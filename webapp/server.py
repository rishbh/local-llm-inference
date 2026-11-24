"""The backend service: one endpoint the page can call, and nothing else.

Why this sits between the browser and Ollama at all: Ollama's API has no
authentication and does far more than answer questions. Anyone who reached it
directly could pull models until the disk filled, or delete the one you are
benchmarking. Exposing this app to the tunnel instead of Ollama means the only
thing reachable from the internet is "ask one question".

    browser  ->  GET  /            the page (served from ./static)
             ->  POST /api/ask     one question, answer streamed back
             ->  GET  /api/health  is the model up

Run it with:  python -m uvicorn server:app --host 127.0.0.1 --port 8000
"""
import asyncio
import base64
import binascii
import collections
import hashlib
import json
import os
import secrets
import time
import typing

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import ollama_client

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="Health Q&A (local model)", docs_url=None, redoc_url=None)

# One answer at a time by default: see the note in config.py about RAM.
_slots = asyncio.Semaphore(config.MAX_CONCURRENT)
_waiting = 0
_waiting_lock = asyncio.Lock()

# Salt is per-process and never stored, so a caller can be counted within a
# single run but not identified across restarts.
_IP_SALT = secrets.token_bytes(16)
_hits = collections.defaultdict(collections.deque)

# Created at startup unless something has already supplied one, which is how
# the tests run the whole app against a fake Ollama.
_client = None
_owns_client = False

# None until Ollama has been asked whether this model can see. Left as None if
# Ollama was down at startup, so /api/health retries rather than reporting
# "no vision" for the rest of the process's life.
_vision = None


class Ask(BaseModel):
    question: str
    # Base64 of the raw file bytes, no "data:" prefix. The page strips the
    # prefix and shrinks the picture before it gets here.
    image: typing.Optional[str] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def caller_id(request):
    """A stable, non-reversible id for rate limiting.

    Behind the tunnel every connection appears to come from Cloudflare, so the
    real address is in CF-Connecting-IP. The value is hashed immediately: the
    app never needs the address itself, only a way to count repeats.
    """
    ip = (request.headers.get("cf-connecting-ip")
          or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    return hashlib.blake2b(_IP_SALT + ip.encode(), digest_size=8).hexdigest()


def rate_limited(who):
    """True if this caller has used up their allowance for the window."""
    if config.RATE_LIMIT_QUESTIONS <= 0:
        return False
    now = time.monotonic()
    q = _hits[who]
    while q and now - q[0] > config.RATE_LIMIT_WINDOW_S:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_QUESTIONS:
        return True
    q.append(now)
    return False


def sniff_image(raw):
    """Return "png" / "jpeg" / "webp", or None if these bytes are not one.

    The browser's declared type is not consulted. An upload field is the one
    place where a stranger hands this machine arbitrary bytes, and a file
    called `x.png` is a .png only if it starts like one.
    """
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(b64):
    """Validate a base64 image and return (clean_base64, bytes, kind).

    Raises HTTPException with a message meant for a person, not a log.
    """
    b64 = (b64 or "").strip()
    if b64.startswith("data:"):
        _, _, b64 = b64.partition(",")      # tolerate a pasted data URL

    # Checked before decoding: 4 MB of base64 is cheap to measure and
    # expensive to decode, and the whole point is not to do the expensive part.
    if len(b64) * 3 // 4 > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That image is too big. The limit is "
                   f"{config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="That image did not upload properly.")

    if len(raw) > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That image is too big. The limit is "
                   f"{config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")

    kind = sniff_image(raw)
    if kind not in config.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPEG and WebP images can be read.")

    # Re-encoded from the bytes that were actually checked, so whitespace or
    # odd padding in the request cannot reach Ollama.
    return base64.b64encode(raw).decode("ascii"), raw, kind


def log_event(**fields):
    """Append one line to the request log.

    Question and answer text are included only when LOG_QUESTIONS is on, which
    is not the default. See the note in config.py before turning it on.
    """
    fields["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")
    except OSError:
        pass          # a log that cannot be written must not break an answer


def sse(obj):
    """One newline-delimited JSON message. The page reads these as they land."""
    return json.dumps(obj, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# middleware
# --------------------------------------------------------------------------- #
# Headroom over MAX_IMAGE_BYTES for base64's 33% overhead and the JSON around
# it. Computed once rather than per request.
_MAX_BODY = int(config.MAX_IMAGE_BYTES * 1.4) + 64 * 1024


@app.middleware("http")
async def limit_body(request, call_next):
    """Refuse an oversized body before anything is read or parsed.

    FastAPI parses the whole JSON body into memory before the handler - and so
    before the access code is checked. Without this, a stranger who merely
    knows the URL could make a 16 GB machine buffer 100 MB bodies all day. The
    declared length is enough: a request that lies about it fails later anyway.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_BODY:
        return JSONResponse(
            {"detail": f"That request is too large. Images are limited to "
                       f"{config.MAX_IMAGE_BYTES // (1024 * 1024)} MB."},
            status_code=413)
    return await call_next(request)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def startup():
    global _client, _owns_client
    if _client is None:
        # Generous read timeout: the per-answer ceiling is enforced below,
        # where a friendly message can still be streamed to the page.
        _client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
        _owns_client = True

    print(f"\n  model      {config.MODEL} via {config.OLLAMA_URL}")
    print(f"  listening  http://{config.HOST}:{config.PORT}")
    print(f"  queue      {config.MAX_CONCURRENT} at a time, "
          f"{config.MAX_QUEUE} waiting, {config.ANSWER_TIMEOUT_S:.0f}s ceiling")
    if config.ACCESS_CODE:
        print(f"  access     code required ({len(config.ACCESS_CODE)} characters)")
    else:
        print("\n  " + "!" * 64)
        print("  WARNING: ACCESS_CODE is not set, so anyone who learns the URL")
        print("           can use this. Fine on localhost. Set one before you")
        print("           put it behind a tunnel.")
        print("  " + "!" * 64)
    if config.LOG_QUESTIONS:
        print("\n  NOTE: LOG_QUESTIONS is on - question and answer text is being")
        print(f"        written to {config.LOG_PATH}")
    print()

    ok, msg = await ollama_client.check_health(_client)
    print(f"  {'ok' if ok else 'PROBLEM'}: {msg}")

    global _vision
    # Only asked if nothing has already answered it, so a test that injects a
    # capability is not overwritten on the way up.
    if ok:
        if _vision is None:
            _vision = await ollama_client.has_vision(_client)
        if _vision:
            print(f"  images     on, up to "
                  f"{config.MAX_IMAGE_BYTES // (1024 * 1024)} MB "
                  f"({', '.join(config.ALLOWED_IMAGE_TYPES)})")
        elif config.ALLOW_IMAGES:
            print(f"  images     off - {config.MODEL} does not report a vision "
                  f"capability")
    print()


@app.on_event("shutdown")
async def shutdown():
    global _client, _owns_client
    if _owns_client and _client:
        await _client.aclose()
        _client, _owns_client = None, False


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    global _vision
    ok, msg = await ollama_client.check_health(_client)
    if ok and _vision is None:
        _vision = await ollama_client.has_vision(_client)
    return JSONResponse(
        {"ok": ok, "detail": msg, "model": config.MODEL,
         "needs_code": bool(config.ACCESS_CODE), "busy": _waiting,
         # The page uses this to decide whether to show the upload button at
         # all, so a text-only model never offers something that would fail.
         "images": bool(ok and _vision),
         "max_image_mb": config.MAX_IMAGE_BYTES // (1024 * 1024)},
        status_code=200 if ok else 503)


@app.post("/api/ask")
async def ask(body: Ask, request: Request, x_access_code: str = Header(default="")):
    global _waiting

    if config.ACCESS_CODE and not secrets.compare_digest(x_access_code,
                                                         config.ACCESS_CODE):
        raise HTTPException(status_code=401, detail="Wrong or missing access code.")

    question = (body.question or "").strip()
    if not question and not body.image:
        raise HTTPException(status_code=400, detail="Please type a question.")
    if len(question) > config.MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Please keep it under {config.MAX_QUESTION_CHARS} characters.")

    image_b64, image_bytes = None, 0
    if body.image:
        if not config.ALLOW_IMAGES:
            raise HTTPException(status_code=400, detail="Images are turned off here.")
        if not _vision:
            raise HTTPException(
                status_code=400,
                detail=f"{config.MODEL} cannot read images.")
        image_b64, raw, _kind = decode_image(body.image)
        image_bytes = len(raw)
        if not question:
            # An image with no words is a real thing people do. Asking nothing
            # is not, so give the model the question they meant.
            question = "What can you tell me about this image?"

    who = caller_id(request)
    if rate_limited(who):
        raise HTTPException(
            status_code=429,
            detail=f"That's {config.RATE_LIMIT_QUESTIONS} questions in "
                   f"{config.RATE_LIMIT_WINDOW_S / 60:.0f} minutes. Give it a rest.")

    async with _waiting_lock:
        if _waiting >= config.MAX_QUEUE:
            raise HTTPException(
                status_code=503,
                detail="Too many people are waiting. Try again in a minute.")
        _waiting += 1
        position = _waiting

    return StreamingResponse(
        _answer(question, who, position, image_b64, image_bytes),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


async def _answer(question, who, position, image_b64=None, image_bytes=0):
    """Wait for a slot, then stream the answer. Always releases the slot."""
    global _waiting
    started = time.monotonic()
    tokens = 0
    status = "ok"
    answer_parts = []

    try:
        if position > config.MAX_CONCURRENT:
            yield sse({"type": "queued", "position": position - config.MAX_CONCURRENT})

        # Encoding the picture happens before any text comes back, so both
        # budgets stretch when there is one.
        slow = config.IMAGE_TIMEOUT_FACTOR if image_b64 else 1.0
        first_budget = config.FIRST_TOKEN_TIMEOUT_S * slow
        total_budget = config.ANSWER_TIMEOUT_S * slow

        async with _slots:
            yield sse({"type": "start"})
            first_token_at = None
            stream = ollama_client.stream_answer(_client, question,
                                                 image_b64=image_b64)

            try:
                while True:
                    budget = (first_budget if first_token_at is None
                              else total_budget - (time.monotonic() - started))
                    if budget <= 0:
                        raise asyncio.TimeoutError()

                    kind, data = await asyncio.wait_for(stream.__anext__(),
                                                        timeout=budget)
                    if kind == "delta":
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        answer_parts.append(data)
                        yield sse({"type": "delta", "text": data})
                    else:
                        tokens = data.get("output_tokens", 0)
                        break
            except StopAsyncIteration:
                pass
            except asyncio.TimeoutError:
                status = "timeout"
                yield sse({"type": "error", "message":
                           "The model is taking too long - it usually means this "
                           "machine has run out of memory. Try a shorter question."})
            finally:
                await stream.aclose()

        if status == "ok":
            yield sse({"type": "done",
                       "output_tokens": tokens,
                       "seconds": round(time.monotonic() - started, 1)})

    except ollama_client.OllamaError as e:
        status = "model_error"
        yield sse({"type": "error", "message": str(e)})
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    finally:
        async with _waiting_lock:
            _waiting -= 1
        event = {"caller": who, "status": status, "question_chars": len(question),
                 # Size only. The image itself is never written anywhere: a
                 # photo of a person is far more identifying than their words.
                 "image_bytes": image_bytes,
                 "output_tokens": tokens,
                 "seconds": round(time.monotonic() - started, 1)}
        if config.LOG_QUESTIONS:
            event["question"] = question
            event["answer"] = "".join(answer_parts)
        log_event(**event)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, log_level="info")
