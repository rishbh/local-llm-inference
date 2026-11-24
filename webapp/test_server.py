"""Checks for the web app. No model and no network needed.

    cd webapp && python test_server.py

Ollama is replaced by a fake, so these run in under a second and can be run
on a machine that has never heard of MedGemma.
"""
import ast
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                      # noqa: E402
import ollama_client               # noqa: E402

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print("OK:", msg)


# --------------------------------------------------------------------------- #
# 1. the system prompt has not drifted from the benchmark's
# --------------------------------------------------------------------------- #
def benchmark_prompt():
    """Read DEFAULT_SYSTEM out of local-medgemma.py without importing it.

    Importing would drag in pandas and psutil; parsing the file does not.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "local-medgemma.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEFAULT_SYSTEM":
                    return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_SYSTEM not found in local-medgemma.py")


bench = benchmark_prompt()
assert config.SYSTEM_PROMPT == bench, (
    "The site's system prompt has drifted from the benchmark's.\n"
    "  benchmark: " + repr(bench[:90]) + "...\n"
    "  webapp:    " + repr(config.SYSTEM_PROMPT[:90]) + "...\n"
    "Copy DEFAULT_SYSTEM from local-medgemma.py into config.py, or the "
    "benchmark stops predicting what users will see.")
ok("the site and the benchmark use the same system prompt")


# --------------------------------------------------------------------------- #
# 2. the Ollama payload matches the benchmark's settings
# --------------------------------------------------------------------------- #
p = ollama_client.build_payload("why does my throat hurt?")
assert p["stream"] is True and p["think"] is False, p
assert p["messages"][0]["role"] == "system", p["messages"]
assert p["messages"][1]["content"] == "why does my throat hurt?", p["messages"]
o = p["options"]
assert o["temperature"] == 0, o
assert o["repeat_penalty"] == 1.1 and o["repeat_last_n"] == 256, o
assert o["num_predict"] == 1600 and o["num_ctx"] == 4096, o
assert "images" not in p["messages"][1], p["messages"]
ok("the request to Ollama carries the benchmark's sampling settings")


# --------------------------------------------------------------------------- #
# 2b. an image rides in the user message, and only then does the prompt grow
# --------------------------------------------------------------------------- #
pi = ollama_client.build_payload("what is this?", image_b64="QUJD")
assert pi["messages"][1]["images"] == ["QUJD"], pi["messages"]
assert pi["messages"][0]["content"].startswith(config.SYSTEM_PROMPT), "prompt replaced"
assert config.IMAGE_NOTE in pi["messages"][0]["content"], "image note missing"
# The text-only path must stay exactly what the benchmark measured.
assert p["messages"][0]["content"] == config.SYSTEM_PROMPT, "text prompt grew"
assert pi["options"] == p["options"], "sampling differs when an image is sent"
ok("an image is attached to the user message and only then is IMAGE_NOTE added")


# --------------------------------------------------------------------------- #
# 3. the thinking block never reaches the browser
# --------------------------------------------------------------------------- #
def run_stripper(chunks):
    s = ollama_client.ThinkingStripper()
    return "".join(s.feed(c) for c in chunks) + s.flush()

cases = [
    (["Plain answer."], "Plain answer."),
    (["<unused94>thought\nplan plan\n<unused95>\nReal answer."], "Real answer."),
    # split across chunks, the way a stream actually arrives
    (["<unu", "sed94>", "thinking...", "<unused", "95>", "The answer."], "The answer."),
    (["<unused94>", "a" * 500, "<unused95>Short."], "Short."),
    # a lone "<" at the start must not be swallowed
    (["<b>bold</b> answer"], "<b>bold</b> answer"),
]
for chunks, want in cases:
    got = run_stripper(chunks)
    assert got == want, (chunks, got, want)

# a thought block that never closes leaks nothing
assert run_stripper(["<unused94>", "forever and ever"]) == ""
ok("thinking markers are stripped from the stream, however they are chunked")


# --------------------------------------------------------------------------- #
# the rest need the app, so only run them if FastAPI is installed
# --------------------------------------------------------------------------- #
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("\nSKIPPED the HTTP checks: pip install -r requirements.txt")
    print("\n%d/%d CHECKS PASSED" % (len(PASSED), len(PASSED)))
    raise SystemExit(0)

import server                      # noqa: E402


class FakeStream:
    """Stands in for httpx's streaming response."""

    def __init__(self, lines, status=200):
        self.lines, self.status_code = lines, status

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def aread(self): return b""

    async def aiter_lines(self):
        for line in self.lines:
            await asyncio.sleep(0)
            yield line


def fake_client(lines, status=200, vision=False):
    class C:
        sent = []               # every payload Ollama was asked to answer

        def stream(self, method, url, json=None):
            C.sent.append(json)
            return FakeStream(lines, status)

        async def get(self, url, timeout=None):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self): return {"models": [{"name": config.MODEL}]}
            return R()

        async def post(self, url, json=None, timeout=None):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    return {"capabilities": ["completion"] +
                            (["vision"] if vision else [])}
            return R()
    C.sent = []
    return C()


def chat_lines(words, tokens=12):
    out = [json.dumps({"message": {"content": w}, "done": False}) for w in words]
    out.append(json.dumps({"message": {"content": ""}, "done": True,
                           "eval_count": tokens, "prompt_eval_count": 40,
                           "done_reason": "stop"}))
    return out


def read_ndjson(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


# ---- 4. a question streams back, and the answer is assembled -------------
config.ACCESS_CODE = ""
server._client = fake_client(chat_lines(["See ", "a ", "doctor."]))
with TestClient(server.app) as c:
    r = c.post("/api/ask", json={"question": "my throat hurts"})
    assert r.status_code == 200, r.text
    msgs = read_ndjson(r)
    kinds = [m["type"] for m in msgs]
    assert kinds[0] == "start" and kinds[-1] == "done", kinds
    text = "".join(m["text"] for m in msgs if m["type"] == "delta")
    assert text == "See a doctor.", repr(text)
    assert msgs[-1]["output_tokens"] == 12, msgs[-1]
ok("a question streams back token by token and ends with a done message")


# ---- 5. the access code is enforced --------------------------------------
config.ACCESS_CODE = "let-me-in"
server._client = fake_client(chat_lines(["hi"]))
with TestClient(server.app) as c:
    assert c.post("/api/ask", json={"question": "hello"}).status_code == 401
    assert c.post("/api/ask", json={"question": "hello"},
                  headers={"X-Access-Code": "wrong"}).status_code == 401
    good = c.post("/api/ask", json={"question": "hello"},
                  headers={"X-Access-Code": "let-me-in"})
    assert good.status_code == 200, good.text
ok("the access code is required and checked")


# ---- 6. empty and oversized questions are refused ------------------------
config.ACCESS_CODE = ""
with TestClient(server.app) as c:
    assert c.post("/api/ask", json={"question": "   "}).status_code == 400
    big = c.post("/api/ask", json={"question": "x" * (config.MAX_QUESTION_CHARS + 1)})
    assert big.status_code == 400, big.status_code
ok("empty and over-long questions are refused before the model is woken")


# ---- 7. the rate limit bites ---------------------------------------------
config.RATE_LIMIT_QUESTIONS = 3
server._hits.clear()
server._client = fake_client(chat_lines(["ok"]))
with TestClient(server.app) as c:
    codes = [c.post("/api/ask", json={"question": "q"}).status_code for _ in range(5)]
assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429], codes
ok("a caller past the rate limit gets 429, not an answer")
config.RATE_LIMIT_QUESTIONS = 10
server._hits.clear()


# ---- 8. a stalled model times out instead of hanging forever -------------
class StallStream(FakeStream):
    async def aiter_lines(self):
        await asyncio.sleep(30)        # longer than the budget below
        yield json.dumps({"message": {"content": "too late"}, "done": True})


class StallClient:
    def stream(self, method, url, json=None): return StallStream([])
    async def get(self, url, timeout=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"models": [{"name": config.MODEL}]}
        return R()


config.FIRST_TOKEN_TIMEOUT_S = 0.3
server._client = StallClient()
with TestClient(server.app) as c:
    msgs = read_ndjson(c.post("/api/ask", json={"question": "slow one"}))
assert any(m["type"] == "error" and "too long" in m["message"] for m in msgs), msgs
ok("a model that stalls produces an error, not an endless request")
config.FIRST_TOKEN_TIMEOUT_S = 45.0


# ---- 9. the slot is released after a failure, so the queue never jams ----
server._client = fake_client(chat_lines(["fine"]))
with TestClient(server.app) as c:
    r = c.post("/api/ask", json={"question": "after the timeout"})
    assert r.status_code == 200, r.text
    assert "".join(m.get("text", "") for m in read_ndjson(r)) == "fine"
assert server._waiting == 0, server._waiting
assert server._slots._value == config.MAX_CONCURRENT, server._slots._value
ok("the queue slot is released even when an answer fails")


# ---- 10. health reports the model and whether a code is needed -----------
config.ACCESS_CODE = "secret"
server._client = fake_client([])
with TestClient(server.app) as c:
    h = c.get("/api/health").json()
assert h["ok"] is True and h["needs_code"] is True and h["model"] == config.MODEL, h
ok("/api/health reports the model, and that a code is needed")
config.ACCESS_CODE = ""


# ---- 11. the page is served and carries the disclaimer -------------------
with TestClient(server.app) as c:
    page = c.get("/")
    assert page.status_code == 200
    body = page.text
    for needed in ["not medical advice", "112", "experimental", "agree",
                   # the photo warning, and the upload control it refers to
                   "be careful with photos", 'id="attach"']:
        assert needed.lower() in body.lower(), f"page is missing {needed!r}"
    assert c.get("/app.js").status_code == 200
    assert c.get("/style.css").status_code == 200
ok("the page loads with the disclaimer, and its assets are served")


# ---- 12. Ollama being down is reported, not swallowed --------------------
server._client = fake_client(["{\"error\":\"model not found\"}"])
with TestClient(server.app) as c:
    msgs = read_ndjson(c.post("/api/ask", json={"question": "anything"}))
assert any(m["type"] == "error" and "model not found" in m["message"] for m in msgs), msgs
ok("an error from Ollama reaches the page instead of being swallowed")


# ---- 13. only real PNG / JPEG / WebP bytes are accepted ------------------
PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQ"
           "DwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
import base64 as _b64                # noqa: E402

assert server.sniff_image(_b64.b64decode(PNG_1PX)) == "png"
assert server.sniff_image(b"\xff\xd8\xff\xe0 jfif") == "jpeg"
assert server.sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
# The things an upload field actually gets handed
assert server.sniff_image(b"GIF89a") is None            # wrong image format
assert server.sniff_image(b"%PDF-1.4") is None          # a document
assert server.sniff_image(b"MZ\x90\x00") is None        # a Windows executable
assert server.sniff_image(b"<svg xmlns=") is None       # scriptable, so refused
assert server.sniff_image(b"") is None
ok("images are identified by their bytes, so a renamed file is still refused")


# ---- 14. an attached image reaches Ollama ---------------------------------
config.ACCESS_CODE = ""
server._vision = True
fc = fake_client(chat_lines(["A ", "rash."]))
server._client = fc
with TestClient(server.app) as c:
    r = c.post("/api/ask", json={"question": "what is this?", "image": PNG_1PX})
    assert r.status_code == 200, r.text
    assert "".join(m.get("text", "") for m in read_ndjson(r)) == "A rash."
sent = fc.sent[-1]
assert sent["messages"][1]["images"], list(sent["messages"][1])
assert _b64.b64decode(sent["messages"][1]["images"][0])[:8] == b"\x89PNG\r\n\x1a\n"
assert config.IMAGE_NOTE in sent["messages"][0]["content"]
ok("an attached image is forwarded to Ollama with the image note")


# ---- 15. an image with no question still asks something -------------------
fc = fake_client(chat_lines(["ok"]))
server._client = fc
with TestClient(server.app) as c:
    r = c.post("/api/ask", json={"question": "", "image": PNG_1PX})
    assert r.status_code == 200, r.text
assert fc.sent[-1]["messages"][1]["content"].strip(), "empty question reached the model"
ok("an image with no words still sends the model a question")


# ---- 16. junk, corrupt and oversized uploads are refused ------------------
server._client = fake_client(chat_lines(["x"]))
with TestClient(server.app) as c:
    # not an image at all
    bad = c.post("/api/ask", json={"question": "q", "image":
                                   _b64.b64encode(b"%PDF-1.4 hello").decode()})
    assert bad.status_code == 400, (bad.status_code, bad.text)
    # not even valid base64
    broken = c.post("/api/ask", json={"question": "q", "image": "!!!not base64!!!"})
    assert broken.status_code == 400, broken.status_code
    # past the size ceiling: stopped by the body-size middleware, before
    # FastAPI parses several megabytes of JSON on a 16 GB machine
    huge = c.post("/api/ask", json={"question": "q",
                                    "image": "A" * (config.MAX_IMAGE_BYTES * 2)})
    assert huge.status_code == 413, huge.status_code

# ...and decode_image refuses an oversized image on its own, so the ceiling
# does not depend on the middleware alone.
try:
    server.decode_image("A" * (config.MAX_IMAGE_BYTES * 2))
    raise AssertionError("decode_image accepted an oversized image")
except Exception as e:
    assert getattr(e, "status_code", None) == 413, e
ok("junk, corrupt and oversized uploads are refused with a clear status")


# ---- 17. a model with no vision encoder refuses images --------------------
server._vision = False
server._client = fake_client(chat_lines(["x"]))
with TestClient(server.app) as c:
    blocked = c.post("/api/ask", json={"question": "q", "image": PNG_1PX})
    assert blocked.status_code == 400, blocked.status_code
    assert "cannot read images" in blocked.json()["detail"], blocked.text
    # text still works perfectly well
    assert c.post("/api/ask", json={"question": "q"}).status_code == 200
ok("a text-only model refuses an image instead of failing deep inside Ollama")


# ---- 18. health tells the page whether to show the upload button ----------
server._vision = None
server._client = fake_client([], vision=True)
with TestClient(server.app) as c:
    assert c.get("/api/health").json()["images"] is True

server._vision = None
server._client = fake_client([], vision=False)
with TestClient(server.app) as c:
    h = c.get("/api/health").json()
assert h["images"] is False and h["max_image_mb"] >= 1, h
ok("/api/health reports whether the installed model can actually see")


print("\n%d/%d CHECKS PASSED" % (len(PASSED), len(PASSED)))
