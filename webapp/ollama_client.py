"""The only part of the app that knows Ollama exists.

Streams an answer token by token. Streaming is not a nicety here: the
benchmark's median answer took 26 s and its p95 took 74 s, but the first token
arrived after 3.4 s. Waiting 26 s at a spinner feels broken; watching text
appear after 3 s does not.
"""
import json

import httpx

import config


class OllamaError(RuntimeError):
    """Ollama could not be reached, or refused the request."""


class ThinkingStripper:
    """Drop a <unused94>thought ... <unused95> block from a live stream.

    MedGemma 1.5 is a thinking model. Ollama usually hands the reasoning back
    in its own field, but when it does not, the markers land in the visible
    text instead. The benchmark strips them after the fact; here the text is
    already on its way to a browser, so it has to be done as it arrives.

    Text is held back only until it is clear it is not a marker, which is at
    most a few characters.
    """

    OPEN, CLOSE = "<unused94>", "<unused95>"
    START, THINKING, ANSWER = 0, 1, 2

    def __init__(self):
        self.state = self.START
        self.buf = ""

    def feed(self, chunk):
        """Return the portion of `chunk` that is safe to show."""
        if self.state == self.ANSWER:
            return chunk

        self.buf += chunk

        if self.state == self.START:
            stripped = self.buf.lstrip()
            if stripped.startswith(self.OPEN):
                self.state = self.THINKING
                self.buf = stripped[len(self.OPEN):]
            elif self.OPEN.startswith(stripped[:len(self.OPEN)]) and stripped:
                return ""          # still could become the marker: keep waiting
            else:
                self.state = self.ANSWER
                out, self.buf = self.buf, ""
                return out

        if self.state == self.THINKING:
            i = self.buf.find(self.CLOSE)
            if i == -1:
                # Keep only enough tail to recognise a marker split across chunks.
                self.buf = self.buf[-(len(self.CLOSE) - 1):]
                return ""
            self.state = self.ANSWER
            out = self.buf[i + len(self.CLOSE):]
            self.buf = ""
            return out.lstrip()

        return ""

    def flush(self):
        """Whatever is still held back once the stream ends."""
        if self.state == self.START and self.buf:
            self.state = self.ANSWER
            out, self.buf = self.buf, ""
            return out
        return ""


def build_payload(question, system_prompt=None, image_b64=None):
    """The request body sent to Ollama, kept in one place so it is testable.

    `image_b64` is base64 of the raw file bytes with no `data:` prefix, which
    is the only form Ollama's `images` field accepts. When one is present the
    system prompt grows by IMAGE_NOTE; without one the prompt is byte-identical
    to the benchmark's, so the measured path stays the measured path.
    """
    system = system_prompt or config.SYSTEM_PROMPT
    user = {"role": "user", "content": question}
    if image_b64:
        system = system + config.IMAGE_NOTE
        user["images"] = [image_b64]

    return {
        "model": config.MODEL,
        "messages": [
            {"role": "system", "content": system},
            user,
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
            "num_ctx": config.NUM_CTX,
            "repeat_penalty": config.REPEAT_PENALTY,
            "repeat_last_n": config.REPEAT_LAST_N,
        },
    }


async def stream_answer(client, question, system_prompt=None, image_b64=None):
    """Yield ("delta", text) as the answer arrives, then ("done", info).

    Raises OllamaError if Ollama is unreachable or rejects the request.
    """
    stripper = ThinkingStripper()
    payload = build_payload(question, system_prompt, image_b64)

    try:
        async with client.stream("POST", f"{config.OLLAMA_URL}/api/chat",
                                 json=payload) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise OllamaError(
                    f"Ollama returned {resp.status_code}. "
                    f"Is the model '{config.MODEL}' pulled?")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue            # a partial line; the next one completes it

                if obj.get("error"):
                    raise OllamaError(str(obj["error"]))

                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    visible = stripper.feed(piece)
                    if visible:
                        yield "delta", visible

                if obj.get("done"):
                    tail = stripper.flush()
                    if tail:
                        yield "delta", tail
                    yield "done", {
                        "output_tokens": obj.get("eval_count", 0),
                        "prompt_tokens": obj.get("prompt_eval_count", 0),
                        "done_reason": obj.get("done_reason", ""),
                    }
                    return

    except httpx.HTTPError as e:
        raise OllamaError(f"Could not reach Ollama at {config.OLLAMA_URL}: {e}")

    # The stream ended without a done marker: treat it as a truncated answer
    # rather than pretending it finished cleanly.
    tail = stripper.flush()
    if tail:
        yield "delta", tail
    yield "done", {"output_tokens": 0, "prompt_tokens": 0, "done_reason": "incomplete"}


async def check_health(client):
    """Return (ok, message). Used by /api/health and at startup."""
    try:
        r = await client.get(f"{config.OLLAMA_URL}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception as e:
        return False, f"Ollama is not reachable at {config.OLLAMA_URL} ({e})"

    base = config.MODEL.split(":")[0]
    if not any(n == config.MODEL or n.split(":")[0] == base for n in names):
        return False, (f"Ollama is running but '{config.MODEL}' is not pulled. "
                       f"Installed: {', '.join(names) or 'nothing'}")
    return True, f"{config.MODEL} ready"


async def has_vision(client):
    """True if Ollama says this model can accept images.

    Asked rather than assumed. MedGemma 4B ships a vision encoder and the 27B
    text model does not, and a quantisation can be built either way, so the
    only honest answer comes from the model that is actually installed. An
    older Ollama that does not report capabilities at all answers False, which
    hides the upload button rather than offering something that would fail.
    """
    if not config.ALLOW_IMAGES:
        return False
    try:
        r = await client.post(f"{config.OLLAMA_URL}/api/show",
                              json={"model": config.MODEL}, timeout=10.0)
        r.raise_for_status()
        caps = r.json().get("capabilities") or []
    except Exception:
        return False
    return "vision" in [str(c).lower() for c in caps]
