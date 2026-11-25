# webapp — a test site in front of the local model

A small chat page that asks MedGemma a health question and streams the answer
back, so people other than you can try the model. Everything runs on your PC;
Cloudflare only carries the traffic.

```
            internet
                │
                ▼
      ┌───────────────────┐
      │   cloudflared     │   outbound connection, no open ports
      └─────────┬─────────┘
                │  http://127.0.0.1:8000
   ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  your PC
                ▼
      ┌───────────────────┐
      │  server.py        │  ← static/  the page the browser loads
      │  (FastAPI)        │  ← the access code, queue, timeout, rate limit
      └─────────┬─────────┘
                │  http://127.0.0.1:11434
                ▼
      ┌───────────────────┐
      │  Ollama           │
      │  medgemma1.5      │
      └───────────────────┘
```

**The tunnel points at `server.py`, never at Ollama.** Ollama's API has no
authentication and can pull and delete models. Anyone who found that URL could
wipe your model. The only thing exposed here is "ask one question".

| File | What it is |
|---|---|
| `server.py` | The backend service: `/api/ask`, `/api/health`, serves the page |
| `ollama_client.py` | The only file that knows Ollama exists |
| `config.py` | Every setting, read from the environment |
| `static/` | The frontend: one HTML page, one stylesheet, one script |
| `test_server.py` | 19 checks, no model or network needed |

---

## How many terminals

**There is no separate frontend server.** One Python process serves both the
page and the API, so this needs two terminals, not three.

| | What | Terminal |
|---|---|---|
| 1 | Ollama — the model | usually already running |
| 2 | This app — page **and** API | one terminal |
| 3 | `cloudflared` — only to share it | one terminal |

---

## Setup (once)

```bash
cd ~/Me-work/github/local-llm-inference
```

```bash
pip install -r webapp/requirements.txt
```

These are separate from the benchmark's requirements — no pandas, no datasets.

```bash
cp webapp/env.example webapp/.env
```

Open `webapp/.env` in a text editor and set a code:

```
ACCESS_CODE=pick-something-here
```

Leave it empty and anyone who learns the URL can use the site. You only do this
part once.

---

## Terminal 1 — Ollama

If the Ollama app is already running in your system tray, skip this. To check:

```bash
curl http://localhost:11434/api/tags
```

JSON listing your models means it is running. Nothing back means it is not:

```bash
ollama serve
```

Leave it open.

---

## Terminal 2 — this app

```bash
cd ~/Me-work/github/local-llm-inference/webapp
```

```bash
python server.py
```

You should see:

```
  model      medgemma1.5 via http://127.0.0.1:11434
  listening  http://127.0.0.1:8000
  queue      1 at a time, 8 waiting, 120s ceiling
  access     code required (19 characters)

  ok: medgemma1.5 ready
  images     on, up to 4 MB (png, jpeg, webp)
```

If the `ok:` line says `PROBLEM` instead, Ollama is not up or the model is not
pulled — the message says which. Leave this terminal open.

The `images` line reports whether your installed model can read pictures. If
it says `off`, that is the honest answer from Ollama, not a setting to
override — see **Image questions** below.

### Check it locally before going further

Open **http://127.0.0.1:8000**.

1. The disclaimer appears — tick the box, type your access code, click Continue
2. Click one of the example questions
3. Text should start appearing within a few seconds and keep flowing

A **green dot** beside the title means the model is reachable; red means it is
not, and hovering gives the reason. If this does not work, a tunnel will not
fix it — fix it here first.

---

## Terminal 3 — the tunnel (only when sharing)

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a `https://….trycloudflare.com` URL. Give that **and the access
code** to your testers. Closing this terminal takes the site down, and the URL
is different next time.

There is no `--http-host-header` flag this time. That was only needed because
Ollama checks the hostname; this app does not.

---

## Stopping

`Ctrl+C` in each terminal, in any order.

---

## What to expect on the real model

| | |
|---|---|
| Time per answer | about 26 s, streaming throughout |
| Time per **image** answer | much longer — the picture is encoded before any text appears |
| Someone asking while you are | sees "1 question ahead of you" |
| Nothing within 45 s | memory warning — the ceiling working, not a crash |

That last one is worth taking seriously: the benchmark peaked at 16.78 GB of
16.9 GB. Close the browser's other tabs and anything heavy before a session.

---

## Image questions

MedGemma 4B ships a vision encoder, so it can be asked about a picture as well
as about text. The upload button appears **only** if Ollama reports that your
installed model has one. To see what yours says:

```bash
ollama show medgemma1.5
```

If `vision` is not in the capabilities list, the button stays hidden and an
image sent by hand is refused with a plain message. `ALLOW_IMAGES=1` does not
override that — a text-only model cannot be talked into seeing.

### What happens to a picture

| Step | Where | Why |
|---|---|---|
| Oversized request refused unread | `server.py` middleware | FastAPI parses the body *before* the access code is checked, so without this a stranger who knows the URL could make the machine buffer 100 MB bodies |
| Shrunk to 1024px, re-encoded as JPEG | the browser | The encoder works at 896px, so anything larger is uploaded and then discarded. Re-encoding also **drops the EXIF block, which is where a phone writes GPS coordinates** |
| Checked by its first bytes | `server.py` | A file called `x.png` is a PNG only if it starts like one. PDFs, executables and SVGs are refused |
| Size checked before decoding | `server.py` | 4 MB of base64 is cheap to measure and expensive to decode |
| Sent as `images: [...]` | `ollama_client.py` | Ollama's own field for this. Never a `data:` URL |
| Size logged, image never written | `server.py` | A photo of a person is far more identifying than their words |

### It will be slow

Encoding the picture happens **before the first token appears**, so an image
question on a CPU feels much slower than a text one. Both timeouts are
multiplied by `IMAGE_TIMEOUT_FACTOR` (3) when an image is attached — 135 s to
the first token instead of 45, 360 s overall instead of 120. The page says
"Reading the image… this takes longer" rather than letting it look stuck.

If image questions still time out, raise the factor rather than lowering the
protection on text questions.

### Two warnings

**The benchmark has never sent an image.** Your 38% pass rate is a text-only
number. `SYSTEM_PROMPT` stays byte-identical to the benchmark's
`DEFAULT_SYSTEM`, and `IMAGE_NOTE` is appended *only* when a picture is
attached — so the measured path stays measured, but nothing here tells you how
well the model reads images. Assume worse than text until you have graded it.

**A photo of a patient is health data.** The disclaimer asks testers not to
include a face, a name band, or anything identifying in the frame, and the
browser strips GPS before upload. Neither of those is consent. Under the DPDP
rules a patient photograph is sensitive personal data, so do not collect
images from testers for a training set without deciding that question
properly first.

---

## What it does to protect a 16 GB machine

The benchmark peaked at 16.78 GB of 16.9 GB answering **one** question, and two
of its 50 questions took 657 s and 844 s once the machine started swapping.
The defaults come straight from that:

| Setting | Default | Why |
|---|---|---|
| `MAX_CONCURRENT` | 1 | There is no RAM for a second answer. Extra askers queue. |
| `MAX_QUEUE` | 8 | Past this, people get "try again in a minute" instead of a silent wait |
| `FIRST_TOKEN_TIMEOUT_S` | 45 | If nothing has arrived by then, the machine is thrashing |
| `ANSWER_TIMEOUT_S` | 120 | One stalled answer must not block the queue for 14 minutes |
| `RATE_LIMIT_QUESTIONS` | 10 per 10 min | Per caller, so one person can't monopolise the queue |

The page shows "3 questions ahead of you" while waiting, so a queue looks like a
queue rather than a hang.

Answers stream. The benchmark's median answer took 26 s but its first token
arrived after 3.4 s — streaming is the difference between "broken" and "thinking".

---

## Keeping it honest

`config.SYSTEM_PROMPT` and the sampling settings are **copies of the
benchmark's defaults**. `test_server.py` reads `DEFAULT_SYSTEM` out of
`../local-medgemma.py` and fails if the two have drifted. If the site answers
with a different prompt than the benchmark, your 38% pass rate stops predicting
what users will see.

Change one, change both.

---

## Logging and personal data

`LOG_QUESTIONS` is **off** by default, and should stay off until you have
decided what you are doing about consent and retention. With it off, the log
holds timings, token counts and a salted hash of the caller — never the
question text.

A public health Q&A page collects real symptoms from real people. That is
sensitive personal data under the DPDP rules, not test data. If you want
questions for your eval set, collecting them from named testers who know what
they signed up for is a different thing from quietly logging whoever finds the
URL.

---

## What this is not

The disclaimer the page makes people accept is not decoration. In the 19 Nov
grading run this model passed **19 of 50** questions and produced **11 unsafe
answers** — invented drug names, wrong genes, and a treatment for the wrong
disease entirely. Keep the site invite-only, keep the disclaimer, and don't
describe it to testers as anything but an experiment.

---

## Tests

```bash
cd webapp && python test_server.py
```

19 checks: the prompt matches the benchmark, sampling is right, thinking
markers never reach the browser, streaming works, the access code is enforced,
bad input is refused, the rate limit bites, a stalled model times out, the
queue slot is always released, health reports correctly, the page serves with
its disclaimer, and Ollama errors surface instead of being swallowed.

On images: an attached picture reaches Ollama in the `images` field with
`IMAGE_NOTE` added, the text-only prompt does not grow, an image with no words
still asks something, uploads are identified by their bytes so a renamed file
is refused, corrupt and oversized uploads get a clear status, a model with no
vision capability refuses images instead of failing deep inside Ollama, and
health reports whether the installed model can actually see.

The first four run without FastAPI installed; the rest are skipped if it
isn't.
