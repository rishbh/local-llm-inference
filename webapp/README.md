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
| `test_server.py` | 12 checks, no model or network needed |

---

## Run it

**1. Install the dependencies** (separate from the benchmark's — no pandas here):

```bash
pip install -r webapp/requirements.txt
```

**2. Make your settings file:**

```bash
cp webapp/env.example webapp/.env
```

Open `webapp/.env` and set `ACCESS_CODE` to anything you like. Leave it empty
and anyone who learns the URL can use the site.

**3. Start it** (Ollama must already be running):

```bash
cd webapp && python server.py
```

Open http://127.0.0.1:8000. If the dot next to the title is green, the model is
reachable. Red means Ollama is down or the model isn't pulled — hover for the
reason.

**4. Put it on the internet:**

```bash
cloudflared tunnel --url http://localhost:8000
```

Share the `https://….trycloudflare.com` URL and the access code. Closing that
terminal takes the site down.

Note there is no `--http-host-header` flag this time. That was only needed
because Ollama checks the hostname; this app doesn't.

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

12 checks: the prompt matches the benchmark, sampling is right, thinking
markers never reach the browser, streaming works, the access code is enforced,
bad input is refused, the rate limit bites, a stalled model times out, the
queue slot is always released, health reports correctly, the page serves with
its disclaimer, and Ollama errors surface instead of being swallowed.

The first three run without FastAPI installed; the rest are skipped if it
isn't.
