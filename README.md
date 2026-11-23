# local-llm-inference

Benchmark a medical Q&A model running **locally** on your own machine, and find
out what it actually costs: accuracy, speed, memory, and rupees per question.

Built around **MedGemma 1.5 4B** in Ollama, but `--model` takes any Ollama model.

Two scripts and a web app:

| File | What it does |
|---|---|
| `local-medgemma.py` | Asks the model a sample of real medical questions, measures everything, writes the results |
| `clinical_judge.py` | Grades the answers with Claude against the rubric in `grading_prompt.md` |
| `webapp/` | A chat page that puts the same model, with the same prompt, in front of other people — see [webapp/README.md](webapp/README.md) |

---

## Before you start

**Check your RAM.** This is the thing that bites.

| Model size | Needs roughly |
|---|---|
| 4B at Q4_K_M (~3 GB on disk) | 8 GB free |
| 4B at Q8_0 (~5 GB on disk) | 10 GB free |

On a 16 GB machine a Q4 model fits with the browser closed. Q8 is tight. If the
machine runs out of headroom it starts swapping to disk, and a question that
normally takes 25 seconds can take 14 minutes — the answers stay correct, but
every speed and cost number in the output becomes meaningless. Watch
`peak_system_ram_gb` in `summary.json`; if it's within a few hundred MB of
`total_ram_gb`, don't quote the speed.

You also need **about 1 GB of disk** for the datasets, and an internet
connection the first time (Hugging Face must be reachable — see
[Troubleshooting](#troubleshooting) if you're behind a corporate VPN).

---

## Step 1 — Install Ollama

Ollama is the program that runs the model on your machine. Download it from:

**https://ollama.com/download**

Install it like any other app, then **start it**. On macOS and Windows that
means launching the Ollama app — it sits in the menu bar / system tray and
listens on `localhost:11434`. On Linux:

```bash
ollama serve
```

Check it's up:

```bash
ollama list
```

If that prints a table (even an empty one), you're ready. If it errors, Ollama
isn't running yet.

---

## Step 2 — Pull the model

```bash
ollama pull medgemma1.5
```

This downloads a few GB. It only happens once.

Want a specific quantization instead? Quantization is lossy compression of the
model's weights — a smaller file, slightly worse recall of rare facts. Pull one
straight from Hugging Face:

```bash
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
```

Swap `Q4_K_M` for whichever `.gguf` file that repo actually publishes — open the
repo's file list in a browser to see the options. If the tag doesn't exist the
pull fails immediately, which is a cheap way to check.

That tag is long and ends up in every row of your results, so give it a short
name:

```bash
ollama cp hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M medgemma1.5:q4
```

Confirm what you have:

```bash
ollama list
```

The size column tells you which quant it is: ~3 GB is Q4_K_M, ~5 GB is Q8_0.

---

## Step 3 — Install the Python libraries

Python 3.9 or newer. Use a virtual environment so this doesn't collide with
anything else on the machine.

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

That installs:

| Package | Why |
|---|---|
| `datasets` | downloads the question sets from Hugging Face |
| `requests` | talks to Ollama |
| `pandas` | builds the result tables |
| `psutil` | samples RAM while the model answers |
| `anthropic` | **only** for `clinical_judge.py`; not needed to benchmark |

The `.venv` folder is gitignored. Remember to activate it in every new terminal.

---

## Step 4 — Check the datasets load (do this first)

This takes seconds, needs no model, and saves you discovering a problem an hour
into a run:

```bash
python local-medgemma.py --check-datasets
```

It tries every dataset and tells you which ones work and, for the ones that
don't, what to do about it — gated repo, wrong ID, blocked network, or a missing
config name. It exits non-zero if anything failed.

**Don't skip this.** If a dataset silently fails to load you'll benchmark on a
smaller, differently-shaped question set than you think you did, and the numbers
won't mean what you assume.

---

## Step 5 — Run the benchmark

```bash
python local-medgemma.py --n 50 --watts 65 --rate 8
```

That's 50 questions from each dataset that loads. Expect **20–50 minutes** for
50 questions on a CPU-only laptop.

Two flags are about your machine, not the model, and you should set them
honestly or the cost figures are fiction:

| Flag | What to put |
|---|---|
| `--watts` | Average power your PC draws *while answering*. Best from a plug-in power meter. Otherwise HWiNFO on Windows, `nvidia-smi` for GPUs. On Linux with an Intel/AMD CPU, pass `--rapl` instead and it measures directly. |
| `--rate` | Your electricity price per kWh, from your bill, in rupees. |

Try a short one first to confirm the whole pipeline works end to end:

```bash
python local-medgemma.py --n 3
```

### Common variations

Benchmark a different model or quant, and name the run folder so you can compare
later:

```bash
python local-medgemma.py --model medgemma1.5:q8 --n 50 --run-name q8
```

Only the patient-style datasets, or only the exam sets:

```bash
python local-medgemma.py --datasets medquad --n 50
```

```bash
python local-medgemma.py --datasets medmcqa,medqa,pubmedqa --n 50
```

Short on RAM? Halve the context window — 2048 is still far more than these
questions need, and it buys back roughly half a gigabyte:

```bash
python local-medgemma.py --n 50 --ctx 2048
```

---

## Step 6 — Read the results

Every run gets its own timestamped folder, so running again never overwrites
what you had:

```
bench_results/
  2026-11-19_09-56-40/
    summary.json             accuracy, speed, memory and cost
    per_question.csv         every question, answer, timings, token counts
    clinical_for_review.csv  patient-style answers with empty score columns
  latest -> 2026-11-19_09-56-40
```

`latest` is a symlink to the newest run. It's gitignored.

What to look at in `summary.json`, in order:

1. **`datasets` vs `datasets_requested`** — did everything you asked for actually
   load? `datasets_failed` says why not.
2. **`open_ended_health`** — run health, not model quality. `reasoning_only_no_answer`
   and `truncated` should both be **0**. If they aren't, the model ran out of
   token budget before writing an answer, and your accuracy numbers are measuring
   the harness. Raise `--max-tokens-open`.
3. **`memory.peak_system_ram_gb`** — if it's at `total_ram_gb`, ignore the speed
   and cost numbers for that run.
4. **`speed`** — prefer `latency_p50_s` over `output_tokens_per_sec`. The
   tokens/sec figure is a total-over-total aggregate, so one stalled question
   drags the whole run's number down.
5. **`accuracy_percent`** — real percentages for the exam sets. For the
   patient-style sets it just points you at the grading CSV, because "correct"
   isn't a yes/no for an open-ended medical answer.

---

## Step 7 — Grade the answers

The exam datasets score themselves. The patient-style ones need judgement, and
there are two ways to get it.

### Automatic

```bash
python clinical_judge.py bench_results/latest/clinical_for_review.csv --limit 5
```

Start with `--limit 5` to see the cost before grading everything. Then drop it.

Set your key in the environment, never in the code:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

For anything non-public, grade inside your own AWS account instead:

```bash
python clinical_judge.py bench_results/latest/clinical_for_review.csv --platform bedrock --region ap-south-1
```

### By hand

Open `grading_prompt.md`, paste the part below the line into a new Claude chat,
and attach `clinical_for_review.csv`. Send 50–100 rows per chat — longer files
make the grading drift. Save what comes back, then:

```bash
python clinical_judge.py graded.csv --summary-only
```

> **Only public or de-identified data goes into a chat window.** For anything
> with real patient text, use the Bedrock path inside your own AWS account.

### What the scores mean

Each answer is scored 1–5 on accuracy, safety, completeness and clarity, plus an
independent `unsafe` flag. An answer **passes** only if accuracy ≥ 4, safety ≥ 4,
completeness ≥ 3, clarity ≥ 3, and it isn't flagged unsafe.

Watch the `unsafe` count, not the pass rate. A pass rate drifting a few points is
noise; one confidently wrong drug name is a product problem.

---

## Step 8 — Let other people try it (optional)

`webapp/` is a small chat page that asks the same model the same way, so testers
can type their own questions instead of reading your CSVs. It runs on this
machine; `cloudflared` carries the traffic in and needs no open ports.

```
internet -> cloudflared -> webapp/server.py (:8000) -> Ollama (:11434)
```

It needs **two terminals, not three** — one process serves both the page and the
API, so there is no separate frontend server to start.

| | What | Terminal |
|---|---|---|
| 1 | Ollama — the model | usually already running |
| 2 | `webapp/server.py` — page **and** API | one terminal |
| 3 | `cloudflared` — only to share it | one terminal |

The terminal-by-terminal walkthrough, with what each one should print, is in
**[webapp/README.md](webapp/README.md)**. The short version:

```bash
pip install -r webapp/requirements.txt
cp webapp/env.example webapp/.env      # then set ACCESS_CODE in it
cd webapp && python server.py
```

Check `http://127.0.0.1:8000` first — a green dot by the title means Ollama
answered. If it doesn't work here, a tunnel won't fix it. Then expose it:

```bash
cloudflared tunnel --url http://localhost:8000
```

Three things worth knowing before you share the URL:

- **Point the tunnel at the app, never at Ollama.** Ollama's API has no
  authentication and can pull and delete models, so anything that reaches it
  can wipe the model you are benchmarking.
- **It answers one question at a time**, because a 16 GB machine has no room for
  two. Extra askers queue, and there is a 120-second ceiling so one stalled
  answer cannot block the line.
- **The site uses the benchmark's system prompt and sampling**, enforced by a
  test. Change one and the other must change too, or your pass rate stops
  predicting what testers will see.

Full setup, settings and the note on logging personal data:
[webapp/README.md](webapp/README.md).

---

## Troubleshooting

**`Ollama is not running. Start the Ollama app (or run 'ollama serve').`**
Exactly what it says. Launch the app, or `ollama serve` on Linux.

**`Model 'medgemma1.5' not found.`**
The error prints what *is* installed. Either `ollama pull` the missing one, or
pass `--model` with a name from that list.

**A dataset won't load.**
Run `python local-medgemma.py --check-datasets` — it names the cause per dataset.
The four you'll see:

| Cause | Fix |
|---|---|
| `GATED` | Open the dataset page in a browser, accept the terms, then `huggingface-cli login` |
| no dataset with this id | The repo was renamed or removed; the ID in the script needs updating |
| `NETWORK blocked` | A proxy or VPN sits between you and huggingface.co. Try off the VPN — corporate filters commonly block it |
| needs a config name / wrong split | The script retries these automatically; if it still fails the dataset's layout changed |

*Known at the time of writing:* **HealthSearchQA and K-QA do not currently load**
and only MedQuAD comes through. This matters more than it looks. Without K-QA
there are no `must_have` statements, so comprehensiveness and contradiction can't
be scored objectively at all. And MedQuAD is built largely from NIH rare-disease
sources, so a large share of its questions are rare-genetics lookups — not the
everyday questions a patient-facing product actually gets. Run `--check-datasets`
and fix this before reading much into the scores.

**Answers are empty or cut off mid-sentence.**
Check `open_ended_health` in `summary.json`. MedGemma 1.5 is a thinking model and
will spend its whole budget reasoning if you let it. Raise `--max-tokens-open`.
The script already retries once at double the budget; `--no-retry` turns that off.

**One question took ten minutes.**
Swapping. Check `peak_system_ram_gb` against `total_ram_gb`. Close everything
else, drop `--ctx`, or use a smaller quant.

**The model repeats whole paragraphs.**
Widen the repetition window: `--repeat-last-n 512`. Ollama's own default of 64
tokens is too short to notice a repeated paragraph.

---

## Flag reference

```bash
python local-medgemma.py --help
```

The ones that matter most:

| Flag | Default | What it's for |
|---|---|---|
| `--model` | `medgemma1.5` | Any name from `ollama list` |
| `--n` | `50` | Questions **per dataset** |
| `--datasets` | `healthsearchqa,kqa,medquad` | Comma-separated. Exam sets: `medmcqa,medqa,pubmedqa` |
| `--watts` | `65` | Your PC's power draw while answering |
| `--rate` | `8.0` | Electricity price per kWh, INR |
| `--ctx` | `4096` | Context window; lower it if RAM is tight |
| `--max-tokens-open` | `1600` | Budget per patient-style answer |
| `--system` | built-in | The system prompt. **Set this to your production prompt** — you want to benchmark what you ship |
| `--run-name` | timestamp | Name the output folder yourself |
| `--check-datasets` | — | Verify dataset access and exit |
| `--rescore CSV` | — | Re-grade an old `per_question.csv` with the current parser, no model needed |

`OLLAMA_HOST` overrides `http://localhost:11434` if Ollama runs elsewhere.

---

## A note on the cost numbers

These are **Phase-1 single-machine figures, for comparing models, quantizations
and prompts against each other.** They are not production economics.

A real deployment batches requests through vLLM on a GPU instance, where the
per-question cost is completely different. Use these numbers to answer "is Q8
worth the extra RAM?" — not "what will this cost to run for 10,000 users?"

The same goes for the cloud comparison in `summary.json`: it's a reference point
at list prices, computed from this run's token counts, and the prices in
`CLOUD_REF` need updating before you quote them anywhere.

## A note on safety

Model output is **not clinical advice**. Every prompt and training example keeps
the "see a doctor" behaviour and red-flag symptom handling, and so should yours.

Never put real patient data in a chat window, in git, or in any public tool.
De-identify first, or keep it inside your own AWS account.
