"""
MedGemma local benchmark: accuracy + speed + memory + inference cost.

What it does
  1. Downloads public medical Q&A datasets from Hugging Face (small samples).
  2. Sends each question to MedGemma running locally in Ollama.
  3. Measures accuracy, speed (tokens/sec), latency, and RAM use.
  4. Converts energy use into cost (electricity + PC wear) per question
     and per 1M tokens, so you can compare against cloud options.

Setup (one time)
  1. Install Ollama:  https://ollama.com/download
  2. Download the model (pick one; see README section in the chat):
       ollama pull medgemma1.5
     or a specific quantization from Hugging Face:
       ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
  3. pip install datasets requests pandas psutil

Run
  python local-medgemma.py --model medgemma1.5 --n 50 --watts 65 --rate 8

  --watts   Average power your PC draws WHILE the model is answering.
            Best: read it from a plug-in power meter. Otherwise see the
            chat notes (HWiNFO on Windows, nvidia-smi for GPUs).
            On Linux with Intel/AMD CPUs, --rapl measures CPU energy directly.
  --rate    Your electricity price per kWh (from your bill), in rupees.
  --think   Let the model reason before answering (slower, needs a much
            bigger token budget). Off by default: MedGemma 1.5 is a thinking
            model, and its reasoning block eats the whole budget before the
            answer appears.

Re-score an old run with the current parser (no model needed):
  python local-medgemma.py --rescore bench_results/per_question.csv

Datasets
  Default (the product use case - how people actually ask):
    healthsearchqa  real consumer health search queries (no reference answer)
    kqa             real patient questions + physician-written facts a good
                    answer must contain -> automatically scoreable
    medquad         patient questions with NIH reference answers
  Exam sets, as a cheap regression check (--datasets medmcqa,medqa,pubmedqa):
    medmcqa, medqa (multiple choice), pubmedqa (yes/no/maybe)

Outputs (in ./bench_results/)
  per_question.csv        every question, the answer, correct/incorrect, timings
  clinical_for_review.csv patient-style questions + the model's answers, with
                          empty score columns. Grade it either way:
                            python clinical_judge.py bench_results/clinical_for_review.csv
                          or send it to Claude with grading_prompt.md by hand.
  summary.json            accuracy, speed, and cost summary
"""

import argparse
import json
import os
import re
import statistics
import threading
import time

import pandas as pd
import psutil
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NS = 1e9  # Ollama reports durations in nanoseconds

# Reference cloud price for comparison (per 1M tokens, USD). Update before use.
CLOUD_REF = {
    "Claude Opus 5.5": (4.00, 20.00),
    "Claude Sonnet 5.5": (2.00, 10.00),
    "Claude Haiku 5.5": (0.10, 0.50),
}


# ----------------------------------------------------------------------------
# 1. Datasets
# ----------------------------------------------------------------------------
def _field(row, *names):
    """First present, non-empty field from a row, trying several spellings."""
    for name in names:
        if name in row and row[name] not in (None, "", []):
            return row[name]
    return ""


def _as_list(v):
    """K-QA statement columns come back as a list, or as a JSON / newline string."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(x).strip() for x in json.loads(s) if str(x).strip()]
        except Exception:
            pass
    return [p.strip() for p in re.split(r"\n|\s*\|\s*", s) if p.strip()]


def _try_load(label, candidates, seed, n):
    """Try each (repo_id, config, split) in turn; return the first that loads.

    Dataset repos get renamed and re-uploaded, so the first id is the one that
    worked most recently and the rest are fallbacks. The id that succeeded is
    printed so you know what you actually benchmarked against.
    """
    from datasets import load_dataset

    errors = []
    for repo, config, split in candidates:
        try:
            ds = load_dataset(repo, config, split=split) if config else \
                load_dataset(repo, split=split)
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
            print(f"  {label}: loaded {len(ds)} from {repo}"
                  f"{'/' + config if config else ''} [{split}]")
            return ds
        except Exception as e:
            errors.append(f"{repo}: {type(e).__name__}: {str(e)[:120]}")
    print(f"  {label}: SKIPPED - none of the candidate repos loaded")
    for e in errors:
        print(f"      {e}")
    return None


def load_samples(n, seed=42):
    """Return a dict of dataset_name -> list of question dicts."""
    out = {}
    letters = "ABCD"

    # ---- Patient-style questions: the day-to-day clinical use case -----------

    # HealthSearchQA (Google, Med-PaLM): real consumer health search queries.
    # No reference answer - these are graded by rubric, not by matching.
    ds = _try_load("healthsearchqa", [
        ("katielink/healthsearchqa", None, "train"),
        ("openlifescienceai/healthsearchqa", None, "train"),
        ("Kabatubare/healthsearchqa", None, "train"),
    ], seed, n)
    if ds is not None:
        out["healthsearchqa"] = [{
            "kind": "open",
            "question": str(_field(r, "question", "Question", "query", "text")),
            "answer": "",            # no reference answer exists for this set
            "must_have": [], "nice_to_have": [],
        } for r in ds]

    # K-QA: real patient questions, with physician-written statements that a
    # good answer MUST contain. Those statements make it automatically
    # scoreable (comprehensiveness + contradiction), unlike the other two.
    ds = _try_load("kqa", [
        ("Itaykhealth/K-QA", None, "train"),
        ("Itaykhealth/K-QA", "default", "train"),
        ("nvidia/K-QA", None, "train"),
    ], seed, n)
    if ds is not None:
        out["kqa"] = [{
            "kind": "open",
            "question": str(_field(r, "Question", "question")),
            "answer": str(_field(r, "Free_form_answer", "free_form_answer", "answer")),
            "must_have": _as_list(_field(r, "Must_have", "must_have")),
            "nice_to_have": _as_list(_field(r, "Nice_to_have", "nice_to_have")),
        } for r in ds]

    # MedQuAD: patient-style questions with NIH reference answers.
    ds = _try_load("medquad", [
        ("keivalya/MedQuad-MedicalQnADataset", None, "train"),
        ("lavita/MedQuAD", None, "train"),
    ], seed, n)
    if ds is not None:
        out["medquad"] = [{
            "kind": "open",
            "question": str(_field(r, "Question", "question")),
            "answer": str(_field(r, "Answer", "answer")),
            "must_have": [], "nice_to_have": [],
        } for r in ds]

    # ---- Exam questions: cheap regression check, not the product use case ----
    from datasets import load_dataset

    # MedMCQA: Indian medical entrance exam questions (AIIMS / NEET-PG style).
    # The public test split has no answers, so we use the validation split.
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out["medmcqa"] = [{
            "kind": "mcq",
            "question": r["question"],
            "options": [r["opa"], r["opb"], r["opc"], r["opd"]],
            "answer": letters[int(r["cop"])],
            "subject": r.get("subject_name", ""),
        } for r in ds]
    except Exception as e:
        print("Skipping MedMCQA:", e)

    # MedQA (USMLE, 4 options): US medical licensing exam questions.
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out["medqa"] = [{
            "kind": "mcq",
            "question": r["question"],
            "options": [r["options"][k] for k in letters],
            "answer": r["answer_idx"],
            "subject": "",
        } for r in ds]
    except Exception as e:
        print("Skipping MedQA:", e)

    # PubMedQA (expert-labeled): yes / no / maybe from a research abstract.
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out["pubmedqa"] = [{
            "kind": "yesno",
            "question": r["question"],
            "context": " ".join(r["context"]["contexts"]),
            "answer": r["final_decision"].lower(),
        } for r in ds]
    except Exception as e:
        print("Skipping PubMedQA:", e)

    return out


def build_prompt(item):
    if item["kind"] == "mcq":
        opts = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", item["options"]))
        return (f"{item['question']}\n\n{opts}\n\n"
                "Reply with the correct option only, in exactly this format:\n"
                "Answer: X\n"
                "where X is A, B, C, or D. Do not explain.")
    if item["kind"] == "yesno":
        return (f"Context: {item['context']}\n\nQuestion: {item['question']}\n\n"
                "Reply in exactly this format:\n"
                "Answer: X\n"
                "where X is yes, no, or maybe. Do not explain.")
    return item["question"]  # open-ended, patient style: ask it exactly as a user would


# Day-to-day clinical Q&A is a product with a system prompt, not a raw model
# call. Benchmark the thing you ship: change this to your production prompt.
DEFAULT_SYSTEM = (
    "You are a medical information assistant answering questions from patients. "
    "Begin your reply with the answer itself. Do not write out your reasoning, "
    "a plan, or numbered thinking steps first. "
    "Answer in plain language a non-medical person can follow. Be accurate and "
    "specific. Put any red-flag symptoms that need urgent care, and when the "
    "person should see a doctor, near the start rather than the end. Say "
    "\"I don't know\" rather than guessing a gene name, a number, or a drug "
    "name you are not sure of. Do not diagnose, and do not give a dose or a "
    "prescription. If the question is unclear or unsafe to answer without an "
    "examination, say so."
)


# MedGemma 1.5 / Gemma-3 thinking models wrap their reasoning in <unusedNN>
# marker tokens. When Ollama does not strip them, the block leaks into the
# answer text -- and if the token budget runs out mid-thought there is no
# answer at all. Everything up to the closing marker is reasoning, not answer.
THINK_OPEN = re.compile(r"<unused9[0-9]>\s*(?:thought|think)?\s*", re.I)
THINK_CLOSE = re.compile(r"<unused9[0-9]>|</think(?:ing)?>|</thought>", re.I)


def split_thinking(text):
    """Return (answer_text, thinking_text, thinking_unterminated).

    thinking_unterminated=True means the model was still reasoning when it ran
    out of tokens, so there is no answer to grade -- a budget problem, not a
    knowledge problem.
    """
    t = (text or "").strip()
    # Ollama >= 0.9 style tags, if the template emits them
    m = re.match(r"\s*<think(?:ing)?>(.*?)</think(?:ing)?>\s*", t, re.S | re.I)
    if m:
        return t[m.end():].strip(), m.group(1).strip(), False
    m = THINK_OPEN.search(t)
    if not m:
        return t, "", False
    rest = t[m.end():]
    close = THINK_CLOSE.search(rest)
    if close:
        return rest[close.end():].strip(), rest[:close.start()].strip(), False
    return "", rest.strip(), True  # ran out of budget inside the thought block


def _match_option(text, options):
    """Last resort: the model wrote out an option verbatim instead of a letter."""
    low = text.lower()
    hits = [l for l, o in zip("ABCD", options)
            if o and len(o.strip()) > 3 and o.strip().lower() in low]
    return hits[0] if len(hits) == 1 else ""


def grade(item, text, thinking_fallback=True, thinking_field=""):
    """Return (predicted, correct_or_None). predicted == "" means unparseable.

    thinking_field: reasoning Ollama returned in its own message.thinking field
    (so it never appears in `text`).
    """
    answer, thinking, unterminated = split_thinking(text)
    thinking = thinking or (thinking_field or "")
    if item["kind"] == "open":
        return "", None  # open-ended: needs clinician / LLM-judge review

    def parse(t, strict=False):
        """strict=True: only an explicit "the answer is X" counts. Used when
        reading a reasoning block, where stray letters and a loose "yes" are
        part of the model weighing options, not its conclusion."""
        t = t.strip()
        if not t:
            return ""
        if item["kind"] == "mcq":
            for line in ([] if strict else t.splitlines()):
                line = line.strip()
                if not line:
                    continue
                # "Answer: B", "B", "B.", "(B)", "**B**", "- B)", "### B"
                m = re.match(r"^(?:answer|option|choice)?\s*[:\-]?\s*[*#>(\[\s]*"
                             r"([ABCD])(?=[\s.):,*\]]|$)", line, re.I)
                if m:
                    return m.group(1).upper()
                break  # only the first non-empty line counts as a bare letter
            m = (re.search(r"(?:answer|option|choice)\s*(?:is|:|=)\s*\**\(?([ABCD])\b", t, re.I)
                 or re.search(r"\b([ABCD])\s*(?:is|would be)\s+(?:the\s+)?correct", t, re.I)
                 or re.search(r"\*\*\(?([ABCD])\)?\*\*", t))
            if m:
                return m.group(1).upper()
            return "" if strict else _match_option(t, item.get("options") or [])
        # yes / no / maybe
        for line in ([] if strict else t.splitlines()):
            line = line.strip().lower()
            if not line:
                continue
            m = re.match(r"^(?:answer|final answer)?\s*[:\-]?\s*[*#>\s]*(yes|no|maybe)\b", line)
            if m:
                return m.group(1)
            break
        m = re.search(r"(?:answer|conclusion)\s*(?:is|:|=)\s*\**(yes|no|maybe)\b", t, re.I)
        if not m and not strict:
            m = re.search(r"\b(yes|no|maybe)\b", t, re.I)
        return m.group(1).lower() if m else ""

    pred = parse(answer)
    # If the model never got out of its thought block, its conclusion may still
    # be in there. Only used when there is no answer text at all, and flagged
    # in the CSV so you can see how much of the score rests on it.
    if not pred and thinking_fallback and thinking and not answer:
        pred = parse(thinking[-600:], strict=True)
    return pred, (pred == item["answer"] if pred else False)


# ----------------------------------------------------------------------------
# 2. Running the model + measuring
# ----------------------------------------------------------------------------
class RamSampler(threading.Thread):
    """Samples system RAM in the background to find the peak."""
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self.running = interval, 0, True

    def run(self):
        while self.running:
            self.peak = max(self.peak, psutil.virtual_memory().used)
            time.sleep(self.interval)


def rapl_energy_uj():
    """CPU package energy counter on Linux (Intel/AMD). Returns None if unavailable."""
    path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    try:
        with open(path) as f:
            return int(f.read())
    except Exception:
        return None


_THINK_FIELD_OK = True  # cleared if this Ollama / model rejects the "think" field


def ask(model, prompt, max_tokens, ctx, think=False, system=None):
    global _THINK_FIELD_OK
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": ctx},
    }
    if _THINK_FIELD_OK:
        payload["think"] = bool(think)
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    if r.status_code == 400 and _THINK_FIELD_OK:
        # Older Ollama, or a model with no thinking support: retry without it.
        # The reasoning block (if any) is then stripped from the text instead.
        _THINK_FIELD_OK = False
        print("  note: this Ollama/model does not accept the 'think' field; "
              "falling back to stripping the reasoning block from the output.")
        payload.pop("think")
        t0 = time.perf_counter()
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    wall = time.perf_counter() - t0
    j = r.json()
    msg = j.get("message", {})
    return {
        "text": msg.get("content", ""),
        # Ollama puts reasoning here when it understands the model's think tags
        "thinking": msg.get("thinking") or "",
        "done_reason": j.get("done_reason", ""),
        "wall_s": wall,
        "prompt_tokens": j.get("prompt_eval_count", 0),
        "output_tokens": j.get("eval_count", 0),
        "prompt_s": j.get("prompt_eval_duration", 0) / NS,   # ~ time to first token
        "gen_s": j.get("eval_duration", 0) / NS,
        "load_s": j.get("load_duration", 0) / NS,
    }


def check_ollama(model):
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=5).json()
    except Exception:
        raise SystemExit("Ollama is not running. Start the Ollama app (or run `ollama serve`).")
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == model or n.startswith(model + ":") for n in names):
        raise SystemExit(f"Model '{model}' not found. Run: ollama pull {model}\nInstalled: {names}")


# ----------------------------------------------------------------------------
# 3. Cost maths
# ----------------------------------------------------------------------------
def cost_summary(total_s, prompt_tok, out_tok, n_questions, watts, rate,
                 pc_cost, pc_years, usd_inr, rapl_kwh=None):
    kwh = rapl_kwh if rapl_kwh is not None else watts * total_s / 3.6e6
    electricity = kwh * rate
    # PC wear: spread the PC's price over its working life (if used for this 24/7)
    life_s = pc_years * 365 * 24 * 3600
    hardware = pc_cost / life_s * total_s if pc_cost else 0.0
    total = electricity + hardware
    tokens = prompt_tok + out_tok
    cloud = {}
    for name, (pin, pout) in CLOUD_REF.items():
        usd = prompt_tok / 1e6 * pin + out_tok / 1e6 * pout
        cloud[name] = round(usd * usd_inr / n_questions * 1000, 2)
    return {
        "energy_kwh": round(kwh, 5),
        "energy_source": "RAPL (measured CPU only)" if rapl_kwh is not None else f"{watts} W x time",
        "electricity_inr": round(electricity, 4),
        "hardware_wear_inr": round(hardware, 4),
        "total_inr": round(total, 4),
        "inr_per_1000_questions": round(total / n_questions * 1000, 2),
        "inr_per_1M_total_tokens": round(total / tokens * 1e6, 2) if tokens else None,
        "inr_per_1M_output_tokens": round(total / out_tok * 1e6, 2) if out_tok else None,
        "cloud_reference_inr_per_1000_same_questions": cloud,
        "max_questions_per_day_one_user_at_a_time": int(86400 / (total_s / n_questions)),
    }


# ----------------------------------------------------------------------------
# 4. Scoring report
# ----------------------------------------------------------------------------
BASELINE = {"mcq": 25.0, "yesno": 33.3}  # score from guessing at random


def accuracy_breakdown(df):
    """Per-dataset accuracy that keeps unparseable answers visible.

    An answer the parser could not read counts as wrong in accuracy_percent
    (that is the honest end-to-end number), but it is also reported on its own
    so a harness problem can never be mistaken for a weak model.
    """
    out = {}
    for d, g in df[df.kind != "open"].groupby("dataset"):
        n = len(g)
        parsed = g[g.predicted.astype(str).str.strip() != ""]
        correct = int(g.correct.astype(bool).sum())
        out[d] = {
            "accuracy_percent": round(100 * correct / n, 1) if n else None,
            "accuracy_of_parsed_percent": (
                round(100 * int(parsed.correct.astype(bool).sum()) / len(parsed), 1)
                if len(parsed) else None),
            "random_baseline_percent": BASELINE.get(g.kind.iloc[0]),
            "questions": n,
            "unparsed": n - len(parsed),
            "truncated": int(g.truncated.astype(bool).sum()) if "truncated" in g else None,
            "answer_read_from_reasoning_block": (
                int(g.answer_from_reasoning.astype(bool).sum())
                if "answer_from_reasoning" in g else None),
        }
    return out


def warn_about_quality(acc):
    """Print loud warnings when the numbers describe the harness, not the model."""
    msgs = []
    for d, s in acc.items():
        if not isinstance(s, dict) or "accuracy_percent" not in s:
            continue  # open-ended sets are graded by rubric, not scored here
        n = s["questions"]
        if s["unparsed"] and s["unparsed"] / n > 0.05:
            msgs.append(f"{d}: {s['unparsed']}/{n} answers could not be read by the parser. "
                        f"Accuracy is understated. Check answer_text in per_question.csv.")
        if s.get("truncated") and s["truncated"] / n > 0.05:
            msgs.append(f"{d}: {s['truncated']}/{n} answers hit the token budget. "
                        f"Re-run with a larger --max-tokens.")
        base = s["random_baseline_percent"]
        if base and s["accuracy_percent"] is not None and s["accuracy_percent"] < base:
            msgs.append(f"{d}: {s['accuracy_percent']}% is below the {base}% you would get by "
                        f"guessing. Something is wrong with the run, not just the model.")
    if msgs:
        print("\n" + "=" * 70)
        for m in msgs:
            print("WARNING  " + m)
        print("=" * 70)


def rescore(path):
    """Re-grade a saved per_question.csv with the current parser.

    Lets you recover a run that was scored by an older/looser parser without
    paying for the model time again. Writes <name>_rescored.csv next to it.
    """
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    df = pd.read_csv(path, engine="python").fillna("")
    preds, oks = [], []
    for _, r in df.iterrows():
        item = {"kind": r["kind"], "answer": str(r["expected"]),
                "options": str(r.get("options", "")).split(" | ")}
        pred, ok = grade(item, str(r["answer_text"]))
        preds.append(pred)
        oks.append(ok)
    df["predicted"], df["correct"] = preds, oks
    if "truncated" not in df:
        df["truncated"] = [bool(split_thinking(str(t))[2]) for t in df.answer_text]
    acc = accuracy_breakdown(df)
    out = re.sub(r"\.csv$", "", path) + "_rescored.csv"
    df.to_csv(out, index=False)
    print(json.dumps(acc, indent=2))
    warn_about_quality(acc)
    print(f"\nRe-scored rows written to {os.path.abspath(out)}")
    return acc


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="medgemma1.5")
    p.add_argument("--n", type=int, default=50, help="questions per dataset")
    p.add_argument("--watts", type=float, default=65, help="avg PC power while answering")
    p.add_argument("--rate", type=float, default=8.0, help="electricity price per kWh (INR)")
    p.add_argument("--pc-cost", type=float, default=0, help="PC price in INR (0 = ignore wear)")
    p.add_argument("--pc-years", type=float, default=4)
    p.add_argument("--usd-inr", type=float, default=88.0, help="exchange rate for cloud comparison")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--rapl", action="store_true", help="Linux: measure CPU energy directly")
    p.add_argument("--out", default="bench_results")
    p.add_argument("--datasets", default="healthsearchqa,kqa,medquad",
                   help="comma-separated. Default is the patient-style sets (the product "
                        "use case). Exam sets: medmcqa,medqa,pubmedqa")
    p.add_argument("--system", default=DEFAULT_SYSTEM,
                   help="system prompt used for patient-style questions "
                        "(set to your production prompt); '' for none")
    p.add_argument("--think", action="store_true",
                   help="let the model reason before answering (slower, needs a big budget)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="token budget per graded answer (default 160, or 1024 with --think)")
    p.add_argument("--max-tokens-open", type=int, default=None,
                   help="token budget per patient-style answer (default 1600, 2400 with --think)")
    p.add_argument("--no-retry", action="store_true",
                   help="do not re-ask when an answer comes back as reasoning only")
    p.add_argument("--retry-factor", type=float, default=2.0,
                   help="budget multiplier for that retry (default 2x)")
    p.add_argument("--rescore", metavar="CSV",
                   help="re-grade an existing per_question.csv with the current parser and exit")
    a = p.parse_args()

    if a.rescore:
        rescore(a.rescore)
        return

    # A thinking model needs room for the reasoning AND the answer. Too small a
    # budget scores the truncation, not the model.
    #
    # 1600 is not generous, it is the floor. MedGemma 1.5 emits its <unusedNN>
    # thought block even with think=False, and on the 17 Nov 2026 run that block
    # alone ate 400-600 tokens: at a 600 budget, 35 of 50 answers were pure
    # reasoning with no answer attached. Budget for the thought plus the answer.
    if a.max_tokens is None:
        a.max_tokens = 1024 if a.think else 160
    if a.max_tokens_open is None:
        a.max_tokens_open = 2400 if a.think else 1600

    check_ollama(a.model)
    os.makedirs(a.out, exist_ok=True)

    print("Loading datasets...")
    data = load_samples(a.n)
    wanted = {d.strip() for d in a.datasets.split(",")}
    data = {k: v for k, v in data.items() if k in wanted}
    if not data:
        raise SystemExit("No datasets loaded. Check your internet connection.")

    print("Warming up (first load of the model into memory)...")
    warm = ask(a.model, "Say OK.", 5, a.ctx, think=a.think)
    print(f"  model load time: {warm['load_s']:.1f}s")
    print(f"  reasoning: {'on' if a.think else 'off'}; "
          f"budget {a.max_tokens} tokens (graded) / {a.max_tokens_open} (open-ended)")

    ram = RamSampler(); ram.start()
    e0 = rapl_energy_uj() if a.rapl else None
    t_start = time.perf_counter()

    rows = []
    for name, items in data.items():
        max_tok = a.max_tokens_open if items[0]["kind"] == "open" else a.max_tokens
        for i, it in enumerate(items, 1):
            sys_prompt = a.system if (it["kind"] == "open" and a.system) else None
            res = ask(a.model, build_prompt(it), max_tok, a.ctx,
                      think=a.think, system=sys_prompt)
            # Reasoning either comes back in its own field or inline in the text
            answer, inline_think, unterminated = split_thinking(res["text"])
            thinking = res["thinking"] or inline_think

            # The model reasoned until the budget ran out and never answered.
            # That is a harness result, not a model result, so pay once more
            # for a real answer instead of scoring a 1.
            retried, wasted_s = False, 0.0
            if unterminated and not a.no_retry:
                big = int(max_tok * a.retry_factor)
                print(f"    reasoning-only at {max_tok} tokens, retrying at {big}")
                res2 = ask(a.model, build_prompt(it), big, a.ctx,
                           think=a.think, system=sys_prompt)
                ans2, think2, unterm2 = split_thinking(res2["text"])
                retried = True
                # Keep the retry only if it actually got us an answer. Either
                # way the discarded call burned real electricity, so its time
                # stays in the cost total (but out of the latency figures).
                if ans2.strip():
                    wasted_s = res["wall_s"]
                    res, answer, unterminated = res2, ans2, unterm2
                    thinking = res2["thinking"] or think2
                else:
                    wasted_s = res2["wall_s"]
                    print("    still reasoning-only; keeping the first response")
            pred, ok = grade(it, res["text"], thinking_field=res["thinking"])
            pred_clean, _ = grade(it, res["text"], thinking_fallback=False)
            truncated = res["done_reason"] == "length" or unterminated
            rows.append({"dataset": name, "kind": it["kind"], "question": it["question"],
                         "expected": it["answer"] if it["kind"] != "open" else "",
                         "predicted": pred, "correct": ok,
                         "answer_text": answer or res["text"],
                         "clean_answer": answer,
                         "thinking_text": thinking,
                         "truncated": truncated,
                         "no_answer_only_thinking": unterminated,
                         "retried": retried,
                         "retry_wasted_s": round(wasted_s, 2),
                         "answer_from_reasoning": bool(pred) and not pred_clean,
                         "options": " | ".join(it.get("options", [])),
                         "must_have": " | ".join(it.get("must_have", [])),
                         "nice_to_have": " | ".join(it.get("nice_to_have", [])),
                         "reference_answer": it["answer"] if it["kind"] == "open" else "",
                         **{k: v for k, v in res.items() if k not in ("text", "thinking")}})
            gen_tps = res["output_tokens"] / res["gen_s"] if res["gen_s"] else 0
            flag = " [TRUNCATED]" if truncated else ""
            verdict = "" if ok is None else (
                ", correct" if ok else (", UNPARSED" if not pred else ", wrong"))
            print(f"[{name} {i}/{len(items)}] {res['wall_s']:.1f}s, "
                  f"{gen_tps:.1f} tok/s{verdict}{flag}")

    run_s = time.perf_counter() - t_start
    ram.running = False
    e1 = rapl_energy_uj() if a.rapl else None
    rapl_kwh = (e1 - e0) / 3.6e12 if (e0 is not None and e1 is not None and e1 >= e0) else None
    if a.rapl and rapl_kwh is None:
        print("RAPL not available (try sudo, or Linux only). Falling back to --watts.")

    df = pd.DataFrame(rows)
    # Time the model was actually busy answering, including retries we threw away
    total_s = float(df.wall_s.sum()) + float(df.get("retry_wasted_s", pd.Series([0.0])).sum())
    df.drop(columns=["reference_answer"]).to_csv(f"{a.out}/per_question.csv", index=False)
    open_df = df[df.kind == "open"].reset_index(drop=True)
    review_path = f"{a.out}/clinical_for_review.csv"
    if len(open_df):
        review = pd.DataFrame({
            "id": [f"Q{i+1:03d}" for i in range(len(open_df))],
            "dataset": open_df.dataset,
            "question": open_df.question,
            "reference_answer": open_df.reference_answer,
            # the answer with any reasoning block stripped out
            "model_answer": open_df.clean_answer.where(
                open_df.clean_answer.str.strip().astype(bool), open_df.answer_text),
            # K-QA only: physician-written facts a good answer must contain
            "must_have": open_df.must_have,
            "nice_to_have": open_df.nice_to_have,
            "model": a.model,
            "truncated": open_df.truncated,
            "output_tokens": open_df.output_tokens,
            "latency_s": open_df.wall_s.round(1),
            # Empty columns for the judge (or a clinician) to fill in
            "accuracy_1to5": "", "safety_1to5": "", "completeness_1to5": "",
            "clarity_1to5": "", "unsafe": "", "verdict": "", "comments": "",
        })
        # utf-8-sig so the file opens correctly in Excel
        review.to_csv(review_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(review)} patient-style answers for grading: "
              f"{os.path.abspath(review_path)}")
        print("  Grade them with:  python clinical_judge.py "
              f"{review_path}   (automatic)")
        print("  or send that file to Claude with grading_prompt.md  (manual)")

    # Is this run even gradeable? A patient-style answer that got cut off, or
    # that is nothing but a reasoning block, measures the token budget rather
    # than the model. Say so here instead of letting it sink the rubric scores.
    open_health = {}
    if len(open_df):
        n = len(open_df)
        no_answer = int(open_df.no_answer_only_thinking.sum())
        cut = int(open_df.truncated.sum())
        open_health = {
            "answers": n,
            "reasoning_only_no_answer": no_answer,
            "truncated": cut,
            "retried_for_more_budget": int(open_df.get("retried", pd.Series(dtype=bool)).sum()),
            "budget_used_fully_percent": round(100 * cut / n, 1),
        }
        if no_answer:
            print(f"\nWARNING: {no_answer}/{n} patient-style answers were reasoning only, "
                  f"with no answer attached. Raise --max-tokens-open above "
                  f"{a.max_tokens_open}.")
        if cut / n > 0.25:
            print(f"\nWARNING: {cut}/{n} patient-style answers hit the "
                  f"{a.max_tokens_open}-token ceiling and were cut off mid-sentence. "
                  "Rubric scores from this run measure the budget, not the model.")

    acc = accuracy_breakdown(df)
    for name, items in data.items():
        if items and items[0]["kind"] == "open":
            acc[name] = {"questions": len(items),
                         "scored_by": "rubric grading - see clinical_for_review.csv"}
    gen_tps = df.output_tokens.sum() / df.gen_s.sum()
    prompt_tps = df.prompt_tokens.sum() / df.prompt_s.sum() if df.prompt_s.sum() else None
    lat = sorted(df.wall_s)

    summary = {
        "model": a.model,
        "questions": len(df),
        "datasets": list(data),
        "reasoning_enabled": a.think,
        "max_tokens": {"graded": a.max_tokens, "open_ended": a.max_tokens_open},
        "system_prompt": a.system,
        "open_ended_health": open_health,
        "accuracy_percent": acc,
        "speed": {
            "output_tokens_per_sec": round(gen_tps, 1),
            "prompt_tokens_per_sec": round(prompt_tps, 1) if prompt_tps else None,
            "avg_time_to_first_token_s": round(df.prompt_s.mean(), 2),
            "latency_p50_s": round(statistics.median(lat), 2),
            "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2),
            "avg_output_tokens": round(df.output_tokens.mean(), 1),
            "avg_prompt_tokens": round(df.prompt_tokens.mean(), 1),
        },
        "memory": {
            "peak_system_ram_gb": round(ram.peak / 1e9, 2),
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "cost": cost_summary(total_s, int(df.prompt_tokens.sum()), int(df.output_tokens.sum()),
                             len(df), a.watts, a.rate, a.pc_cost, a.pc_years, a.usd_inr, rapl_kwh),
        "total_run_time_min": round(run_s / 60, 1),
    }
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    warn_about_quality(acc)
    print(f"\nSaved results to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()