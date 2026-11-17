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
  python medgemma_local_benchmark.py --model medgemma1.5 --n 50 --watts 65 --rate 8

  --watts   Average power your PC draws WHILE the model is answering.
            Best: read it from a plug-in power meter. Otherwise see the
            chat notes (HWiNFO on Windows, nvidia-smi for GPUs).
            On Linux with Intel/AMD CPUs, --rapl measures CPU energy directly.
  --rate    Your electricity price per kWh (from your bill), in rupees.

Outputs (in ./bench_results/)
  per_question.csv   every question, the model's answer, correct/incorrect, timings
  medquad_for_review.csv  patient-style questions + MedGemma's answers + NIH
                          reference answers, with empty score columns. Send this
                          file to Claude with grading_prompt.md to get it scored.
  summary.json       accuracy, speed, and cost summary
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
CLOUD_REF = {"Claude Sonnet 5 (input/output)": (2.00, 10.00)}


# ----------------------------------------------------------------------------
# 1. Datasets
# ----------------------------------------------------------------------------
def load_samples(n, seed=42):
    """Return a dict of dataset_name -> list of question dicts."""
    from datasets import load_dataset

    out = {}
    letters = "ABCD"

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

    # MedQuAD: patient-style questions with NIH reference answers (open-ended).
    try:
        ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out["medquad"] = [{
            "kind": "open",
            "question": r["Question"],
            "answer": r["Answer"],
        } for r in ds]
    except Exception as e:
        print("Skipping MedQuAD:", e)

    return out


def build_prompt(item):
    if item["kind"] == "mcq":
        opts = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", item["options"]))
        return (f"{item['question']}\n\n{opts}\n\n"
                "Answer with the single letter of the correct option (A, B, C, or D) "
                "on the first line, then one short sentence of reasoning.")
    if item["kind"] == "yesno":
        return (f"Context: {item['context']}\n\nQuestion: {item['question']}\n\n"
                "Answer with one word on the first line: yes, no, or maybe. "
                "Then one short sentence of reasoning.")
    return item["question"]  # open-ended, patient style


def grade(item, text):
    t = text.strip()
    if item["kind"] == "mcq":
        first = t.splitlines()[0] if t else ""
        # Prefer a letter at the start of the first line: "B", "B.", "(B)", "**B**"
        m = re.match(r"^[\s*(\[]*([ABCD])\b", first)
        # Otherwise look for "answer is B" / "option B" anywhere
        m = m or re.search(r"(?:answer|option)\s*(?:is|:)?\s*\(?([ABCD])\b", t, re.I)
        pred = m.group(1).upper() if m else ""
        return pred, pred == item["answer"]
    if item["kind"] == "yesno":
        m = re.search(r"\b(yes|no|maybe)\b", t.lower()[:200])
        pred = m.group(1) if m else ""
        return pred, pred == item["answer"]
    return "", None  # open-ended: needs human (clinician) review


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


def ask(model, prompt, max_tokens, ctx):
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": ctx},
    }, timeout=900)
    r.raise_for_status()
    wall = time.perf_counter() - t0
    j = r.json()
    return {
        "text": j["message"]["content"],
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
# 4. Main
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
    p.add_argument("--datasets", default="medmcqa,medqa,pubmedqa,medquad",
                   help="comma-separated; e.g. --datasets medquad to test only patient questions")
    a = p.parse_args()

    check_ollama(a.model)
    os.makedirs(a.out, exist_ok=True)

    print("Loading datasets...")
    data = load_samples(a.n)
    wanted = {d.strip() for d in a.datasets.split(",")}
    data = {k: v for k, v in data.items() if k in wanted}
    if not data:
        raise SystemExit("No datasets loaded. Check your internet connection.")

    print("Warming up (first load of the model into memory)...")
    warm = ask(a.model, "Say OK.", 5, a.ctx)
    print(f"  model load time: {warm['load_s']:.1f}s")

    ram = RamSampler(); ram.start()
    e0 = rapl_energy_uj() if a.rapl else None
    t_start = time.perf_counter()

    rows = []
    for name, items in data.items():
        max_tok = 400 if items[0]["kind"] == "open" else 120
        for i, it in enumerate(items, 1):
            res = ask(a.model, build_prompt(it), max_tok, a.ctx)
            pred, ok = grade(it, res["text"])
            rows.append({"dataset": name, "kind": it["kind"], "question": it["question"],
                         "expected": it["answer"] if it["kind"] != "open" else "",
                         "predicted": pred, "correct": ok, "answer_text": res["text"],
                         "reference_answer": it["answer"] if it["kind"] == "open" else "",
                         **{k: v for k, v in res.items() if k != "text"}})
            gen_tps = res["output_tokens"] / res["gen_s"] if res["gen_s"] else 0
            print(f"[{name} {i}/{len(items)}] {res['wall_s']:.1f}s, {gen_tps:.1f} tok/s"
                  + ("" if ok is None else f", {'correct' if ok else 'wrong'}"))

    run_s = time.perf_counter() - t_start
    ram.running = False
    e1 = rapl_energy_uj() if a.rapl else None
    rapl_kwh = (e1 - e0) / 3.6e12 if (e0 is not None and e1 is not None and e1 >= e0) else None
    if a.rapl and rapl_kwh is None:
        print("RAPL not available (try sudo, or Linux only). Falling back to --watts.")

    df = pd.DataFrame(rows)
    total_s = float(df.wall_s.sum())  # time the model was actually busy answering
    df.drop(columns=["reference_answer"]).to_csv(f"{a.out}/per_question.csv", index=False)
    open_df = df[df.kind == "open"].reset_index(drop=True)
    if len(open_df):
        review = pd.DataFrame({
            "id": [f"Q{i+1:03d}" for i in range(len(open_df))],
            "question": open_df.question,
            "reference_answer": open_df.reference_answer,
            "model_answer": open_df.answer_text,
            "model": a.model,
            "output_tokens": open_df.output_tokens,
            "latency_s": open_df.wall_s.round(1),
            # Empty columns for Claude (or a clinician) to fill in
            "accuracy_1to5": "", "safety_1to5": "", "completeness_1to5": "",
            "clarity_1to5": "", "unsafe": "", "verdict": "", "comments": "",
        })
        # utf-8-sig so the file opens correctly in Excel
        review.to_csv(f"{a.out}/medquad_for_review.csv", index=False, encoding="utf-8-sig")
        print(f"Saved {len(review)} patient-style answers for review: "
              f"{os.path.abspath(a.out)}/medquad_for_review.csv")

    acc = {d: round(100 * g["correct"].astype(bool).mean(), 1)
           for d, g in df[df.kind != "open"].groupby("dataset")}
    if "medquad" in data:
        acc["medquad"] = "graded separately (see medquad_for_review.csv)"
    gen_tps = df.output_tokens.sum() / df.gen_s.sum()
    prompt_tps = df.prompt_tokens.sum() / df.prompt_s.sum() if df.prompt_s.sum() else None
    lat = sorted(df.wall_s)

    summary = {
        "model": a.model,
        "questions": len(df),
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
    print(f"\nSaved results to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()