"""Scores the classifiers on data/test.csv and writes eval/results.md and eval/predictions.csv.

  python eval/run_eval.py                       keyword rules and TF-IDF only (seconds, CPU)
  python eval/run_eval.py --llm local           plus a local open model (default Qwen/Qwen2.5-1.5B-Instruct)
  python eval/run_eval.py --llm gemini          plus Gemini (needs Google Cloud or an API key)

Scores are split by wording: "seen" notes use words that appear in train.csv, "new" notes use words that never do.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wofc import keywords  # noqa: E402
from wofc.ml import TfidfClassifier  # noqa: E402


def load(name: str) -> list[dict]:
    with open(ROOT / "data" / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(rows: list[dict]) -> dict:
    n = len(rows)
    covered = [r for r in rows if r["pred_component"] and r["pred_failure"]]
    both = [r for r in rows if r["pred_component"] == r["component"] and r["pred_failure"] == r["failure"]]
    return {
        "n": n,
        "component": sum(r["pred_component"] == r["component"] for r in rows) / n,
        "failure": sum(r["pred_failure"] == r["failure"] for r in rows) / n,
        "both": len(both) / n,
        "labeled": len(covered) / n,
        "right_when_labeled": (len(both) / len(covered)) if covered else float("nan"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["none", "local", "gemini"], default="none")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results.md",
                    help="report path; predictions.csv is written next to it")
    ap.add_argument("--model", help="model ID for --llm (default: Qwen/Qwen2.5-1.5B-Instruct or gemini-3.5-flash)")
    args = ap.parse_args(argv)
    train, test = load("train.csv"), load("test.csv")

    methods = {"Keyword rules": keywords.classify,
               "TF-IDF + logistic regression": TfidfClassifier().fit(
                   [r["text"] for r in train], [r["component"] for r in train],
                   [r["failure"] for r in train]).classify}
    confident = TfidfClassifier(min_confidence=0.6).fit(
        [r["text"] for r in train], [r["component"] for r in train], [r["failure"] for r in train])
    methods["TF-IDF, review below 0.6 confidence"] = confident.classify
    model_name = ""
    if args.llm != "none":
        from wofc.llm import GeminiModel, LLMClassifier, LocalModel, cascade

        model = LocalModel(args.model) if args.llm == "local" and args.model else \
            LocalModel() if args.llm == "local" else GeminiModel(args.model)
        model_name = model.name
        llm = LLMClassifier(model)
        methods[f"LLM ({model.name}) + validation"] = llm.classify
        methods["Keyword rules, then the LLM for the rest"] = cascade(keywords.classify, llm.classify)

    predictions, results = [], {}
    for name, classify in methods.items():
        started = time.time()
        rows = []
        for r in test:
            p = classify(r["text"])
            rows.append({**r, "method": name, "pred_component": p.get("component"), "pred_failure": p.get("failure"),
                         "problems": "; ".join(p.get("problems", [])), "raw": p.get("raw", "")})
        predictions += rows
        results[name] = {w: score([r for r in rows if r["wording"] == w]) for w in ("seen", "new")}
        results[name]["all"] = score(rows)
        print(f"{name}: {time.time() - started:.0f}s")

    with open(args.out.with_name("predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0]))
        w.writeheader()
        w.writerows(predictions)

    pct = lambda x: "n/a" if x != x else f"{x:.0%}"  # noqa: E731
    lines = ["# Results", "",
             f"Test set: {len(test)} made-up work orders, 60 with wording seen in training and 60 with new wording. "
             f"Trained on {len(train)}.{' LLM: ' + model_name + '.' if model_name else ''}", "",
             "| Method | Wording | Both labels right | Component | Failure | Labeled automatically | Right when labeled |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, by in results.items():
        for wording in ("seen", "new", "all"):
            s = by[wording]
            lines.append(f"| {name} | {wording} | {pct(s['both'])} | {pct(s['component'])} | {pct(s['failure'])} | "
                         f"{pct(s['labeled'])} | {pct(s['right_when_labeled'])} |")
    llm_rows = [r for r in predictions if r["method"].startswith("LLM")]
    if llm_rows:
        flagged = [r for r in llm_rows if r["problems"]]
        wrong_but_valid = [r for r in llm_rows if not r["problems"] and
                           (r["pred_component"], r["pred_failure"]) != (r["component"], r["failure"])]
        lines += ["", "## The LLM's answers that went to review", "",
                  f"{len(flagged)} of {len(llm_rows)} failed validation and went to a person. "
                  f"{len(wrong_but_valid)} passed validation but were wrong.", "",
                  "| Work order | Text | Why it went to review |", "| --- | --- | --- |"]
        for r in flagged[:15]:
            lines.append(f"| {r['id']} | {r['text']} | {r['problems']} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
