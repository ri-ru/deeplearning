"""Stage 11: LLM zero-shot upper bound on the Stage 10 human eval (n=100).

Several frontier LLMs are asked, zero-shot, whether each (explanation, comment)
pair conveys the same conclusion. Tells us the ceiling: if even a top LLM can
only hit ~80% on this set, our multi_gen at 81% is near state-of-art. If they
hit 95%+, we have room.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import LABEL2ID, REPO, save_json  # noqa: E402

EVAL_CSV = REPO / "overnight" / "results" / "stage10" / "claude_human_eval.csv"
OUT_DIR = REPO / "overnight" / "results" / "stage11"


PROMPT = """You will see two reactions to a social media post. Decide whether they reach
the SAME conclusion about whether the post is hate speech, or DIFFERENT
conclusions.

Reaction A: {a}

Reaction B: {b}

Answer with exactly one word: equivalent (same conclusion) or unequivalent
(different conclusion).

Answer:"""


def call_openrouter(model: str, prompt: str, max_tokens: int = 8) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower()


def judge_one_model(model_id: str, df: pd.DataFrame) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    raws: list[str] = []
    for i, r in df.iterrows():
        try:
            ans = call_openrouter(model_id, PROMPT.format(a=r["explanation"], b=r["comment"]))
        except Exception as e:
            print(f"  [{i}] {model_id}: error: {e}", flush=True)
            continue
        first = ans.split()[0] if ans else ""
        pred = 1 if "uneq" in first or "diff" in first else 0
        y_pred.append(pred)
        y_true.append(LABEL2ID[r["label_verified"]])
        raws.append(ans)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(df)}] {model_id}", flush=True)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "model": model_id,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm,
        "acc_equivalent": float((y_pred[y_true == 0] == 0).mean()) if (y_true == 0).any() else None,
        "acc_unequivalent": float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else None,
        "raw_samples": raws[:10],
    }


MODELS = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v3.1",
    "anthropic/claude-haiku-4.5",  # the generator itself; sanity check
    "qwen/qwen-2.5-72b-instruct",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[stage11] OPENROUTER_API_KEY not set; abort.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EVAL_CSV)
    df = df[df["label_verified"].isin(LABEL2ID)].reset_index(drop=True)
    print(f"[stage11] eval set: {len(df)} pairs from {EVAL_CSV}", flush=True)
    print(f"[stage11] label balance: {dict(df['label_verified'].value_counts())}", flush=True)

    results = {}
    for model_id in args.models.split(","):
        model_id = model_id.strip()
        print(f"\n=== {model_id} ===", flush=True)
        t0 = time.time()
        try:
            res = judge_one_model(model_id, df)
            results[model_id] = res
            print(f"  acc={res['accuracy']:.3f}  f1={res['f1_macro']:.3f}  "
                  f"eq={res['acc_equivalent']:.2f}/uneq={res['acc_unequivalent']:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results[model_id] = {"error": str(e)}
        save_json(OUT_DIR / "ceiling.json", results)

    # OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {ts}  STAGE 11: LLM zero-shot ceiling on Stage 10 set (n=100)\n",
        "Each model called once per pair via OpenRouter, zero-shot. Same 100 Claude-Haiku-generated pairs.",
        "",
        "| model | acc | f1_macro | eq / uneq |",
        "|---|---|---|---|",
    ]
    for m, r in results.items():
        if "error" in r:
            lines.append(f"| {m} | ERROR: {r['error']} | | |")
        else:
            lines.append(f"| {m} | {r['accuracy']*100:.1f}% | {r['f1_macro']:.3f} | "
                         f"{r['acc_equivalent']*100:.0f}%/{r['acc_unequivalent']*100:.0f}% |")
    lines.append("")
    lines.append("Compare to Stage 10 trained classifiers: poster 59% / v04 66% / multi_gen 81%.")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[stage11] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
