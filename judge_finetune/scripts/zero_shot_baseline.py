"""Zero-shot baseline: ask an LLM whether two rationales are equivalent.

This is the comparison your fine-tuned 125M-param judge needs to beat
(or get close to) on OOD domains. If it does, that's the paper.

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/zero_shot_baseline.py \
        --csv /path/to/reformulated_full.csv \
        --provider anthropic \
        --model claude-haiku-4-5 \
        --n 500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DataConfig, load_and_split  # noqa

PROMPT = """You are evaluating whether two explanations of the same social media post convey the same meaning.

Explanation A:
{a}

Explanation B:
{b}

Do A and B express the same overall judgement and reasoning? Answer with exactly one word: "equivalent" or "unequivalent"."""


def call_anthropic(model: str, a: str, b: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}],
    )
    return msg.content[0].text.strip().lower()


def call_openai(model: str, a: str, b: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}],
    )
    return resp.choices[0].message.content.strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=500, help="sample N rows from OOD test set")
    p.add_argument("--out", default="runs/zero_shot.json")
    args = p.parse_args()

    _, _, test_ds = load_and_split(DataConfig(csv_path=args.csv))
    df = test_ds.to_pandas()
    df = df.sample(n=min(args.n, len(df)), random_state=42).reset_index(drop=True)

    call = call_anthropic if args.provider == "anthropic" else call_openai

    preds, labels, domains = [], [], []
    for i, row in df.iterrows():
        try:
            ans = call(args.model, row["text_a"], row["text_b"])
        except Exception as e:
            print(f"[{i}] error: {e}")
            continue
        # Map any "equivalent"/"unequivalent" substring; default to equivalent on garbage.
        pred = 1 if "unequivalent" in ans or "not equivalent" in ans else 0
        preds.append(pred)
        labels.append(int(row["label"]))
        domains.append(row["domain"])
        if i % 25 == 0:
            print(f"[{i}/{len(df)}] {ans}")

    preds = np.array(preds); labels = np.array(labels)
    out = {
        "provider": args.provider,
        "model": args.model,
        "n": len(preds),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_unequivalent": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "per_domain": {},
    }
    for dom in sorted(set(domains)):
        mask = np.array([d == dom for d in domains])
        out["per_domain"][dom] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], preds[mask])),
            "f1_macro": float(f1_score(labels[mask], preds[mask], average="macro")),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
