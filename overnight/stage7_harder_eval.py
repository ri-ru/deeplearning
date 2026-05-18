"""Stage 7: generate a fresh held-out human-style eval set.

The original manual + Claude sets total 40 pairs, which is small. This stage
uses GPT-4o-mini (already in the project's API quota) to write ~50 fresh
human-style comments in two distinct voices we haven't seen yet:
  - "skeptical reader" voice: argues against the AI's rationale
  - "agreeing reader" voice: agrees with the AI's rationale in plain language

For each sampled OOD-test post + Qwen explanation, we generate ONE comment.
Half the time we ask GPT for an *agreeing* comment ("equivalent" label),
half the time *disagreeing* ("unequivalent"). We then verify each generated
comment with a simple consistency check (GPT-4o-mini reads the original
explanation and the new comment and answers SAME/DIFFERENT). Only pairs
with verified labels are kept.

The resulting CSV is run through both the baseline (poster) checkpoint and
the v04 winner.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, TEST_DOMAINS, save_json, _metrics_from_arrays, _batched_forward,
)

OUT_CSV = REPO / "overnight" / "results" / "stage7" / "harder_eval.csv"
RESULTS_DIR = REPO / "overnight" / "results" / "stage7"


AGREE_PROMPT = """You are a human reader giving your own short reaction to a social-media post.
You have just seen an AI's explanation of why this post is or isn't hate speech.

Your job: write a 1-2 sentence comment that AGREES with the AI's reasoning
but uses YOUR OWN words. Sound like a real Reddit reader, not an academic.
Casual, contractions OK, no markdown, no quotes.

Post: {post}
AI's explanation: {explanation}

Your reaction (1-2 sentences, agreeing in your own words):"""

DISAGREE_PROMPT = """You are a human reader giving your own short reaction to a social-media post.
You have just seen an AI's explanation of why this post is or isn't hate speech.

Your job: write a 1-2 sentence comment that DISAGREES with the AI's reasoning.
You should arrive at the opposite conclusion (if the AI said "not hate", you
think it IS hate; if the AI said "hate", you think it ISN'T). Sound like a
real Reddit reader, not an academic. Casual, contractions OK, no markdown.

Post: {post}
AI's explanation: {explanation}

Your reaction (1-2 sentences, disagreeing in your own words):"""

VERIFY_PROMPT = """Two reactions to the same social-media post. Do they reach the SAME conclusion
about whether the post is hate speech, or DIFFERENT conclusions?

Reaction A: {a}
Reaction B: {b}

Answer with exactly one word: SAME or DIFFERENT."""


def call_openai(model: str, prompt: str, max_tokens: int = 200) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate(n_per_label: int, model: str = "gpt-4o-mini") -> pd.DataFrame:
    """Sample n_per_label OOD-test rows, generate one human-style comment
    per row, alternating equivalent/unequivalent labels.
    """
    df = pd.read_csv(REPO / "reformulated_full.csv")
    df = df.dropna(subset=["explanation", "post", "domain"]).copy()
    df = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)

    rng = random.Random(42)
    sampled_indices = rng.sample(range(len(df)), k=min(2 * n_per_label, len(df)))

    rows = []
    for i, idx in enumerate(sampled_indices):
        target = "equivalent" if i % 2 == 0 else "unequivalent"
        post = str(df.loc[idx, "post"])[:1000]
        explanation = str(df.loc[idx, "explanation"])[:1500]
        prompt = AGREE_PROMPT if target == "equivalent" else DISAGREE_PROMPT
        try:
            comment = call_openai(model, prompt.format(post=post, explanation=explanation))
        except Exception as e:
            print(f"  [{i}] generation error: {e}", flush=True)
            continue
        # verify with second call
        try:
            verify = call_openai(model, VERIFY_PROMPT.format(a=explanation, b=comment), max_tokens=4)
            verdict = verify.upper().strip().split()[0] if verify else ""
        except Exception as e:
            print(f"  [{i}] verify error: {e}", flush=True)
            continue
        verified_label = "equivalent" if "SAME" in verdict else "unequivalent" if "DIFFERENT" in verdict else None
        if verified_label is None:
            print(f"  [{i}] verify garbage: {verify}", flush=True)
            continue
        if verified_label != target:
            # GPT generated a comment that doesn't match its intended target;
            # keep it anyway with the verifier's label (these are the *hard*
            # cases and human-realistic).
            pass
        rows.append({
            "domain": df.loc[idx, "domain"],
            "post": post,
            "explanation": explanation,
            "comment": comment,
            "label_target": target,
            "label_verified": verified_label,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{2*n_per_label}] done; kept {len(rows)} so far", flush=True)
    return pd.DataFrame(rows)


def evaluate_checkpoint(ckpt: Path, df: pd.DataFrame) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.train(False)
    pairs = [(r["explanation"], r["comment"], LABEL2ID[r["label_verified"]])
             for _, r in df.iterrows() if r["label_verified"] in LABEL2ID]
    labels, preds, probs = _batched_forward(pairs, tok, model, device)
    return _metrics_from_arrays(labels, preds, probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_label", type=int, default=25,
                        help="Number of pairs per intended label (total 2*N).")
    parser.add_argument("--gen_model", default="gpt-4o-mini")
    parser.add_argument("--skip_generation", action="store_true",
                        help="reuse cached harder_eval.csv if it exists")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_generation and OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print(f"[stage7] reusing cached {OUT_CSV}  n={len(df)}", flush=True)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[stage7] OPENAI_API_KEY not set; abort.")
            return 1
        print(f"[stage7] generating {2*args.n_per_label} pairs with {args.gen_model}", flush=True)
        df = generate(args.n_per_label, args.gen_model)
        df.to_csv(OUT_CSV, index=False)
        print(f"[stage7] saved {OUT_CSV}  n_kept={len(df)}", flush=True)

    print(f"\n[stage7] label balance: {dict(df['label_verified'].value_counts())}")

    # Evaluate two checkpoints: poster baseline and v04 winner
    checkpoints = {
        "poster_baseline": REPO / "judge_finetune" / "runs" / "roberta_base" / "checkpoints" / "checkpoint-1576",
        "v04_style_mix_qwen7b": REPO / "overnight" / "runs" / "v04_style_mix_qwen7b" / "checkpoints",
    }
    # find checkpoint-X subdirs for the trainer-style ones
    out = {}
    for name, ck in checkpoints.items():
        if ck.exists() and (ck / "config.json").exists():
            target = ck
        else:
            cands = sorted(ck.glob("checkpoint-*")) if ck.exists() else []
            if not cands:
                print(f"[stage7] {name}: checkpoint not found at {ck}; skipping")
                continue
            target = cands[-1]
        print(f"\n[stage7] scoring {name}  ({target}) ...", flush=True)
        m = evaluate_checkpoint(target, df)
        out[name] = m
        print(f"  n={m['n']}  acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  "
              f"eq={m.get('acc_equivalent',0):.2f}/uneq={m.get('acc_unequivalent',0):.2f}", flush=True)

    save_json(RESULTS_DIR / "scores.json", out)

    # Append to OVERNIGHTLOG.md
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    block = [f"\n## {ts}  STAGE 7: harder eval ({len(df)} GPT-generated human-style pairs)\n",
             "Generated by GPT-4o-mini on fresh OOD-test posts, in two voices (agreeing/disagreeing).",
             "Verified by a second GPT-4o-mini call. Both checkpoints scored on same set.\n\n",
             "| model | acc | f1_macro | eq / uneq |\n|---|---|---|---|"]
    for name, m in out.items():
        block.append(f"| {name} | {m['accuracy']*100:.1f}% | {m['f1_macro']:.3f} | "
                     f"{m.get('acc_equivalent',0)*100:.0f}%/{m.get('acc_unequivalent',0)*100:.0f}% |")
    block.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")
    print("[stage7] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
