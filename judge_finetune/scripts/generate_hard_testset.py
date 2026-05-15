"""Generate a hard test set with genuinely subtle semantic pairs.

Two types of hard examples:
  hard_unequivalent: similar structure/vocabulary to the original, but opposite
                     conclusion - NOT just inserting "not"
  hard_equivalent:   completely different vocabulary and sentence structure,
                     but identical meaning and verdict

Usage:
    python scripts/generate_hard_testset.py \
        --csv ../reformulated_full.csv \
        --n 200 \
        --out runs/hard_testset.csv

Cost: ~200 API calls × ~250 tokens = ~$0.008 with gpt-4o-mini
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DataConfig, load_and_split

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HARD_UNEQUIVALENT_PROMPT = """You are building a dataset to train a semantic equivalence classifier for hate speech research.

Here is an explanation of why a social media post is or isn't hate speech:
\"\"\"{original}\"\"\"

Rewrite this explanation so it reaches the OPPOSITE conclusion, but follow these strict rules:
- Use similar sentence structure and vocabulary to the original
- Do NOT just insert "not", "doesn't", or "no longer" - that is too easy to detect
- Instead, change the REASONING: swap which group is targeted, reframe the intent, change what the evidence shows
- The rewrite must be plausible and coherent on its own
- It should take genuine reading to notice the conclusion changed

Output only the rewritten explanation. No preamble, no quotes."""

HARD_EQUIVALENT_PROMPT = """You are building a dataset to train a semantic equivalence classifier for hate speech research.

Here is an explanation of why a social media post is or isn't hate speech:
\"\"\"{original}\"\"\"

Rewrite this explanation so it preserves the EXACT same conclusion and verdict, but follow these strict rules:
- Use completely different vocabulary and sentence structure
- Do NOT copy any phrase longer than 3 words from the original
- Write it as if a different person with a different style wrote it
- The final verdict (hate / not hate / offensive / neutral / etc.) must be identical

Output only the rewritten explanation. No preamble, no quotes."""


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bdon't\b", r"\bdoesn't\b",
    r"\bdoes not\b", r"\bdo not\b", r"\bwithout\b", r"\bnever\b",
    r"\bneither\b", r"\bnor\b",
]

def _negation_count(text: str) -> int:
    text = text.lower()
    return sum(len(re.findall(p, text)) for p in NEGATION_PATTERNS)

def is_lazy_negation(original: str, synthetic: str) -> bool:
    """True if the synthetic just added negation words the original didn't have."""
    orig_neg = _negation_count(original)
    synth_neg = _negation_count(synthetic)
    # If synthetic has 2+ more negations than original, likely a lazy flip
    return (synth_neg - orig_neg) >= 2

def word_overlap(a: str, b: str) -> float:
    """Fraction of words in b that also appear in a."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wb:
        return 0.0
    return len(wa & wb) / len(wb)

def is_too_similar(original: str, synthetic: str, threshold: float = 0.85) -> bool:
    """True if the equivalent rewrite copied too much from the original."""
    return word_overlap(original, synthetic) > threshold


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(client: OpenAI, model: str, prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=300,
                temperature=0.9,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {e} - waiting {wait}s")
            time.sleep(wait)
    return None


def generate_hard_pair(
    client: OpenAI,
    model: str,
    original: str,
    kind: str,  # "unequivalent" or "equivalent"
) -> dict | None:
    prompt = (
        HARD_UNEQUIVALENT_PROMPT if kind == "unequivalent"
        else HARD_EQUIVALENT_PROMPT
    ).format(original=original)

    synthetic = generate_one(client, model, prompt)
    if synthetic is None:
        return None

    # Quality gates
    flags = []
    if kind == "unequivalent" and is_lazy_negation(original, synthetic):
        flags.append("lazy_negation")
    if kind == "equivalent" and is_too_similar(original, synthetic):
        flags.append("too_similar")

    return {
        "original": original,
        "synthetic": synthetic,
        "label": kind,
        "quality_flags": ",".join(flags) if flags else "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--n", type=int, default=200,
                   help="total pairs to generate (half unequivalent, half equivalent)")
    p.add_argument("--out", default="runs/hard_testset.csv")
    p.add_argument("--domain", default=None,
                   help="restrict source examples to one domain (e.g. anatomy)")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")
    client = OpenAI(api_key=api_key)

    # Load OOD test set as source of real explanations
    _, _, test_ds = load_and_split(DataConfig(csv_path=args.csv))
    df = test_ds.to_pandas()

    if args.domain:
        df = df[df["domain"] == args.domain].reset_index(drop=True)
        print(f"[source] restricted to domain '{args.domain}': {len(df)} rows")

    n_each = args.n // 2
    # Sample from existing equivalents as seeds for hard-unequivalent generation
    # (we know these are real equivalent explanations, so flipping them is meaningful)
    seed_uneq = df[df["label"] == 0].sample(n=min(n_each, len(df[df["label"] == 0])),
                                             random_state=7).reset_index(drop=True)
    seed_eq   = df[df["label"] == 0].sample(n=min(n_each, len(df[df["label"] == 0])),
                                             random_state=13).reset_index(drop=True)

    rows = []
    total = len(seed_uneq) + len(seed_eq)
    done = 0

    print(f"Generating {len(seed_uneq)} hard-unequivalent pairs...")
    for _, row in seed_uneq.iterrows():
        result = generate_hard_pair(client, args.model, row["text_a"], "unequivalent")
        done += 1
        if result:
            result["domain"] = row["domain"]
            result["source_post_id"] = row["post_id"]
            rows.append(result)
            flag_str = f"  [{result['quality_flags']}]" if result["quality_flags"] != "ok" else ""
            print(f"[{done}/{total}] unequivalent{flag_str}")
        else:
            print(f"[{done}/{total}] unequivalent - FAILED")

    print(f"\nGenerating {len(seed_eq)} hard-equivalent pairs...")
    for _, row in seed_eq.iterrows():
        result = generate_hard_pair(client, args.model, row["text_a"], "equivalent")
        done += 1
        if result:
            result["domain"] = row["domain"]
            result["source_post_id"] = row["post_id"]
            rows.append(result)
            flag_str = f"  [{result['quality_flags']}]" if result["quality_flags"] != "ok" else ""
            print(f"[{done}/{total}] equivalent{flag_str}")
        else:
            print(f"[{done}/{total}] equivalent - FAILED")

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Summary
    n_ok      = (out_df["quality_flags"] == "ok").sum()
    n_flagged = (out_df["quality_flags"] != "ok").sum()
    print(f"\n=== Done ===")
    print(f"Total generated: {len(out_df)}")
    print(f"  ok:      {n_ok}")
    print(f"  flagged: {n_flagged}  (lazy negation or too-similar - still usable, just lower quality)")
    print(f"Saved to: {args.out}")
    print()
    print("--- 3 sample hard-unequivalent pairs ---")
    sample = out_df[out_df["label"] == "unequivalent"].head(3)
    for _, r in sample.iterrows():
        print(f"ORIGINAL:  {r['original'][:180]}")
        print(f"SYNTHETIC: {r['synthetic'][:180]}")
        print(f"FLAGS:     {r['quality_flags']}\n")


if __name__ == "__main__":
    main()
