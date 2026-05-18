"""Stage 8: estimate label noise in the actual training data.

The poster model fine-tuned on reformulated_full.csv. The synthetic
"unequivalent" pairs were supposed to flip the original rationale's
conclusion, but a previous spot-check on the regenerated final_batch.csv
showed 50% of "unequivalent" pairs were actually saying the SAME thing.
We never measured the noise rate on the actual training data.

This stage:
  1. Samples N pairs from reformulated_full.csv, stratified by (label, domain).
  2. Asks GPT-4o-mini, for each pair: do the two rationales reach the SAME
     conclusion about whether the post is hate speech, or DIFFERENT? (No
     dataset context, no labels — blind judgment.)
  3. Cross-tabulates judgment vs stored label.
  4. Also audits the 4 improved-Qwen-7B reformulation CSVs that v04 uses
     as augmentation — same procedure.

If the noise rate is high, that's the ceiling on any model trained on
this data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import LABEL2ID, REPO, save_json  # noqa: E402

RESULTS_DIR = REPO / "overnight" / "results" / "stage8"

JUDGE_PROMPT = """You will see two rationales about whether a social media post is hate speech.
They might be paraphrases of each other, or they might reach opposite conclusions.

Answer with exactly one word: SAME or DIFFERENT.

SAME = both rationales reach the same overall verdict (both say it IS hate /
       both say it ISN'T hate / both say it's offensive but not hate, etc).
DIFFERENT = the rationales reach opposite verdicts.

Rationale A: {a}

Rationale B: {b}

Answer (SAME or DIFFERENT):"""


def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().upper()


def audit_pairs(pairs: list[tuple[str, str, str]], model: str = "gpt-4o-mini",
                stop_on_errors: int = 5) -> list[dict]:
    """pairs is [(text_a, text_b, stored_label), ...]. Returns list of dicts."""
    rows: list[dict] = []
    err = 0
    for i, (a, b, lbl) in enumerate(pairs):
        try:
            v = call_openai(model, JUDGE_PROMPT.format(a=a[:1500], b=b[:1500]))
        except Exception as e:
            err += 1
            print(f"  [{i}] error: {e}", flush=True)
            if err > stop_on_errors:
                print(f"  too many errors ({err}); aborting")
                break
            continue
        first = v.split()[0] if v else ""
        verdict = ("same" if "SAME" in first else
                   "different" if "DIFFERENT" in first else "garbage")
        # Map stored label to expected verdict: equivalent -> SAME, unequivalent -> DIFFERENT
        expected = "same" if lbl == "equivalent" else "different"
        rows.append({
            "stored_label": lbl,
            "expected_verdict": expected,
            "judge_verdict": verdict,
            "agrees": verdict == expected,
            "raw": v,
        })
        if (i + 1) % 20 == 0:
            agree = sum(1 for r in rows if r["agrees"])
            print(f"  [{i+1}/{len(pairs)}] kept {len(rows)}, judge agrees {agree}/{len(rows)} = "
                  f"{agree/max(len(rows),1)*100:.0f}%", flush=True)
    return rows


def summarize(rows: list[dict], name: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"name": name, "n": 0}
    agree_total = sum(1 for r in rows if r["agrees"])
    by_label: dict[str, dict] = {}
    for lbl in ("equivalent", "unequivalent"):
        sub = [r for r in rows if r["stored_label"] == lbl]
        if not sub:
            continue
        agree = sum(1 for r in sub if r["agrees"])
        garbage = sum(1 for r in sub if r["judge_verdict"] == "garbage")
        by_label[lbl] = {
            "n": len(sub),
            "agree_pct": agree / len(sub) * 100,
            "disagree_pct": (len(sub) - agree - garbage) / len(sub) * 100,
            "garbage_pct": garbage / len(sub) * 100,
        }
    return {
        "name": name,
        "n": n,
        "overall_agreement_pct": agree_total / n * 100,
        "by_stored_label": by_label,
    }


def sample_from_csv(csv_path: Path, n_per_strata: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["explanation", "synthetic", "meaning"]).copy()
    df["meaning"] = df["meaning"].astype(str).str.lower().str.strip()
    df = df[df["meaning"].isin(LABEL2ID)].copy()
    if "domain" not in df.columns:
        df["domain"] = "augmented"
    rng = random.Random(seed)
    sampled = []
    for (label, domain), group in df.groupby(["meaning", "domain"]):
        k = min(n_per_strata, len(group))
        idxs = rng.sample(range(len(group)), k)
        sampled.append(group.iloc[idxs])
    return pd.concat(sampled, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_main", type=int, default=12,
                        help="Per stratum (label × domain) for main CSV. "
                             "Default 12 with 2 labels x 8 domains = ~192 pairs.")
    parser.add_argument("--n_aug", type=int, default=15,
                        help="Per stratum for the Qwen-7B augmentation files (much smaller).")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[stage8] OPENAI_API_KEY not set; abort.")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Main training CSV ---
    main_csv = REPO / "reformulated_full.csv"
    print(f"\n=== auditing {main_csv.name} ===", flush=True)
    sampled = sample_from_csv(main_csv, args.n_main)
    print(f"[main] sampled {len(sampled)} pairs across {len(set(sampled['domain']))} domains, "
          f"label balance {dict(sampled['meaning'].value_counts())}", flush=True)
    main_pairs = [(r["explanation"], r["synthetic"], r["meaning"])
                  for _, r in sampled.iterrows()]
    main_rows = audit_pairs(main_pairs, args.model)
    main_summary = summarize(main_rows, "reformulated_full.csv")
    main_summary["sampled_per_stratum"] = args.n_main
    save_json(RESULTS_DIR / "main_audit.json", {"summary": main_summary, "rows": main_rows})
    print(f"\n[main] overall judge-agreement: {main_summary['overall_agreement_pct']:.1f}%")
    print(f"[main] by stored label:")
    for lbl, st in main_summary["by_stored_label"].items():
        print(f"   {lbl:<14} n={st['n']:>3}  agree={st['agree_pct']:.1f}%  "
              f"disagree={st['disagree_pct']:.1f}%  garbage={st['garbage_pct']:.1f}%")

    # --- 2. Qwen-7B augmentation CSVs ---
    aug_summaries = []
    for p in sorted((REPO / "synthetic_data_generation_improved").glob("reformulated*.csv")):
        print(f"\n=== auditing {p.name} ===", flush=True)
        try:
            sampled_aug = sample_from_csv(p, args.n_aug)
        except Exception as e:
            print(f"[aug] failed to sample {p}: {e}")
            continue
        print(f"[aug] sampled {len(sampled_aug)} pairs", flush=True)
        aug_pairs = [(r["explanation"], r["synthetic"], r["meaning"])
                     for _, r in sampled_aug.iterrows()]
        aug_rows = audit_pairs(aug_pairs, args.model)
        s = summarize(aug_rows, p.name)
        s["sampled_per_stratum"] = args.n_aug
        aug_summaries.append(s)
        print(f"[aug] {p.name}: overall {s['overall_agreement_pct']:.1f}%")

    save_json(RESULTS_DIR / "aug_audit.json", {"summaries": aug_summaries})

    # --- 3. Append OVERNIGHTLOG.md
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## {ts}  STAGE 8: training-data label noise audit\n",
             f"GPT-4o-mini blind-judges each pair: SAME conclusion or DIFFERENT?\n",
             "**`reformulated_full.csv` (the actual training data, n={n}):**".format(n=main_summary["n"]),
             f"- overall judge-agreement: **{main_summary['overall_agreement_pct']:.1f}%**",
             "",
             "| stored label | n | judge agrees | judge disagrees | garbage |",
             "|---|---|---|---|---|"]
    for lbl, st in main_summary["by_stored_label"].items():
        lines.append(f"| {lbl} | {st['n']} | {st['agree_pct']:.1f}% | "
                     f"{st['disagree_pct']:.1f}% | {st['garbage_pct']:.1f}% |")
    if aug_summaries:
        lines.append("")
        lines.append("**Qwen-7B augmentation files (the data v04 added):**")
        lines.append("")
        lines.append("| file | n | overall agree |")
        lines.append("|---|---|---|")
        for s in aug_summaries:
            lines.append(f"| {s['name']} | {s['n']} | {s['overall_agreement_pct']:.1f}% |")
    lines.append("")
    lines.append("Interpretation: any noise rate >5% means a meaningful chunk of the "
                 "training labels are wrong. If 30% of 'unequivalent' pairs are actually "
                 "saying the same thing as the original, the model is being told 'these are "
                 "different' on examples that aren't, and there's no clean signal it can latch "
                 "onto except *style*.")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[stage8] done. results -> {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
