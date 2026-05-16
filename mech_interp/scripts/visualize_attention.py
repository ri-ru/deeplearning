"""Print which tokens the attention probe pooled over.

Per sample, sorts tokens by alpha weight and prints the top-k along with
RoBERTa's own prediction. Groups the printout by (true_label, model_correct?)
so it's easy to skim what the model was looking at on its wins vs losses.

Also dumps an aggregate "top tokens overall" by averaging alpha over each
group.

Usage:
    python visualize_attention.py --layer 12 --split manual --top_k 8
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO / "mech_interp" / "probe_runs"


def load_alphas(runs_dir: Path, layer: int, split: str) -> list[dict]:
    path = runs_dir / f"layer_{layer:02d}" / f"{split}_alpha.json"
    return json.loads(path.read_text())


def fmt_token(tok: str) -> str:
    return tok.replace("Ġ", "_").replace("Ċ", "\\n")


def render_sample(rec: dict, top_k: int) -> str:
    tokens = rec["tokens"]
    alpha = rec["alpha"]
    # ignore <s> </s> </s> separators when picking top tokens, but keep them in mass
    skip_specials = {"<s>", "</s>", "<pad>"}
    ranked = sorted(
        [(t, a, i) for i, (t, a) in enumerate(zip(tokens, alpha))
         if t not in skip_specials],
        key=lambda x: -x[1],
    )[:top_k]
    parts = []
    for t, a, _ in ranked:
        bar = "█" * int(round(a * 40))
        parts.append(f"    {a:.3f} {bar} {fmt_token(t)}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--split", choices=["manual", "claude", "val"], default="manual")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = load_alphas(args.runs_dir, args.layer, args.split)
    if args.limit:
        recs = recs[: args.limit]

    print(f"\n=== {args.split} @ layer {args.layer}  ({len(recs)} samples) ===\n")

    groups = defaultdict(list)
    for r in recs:
        true = "EQUIV" if r["label"] == 0 else "UNEQUIV"
        model_correct = (r["label"] == r["model_pred"])
        key = (true, "✓" if model_correct else "✗")
        groups[key].append(r)

    # Sort: equiv first, then unequiv; correct first within each
    for true in ("EQUIV", "UNEQUIV"):
        for mark in ("✓", "✗"):
            key = (true, mark)
            if key not in groups: continue
            grp = groups[key]
            roberta_label = ["equivalent", "unequivalent"][grp[0]["model_pred"]] if mark == "✓" \
                else "wrong"
            print(f"\n--- true={true}   roberta={mark}  ({len(grp)} samples) ---")
            for r in grp:
                m_pred = "equiv" if r["model_pred"] == 0 else "uneq"
                m_p = r["model_prob_uneq"]
                pp = r["probe_prob"]
                print(f"\n  [{r['sample_id']}]  roberta={m_pred} (p_uneq={m_p:.2f})   probe_p_uneq={pp:.2f}")
                print(render_sample(r, args.top_k))

    # Aggregate: average alpha mass per token surface form, separated by true label
    agg = {"EQUIV": defaultdict(lambda: [0.0, 0]),
           "UNEQUIV": defaultdict(lambda: [0.0, 0])}
    for r in recs:
        bucket = agg["EQUIV"] if r["label"] == 0 else agg["UNEQUIV"]
        for t, a in zip(r["tokens"], r["alpha"]):
            if t in {"<s>", "</s>", "<pad>"}: continue
            bucket[t][0] += a
            bucket[t][1] += 1
    print("\n\n=== mean alpha per token (top 15 each, min 3 occurrences) ===")
    for label in ("EQUIV", "UNEQUIV"):
        rows = [(t, v[0] / v[1], v[1]) for t, v in agg[label].items() if v[1] >= 3]
        rows.sort(key=lambda x: -x[1])
        print(f"\n-- {label} --")
        for t, mean_a, n in rows[:15]:
            bar = "█" * int(round(mean_a * 60))
            print(f"  {mean_a:.4f} ({n:3d}x) {bar} {fmt_token(t)}")


if __name__ == "__main__":
    main()
