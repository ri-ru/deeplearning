"""Stage 14b: split the manual+claude n=40 set into its two halves and rerun.

manual labelling.csv          = 20 pairs hand-written by Veronika.
claude_test_labels.csv        = 20 pairs written by Claude 4.6.

Stage 14 evaluated on these COMBINED. This script evaluates on each
separately, so we can see how each model does on actually-handwritten
text vs Claude-4.6-written text.

Same 12 checkpoints as Stage 14.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, _batched_forward, _metrics_from_arrays, load_human_csv, save_json,
)
from overnight.stage14_human_eval_matrix import resolve_ckpt, MODELS  # noqa: E402

RESULTS_DIR = REPO / "overnight" / "results" / "stage14b"


def load_split_sets() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # Veronika handwritten (the cleanest eval we have)
    rows = []
    for r in load_human_csv(REPO / "manual labelling.csv"):
        rows.append({"text_a": r["explanation"], "text_b": r["comment"],
                     "label": LABEL2ID[r["label"]]})
    out["manual_handwritten_n20"] = pd.DataFrame(rows)

    # Claude 4.6 written (different model from anything in training)
    rows = []
    for r in load_human_csv(REPO / "roberta_training" / "claude_test_labels.csv"):
        rows.append({"text_a": r["explanation"], "text_b": r["comment"],
                     "label": LABEL2ID[r["label"]]})
    out["claude46_n20"] = pd.DataFrame(rows)

    return out


def score_one(ckpt: Path, df: pd.DataFrame) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.train(False)
    pairs = list(zip(
        df["text_a"].astype(str).tolist(),
        df["text_b"].astype(str).tolist(),
        df["label"].astype(int).tolist(),
    ))
    labels, preds, probs = _batched_forward(pairs, tok, model, device)
    return _metrics_from_arrays(labels, preds, probs)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_sets = load_split_sets()
    print(f"[stage14b] eval sets: {[(n, len(df)) for n, df in eval_sets.items()]}", flush=True)
    print(f"[stage14b] label balance:", flush=True)
    for n, df in eval_sets.items():
        bal = dict(df["label"].value_counts())
        print(f"  {n}: {bal}", flush=True)

    matrix: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        ck = resolve_ckpt(m)
        if ck is None:
            print(f"[stage14b] {m}: no checkpoint; skipping", flush=True)
            continue
        matrix[m] = {}
        for eval_name, df in eval_sets.items():
            try:
                r = score_one(ck, df)
                matrix[m][eval_name] = {
                    "n": r["n"], "acc": r["accuracy"], "f1": r["f1_macro"],
                    "auroc": r.get("auroc"),
                    "acc_eq": r.get("acc_equivalent"), "acc_uneq": r.get("acc_unequivalent"),
                }
                a = r["accuracy"]; eq = r.get("acc_equivalent",0) or 0; uneq = r.get("acc_unequivalent",0) or 0
                print(f"  {m:>22}  on {eval_name:>24}  acc={a:.3f}  eq={eq*100:.0f}%/uneq={uneq*100:.0f}%",
                      flush=True)
            except Exception as e:
                matrix[m][eval_name] = {"error": str(e)}
                print(f"  {m:>22}  on {eval_name:>24}  FAILED: {e}", flush=True)
        save_json(RESULTS_DIR / "matrix.json", matrix)

    # OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    eval_names = list(eval_sets.keys())
    lines = [
        f"\n## {ts}  STAGE 14b: handwritten vs Claude-4.6 split eval\n",
        f"Split the manual+claude n=40 set into its two halves. "
        f"manual_handwritten_n20 = Veronika's handwriting (no LLM at all). "
        f"claude46_n20 = Claude 4.6 (NOT in any of our training sets, but same family as Haiku-4.5).\n",
        f"Format: acc (eq% / uneq%)\n",
        "| model | manual handwritten (n=20) | Claude 4.6 (n=20) |",
        "|---|---|---|",
    ]
    for m, row in matrix.items():
        cells = [m]
        for e in eval_names:
            v = row.get(e, {})
            if "acc" in v:
                eq = (v.get("acc_eq") or 0) * 100
                un = (v.get("acc_uneq") or 0) * 100
                cells.append(f"{v['acc']*100:.0f}% ({eq:.0f}/{un:.0f})")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[stage14b] done. matrix -> {RESULTS_DIR / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
