"""Evaluate the trained RoBERTa judge on the hard test set.

Compares performance on:
  - easy examples (original OOD test set, anatomy domain)
  - hard examples (newly generated, anatomy domain)
  - hard ok-only (flagged lazy-negation examples removed)

Usage:
    python scripts/eval_hard_testset.py \
        --checkpoint runs/roberta_base/checkpoints \
        --hard runs/hard_testset_anatomy.csv \
        --csv ../reformulated_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DataConfig, load_and_split, LABEL2ID


def predict(model, tokenizer, text_a_list, text_b_list, batch_size=64, device="cuda"):
    model.eval()
    all_probs = []
    for i in range(0, len(text_a_list), batch_size):
        batch_a = text_a_list[i:i+batch_size]
        batch_b = text_b_list[i:i+batch_size]
        enc = tokenizer(
            batch_a, batch_b,
            truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def report(name, labels, preds):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")
    f1u = f1_score(labels, preds, pos_label=1, zero_division=0)
    print(f"  {name:<35}  n={len(labels):4d}  acc={acc:.3f}  f1_macro={f1:.3f}  f1_uneq={f1u:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/roberta_base/checkpoints")
    p.add_argument("--hard", default="runs/hard_testset_anatomy.csv")
    p.add_argument("--csv", default="../reformulated_full.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Find checkpoint dir
    ckpt_path = Path(args.checkpoint)
    subdirs = [d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if subdirs:
        ckpt_path = sorted(subdirs, key=lambda x: int(x.name.split("-")[-1]))[-1]
    print(f"[checkpoint] {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path)).to(device)

    # ---- Easy baseline: original anatomy test set
    _, _, test_ds = load_and_split(DataConfig(csv_path=args.csv))
    test_df = test_ds.to_pandas()
    anatomy_df = test_df[test_df["domain"] == "anatomy"].reset_index(drop=True)

    probs = predict(model, tokenizer, list(anatomy_df["text_a"]), list(anatomy_df["text_b"]), device=device)
    preds = probs.argmax(axis=1)

    print("\n=== Results ===")
    report("EASY  (original anatomy OOD)", anatomy_df["label"].values, preds)

    # ---- Hard test set
    hard_df = pd.read_csv(args.hard)
    hard_df["label_int"] = hard_df["label"].map({"equivalent": 0, "unequivalent": 1})
    hard_df = hard_df.dropna(subset=["label_int", "original", "synthetic"])

    probs_h = predict(model, tokenizer, list(hard_df["original"]), list(hard_df["synthetic"]), device=device)
    preds_h = probs_h.argmax(axis=1)
    hard_df["pred"] = preds_h
    hard_df["p_unequivalent"] = probs_h[:, 1]

    report("HARD  (all generated pairs)", hard_df["label_int"].values, preds_h)

    ok_mask = hard_df["quality_flags"] == "ok"
    if ok_mask.sum() > 0:
        report("HARD  (ok only, no lazy negation)", hard_df["label_int"][ok_mask].values, preds_h[ok_mask])

    uneq_mask = hard_df["label"] == "unequivalent"
    eq_mask   = hard_df["label"] == "equivalent"
    if uneq_mask.sum() > 0:
        report("HARD  (unequivalent only)", hard_df["label_int"][uneq_mask].values, preds_h[uneq_mask])
    if eq_mask.sum() > 0:
        report("HARD  (equivalent only)", hard_df["label_int"][eq_mask].values, preds_h[eq_mask])

    # ---- Show worst failures
    failures = hard_df[hard_df["label_int"] != hard_df["pred"]].copy()
    failures = failures.sort_values("p_unequivalent", ascending=False)
    print(f"\n=== {len(failures)} failures out of {len(hard_df)} hard examples ===")
    print(f"    ({len(failures[failures['label']=='unequivalent'])} missed unequivalent, "
          f"{len(failures[failures['label']=='equivalent'])} false unequivalent)\n")

    print("--- Top 3 most confident wrong predictions ---")
    for _, row in failures.head(3).iterrows():
        print(f"  TRUE:      {row['label']}")
        print(f"  PREDICTED: {'unequivalent' if row['pred']==1 else 'equivalent'}  (p_uneq={row['p_unequivalent']:.3f})")
        print(f"  ORIGINAL:  {str(row['original'])[:160]}")
        print(f"  SYNTHETIC: {str(row['synthetic'])[:160]}")
        print()

    out = Path(args.hard).parent / (Path(args.hard).stem + "_eval.csv")
    hard_df.to_csv(out, index=False)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
