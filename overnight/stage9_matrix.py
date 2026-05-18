"""Stage 9b: build the cross-generator transfer matrix.

After stage9_generate.py has written overnight/cross_gen/<slug>.csv:
  1. For each generator slug, train one roberta-base on JUST that 200 pairs
     (3 epochs since tiny). Save the checkpoint.
  2. For each (train_slug, test_slug) pair, score the trained classifier
     on the held-out test_slug pairs. Also evaluate on the human eval set
     (manual + claude = 40 pairs).
  3. Also train one "multi-gen" classifier on the union of all generators'
     pairs (~1200) and add it as an extra row.

Output: overnight/results/stage9/matrix.json (rows = train_slug, cols = test_slug),
plus a markdown table appended to OVERNIGHTLOG.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, _batched_forward, _metrics_from_arrays, find_checkpoint,
    load_human_csv, save_json,
)

CROSS_DIR = REPO / "overnight" / "cross_gen"
RUNS_DIR = REPO / "overnight" / "runs" / "stage9"
RESULTS_DIR = REPO / "overnight" / "results" / "stage9"


def load_generator_csv(slug: str) -> pd.DataFrame:
    p = CROSS_DIR / f"{slug}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df = df.dropna(subset=["explanation", "synthetic", "meaning"]).copy()
    df = df[df["meaning"].isin(LABEL2ID)].copy()
    df["label"] = df["meaning"].map(LABEL2ID).astype(int)
    df = df.rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    return df.reset_index(drop=True)


def make_train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Split by post_id so the same source post never appears in both splits."""
    unique_posts = df["post_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_posts)
    n_test = max(1, int(len(unique_posts) * test_frac))
    test_posts = set(unique_posts[:n_test])
    train_posts = set(unique_posts[n_test:])
    train_df = df[df["post_id"].astype(str).isin(train_posts)].reset_index(drop=True)
    test_df = df[df["post_id"].astype(str).isin(test_posts)].reset_index(drop=True)
    return train_df, test_df


def tokenize(ds, tok, max_length: int = 256):
    def _t(b):
        return tok(b["text_a"], b["text_b"], truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label",)]
    return ds.map(_t, batched=True, remove_columns=cols)


def train_classifier(train_df: pd.DataFrame, slug: str, epochs: int = 3, seed: int = 42) -> Path:
    set_seed(seed)
    run_dir = RUNS_DIR / slug
    if (run_dir / "checkpoints").exists():
        try:
            return find_checkpoint(run_dir)
        except FileNotFoundError:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize(Dataset.from_pandas(train_df[["text_a", "text_b", "label"]]), tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"}, label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        save_strategy="epoch", save_total_limit=1,
        eval_strategy="no",
        bf16=torch.cuda.is_available(), fp16=False,
        logging_steps=20, report_to="none",
        dataloader_num_workers=0, seed=seed,
    )
    Trainer(model=model, args=targs,
            train_dataset=train_ds,
            processing_class=tok, data_collator=collator).train()
    return find_checkpoint(run_dir)


def score_on(ckpt: Path, test_df: pd.DataFrame) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.train(False)
    pairs = list(zip(test_df["text_a"].astype(str).tolist(),
                     test_df["text_b"].astype(str).tolist(),
                     test_df["label"].astype(int).tolist()))
    labels, preds, probs = _batched_forward(pairs, tok, model, device)
    return _metrics_from_arrays(labels, preds, probs)


def load_human() -> pd.DataFrame:
    """Load the human eval set (manual + Claude) as a labeled DataFrame."""
    rows = []
    for src in [REPO / "manual labelling.csv",
                REPO / "roberta_training" / "claude_test_labels.csv"]:
        for r in load_human_csv(src):
            rows.append({"text_a": r["explanation"], "text_b": r["comment"],
                         "label": LABEL2ID[r["label"]], "post_id": "human"})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slugs", default=None,
                        help="Comma-sep generator slugs to include. Default: all CSVs in cross_gen/.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--skip_multi", action="store_true",
                        help="Don't train the multi-generator union model.")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.slugs:
        slugs = [s.strip() for s in args.slugs.split(",")]
    else:
        slugs = sorted(p.stem for p in CROSS_DIR.glob("*.csv") if not p.stem.startswith("_"))
    print(f"[stage9] generators in matrix: {slugs}", flush=True)

    # Per-generator splits
    train_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for s in slugs:
        df = load_generator_csv(s)
        tr, te = make_train_test_split(df)
        train_dfs[s] = tr
        test_dfs[s] = te
        print(f"  {s}: train={len(tr)} test={len(te)}", flush=True)

    # Human held-out
    human_df = load_human()
    print(f"  human (eval-only): n={len(human_df)}", flush=True)

    # Train per-generator
    ckpts: dict[str, Path] = {}
    for s in slugs:
        print(f"\n[train] {s}: {len(train_dfs[s])} pairs x {args.epochs} epochs", flush=True)
        t0 = time.time()
        try:
            ckpts[s] = train_classifier(train_dfs[s], s, epochs=args.epochs)
            print(f"[train] {s}: done in {time.time()-t0:.0f}s -> {ckpts[s]}", flush=True)
        except Exception as e:
            print(f"[train] {s}: FAILED {e}", flush=True)

    # Optional: union of all generators
    if not args.skip_multi and len(slugs) > 1:
        union = pd.concat([train_dfs[s] for s in slugs], ignore_index=True)
        print(f"\n[train] multi_gen: {len(union)} pairs (union)", flush=True)
        try:
            ckpts["multi_gen"] = train_classifier(union, "multi_gen", epochs=args.epochs)
        except Exception as e:
            print(f"[train] multi_gen: FAILED {e}", flush=True)

    # Cross-eval
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for tr_slug, ckpt in ckpts.items():
        matrix[tr_slug] = {}
        # Eval on every test set
        for te_slug, te_df in test_dfs.items():
            try:
                m = score_on(ckpt, te_df)
                matrix[tr_slug][te_slug] = {
                    "n": m["n"], "acc": m["accuracy"], "f1": m["f1_macro"],
                    "acc_eq": m.get("acc_equivalent"), "acc_uneq": m.get("acc_unequivalent"),
                }
                print(f"  train={tr_slug:>18}  test={te_slug:>18}  "
                      f"acc={m['accuracy']:.3f}  f1={m['f1_macro']:.3f}", flush=True)
            except Exception as e:
                matrix[tr_slug][te_slug] = {"error": str(e)}
                print(f"  train={tr_slug:>18}  test={te_slug:>18}  FAILED {e}", flush=True)
        # Eval on human held-out
        try:
            m = score_on(ckpt, human_df)
            matrix[tr_slug]["human"] = {
                "n": m["n"], "acc": m["accuracy"], "f1": m["f1_macro"],
                "acc_eq": m.get("acc_equivalent"), "acc_uneq": m.get("acc_unequivalent"),
            }
            print(f"  train={tr_slug:>18}  test={'human':>18}  acc={m['accuracy']:.3f}", flush=True)
        except Exception as e:
            matrix[tr_slug]["human"] = {"error": str(e)}

    save_json(RESULTS_DIR / "matrix.json", matrix)

    # Markdown table for OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    cols = slugs + (["human"] if any("human" in v for v in matrix.values()) else [])
    lines = [f"\n## {ts}  STAGE 9: cross-generator transfer matrix\n",
             "Trained one roberta-base on each generator's 200 paraphrase pairs "
             "(80% train / 20% test split by source post). Then cross-evaluated. "
             "Diagonal = in-distribution (same generator for train + test).\n"]
    # Accuracy table
    header = "| train \\ test | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines.append(header); lines.append(sep)
    for tr in matrix:
        row = [tr]
        for te in cols:
            cell = matrix[tr].get(te, {})
            row.append(f"{cell.get('acc'):.2f}" if "acc" in cell else "-")
        lines.append("| " + " | ".join(row) + " |")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[stage9] matrix.json + OVERNIGHTLOG updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
