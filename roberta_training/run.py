"""
Fine-tune RoBERTa on reformulated_full.csv + few-shot synthetic data.

Usage:
    python run.py --synthetic synthetic_fewshot.csv

Required files in the same folder:
    - reformulated_full.csv       (original 52k training data)
    - synthetic_fewshot.csv       (output of reformulate.py)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    set_seed,
)

# ── constants ──────────────────────────────────────────────────────────────────

TRAIN_DOMAINS = ("dynahate", "cad", "measure", "socialbias", "toraman")
TEST_DOMAINS  = ("toxigen", "gab", "anatomy")
LABEL2ID = {"equivalent": 0, "unequivalent": 1}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── data preparation ───────────────────────────────────────────────────────────

def prepare_synthetic(synthetic_csv: str) -> str:
    """Convert reformulate.py output to the format expected by the trainer."""
    df = pd.read_csv(synthetic_csv)
    out = pd.DataFrame({
        "original":      df["explanation"],
        "synthetic":     df["synthetic"],
        "label":         df["meaning"],
        "quality_flags": "ok",
        "domain":        df.get("domain", "augmented"),
    })
    out = out.dropna(subset=["original", "synthetic", "label"])
    out = out[out["label"].isin(LABEL2ID)]
    prepared_path = synthetic_csv.replace(".csv", "_prepared.csv")
    out.to_csv(prepared_path, index=False)
    print(f"[prep] {len(out)} synthetic rows saved to {prepared_path}")
    return prepared_path


# ── dataset loading ────────────────────────────────────────────────────────────

def load_and_split(csv_path: str, augment_csv: str | None,
                   val_frac: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)

    label_col = next(
        (c for c in ["equivalence", "judgement", "is_equivalent"] if c in df.columns),
        None
    )
    if label_col is None:
        last = df.columns[-1]
        if set(df[last].dropna().astype(str).str.lower().unique()[:5]).issubset({"equivalent", "unequivalent"}):
            label_col = last
    if label_col is None:
        raise ValueError("Could not find equivalence label column.")

    text_b_col = "synthetic" if "synthetic" in df.columns else "reformulated"
    df = df.dropna(subset=["explanation", text_b_col, label_col, "domain"])
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(LABEL2ID)]
    df = df.rename(columns={"explanation": "text_a", text_b_col: "text_b", label_col: "label_str"})
    df["label"] = df["label_str"].map(LABEL2ID).astype(int)
    df["post_id"] = df.get("uid", df.get("id", range(len(df)))).astype(str)

    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    test_df    = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)

    val_chunks = []
    for _, grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=val_frac, random_state=seed))
    val_df   = pd.concat(val_chunks).reset_index(drop=True)
    val_ids  = set(val_df["post_id"])
    train_df = train_pool[~train_pool["post_id"].isin(val_ids)].reset_index(drop=True)

    if augment_csv is not None:
        aug = pd.read_csv(augment_csv)
        aug = aug[aug["quality_flags"] == "ok"].copy()
        aug["label"] = aug["label"].map(LABEL2ID).astype(int)
        aug = aug.rename(columns={"original": "text_a", "synthetic": "text_b"})
        aug["post_id"] = "aug_" + aug.index.astype(str)
        aug["domain"]  = aug.get("domain", "augmented")
        aug = aug[["text_a", "text_b", "label", "domain", "post_id"]]
        train_df = pd.concat([train_df, aug], ignore_index=True)
        print(f"[data] injected {len(aug)} synthetic rows into train")

    keep = ["text_a", "text_b", "label", "domain", "post_id"]
    return (
        Dataset.from_pandas(train_df[keep]),
        Dataset.from_pandas(val_df[keep]),
        Dataset.from_pandas(test_df[keep]),
    )


def tokenize_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _tok(batch):
        return tokenizer(batch["text_a"], batch["text_b"],
                         truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label", "domain", "post_id")]
    return ds.map(_tok, batched=True, remove_columns=cols)


# ── metrics ────────────────────────────────────────────────────────────────────

def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics_overall(eval_pred):
    logits, labels = eval_pred
    probs = softmax_np(logits)
    preds = probs.argmax(axis=-1)
    return {
        "accuracy":         accuracy_score(labels, preds),
        "f1_macro":         f1_score(labels, preds, average="macro"),
        "f1_unequivalent":  f1_score(labels, preds, pos_label=1),
        "auroc":            roc_auc_score(labels, probs[:, 1]),
    }


def per_domain_breakdown(logits, labels, domains):
    probs = softmax_np(logits)
    preds = probs.argmax(axis=-1)
    out = {"overall": {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_unequivalent": float(f1_score(labels, preds, pos_label=1)),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }}
    for dom in sorted(set(domains)):
        mask = np.array([d == dom for d in domains])
        if not mask.sum():
            continue
        y, p, pr = labels[mask], preds[mask], probs[mask, 1]
        try:
            auroc = float(roc_auc_score(y, pr))
        except ValueError:
            auroc = float("nan")
        out[dom] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "f1_unequivalent": float(f1_score(y, p, pos_label=1, zero_division=0)),
            "auroc": auroc,
        }
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic",  required=True,
                   help="CSV output of reformulate.py (e.g. synthetic_fewshot.csv)")
    p.add_argument("--base_csv",   default="reformulated_full.csv",
                   help="Original 52k training data")
    p.add_argument("--output_dir", default="runs/roberta_fewshot")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--max_length", type=int,   default=256)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — prepare synthetic CSV
    augment_csv = prepare_synthetic(args.synthetic)

    # Step 2 — load data
    train_ds, val_ds, test_ds = load_and_split(args.base_csv, augment_csv, seed=args.seed)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # Step 3 — tokenize
    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_tok = tokenize_dataset(train_ds, tok, args.max_length)
    val_tok   = tokenize_dataset(val_ds,   tok, args.max_length)
    test_tok  = tokenize_dataset(test_ds,  tok, args.max_length)

    test_domains  = list(test_ds["domain"])
    test_post_ids = list(test_ds["post_id"])
    test_text_a   = list(test_ds["text_a"])
    test_text_b   = list(test_ds["text_b"])

    # Step 4 — model
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=False,
        fp16=False,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tok,
        data_collator=collator,
        compute_metrics=compute_metrics_overall,
    )

    # Step 5 — train
    trainer.train()

    # Step 6 — evaluate on OOD test set
    pred_out = trainer.predict(test_tok)
    breakdown = per_domain_breakdown(pred_out.predictions, pred_out.label_ids, test_domains)

    print("\n=== OOD test results ===")
    print(json.dumps(breakdown, indent=2))

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(breakdown, f, indent=2)

    probs = softmax_np(pred_out.predictions)
    preds = probs.argmax(axis=1)
    pd.DataFrame({
        "post_id": test_post_ids,
        "domain":  test_domains,
        "text_a":  test_text_a,
        "text_b":  test_text_b,
        "label":   pred_out.label_ids,
        "pred":    preds,
        "p_unequivalent": probs[:, 1],
    }).to_csv(run_dir / "test_predictions.csv", index=False)

    print(f"\n[done] results in {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
