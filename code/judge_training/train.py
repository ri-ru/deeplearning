"""Fine-tune a transformer encoder for rationale-equivalence classification.

"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,

    TrainingArguments,
    set_seed,
)

from .data import DataConfig, ID2LABEL, LABEL2ID, load_and_split, tokenize_dataset
from .metrics import compute_metrics_overall, per_domain_breakdown


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data
    data_cfg = DataConfig(
        csv_path=cfg["csv_path"],
        max_length=cfg["max_length"],
        val_frac=cfg.get("val_frac", 0.1),
        seed=cfg.get("seed", 42),
        max_samples=cfg.get("max_samples"),
        augment_csv=cfg.get("augment_csv"),
    )
    train_ds, val_ds, test_ds = load_and_split(data_cfg)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"[data] train domains: {sorted(set(train_ds['domain']))}")
    print(f"[data] test domains:  {sorted(set(test_ds['domain']))}")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_tok = tokenize_dataset(train_ds, tok, cfg["max_length"])
    val_tok = tokenize_dataset(val_ds, tok, cfg["max_length"])
    test_tok = tokenize_dataset(test_ds, tok, cfg["max_length"])

    # dont need for test set
    test_domains = list(test_ds["domain"])
    test_post_ids = list(test_ds["post_id"])
    test_text_a = list(test_ds["text_a"])
    test_text_b = list(test_ds["text_b"])

    # ---- Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    # ---- Training
    use_bf16 = cfg.get("bf16", True) and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg.get("eval_batch_size", cfg["batch_size"] * 2),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
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

    trainer.train()

    # ---- Final OOD evals  
    pred_out = trainer.predict(test_tok)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    breakdown = per_domain_breakdown(logits, labels, test_domains)

    print("\n=== OOD test results ===")
    print(json.dumps(breakdown, indent=2))

    # persistencee for report
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"config": cfg, "test": breakdown}, f, indent=2)

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    err_df = pd.DataFrame({
        "post_id": test_post_ids,
        "domain": test_domains,
        "text_a": test_text_a,
        "text_b": test_text_b,
        "label": labels,
        "pred": preds,
        "p_unequivalent": probs[:, 1],
    })
    err_df.to_csv(run_dir / "test_predictions.csv", index=False)
    print(f"[done] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
