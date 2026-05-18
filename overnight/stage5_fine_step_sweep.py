"""Fine-grained training-dynamics sweep around the step-400 sweet spot.

Stage 3 saved every 200 steps and found that step 400 was the only snapshot
where manual_acc was 95%. We don't know if that's a 1-step fluke (early in
training, the model is still randomly drifting) or a real ~100-step window.

This stage retrains baseline from scratch and saves a checkpoint every
25 steps from step 100 to step 800, then scores each. Total 28 snapshots.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, TEST_DOMAINS, TRAIN_DOMAINS,
    load_full_csv, save_json, score_all,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage5"
RUNS_DIR = REPO / "overnight" / "runs" / "stage5_fine_sweep"


def tokenize(ds, tok, max_length=256):
    def _t(b):
        return tok(b["text_a"], b["text_b"], truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label", "domain")]
    return ds.map(_t, batched=True, remove_columns=cols)


def make_split(df, seed=42, val_frac=0.1):
    import pandas as pd
    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    val_chunks = []
    for _, grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=val_frac, random_state=seed))
    val_df = pd.concat(val_chunks).reset_index(drop=True)
    val_idx = set(val_df.index)
    train_df = train_pool[~train_pool.index.isin(val_idx)].reset_index(drop=True)
    return train_df, val_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_full_csv().rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    train_df, val_df = make_split(df, seed=args.seed)
    print(f"[stage5] train={len(train_df)}  val={len(val_df)}", flush=True)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize(Dataset.from_pandas(train_df[["text_a", "text_b", "label", "domain"]]), tok)
    val_ds = tokenize(Dataset.from_pandas(val_df[["text_a", "text_b", "label", "domain"]]), tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"}, label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=str(RUNS_DIR / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=64,
        eval_strategy="no",
        bf16=torch.cuda.is_available(), fp16=False,
        logging_steps=50, report_to="none",
        dataloader_num_workers=0, seed=args.seed,
    )
    Trainer(model=model, args=targs,
            train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tok, data_collator=collator).train()

    ckpt_root = RUNS_DIR / "checkpoints"
    snapshots = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    print(f"[stage5] scoring {len(snapshots)} fine-grained snapshots ...", flush=True)
    timeline = []
    for ck in snapshots:
        step = int(ck.name.split("-")[-1])
        t0 = time.time()
        try:
            m = score_all(ck)
        except Exception as e:
            print(f"[stage5] {ck.name} failed: {e}", flush=True)
            continue
        row = {
            "step": step,
            "ood_f1": m["ood_overall"]["f1_macro"],
            "ood_acc": m["ood_overall"]["accuracy"],
            "manual_acc": m["manual"]["accuracy"],
            "manual_acc_eq": m["manual"].get("acc_equivalent"),
            "manual_acc_uneq": m["manual"].get("acc_unequivalent"),
            "claude_acc": m["claude"]["accuracy"],
            "claude_acc_eq": m["claude"].get("acc_equivalent"),
            "claude_acc_uneq": m["claude"].get("acc_unequivalent"),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        timeline.append(row)
        print(f"  step={step:>4}  ood_f1={row['ood_f1']:.3f}  "
              f"manual={row['manual_acc']:.2f} (eq={row['manual_acc_eq']:.2f}/un={row['manual_acc_uneq']:.2f})  "
              f"claude={row['claude_acc']:.2f}", flush=True)
        save_json(RESULTS_DIR / "fine_timeline.json", {"timeline": timeline})

    # Find sweet-spot rows
    if timeline:
        ranked = sorted(timeline, key=lambda r: (r["manual_acc"] + r["claude_acc"]), reverse=True)
        print("\n[stage5] top-5 snapshots by manual+claude sum:")
        for r in ranked[:5]:
            print(f"  step={r['step']:>4}  manual={r['manual_acc']:.2f}  claude={r['claude_acc']:.2f}  "
                  f"ood_f1={r['ood_f1']:.3f}")

    print("[stage5] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
