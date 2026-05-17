"""Stage 3: training-dynamics analysis.

Train a fresh roberta-base for 3 epochs, but save a checkpoint every 1/4
epoch and score each on val + manual + Claude. The story we want to see:
   - val/OOD accuracy climbs smoothly toward 0.97
   - manual accuracy initially climbs, then plateaus or drops as the
     model overfits the Qwen style shortcut.

If the manual accuracy peaks early and then degrades, that's direct evidence
that the shortcut is what the second half of training is learning.

Resumable: one snapshot per quarter-epoch; results JSON written incrementally.
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, TEST_DOMAINS, TRAIN_DOMAINS,
    find_checkpoint, load_full_csv, save_json, score_all,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage3"
RUNS_DIR = REPO / "overnight" / "runs" / "training_dynamics"


def tokenize(ds: Dataset, tok, max_length: int) -> Dataset:
    def _t(batch):
        return tok(batch["text_a"], batch["text_b"],
                   truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label", "domain")]
    return ds.map(_t, batched=True, remove_columns=cols)


def make_split(df, seed=42, val_frac=0.1):
    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    val_chunks = []
    for _, grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=val_frac, random_state=seed))
    import pandas as pd
    val_df = pd.concat(val_chunks).reset_index(drop=True)
    val_ids = set(val_df.index.astype(str))
    train_df = train_pool[~train_pool.index.astype(str).isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


class CheckpointAtFractions(TrainerCallback):
    """Force a checkpoint at given fraction-of-epoch marks (e.g. 0.25, 0.5)."""

    def __init__(self, fractions: list[float]):
        self.fractions = sorted(fractions)
        self.fired = set()

    def on_step_end(self, args, state, control, **kwargs):
        # state.epoch is a float (0.0 -> args.num_train_epochs)
        for f in self.fractions:
            key = round(f, 3)
            if state.epoch >= f and key not in self.fired:
                control.should_save = True
                self.fired.add(key)
        return control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_full_csv().rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    train_df, val_df = make_split(df, seed=args.seed)
    print(f"[stage3] train={len(train_df)}  val={len(val_df)}", flush=True)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize(Dataset.from_pandas(train_df[["text_a", "text_b", "label", "domain"]]), tok, 256)
    val_ds = tokenize(Dataset.from_pandas(val_df[["text_a", "text_b", "label", "domain"]]), tok, 256)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"},
        label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    # We use steps-based saves: ~12 checkpoints across 3 epochs (every ~250 steps
    # for 34k samples / batch 32 / 3 epochs ≈ 3225 steps).
    targs = TrainingArguments(
        output_dir=str(RUNS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        save_strategy="steps",
        save_steps=200,            # frequent snapshots
        save_total_limit=20,
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=4,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=collator,
    )
    trainer.train()

    # Score every saved checkpoint
    ckpt_root = RUNS_DIR / "checkpoints"
    snapshots = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    print(f"[stage3] scoring {len(snapshots)} snapshots ...", flush=True)
    timeline: list[dict] = []
    for ck in snapshots:
        step = int(ck.name.split("-")[-1])
        t0 = time.time()
        try:
            metrics = score_all(ck)
        except Exception as e:
            print(f"[stage3] {ck.name} failed: {e}", flush=True)
            continue
        row = {
            "step": step,
            "manual_acc": metrics["manual"]["accuracy"],
            "manual_f1": metrics["manual"]["f1_macro"],
            "claude_acc": metrics["claude"]["accuracy"],
            "claude_f1": metrics["claude"]["f1_macro"],
            "ood_acc": metrics["ood_overall"]["accuracy"],
            "ood_f1": metrics["ood_overall"]["f1_macro"],
            "manual_acc_eq": metrics["manual"].get("acc_equivalent"),
            "manual_acc_uneq": metrics["manual"].get("acc_unequivalent"),
            "claude_acc_eq": metrics["claude"].get("acc_equivalent"),
            "claude_acc_uneq": metrics["claude"].get("acc_unequivalent"),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        timeline.append(row)
        print(f"  step={step:>5}  ood_f1={row['ood_f1']:.3f}  "
              f"manual_acc={row['manual_acc']:.3f}  claude_acc={row['claude_acc']:.3f}", flush=True)
        # incremental save
        save_json(RESULTS_DIR / "timeline.json", {"timeline": timeline})

    save_json(RESULTS_DIR / "timeline.json", {"timeline": timeline})
    print("\n[stage3] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
