"""Stage 1: data-mitigation variant sweep.

Predefined variants. Each one filters or augments reformulated_full.csv
(no API calls), trains roberta-base with the same hyperparameters as the
poster baseline, then scores on OOD + manual + Claude.

Resumable: each variant writes its own results JSON. Re-running this script
will skip variants whose results file already exists.

Variants:
    v00_baseline             - full data, 3 epochs (sanity-check)
    v01_drop_high_cosine     - drop unequivalent pairs with cosine sim > 0.85
    v02_drop_neg_shortcut    - drop unequivalent pairs that have the
                               negation-regex shortcut (negation in text_b
                               that's not in text_a)
    v03_balance_neg          - reweight negation cell to break correlation
    v04_style_mix_qwen7b     - + 4 improved-Qwen-7B reformulation CSVs
    v05_one_epoch            - same data, only 1 epoch (does shortcut
                               emerge at higher epoch count?)
    v06_combined             - v01 + v02 + v04 stacked
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, TEST_DOMAINS, TRAIN_DOMAINS,
    find_checkpoint, has_negation_flip, load_full_csv, save_json, score_all,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage1"
RUNS_DIR = REPO / "overnight" / "runs"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach helper columns used by the filter recipes."""
    df = df.copy()
    df["neg_flip"] = [
        has_negation_flip(a, b) for a, b in zip(df["explanation"], df["synthetic"])
    ]
    vec = TfidfVectorizer()
    vec.fit(pd.concat([df["explanation"].astype(str), df["synthetic"].astype(str)]))
    a = vec.transform(df["explanation"].astype(str))
    b = vec.transform(df["synthetic"].astype(str))
    sims = np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(a.shape[0])])
    df["cosine_sim"] = sims
    return df


# --- variant recipes ------------------------------------------------------
def _baseline(df: pd.DataFrame) -> pd.DataFrame:
    return df


def _drop_high_cosine(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df["label"] == 1) & (df["cosine_sim"] > 0.85))].reset_index(drop=True)


def _drop_neg_shortcut(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df["label"] == 1) & (df["neg_flip"] == 1))].reset_index(drop=True)


def _balance_neg(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample equivalent-with-negation 2x, downsample unequivalent-with-negation by 50%."""
    eq_neg = df[(df["label"] == 0) & (df["neg_flip"] == 1)]
    uneq_neg = df[(df["label"] == 1) & (df["neg_flip"] == 1)]
    eq_no = df[(df["label"] == 0) & (df["neg_flip"] == 0)]
    uneq_no = df[(df["label"] == 1) & (df["neg_flip"] == 0)]
    upsampled_eq_neg = pd.concat([eq_neg, eq_neg], ignore_index=True)
    downsampled_uneq_neg = uneq_neg.sample(frac=0.5, random_state=42)
    return pd.concat([upsampled_eq_neg, eq_no, downsampled_uneq_neg, uneq_no],
                     ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def _style_mix_qwen7b(df: pd.DataFrame) -> pd.DataFrame:
    """Augment train pool with the 4 improved-Qwen-7B reformulations."""
    aug_paths = list((REPO / "synthetic_data_generation_improved").glob("reformulated*.csv"))
    aug_rows = []
    for p in aug_paths:
        sub = pd.read_csv(p)
        if "explanation" not in sub.columns or "synthetic" not in sub.columns:
            continue
        sub = sub.dropna(subset=["explanation", "synthetic", "meaning", "domain"]).copy()
        sub["meaning"] = sub["meaning"].astype(str).str.lower().str.strip()
        sub = sub[sub["meaning"].isin(LABEL2ID)].copy()
        sub["label"] = sub["meaning"].map(LABEL2ID).astype(int)
        aug_rows.append(sub)
    if not aug_rows:
        return df
    aug = pd.concat(aug_rows, ignore_index=True)
    aug["neg_flip"] = [
        has_negation_flip(a, b) for a, b in zip(aug["explanation"], aug["synthetic"])
    ]
    aug["cosine_sim"] = 0.0
    common = ["id", "explanation", "domain", "label", "post", "target", "uid",
              "synthetic", "meaning", "neg_flip", "cosine_sim"]
    common = [c for c in common if c in df.columns and c in aug.columns]
    return pd.concat([df[common], aug[common]], ignore_index=True)


def _combined(df: pd.DataFrame) -> pd.DataFrame:
    out = _drop_high_cosine(df)
    out = _drop_neg_shortcut(out)
    out = _style_mix_qwen7b(out)
    return out.reset_index(drop=True)


def _append_overnight_log(name: str, epochs: int, metrics: dict) -> None:
    log_path = REPO / "OVERNIGHTLOG.md"
    m = metrics["manual"]; o = metrics["ood_overall"]; c = metrics["claude"]
    cm = c.get("pred_balance", {})
    pred_uneq = (int(cm.get("1", 0)) / max(c.get("n", 1), 1)) * 100
    ts = time.strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## {ts}  {name}  (stage1)\n"
        f"- recipe: {name} ({epochs} epochs)\n"
        f"- OOD F1: {o['f1_macro']:.3f}  manual_acc: {m['accuracy']*100:.1f}% "
        f"(eq={m.get('acc_equivalent', 0)*100:.0f}%/uneq={m.get('acc_unequivalent', 0)*100:.0f}%)  "
        f"claude_acc: {c['accuracy']*100:.1f}% ({pred_uneq:.0f}% predicted unequivalent)\n"
        f"- interpretation: _auto-logged; review later_\n"
        f"- file: `overnight/results/stage1/{name}.json`\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


VARIANTS: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame], int]] = [
    ("v00_baseline",         _baseline,         3),
    ("v01_drop_high_cosine", _drop_high_cosine, 3),
    ("v02_drop_neg_shortcut",_drop_neg_shortcut,3),
    ("v03_balance_neg",      _balance_neg,      3),
    ("v04_style_mix_qwen7b", _style_mix_qwen7b, 3),
    ("v05_one_epoch",        _baseline,         1),
    ("v06_combined",         _combined,         3),
]


def make_split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.1):
    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    test_df = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)
    val_chunks = []
    for _, grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=val_frac, random_state=seed))
    val_df = pd.concat(val_chunks, ignore_index=False)
    val_ids = set(val_df.index.astype(str))
    train_df = train_pool[~train_pool.index.astype(str).isin(val_ids)].reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df, test_df


def tokenize(ds: Dataset, tok, max_length: int) -> Dataset:
    def _t(batch):
        return tok(batch["text_a"], batch["text_b"],
                   truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label", "domain")]
    return ds.map(_t, batched=True, remove_columns=cols)


def train_variant(name: str, prep_fn: Callable, epochs: int, seed: int = 42) -> Path:
    set_seed(seed)
    run_dir = RUNS_DIR / name
    if (run_dir / "checkpoints").exists():
        try:
            return find_checkpoint(run_dir)
        except FileNotFoundError:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] loading + preparing data ...", flush=True)
    df = add_features(load_full_csv())
    df = prep_fn(df)
    df = df.rename(columns={"explanation": "text_a", "synthetic": "text_b"})

    train_df, val_df, _ = make_split(df, seed=seed)
    print(f"[{name}] train={len(train_df)}  val={len(val_df)}", flush=True)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize(Dataset.from_pandas(train_df[["text_a", "text_b", "label", "domain"]]), tok, 256)
    val_ds = tokenize(Dataset.from_pandas(val_df[["text_a", "text_b", "label", "domain"]]), tok, 256)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"},
        label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=4,
        seed=seed,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=collator,
    )
    trainer.train()
    return find_checkpoint(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma-sep variant names to run; default = all")
    parser.add_argument("--skip", default=None, help="comma-sep variant names to skip")
    parser.add_argument("--cleanup_checkpoints", action="store_true",
                        help="delete checkpoints after scoring to save disk")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, prep_fn, epochs in VARIANTS:
        if only is not None and name not in only:
            continue
        if name in skip:
            continue

        result_path = RESULTS_DIR / f"{name}.json"
        if result_path.exists():
            print(f"[{name}] result already exists; skipping. To rerun, delete {result_path}", flush=True)
            continue

        t0 = time.time()
        print(f"\n=== [{name}] starting ({epochs} epoch{'s' if epochs > 1 else ''}) ===", flush=True)
        try:
            ckpt = train_variant(name, prep_fn, epochs)
            print(f"[{name}] training done, scoring ...", flush=True)
            metrics = score_all(ckpt)
            metrics["_meta"] = {
                "variant": name, "epochs": epochs,
                "checkpoint": str(ckpt),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            save_json(result_path, metrics)
            print(f"[{name}] manual_acc={metrics['manual']['accuracy']:.3f}  "
                  f"claude_acc={metrics['claude']['accuracy']:.3f}  "
                  f"ood_f1={metrics['ood_overall']['f1_macro']:.3f}  "
                  f"({metrics['_meta']['elapsed_sec']:.0f}s)", flush=True)
            _append_overnight_log(name, epochs, metrics)
            if args.cleanup_checkpoints:
                # keep config.json + tokenizer; nuke heavy weights to save disk
                ckpt_root = ckpt.parent
                for sub in ckpt_root.iterdir():
                    if sub.is_dir():
                        shutil.rmtree(sub, ignore_errors=True)
        except Exception as e:
            err_path = RESULTS_DIR / f"{name}.error.txt"
            err_path.write_text(f"{type(e).__name__}: {e}\n")
            print(f"[{name}] FAILED: {e}", flush=True)
            continue

    print("\n[stage1] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
