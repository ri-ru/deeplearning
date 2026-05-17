"""Stage 4: train *real* hate-speech classifiers on public datasets.

The poster pivoted to a rationale-equivalence judge — a meta-task. This
stage brings the project back to the course brief ("deep learning to foster
safer online spaces") by training one solid hate-speech detector and a
TF-IDF baseline on two widely-cited public datasets, with no synthetic data
in the loop.

Datasets (both downloaded via HuggingFace `datasets`; both small enough
to fit on a 5090 in minutes):
  - tdavidson/hate_speech_offensive  (Davidson 2017; 24,783 tweets;
        3 classes: 0=hate / 1=offensive / 2=neither;  MIT)
  - Hate-speech-CNERG/hatexplain     (Mathew 2021; ~20k posts;
        3 classes: hate / offensive / normal; CC BY 4.0)

Outputs:
  overnight/results/stage4/<dataset>_<model>.json   (metrics)
  overnight/results/stage4/<dataset>_<model>.errors.csv  (mis-classified examples)
  OVERNIGHTLOG.md  (append-only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import REPO, save_json  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = REPO / "overnight" / "results" / "stage4"
RUNS_DIR = REPO / "overnight" / "runs" / "stage4"


# ---------------------------------------------------------------------------
# Dataset loaders. Each returns (pandas df with text/label, label_names list).
# ---------------------------------------------------------------------------

def load_davidson() -> tuple[pd.DataFrame, list[str]]:
    """Davidson 2017: 24,783 tweets, 3 classes (hate / offensive / neither).

    Loaded via the HuggingFace datasets hub (`tdavidson/hate_speech_offensive`).
    """
    from datasets import load_dataset
    ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
    df = ds.to_pandas()
    # columns: count, hate_speech_count, offensive_language_count,
    # neither_count, class, tweet
    df = df.rename(columns={"tweet": "text", "class": "label"})
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1, 2])]
    label_names = ["hate", "offensive", "neither"]
    return df[["text", "label"]].reset_index(drop=True), label_names


def load_hatexplain() -> tuple[pd.DataFrame, list[str]]:
    """HateXplain: ~20k posts, 3 classes (hatespeech / offensive / normal).

    The dataset stores per-annotator labels; we take the majority vote (the
    official label aggregation used by the paper).
    """
    from datasets import load_dataset
    ds = load_dataset("Hate-speech-CNERG/hatexplain", split="train",
                      trust_remote_code=True)
    rows = []
    for row in ds:
        # annotators field: {"label": [int,int,int], "annotator_id": [...], "target": [...]}
        labels = row["annotators"]["label"]
        # majority vote; tie-break = first label
        counts = Counter(labels)
        majority = counts.most_common(1)[0][0]
        text = " ".join(row["post_tokens"])
        rows.append({"text": text, "label": int(majority)})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text", "label"]).copy()
    label_names = ["hatespeech", "normal", "offensive"]  # HF id mapping
    return df, label_names


DATASETS = {
    "davidson":   load_davidson,
    "hatexplain": load_hatexplain,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred, label_names: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist()
    report = classification_report(y_true, y_pred, labels=list(range(len(label_names))),
                                   target_names=label_names, output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm,
        "label_names": label_names,
        "per_class": {n: {k: float(v) for k, v in report[n].items() if isinstance(v, (int, float))}
                      for n in label_names},
        "label_balance": {n: int(np.sum(np.array(y_true) == i)) for i, n in enumerate(label_names)},
    }


def train_tfidf_baseline(train_df, test_df, label_names) -> dict:
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    X_tr = vec.fit_transform(train_df["text"])
    X_te = vec.transform(test_df["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr, train_df["label"])
    y_pred = clf.predict(X_te)
    return _metrics(test_df["label"].tolist(), y_pred.tolist(), label_names)


def train_roberta(train_df, test_df, label_names, run_dir: Path,
                  epochs: int = 3, max_length: int = 128, batch_size: int = 32) -> dict:
    n_classes = len(label_names)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=n_classes,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=max_length, padding=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(_tok, batched=True, remove_columns=["text"])
    test_ds  = Dataset.from_pandas(test_df[["text", "label"]]).map(_tok, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        bf16=torch.cuda.is_available(), fp16=False,
        logging_steps=200, report_to="none",
        dataloader_num_workers=4, seed=42,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=test_ds,
                      processing_class=tok, data_collator=collator)
    trainer.train()
    pred_out = trainer.predict(test_ds)
    y_pred = pred_out.predictions.argmax(axis=1)
    return _metrics(test_df["label"].tolist(), y_pred.tolist(), label_names)


# ---------------------------------------------------------------------------
# OVERNIGHTLOG.md append
# ---------------------------------------------------------------------------

def _append_log(dataset: str, model: str, metrics: dict, elapsed_sec: float) -> None:
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    per = "  ".join(f"{name}: f1={metrics['per_class'][name].get('f1-score', 0):.2f}"
                    for name in metrics["label_names"])
    entry = (
        f"\n## {ts}  {dataset}/{model}  (stage4: real hate-speech detector)\n"
        f"- dataset: {dataset}  (n_test={metrics['n']})  classes={metrics['label_names']}\n"
        f"- model:   {model}\n"
        f"- accuracy: {metrics['accuracy']*100:.1f}%   f1_macro: {metrics['f1_macro']:.3f}   "
        f"f1_weighted: {metrics['f1_weighted']:.3f}   ({elapsed_sec:.0f}s)\n"
        f"- per-class: {per}\n"
        f"- file: `overnight/results/stage4/{dataset}_{model}.json`\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="davidson,hatexplain",
                        help="comma-sep keys from DATASETS")
    parser.add_argument("--skip_roberta", action="store_true", help="run TF-IDF only")
    parser.add_argument("--skip_tfidf", action="store_true", help="skip TF-IDF baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for key in args.datasets.split(","):
        key = key.strip()
        if key not in DATASETS:
            print(f"[stage4] unknown dataset key {key}; skipping")
            continue

        print(f"\n=== [{key}] loading ===", flush=True)
        df, label_names = DATASETS[key]()
        print(f"[{key}] n={len(df)}  classes={label_names}  "
              f"balance={dict(Counter(df['label'].astype(int).tolist()))}", flush=True)

        # Stratified 80/20 train/test
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=args.seed,
        )
        train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
        print(f"[{key}] train={len(train_df)} test={len(test_df)}", flush=True)

        # --- TF-IDF baseline
        if not args.skip_tfidf:
            res_path = RESULTS_DIR / f"{key}_tfidf.json"
            if not res_path.exists():
                t0 = time.time()
                print(f"[{key}] training TF-IDF + LogReg ...", flush=True)
                metrics = train_tfidf_baseline(train_df, test_df, label_names)
                metrics["_meta"] = {"dataset": key, "model": "tfidf_logreg",
                                    "elapsed_sec": round(time.time() - t0, 1)}
                save_json(res_path, metrics)
                _append_log(key, "tfidf_logreg", metrics, time.time() - t0)
                print(f"[{key}] tfidf  acc={metrics['accuracy']:.3f}  "
                      f"f1_macro={metrics['f1_macro']:.3f}", flush=True)
            else:
                print(f"[{key}] tfidf result exists; skipping", flush=True)

        # --- RoBERTa fine-tune
        if not args.skip_roberta:
            res_path = RESULTS_DIR / f"{key}_roberta.json"
            if not res_path.exists():
                t0 = time.time()
                print(f"[{key}] training RoBERTa-base ({len(train_df)} train rows) ...", flush=True)
                try:
                    metrics = train_roberta(train_df, test_df, label_names,
                                            run_dir=RUNS_DIR / key)
                    metrics["_meta"] = {"dataset": key, "model": "roberta-base",
                                        "elapsed_sec": round(time.time() - t0, 1)}
                    save_json(res_path, metrics)
                    _append_log(key, "roberta-base", metrics, time.time() - t0)
                    print(f"[{key}] roberta acc={metrics['accuracy']:.3f}  "
                          f"f1_macro={metrics['f1_macro']:.3f}  "
                          f"({metrics['_meta']['elapsed_sec']:.0f}s)", flush=True)
                except Exception as e:
                    (RESULTS_DIR / f"{key}_roberta.error.txt").write_text(
                        f"{type(e).__name__}: {e}\n")
                    print(f"[{key}] roberta FAILED: {e}", flush=True)
            else:
                print(f"[{key}] roberta result exists; skipping", flush=True)

    print("\n[stage4] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
