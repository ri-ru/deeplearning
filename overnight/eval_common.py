"""Shared scoring helpers used by every overnight stage.

Each scorer returns a dict that is JSON-serializable so the final report
generator can just glob `overnight/results/*.json` and tabulate.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
LABEL2ID = {"equivalent": 0, "unequivalent": 1}
ID2LABEL = {0: "equivalent", 1: "unequivalent"}
TRAIN_DOMAINS = {"dynahate", "cad", "measure", "socialbias", "toraman"}
TEST_DOMAINS = {"toxigen", "gab", "anatomy"}
NEG = re.compile(r"\b(not|n't|never|no|none|nothing|neither|nor|cannot|can't)\b", re.I)


def load_full_csv() -> pd.DataFrame:
    df = pd.read_csv(REPO / "reformulated_full.csv")
    df = df.dropna(subset=["explanation", "synthetic", "meaning", "domain"]).copy()
    df["meaning"] = df["meaning"].astype(str).str.lower().str.strip()
    df = df[df["meaning"].isin(LABEL2ID)].copy()
    df["label"] = df["meaning"].map(LABEL2ID).astype(int)
    return df.reset_index(drop=True)


def load_human_csv(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) < 4 or not row[1].strip() or not row[2].strip() or not row[3].strip():
                continue
            label = row[3].strip().lower()
            if label not in LABEL2ID:
                continue
            out.append({
                "post": row[0].strip(),
                "explanation": row[1].strip(),
                "comment": row[2].strip(),
                "label": label,
            })
    return out


def has_negation_flip(text_a: str, text_b: str) -> int:
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0
    a_negs = {m.group(0).lower() for m in NEG.finditer(text_a)}
    b_negs = {m.group(0).lower() for m in NEG.finditer(text_b)}
    return int(bool(b_negs - a_negs))


def _score_pairs(pairs: list[tuple[str, str, int]], tokenizer, model, device) -> dict[str, Any]:
    labels, preds, probs = [], [], []
    for a, b, y in pairs:
        inputs = tokenizer(a, b, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0].cpu().numpy()
        p = np.exp(logits - logits.max()); p = p / p.sum()
        preds.append(int(p.argmax()))
        probs.append(float(p[1]))
        labels.append(int(y))

    y_arr = np.array(labels)
    p_arr = np.array(preds)
    cm = confusion_matrix(y_arr, p_arr, labels=[0, 1]).tolist()
    metrics: dict[str, Any] = {
        "n": int(len(pairs)),
        "accuracy": float(accuracy_score(y_arr, p_arr)),
        "f1_macro": float(f1_score(y_arr, p_arr, average="macro", zero_division=0)),
        "label_balance": dict(Counter(labels)),
        "pred_balance": dict(Counter(preds)),
        "confusion_matrix": cm,
        "mean_p_unequivalent": float(np.mean(probs)) if probs else 0.0,
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_arr, probs))
    except ValueError:
        metrics["auroc"] = None
    for cls_id, cls_name in ID2LABEL.items():
        mask = y_arr == cls_id
        if mask.sum() == 0:
            continue
        metrics[f"acc_{cls_name}"] = float((p_arr[mask] == cls_id).mean())
        metrics[f"n_{cls_name}"] = int(mask.sum())
    return metrics


def score_all(checkpoint_dir: Path, ood_test_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Score a checkpoint on (a) OOD test split, (b) manual, (c) Claude."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir)).to(device).eval()

    if ood_test_df is None:
        df = load_full_csv()
        ood_test_df = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)

    ood_pairs = list(zip(
        ood_test_df["explanation"].astype(str).tolist(),
        ood_test_df["synthetic"].astype(str).tolist(),
        ood_test_df["label"].astype(int).tolist(),
    ))
    ood_metrics = _score_pairs(ood_pairs, tokenizer, model, device)

    per_domain: dict[str, Any] = {}
    for dom in sorted(ood_test_df["domain"].unique()):
        sub = ood_test_df[ood_test_df["domain"] == dom]
        sub_pairs = list(zip(
            sub["explanation"].astype(str).tolist(),
            sub["synthetic"].astype(str).tolist(),
            sub["label"].astype(int).tolist(),
        ))
        per_domain[dom] = _score_pairs(sub_pairs, tokenizer, model, device)

    manual = load_human_csv(REPO / "manual labelling.csv")
    manual_pairs = [(r["explanation"], r["comment"], LABEL2ID[r["label"]]) for r in manual]
    manual_metrics = _score_pairs(manual_pairs, tokenizer, model, device)

    claude = load_human_csv(REPO / "roberta_training" / "claude_test_labels.csv")
    claude_pairs = [(r["explanation"], r["comment"], LABEL2ID[r["label"]]) for r in claude]
    claude_metrics = _score_pairs(claude_pairs, tokenizer, model, device)

    return {
        "ood_overall": ood_metrics,
        "ood_per_domain": per_domain,
        "manual": manual_metrics,
        "claude": claude_metrics,
    }


def find_checkpoint(run_dir: Path) -> Path:
    cands = sorted(
        (p for p in (run_dir / "checkpoints").iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not cands:
        raise FileNotFoundError(f"No checkpoint-* under {run_dir / 'checkpoints'}")
    return cands[-1]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
