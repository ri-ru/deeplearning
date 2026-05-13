"""Metrics: overall + per-domain breakdown for OOD analysis."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics_overall(eval_pred):
    """For HF Trainer. Receives EvalPrediction(predictions, label_ids)."""
    logits, labels = eval_pred
    probs = softmax_np(logits)
    preds = probs.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_unequivalent": f1_score(labels, preds, pos_label=1),
        "auroc": roc_auc_score(labels, probs[:, 1]),
    }


def per_domain_breakdown(
    logits: np.ndarray, labels: np.ndarray, domains: list[str]
) -> dict:
    """Returns metrics per domain plus a confusion matrix overall."""
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
        if mask.sum() == 0:
            continue
        y = labels[mask]
        p = preds[mask]
        pr = probs[mask, 1]
        try:
            auroc = float(roc_auc_score(y, pr))
        except ValueError:
            auroc = float("nan")  # single-class slice
        out[dom] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro")),
            "f1_unequivalent": float(f1_score(y, p, pos_label=1, zero_division=0)),
            "auroc": auroc,
        }
    return out
