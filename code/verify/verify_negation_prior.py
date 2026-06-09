"""Complementary shortcut-evidence check that does NOT need residuals. 
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
PREDS_CSV = ROOT / "judge_finetune" / "runs" / "roberta_base" / "test_predictions.csv"

NEG = re.compile(r"\b(not|n't|never|no|none|nothing|neither|nor|cannot|can't)\b", re.I)


def has_negation_flip(text_a: str, text_b: str) -> int:
    """1 if text_b contains a negation token that text_a does not."""
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0
    a_negs = set(m.group(0).lower() for m in NEG.finditer(text_a))
    b_negs = set(m.group(0).lower() for m in NEG.finditer(text_b))
    return int(bool(b_negs - a_negs))


def main() -> int:
    df = pd.read_csv(PREDS_CSV)
    df["neg_in_b_not_a"] = [
        has_negation_flip(a, b) for a, b in zip(df["text_a"], df["text_b"])
    ]
    df["correct"] = (df["pred"] == df["label"]).astype(int)

    print(f"[data] n={len(df)}  pos_rate={df['label'].mean():.3f}")

    # Joint distribution
    ct = pd.crosstab(df["label"], df["neg_in_b_not_a"], margins=True, margins_name="ALL")
    print("\nJoint count (rows=true label, cols=has_neg_flip):")
    print(ct.to_string())

    print("\nModel accuracy by (label, has_neg_flip):")
    acc_table = df.groupby(["label", "neg_in_b_not_a"])["correct"].agg(["mean", "count"])
    print(acc_table.to_string())

    # Regex-only AUC vs the true label
    auc = roc_auc_score(df["label"], df["neg_in_b_not_a"])
    print(f"\nRegex-only AUC (has_neg_flip vs true label, OOD test): {auc:.4f}")

    # Logistic regression using JUST the negation feature on train, evaluated on test
    # (re-uses test split via the same predictions file)
    feat = df[["neg_in_b_not_a"]].values
    # Self-test (proxy for the prior strength)
    clf = LogisticRegression()
    clf.fit(feat, df["label"])
    preds = clf.predict(feat)
    f1 = f1_score(df["label"], preds, average="macro")
    print(f"LR(neg_flip_only) F1 on the SAME OOD test split: {f1:.4f}")

    # How well does the regex predict the MODEL's own prediction?
    auc_model_match = roc_auc_score(df["pred"], df["neg_in_b_not_a"])
    print(f"Regex AUC vs MODEL prediction (not true label): {auc_model_match:.4f}")

    out = {
        "regex_auc_vs_label":      float(auc),
        "regex_auc_vs_model_pred": float(auc_model_match),
        "lr_neg_only_f1_macro":    float(f1),
        "accuracy_by_cell": {
            f"label={int(k[0])}_flip={int(k[1])}": {
                "acc":   float(v["mean"]),
                "count": int(v["count"]),
            } for k, v in acc_table.iterrows()
        },
    }
    (ROOT / "verify_negation_prior_results.json").write_text(json.dumps(out, indent=2))
    print("\n[done] wrote verify_negation_prior_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
