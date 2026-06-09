"""Verify the OOD failure 
- 55% accuracy on the 19 manually-labelled human-written rationales
  (`manual labelling.csv`).

"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "judge_finetune" / "runs" / "roberta_base" / "checkpoints" / "checkpoint-1576"
LABEL2ID = {"equivalent": 0, "unequivalent": 1}
ID2LABEL = {0: "equivalent", 1: "unequivalent"}


def load_human_pairs(path: Path) -> list[dict]:
    """Both 'manual labelling.csv' and 'claude_test_labels.csv' share a
    'Table 1' / header / data layout: row0 = "Table 1", row1 = header,
    rows 2+ = (post, explanation, human_or_claude_comment, label, ...).
    """
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
                "post":        row[0].strip(),
                "explanation": row[1].strip(),
                "comment":     row[2].strip(),
                "label":       label,
            })
    return out


def score(pairs: list[dict], tokenizer, model, device) -> dict:
    labels, preds, probs = [], [], []
    for r in pairs:
        inputs = tokenizer(
            r["explanation"], r["comment"],
            return_tensors="pt", truncation=True, max_length=256,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0].cpu().numpy()
        p = np.exp(logits - logits.max()); p = p / p.sum()
        preds.append(int(p.argmax()))
        probs.append(float(p[1]))
        labels.append(LABEL2ID[r["label"]])

    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1]).tolist()

    res = {
        "n":               len(pairs),
        "accuracy":        float(accuracy_score(labels_arr, preds_arr)),
        "f1_macro":        float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
        "label_balance":   dict(Counter(labels)),
        "pred_balance":    dict(Counter(preds)),
        "confusion_matrix": cm,
        "mean_p_unequivalent": float(np.mean(probs)),
    }
    for cls_id, cls_name in ID2LABEL.items():
        mask = labels_arr == cls_id
        if mask.sum() == 0:
            continue
        res[f"acc_{cls_name}"] = float((preds_arr[mask] == cls_id).mean())
        res[f"n_{cls_name}"]   = int(mask.sum())
    return res


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[ckpt]   {CKPT}")
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CKPT)).to(device).eval()

    manual = load_human_pairs(ROOT / "manual labelling.csv")
    claude = load_human_pairs(ROOT / "roberta_training" / "claude_test_labels.csv")
    print(f"[data] manual={len(manual)}  claude={len(claude)}")

    results = {
        "manual": score(manual, tokenizer, model, device),
        "claude": score(claude, tokenizer, model, device),
    }
    print("\n=== MANUAL (Veronika's human-written comments) ===")
    print(json.dumps(results["manual"], indent=2))
    print(f"  poster claim: 55%  |  recomputed acc: {results['manual']['accuracy']*100:.1f}%")

    print("\n=== CLAUDE (Claude-4.6-written rationales) ===")
    print(json.dumps(results["claude"], indent=2))
    cm = results["claude"]["confusion_matrix"]
    pred_uneq_total = cm[0][1] + cm[1][1]
    print(f"  poster claim: severe bias toward one class on Claude")
    print(f"  predicted 'unequivalent': {pred_uneq_total}/{results['claude']['n']} = "
          f"{pred_uneq_total/results['claude']['n']*100:.1f}%")

    (ROOT / "verify_ood_results.json").write_text(json.dumps(results, indent=2))
    print("\n[done] results -> verify_ood_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
