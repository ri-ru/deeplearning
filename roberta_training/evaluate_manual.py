"""
Evaluate the trained RoBERTa model on the manual labelling CSV.

Usage:
    python evaluate_manual.py --checkpoint runs/roberta_fewshot/checkpoints/checkpoint-XXXX
"""
import argparse
import csv
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LABEL2ID = {"equivalent": 0, "unequivalent": 1}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def load_manual_labels(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip "Table 1"
        next(reader)  # skip header row
        for row in reader:
            if len(row) < 4 or not row[1].strip() or not row[2].strip():
                continue
            rows.append({
                "post":          row[0].strip(),
                "explanation":   row[1].strip(),
                "human_comment": row[2].strip(),
                "human_label":   row[3].strip().lower(),
            })
    return rows


def find_checkpoint(checkpoint_arg: str) -> str:
    """If user passed the checkpoints/ dir, find the latest checkpoint inside."""
    p = Path(checkpoint_arg)
    if p.name.startswith("checkpoint-"):
        return str(p)
    # look for checkpoint-XXXX subdirs
    candidates = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if candidates:
        return str(candidates[-1])
    return checkpoint_arg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint dir, e.g. runs/roberta_fewshot/checkpoints")
    p.add_argument("--manual_labels", default="manual_labelling.csv")
    args = p.parse_args()

    checkpoint = find_checkpoint(args.checkpoint)
    print(f"Loading model from: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    rows = load_manual_labels(args.manual_labels)
    print(f"Loaded {len(rows)} manual labels\n")

    correct = 0
    print(f"{'#':<3} {'Human':<14} {'Model':<14} {'Confidence':>10}  Match  Explanation[:60]")
    print("-" * 100)

    for i, row in enumerate(rows, 1):
        inputs = tokenizer(
            row["explanation"],
            row["human_comment"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()

        probs      = softmax(logits)
        pred_id    = int(probs.argmax())
        pred_label = ID2LABEL[pred_id]
        confidence = probs[pred_id]
        match      = "✓" if pred_label == row["human_label"] else "✗"
        if pred_label == row["human_label"]:
            correct += 1

        print(f"{i:<3} {row['human_label']:<14} {pred_label:<14} {confidence:>10.1%}  {match}     {row['explanation'][:60]}")

    print("-" * 100)
    print(f"\nAccuracy on manual labels: {correct}/{len(rows)} = {correct/len(rows):.1%}")


if __name__ == "__main__":
    main()
