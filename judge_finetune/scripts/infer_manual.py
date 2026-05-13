"""Run the fine-tuned RoBERTa judge on manual labelling.csv and print row-by-row predictions.

Usage (from judge_finetune/):
    .venv\\Scripts\\python scripts\\infer_manual.py
"""
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs" / "roberta_base" / "checkpoints" / "checkpoint-1576"
CSV = ROOT.parent / "manual labelling.csv"

# The CSV's first physical line is a literal "Table 1" cell, so skip it; the
# real header (Post, Explanation, , , , , ) is on the second line. The 3rd
# column has no name in the header but holds the manual reformulation, and
# the 4th holds the Equivalent / Unequivalent label.
df = pd.read_csv(CSV, skiprows=1)
df = df.iloc[:, :4]
df.columns = ["post", "text_a", "text_b", "label_str"]
df = df.dropna(subset=["text_a", "text_b", "label_str"]).reset_index(drop=True)
df["label"] = df["label_str"].str.strip().str.lower().map(
    {"equivalent": 0, "unequivalent": 1}
)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT).to(device)
model.train(False)  # inference mode (equivalent to model.eval())

with torch.no_grad():
    enc = tok(
        list(df["text_a"]),
        list(df["text_b"]),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    logits = model(**enc).logits.float().cpu()

probs = torch.softmax(logits, dim=-1).numpy()
preds = probs.argmax(axis=1)
true = df["label"].to_numpy()
correct = (preds == true).sum()

print(f"\n=== {len(df)} examples — accuracy {correct}/{len(df)} = {correct/len(df):.1%} ===\n")

ID2LABEL = {0: "equivalent", 1: "unequivalent"}
for i, row in df.iterrows():
    mark = "OK" if preds[i] == row["label"] else "X "
    print(f"[{i:2d}] {mark}  true={ID2LABEL[row['label']]:<13} "
          f"pred={ID2LABEL[preds[i]]:<13} p_uneq={probs[i, 1]:.3f}")
    print(f"     post : {str(row['post'])[:120]}")
    print(f"     A    : {row['text_a'][:120]}")
    print(f"     B    : {row['text_b'][:120]}")
    print()

out = df.assign(pred=[ID2LABEL[p] for p in preds],
                p_unequivalent=probs[:, 1]).drop(columns=["label"])
out_path = ROOT / "runs" / "roberta_base" / "manual_predictions.csv"
out.to_csv(out_path, index=False)
print(f"[saved] {out_path}")
