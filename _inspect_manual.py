"""Run multi_gen on each of the 20 handwritten pairs and show predictions."""
import sys, csv
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from overnight.eval_common import LABEL2ID, load_human_csv

ID2LABEL = {0: "equivalent", 1: "unequivalent"}

ckpt = REPO / "overnight" / "runs" / "stage9" / "multi_gen" / "checkpoints"
ckpt = sorted(ckpt.glob("checkpoint-*"))[-1]
print(f"multi_gen ckpt: {ckpt}\n")

tok = AutoTokenizer.from_pretrained(str(ckpt))
model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to("cuda")
model.train(False)

rows = load_human_csv(REPO / "manual labelling.csv")
print(f"n={len(rows)}\n")

correct = 0
for i, r in enumerate(rows):
    inputs = tok(r["explanation"], r["comment"], return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    p = np.exp(logits - logits.max()); p = p / p.sum()
    pred = int(p.argmax())
    true = LABEL2ID[r["label"]]
    ok = pred == true
    correct += ok
    mark = "OK " if ok else "WRONG"
    print(f"--- {i+1:>2}  true={r['label']:<13} pred={ID2LABEL[pred]:<13} p_uneq={p[1]:.3f}  [{mark}]")
    print(f"     post: {r['post'][:80]}")
    print(f"     explanation: {r['explanation'][:160]}")
    print(f"     comment: {r['comment'][:160]}")
    print()

print(f"overall: {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%")
