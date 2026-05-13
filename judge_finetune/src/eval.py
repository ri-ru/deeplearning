"""Re-run evaluation on a saved checkpoint without retraining.

Usage:
    python -m src.eval --checkpoint runs/roberta_base/checkpoints/checkpoint-XXXX \
                       --csv /path/to/reformulated_full.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import DataConfig, load_and_split, tokenize_dataset
from .metrics import per_domain_breakdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()

    _, _, test_ds = load_and_split(DataConfig(csv_path=args.csv, max_length=args.max_length))
    domains = list(test_ds["domain"])
    test_tok = tokenize_dataset(test_ds, tok, args.max_length)
    test_tok = test_tok.with_format("torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tok)
    loader = DataLoader(test_tok, batch_size=args.batch_size, collate_fn=collator)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)

    import numpy as np
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    breakdown = per_domain_breakdown(logits, labels, domains)
    out_path = Path(args.out) if args.out else Path(args.checkpoint) / "eval.json"
    out_path.write_text(json.dumps(breakdown, indent=2))
    print(json.dumps(breakdown, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
