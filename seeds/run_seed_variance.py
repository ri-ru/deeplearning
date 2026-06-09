"""RoBERTa judge — multi-seed variance experiment (final-submission-2 layout).

Trains the SAME roberta-base config under several seeds and reports mean +/-
std of the OOD-test metrics. The OOD test set is the 3 held-out domains
(toxigen, gab, anatomy) and does NOT depend on the seed, so the spread is
clean run-to-run training variance (weight init + data shuffle + train/val
carve), not evaluation noise.

Faithful re-implementation of the original judge_finetune/src/train.py loop,
reusing code/judge_training/{data,metrics}.py and pointing at the reorganized
paths (data/reformulated_full.csv, outputs under seeds/).

Run from the repo root with the project venv:
    & "C:\\Users\\jan\\downloads\\nml_project_proposal\\.venv\\Scripts\\python.exe" seeds/run_seed_variance.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "judge_training"))
import data as D          # noqa: E402
import metrics as M       # noqa: E402

SEEDS = [42, 1, 2]
CSV = str(REPO / "data" / "reformulated_full.csv")
RUNS = REPO / "seeds" / "runs"
RESULTS = REPO / "seeds" / "results"
METRIC_KEYS = ["accuracy", "f1_macro", "f1_unequivalent", "auroc"]

# Hyperparameters identical to the original configs/roberta_base.yaml.
CFG = dict(model_name="roberta-base", max_length=256, batch_size=32,
           eval_batch_size=128, epochs=3, learning_rate=2e-5,
           weight_decay=0.01, warmup_ratio=0.06)


def train_one(seed: int) -> dict:
    run_dir = RUNS / f"roberta_base_seed{seed}"
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        print(f"[seed {seed}] metrics.json exists -> reusing", flush=True)
        return json.loads(metrics_file.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[seed {seed}] training ...", flush=True)

    set_seed(seed)
    dc = D.DataConfig(csv_path=CSV, max_length=CFG["max_length"], seed=seed)
    train_ds, val_ds, test_ds = D.load_and_split(dc)
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    tok = AutoTokenizer.from_pretrained(CFG["model_name"])
    train_tok = D.tokenize_dataset(train_ds, tok, CFG["max_length"])
    val_tok = D.tokenize_dataset(val_ds, tok, CFG["max_length"])
    test_tok = D.tokenize_dataset(test_ds, tok, CFG["max_length"])
    test_domains = list(test_ds["domain"])

    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"], num_labels=2, id2label=D.ID2LABEL, label2id=D.LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    use_bf16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["batch_size"],
        per_device_eval_batch_size=CFG["eval_batch_size"],
        learning_rate=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
        warmup_ratio=CFG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=use_bf16,
        fp16=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=4,
        seed=seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        processing_class=tok, data_collator=collator,
        compute_metrics=M.compute_metrics_overall,
    )
    trainer.train()

    pred = trainer.predict(test_tok)
    breakdown = M.per_domain_breakdown(pred.predictions, pred.label_ids, test_domains)
    out = {"config": {**CFG, "seed": seed}, "test": breakdown}
    metrics_file.write_text(json.dumps(out, indent=2))
    print(f"[seed {seed}] done -> {metrics_file}", flush=True)
    return out


def stats(xs):
    return {"values": [round(x, 4) for x in xs], "mean": round(statistics.mean(xs), 4),
            "std": round(statistics.pstdev(xs), 4),
            "std_sample": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_seed = {s: train_one(s) for s in SEEDS}

    domains = sorted(d for d in per_seed[SEEDS[0]]["test"] if d != "overall")
    overall = {k: [per_seed[s]["test"]["overall"][k] for s in SEEDS] for k in METRIC_KEYS}
    per_domain = {dom: {"f1_macro": [per_seed[s]["test"][dom]["f1_macro"] for s in SEEDS],
                        "accuracy": [per_seed[s]["test"][dom]["accuracy"] for s in SEEDS]}
                  for dom in domains}
    summary = {"seeds": SEEDS,
               "overall": {k: stats(v) for k, v in overall.items()},
               "per_domain": {dom: {m: stats(v) for m, v in mv.items()}
                              for dom, mv in per_domain.items()}}
    (RESULTS / "seed_variance_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== ROBERTA SEED VARIANCE (OOD test) =====")
    for k in METRIC_KEYS:
        s = summary["overall"][k]
        print(f"  {k:16s} {s['mean']:.4f} +/- {s['std']:.4f}  {s['values']}")
    for dom in domains:
        s = summary["per_domain"][dom]["f1_macro"]
        print(f"  f1[{dom:8s}] {s['mean']:.4f} +/- {s['std']:.4f}  {s['values']}")
    print(f"\nWrote {RESULTS / 'seed_variance_summary.json'}")


if __name__ == "__main__":
    main()
