# Rationale Equivalence Judge

Fine-tunes a transformer encoder to classify whether two hate-speech rationales are semantically equivalent.

**Task:** binary sentence-pair classification.
**Input:** `(explanation, meaning)` - original rationale + reformulation.
**Label:** `equivalent` (0) / `unequivalent` (1).
**Data:** ~52k rows, balanced 50/50, 8 domains.

## OOD split (the headline experiment)

- **Train domains:** `dynahate, cad, measure, socialbias, toraman`
- **Test domains:** `toxigen, gab, anatomy`

Random splits are uninformative - the model memorizes per-domain stylistic cues. Domain-OOD split is what makes the generalization claim meaningful.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Quick smoke test (2k samples, ~1 min on 5090)
python -m src.train --config configs/smoke.yaml

# Full RoBERTa-base run (~10-15 min on 5090)
python -m src.train --config configs/roberta_base.yaml

# DeBERTa-v3-large with bf16 (~30-45 min on 5090)
python -m src.train --config configs/deberta_large.yaml
```

## Evaluation

`src/train.py` automatically reports:
- Overall test F1, accuracy, AUROC
- Per-domain F1 on the OOD test set (this is the table you want in the report)
- Confusion matrix (false-equivalent vs false-unequivalent rates)

Outputs land in `runs/<run_name>/` with metrics JSON, checkpoint, and a CSV of per-example predictions for error analysis.

## Files

- `src/data.py` - dataset loading, domain split, tokenization
- `src/train.py` - training loop using HF Trainer
- `src/eval.py` - standalone eval on a saved checkpoint (for re-running metrics without retraining)
- `configs/*.yaml` - model + hyperparam configs
- `scripts/zero_shot_baseline.py` - zero-shot GPT-4o / Claude baseline for comparison
