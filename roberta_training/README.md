# roberta_training

Few-shot augmented RoBERTa run. Loads `reformulated_full.csv` and appends
`synthetic_data_generation_improved/reformulated (4).csv` into the training
pool, OOD-split by domain (train: dynahate/cad/measure/socialbias/toraman;
test: toxigen/gab/anatomy), 3 epochs.

## Scripts

- `run.py` - train + eval, writes `runs/roberta_fewshot/`.
- `evaluate_manual.py` - score a checkpoint against the 20 manual labels.
  See `TEST_SETS.md` for the leakage caveat.
- `test_negation.py` - 20 hand-built "X" / "not X" pairs from
  `reformulated_full.csv`. Sanity check for the negation-shortcut story.

## Reproduce

```bash
pip install -r requirements.txt
python run.py \
    --synthetic ../synthetic_data_generation_improved/"reformulated (4).csv" \
    --base_csv ../reformulated_full.csv \
    --output_dir runs/roberta_fewshot
```

Best val f1_macro = 0.9898 at step 1580 (epoch 2).

## Results in `runs/roberta_fewshot/`

| file | what |
|------|------|
| `metrics.json` | OOD test per-domain accuracy / f1 / AUROC |
| `test_predictions.csv` | per-row predictions on the 17,691-row OOD test set |
| `trainer_state.json` | full training log |
| `config.json` | RoBERTa-base config |

Weights (`model.safetensors`, `optimizer.pt`) gitignored, see
`../EXCLUDED_FILES.md`.

## Headline (`metrics.json`)

| domain | n | acc | f1_macro |
|--------|---|-----|----------|
| overall | 17691 | 0.968 | 0.968 |
| anatomy | 5418 | 0.939 | 0.939 |
| gab | 5034 | 0.977 | 0.977 |
| toxigen | 7239 | 0.983 | 0.983 |

## Caveats

1. Base data has the ~50% mislabeling problem documented in
   `../audit_results.txt`. 50 rows of few-shot augment is too small to
   move the gradient much, so most of what RoBERTa learns here is still
   the negation shortcut.
2. `evaluate_manual.py` is not an independent test - see `TEST_SETS.md`.
