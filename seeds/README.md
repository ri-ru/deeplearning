# `seeds/` — multiple-seed variance experiments

Re-runs every trainable component under multiple seeds to report mean ± std
of its performance. In every sweep the **OOD test set is fixed** (the 3
held-out domains — toxigen, gab, anatomy, n=17,691); the domain split does not
depend on the seed, so each spread is clean run-to-run variance, not eval noise.

All scripts run from the repo root with the project venv and read from the
reorganized layout (`data/`, `code/`). Heavy artifacts (`seeds/runs/`,
`seeds/extracts/`) are gitignored; only the scripts and `seeds/results/*.json`
are tracked.

| Script | Sweep | Output |
|---|---|---|
| `run_seed_variance.py` | RoBERTa judge, 3 seeds (42, 1, 2) — full retrain | `results/seed_variance_summary.json` |
| `tfidf_seed_sweep.py` | TF-IDF + LogReg baseline, 5 seeds (0–4) | `results/tfidf_seed_sweep.json` |
| `probe_seed_sweep.py` | per-layer probes, 10 seeds — re-split fixed residuals | `results/probe_seed_sweep.json` |
| `end2end_interp_sweep.py` | per-layer probes across the 3 independent checkpoints | `results/end2end_interp_sweep.json` |

## How to run

```powershell
$py = "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe"
& $py seeds/run_seed_variance.py          # ~36 min on a 5090 (3x RoBERTa fine-tune)
& $py seeds/tfidf_seed_sweep.py           # ~2 min

# the two interp sweeps need residuals extracted from the seed-42 checkpoint first:
& $py code/interpretability/build_manifest.py `
      --out seeds/extracts/manifest.csv `
      --synthetic_csv data/reformulated_full.csv --manual_csv data/manual_labelling.csv
& $py code/interpretability/extract_residuals.py `
      --checkpoint seeds/runs/roberta_base_seed42/checkpoints/checkpoint-* `
      --out_dir seeds/extracts --manifest seeds/extracts/manifest.csv --max_length 128 --batch_size 64
& $py seeds/probe_seed_sweep.py --seeds 0 1 2 3 4 5 6 7 8 9
& $py seeds/end2end_interp_sweep.py
```

Hyperparameters for the RoBERTa runs match the original judge config
(roberta-base, max_length 256, batch 32, 3 epochs, lr 2e-5, weight_decay 0.01,
warmup_ratio 0.06, bf16).
