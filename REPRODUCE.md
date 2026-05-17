# Reproducing the EE-559 poster (Group 26) — verification log

Everything below was run on **2026-05-17** on Windows 11 with an RTX 5090 (CUDA 12.8).
All commits live on the branch `verify-poster-claims` and have **not been pushed**.

```
git log verify-poster-claims --oneline ^main
040e76f verify mech-interp ...
c5188cb verify OOD ...
cf2b955 verify poster ...
```

## 0. Environment

Python venv with PyTorch 2.11.0+cu128, transformers 5.7.0, etc.:

```powershell
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\Activate.ps1"
# or call the interpreter directly:
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe"
```

All scripts are run from the repo root: `C:\Users\jan\Downloads\deep learning`.

On Windows you also need `$env:PYTHONIOENCODING="utf-8"` before running anything
that prints Greek letters (the probe script prints `ρ` for Pearson correlation).

## 1. Data

| File | Rows | Notes |
|---|---|---|
| `reformulated_full.csv` | 52,675 (52,137 usable) | Qwen2.5-3B-Instruct rewrites of HateCoT rationales. Columns: `id, explanation, domain, label, post, target, uid, synthetic, meaning`. `meaning ∈ {equivalent, unequivalent}` is the binary label. |
| `manual labelling.csv` | 20 usable | Veronika's hand-written rationale comments against Qwen explanations. Layout: row 0 = "Table 1", row 1 = header, then `post / explanation / human_comment / label`. |
| `roberta_training/claude_test_labels.csv` | 20 usable | Same schema, comments written by Claude 4.6. Used for the "different generator → severe bias" claim. |

Domain split (hard-coded in `judge_finetune/src/data.py`):
- **train/val (5 domains):** dynahate, cad, measure, socialbias, toraman → 34,446 rows
- **OOD test (3 domains):** anatomy, gab, toxigen → 17,691 rows

## 2. Trained checkpoint

The headline RoBERTa numbers in the poster come from:
`judge_finetune/runs/roberta_base/checkpoints/checkpoint-1576`

Config (`judge_finetune/configs/roberta_base.yaml`): roberta-base, max_length=256,
batch=32, lr=2e-5, epochs=3, bf16, weight_decay=0.01, warmup_ratio=0.06, seed=42.
Training is re-runnable but takes ~10–15 min on a 5090:

```powershell
cd "C:\Users\jan\Downloads\deep learning\judge_finetune"
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" -m src.train --config configs/roberta_base.yaml
```

No retraining was required for the verification below — the checkpoint on disk
is the one that produced the poster numbers.

## 3. Verification scripts and results

Each section gives the exact command and the reproduced output. Numbers in
**bold** are the poster's claim; the value after the slash is what I observed.

### 3.1 TF-IDF + GPT-4o-mini + RoBERTa OOD + cosine shortcut

```powershell
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" verify_poster.py
```

This (a) trains the TF-IDF baseline, (b) reads the cached GPT-4o-mini result,
(c) recomputes RoBERTa F1 from saved test predictions, (d) recomputes
TF-IDF cosine-sim histogram over all 52,137 pairs.

Reproduced (`verify_poster_results.json`):

| Quantity | Poster | Recomputed |
|---|---|---|
| TF-IDF F1 (macro, OOD test) | **0.789** | 0.7891 |
| GPT-4o-mini F1 (n=500 from OOD test) | **0.964** | 0.9637 |
| RoBERTa F1 (overall, OOD test, n=17,691) | **0.968** | 0.9681 |
| RoBERTa F1 per domain | — | anatomy 0.9488, gab 0.9760, toxigen 0.9769 |
| Equivalent pairs with cosine sim > 0.9 | **21.6 %** | 21.60 % |
| Unequivalent pairs with cosine sim > 0.9 | **50.9 %** | 50.87 % |
| Unequivalent cosine sim mean / median | — | 0.784 / 0.904 |

### 3.2 OOD failure on human-written + Claude-written rationales

```powershell
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" verify_ood.py
```

Loads `checkpoint-1576`, runs the manual + Claude CSVs through it. Reproduced
(`verify_ood_results.json`):

**Manual set (n=20, 13 equivalent / 7 unequivalent):**
- Accuracy: **55 %** poster / **55.0 %** recomputed
- F1 macro: 0.5396
- Confusion matrix `[[TN,FP],[FN,TP]] = [[4,9],[0,7]]`
- Per-class accuracy: equivalent **31 %**, unequivalent **100 %**

**Claude set (n=20, 10/10 balanced):**
- Accuracy: 65 %
- 17/20 = **85 %** of predictions are "unequivalent" → severe bias confirmed
- Per-class accuracy: equivalent **30 %**, unequivalent **100 %**

The model never gets an "unequivalent" pair wrong on either set, but
mis-flips 7/10 equivalent pairs to "unequivalent" — i.e. once style is not
Qwen-prose, it predicts unequivalent.

### 3.3 Negation-regex shortcut on the OOD test set

```powershell
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" verify_negation_prior.py
```

Cross-tabulates the model's saved OOD predictions against a regex feature
`has_neg_flip = (text_b contains a negation token text_a does not)`.

Reproduced (`verify_negation_prior_results.json`):

| Cell | n | Model accuracy |
|---|---|---|
| label=equivalent, no flip | 8,714 | 97.5 % |
| label=equivalent, flip | 157 | **100.0 %** |
| label=unequivalent, no flip | 4,783 | **92.7 %** |
| label=unequivalent, flip | 4,037 | **99.97 %** |

- Regex AUC vs the true label: **0.72**.
- Regex AUC vs the model's own prediction: **0.72** (same).
- 4,037 / 4,194 examples that have a negation flip are unequivalent
  → the regex is itself the shortcut.

### 3.4 Mechanistic-interpretability probes (full rerun)

Two-step. First re-extract the per-layer [CLS] residual stream for all
52,157 (synthetic + 20 manual) pairs:

```powershell
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" `
  "mech_interp\scripts\extract_residuals.py" `
  --checkpoint "judge_finetune\runs\roberta_base\checkpoints\checkpoint-1576" `
  --batch_size 64 --max_length 128
```

Runs in ~38 s on the 5090. Produces `mech_interp/extracts/layer_{00..12}.pt`
(13 files, ~155 MB each, ~2 GB total — already in `.gitignore`).

Then train the per-layer logistic probes for (a) the equivalence label and
(b) a regex "text_b contains a negation token":

```powershell
$env:PYTHONIOENCODING="utf-8"
& "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe" `
  "mech_interp\scripts\train_probes.py"
```

Reproduced (`mech_interp/results/probe_metrics.json`,
`mech_interp/figures/probe_auc_layer_sweep.png`,
`mech_interp/figures/probe_prediction_correlation.png`):

| Layer | equiv AUC (val) | neg AUC (val) | equiv AUC (manual, n=20) |
|---|---|---|---|
| 0 (embeddings) | 0.500 | 0.500 | 0.500 |
| 1 | 0.754 | 0.845 | 0.527 |
| 4 | 0.888 | 0.909 | 0.538 |
| 6 | 0.957 | 0.925 | 0.670 |
| 9 | 0.999 | 0.967 | 0.989 |
| 12 | **0.999** | **0.960** | 0.846 |

- The "regex-only prior" AUC vs the true label on synthetic is 0.6086.
- The negation-presence feature is decoded from [CLS] almost as well as the
  actual equivalence label at every deep layer.
- Peak Pearson correlation between the two probes' predictions is **0.36** at
  layer 4 (not >0.85). So negation is one of several stylistic cues the
  representation encodes, not a single localized shortcut circuit. The
  script's automatic verdict line correctly says
  `peak ρ at layer 0: nan  (no clear localization)`.

The original probe results (saved before the rerun) gave layer-12 equiv AUC
0.9946 / neg AUC 0.9756; the differences are within probe-training variance.

## 4. Files added by this verification

```
verify_poster.py               # §3.1 — TF-IDF, GPT cache, RoBERTa, cosine
verify_poster_results.json
verify_ood.py                  # §3.2 — manual + Claude
verify_ood_results.json
verify_negation_prior.py       # §3.3 — regex cross-tab on OOD preds
verify_negation_prior_results.json
mech_interp/results/probe_metrics.json   # §3.4 (overwritten by rerun)
mech_interp/figures/probe_auc_layer_sweep.png
mech_interp/figures/probe_prediction_correlation.png
```

## 5. One-shot reproduce

```powershell
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe"

cd "C:\Users\jan\Downloads\deep learning"

& $py verify_poster.py
& $py verify_ood.py
& $py verify_negation_prior.py
& $py "mech_interp\scripts\extract_residuals.py" `
       --checkpoint "judge_finetune\runs\roberta_base\checkpoints\checkpoint-1576" `
       --batch_size 64 --max_length 128
& $py "mech_interp\scripts\train_probes.py"
```

Wall-clock: ~2 minutes end-to-end (TF-IDF ~30 s, residual extract ~40 s,
probes ~30 s, the rest are negligible).
