# Overnight experiment pipeline

Three-stage, resumable, pre-defined experiment sweep that runs without
external API calls. All artifacts land in `overnight/`:

```
overnight/
  eval_common.py                # shared scoring helpers
  stage1_mitigations.py         # data-mitigation variant sweep
  stage2_interp.py              # per-layer probes + negation ablation
  stage3_training_dynamics.py   # checkpoint-by-checkpoint dynamics
  summarize.py                  # writes OVERNIGHT_REPORT.md
  run_all.py                    # top-level orchestrator (s1 -> s2 -> s3)
  results/                      # per-experiment JSON (tracked in git)
  runs/                         # training checkpoints (gitignored, heavy)
  extracts/                     # residual .pt files (gitignored, heavy)
  run.log                       # append-only run log (gitignored)
```

## How to launch

```powershell
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe"

cd "C:\Users\jan\Downloads\deep learning"
& $py "overnight\run_all.py" --cleanup_checkpoints
```

`--cleanup_checkpoints` deletes the Stage-1 model weights after each variant
is scored. The per-variant JSON survives. Saves ~500 MB per variant.

To run just one stage:

```powershell
& $py "overnight\run_all.py" --skip s2,s3      # stage 1 only
& $py "overnight\stage1_mitigations.py" --only v01_drop_high_cosine,v02_drop_neg_shortcut
```

To peek at partial results while it's running:

```powershell
& $py "overnight\summarize.py"
# then open OVERNIGHT_REPORT.md
```

## Stage 1 — 6 data-mitigation variants

Each variant filters or augments `reformulated_full.csv`, trains a fresh
`roberta-base` (3 epochs, same hyperparams as the poster baseline), scores
on (a) OOD test, (b) the 20 manual-labelled pairs, (c) the 20 Claude pairs.

| id | recipe |
|---|---|
| `v00_baseline` | unmodified data, 3 epochs (sanity-check: should reproduce 0.968 OOD F1, ~55 % manual) |
| `v01_drop_high_cosine` | drop unequivalent pairs with TF-IDF cosine sim > 0.85 |
| `v02_drop_neg_shortcut` | drop unequivalent pairs whose `text_b` introduces a negation `text_a` does not have |
| `v03_balance_neg` | upsample equivalent+negation 2x, downsample unequivalent+negation 0.5x |
| `v04_style_mix_qwen7b` | add the 4 improved-Qwen-7B reformulations to train |
| `v05_one_epoch` | full data, only 1 epoch (does the shortcut emerge later in training?) |
| `v06_combined` | v01 ∩ v02 ∪ v04 |

Wall-clock per variant: ~3 min training + ~10–30 s batched scoring on a 5090.

## Stage 2 — per-layer probes + negation-direction ablation

Re-extracts the [CLS] residual stream at every transformer layer for
(synthetic + manual + Claude) using either the baseline checkpoint or the
best Stage-1 mitigation. Trains three probe types per layer:
- linear logreg on the equivalence label
- linear logreg on the negation-flip regex
- 2-layer MLP on the equivalence label

Then computes the **negation direction** = the logreg coefficient vector of
the negation probe. Projects this direction out of [CLS] and re-trains the
equivalence probe. The drop in val AUC tells you how much the model was
leaning on the negation feature at that layer.

## Stage 3 — training dynamics

Trains the baseline from scratch with frequent checkpoints (every 200 steps,
~12 snapshots across 3 epochs) and scores each one on val + manual +
Claude. The story we expect: OOD F1 climbs monotonically to ~0.97, but
manual accuracy might peak early and then degrade as the model overfits
the Qwen-prose shortcut.

## Output

`overnight/summarize.py` produces `OVERNIGHT_REPORT.md` with three
markdown tables: Stage-1 leaderboard sorted by manual accuracy, Stage-2
per-layer probe metrics with shortcut-tax column, Stage-3 step-by-step
timeline.
