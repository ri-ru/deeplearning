# Can RoBERTa Judge If Two Hate-Speech Rationales Mean the Same Thing?

**EE-559 Deep Learning — Group 26**
Veronika Wannack, Flavia Wallenhorst, Dimitri Pelgrin (EPFL)

The deliverable is the notebook **`synthetic_data_generation.ipynb`**. Run it top
to bottom. Part I (data generation) was run on the GPU cluster (vLLM + Qwen2.5-3B
+ an OpenAI key for the flip-rate audit); Part II reads results back from the
cached artifacts in `results/`, so the analysis and figures rebuild on a plain
CPU clone in seconds.

## Folder layout

```
synthetic_data_generation.ipynb   the notebook (the whole story, end to end)
README.md
data/        inputs the notebook reads
results/     cached outputs the notebook displays (numbers/figures come from here)
code/        the scripts that produced everything in results/  (run on the cluster)
```

### `data/`
| File | Used in | What it is |
|---|---|---|
| `hatecot_final_D3.csv` | Part I | raw HateCoT posts + human rationales (generation source) |
| `manual_labelling.csv` | Part I | hand-labelled examples used as few-shot demonstrations |
| `reformulated_full.csv` | Part II | the ~52k generated `(explanation, synthetic, meaning)` pairs |

### `results/`
| File | Produced by (`code/…`) |
|---|---|
| `verify_poster_results.json` | `verify/verify_poster.py` |
| `verify_ood_results.json` | `verify/verify_ood.py` |
| `verify_negation_prior_results.json` | `verify/verify_negation_prior.py` |
| `probe_metrics.json` | `interpretability/` (build_manifest → extract_residuals → train_probes) |
| `roberta_fewshot_trainer_state.json` | `roberta_baseline/run.py` |
| `cross_gen/*.csv` | `generator_diversity/stage9_generate.py` |
| `overnight/stage6_baseline_layer12.json` | `generator_diversity/stage6_multi_ablation.py` |
| `overnight/stage14_matrix.json` | `generator_diversity/stage14_human_eval_matrix.py` |
| `overnight/stage14b_matrix.json` | `generator_diversity/stage14b_human_split.py` |

### `code/` — provenance for every reported number
These ran on the cluster; the notebook's Part II cells are short excerpts of them.

| Folder | What it does |
|---|---|
| `verify/` | re-evaluates the trained judge on held-out / human / Claude rationales → the `verify_*` results |
| `judge_training/` | the RoBERTa-base sentence-pair judge: domain split (`data.py`), metrics (`metrics.py`), training loop (`train.py`) |
| `interpretability/` | per-layer [CLS] probes for the equivalence label and the negation shortcut |
| `generator_diversity/` | the multi-generator fix: regenerate pairs from 5 LLMs and re-evaluate (stages 6/9/14/14b) |
| `roberta_baseline/` | the few-shot RoBERTa baseline run that produced `roberta_fewshot_trainer_state.json` |

> Note: Part I's full generation code lives **inside the notebook** (Sections 0–5), not
> in `code/`. `code/` holds the Part II training/analysis scripts.
