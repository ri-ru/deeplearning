# Running SameSense

Reading saved results is local and dependency-free. Training downloads a base model; generation and historical analysis need additional infrastructure.

## Inspect the saved results

From a clone, with Python 3.10 or newer:

```bash
python3 scripts/project_summary.py
python3 scripts/project_summary.py --json
```

The summary reads data and metrics relative to the script, so it also works when invoked from another directory. It counts usable labelled pairs, excludes empty/invalid generation outputs, and displays the original and multi-generator accuracy side by side. No inference, paid API calls, or new evaluation is performed.

## Train a new RoBERTa judge

Use Python 3.11 or 3.12. A CUDA GPU is recommended for the full configuration; the smoke configuration is small enough to check the pipeline on CPU, though model download and initialization still take time.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd code
python -m judge_training.train --config ../configs/roberta_smoke.yaml
```

For the full single-generator training configuration:

```bash
python -m judge_training.train --config ../configs/roberta_base.yaml
```

Run these training commands from `code/`: CSV paths in the YAML configurations are relative to that directory. Both use the committed `data/reformulated_full.csv` and download `roberta-base` from Hugging Face. No LLM API key is needed. BF16 is enabled only when CUDA is available; change the configuration if your GPU does not support it.

Outputs go to `code/runs/<run_name>/`:

| Output | Contents |
|---|---|
| `checkpoints/` | Saved model checkpoints from training |
| `metrics.json` | Configuration and test metrics, including the domain breakdown |
| `test_predictions.csv` | Pair texts, labels, predictions, and non-equivalence probabilities |

The smoke configuration caps the training pool at 400 rows and test sample at 100, for one epoch. It is a pipeline check, not a reported model result. The full configuration trains for three epochs. Neither command reproduces the multi-generator training run by itself.

## Read or rebuild the notebook

[summary_notebook.ipynb](../summary_notebook.ipynb) can be read on GitHub without installing anything. It combines executable generation/training cells with illustrative excerpts; **do not use “Run All” as a CPU quick start**.

To rebuild the cached-result appendix locally, install the training dependencies plus the notebook tools:

```bash
python -m pip install -r requirements.txt jupyterlab matplotlib
jupyter lab summary_notebook.ipynb
```

Run Jupyter from the repository root, then execute **Appendix A** in order. Its cells read the committed JSON/CSV artifacts; the first also refits the TF-IDF baseline on CPU. Earlier cells require vLLM, a compatible CUDA environment, model downloads, or API access and are separate from this route.

## Historical generation and analysis

The scripts in `code/generator_diversity/`, `code/interpretability/`, `code/verify/`, and `seeds/` preserve how the original runs were produced. Several still refer to the original cluster's directories or modules (`overnight/`, `judge_finetune/`, `mech_interp/`) and require adaptation, model checkpoints, and/or activation extracts not included in this repository. They are provenance, not a promised one-command reproduction.

The cross-generator CSVs and small evaluation JSONs are included, so you can inspect the completed outputs without rerunning those jobs. API keys, downloaded models, activation dumps, and local environments stay out of Git.

[Back to the overview](../README.md) · [Validation and artifact map](validation.md)
