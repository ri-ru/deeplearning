# SameSense

A 125M-parameter RoBERTa judge for semantic equivalence between moderation explanations, with cross-generator evaluation, linear probes, and directional ablations.

[![Repository checks](https://github.com/ri-ru/samesense/actions/workflows/checks.yml/badge.svg)](https://github.com/ri-ru/samesense/actions/workflows/checks.yml)

[Notebook](summary_notebook.ipynb) · [Setup](guides/usage.md) · [Evaluation details](guides/validation.md)

## Summary

We built the data generation, training, and evaluation pipeline for a compact sentence-pair classifier that can replace an LLM call for this specific judging task. It predicts whether two explanations are **equivalent** or **non-equivalent**; it does not classify the original post as hate speech.

- Generated 52,137 labelled pairs and fine-tuned RoBERTa with held-out-domain evaluation.
- Compared against TF-IDF and LLM baselines, then used layer probes and ablations to investigate failures on unfamiliar writing styles.
- Trained a second version on 998 pairs from five generators, improving accuracy on the evaluation sets below.

The central finding was a generalization failure: **0.968 macro-F1 on 17,691 held-out-domain synthetic pairs** (TF-IDF: 0.789) concealed sensitivity to writing style. Probes and directional ablations investigated the shortcuts; training on more varied generators improved the recorded style-shift results.

## Run the summary

```bash
git clone https://github.com/ri-ru/samesense.git
cd samesense
python3 scripts/project_summary.py
```

Requires Python 3.10+. Reads the saved data and metrics without a GPU or API key; it does not run inference. Add `--json` for structured output. Training instructions are in [the setup guide](guides/usage.md).

## Results

Accuracy of the original single-generator judge and the later multi-generator version:

| Evaluation set | Examples | Single-generator | Multi-generator |
|---|---:|---:|---:|
| Human-written rationales | 20 | 55% | **95%** |
| Claude-written rationales | 20 | 65% | **85%** |
| GPT-4o-mini rationales | 50 | 54% | **72%** |
| Claude Haiku rationales | 100 | 59% | **81%** |

The human/Claude sets are small; these scores are not a production-accuracy claim. See [evaluation details](guides/validation.md) for the source artifacts and limitations.

## Code and data

| Location | What you will find |
|---|---|
| [summary_notebook.ipynb](summary_notebook.ipynb) | Implementation narrative, code excerpts, tables, and figures |
| [code/judge_training/](code/judge_training/) | Data loading, domain splits, tokenization, metrics, and fine-tuning |
| [code/generator_diversity/](code/generator_diversity/) | Multi-model pair generation and comparison scripts |
| [code/interpretability/](code/interpretability/) | Activation extraction, layer-wise probes, and shortcut analysis |
| [code/verify/](code/verify/) | Baseline, domain-shift, and negation checks |
| [configs/](configs/) | Full and smoke-test training configurations |
| [data/](data/) · [results/](results/) · [seeds/](seeds/) | Source data, cached outputs, and repeated-seed checks |

Model weights are not included. Some historical analysis scripts require the original cluster layout; see the setup guide before rerunning them.

---

Built by Veronika Wannack with teammates Flavia Wallenhorst and Dimitri Pelgrin.
