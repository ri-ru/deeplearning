# Validation and artifact map

SameSense was built as an end-to-end semantic-equivalence pipeline, then checked against unfamiliar domains, writing styles, and simple baselines. This page separates what was implemented from what the recorded scores establish.

## The model comparison

The README and `scripts/project_summary.py` use these recorded accuracy values:

| Set | n | Original judge | Multi-generator judge | Source |
|---|---:|---:|---:|---|
| Human-written | 20 | 55% | 95% | [stage14b_matrix.json](../results/overnight/stage14b_matrix.json) |
| Claude-written | 20 | 65% | 85% | [stage14b_matrix.json](../results/overnight/stage14b_matrix.json) |
| GPT-4o-mini | 50 | 54% | 72% | [stage14_matrix.json](../results/overnight/stage14_matrix.json) |
| Claude Haiku | 100 | 59% | 81% | [stage14_matrix.json](../results/overnight/stage14_matrix.json) |

The compared recipes are `poster_baseline` and `multi_gen`. The JSON files also contain F1, AUROC, class-specific accuracy, and leave-one-generator-out runs.

## Why the initial score was not enough

The original judge recorded **0.968 macro-F1 on 17,691 held-out-domain synthetic pairs**, while the TF-IDF baseline recorded **0.789** on the same-sized test set. Those results are in [verify_poster_results.json](../results/verify_poster_results.json).

Held-out *domains* are not the same as held-out *generators*. The human-written and Claude-written checks exposed a different weakness: the original judge achieved only 55% and 65% accuracy, respectively. Layer probes, negation checks, and directional ablations were used to investigate stylistic shortcuts. Training on a varied generator pool improved the recorded accuracy on all four sets above; this does not prove that every shortcut was removed.

The five usable generator files contain **998 pairs**: 200 each from Claude Haiku, DeepSeek, Llama, and GPT-4o-mini, and 198 from Qwen. The `mistral_7b.csv` file has no usable rows and is excluded. This inventory is computed from the CSVs, not rounded to an advertised 1,000.

## Evaluation boundaries

- The 20-example human/Claude checks are small. A one-example change moves accuracy by five percentage points, and they do not support a broad production-accuracy claim.
- These are historical recorded runs, not independently replicated results or a fresh validation. The datasets and protocols differ, so the large synthetic score and small writing-style scores are not interchangeable.
- Five generators contribute to the varied training pool. A score on text from a familiar generator does not establish performance on a genuinely new generator.
- Equivalence labels simplify subjective moderation reasoning into a binary decision. This system compares rationales; it is not itself a hate-speech detector or an automated moderation service.
- Trained weights, full activation extracts, and some original evaluation inputs are not bundled. Cached results can be inspected; full inference-level reproduction requires those artifacts or a new training run.

## Source artifacts

| Artifact | What it records | Producer |
|---|---|---|
| [verify_poster_results.json](../results/verify_poster_results.json) | TF-IDF, RoBERTa, an LLM baseline on a 500-pair subset, and lexical similarity | `code/verify/verify_poster.py` |
| [verify_ood_results.json](../results/verify_ood_results.json) | Human/Claude style-shift checks and confusion matrices | `code/verify/verify_ood.py` |
| [verify_negation_prior_results.json](../results/verify_negation_prior_results.json) | Label predictability from negation patterns | `code/verify/verify_negation_prior.py` |
| [probe_metrics.json](../results/probe_metrics.json) | Per-layer equivalence and shortcut probes | `code/interpretability/` |
| [stage6_baseline_layer12.json](../results/overnight/stage6_baseline_layer12.json) | Layer-12 directional-ablation sweep | `code/generator_diversity/stage6_multi_ablation.py` |
| [stage14_matrix.json](../results/overnight/stage14_matrix.json) · [stage14b_matrix.json](../results/overnight/stage14b_matrix.json) | Multi-generator and leave-one-out comparisons | `code/generator_diversity/stage14*.py` |
| [cross_gen/](../results/cross_gen/) | Generated pair texts, labels, and metadata | `code/generator_diversity/stage9_generate.py` |
| [seed_variance_summary.json](../seeds/results/seed_variance_summary.json) | Aggregate repeated-seed outputs | `seeds/` |

See [the notebook](../summary_notebook.ipynb) for the implementation narrative and [the running guide](usage.md) for executable entry points.
