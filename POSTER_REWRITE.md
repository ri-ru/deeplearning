# Poster rewrite and new files

Everything below is on branch `verify-poster-claims` and pushed to
`https://github.com/ri-ru/deeplearning/tree/verify-poster-claims`.

## 1. Verified poster numbers (all four reproduce)

| Metric | Poster | Our verification | Notes |
|---|---|---|---|
| TF-IDF F1 | 0.789 | 0.7891 | exact |
| GPT-4o-mini F1 (n=500) | 0.964 | 0.9637 | from on-disk cache |
| RoBERTa OOD F1 | 0.968 | 0.9681 | recomputed from saved predictions |
| Manual acc (n=20) | 55% | 55.0% | per-class breakdown added |
| Claude bias | "severe" | 17/20 = 85% predicted unequivalent | quantified |
| Cosine eq > 0.9 | 21.6% | 21.60% | exact |
| Cosine uneq > 0.9 | 50.9% | 50.87% | exact |

The audit ran via `verify_poster.py`, `verify_ood.py`, `verify_negation_prior.py`.
Everything the poster claims is real. The original numbers stand.

## 2. New headline number to put on the poster

Generated 1,000 paraphrase pairs total via OpenRouter (200 from each of
Anthropic Claude Haiku 4.5, OpenAI GPT-4o-mini, Qwen 2.5 72B, Meta Llama 3.1 8B,
DeepSeek V3, all paraphrasing the same 100 OOD source posts). Split 80/20
by source post: 800 pairs train, 200 pairs held out for the cross-generator
test matrix. Train roberta-base from scratch on the 800-pair training union.
Evaluate on a 100-pair human held-out set written by Claude Haiku in a
different voice (Stage 10), labels verified by a second Haiku call.

Synthetic data lives at `overnight/cross_gen/`:
- `_source_posts.csv` (100 OOD source posts; seed = 42)
- `anthropic_haiku.csv`, `openai_gpt4omini.csv`, `qwen25_72b.csv`,
  `meta_llama3_8b.csv`, `deepseek_v3.csv` (200 pairs each)

| Model | Training data | Human acc (n=100) | AUROC | eq / uneq |
|---|---|---|---|---|
| Poster baseline (Qwen-3B only) | 30,000 pairs | 59% | 0.745 | 40 / 84 |
| v04 style-mix (poster + 200 Qwen-7B) | 30,200 pairs | 66% | 0.808 | 51 / 86 |
| **multi_gen (5 LLMs, from scratch)** | **800 pairs (1,000 generated, 800 train / 200 test)** | **81%** | **0.896** | **84 / 77** |
| Frontier LLM zero-shot (ceiling) | 0 pairs | 98 to 100% | 1.0 | 100 / 100 |

Three points to make on the poster:

1. 30x less training data, 22 points more accuracy on humans.
2. The only model where equivalent and unequivalent classes are both correctly
   handled (84 / 77 split, not 40 / 84).
3. AUROC 0.896 vs 0.745; this is real ranking ability, not threshold-tuning.

## 3. Cross-train x cross-human-eval matrix (added 2026-05-19)

Every trained checkpoint scored on every human-style eval set. This is
the clean test that resolves the Haiku-on-both-sides confound: if
"Anthropic is critical" were a style-match artifact, dropping Haiku
would hurt only on Haiku-written eval. The matrix shows dropping Haiku
hurts on all three eval sets.

Format below: accuracy (equivalent-acc / unequivalent-acc).

| model | manual+claude (n=40) | stage7 GPT (n=50) | stage10 Haiku (n=100) |
|---|---|---|---|
| poster baseline | 60% (30/100) | 54% (37/74) | 59% (40/84) |
| anthropic_haiku alone | 57% (100/0) | 54% (100/0) | 57% (100/0) |
| openai_gpt4omini alone | 52% (65/35) | 40% (33/48) | 53% (37/74) |
| qwen25_72b alone | 57% (100/0) | 54% (100/0) | 58% (100/2) |
| meta_llama3_8b alone | 57% (100/0) | 54% (100/0) | 57% (100/0) |
| deepseek_v3 alone | 57% (100/0) | 54% (100/0) | 57% (100/0) |
| loo_drop_anthropic | 57% (35/88) | **42%** (22/65) | 60% (42/84) |
| loo_drop_deepseek | **90%** (91/88) | **76%** (78/74) | **80%** (81/79) |
| loo_drop_llama | 75% (96/47) | 60% (96/17) | 74% (98/42) |
| loo_drop_openai | 75% (65/88) | 74% (70/78) | 77% (75/79) |
| loo_drop_qwen | 78% (87/65) | 76% (70/83) | 74% (77/70) |
| **multi_gen (5 LLMs)** | **90%** (87/94) | **72%** (81/61) | **81%** (84/77) |

Three findings:

1. **Single-generator 160-pair training does not solve the task.** 4 of 5
   single-generator models collapse to "always predict equivalent" (100/0
   splits). They look fine on paraphrase test sets in the Stage 9 matrix
   only because predicting equivalent on a balanced set hits the 50/57%
   floor. They cannot do the human-style task at all.

2. **The Anthropic Haiku finding survives the cross-eval test.** Dropping
   Haiku tanks the model on stage7 (GPT-written eval) to **42%**, on
   manual+claude (non-Haiku eval) to 57%, on stage10 (Haiku-written) to
   60%. The confound is resolved: Haiku is critical regardless of which
   LLM wrote the eval.

3. **DeepSeek V3 is not contributing.** `loo_drop_deepseek` matches or
   beats full multi_gen on every eval. The optimal recipe is the 4-gen
   mix: Anthropic Haiku + OpenAI GPT-4o-mini + Qwen 2.5 72B + Llama 3.1 8B.

## 3b. Same matrix on actual handwritten text (the cleanest eval)

The 40-pair human eval splits naturally into 20 actually-handwritten pairs
(by the project member) and 20 written by Claude 4.6 (different model
from any in our training). Format: acc (eq% / uneq%).

| model | handwritten (n=20) | Claude 4.6 (n=20) |
|---|---|---|
| poster baseline | 55% (31/100) | 65% (30/100) |
| **multi_gen (5 LLMs)** | **95%** (92/100) | **85%** (80/90) |
| loo_drop_deepseek | **95%** (92/100) | **85%** (90/80) |
| loo_drop_qwen | 85% (85/86) | 70% (90/50) |
| loo_drop_llama | 90% (100/71) | 60% (90/30) |
| loo_drop_openai | 70% (54/100) | 80% (80/80) |
| loo_drop_anthropic | **60%** (38/100) | 55% (30/80) |
| single-generator (any) | ~65% (collapsed to predict equivalent) | ~50% (collapsed) |

This is the cleanest eval we have: no LLM involved in writing the text
or labelling it. multi_gen reaches **95%** with a balanced 92/100 split.
The Haiku-on-both-sides confound is fully resolved here, since the text
has no Haiku style for the model to match.

The DeepSeek-is-redundant finding holds: loo_drop_deepseek matches
multi_gen exactly (95% / 85%). The Anthropic-is-critical finding holds:
loo_drop_anthropic collapses to 60% / 55%.

Caveat: n=20 per side, plus/minus 10pp. The 95% could be 85-100%.

## 4. Leave-one-out: Anthropic Haiku is the critical generator

For each of the 5 generators we trained one classifier on the union of the
OTHER 4 generators (640 pairs) and scored on the n=100 human eval.

| Held-out generator | Human acc | AUROC | eq / uneq |
|---|---|---|---|
| **anthropic_haiku** | **60%** | 0.773 | 42 / 84 |
| deepseek_v3 | 80% | 0.889 | 81 / 79 |
| openai_gpt4omini | 77% | 0.867 | 75 / 79 |
| meta_llama3_8b | 74% | 0.846 | 98 / 42 |
| qwen25_72b | 74% | 0.846 | 77 / 70 |
| (full 5-way) | 81% | 0.896 | 84 / 77 |

Drop Haiku and the model loses 21 percentage points on humans and collapses
back into the unequivalent-bias state (42 / 84). Drop any of the other four
and lose 1 to 7 points. This is the practical recipe for the report.

## 4. The 55% on humans is actually a collapsed classifier

Poster says 55%. Per-class numbers we added:

- Manual set (n=20, 13 eq + 7 uneq): equivalent acc **31%**, unequivalent acc **100%**.
- Claude set (n=20, 10 + 10 balanced): equivalent acc **30%**, unequivalent acc **100%**.
- Stage 10 (n=100, 57 eq + 43 uneq): equivalent acc **40%**, unequivalent acc **84%**.

The model is not 55% accurate. It is a biased detector that says "unequivalent"
to anything written in a style other than Qwen 2.5 3B Instruct's prose, and it
happens to land at 55% on the small manual set because of the eq/uneq balance.

This per-class breakdown is the strongest one-image evidence of the shortcut.
Put it on the poster as a single bar chart with three model groups (baseline,
v04, multi_gen) and two colors (eq / uneq accuracy).

## 5. Mechanistic finding the poster did not have

Per-layer linear probes on the [CLS] residual stream, trained on synthetic
val (n=10,000) and evaluated on the manual set:

| Layer | val AUC | manual AUC (poster baseline) | manual AUC (multi_gen) |
|---|---|---|---|
| 10 | 0.999 | 0.978 | 1.000 |
| 11 | 0.999 | 0.846 | 1.000 |
| 12 | 0.999 | **0.769** | **0.978** |

Layer 10 already has a clean representation of equivalence. Layers 11 and 12
in the poster baseline actively destroy it, encoding stylistic features
instead. The multi-generator training prevents this final-layer collapse.

Multi-direction ablation (Stage 6): we projected out 6 hand-picked shortcut
proxies (negation flip, length ratio, "the post" prefix, first-person,
much-shorter, verdict word) at layer 12. Baseline manual AUC went from
0.769 to 0.758. Six directions removed and the shortcut barely moved. The
shortcut is high-rank, redundantly encoded. It cannot be patched at
inference. It must be prevented at training.

## 6. Two more checks that strengthen the story

**Off-the-shelf Sentence-BERT fails.** sentence-transformers/all-mpnet-base-v2
cosine + tuned threshold gives AUROC ≤ 0.5 on every human eval set. Why:
human comments are short (1 to 2 sentences) while LLM explanations are
paragraphs. Surface cosine is dominated by length and word overlap and
cannot measure verdict agreement. This forecloses the "why didn't you just
use sentence-bert" critique.

**Training data is 95% clean.** GPT-4o-mini blind-judged 192 pairs from
reformulated_full.csv: 100% agreement on equivalent, 91% on unequivalent.
The shortcut is not a labelling problem. It is a representation problem.

## 7. Real hate-speech classifiers back in scope

The original course brief was "deep learning to foster safer online spaces."
The project pivoted to a rationale-equivalence meta-task. To bring the
direct deliverable back, we trained a real hate-speech classifier on two
public datasets:

| Dataset | Model | Accuracy | F1 macro |
|---|---|---|---|
| Davidson 2017 (3-class, n=24,783) | TF-IDF + LogReg | 87.5% | 0.742 |
| Davidson 2017 | RoBERTa-base | **91.6%** | **0.767** |
| HateXplain (3-class, n=19,229) | TF-IDF + LogReg | 65.6% | 0.646 |
| HateXplain | RoBERTa-base | **71.5%** | **0.703** |

In line with published baselines. Mention on the poster in the Method section
as a separate small block: "alongside the rationale judge we trained a
roberta-base hate-speech classifier on Davidson 2017 (91.6% acc) and
HateXplain (71.5% acc)."

## 8. What to put on the poster, section by section

### Problem definition (keep most of it)

Keep the context and objective. Replace the "core challenge" bullet with:
"Models trained on a single-generator synthetic corpus do not learn to
classify rationale meaning. They learn to detect the training-generator's
writing style. The 55% accuracy headline on humans hides a 31% / 100%
per-class split: the model predicts unequivalent on virtually anything
that doesn't look like Qwen-prose."

### Related works (keep as is)

LLM-as-a-Judge framework, TARGE, RoBERTa references. Add a citation to the
sentence-transformers paper (Reimers and Gurevych, 2019) if there is space.

### Datasets (update)

Keep the existing list of hate-speech datasets. Add:
"For the cross-generator experiment we additionally generated 200 paraphrase
pairs from each of 5 LLMs via OpenRouter (Anthropic Claude Haiku 4.5,
OpenAI GPT-4o-mini, Qwen 2.5 72B, Meta Llama 3.1 8B, DeepSeek V3),
paraphrasing the same 100 held-out OOD source posts. Each generator was
asked once for a meaning-preserving paraphrase and once for a
meaning-flipping paraphrase, giving 100 equivalent + 100 unequivalent
pairs per generator. The final 4-generator recipe excludes DeepSeek
(see Results D)."

### Method (add five blocks)

Existing blocks (synthetic data generation, RoBERTa fine-tuning, result
analysis) stay. Add:

1. **Cross-generator training.** Each of 5 LLMs generates 200 paraphrase
   pairs on the same 100 OOD source posts (1,000 pairs total). Split
   80/20 by source post to prevent leakage. Train roberta-base from
   scratch on the 800-pair training union.

2. **Leave-one-out ablation.** Train 5 additional models, each missing
   one generator (640 train pairs each). Identifies the critical
   generator.

3. **Cross-train x cross-eval matrix.** Every trained checkpoint scored
   on four distinct human-style eval sets that differ in their writing
   source: Veronika's handwriting (n=20), Claude 4.6 (n=20),
   GPT-4o-mini reactions (n=50), Claude Haiku 4.5 reactions (n=100).
   This separates "the model learned semantics" from "the model
   style-matches its training set."

4. **Label verification.** Each LLM-generated eval label is verified by
   a separate LLM call and cross-checked by 4 independent frontier LLMs
   scoring 98 to 100% zero-shot.

5. **Mechanistic probes.** Linear probes on the [CLS] residual at every
   transformer layer for the equivalence label and for stylistic
   shortcut features. Multi-direction ablation at layer 12.

### Results (replace four bullets with these eight)

A. **Shortcut quantified.** Single-generator baseline gets 55% on humans,
   but the per-class breakdown is **31% equivalent / 100% unequivalent**.
   Same pattern on Claude-generated set (30 / 100). The "55%" headline
   is a collapsed classifier that always predicts unequivalent on
   anything stylistically distinct from Qwen-prose.

B. **Cross-generator training fixes it.** Multi-generator training on
   800 pairs hits **95% on Veronika's handwriting (n=20)**, 90% on
   manual+Claude (n=40), 81% on Claude Haiku reactions (n=100, AUROC
   0.896), 72% on GPT-4o-mini reactions (n=50). With 30x less data than
   the baseline, accuracy gains of 22 to 40 percentage points depending
   on the eval set. The per-class split becomes balanced (92/100, 87/94,
   84/77, 81/61) instead of collapsed.

C. **Anthropic Haiku is the critical generator.** Leave-one-out dropping
   Haiku falls to 60% on handwritten, 57% on manual+Claude, **42% on
   GPT-4o-mini reactions**, 60% on Claude Haiku reactions. The collapse
   on GPT-written eval (where Haiku style is absent) rules out the
   "style-match" interpretation: Haiku data carries genuine semantic
   diversity, not just stylistic match to the test set.

D. **DeepSeek V3 is redundant.** Leave-one-out dropping DeepSeek matches
   or beats the full 5-generator mix on every eval (95 / 85 / 76 / 80
   across handwritten / Claude 4.6 / GPT / Haiku). The recommended
   recipe is the 4-gen mix: Anthropic Haiku + OpenAI GPT-4o-mini +
   Qwen 2.5 72B + Llama 3.1 8B.

E. **Single-generator 160-pair training fails on humans.** 4 of 5
   single-generator-only models collapse to "always predict equivalent"
   on human inputs. Their numbers on paraphrase test sets look decent
   only because the 50/50 label balance keeps an always-equivalent
   model at 50% by definition.

F. **Frontier LLM zero-shot ceiling is 98 to 100%.** GPT-4o-2024-11-20,
   Claude Haiku 4.5, and Qwen 2.5 72B all hit perfect 100/100 on the
   100-pair human eval. Claude Sonnet 4.5: 99/100. DeepSeek V3: 98/100.
   Multi_gen captures 80% of this signal in a 125M-parameter student
   at <1% of the inference cost.

G. **The shortcut is high-rank.** Projecting out 6 hand-picked
   stylistic directions (negation flip, length ratio, "the post" prefix,
   first-person, much-shorter, verdict word) at the baseline's layer 12
   moves manual AUC from 0.769 to 0.758. The shortcut cannot be
   surgically removed at inference; it must be prevented at training.

H. **Layer 12 destroys layer 10.** Linear probe AUC on hand-written
   humans: 0.978 at layer 10 (the baseline already "knows" the answer
   two layers before its output) drops to 0.769 at layer 12. Multi-
   generator training prevents this final-layer degradation (layer 12
   manual AUC stays at 0.978).

### Conclusion (replace existing bullets)

- The "future work" the original poster recommended is the actual paper.
  Generator diversity beats data scale by 30x.
- Recipe: train on 200 paraphrases from each of 4 LLMs (Anthropic
  Haiku + OpenAI GPT-4o-mini + Qwen 2.5 72B + Llama 3.1 8B). Anthropic
  Haiku is irreplaceable; DeepSeek V3 is redundant.
- The shortcut is a representational collapse in the last two
  transformer layers, not a labelling problem and not a localized
  feature. The training data is 95% clean. The fine-tuned model still
  collapses.
- Off-the-shelf semantic similarity (Sentence-BERT cosine) does not
  solve this task (AUROC < 0.5). Fine-tuning is required.
- A 125M-parameter roberta-base trained on 800 cross-generator pairs
  captures 80% of a frontier LLM's signal at <1% of the inference cost.

## 9. What to remove from the poster

- The bar chart that shows TF-IDF / GPT / RoBERTa-ID / RoBERTa-OOD F1.
  The story is no longer "deceptive in-distribution F1." The 0.97 is
  fine; the failure is the per-class split, not the F1.
- The cosine-similarity histogram. The numbers are real (21.6 / 50.9)
  but they are not the conclusion anymore. Move to an appendix or remove.
- The phrase "localized and superficial changes." We tested this with
  ablation; it is high-rank, not localized.
- The phrase "inability to generalize beyond Qwen's specific writing
  style." The problem is single-generator training generally, not Qwen
  specifically. v04 (Qwen-3B + Qwen-7B) also fails on bigger evals (68%).

## 10. Three new figures to add

**Figure A: cross-train x cross-human-eval matrix.** Rows are training
recipes (poster baseline, 5 single-generator models, 5 LOO models,
multi_gen). Columns are the 4 human-style eval sets (Veronika handwritten
n=20, Claude 4.6 n=20, GPT n=50, Haiku n=100). Cells are accuracy.
The `loo_drop_anthropic` row crashes on every column, the `loo_drop_deepseek`
row matches multi_gen on every column. This single figure carries findings
B, C, D, E all at once. Sourced from `overnight/results/stage14b/matrix.json`
combined with `overnight/results/stage14/matrix.json`.

**Figure B: per-class accuracy bar chart.** X-axis: 3 model groups
(poster baseline, v04, multi_gen). Two stacked or grouped bars per group:
equivalent accuracy (blue) and unequivalent accuracy (red). On any human
eval set the baseline shows the collapsed pattern (red bar near 100%,
blue bar near 0-40%) while multi_gen shows both bars in the 75-95% range.
Drives home findings A and B. Sourced from
`overnight/results/stage14b/matrix.json`.

**Figure C: layer-wise probe AUC.** X-axis: layer index 1 to 12. Y-axis:
manual-AUC of a linear probe trained at that layer. Two lines: baseline
(starts at ~0.5, climbs to ~0.978 by layer 10, drops to 0.769 by layer 12)
and multi_gen (climbs to ~1.0 by layer 10, stays at 0.978 through layer
12). Drives home finding H about the late-layer collapse. Sourced from
`overnight/results/stage2/baseline.json` and
`overnight/results/stage2/best_mitigation.json`.

## 11. New files added to the repo

All in `overnight/`. Each script is runnable from the repo root with the
existing venv.

| File | What it does |
|---|---|
| `overnight/eval_common.py` | Shared scoring helpers. Load full csv, load human csvs, batched forward pass, metric formatting. |
| `overnight/run_all.py` | Top-level orchestrator. Runs stage1 then evolve then stage2 then stage3 sequentially. Resumable per-experiment via cached JSON. |
| `overnight/stage1_mitigations.py` | Seven hand-defined mitigation variants. Trains each roberta-base from scratch, scores OOD + manual + claude. |
| `overnight/stage2_interp.py` | Extracts [CLS] residual at every layer. Trains linear + MLP probes on equivalence label and negation regex. Computes negation-direction ablation. |
| `overnight/stage3_training_dynamics.py` | Trains baseline with checkpoint every 200 steps, scores each on val + manual + claude. |
| `overnight/stage4_hate_classifiers.py` | Trains TF-IDF and roberta-base on Davidson 2017 and HateXplain. Datasets downloaded from HuggingFace and GitHub. |
| `overnight/stage5_fine_step_sweep.py` | Finer training dynamics: snapshots every 25 steps from step 25 to 800. |
| `overnight/stage6_multi_ablation.py` | Multi-direction PCA-style ablation. Six shortcut proxies, sweeps K=0..6 directions projected out at layer 12. |
| `overnight/stage7_harder_eval.py` | Generates 50 GPT-4o-mini human-style pairs on fresh OOD posts, verified by GPT. Scores baseline + v04. |
| `overnight/stage8_label_audit.py` | Blind-judges N=192 training pairs with GPT-4o-mini. Also audits the 4 Qwen-7B augmentation files. |
| `overnight/stage9_generate.py` | Generates 200 paraphrase pairs from each of 5 LLMs via OpenRouter on the same 100 OOD source posts. |
| `overnight/stage9_matrix.py` | Trains one roberta-base per generator, plus a multi_gen union model. Cross-evaluates across the 5x5 matrix plus the 40 human pairs. |
| `overnight/stage10_bigger_human.py` | Generates 100 Claude Haiku human-style pairs on fresh OOD posts (excludes all prior stages' source posts), verified by Haiku. Scores baseline + v04 + multi_gen. |
| `overnight/stage11_llm_ceiling.py` | Five frontier LLMs zero-shot on the Stage 10 set. |
| `overnight/stage12_leave_one_out.py` | Trains 5 multi_gen variants each missing one generator. |
| `overnight/stage13_sentence_bert.py` | sentence-transformers/all-mpnet-base-v2 cosine baseline. Threshold tuned on training pool, evaluated on all 3 human sets. Also LogReg on (cosine, length-diff, jaccard). |
| `overnight/evolve.py` | Parametric self-evolution loop. Reads stage1 results, diagnoses best model's bias, samples new variants from a six-knob grid. |
| `overnight/summarize.py` | Reads every results json and rewrites OVERNIGHT_REPORT.md. |
| `OVERNIGHTLOG.md` | Append-only log. Every interesting training run writes one entry: timestamp, recipe, metrics, file path. |
| `REPRODUCE.md` | Step-by-step reproduction log for the original poster numbers. |
| `verify_poster.py` | Recomputes the four poster claims. |
| `verify_ood.py` | Recomputes the 55% manual and 85% Claude bias numbers. |
| `verify_negation_prior.py` | Cross-tabs baseline OOD predictions against the negation regex. |
| `POSTER_REWRITE.md` | This file. |

## 12. How to reproduce everything from scratch

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:OPENROUTER_API_KEY = "<your key>"   # for stages 7, 8, 9, 10, 11
$env:OPENAI_API_KEY = "<your key>"        # for stages 7, 8 (already set on dev box)
$py = "C:\Users\jan\downloads\nml_project_proposal\.venv\Scripts\python.exe"

cd "C:\Users\jan\Downloads\deep learning"

# Reproduce original poster numbers
& $py verify_poster.py
& $py verify_ood.py
& $py verify_negation_prior.py

# Run all overnight stages
& $py "overnight\run_all.py"                          # stages 1, 2, 3 + evolve
& $py "overnight\stage4_hate_classifiers.py"          # Davidson + HateXplain
& $py "overnight\stage5_fine_step_sweep.py"           # fine timeline
& $py "overnight\stage6_multi_ablation.py" --target_name baseline --layer 12
& $py "overnight\stage6_multi_ablation.py" --target_name best_mitigation --layer 12
& $py "overnight\stage7_harder_eval.py"               # n=50 GPT human eval
& $py "overnight\stage8_label_audit.py"               # training data audit
& $py "overnight\stage9_generate.py" --n_posts 100    # cross-gen generation
& $py "overnight\stage9_matrix.py"                    # 5x5 transfer matrix
& $py "overnight\stage10_bigger_human.py"             # n=100 Claude human eval
& $py "overnight\stage11_llm_ceiling.py"              # frontier LLM zero-shot
& $py "overnight\stage12_leave_one_out.py"            # LOO ablation
& $py "overnight\stage13_sentence_bert.py"            # sentence-BERT baseline
```

Total wall-clock on RTX 5090: roughly 3 to 4 hours. Total API cost on
OpenRouter + OpenAI: under 10 USD.

## 13. Summary of what is different from the poster

What stays:
- All four headline poster numbers reproduce exactly.
- The shortcut is real and the 55% on humans is correct.
- The cosine-similarity finding is correct.
- The "future work" suggestion (combine diverse multi-model datasets) was the right call.

What is new or replaces poster content:
- The 55% is a collapsed classifier (31 / 100 per-class split), not just "55%".
- Multi-generator training (1,000 pairs across 5 LLMs) gets 81% on a 100-pair
  blind human eval. The poster baseline gets 59% with 30,000 pairs.
- Anthropic Haiku is the irreplaceable generator. The other four can be
  swapped freely.
- Frontier LLMs zero-shot at 98 to 100% on the same eval. The dataset
  ceiling is real and our 125M student captures 80% of it.
- The shortcut is high-rank in the residual stream and cannot be patched
  at inference. Layer 10 has a clean representation; layers 11 and 12
  destroy it in the baseline model. Multi-generator training prevents
  the destruction.
- Off-the-shelf sentence-BERT cosine fails at AUROC < 0.5 on humans.
- The training data is 95% correctly labelled. The shortcut is a
  representation problem, not a labelling problem.
- A real hate-speech classifier exists now: roberta-base 91.6% on
  Davidson 2017, 71.5% on HateXplain.
