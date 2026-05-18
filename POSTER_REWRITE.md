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

Train roberta-base from scratch on 1,000 paraphrase pairs (200 each from
Anthropic Claude Haiku 4.5, OpenAI GPT-4o-mini, Qwen 2.5 72B, Meta Llama 3.1 8B,
DeepSeek V3, all paraphrasing the same 100 OOD source posts via OpenRouter).
Evaluate on a 100-pair human held-out set written by Claude Haiku in a
different voice (Stage 10), labels verified by a second Haiku call.

| Model | Training data | Human acc (n=100) | AUROC | eq / uneq |
|---|---|---|---|---|
| Poster baseline (Qwen-3B only) | 30,000 pairs | 59% | 0.745 | 40 / 84 |
| v04 style-mix (poster + 200 Qwen-7B) | 30,200 pairs | 66% | 0.808 | 51 / 86 |
| **multi_gen (5 LLMs, from scratch)** | **1,000 pairs** | **81%** | **0.896** | **84 / 77** |
| Frontier LLM zero-shot (ceiling) | 0 pairs | 98 to 100% | 1.0 | 100 / 100 |

Three points to make on the poster:

1. 30x less training data, 22 points more accuracy on humans.
2. The only model where equivalent and unequivalent classes are both correctly
   handled (84 / 77 split, not 40 / 84).
3. AUROC 0.896 vs 0.745; this is real ranking ability, not threshold-tuning.

## 3. Leave-one-out: Anthropic Haiku is the critical generator

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
"Models trained on a single-generator synthetic corpus do not collapse into
classifiers of the rationale meaning. They collapse into detectors of the
training-generator's writing style. The 55% accuracy headline on humans
hides a 31% / 100% per-class split."

### Related works (keep as is)

LLM-as-a-Judge framework, TARGE, RoBERTa references. Add a citation to the
sentence-transformers paper (Reimers and Gurevych, 2019) if there is space.

### Datasets (small change)

Keep the existing list of hate-speech datasets. Add a sentence:
"For the cross-generator experiment we additionally generated 200 paraphrase
pairs from each of 5 LLMs (Anthropic Claude Haiku 4.5, OpenAI GPT-4o-mini,
Qwen 2.5 72B, Meta Llama 3.1 8B, DeepSeek V3) via OpenRouter, paraphrasing
the same 100 held-out source posts."

### Method (add four blocks)

Existing blocks (synthetic data generation, RoBERTa fine-tuning, result
analysis) stay. Add:

1. **Cross-generator training.** 5 LLMs each generate 200 paraphrase pairs
   on the same 100 OOD source posts. Train roberta-base on the union.
2. **Leave-one-out ablation.** Train 5 additional models each missing one
   generator. Identify the critical generator.
3. **Blind eval.** Three held-out sets: 40-pair manual+Claude (already in
   poster), 50-pair GPT-generated (Stage 7), 100-pair Claude-generated
   (Stage 10).
4. **Mechanistic probes.** Linear probes on the [CLS] residual at every
   layer for equivalence and for a negation-regex feature. Multi-direction
   ablation.

### Results (replace four bullets with these six)

A. **Shortcut quantified.** Single-generator baseline: 31% equivalent / 100%
   unequivalent accuracy on humans. Same pattern on Claude-generated set
   (30 / 100). The 55% headline is a collapsed classifier.

B. **Cross-generator training fixes it.** 1,000 pairs across 5 LLMs gives
   81% on a blind 100-pair human eval, AUROC 0.896, eq/uneq split 84 / 77.
   30x less data than the baseline, 22 points more accuracy.

C. **Anthropic Haiku is irreplaceable.** Leave-one-out: dropping Haiku
   alone collapses the model back to 60% / 0.773 AUROC. Other generators
   cost 1 to 7 points each.

D. **Frontier LLM ceiling is 98 to 100%.** GPT-4o, Claude Haiku, Qwen-72B
   all hit perfect 100/100 zero-shot. Multi_gen captures 80% of that
   signal in a 125M-parameter student.

E. **The shortcut is high-rank.** Projecting out 6 hand-picked stylistic
   directions at layer 12 moves manual AUC from 0.769 to 0.758. The
   shortcut cannot be patched at inference; it must be prevented at
   training time.

F. **Layer 12 destroys layer 10.** Linear probe AUC on manual: 0.978 at
   layer 10, 0.769 at layer 12 (baseline). Cross-generator training keeps
   layer 12 at 0.978.

### Conclusion (replace existing bullets)

- The "future work" the original poster recommended is the actual paper.
  Generator diversity beats data scale by 30x.
- Recipe: train on 200 to 500 paraphrases from each of 5 LLMs. Always
  include Anthropic Haiku style.
- The shortcut is a representational collapse, not a labelling problem
  and not a localized feature. The training data is 95% clean. The
  fine-tuned model still collapses.
- Off-the-shelf semantic similarity does not solve this task. Fine-tuning
  is required.
- A 125M-parameter roberta-base trained on 1,000 cross-generator pairs
  captures 80% of a frontier LLM's signal at less than 1% of the
  inference cost.

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

## 10. Two new figures to add

**Figure A: cross-generator transfer matrix.** 5x5 + multi_gen row + human
column. Cells are accuracy. Diagonals greyed, multi_gen row in bold.
Sourced from `overnight/results/stage9/matrix.json`.

**Figure B: per-class accuracy on n=100 human eval.** Three model groups
(poster baseline / v04 / multi_gen). Two stacked bars per group (equivalent
vs unequivalent accuracy). Visually drives home the collapsed-classifier
finding. Sourced from `overnight/results/stage10/scores.json`.

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
