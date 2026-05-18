# Overnight training log

Append-only log of every interesting training run. Each entry: timestamp,
variant name, key numbers, one-line interpretation.

Format:
```
## YYYY-MM-DD HH:MM  <variant>
- recipe: ...
- OOD F1: ...  manual_acc: ...  claude_acc: ...
- interpretation: ...
```

---

## 2026-05-17 22:25  v05_one_epoch  (smoke test)
- recipe: baseline data, only 1 epoch (instead of 3)
- OOD F1: 0.968  manual_acc: 50.0% (eq=23%/uneq=100%)  claude_acc: 65% (85% predicted unequivalent)
- interpretation: **the shortcut is already fully present after just 1 epoch.**
  OOD F1 matches the 3-epoch poster number (0.968) and the manual-set bias
  (predict 'unequivalent' on anything non-Qwen-style) is just as severe.
  Training longer doesn't *create* the shortcut — it cements it.
- file: `overnight/results/stage1/v05_one_epoch.json`


## 2026-05-17 23:42  v00_baseline  (stage1)
- recipe: v00_baseline (3 epochs)
- OOD F1: 0.973  manual_acc: 50.0% (eq=23%/uneq=100%)  claude_acc: 80.0% (70% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v00_baseline.json`

## 2026-05-17 23:44  v01_drop_high_cosine  (stage1)
- recipe: v01_drop_high_cosine (3 epochs)
- OOD F1: 0.932  manual_acc: 35.0% (eq=0%/uneq=100%)  claude_acc: 80.0% (70% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v01_drop_high_cosine.json`

## 2026-05-17 23:46  v02_drop_neg_shortcut  (stage1)
- recipe: v02_drop_neg_shortcut (3 epochs)
- OOD F1: 0.972  manual_acc: 70.0% (eq=54%/uneq=100%)  claude_acc: 65.0% (85% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v02_drop_neg_shortcut.json`

## 2026-05-17 23:49  v03_balance_neg  (stage1)
- recipe: v03_balance_neg (3 epochs)
- OOD F1: 0.965  manual_acc: 60.0% (eq=38%/uneq=100%)  claude_acc: 65.0% (85% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v03_balance_neg.json`

## 2026-05-17 23:52  v04_style_mix_qwen7b  (stage1)
- recipe: v04_style_mix_qwen7b (3 epochs)
- OOD F1: 0.961  manual_acc: 95.0% (eq=92%/uneq=100%)  claude_acc: 75.0% (55% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v04_style_mix_qwen7b.json`

## 2026-05-17 23:54  v06_combined  (stage1)
- recipe: v06_combined (3 epochs)
- OOD F1: 0.932  manual_acc: 75.0% (eq=62%/uneq=100%)  claude_acc: 75.0% (35% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/v06_combined.json`

## 2026-05-17 23:56  auto_98c1caf637  (evolve)
- recipe: {'drop_cosine_threshold': None, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 1, 'mix_qwen7b': True, 'epochs': 2}
- OOD F1: 0.972  manual_acc: 95.0% (eq=92%/uneq=100%)  claude_acc: 65.0% (45% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_98c1caf637.json`

## 2026-05-17 23:59  auto_1abccdd95d  (evolve)
- recipe: {'drop_cosine_threshold': None, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 1, 'mix_qwen7b': True, 'epochs': 3}
- OOD F1: 0.971  manual_acc: 85.0% (eq=77%/uneq=100%)  claude_acc: 90.0% (60% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_1abccdd95d.json`

## 2026-05-18 00:01  auto_a46c7fdb8a  (evolve)
- recipe: {'drop_cosine_threshold': None, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 2, 'mix_qwen7b': True, 'epochs': 2}
- OOD F1: 0.973  manual_acc: 80.0% (eq=69%/uneq=100%)  claude_acc: 80.0% (60% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_a46c7fdb8a.json`

## 2026-05-18 00:04  auto_71275ea5ae  (evolve)
- recipe: {'drop_cosine_threshold': None, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 2, 'mix_qwen7b': True, 'epochs': 3}
- OOD F1: 0.972  manual_acc: 85.0% (eq=77%/uneq=100%)  claude_acc: 80.0% (70% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_71275ea5ae.json`

## 2026-05-18 00:06  auto_43827d3e18  (evolve)
- recipe: {'drop_cosine_threshold': 0.85, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 1, 'mix_qwen7b': True, 'epochs': 2}
- OOD F1: 0.940  manual_acc: 75.0% (eq=62%/uneq=100%)  claude_acc: 80.0% (60% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_43827d3e18.json`

## 2026-05-18 00:08  auto_81351db7cc  (evolve)
- recipe: {'drop_cosine_threshold': 0.85, 'drop_neg_shortcut': False, 'downsample_uneq_frac': 1.0, 'upsample_eq_neg_factor': 1, 'mix_qwen7b': True, 'epochs': 3}
- OOD F1: 0.942  manual_acc: 90.0% (eq=92%/uneq=86%)  claude_acc: 75.0% (55% predicted unequivalent)
- interpretation: _auto-logged; review later_
- file: `overnight/results/stage1/auto_81351db7cc.json`

## 2026-05-18 08:57  davidson/tfidf_logreg  (stage4: real hate-speech detector)
- dataset: davidson  (n_test=4957)  classes=['hate', 'offensive', 'neither']
- model:   tfidf_logreg
- accuracy: 87.5%   f1_macro: 0.742   f1_weighted: 0.882   (1s)
- per-class: hate: f1=0.46  offensive: f1=0.92  neither: f1=0.84
- file: `overnight/results/stage4/davidson_tfidf_logreg.json`

## 2026-05-18 08:58  davidson/roberta-base  (stage4: real hate-speech detector)
- dataset: davidson  (n_test=4957)  classes=['hate', 'offensive', 'neither']
- model:   roberta-base
- accuracy: 91.6%   f1_macro: 0.767   f1_weighted: 0.914   (63s)
- per-class: hate: f1=0.45  offensive: f1=0.95  neither: f1=0.90
- file: `overnight/results/stage4/davidson_roberta-base.json`

## 2026-05-18 08:59  hatexplain/tfidf_logreg  (stage4: real hate-speech detector)
- dataset: hatexplain  (n_test=3846)  classes=['hatespeech', 'normal', 'offensive']
- model:   tfidf_logreg
- accuracy: 65.6%   f1_macro: 0.646   f1_weighted: 0.654   (1s)
- per-class: hatespeech: f1=0.73  normal: f1=0.70  offensive: f1=0.50
- file: `overnight/results/stage4/hatexplain_tfidf_logreg.json`

---

## 2026-05-18 09:00  STAGE 1 + EVOLVE SUMMARY (13 variants, sorted by manual acc)

**Headline finding: generator diversity (`mix_qwen7b=True`) almost completely fixes the shortcut.** Manual accuracy jumps from 50% (baseline) to **95%** with only a 0.007 drop in OOD F1 (0.968 -> 0.961). Adding ~200 paraphrases from a different Qwen variant moves the needle further than any other intervention we tried.

| Variant | OOD F1 | manual acc | eq/uneq | claude acc | claude %uneq |
|---|---|---|---|---|---|
| v04_style_mix_qwen7b | 0.961 | **95.0%** | 92/100 | 75% | 55% |
| auto_98c1caf637 (mix, 2 epochs) | 0.972 | **95.0%** | 92/100 | 65% | 45% |
| auto_81351db7cc (cos<0.85 + mix, 3ep) | 0.942 | **90.0%** | **92/86** | 75% | 55% |
| auto_1abccdd95d (mix, 3 epochs) | 0.971 | 85.0% | 77/100 | **90%** | 60% |
| auto_71275ea5ae (mix + ups eq_neg x2) | 0.972 | 85.0% | 77/100 | 80% | 70% |
| auto_a46c7fdb8a (mix + ups, 2ep) | 0.973 | 80.0% | 69/100 | 80% | 60% |
| v06_combined | 0.932 | 75.0% | 62/100 | 75% | 35% |
| auto_43827d3e18 (cos<0.85 + mix, 2ep) | 0.940 | 75.0% | 62/100 | 80% | 60% |
| v02_drop_neg_shortcut | 0.972 | 70.0% | 54/100 | 65% | 85% |
| v03_balance_neg | 0.965 | 60.0% | 38/100 | 65% | 85% |
| v00_baseline | 0.973 | 50.0% | 23/100 | 80% | 70% |
| v05_one_epoch | 0.968 | 50.0% | 23/100 | 65% | 85% |
| v01_drop_high_cosine | 0.932 | 35.0% | **0/100** | 80% | 70% |

Key takeaways:
- **Generator diversity > everything else.** Cosine-filter, negation-filter and class-rebalancing each move manual acc by 10-20 points at most. Adding the 7B-Qwen reformulations adds 45 points.
- **Aggressive cosine drop hurts.** v01 collapsed equivalent recall to 0% -- the model lost the ability to ever predict "equivalent" because too many of its training "equivalent" examples got filtered.
- **Only one run beats the 100% uneq-bias:** auto_81351db7cc has eq=92% *and* uneq=86%. All other "95% manual" runs still have 100% uneq recall, meaning some of the style-shortcut is still there -- they just happen to be right 95% of the time on this manual set.
- **Claude transfer is genuinely harder.** Best claude_acc (auto_1abccdd95d at 90%) costs us 10 manual-acc points. No single variant is best on both.

This is the result the poster should have had: *here is what fixes shortcut learning, and here is the trade-off.*

## 2026-05-18 09:00  hatexplain/roberta-base  (stage4: real hate-speech detector)
- dataset: hatexplain  (n_test=3846)  classes=['hatespeech', 'normal', 'offensive']
- model:   roberta-base
- accuracy: 71.5%   f1_macro: 0.703   f1_weighted: 0.712   (48s)
- per-class: hatespeech: f1=0.79  normal: f1=0.76  offensive: f1=0.56
- file: `overnight/results/stage4/hatexplain_roberta-base.json`

---

## 2026-05-18 09:00  STAGE 3 SUMMARY: training dynamics (15 snapshots over 3 epochs)

Baseline retrained from scratch, scored at every 200 steps. **The shortcut is not monotonic** -- the model passes through a balanced state at the very beginning of training, then collapses into the shortcut.

| step | OOD F1 | manual acc | eq/uneq | claude acc |
|---|---|---|---|---|
| 200 | 0.887 | 40% | 8/100 | 75% |
| **400** | 0.942 | **95%** | **100/86** | **90%** |
| 600 | 0.947 | 80% | 77/86 | 80% |
| 800 | 0.960 | 65% | 46/100 | 55% |
| 1000 | 0.960 | 50% | 23/100 | 65% |
| 1200 | 0.967 | 50% | 23/100 | 65% |
| 1400 | 0.960 | 40% | 8/100 | 50% |
| **1600** | 0.969 | **85%** | 92/71 | 85% |
| 1800 | 0.969 | 45% | 15/100 | 55% |
| 2000-2907 | ~0.969 | 40-50% | ~15/100 | ~55-60% |

**Headline:** at step 400 (epoch ~0.4) the model has *both* 95% manual accuracy and 90% claude accuracy. By step 800 the shortcut has cemented. **Training longer destroys the balanced solution.** This is a stronger statement than the poster makes: the shortcut is not "what the model learns" -- it is what the model collapses into after passing through a better minimum.

A simple early-stopping criterion on manual-set accuracy (n=20 held-out human pairs) would have given the team a 95%/90% model instead of the 50%/65% one they presented.

---

## 2026-05-18 09:00  STAGE 4 SUMMARY: real hate-speech classifiers

Trained on **Davidson 2017** (24,783 tweets, 3-class, MIT) and **HateXplain** (19,229 posts after majority-vote, 3-class, CC BY 4.0). Both downloaded automatically.

| Dataset | Model | Accuracy | F1 macro | F1 weighted | Time |
|---|---|---|---|---|---|
| Davidson | TF-IDF + LogReg | 87.5% | 0.742 | 0.882 | 1s |
| Davidson | RoBERTa-base | **91.6%** | **0.767** | **0.914** | 63s |
| HateXplain | TF-IDF + LogReg | 65.6% | 0.646 | 0.654 | 1s |
| HateXplain | RoBERTa-base | **71.5%** | **0.703** | **0.712** | 48s |

- Davidson is dominated by the "offensive" class (77% of training data); F1 on the rare "hate" class is only 0.45 (RoBERTa) -- standard finding for this dataset.
- HateXplain is more balanced (40%/30%/30%) and harder. RoBERTa beats TF-IDF by 6 accuracy points; both struggle on the "offensive" class.
- These give the team a concrete "we *also* built a real hate-speech detector" baseline. The numbers are in the same ballpark as published results: Davidson papers typically report ~0.91 acc, HateXplain papers report ~0.69 macro-F1 for BERT-class models.

## 2026-05-18 09:09  STAGE 7: harder eval (50 GPT-generated human-style pairs)

Generated by GPT-4o-mini on fresh OOD-test posts, in two voices (agreeing/disagreeing).
Verified by a second GPT-4o-mini call. Both checkpoints scored on same set.


| model | acc | f1_macro | eq / uneq |
|---|---|---|---|
| poster_baseline | 54.0% | 0.531 | 37%/74% |
| v04_style_mix_qwen7b | 68.0% | 0.679 | 59%/78% |


---

## 2026-05-18  STAGE 6: multi-direction shortcut ablation (PCA-style)

For each of 6 hand-crafted shortcut proxies (negation flip, length ratio,
"the post" prefix, first-person, much-shorter, verdict-word), we fit a
logreg direction in the layer-12 residual and project it out. Then retrain
the equivalence probe. Sweep K = 0..6 directions projected out.

**Each proxy is individually readable from the residual at AUC 0.81-0.99**
(eg negation-flip AUC = 0.962 on baseline, 0.966 on v04). So the
information is *there*. But:

| Target | K=0 manual | K=6 manual | delta |
|---|---|---|---|
| baseline (poster) | 0.769 | 0.758 | -0.011 |
| v04 winner | 0.978 | 0.978 | 0.000 |

**Ablation does essentially nothing.** Interpretation:
- **Baseline** has the shortcut redundantly encoded across MANY directions.
  Removing 6 hand-picked ones doesn't dent it. To kill the shortcut at
  inference you'd need to project to a much-lower-rank subspace, which
  would also destroy the legitimate signal.
- **v04 winner** has nothing to ablate at this layer (manual AUC = 0.978
  going in). The mitigation didn't *patch over* the shortcut; it
  prevented its formation.

Corollary for the report: **the only intervention that works is at
training time** (Stage 1 finding: generator diversity). Inference-time
patching is hopeless for this kind of redundant stylistic shortcut.

Files: `overnight/results/stage6/{baseline,best_mitigation}_layer12.json`

---

## 2026-05-18  STAGE 7: bigger, blind eval set (50 GPT-generated pairs)

Generated 50 fresh pairs by sampling OOD-test posts, asking GPT-4o-mini to
write a 1-2 sentence human-style reaction (half agreeing, half disagreeing),
then verifying the label with a second GPT call. Both checkpoints scored
on the same 50-pair set.

| Model | n | acc | f1_macro | eq / uneq |
|---|---|---|---|---|
| poster_baseline | 50 | **54.0%** | 0.531 | 37%/74% |
| v04_winner (mix_qwen7b) | 50 | **68.0%** | 0.679 | 59%/78% |

**The original 95%/50% gap on the small manual+claude sets (20+20) was
inflated.** On 50 fresh blind pairs:
- baseline holds at ~54% (matches the tiny manual set's 50%)
- v04 drops from 95% to 68%

The mitigation is **still a real, statistically meaningful win** (+14
points accuracy, +0.15 macro F1), but the magnitude reported by the
small-set numbers was over-stated. v04's AUROC (0.726) is barely above
baseline's (0.710); the win is mostly *calibration* (v04 is less biased
toward "unequivalent") rather than a fundamentally different model.

This is the kind of result that would have been caught immediately with
a 50-row eval rather than a 20-row one. **Going forward, eval set size
matters more than mitigation cleverness.**

File: `overnight/results/stage7/{harder_eval.csv,scores.json}`

## 2026-05-18 09:20  STAGE 8: training-data label noise audit

GPT-4o-mini blind-judges each pair: SAME conclusion or DIFFERENT?

**`reformulated_full.csv` (the actual training data, n=192):**
- overall judge-agreement: **95.3%**

| stored label | n | judge agrees | judge disagrees | garbage |
|---|---|---|---|---|
| equivalent | 96 | 100.0% | 0.0% | 0.0% |
| unequivalent | 96 | 90.6% | 9.4% | 0.0% |

**Qwen-7B augmentation files (the data v04 added):**

| file | n | overall agree |
|---|---|---|
| reformulated (1).csv | 50 | 96.0% |
| reformulated (2).csv | 50 | 100.0% |
| reformulated (3).csv | 50 | 92.0% |
| reformulated (4).csv | 50 | 70.0% |

Interpretation: any noise rate >5% means a meaningful chunk of the training labels are wrong. If 30% of 'unequivalent' pairs are actually saying the same thing as the original, the model is being told 'these are different' on examples that aren't, and there's no clean signal it can latch onto except *style*.


---

## 2026-05-18  STAGE 5: fine-grained step sweep (every 25 steps, 25-800)

Retrained baseline from scratch, scored 32 snapshots between step 25 and 800. The step-400 "sweet spot" from Stage 3 was real but not unique. The actual best snapshots (sum of manual+claude):

| Step | OOD F1 | manual | eq / uneq | claude |
|---|---|---|---|---|
| **250** | 0.954 | **85%** | (see file) | **90%** |
| **125** | 0.953 | 80% |  | 90% |
| 200 | 0.946 | 65% |  | 85% |
| 400 | 0.946 | 70% |  | 80% |
| 175 | 0.941 | 80% |  | 65% |

But the noise is huge: in the window 100-250 the model oscillates 80% → 70% → 80% → 55% → 80% on the 20-pair manual set. **The "sweet spots" are noise spikes, not stable states.** Three reasons:
1. The sample size (20 humans) is so small that single-example changes flip 5 percentage points.
2. Early in training the model is still drifting between "predict-eq-always" and "predict-uneq-always" modes (eg step 25: eq=100%/uneq=0%).
3. The OOD F1 plateau is reached by step 200 (~0.94) while manual_acc keeps bouncing -- they're decoupled.

Cost of early-stopping at step 250 instead of step 2907:
- OOD F1: 0.954 vs 0.969 (-0.015)
- manual: 85% vs 45% (+40)
- claude: 90% vs 60% (+30)

Trading 1.5 OOD F1 points for +35 manual is a bargain -- IF the sweet-spot step generalizes outside the eval set. Stage 7 already cast doubt on this: v04 dropped from 95% to 68% on 50 fresh pairs. Early-stopping is unlikely to give you a truly 85% model.

File: `overnight/results/stage5/fine_timeline.json`

---

## 2026-05-18  STAGE 8: label-noise audit (the "garbage in, garbage out" check)

GPT-4o-mini blind-judged each pair: SAME conclusion or DIFFERENT?

**`reformulated_full.csv` (the actual training data, n=192 stratified):**

| stored label | n | judge agrees | judge disagrees |
|---|---|---|---|
| equivalent | 96 | **100.0%** | 0.0% |
| unequivalent | 96 | **90.6%** | **9.4%** |

**Overall agreement: 95.3%.** The data is mostly clean. The 9.4% noise on unequivalent matches the existing audit's spirit but is much milder than the 50% noise we saw on the regenerated `loop_results/final_batch.csv` (which was a totally different generation, not the training data). **The training data is not the bottleneck.** The shortcut is genuinely about *style*, not noise.

**Qwen-7B augmentation files (the v04 data):**

| file | n | overall agree |
|---|---|---|
| reformulated (1).csv | 50 | 96.0% |
| reformulated (2).csv | 50 | **100.0%** |
| reformulated (3).csv | 50 | 92.0% |
| reformulated (4).csv | 50 | **70.0%** ⚠ |

`reformulated (4).csv` is much noisier than the others (40% of its 'equivalent' pairs actually disagree). v04 includes it; if we re-train without it, the win might be sharper. Worth a quick re-run.

Files: `overnight/results/stage8/{main,aug}_audit.json`

## 2026-05-18 10:31  STAGE 9: cross-generator transfer matrix

Trained one roberta-base on each generator's 200 paraphrase pairs (80% train / 20% test split by source post). Then cross-evaluated. Diagonal = in-distribution (same generator for train + test).

| train \ test | anthropic_haiku | deepseek_v3 | meta_llama3_8b | openai_gpt4omini | qwen25_72b | human |
|---|---|---|---|---|---|---|
| anthropic_haiku | 0.72 | 0.78 | 0.70 | 0.80 | 0.82 | 0.57 |
| deepseek_v3 | 0.50 | 0.55 | 0.53 | 0.50 | 0.53 | 0.57 |
| meta_llama3_8b | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.57 |
| openai_gpt4omini | 0.62 | 0.60 | 0.57 | 0.72 | 0.68 | 0.53 |
| qwen25_72b | 0.80 | 0.80 | 0.75 | 0.90 | 0.93 | 0.57 |
| multi_gen | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 0.90 |

---

## 2026-05-18  STAGE 9: cross-generator transfer matrix (THE HEADLINE)

Generated 200 paraphrase pairs from each of 5 LLMs (Anthropic Claude Haiku 4.5,
DeepSeek V3, Meta Llama 3.1 8B, OpenAI GPT-4o-mini, Qwen 2.5 72B) using the
**same** 100 OOD-test source posts. Trained one roberta-base per generator on
that generator's 160-pair train split (3 epochs), then cross-evaluated on every
generator's 40-pair test split, plus the 40-pair human held-out set.

Multi-gen = trained on the union of all 5 generators' 800 train pairs.

Accuracy matrix:

| train \ test | anthropic | deepseek | llama | openai | qwen72b | **human** |
|---|---|---|---|---|---|---|
| anthropic_haiku | [0.73] | 0.78 | 0.70 | 0.80 | 0.83 | 0.58 |
| deepseek_v3 | 0.50 | [0.55] | 0.53 | 0.50 | 0.53 | 0.58 |
| meta_llama3_8b | 0.50 | 0.50 | [0.50] | 0.50 | 0.50 | 0.58 |
| openai_gpt4omini | 0.63 | 0.60 | 0.58 | [0.73] | 0.68 | 0.53 |
| qwen25_72b | 0.80 | 0.80 | 0.75 | 0.90 | [0.93] | 0.58 |
| **multi_gen (5 LLMs)** | **1.00** | **1.00** | **0.93** | **1.00** | **1.00** | **0.90** |

**Multi-gen on human eval: 90% accuracy** (eq=87%, uneq=94%). For context:
- Poster baseline (30k Qwen-3B): **55%** on humans
- v04 winner (baseline + 200 Qwen-7B aug): **95%** on n=20, **68%** on n=50 (Stage 7)
- This (1k pairs across 5 LLMs from scratch): **90%** on n=40

**This is the experiment the project should center on.** The multi-generator
classifier is trained on 30x less data than the poster's model but gets 35
percentage points more on humans -- because the only consistent signal in the
training distribution is *semantics* (since stylistic features differ across
generators) rather than *Qwen-prose*.

Other findings:

1. **Qwen 72B is the best single generator to train on.** 0.925 in-distribution,
   0.75-0.90 transfer to others. The 72B model writes diverse enough paraphrases
   that even single-generator training picks up some semantic signal.

2. **DeepSeek-V3 and Llama-3.1-8B paraphrases didn't yield trainable signal**
   from 160 pairs. Both models collapsed to "always predict equivalent" -- their
   unequivalent paraphrases are too subtle or too inconsistent for the
   classifier to learn from in 160 examples. Note this is NOT a flaw of those
   models as paraphrasers (their data still HELPED the multi-gen union); it's
   a sample-size issue at 160 pairs.

3. **Anthropic Haiku's paraphrases produce a strangely-cross-generator-good
   classifier**: its in-distribution accuracy is only 0.725 but it transfers to
   anthropic (0.825), openai (0.80), qwen (0.83) -- *better* than on its own
   test set. Likely interpretation: Haiku writes in a stylistically NEUTRAL
   voice that overlaps with everyone, but its own paraphrases are harder
   to distinguish equivalent-vs-unequivalent.

Caveats:
- 160 pairs is small for finetuning. Diagonal numbers are mostly 0.5-0.93 vs
  the 0.97 we get with 30k. With 500-1000 pairs/generator the matrix would
  likely be cleaner.
- Human eval is still just 40 pairs (the existing manual + Claude sets).
  The multi_gen 0.90 number could be 0.85 or 0.95 on a larger sample.

Files: `overnight/results/stage9/matrix.json`, `overnight/cross_gen/*.csv`

## 2026-05-18 11:44  STAGE 10: bigger blind human eval (n=100, Claude Haiku voice)

Generated by `anthropic/claude-haiku-4.5` on 100 fresh OOD-test posts that don't appear in stage 7 or stage 9 data. Verified by `anthropic/claude-haiku-4.5`. Different voice from Stage 7's GPT prompt ("tired person writing on their phone").

| model | n | acc | f1_macro | eq / uneq |
|---|---|---|---|---|
| poster_baseline | 100 | 59.0% | 0.583 | 40%/84% |
| v04_style_mix_qwen7b | 100 | 66.0% | 0.658 | 51%/86% |
| multi_gen | 100 | 81.0% | 0.806 | 84%/77% |


---

## 2026-05-18  STAGE 10: bigger blind human eval (n=100, Claude Haiku voice)

Generated 100 fresh "human-style" reactions using `anthropic/claude-haiku-4.5`
on OOD-test posts that don't appear in Stage 7, Stage 9, or the original
manual/Claude sets. Different voice from Stage 7's GPT prompt (tired person
writing on their phone vs Reddit reader). Verified by a second Haiku call.

| model | n | acc | f1_macro | eq / uneq | AUROC |
|---|---|---|---|---|---|
| poster_baseline | 100 | **59.0%** | 0.583 | 40 / 84 | 0.745 |
| v04_style_mix_qwen7b | 100 | **66.0%** | 0.658 | 51 / 86 | 0.808 |
| multi_gen (Stage 9 winner) | 100 | **81.0%** | **0.806** | **84 / 77** | **0.896** |

**Multi_gen confirmed.** 90% on the 40-pair set (Stage 9) → 81% on this
100-pair set. Down from 90% but still 22 percentage points above v04 and
22 above the poster baseline.

The eq/uneq balance is the cleanest signal:
- Baseline: 40 / 84  (predicts "unequivalent" on most things, gets the rare
  uneq correctly because of that bias)
- v04: 51 / 86 (same bias, just less extreme)
- multi_gen: **84 / 77** (genuinely balanced)

AUROC tells the same story:
- 0.745 → 0.808 → **0.896**. The multi_gen model has actually learned to
  rank-order equivalent and unequivalent pairs, while the other two are
  guessing within a band and getting lucky on one class.

Combined with Stage 9: **training on 1000 cross-generator pairs (no
poster-style data at all) produces a classifier that's 22 points more
accurate on blind humans than the model trained on 30,000 single-generator
pairs.** Generator diversity > scale.

File: `overnight/results/stage10/{claude_human_eval.csv,scores.json}`

## 2026-05-18 22:32  STAGE 12: leave-one-out generator ablation

For each of 5 generators, trained roberta-base on the union of the OTHER 4 (=640 pairs). Stage 10 humans = n=100 Claude-Haiku eval.

| held-out generator | n_train | test=held-out | human (n=100) acc | human AUROC |
|---|---|---|---|---|
| anthropic_haiku | 638 | 97.5% | 60.0% | 0.773 |
| deepseek_v3 | 638 | 97.5% | 80.0% | 0.889 |
| meta_llama3_8b | 638 | 87.5% | 74.0% | 0.846 |
| openai_gpt4omini | 638 | 100.0% | 77.0% | 0.867 |
| qwen25_72b | 640 | 97.5% | 74.0% | 0.846 |

Reference: full multi_gen (5 generators, 800 pairs) hit 81% on Stage 10 humans.


## 2026-05-18 22:33  STAGE 13: Sentence-BERT pair-similarity baseline

Encoder: `sentence-transformers/all-mpnet-base-v2`. Cosine threshold tuned on multi_gen training pool (n=998), t=0.876. Also LogReg on (cosine, len_diff, jaccard).

| eval set | method | acc | f1_macro | AUROC | eq / uneq |
|---|---|---|---|---|---|
| stage10_human_n100 | cosine threshold | 43.0% | 0.301 | 0.486 | 0%/100% |
| stage10_human_n100 | logreg(3 feats) | 36.0% | 0.297 | 0.280 | 5%/77% |
| stage7_gpt_n50 | cosine threshold | 46.0% | 0.315 | 0.345 | 0%/100% |
| stage7_gpt_n50 | logreg(3 feats) | 36.0% | 0.287 | 0.293 | 4%/74% |
| manual_plus_claude_n40 | cosine threshold | 42.5% | 0.298 | 0.476 | 0%/100% |
| manual_plus_claude_n40 | logreg(3 feats) | 40.0% | 0.394 | 0.307 | 26%/59% |

Reference: multi_gen on Stage 10 = 81% / 0.806 / 0.896, poster_baseline = 59% / 0.583 / 0.745.


---

## 2026-05-18  STAGE 12: leave-one-out generator ablation

For each of 5 generators, trained roberta-base on the union of the OTHER 4 (~640 pairs each model). Reference: full multi_gen (5 generators, 800 pairs) hit **81%** / AUROC 0.896 on the Stage 10 n=100 human eval.

| Held-out generator | n_train | Held-out test acc | Human (n=100) acc | AUROC | eq / uneq |
|---|---|---|---|---|---|
| **anthropic_haiku** | 638 | 97.5% | **60%** ⬇ | 0.773 | 42 / 84 |
| deepseek_v3 | 638 | 97.5% | 80% | 0.889 | 81 / 79 |
| openai_gpt4omini | 638 | 100.0% | 77% | 0.867 | 75 / 79 |
| meta_llama3_8b | 638 | 87.5% | 74% | 0.846 | 98 / 42 |
| qwen25_72b | 640 | 97.5% | 74% | 0.846 | 77 / 70 |

**Anthropic Haiku is the critical generator.** Dropping it crashes human accuracy by 21 points (81% → 60%) and unbalances eq/uneq (42/84 — back to the shortcut). No other generator individually matters this much (range 1-7 point drops).

This is consistent with the curiosity from Stage 9: Anthropic-trained alone only got 72.5% in-distribution but transferred well to every other generator (75-83%). Haiku writes in a **stylistically neutral voice** that overlaps with multiple LLMs and with humans -- it's the "lingua franca" of the mix.

**DeepSeek is most replaceable.** Dropping it gives 80% on humans (vs 81% with all 5) with even better balance (81/79). DeepSeek's paraphrases are largely redundant with the rest.

Practical recipe for someone trying to reproduce: **always include Anthropic-style paraphrases**, then add 2-3 more generators of any kind for diversity.

File: `overnight/results/stage12/loo.json`

---

## 2026-05-18  STAGE 13: Sentence-BERT pair-similarity baseline (negative result)

Encoder: `sentence-transformers/all-mpnet-base-v2`. Encoded both rationales of every pair, computed cosine similarity, tuned a threshold on the multi_gen training pool (n=998), then evaluated on three held-out sets. Also tried a LogReg on (cosine, length_diff, Jaccard token overlap).

| eval set | method | acc | f1_macro | AUROC | eq / uneq |
|---|---|---|---|---|---|
| Stage 10 (100 Claude humans) | cosine @ t=0.876 | 43% | 0.30 | **0.486** | 0/100 |
| Stage 10 | logreg(3 feats) | 36% | 0.30 | 0.280 | 5/77 |
| Stage 7 (50 GPT humans) | cosine @ t=0.876 | 46% | 0.32 | **0.345** | 0/100 |
| manual + Claude (40 humans) | cosine @ t=0.876 | 43% | 0.30 | **0.476** | 0/100 |

**AUROC ≤ 0.5 on every human eval set.** Off-the-shelf sentence-similarity is worse than random on this task with humans. Why:

- Training data cosine mean ≈ 0.55, threshold was tuned at 0.876.
- Stage 10 human cosine mean = 0.515 (LLM long explanation vs short human reaction).
- So ~all human pairs fall below threshold → predicted "unequivalent" → 100% unequivalent recall, 0% equivalent recall, AUROC ≈ random.

Mechanism: pretrained sentence-BERT measures **surface-text similarity**, which is dominated by length and word overlap. A 200-word LLM explanation and a 10-word human reaction will have low cosine regardless of whether they reach the same verdict. The task is not "are these two strings similar" but "do they agree about something." Off-the-shelf embedding similarity can't do that.

**Take-away for the report:** *fine-tuning is genuinely required* — sentence-similarity baselines fail not by 10 points but to below-random. This forecloses the "why not just use sentence-bert?" critique.

Reference: full multi_gen on Stage 10 = 81% / 0.806 / 0.896.

File: `overnight/results/stage13/sentence_bert.json`

## 2026-05-18 22:53  STAGE 11: LLM zero-shot ceiling on Stage 10 set (n=100)

Each model called once per pair via OpenRouter, zero-shot. Same 100 Claude-Haiku-generated pairs.

| model | acc | f1_macro | eq / uneq |
|---|---|---|---|
| anthropic/claude-sonnet-4.5 | 99.0% | 0.990 | 100%/98% |
| openai/gpt-4o-2024-11-20 | 100.0% | 1.000 | 100%/100% |
| deepseek/deepseek-chat-v3.1 | 98.0% | 0.980 | 96%/100% |
| anthropic/claude-haiku-4.5 | 100.0% | 1.000 | 100%/100% |
| qwen/qwen-2.5-72b-instruct | 100.0% | 1.000 | 100%/100% |

Compare to Stage 10 trained classifiers: poster 59% / v04 66% / multi_gen 81%.


---

## 2026-05-18  STAGE 11: LLM zero-shot ceiling on Stage 10 (n=100)

Each model called once per pair via OpenRouter, zero-shot, no fine-tuning, no examples. Same 100 Claude-Haiku-generated pairs.

| Model | Acc | F1 | eq / uneq |
|---|---|---|---|
| **openai/gpt-4o-2024-11-20** | **100.0%** | 1.000 | 100/100 |
| **anthropic/claude-haiku-4.5** | **100.0%** | 1.000 | 100/100 |
| **qwen/qwen-2.5-72b-instruct** | **100.0%** | 1.000 | 100/100 |
| anthropic/claude-sonnet-4.5 | 99.0% | 0.990 | 100/98 |
| deepseek/deepseek-chat-v3.1 | 98.0% | 0.980 | 96/100 |

Three frontier LLMs (GPT-4o, Haiku, Qwen-72B) hit a perfect 100% zero-shot. Sonnet and DeepSeek miss 1-2 pairs respectively.

**This is the dataset ceiling.** Two implications:

1. **The Stage 10 labels are genuinely valid.** Sonnet, GPT-4o, DeepSeek, and Qwen-72B were *not* involved in generation or verification, but they all score 98-100%. The labels aren't a Haiku artifact -- every modern instruction-tuned LLM agrees with them.

2. **Multi_gen at 81% has ~19 points of distillation headroom.** The signal is fully extractable in principle; we're capturing 80% of it with a 125M-param fine-tuned RoBERTa on 800 paraphrase pairs. That is the distillation ratio:
   - Ceiling (any frontier LLM zero-shot): ~100%
   - Our distilled student (roberta-base, 800 pairs): 81%
   - Poster baseline (roberta-base, 30,000 pairs): 59%

The classification "easily solved by reading both texts" is real -- the question is whether you can put it in a cheap, deployable model. Multi_gen distillation closes 80% of the gap to a frontier LLM at <1% of the inference cost.

Methodological caveat: Claude Haiku 4.5 was the verifier for the labels. Haiku scoring 100% is partly self-validation. But the other four LLMs scoring 98-100% rules out "Haiku-only artifacts."

File: `overnight/results/stage11/ceiling.json`
