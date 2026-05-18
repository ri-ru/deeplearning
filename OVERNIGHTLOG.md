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
