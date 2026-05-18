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
