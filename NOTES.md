# EE559 Group Mini-Project - Notes & Findings

## What this project is

We are building a **rationale-equivalence judge** - a binary classifier that takes two hate-speech rationales and decides if they mean the same thing.

- **Input:** two texts (original explanation + reformulated version)
- **Output:** `equivalent` or `unequivalent`
- **Why it matters:** once trained, anyone can use it to check whether a model's generated reasoning agrees with their ground truth, without paying GPT-4 per call

This is the core artifact. It is not a hate speech classifier itself - it is a tool for *evaluating* hate speech classifiers. The paper needs to justify this clearly in the intro.

---

## The dataset (`reformulated_full.csv`)

- **52,137 rows**, exactly 50/50 balanced (26,069 equivalent / 26,068 unequivalent)
- **Source:** HateCoT dataset - real social media posts with human-written explanations
- **How Flavia generated it:** took each human explanation and ran GPT-4 twice:
  1. "Paraphrase this, keep the same meaning" → `equivalent` row
  2. "Rewrite this but change the conclusion" → `unequivalent` row (adversarial)
- **Columns:** `id`, `explanation` (original rationale), `domain`, `label` (hate/benign), `post`, `target`, `uid`, `synthetic` (reformulated rationale), `meaning` (the label: equivalent/unequivalent)

### Example - equivalent
- Post: *"you keep thinking im black"*
- Explanation: *"benign, no harmful language, no stereotypes"*
- Synthetic: *"benign because it lacks offensive language, no harmful intent"*
- Label: **equivalent** (same conclusion, different words)

### Example - unequivalent
- Post: *"Same. At first I thought I just misread it..."*
- Explanation: *"does not target any identity, general confusion, Neutral content"*
- Synthetic: *"targets a specific identity for abuse, promotes hate, Hate speech"*
- Label: **unequivalent** (conclusion completely flipped)

Data is bad: 

  [3] ORIGINAL:  "should be classified as Person Directed Abuse"
  [3] SYNTHETIC: "should NOT be classified as Person Directed Abuse"

  [5] ORIGINAL:  "should be classified as Hate"
  [5] SYNTHETIC: "should be classified as not Hate"

  [6] ORIGINAL:  "should be classified as Offensive"
  [6] SYNTHETIC: "should NOT be classified as Offensive"

---

## Domain split

| Split | Domains | Size |
|-------|---------|------|
| Train | dynahate, cad, measure, socialbias, toraman | ~25k |
| Val | carved from train (10%) | ~3.4k |
| OOD Test | toxigen, gab, anatomy | ~17.7k |

The key scientific claim: the model generalizes to **unseen domains** (OOD). This is what makes it interesting - not just in-domain accuracy.

---

## What fine-tuning is

RoBERTa-base was pre-trained by Meta on 160GB of English text to predict masked words. It already understands meaning, entailment, negation, paraphrase.

Fine-tuning = take those 125M pre-trained weights as a starting point, throw away the vocabulary prediction head, add a new 2-output classification head, and nudge all weights slightly on your labeled data.

The model input is: `[CLS] explanation [SEP] synthetic [SEP]` - both texts fed as a sentence pair. The `[CLS]` token's final hidden state becomes a vector that the classification head reads.

**Key hyperparameters:**
- Learning rate: `2e-5` (small - don't destroy pre-trained weights)
- Epochs: 3
- Batch size: 32
- Max length: 256 tokens
- bf16: true (avoids NaN on 5090, which had issues with fp16)

---

## Training results (full run - `configs/roberta_base.yaml`)

### Training loss curve
```
epoch 0.06:  loss = 0.697   ← random guessing baseline (ln(2) = 0.693)
epoch 0.19:  loss = 0.137   ← already learned something real
epoch 1.0:   eval F1 = 0.985, eval accuracy = 98.5%
epoch 2.0:   eval F1 = 0.989, eval accuracy = 98.9%  ← best checkpoint saved
epoch 3.0:   eval F1 = 0.989, eval accuracy = 98.9%  ← no improvement
```

### OOD test results

| Domain | n | Accuracy | F1 macro | AUROC |
|--------|---|----------|----------|-------|
| **Overall** | **17,691** | **96.8%** | **0.968** | **0.997** |
| anatomy | 5,418 | 94.9% | 0.949 | 0.993 |
| gab | 5,034 | 97.6% | 0.976 | 0.998 |
| toxigen | 7,239 | 97.7% | 0.977 | 0.998 |

**anatomy is the weakest domain** - likely because its rationales are shorter and more ambiguous. Error analysis material.

### Confusion matrix (OOD overall)
```
                  predicted equivalent   predicted unequivalent
true equivalent        8655                    216
true unequivalent       349                   8471
```
Model makes more mistakes on unequivalent examples (349 wrong) than equivalent (216 wrong). It is slightly biased toward calling things equivalent when uncertain.

### Smoke test comparison (1,783 train examples, 1 epoch)
| Domain | Smoke F1 | Full F1 |
|--------|----------|---------|
| anatomy | 0.787 | **0.949** |
| gab | 0.981 | 0.976 |
| toxigen | 0.957 | **0.977** |

More data helped anatomy massively. Gab was already nearly saturated.

---

## Artifacts

All in `runs/roberta_base/`:
- `checkpoints/` - the trained model (load with `from transformers import pipeline`)
- `metrics.json` - the results table (goes in the paper)
- `test_predictions.csv` - every test example with `label`, `pred`, `p_unequivalent` (use for error analysis)

---

## Zero-shot baseline (in progress)

Running GPT-4o-mini on 500 OOD test samples via `scripts/zero_shot_baseline.py`.

**Why 500 not 17k:** costs money per API call. 500 samples gives ±3% confidence interval on F1, which is tight enough. RoBERTa runs free on local GPU so we ran it on everything.

**Why we run this:** the paper needs a comparison. The story is:
> "A 125M parameter model fine-tuned in 3 minutes on a single GPU matches or beats a large LLM called via API - making it practical to run locally without per-call costs."

### Results - GPT-4o-mini zero-shot (500 samples)

| Domain | n | Accuracy | F1 macro |
|--------|---|----------|----------|
| **Overall** | **500** | **96.4%** | **0.964** |
| anatomy | 155 | 91.6% | 0.914 |
| gab | 157 | 99.4% | 0.994 |
| toxigen | 188 | 97.9% | 0.979 |

### Head-to-head comparison (full results table)

| Model | Overall F1 | anatomy | gab | toxigen |
|-------|-----------|---------|-----|---------|
| TF-IDF + LogReg (no neural net) | 0.789 | 0.759 | 0.776 | 0.815 |
| GPT-4o-mini zero-shot | 0.964 | 0.914 | **0.994** | 0.979 |
| **RoBERTa-base fine-tuned (ours)** | **0.968** | **0.949** | 0.976 | **0.977** |

**Key finding:** A 125M parameter model fine-tuned in 3 minutes on a single GPU **matches GPT-4o-mini** overall and **beats it on anatomy** (0.949 vs 0.914). The TF-IDF baseline shows the task is not trivially solvable by keyword matching - 18 points of the gap reflects genuine semantic understanding that RoBERTa learns.

**The data is not "too easy":** TF-IDF gets 78.9% - if the task were just surface keyword matching it would score 95%+. The remaining 18-point gap to RoBERTa is real semantic understanding.

---

## Paper framing (important)

**The abstract currently talks about adversarial labels - this needs to be fixed.** The dataset pivoted to rationale-equivalence, not label-correctness. Fix this before submission or the grader will catch it.

**Justification for being in-scope for EE559:** the course asks for hate speech detection. Frame the contribution as: *"Hate speech detection is bottlenecked by evaluation cost. We build the evaluator that lets practitioners cheaply check whether any classifier's reasoning aligns with their values."* The UN definition of hate speech (no universal legal definition, inherently subjective) motivates the practitioner-anchored approach.

**Cite HateCoT paper** - the dataset source. Currently missing from the report.

---

## Progress log

### Session 2025-05-02
- Set up nml_project_proposal venv (Python 3.14.4, torch 2.11.0+cu128) for this project
- Installed missing packages: transformers 5.7.0, datasets 4.8.5, accelerate 1.13.0, scikit-learn 1.8.0
- Fixed transformers 5.x API break: `tokenizer=` → `processing_class=` in Trainer
- Smoke test passed (1783 train, 1 epoch): OOD F1 = 0.915
- Full run completed (25212 train, 3 epochs, ~3 min on RTX 5090): OOD F1 = 0.968
- GPT-4o-mini zero-shot baseline: OOD F1 = 0.964 on 500 samples
- TF-IDF + logistic regression dumb baseline: OOD F1 = 0.789
- Identified project drift: original plan (adversarial label detection) ≠ what was built (rationale equivalence)
- Wrote METHODOLOGY.md with honest account of what was planned vs. built, and how to reconnect

---

## What to do next

1. ~~**Get baseline results**~~ - done. RoBERTa matches GPT-4o-mini overall, beats it on anatomy.
2. **Decision required:** Option A (rewrite paper around rationale equivalence) vs Option B (add hate speech pipeline experiment to reconnect to course objective). See METHODOLOGY.md.
3. **Fix the abstract** - currently placeholder "A very nice abstract". Must match whichever option is chosen.
4. **Add HateCoT citation** - missing from bibliography entirely.
5. **Error analysis** - open `runs/roberta_base/test_predictions.csv`, filter `label != pred`, look at anatomy failures. Short paragraph for the paper.
