# Project Synthesis — What We Did, What It Means, What To Say

---

## One paragraph summary

We set out to build a "judge" that can tell when a hate speech model's reasoning is wrong. The original plan was to generate adversarially wrong labels using LLMs and train a classifier to spot them. That plan hit technical walls (LLMs refused to generate toxic content, adversarial prompts weren't aggressive enough). Flavia pivoted to a cleaner but narrower task: given two *explanations* of the same post, do they say the same thing? We fine-tuned RoBERTa on 52k such pairs and got 96.8% accuracy on domains the model never saw during training — matching GPT-4o-mini with a model 100x smaller that runs locally for free. The results are genuinely good. The gap between what we planned and what we built is the main thing that needs addressing before the paper is done.

---

## The original idea

**The thesis:** hate speech classifiers make predictable errors. You can simulate those errors synthetically, then train a small model to detect them. Result: a judge that catches when a classifier's output is wrong, without needing human annotation at test time.

**The pipeline:**
1. Real posts with ground-truth hate/benign labels (Jigsaw, ToxiGen, etc.)
2. Prompt GPT-4 with "error profiles" to generate plausible-but-wrong labels:
   - sarcasm-blind: misses ironic hate speech
   - overly-sensitive: flags any mention of a minority group
   - profanity-fixated: over-flags swearing regardless of intent
   - implicit-miss: misses hate speech expressed through metaphor
3. Train a small model to distinguish correct labels from adversarial ones
4. Show it generalizes OOD — train on one dataset, test on another

**Why it was interesting:** directly addresses hate speech detection, forces the judge to understand error modes rather than just pattern-match, produces something useful (a cheap local evaluator).

---

## What actually got built

**Why the plan drifted:**
- Jigsaw → ToxiGen OOD split broke because HateCoT already includes ToxiGen as a sub-domain
- LLM-generated adversarial labels were too weak (profanity profile only flipped 5–10% of labels)
- Anthropic refused ~30% of generation requests for explicit content

**What Flavia built instead:**
Used the HateCoT dataset (real posts + human-written explanations across 8 sub-domains). GPT-4 was used to generate two types of reformulations per explanation:
- **Equivalent:** paraphrase that preserves the meaning
- **Unequivalent:** rewrite that flips the conclusion ("benign" → "hate speech" or vice versa)

Result: 52,137 pairs, 50/50 balanced. The task became: given two rationales, do they agree?

**What shifted:** the input is no longer (post, label) — it's (rationale_A, rationale_B). The post itself is not used. The model doesn't classify hate speech directly — it checks whether two explanations are consistent.

---

## The model and training

- **Model:** RoBERTa-base (125M parameters, pre-trained by Meta on 160GB of English)
- **Fine-tuning:** added a 2-class head, trained on sentence pairs [CLS] explanation [SEP] synthetic [SEP]
- **Split:** trained on 5 domains (dynahate, cad, measure, socialbias, toraman), tested OOD on 3 (toxigen, gab, anatomy)
- **Hardware:** RTX 5090, ~3 minutes total training time
- **Key hyperparameters:** lr=2e-5, 3 epochs, batch 32, bf16 precision

---

## Results

| Model | Overall F1 | anatomy | gab | toxigen |
|-------|-----------|---------|-----|---------|
| TF-IDF + logistic regression | 0.789 | 0.759 | 0.776 | 0.815 |
| GPT-4o-mini zero-shot | 0.964 | 0.914 | **0.994** | 0.979 |
| RoBERTa-base fine-tuned | 0.968 | 0.949 | 0.976 | 0.977 |
| RoBERTa-base + hard augmentation | 0.969 | 0.951 | 0.976 | 0.977 |
| **RoBERTa-large-MNLI fine-tuned** | **0.973** | **0.956** | 0.976 | **0.984** |

**What these numbers mean:**
- TF-IDF at 78.9% proves the task isn't solvable by keyword overlap — genuine semantic understanding is required
- GPT-4o-mini at 96.4% is the "what a big LLM can do zero-shot" ceiling
- RoBERTa-base at 96.8% matches it — 125M params, 3 minutes training, free local inference
- Hard augmentation barely helped (+0.2% anatomy) — LLM-generated hard examples stay in-distribution
- RoBERTa-large-MNLI at 97.3% beats GPT-4o-mini — NLI pre-training cuts unequivalent errors nearly in half (349 → 181 missed)
- Anatomy is the hardest domain for everyone — most different annotation style from the training domains
- Model converged in epoch 1 — by epoch 2 validation F1 plateaued at 98.9%

**Confusion matrix (OOD, 17,691 examples):**
- 8,655 correct equivalents, 216 missed (said unequivalent when equivalent)
- 8,471 correct unequivalents, 349 missed (said equivalent when unequivalent)
- The model is slightly biased toward calling things equivalent when uncertain

---

## Hard test set experiment — key finding

We generated 200 "hard" pairs using GPT-4o-mini with stricter prompts (no lazy negation, changed reasoning not just wording). Then evaluated our trained model:

| Test set | n | F1 macro |
|----------|---|----------|
| Easy — original anatomy OOD (human-reformulated) | 5,418 | 0.949 |
| Hard anatomy — GPT-4o-mini generated | 200 | **1.000** |
| Hard all-domains — GPT-4o-mini generated | 200 | **0.995** |

**The model got 100% on anatomy hard set (0 failures) and 99.5% on all-domains hard set (1 failure in 200).**

This is not the model being impressive. This is a diagnosis:

> The model learned GPT-4o-mini's generation style, not semantic equivalence as a general concept.

Training data was GPT-4o-mini-generated. Hard test set was also GPT-4o-mini-generated. The model recognizes the stylistic fingerprint and gets everything right. The original anatomy set (94.9%) is harder specifically because those are human-reformulated rationales — a different stylistic distribution.

**What this means:** the model is a GPT-4o-mini style classifier disguised as a semantic equivalence judge. A truly robust judge would need training on human-written paraphrases, not LLM-generated ones.

**What this means for the paper:** this is a stronger finding than 96.8% OOD accuracy. It explains why anatomy is hard, why the "hard" test wasn't hard, and gives a clear direction for future work. Write it as a limitation — it shows you understood what the model actually learned.

---

## Honest assessment of the project

**What's good:**
- OOD generalization is real and well-designed — 5 train domains, 3 test domains, no leakage
- Results are strong and reproducible (all code runs, artifacts saved)
- Three-baseline table (TF-IDF / GPT-4o-mini / ours) is a clean story
- The task requires semantic understanding, not keyword matching (TF-IDF gap proves it)

**What's weak:**
- The original thesis (judge for hate speech label correctness) drifted to a narrower task (rationale equivalence) — the paper's intro still describes the original, which is now inconsistent with what was built
- The unequivalent examples are sometimes obvious flips ("neutral" → "hate speech"), which inflates the apparent difficulty
- The model doesn't directly do hate speech detection — it's a meta-level tool, which is on the edge of the course scope
- One epoch is enough to converge — the task may be easier than it looks

**The gap that needs closing:** the paper introduction and the actual system are describing different things. This must be fixed before submission.

---

## Two paths forward

**Option A — Accept the pivot (less work, B-range)**

Rewrite the intro to match what was built. Frame as:
> "Hate speech evaluation is expensive and subjective. We build a lightweight rationale-consistency judge — 125M params, 3 minutes to train, matches GPT-4o-mini — that can run locally without per-call API costs."

Stop trying to frame it as adversarial label detection. The OOD results and the TF-IDF gap are the evidence. The contribution is: a cheap, local, generalizable semantic equivalence checker for hate speech rationales.

**Option B — Add one experiment, reconnect to original thesis (more work, A-range)**

Add ~200 lines of code, one day of work:
1. Take held-out posts + ground-truth hate labels + human rationales
2. Run a hate speech classifier, get its predicted label and its generated rationale
3. Feed (ground_truth_rationale, classifier_rationale) into our judge
4. Measure: does "unequivalent" predict "classifier got the label wrong"?

If the judge's verdict correlates with label errors, we've shown it's a proxy for classification correctness — which is exactly the original thesis. The paper becomes coherent end-to-end.

---

## What to say in a meeting

**If asked "what did you build?"**
> "A semantic equivalence classifier for hate speech rationales. Given two explanations of why a post is or isn't hate speech, it tells you whether they're saying the same thing. It matches GPT-4o-mini at that task but runs locally for free."

**If asked "why is this useful?"**
> "If you're running a hate speech detection system and you want to check whether the model's reasoning aligns with your ground truth, you can use this judge instead of paying for GPT-4 per call. It's reusable, local, and fast."

**If asked "how does it connect to hate speech detection?"**
> "It's a meta-level contribution — not a classifier itself, but a tool for evaluating classifiers. The professor said all directions are valid as long as we justify them. Our justification: hate speech detection is bottlenecked by evaluation cost, and this addresses that."

**If asked about the drift from the original plan:**
> "The original plan was adversarial label generation — we hit walls with content policy refusals and weak adversarial outputs. We pivoted to rationale equivalence because it's a cleaner formulation of the same underlying problem: detecting when a model's reasoning is inconsistent with ground truth. We have one experiment planned that reconnects these directly."

**If asked "is the task too easy?"**
> "TF-IDF gets 78.9% — so about 79% of cases are solvable by keyword overlap. The remaining 18 points our model gains requires genuine semantic understanding. It's not trivial, but it's also not the hardest NLP task. The difficulty lives in the anatomy domain, which is stylistically distinct from anything in training."

**If asked "what would you do differently?"**
> "The adversarial generation pipeline is the interesting unsolved piece. If the error profiles worked — sarcasm-blind, over-sensitive, implicit-miss — we'd have a judge that understands *how* classifiers fail, not just *that* two explanations differ. That's the next experiment."

---

## Files

| File | What it is |
|------|-----------|
| `reformulated_full.csv` | The full 52k dataset |
| `judge_finetune/` | All training code |
| `judge_finetune/runs/roberta_base/metrics.json` | Results table (paper) |
| `judge_finetune/runs/roberta_base/test_predictions.csv` | Every test prediction (error analysis) |
| `judge_finetune/runs/zero_shot_gpt4omini.json` | GPT-4o-mini baseline results |
| `NOTES.md` | Technical notes and progress log |
| `METHODOLOGY.md` | Detailed methodology + what drifted + Option B implementation |
| `SYNTHESIS.md` | This file |
