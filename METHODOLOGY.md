# Methodology - What We Planned, What We Built, What To Do Next

---

## What we originally planned

The original thesis was clean and directly on-topic for the course:

> "Hate speech detection models make predictable errors. We can simulate those errors synthetically, then train a small judge to distinguish a model's correct label from its plausible mistakes."

**The three-step pipeline:**

1. **Ground truth data.** Take real social media posts that already have human-annotated hate speech labels (correct/benign). Source candidates: Jigsaw/Civil Comments (1.8M), ToxiGen, ToxicChat.

2. **Adversarial label generation.** Prompt a large LLM (GPT-4, Claude) with different "error profiles" to generate *plausible but wrong* labels for each post. Error profiles simulate real model failure modes:
   - `sarcasm_blind`: classify this post, but you fail to detect irony
   - `overly_sensitive`: classify this post, but you flag any mention of a minority group
   - `profanity_fixated`: classify this post, but you over-flag profanity regardless of intent
   - `implicit_miss`: classify this post, but you miss hate speech expressed through metaphor

3. **Train the judge.** Fine-tune a small model (RoBERTa, DeBERTa) on triples of (post, label, correct/adversarial). It learns to recognize when a proposed label matches the ground-truth pattern vs. when it looks like a model error.

**Why this is interesting:**
- Directly produces a tool for hate speech model evaluation
- Forces the judge to understand *error modes*, not just agree with GPT-4
- The Prometheus line of work (Kim et al.) showed this works for general LLM judging - nobody had done it specifically for hate speech

---

## What actually got built (and why it drifted)

The original plan hit three walls:

**Wall 1 - Dataset overlap.** The plan was to train on Jigsaw and test on ToxiGen OOD. Problem: HateCoT (the dataset Flavia was already using) already incorporates ToxiGen as one of its 8 sub-domains. Training on one and testing on the other becomes partially in-distribution. The OOD split had to change.

**Wall 2 - Adversarial generation quality.** The LLM-generated adversarial labels weren't aggressive enough. The `profanity_fixated` profile only flipped 5–10% of labels. Many "adversarial" generations were accidentally correct (model said "hate speech" for a genuinely hateful post even when prompted to be wrong). The data would have been too noisy.

**Wall 3 - Content policy refusals.** Anthropic refused ~30% of requests to generate adversarial labels for explicitly toxic content. The profile that needed to generate plausible-but-wrong labels for the hardest cases was exactly the one that got blocked most.

**The pivot Flavia made:**

Instead of (post, correct_label, adversarial_label), she reformulated the task as:

> Given an original rationale explaining a hate speech label and a reformulated version of that rationale, are they saying the same thing?

- **`equivalent`**: GPT-4 paraphrased the original explanation, preserving meaning
- **`unequivalent`**: GPT-4 rewrote the explanation with a flipped conclusion ("benign" ↔ "hate speech")

This produced 52,137 clean pairs, exactly 50/50 balanced, across 8 domains. The adversarial generation problem went away because "generate a paraphrase" and "flip the conclusion" are both easy, non-toxic requests that no model refuses.

**What this is actually measuring:**
Rationale equivalence, not label correctness. The model learns: "do these two explanations agree?" - not "is this label correct given this post?"

---

## The gap between plan and implementation

| | Original plan | What was built |
|---|---|---|
| Input | (post, proposed_label) | (original_rationale, reformulated_rationale) |
| Output | correct / adversarial | equivalent / unequivalent |
| What it detects | wrong hate speech labels | changed/preserved meaning in explanations |
| Directly detects hate speech? | Yes - label is right or wrong | No - just checks if two explanations agree |
| Post content used? | Yes | No |
| Error modes simulated? | Yes (sarcasm-blind, etc.) | No - just meaning flip |

The current system is a **semantic equivalence classifier on rationale pairs**. It is *related* to the original thesis but does not directly fulfill it. It would catch a model that outputs a flipped conclusion, but not a model that outputs a subtly wrong conclusion in similar language.

---

## Results (what we have)

| Model | Overall F1 | anatomy | gab | toxigen |
|-------|-----------|---------|-----|---------|
| TF-IDF + LogReg (dumb baseline) | 0.789 | 0.759 | 0.776 | 0.815 |
| GPT-4o-mini zero-shot | 0.964 | 0.914 | **0.994** | 0.979 |
| **RoBERTa-base fine-tuned (ours)** | **0.968** | **0.949** | 0.976 | **0.977** |

- Trained on 5 domains, tested OOD on 3 never-seen domains
- 3 minutes training on RTX 5090 (25k examples, 3 epochs)
- Matches GPT-4o-mini overall, beats it on anatomy (the hardest domain)
- TF-IDF gap (18 points) proves the task requires genuine semantic understanding, not keyword matching

**Anatomy is consistently the weakest domain** across all models - shorter, more ambiguous rationales that are harder to distinguish semantically.

---

## How to reconnect to the original thesis (Option B)

If we want to make this directly about hate speech detection, one additional experiment:

**The idea:** use the trained judge as a component in a hate speech evaluation pipeline.

1. Take a held-out set of posts with ground-truth hate/benign labels and human rationales (from HateCoT).
2. Run a hate speech classifier (e.g., RoBERTa zero-shot, or fine-tuned on a subset) - get its predicted label *and* its generated rationale.
3. Feed (ground_truth_rationale, classifier_rationale) into our judge.
4. Measure: when the judge says "unequivalent," does the classifier's label also disagree with ground truth? If yes, the judge is a proxy for label correctness.

**What this proves:** our rationale-equivalence judge can surface label errors without seeing the labels directly. That reconnects the whole project to the original thesis: *a judge that catches when a hate speech model's reasoning is wrong.*

**Concretely, what code to write:**
```python
# pseudocode
for post, gt_label, gt_rationale in held_out_set:
    classifier_label, classifier_rationale = hate_classifier(post)
    judge_verdict = our_judge(gt_rationale, classifier_rationale)
    # does judge_verdict == "unequivalent" predict (classifier_label != gt_label)?
```

If the judge's "unequivalent" verdict correlates with label disagreement, we've shown the judge is a useful signal for detecting classification errors - which is the original thesis.

**Effort:** one script, maybe 200 lines, one day of work.

---

## Which paper to write

**Option A - Accept the pivot (less work, B-range grade):**

Rewrite the intro/methodology to match what was actually built. Frame it as:
> "Hate speech detection is subjective and evaluation is expensive. We build a local rationale-consistency judge - a 125M parameter model that matches GPT-4o-mini at inference time, runs for free locally, and generalizes OOD."

The contribution is: efficient, local alternative to LLM-based rationale evaluation. The OOD generalization and TF-IDF gap are the evidence.

**Option B - Add the pipeline experiment (more work, A-range grade):**

Keep everything we have, add one experiment connecting the judge to actual label correctness. The paper's thesis becomes coherent end-to-end:
> "We built a rationale-equivalence judge, and showed it serves as a proxy for label correctness in hate speech detection - catching classifier errors without requiring label comparisons."

This directly fulfills the course objective (deep learning to detect hate speech) and has a novel angle (judge-as-error-detector via rationale comparison).
