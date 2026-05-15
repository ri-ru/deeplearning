# Synthetic Data Generation - Few-Shot Improved

## The problem with previous generation attempts

The original `reformulated_full.csv` dataset has a critical flaw: ~50% of pairs
labelled "unequivalent" are actually mislabeled. GPT-4 was asked to flip the
conclusion but mostly just inserted "NOT" into the original sentence:

> ORIGINAL:  "should be classified as Hate Speech because it contains hostile language"
> UNEQUIV:   "should NOT be classified as Hate Speech because it contains hostile language"

A RoBERTa model trained on this data learns to detect the presence of a negation
token, not genuine semantic difference. The self-improving prompt loop
(`prompt_loop.py`) attempted to fix this over 10 iterations but never converged -
it optimized for cosine similarity and TF-IDF separability, but had no signal
that the generated pairs were actually mislabeled.

## The fix: few-shot generation from manual labels

### What we have

`manual labelling.csv` contains 20 pairs that were manually judged by a human.
Each row has:
- **Post** - the original social media post
- **Explanation** - the AI-generated rationale
- **Human comment** - the human's reasoning for their verdict
- **Label** - `Equivalent` or `Unequivalent`

The 7 Unequivalent cases are especially valuable: they are examples where a human
identified a genuine semantic disagreement, often for subtle reasons:
- A rationale that frames a trans-related post as pathological (discrediting the community)
- A post where disclosing someone's nationality changes the framing
- A post where the AI said "not hate" but the human read racial discrediting

These are exactly the kind of hard cases that the previous generation pipeline
completely missed.

### The approach

Instead of zero-shot generation ("rewrite this to reach the opposite conclusion"),
we inject the manually labelled examples **directly into the generation prompt**
as few-shot demonstrations. The LLM sees concrete examples of what a genuine
semantic flip looks like - not a negation shortcut, but a real change in reasoning
- before it generates anything.

**For unequivalent generation:**
The prompt shows 3–4 manually labelled Unequivalent examples with the human
comment as annotation, so the model understands *why* two rationales disagree.

**For equivalent generation:**
The prompt shows 3–4 Equivalent examples where the human confirmed the meaning
is preserved despite different wording.

### Why this should work better

The manual labels encode human judgment about what makes two rationales genuinely
different vs. superficially different. By putting this judgment in the prompt as
examples, we give the generator a concrete target to imitate rather than an
abstract rule to follow. Previous failure modes (lazy negation, register shift,
length explosion) all stem from the model not having a clear example of what
"good" looks like.

### Quality gates

Not used in this version. The focus is on validating whether few-shot prompting
alone produces better pairs before adding automated filtering on top.

## Environment

Cluster GPU: **Tesla V100-SXM2-32GB**, ~10GB allocated, CUDA 13.0, compute capability 70.
- No bfloat16 support → must use `float16`
- AWQ requires capability 75+ → not supported on V100
- GPTQ requires capability 60+ → works on V100

## How to use

```bash
# Option 1 - GPTQ 4-bit (recommended, keeps 7B quality)
python reformulate.py \
    --input "hatecot_final_D3 (2).csv" \
    --model "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" \
    --quantization gptq \
    --dtype float16 \
    --n-samples 0

# Option 2 - 3B no quantization (~6GB, fallback if GPTQ fails)
python reformulate.py \
    --input "hatecot_final_D3 (2).csv" \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --quantization none \
    --dtype float16 \
    --n-samples 0

# Train RoBERTa with the new synthetic data injected
python -m src.train \
    --config ../judge_finetune/configs/roberta_base_augmented.yaml
```

## Files

- `reformulate.py` - few-shot generation script
- `manual_labelling.csv` - 20 manually labelled pairs used as few-shot examples
- `README.md` - this file

## What to measure

After generating and training, compare against the baseline in `NOTES.md`:

| Model | Overall F1 | anatomy | gab | toxigen |
|-------|-----------|---------|-----|---------|
| RoBERTa-base (original data) | 0.968 | 0.949 | 0.976 | 0.977 |
| RoBERTa-base + few-shot synthetic | ? | ? | ? | ? |

The hypothesis: few-shot generation produces harder, more realistic unequivalent
pairs that force the model to learn genuine semantic reasoning rather than
negation-token shortcuts. This should most visibly improve the **anatomy** domain,
which is the weakest and most likely to contain ambiguous cases the model currently
resolves via surface heuristics.


## Example

System message (equivalent):
  You are a precise paraphrasing assistant...

  Here are examples of good equivalent rewrites, as judged by a human annotator:
  
  Example 1:
    Original:  "The post is benign because it does not contain any harmful language..."
    Rewritten: "I don't find this post particularly disrespectful. There is no cursing..."

  Example 2:
    Original:  "This post does not target any specific identity..."
    Rewritten: "There is no offense to anyone, this post is completely neutral."

  ...

  Now rewrite the following explanation. Output ONLY the rewritten explanation.

    System message (the few-shot part):
  You are a sentence-rewriting assistant...

  Here are examples of genuine semantic flips, as judged by a human annotator:
  
  Example 1:
    Original:  "The post does not contain any abusive speech targeting specific
                group characteristics..."       ← AI rationale from manual labelling
    Rewritten: "I feel it's a bit discredibilizing the trans community..."  ← human version

  Example 2:
    Original:  "This post should be classified as Not Hate because..."
    Rewritten: "I think the person is trying to discredit Obama based on his skin colour..."

  ...
  
  Now rewrite the following explanation so it reaches the opposite conclusion.
  
  User message (the actual input to rewrite):
  The post uses hostile and malicious language to attack and belittle someone...
                          ← one row from your main input CSV (reformulated_full.csv)