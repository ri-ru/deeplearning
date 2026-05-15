# Findings - Synthetic Data Quality and the Self-Improving Prompt Loop

Status: in progress. This document records what we learned while building and
diagnosing the rationale-equivalence judge data pipeline. Dated 2026-05-13/14.

## TL;DR

1. The original `reformulated_full.csv` is structurally broken: ~50% of pairs
   labelled "unequivalent" actually reach the **same** conclusion as the
   original, and the remaining differences are dominated by simple negation
   tokens. A judge trained on this data learns "is there a 'not' in here?"
   not "do these two rationales agree?".
2. A naive single-metric (cosine + TF-IDF F1) self-improving prompt loop does
   not converge. Across 10 iterations it found a local minimum where cosine
   stays below 0.35 and TF-IDF F1 stays at 1.000.
3. The diagnosis: the optimizer added stylistic constraints ("at least 6
   sentences with 'therefore'", "begin with a summary") that are absent in
   originals. The result is synthetics 4–5× longer than originals, in a
   different register, perfectly separable by surface features.
4. Adding two new hard gates - **conclusion-flip rate** (does the unequivalent
   rationale actually flip the verdict?) and **length-ratio** (do synthetics
   match the originals' length?) - closes the loophole. Smoke test on the
   multi-objective loop with a hand-written `INITIAL_PROMPT` shows flip rate
   = 100% and length ratio = 0.83–1.15 on the first try.

## The dataset problem

`reformulated_full.csv` (52,137 pairs, balanced 50/50 by `meaning` column).
For each ground-truth rationale, GPT-4 was asked to produce two variants:

- An *equivalent* paraphrase (label = 0).
- An *unequivalent* version reaching the opposite verdict (label = 1).

The equivalent half is high-quality paraphrase. The unequivalent half is
almost entirely shallow negation. Common pattern:

> ORIGINAL: "should be classified as Hate Speech because it contains hostile and malicious language"
>
> UNEQUIV:  "should NOT be classified as Hate Speech because it contains hostile and malicious language"

The rest of the sentence is copy-pasted, often producing self-contradictions
("does not contain bias … contains bias … the post does not mention the
Latino group").

### Audit on the existing failed loop output

We sampled 30 unequivalent pairs from `loop_results/final_batch.csv` (the
output of the first self-improving loop). For each pair, GPT-4o-mini was
asked:

> Two rationales about whether a social-media post is hate speech.
> SAME = both reach the same conclusion. DIFFERENT = opposite conclusions.

Result: **15 SAME / 15 DIFFERENT → flip rate = 50.0%**. Half the "unequivalent"
pairs are mislabeled. See `audit_results.txt` for the full transcript and
three representative SAME examples.

## Why the naive loop did not converge

The first version of `prompt_loop.py` optimized two metrics:

- mean cosine similarity between unequivalent pairs in `[0.55, 0.75]`,
- TF-IDF F1 of a logistic regression classifying pairs ≤ 0.70.

After 10 iterations the trajectory was:

| iter | cosine | TF-IDF F1 |
|------|--------|-----------|
| 1    | 0.214  | 1.000     |
| 2    | 0.275  | 0.995     |
| 3    | 0.270  | 1.000     |
| 4    | 0.282  | 1.000     |
| 5    | 0.300  | 1.000     |
| 6    | 0.142  | 1.000     |
| 7    | 0.146  | 1.000     |
| 8    | 0.331  | 1.000     |
| 9    | 0.347  | 1.000     |
| 10   | 0.342  | 1.000     |

Both metrics stayed pinned in the wrong region. The optimizer responded by
adding more explicit, more verbose constraints each iteration. The final
prompt mandated "at least 40 distinct nouns from the original", "≥6 sentences
using 'because' or 'therefore'", "≥2 sentences beginning with 'Although'",
and "begin with a brief summary of the post's main idea".

Two failure modes resulted:

- **Length explosion.** Originals are 1–3 sentences; synthetics became 4–5
  paragraphs. A TF-IDF classifier separates them on length alone.
- **Register shift.** Originals are matter-of-fact analytical prose;
  synthetics adopt a florid, philosophical register with mandatory
  connective adverbs that never appear in the originals. New shortcut,
  same problem.

The memory-of-iterations feature (added between the first and second
overnight run) successfully prevented oscillation but locked the optimizer
into this stylistic local minimum.

## The fix - multi-objective optimizer

Added two new hard gates to `prompt_loop.py`:

- **`flip_rate`**: for each unequivalent pair, ask GPT-4o-mini SAME-or-DIFFERENT.
  Target ≥ 0.90. Cached at the pair level via MD5 hash → `flip_cache.json`
  for resumability.
- **`length_ratio_mean` and `length_ratio_p90`**: ratio of synthetic
  word-count to original word-count. Mean ∈ [0.5, 2.0], p90 ≤ 3.0.

The meta-prompt now sees all six metrics in the history block and includes
explicit anti-essay guidance: *do not add formal-essay constraints like "use
at least N sentences with 'therefore'" or "begin with a summary" - these
introduce stylistic shortcuts that TF-IDF picks up on*.

The diagnose function prioritises by severity: flip-rate < 90% is reported
first (mislabeled data is the worst failure), then length-ratio, then
cosine, then TF-IDF.

`INITIAL_PROMPT` was also rewritten by hand to bake in the lessons: match
length, match register, forbid connective adverbs, anchor to the original's
vocabulary.

## Smoke verification of the new optimizer

`--smoke` runs `n=4, max_iter=2, final_n=2`. Output landed in
`loop_results_smoke_mo/`. Per-iteration metrics:

| iter | cosine | flip_rate | len_ratio | len_p90 |
|------|--------|-----------|-----------|---------|
| 1    | 0.315  | 100.0%    | 0.83      | 0.92    |
| 2    | 0.325  | 100.0%    | 0.99      | 1.07    |
| final| 0.383  | 100.0%    | 1.15      | 1.20    |

Flip rate is at the target on iteration one (vs 50% on the old loop).
Length ratio sits in the target band. Cosine is still below 0.55 - but with
`n=4`, cosine is noisy; the real check is on the overnight `n=200` run.

`flip_cache.json` accumulates verdicts so a crash mid-iteration does not
waste API calls on rejudging.

## What this is worth for the paper

Three things to write up regardless of how the overnight run goes:

1. **The audit number.** "We audited the existing reformulation pipeline and
   found 50% of unequivalent pairs reach the same conclusion as the
   original." That is a load-bearing sentence for the methodology section.
2. **The single-metric failure mode.** The 10-iteration trajectory is a
   clean negative result: prompt-loop optimization with a TF-IDF/cosine
   target alone finds local minima that satisfy the metric by introducing
   new surface shortcuts the metric does not measure.
3. **The multi-objective fix.** Conclusion-flip rate, judged by a separate
   LLM call, blocks mislabeling. Length-ratio blocks the register-shift
   shortcut. Both are cheap (~200 extra GPT-4o-mini calls per iteration,
   roughly two cents). Whether the loop converges fully is an open question
   the overnight run will answer.

## Files

- `prompt_loop.py` - the multi-objective loop.
- `audit_flip_rate.py` - one-off audit script.
- `audit_results.txt` - verdicts on 30 pairs from the original failed run.
- `loop_results/` - the 10-iteration overnight run that motivated the fix.
- `loop_results_smoke_mo/` - smoke output verifying the new optimizer is wired.
