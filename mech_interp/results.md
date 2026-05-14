# Mech-Interp Investigation: What Is the Judge Actually Learning?

**Subject.** Fine-tuned RoBERTa-base sentence-pair judge that classifies a
(model rationale, reformulated rationale) pair as `equivalent` (0) or
`unequivalent` (1). Trained on 52k synthetic pairs produced by GPT-4o where
"unequivalent" pairs were generated to have the *opposite conclusion*. The
prompt-loop audit (`audit_flip_rate.py`) showed ~50% of "unequivalent" pairs
have an obvious surface flip — frequently a negation injected into the
synthetic side. We suspect the model learned a negation shortcut rather than
"semantic equivalence of conclusions".

**Phase 1 (this doc).** Residual-stream probing at the `[CLS]` token, every
layer, two linear probes:

- `probe_equiv` — predicts the equivalence label.
- `probe_neg`   — predicts whether the synthetic side contains negation
  (regex `\b(not|n't|never|no|none|nothing|neither|nor)\b`, case-insensitive).

**Methodology.**

- Checkpoint probed: `judge_finetune/runs/smoke/checkpoints/checkpoint-XX/`
  (RoBERTa-base, 2k-subset training, 1 epoch — see "caveat" below).
- Data: 52,137 synthetic pairs (`reformulated_full.csv`) + 20 human-annotated
  pairs (`manual labelling.csv`).
- For each sample, forward pass once with `output_hidden_states=True`,
  store `[CLS]` activation at every layer (13 layers: 1 embedding + 12
  transformer blocks).
- 80/20 random split on the synthetic pool for probe train/val. Manual pairs
  are evaluated by the probes trained on synthetic — they are out-of-domain
  in two senses (different writing voice + true labels, not GPT-4o-judged).
- Logistic regression probes (sklearn), `max_iter=1000`, `C=1.0`. No PCA, no
  whitening — we want raw linear separability.

---

## Hypotheses (stated *before* looking at any numbers)

### H1 — Pure negation shortcut

`probe_neg` AUC tracks `probe_equiv` AUC layer-for-layer, and at some layer
the two probes' *per-sample predicted probabilities* correlate >0.85 on the
val split.

> **What this would mean.** The classifier head is, to first order, a
> negation detector wearing a "judge" hat. Phase 2 patching would then have
> a sharp causal story: zero-ablate the negation direction at that layer
> and watch `unequivalent` predictions collapse on synthetic but not on
> manual.

**Predicted result:** AUC curves visually overlap above some mid layer; ρ ≥ 0.85.

**Actual result:** _(filled in by `run_phase1.sh`)_

### H2 — Mixed feature ("negation is *a* signal, not *the* signal")

`probe_neg` AUC rises but plateaus below `probe_equiv` AUC; per-sample
prediction correlation peaks somewhere in 0.55–0.80.

> **What this would mean.** The shortcut basket also contains things we
> didn't probe (length ratio, sentiment-bearing tokens like "but / however",
> negation of polarity adjectives, n-gram overlap). Calling it
> "the negation shortcut" would be premature; Phase 2 would need either
> multi-feature probes or sparse feature decomposition.

**Predicted result:** `probe_neg` AUC < `probe_equiv` AUC at every layer;
gap widens at the top.

**Actual result:** _(filled in)_

### H3 — Negation is incidental

`probe_neg` AUC stays near base-rate (~AUC of the regex baseline, see
`prior_neg_correlation` below) while `probe_equiv` climbs sharply.

> **What this would mean.** The shortcut is something else entirely
> (likely conclusion-token mismatch, sentiment flip, or some
> domain-specific cue). We'd need to redesign the probe targets — e.g.,
> probe for "the post is hate" yes/no embedded in text_a vs text_b.

**Predicted result:** flat `probe_neg` curve at AUC ≈ 0.5–0.6; clear
`probe_equiv` climb.

**Actual result:** _(filled in)_

### Auxiliary — Manual transfer

If `probe_equiv` AUC on the 20 manual pairs is much lower than its synthetic
val AUC (drop > 0.20), that's independent evidence the synthetic decision
function does not transfer to human-labeled pairs — consistent with a
shortcut even if H1 fails to localize *which* shortcut.

**Predicted result:** large drop (AUC 0.95-ish on synthetic, ≤ 0.70 on manual).

**Actual result:** _(filled in)_

---

## Prior: the regex baseline

Before fitting any probe, we measure `correlation(label, contains_negation_in_synthetic)`
directly on the manifest. This is the AUC ceiling for an "always predict
unequivalent iff negation present" rule.

**Actual:** _(filled in)_

If this prior is already very high (e.g. > 0.75 AUC), the probe curves
become harder to interpret — a strong probe_neg AUC at the top of the
network could just mean the model has propagated text-level features
without doing real composition. We will report it both ways.

---

## Caveat: smoke checkpoint, not the production judge

The full-data RoBERTa-base / roberta-large-mnli checkpoints are not on this
machine (gitignored, 8.3 GB; see `EXCLUDED_FILES.md`). Phase 1 uses the
2k-sample smoke training run as a stand-in. This means:

- Probe AUC numbers are not directly comparable to the paper's reported
  judge performance — a probe AUC of 0.9 on a smoke checkpoint could be
  0.95 or 0.85 on the full one.
- The *shape* of the per-layer curves and the negation-equivalence
  correlation are the load-bearing claims. Those should be qualitatively
  preserved across training scales if the shortcut is real.
- We retrain on the full set before publishing any figure with a "the
  shortcut is at layer L" claim.

---

## Result figures and artifacts

- `figures/probe_auc_layer_sweep.png` — AUC vs layer, both probes, with regex prior as a horizontal line.
- `figures/probe_prediction_correlation.png` — per-layer pearson ρ between probe_equiv and probe_neg predicted probs.
- `results/probe_metrics.json` — full numeric table (AUC, ρ, manual AUC) per layer.
- `extracts/layer_XX.pt` — `[CLS]` residual tensors, per layer, in manifest order.
- `extracts/manifest.csv` — sample_id, source, label, contains_negation_in_synthetic, in_synthetic_not_original, text_a, text_b.

---

## What we will NOT claim from Phase 1 alone

- That zero-ablating the "negation direction" causes the judge to flip
  predictions. That is a Phase-2 patching claim and requires causal
  intervention, not just probe AUC.
- That the model "doesn't use semantic content". A high probe_equiv AUC at
  layer 12 is consistent with both pure shortcut and rich semantics; only
  the *gap* to probe_neg distinguishes them.
- Anything about other heads (e.g. roberta-large-mnli). The smoke run uses
  roberta-base, so layer indices and findings transfer only structurally.
