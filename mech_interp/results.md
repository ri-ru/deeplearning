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

**Actual result:** peak ρ(probe_equiv, probe_neg) on synthetic val = 0.350 at layer 5. At that same layer probe_equiv AUC = 0.878 and probe_neg AUC = 0.934. ρ does not exceed 0.85, so H1 is **not supported** at this threshold.

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

**Actual result:** at the top layer (layer 12) probe_equiv AUC = 0.995 and probe_neg AUC = 0.976; the gap is 0.019. Peak probe_neg AUC across all layers = 0.976 (layer 12); peak probe_equiv AUC = 0.995 (layer 12). 

### H3 — Negation is incidental

`probe_neg` AUC stays near base-rate (~AUC of the regex baseline, see
`prior_neg_correlation` below) while `probe_equiv` climbs sharply.

> **What this would mean.** The shortcut is something else entirely
> (likely conclusion-token mismatch, sentiment flip, or some
> domain-specific cue). We'd need to redesign the probe targets — e.g.,
> probe for "the post is hate" yes/no embedded in text_a vs text_b.

**Predicted result:** flat `probe_neg` curve at AUC ≈ 0.5–0.6; clear
`probe_equiv` climb.

**Actual result:** probe_neg AUC range across layers = [0.500, 0.976], vs regex-only prior 0.609. probe_neg climbs well above the regex prior, so negation is encoded as more than a surface lexical match.

### Auxiliary — Manual transfer

If `probe_equiv` AUC on the 20 manual pairs is much lower than its synthetic
val AUC (drop > 0.20), that's independent evidence the synthetic decision
function does not transfer to human-labeled pairs — consistent with a
shortcut even if H1 fails to localize *which* shortcut.

**Predicted result:** large drop (AUC 0.95-ish on synthetic, ≤ 0.70 on manual).

**Actual result:** best probe_equiv AUC on synthetic val = 0.995 (layer 12); best probe_equiv AUC on manual (n=20) = 0.802 (layer 11). Drop = 0.192. Synthetic→manual drop is < 0.20 — the manual eval does not refute on its own that the model learned semantics.

---

## Observed surprises (not covered by H1/H2/H3)

### S1 — `probe_neg` is *higher than* `probe_equiv` at layers 1–6

Layers 1–6: probe_neg AUC = 0.85→0.94, probe_equiv AUC = 0.75→0.93.
The crossover is at layer 7 (probe_equiv 0.964 > probe_neg 0.943) and
the equivalence probe pulls further ahead through layer 12. This is the
right *shape* for "negation is read off as a near-lexical feature early,
then equivalence is computed on top of it" — which is a strictly weaker
shortcut claim than H1 (it does not require the two probes to agree
per-sample, only that the equivalence representation builds on a
negation representation that emerged first).

### S2 — High AUC on both probes, but per-sample disagreement (ρ ≤ 0.35)

The two probes hit AUC 0.995 and 0.976 at layer 12 but their *predicted
probabilities* correlate only ρ = 0.27–0.35 across all middle and top
layers. Reading this carefully: the model has high-quality linear
representations of *both* features in [CLS], stored in **different
directions**. Whatever direction the equivalence probe found is not
collinear with the negation direction — so "the model is doing
negation detection in disguise" is not the right one-line story.

What the model is using *causally* for the final classifier head is
still open; probe AUC is correlational. This is the Phase-2 question.

### S3 — Mid-layer probe_equiv on manual is **below chance** (anti-correlated)

| layer | probe_equiv AUC (manual, n=20) |
|---|---|
| 3 | 0.451 |
| 4 | 0.396 |
| 5 | 0.330 |
| 6 | 0.495 |
| 7 | 0.297 |

With n=20 every individual number is noisy, but five consecutive layers
sitting at 0.30–0.50 is qualitatively striking. The synthetic-fit
direction at mid layers is *anti-correlated* with the human label —
not merely non-transferring. This is the strongest standalone signal in
Phase 1 that the model is leaning on something that *flips sign* between
the synthetic and manual distributions, then partially recovers at the
top layers (manual AUC 0.69, 0.65, 0.64, 0.80, 0.64 at layers 8–12).

### S4 — probe_neg transfers to manual better than probe_equiv at the top layers

At layers 9–12 probe_neg AUC on manual is 0.60, 0.60, 0.70, 0.70 vs
probe_equiv 0.65, 0.64, 0.80, 0.64. Layer-11 manual is the only place
probe_equiv visibly beats probe_neg on the manual set; everywhere else
they are close or probe_neg leads. The "is there negation in text_b"
direction is the most distribution-robust feature we found.

---

## Prior: the regex baseline

Before fitting any probe, we measure `correlation(label, contains_negation_in_synthetic)`
directly on the manifest. This is the AUC ceiling for an "always predict
unequivalent iff negation present" rule.

**Actual:** regex-only AUC on the synthetic pool = 0.609 (negation regex on text_b alone predicts the equivalence label this well).

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


---
<!-- VERDICT -->

## Summary verdict (auto-filled by finalize_results.py)

- Peak per-sample probe correlation: ρ = 0.350 at layer 5 → H1 not supported at the ρ>0.85 threshold.
- Peak probe_equiv AUC = 0.995 (layer 12); peak probe_neg AUC = 0.976 (layer 12); gap small (≤0.05) → H2 most consistent with data.
- Synthetic→manual probe_equiv AUC drop: 0.192 (synthetic 0.995 → manual 0.802).
- Regex prior: AUC = 0.609.

**Phase-2 hook (revised in light of S1–S4).**

The clean H1 story ("negation IS the shortcut at layer L") did not
materialize: AUCs are similar but predictions disagree per-sample. So
the Phase-2 plan should not be "patch the negation direction and watch
labels flip" — that test would likely come back ambiguous.

Better Phase-2 targets, in order:

1. **Attention pattern audit** at layers 6→7 (the crossover where
   probe_equiv overtakes probe_neg). Look for heads whose [CLS]
   attention concentrates on negation tokens vs polarity-bearing
   adjectives ("hateful", "offensive", "benign", "not", "no").
2. **Activation patching with the *equiv* probe direction** (not the
   negation direction). Replace the layer-L residual stream's projection
   along the probe_equiv weight vector with that from a corrupted pair.
   Measure label flip rate on (a) synthetic and (b) manual.
3. **Multi-feature probes**: add probes for (sentiment-flip in text_b
   conclusion-bearing words), (length ratio bucket), and (n-gram overlap
   bucket). If any of these matches the *direction* of probe_equiv
   better than probe_neg does at the top layers, that's a better
   shortcut candidate.
4. **Re-run extraction on the full RoBERTa-base checkpoint** (not the
   smoke run) once we have it back from compute. The shape S1 (early
   negation, late equiv) and the manual collapse S3 should reproduce or
   the claims weaken.

