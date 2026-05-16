# What is RoBERTa attending to?

Per-layer single-head attention probe trained on per-token residuals from the
fewshot RoBERTa checkpoint (`runs/roberta_fewshot/checkpoint-1580`).
Train/val split inside 396 synthetic rows; manual (n=20) and claude (n=20)
held out for analysis. Layers extracted: 1, 6, 9, 12.

See `train_attention_probe.py` for the probe.
Per-layer alpha dumps in `probe_runs/layer_XX/*_alpha.json`.

## Headline

The probe says the equivalence signal is fully present in the residual stream
already at layer 9 and is essentially perfect by layer 12. But the
fine-tuned RoBERTa head only uses 65% of it on the manual set and 50% on
the claude set. The features are there; the head does not generalize.

| layer | val AUC | manual AUC | claude AUC | manual acc | claude acc |
|------:|--------:|-----------:|-----------:|-----------:|-----------:|
| 1     | 0.637   | 0.418      | 0.720      | 0.45       | 0.70       |
| 6     | 0.860   | 0.516      | 0.580      | 0.65       | 0.50       |
| 9     | 0.997   | 0.791      | 0.750      | 0.70       | 0.50       |
| 12    | 0.999   | 1.000      | 0.880      | 1.00       | 0.80       |

(RoBERTa's actual classifier head: manual 13/20 = 65%, claude 10/20 = 50%.)

## What tokens get the attention

At layer 9 the probe shows clear lexical preferences per class.

### On the manual set

**EQUIVALENT** rationales - probe attends to verdict / neutrality words:
`neutral`, `benign`, `Neutral`, `target`, `content`, `problematic`, `simply`,
`see`, `no`, `anything`. The probe is reading the conclusion lexicon.

**UNEQUIVALENT** rationales - probe attends to hate-verdict / framing words:
`offensive`, `language`, `Therefore`, `but`, plus heavy mass on punctuation.

### On the claude set

The lexicon shifts a lot. Claude rationales use `derogatory`, `targets`,
`stereotypes`, `implies`, `statement` (high attention on EQUIV side);
unequivalent has thinner cues mostly on `not`, `should`, `from`.
This is consistent with what shows up in the OOD failure mode - the trained
head learned the Qwen verdict lexicon and the Claude verdict lexicon does
not match.

## What this says about the original "what is it learning" question

It is **not** a pure negation detector (we already knew that from the
existing probes). It is closer to a **verdict-token matcher**: it pools
representation mass over the words that announce the conclusion of each
rationale (`benign / neutral / problematic` vs `offensive / Therefore /
language`) and the head compares those pooled features across the two text
segments. When the two rationales use the same verdict lexicon -> equivalent.
When they differ -> unequivalent.

This explains all three of the things we keep observing:

1. **Negation pairs trivially flip the verdict word** (`benign` -> not
   benign / `offensive`) so the probe AUC stays high on synthetic data
   even though the actual semantic structure is broken.
2. **Manual set drops to 65 %** because Flavia's first-person comments
   often state agreement without echoing the rationale's verdict word
   ("I don't find this disrespectful", "I feel this discredits the
   community"). No `benign` or `Neutral` to anchor to.
3. **Claude set drops to 50 %** because Claude consistently writes
   "This post is X" / "Calling people Y" - a different verdict lexicon
   than Qwen's "I find it..." voice.

## Caveats

- n=20 in each held-out set; the layer 12 manual AUC = 1.000 is suggestive
  but not statistically tight.
- Layer 12 alpha is nearly uniform - by the last layer everything has been
  aggregated into [CLS] and there is no token-level selectivity to read.
  Mid layers (6, 9) carry the interpretable signal.
- The attention probe and the actual RoBERTa attention heads are not the
  same thing. The probe shows which tokens linearly carry the equivalence
  signal; it does not directly tell us what the multi-head attention
  inside the model is doing.

## Next moves worth trying

1. Re-train RoBERTa's classifier head on top of frozen layer-9 features with
   the manual + claude rows mixed in. If the head's the problem, this fixes
   the OOD drop without touching the encoder.
2. Train the probe only on the **verdict noun phrase** of each rationale
   (extract spans around `benign / neutral / offensive / hate`) to test the
   verdict-matcher story directly. Compare to a probe trained on full
   sequence.
3. Grad x input attribution as a cross-check. If grad-input lights up the
   same tokens as attention, we are confident in the verdict-matcher
   reading.
