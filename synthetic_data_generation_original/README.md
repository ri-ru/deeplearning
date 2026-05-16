# Synthetic Data Generation — Zero-Shot (Original Pipeline)

This is the **first** synthetic data generation pipeline used in the project.
The script reads each row of an input CSV (e.g. `hatecot_final_D3 (2).csv` or
`reformulated_full.csv`), feeds the `explanation` column to a Qwen2.5-3B
instruction-tuned model, and rewrites it twice — once asking for a meaning-
preserving paraphrase (`equivalent`), once asking for a meaning-flipped
rewrite (`unequivalent`).

This pipeline produced the original 52,137-row `reformulated_full.csv` that
sits at the repo root.

## Reconstruction notice

The actual file Flavia ran was not committed to the repo at the time of the
original generation. The script in this folder is reconstructed from
Flavia's Telegram message (2026-05-14):

- The two system prompts (`SYSTEM_PROMPT_EQUIVALENT` and
  `SYSTEM_PROMPT_UNEQUIVALENT`) are pasted verbatim from that message.
- The model (`Qwen/Qwen2.5-3B-Instruct`), `--quantization none`, and
  `--dtype float16` are taken from the same message.
- The script wrapper (argparse, vLLM batch generation, alternating labels)
  mirrors `../synthetic_data_generation_improved/script.py` for consistency.

If Flavia recovers the original file, it should replace this one; the
prompts and config above should match.

## Limitations of this pipeline

The follow-up audit in `../audit_flip_rate.py` found that **~50% of the
"unequivalent" rows produced by this pipeline actually reach the same
conclusion as the original rationale**. The model frequently took the
shortcut of inserting "not" or "does not" rather than producing a genuine
semantic flip.

Concrete example from `../audit_results.txt`:

> ORIGINAL : "should be classified as Hate Speech because it contains hostile language"
> SYNTHETIC: "should NOT be classified as Hate Speech because it contains hostile language"

A RoBERTa model trained on data with this failure mode learns to detect the
presence of a negation token rather than genuine semantic disagreement —
which is the core problem the mech-interp probes in `../mech_interp/` were
designed to diagnose.

The follow-up few-shot pipeline in
`../synthetic_data_generation_improved/` is the response to this finding.

## How to run

```bash
python script.py \
    --input ../reformulated_full.csv \
    --output reformulated_zero_shot.csv \
    --column explanation \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization none \
    --dtype float16 \
    --n-samples 50 --seed 42
```

Output columns: original input columns + `synthetic` (the rewritten text) +
`meaning` (`equivalent` or `unequivalent`).
