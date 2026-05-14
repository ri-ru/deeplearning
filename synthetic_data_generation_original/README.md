# Original zero-shot generation

This is the first generation pass we ran. Each row's `explanation` column gets
rewritten twice by Qwen2.5-3B-Instruct: once as a meaning-preserving
paraphrase (`equivalent`), once as a flipped rewrite (`unequivalent`). Output
of this script is what produced `reformulated_full.csv`.

The actual file Flavia ran was not committed. The two prompts in `script.py`
and the model config are from the Telegram message on May 14.

## Run

```bash
python script.py \
    --input ../reformulated_full.csv \
    --output reformulated_zero_shot.csv \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization none --dtype float16 \
    --n-samples 50 --seed 42
```

## Known problem

The follow-up audit (`../audit_flip_rate.py`) found ~50% of the "unequivalent"
rows from this pipeline actually keep the same conclusion. The model usually
just inserts "not" or "does not" instead of flipping the reasoning. That's why
`../synthetic_data_generation_improved/` exists.
