# Prompt history

I iterated the few-shot prompt 4 times. Only the final one is in `script.py`.

| iter | CSV | rows |
|------|-----|------|
| 1 | `reformulated (1).csv` | 50 |
| 2 | `reformulated (2).csv` | 50 |
| 3 | `reformulated (3).csv` | 50 |
| 4 | `reformulated (4).csv` | 50 (this is what RoBERTa trained on) |

Earlier prompts weren't saved. From the chat: each iteration tweaked the
instruction to avoid Qwen falling into patterns ("I agree" / "I disagree"
prefixes etc.). Iteration 4 has an explicit "do NOT say 'I agree'" guard
in the system prompt.

If you want to regenerate iter 4:

```bash
python script.py \
    --input ../reformulated_full.csv \
    --output reformulated_4_regen.csv \
    --manual_labels manual_labelling.csv \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq --dtype float16 \
    --n-samples 50 --seed 42
```

Iters 1-3 can't be regenerated without the prompts.
