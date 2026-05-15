# Few-shot generation

Round two of the synthetic data. The original `reformulated_full.csv` had the
50% mislabeling problem (see `../audit_results.txt`): GPT-4 mostly just
inserted "NOT" instead of flipping reasoning. The prompt-tuning loop
(`../prompt_loop.py`) couldn't fix it - cosine and TF-IDF gave no signal that
pairs were mislabeled.

Fix: inject the 20-row `manual_labelling.csv` into the generation prompt as
few-shot examples. Qwen sees concrete examples of what a real semantic flip
looks like before generating.

## Cluster

V100-SXM2-32GB, CUDA 13.0, compute capability 70. No bf16. AWQ needs cap 75+
so it doesn't work here; GPTQ works.

## Run

```bash
# 7B GPTQ (preferred)
python script.py --input ../reformulated_full.csv \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --quantization gptq --dtype float16 --n-samples 0

# 3B no quantization (fallback)
python script.py --input ../reformulated_full.csv \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization none --dtype float16 --n-samples 0
```

## Files

- `script.py` - generation
- `manual_labelling.csv` - 20 manual pairs (also used as the few-shot seed)
- `reformulated (1-4).csv` - the 4 iterations; only #4 is what RoBERTa trained on
- `PROMPT_HISTORY.md` - what we know about each iteration

## Baseline to beat

| model | overall | anatomy | gab | toxigen |
|-------|---------|---------|-----|---------|
| RoBERTa-base on original | 0.968 | 0.949 | 0.976 | 0.977 |
| + few-shot synthetic | 0.968 | 0.939 | 0.977 | 0.983 |

50 rows of augment isn't enough to dent the negation shortcut. Anatomy
actually got slightly worse. This is what motivated the mech-interp probes
in `../mech_interp/`.
