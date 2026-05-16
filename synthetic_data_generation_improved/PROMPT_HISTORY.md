# Prompt-Engineering History

Flavia iterated the few-shot Qwen generation prompt four times. Per the
project chat (2026-05-14):

> "I did 4 generations of prompt. I could find the data for each generation
> but I don't recall well each strategy I tried."

The CSVs are preserved here. The exact prompt used to produce iterations 1,
2, and 3 was not committed and is not recoverable from the script alone —
`script.py` contains only the final (iteration 4) prompt.

## What we have

| Iteration | CSV | Rows | Prompt source |
|---|---|---|---|
| 1 | `reformulated (1).csv` | 50 | Not preserved. Only the CSV output exists. |
| 2 | `reformulated (2).csv` | 50 | Not preserved. |
| 3 | `reformulated (3).csv` | 50 | Not preserved. |
| 4 | `reformulated (4).csv` | 50 | Encoded in `script.py:32-65` (`build_system_prompt_equivalent` / `build_system_prompt_unequivalent`). This is the iteration used to train the headline RoBERTa run in `roberta_training/runs/roberta_fewshot/`. |

## What we know about the progression (from the chat)

> "I assessed each output given by a prompt using the instruction + few
> shot and changed slightly the instruction to avoid too obviously
> patterns. For instance, if a certain type of words were coming again and
> again such as I agree/I disagree then I reworked on the prompt to make
> sure we don't have the pitfalls again of Qwen learning patterns instead
> of semantics."

So at least one observable change between iterations was an explicit
instruction in iteration 4 to *not* preface the comment with "I agree" or
"I disagree" (visible in `script.py` as: *"do NOT say 'I agree', just
state your own view directly about the post"*). The other three iterations
likely lack that guard.

## Reproducing what we have

You can regenerate iteration 4 from scratch with:

```bash
python script.py \
    --input ../reformulated_full.csv \
    --output reformulated_4_regen.csv \
    --manual_labels manual_labelling.csv \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq --dtype float16 \
    --n-samples 50 --seed 42
```

Iterations 1–3 cannot be regenerated without recovering the earlier
prompts from Flavia.
