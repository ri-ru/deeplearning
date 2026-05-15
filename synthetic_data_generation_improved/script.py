import argparse
import random
import pandas as pd
from vllm import LLM, SamplingParams


def load_fewshot_examples(manual_labels_path: str) -> dict:
    """Load manual labelling CSV and return few-shot examples split by label."""
    df = pd.read_csv(manual_labels_path, skiprows=1, header=0)
    df.columns = ["post", "explanation", "human_comment", "label"] + list(df.columns[4:])
    df = df[["post", "human_comment", "label"]].dropna(subset=["post", "human_comment", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    return {
        "equivalent":   df[df["label"] == "equivalent"][["post", "human_comment"]].to_dict("records"),
        "unequivalent": df[df["label"] == "unequivalent"][["post", "human_comment"]].to_dict("records"),
    }


def _format_shots(examples: list[dict], n: int, seed: int) -> str:
    rng = random.Random(seed)
    picked = rng.sample(examples, min(n, len(examples)))
    shots = ""
    for i, ex in enumerate(picked, 1):
        shots += (
            f"\nExample {i}:\n"
            f"  Post:    {ex['post'].strip()}\n"
            f"  Comment: {ex['human_comment'].strip()}\n"
        )
    return shots


def build_system_prompt_equivalent(examples: list[dict], n_shots: int = 3, seed: int = 0) -> str:
    shots = _format_shots(examples, n_shots, seed)
    return (
        "You are a human annotator labelling social media posts for hate speech. "
        "You will be given a post and its AI classification. "
        "Write a short, casual, first-person comment stating YOUR OWN view of the post "
        "that supports the AI classification. "
        "Write as if you just read the post yourself - do NOT reference the AI, "
        "do NOT say 'I agree', just state your own view directly about the post. "
        "1-3 sentences, informal tone.\n\n"
        "Here are examples of how human annotators write their comments:"
        f"{shots}\n"
        "---\n"
        "Now write your own comment. Do NOT copy from the examples above. "
        "Output ONLY your comment, nothing else."
    )


def build_system_prompt_unequivalent(examples: list[dict], n_shots: int = 3, seed: int = 0) -> str:
    shots = _format_shots(examples, n_shots, seed)
    return (
        "You are a human annotator labelling social media posts for hate speech. "
        "You will be given a post and its AI classification. "
        "Write a short, casual, first-person comment stating YOUR OWN view of the post "
        "that DISAGREES with the AI classification. "
        "Write as if you just read the post yourself - do NOT reference the AI, "
        "do NOT say 'I disagree', just state your own view directly about the post. "
        "1-3 sentences, informal tone.\n\n"
        "Here are examples of how human annotators write their comments:"
        f"{shots}\n"
        "---\n"
        "Now write your own comment. Do NOT copy from the examples above. "
        "Output ONLY your comment, nothing else."
    )


def build_prompts(tokenizer, sentences, system_prompts):
    prompts = []
    for s, sys in zip(sentences, system_prompts):
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": str(s)},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="hatecot_final_D3 (2).csv",
        help="Path to input CSV/XLSX",
    )
    p.add_argument(
        "--output",
        default="reformulated.csv",
        help="Path to output CSV. Original columns are preserved with an added 'synthetic' column.",
    )
    p.add_argument(
        "--column",
        default="explanation",
        help="Column in input to reformulate",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of rows to sample. Use -1 (or 0) to reformulate the full dataset.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--manual_labels",
        default="manual_labelling.csv",
        help="Path to manual labelling CSV used as few-shot examples in the prompt.",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="AWQ 4-bit fits ~10GB. Use Qwen2.5-72B-Instruct on a big GPU.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "auto"],
        help="Use float16 on V100 (no bf16 hardware).",
    )
    p.add_argument(
        "--quantization",
        default="awq",
        help="Matches the model. Pass 'none' for unquantized models.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args()

    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise ValueError(
            f"Column '{args.column}' not in {list(df.columns)}"
        )

    df = df[df[args.column].notna()]
    df = df[df[args.column].astype(str).str.strip() != ""]
    if args.n_samples and args.n_samples > 0:
        sample = df.sample(
            n=min(args.n_samples, len(df)), random_state=args.seed
        ).reset_index(drop=True)
    else:
        sample = df.reset_index(drop=True)
    originals = (
        "Post: " + sample["post"].astype(str) +
        "\nAI classification: " + sample["label"].astype(str)
    ).tolist()

    quantization = None if args.quantization.lower() == "none" else args.quantization
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        quantization=quantization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    fewshot = load_fewshot_examples(args.manual_labels)
    print(f"[few-shot] loaded {len(fewshot['equivalent'])} equivalent and "
          f"{len(fewshot['unequivalent'])} unequivalent examples from {args.manual_labels}")

    labels = [
        "equivalent" if i % 2 == 0 else "unequivalent"
        for i in range(len(originals))
    ]
    system_prompts = [
        build_system_prompt_equivalent(fewshot["equivalent"], seed=i)
        if l == "equivalent"
        else build_system_prompt_unequivalent(fewshot["unequivalent"], seed=i)
        for i, l in enumerate(labels)
    ]
    prompts = build_prompts(tokenizer, originals, system_prompts)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    synthetic = [o.outputs[0].text.strip() for o in outputs]

    out_df = sample.copy()
    out_df["synthetic"] = synthetic
    out_df["meaning"] = labels
    if args.output.lower().endswith((".xlsx", ".xls")):
        out_df.to_excel(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
