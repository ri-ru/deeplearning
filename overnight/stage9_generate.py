"""Stage 9a: generate 200 rationale-pair paraphrases per LLM.

Same 100 source posts (sampled from the OOD test split) are sent to every
generator. For each post we generate two versions: one EQUIVALENT (paraphrase
preserving meaning) and one UNEQUIVALENT (paraphrase flipping the verdict).
This gives 200 pairs per generator on the same content distribution -- the
only thing that varies is writing style.

Source posts are sampled once and cached so all generators see the same set,
and any rerun is reproducible.

Writes one CSV per generator: overnight/cross_gen/<slug>.csv with columns
    post, explanation, synthetic, meaning, post_id
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import LABEL2ID, REPO, TEST_DOMAINS  # noqa: E402

OUT_DIR = REPO / "overnight" / "cross_gen"
POSTS_CSV = OUT_DIR / "_source_posts.csv"
META_PATH = OUT_DIR / "_meta.json"

# Generators available via OpenRouter (openai-compatible API).
# `slug` = file name; `model` = openrouter model id.
GENERATORS = {
    "openai_gpt4omini":  {"model": "openai/gpt-4o-mini",                "provider": "openrouter"},
    "anthropic_haiku":   {"model": "anthropic/claude-haiku-4.5",        "provider": "openrouter"},
    "meta_llama3_8b":    {"model": "meta-llama/llama-3.1-8b-instruct",  "provider": "openrouter"},
    "deepseek_v3":       {"model": "deepseek/deepseek-chat-v3.1",       "provider": "openrouter"},
    "qwen25_72b":        {"model": "qwen/qwen-2.5-72b-instruct",        "provider": "openrouter"},
    "mistral_7b":        {"model": "mistralai/mistral-7b-instruct",     "provider": "openrouter"},
}

SYSTEM_PROMPT_EQUIVALENT = (
    "You are a precise paraphrasing assistant. Rewrite the user's sentence in "
    "different words while preserving the EXACT meaning, tone, named entities, "
    "numbers, and verdict (hate/not hate). Do not add, omit, or infer information. "
    "Output ONLY the reformulated sentence, no preamble, no quotes, no commentary."
)
SYSTEM_PROMPT_UNEQUIVALENT = (
    "You are a sentence-rewriting assistant. Produce a sentence whose MEANING is "
    "clearly DIFFERENT from the original. Keep similar topic, style, and length, "
    "but apply at least ONE: flip the verdict (hate <-> not hate), swap a key "
    "entity/target group, invert sentiment/stance, or change a key fact. "
    "A reader must immediately see the two sentences assert different things. "
    "Output ONLY the rewritten sentence, no preamble, no quotes, no commentary."
)


def sample_source_posts(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Sample n distinct posts from the OOD test split."""
    df = pd.read_csv(REPO / "reformulated_full.csv")
    df = df.dropna(subset=["explanation", "post", "domain", "uid"]).copy()
    df = df[df["domain"].isin(TEST_DOMAINS)].copy()
    # Sample by unique post (uid) so we don't pick two reformulations of the same source
    by_post = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(by_post)), k=min(n, len(by_post)))
    return by_post.iloc[indices][["uid", "post", "explanation", "domain"]].reset_index(drop=True)


def call_openrouter(model: str, system: str, user: str, max_tokens: int = 400) -> str:
    """OpenAI-compatible API hosted at openrouter.ai."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
        top_p=0.9,
    )
    return resp.choices[0].message.content.strip()


def generate_one(slug: str, model: str, posts: pd.DataFrame, out_path: Path,
                 resume: bool = True, max_retries: int = 3) -> None:
    """Generate two paraphrases per source post (one equiv, one unequiv)."""
    print(f"\n=== generating {slug}  ({model}) ===", flush=True)
    seen_ids: set[tuple[str, str]] = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        for _, r in existing.iterrows():
            seen_ids.add((str(r["post_id"]), r["meaning"]))
        print(f"  resume: {len(seen_ids)} pairs already generated", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    f = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=["post", "explanation", "synthetic", "meaning", "post_id", "domain"])
    if write_header:
        writer.writeheader()

    t0 = time.time()
    n_done = 0
    n_failed = 0
    for _, row in posts.iterrows():
        post_id = str(row["uid"])
        explanation = str(row["explanation"])
        for label in ("equivalent", "unequivalent"):
            if (post_id, label) in seen_ids:
                continue
            system = SYSTEM_PROMPT_EQUIVALENT if label == "equivalent" else SYSTEM_PROMPT_UNEQUIVALENT
            last_err = None
            for attempt in range(max_retries):
                try:
                    text = call_openrouter(model, system, explanation)
                    if text:
                        break
                except Exception as e:
                    last_err = e
                    time.sleep(1 + attempt * 2)
            else:
                n_failed += 1
                print(f"  [{post_id}/{label}] failed: {last_err}", flush=True)
                continue
            writer.writerow({
                "post": str(row["post"]),
                "explanation": explanation,
                "synthetic": text,
                "meaning": label,
                "post_id": post_id,
                "domain": row["domain"],
            })
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  {slug}: {n_done} pairs  ({rate:.1f}/s)", flush=True)

    f.close()
    print(f"[{slug}] done: {n_done} new pairs, {n_failed} failed, "
          f"{time.time()-t0:.0f}s, file={out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_posts", type=int, default=100,
                        help="Distinct source posts (each becomes 2 pairs -> 200 total per generator).")
    parser.add_argument("--only", default=None,
                        help="Comma-sep generator slugs to run; default = all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[stage9] OPENROUTER_API_KEY not set; abort.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if POSTS_CSV.exists():
        posts = pd.read_csv(POSTS_CSV)
        print(f"[stage9] reusing cached source posts: {POSTS_CSV} (n={len(posts)})")
    else:
        posts = sample_source_posts(args.n_posts, args.seed)
        posts.to_csv(POSTS_CSV, index=False)
        print(f"[stage9] sampled {len(posts)} source posts -> {POSTS_CSV}")

    META_PATH.write_text(json.dumps({
        "n_posts": int(len(posts)),
        "pairs_per_generator": int(2 * len(posts)),
        "generators": GENERATORS,
        "seed": args.seed,
    }, indent=2))

    only = set(s.strip() for s in args.only.split(",")) if args.only else set(GENERATORS.keys())

    for slug, cfg in GENERATORS.items():
        if slug not in only:
            continue
        out_path = OUT_DIR / f"{slug}.csv"
        try:
            generate_one(slug, cfg["model"], posts, out_path)
        except KeyboardInterrupt:
            print(f"[{slug}] interrupted")
            raise
        except Exception as e:
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n[stage9] generation phase done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
