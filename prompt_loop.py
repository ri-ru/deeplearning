"""
Self-improving prompt loop for generating hard unequivalent rationale pairs.

Each iteration:
  1. Sample 500 originals from the CSV
  2. Generate synthetic unequivalent rationales using current_prompt + gpt-4o-mini
  3. Evaluate: cosine sim distribution + TF-IDF F1
  4. If targets met: stop
  5. Feed worst examples to gpt-4o to improve the prompt
  6. Repeat

Targets:
  - Unequivalent cosine sim mean  <= 0.68
  - Unequivalent cosine sim > 0.9 <= 15%
  - TF-IDF F1                     <= 0.65

Usage:
  export OPENAI_API_KEY=sk-...
  python prompt_loop.py --csv reformulated_full.csv --max_iter 10 --n 500
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

# ── targets ──────────────────────────────────────────────────────────────────
TARGET_SIM_MIN   = 0.55   # unequivalent pairs must not be too different
TARGET_MEAN_SIM  = 0.68   # unequivalent pairs must not be too similar
TARGET_PCT_HIGH  = 0.15   # fraction with sim > 0.9
TARGET_TFIDF_F1  = 0.65

# ── initial generation prompt ─────────────────────────────────────────────────
INITIAL_PROMPT = """\
You are given a hate-speech classification rationale written by an annotator.
Your task: write an UNEQUIVALENT rationale for the same social media post.

Rules:
- Reach a DIFFERENT classification label (e.g. if the original says "Not Hate", conclude "Hate", and vice versa)
- Use DIFFERENT reasoning and evidence — do not simply negate or copy the original sentences
- Vary your vocabulary significantly: avoid repeating the original's phrases word-for-word
- Write a fluent, standalone explanation at a similar length to the original
- Do NOT reference the original rationale or say things like "unlike the above"

Original rationale:
{explanation}

Write only the new rationale, nothing else."""

# ── meta-prompt for prompt optimisation ──────────────────────────────────────
META_PROMPT = """\
You are improving a data-generation prompt used to create "unequivalent" hate-speech rationale pairs.

Goal: the generated synthetic rationale should discuss the SAME social media post as the original but:
  - Reach a different classification label
  - Use different reasoning — NOT a wholesale rewrite about a completely different topic
  - Share enough vocabulary with the original that a TF-IDF cosine similarity between the pair is 0.55–0.75
  - NOT be trivially detectable as unequivalent (either by near-identical text OR by completely different topic)

The ideal synthetic looks like: same post, same domain vocabulary, different interpretation and conclusion.

Current prompt:
<<<
{current_prompt}
>>>

Evaluation results after {n} samples:
  - TF-IDF cosine similarity mean:  {mean_sim:.3f}  (target 0.55–0.75)
  - % pairs with cosine sim > 0.9:  {pct_high:.1%}  (target ≤ {target_pct})
  - TF-IDF classifier F1:           {tfidf_f1:.3f}  (target ≤ {target_f1})

Diagnosis guidance:
  - If cosine sim mean is BELOW 0.55: synthetics are too different from originals — they read like
    completely unrelated text rather than an alternative interpretation of the same post.
    Fix: instruct the model to explicitly discuss the SAME post, reference the same key phrases or
    topics from the original, just reinterpret them differently.
  - If cosine sim mean is ABOVE 0.75: synthetics are too similar — mostly copy-paste with a flipped verdict.
    Fix: instruct the model to use different sentence structures and paraphrase key claims.

Current problem: {diagnosis}

Here are 5 example pairs to illustrate the issue:

{failure_examples}

Write an improved generation prompt that fixes the diagnosed problem above.
Output ONLY the improved prompt text (the {{explanation}} placeholder must remain), nothing else."""


def compute_metrics(originals: list[str], synthetics: list[str], labels: list[int]):
    """labels: 0=equivalent, 1=unequivalent"""
    vec = TfidfVectorizer(max_features=20000)
    all_text = originals + synthetics
    vec.fit(all_text)

    # pairwise cosine sim between each original and its synthetic
    a = vec.transform(originals)
    b = vec.transform(synthetics)
    sims = np.array([cosine_similarity(a[i], b[i])[0][0] for i in range(len(originals))])

    uneq_mask = np.array(labels) == 1
    uneq_sims = sims[uneq_mask]

    mean_sim = float(uneq_sims.mean())
    pct_high = float((uneq_sims > 0.9).mean())

    # TF-IDF F1: concat pair as single string, train LogReg
    combined = [o + " [SEP] " + s for o, s in zip(originals, synthetics)]
    vec2 = TfidfVectorizer(max_features=20000)
    X = vec2.fit_transform(combined)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    preds = clf.predict(X)
    tfidf_f1 = float(f1_score(y, preds, average="macro"))

    return mean_sim, pct_high, tfidf_f1, sims


def generate_synthetic(client: OpenAI, prompt_template: str, explanation: str, model="gpt-4o-mini") -> str:
    prompt = prompt_template.replace("{explanation}", explanation)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


def optimize_prompt(client: OpenAI, current_prompt: str, metrics: dict,
                    failure_examples: list[dict], n: int) -> str:
    examples_str = ""
    for i, ex in enumerate(failure_examples, 1):
        examples_str += (
            f"\nExample {i} (sim={ex['sim']:.3f}):\n"
            f"  ORIGINAL:  {ex['original'][:300]}\n"
            f"  SYNTHETIC: {ex['synthetic'][:300]}\n"
        )

    mean_sim = metrics["mean_sim"]
    if mean_sim < 0.55:
        diagnosis = (
            f"cosine sim mean is {mean_sim:.3f} — TOO LOW. The synthetics read like completely "
            f"unrelated text rather than alternative interpretations of the same post. "
            f"The model is ignoring the original and generating from scratch."
        )
    elif mean_sim > 0.75:
        diagnosis = (
            f"cosine sim mean is {mean_sim:.3f} — TOO HIGH. The synthetics are near-copies of "
            f"the originals with only a verdict word changed. The model is not reasoning differently."
        )
    else:
        diagnosis = (
            f"cosine sim mean is {mean_sim:.3f} — in range, but TF-IDF F1 is still too high. "
            f"The classifier is finding another shortcut signal. Try increasing vocabulary diversity further."
        )

    meta = META_PROMPT.format(
        current_prompt=current_prompt,
        n=n,
        mean_sim=metrics["mean_sim"],
        pct_high=metrics["pct_high"],
        tfidf_f1=metrics["tfidf_f1"],
        target_mean=TARGET_MEAN_SIM,
        target_pct=TARGET_PCT_HIGH,
        target_f1=TARGET_TFIDF_F1,
        failure_examples=examples_str,
        diagnosis=diagnosis,
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[{"role": "user", "content": meta}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default="reformulated_full.csv")
    p.add_argument("--n",        type=int, default=500)
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--out",      default="loop_results")
    args = p.parse_args()

    client = OpenAI()  # reads OPENAI_API_KEY from env
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    # sample n/2 equivalent originals — we'll pair each with a generated unequivalent
    eq_pool = df[df["meaning"] == "equivalent"].reset_index(drop=True)
    sample = eq_pool.sample(n=args.n, random_state=42).reset_index(drop=True)

    current_prompt = INITIAL_PROMPT
    history = []

    for iteration in range(1, args.max_iter + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # generate synthetics for each sampled original
        originals, synthetics, labels = [], [], []
        for i, row in sample.iterrows():
            orig = row["explanation"]
            # add original equivalent pair
            originals.append(orig)
            synthetics.append(row["synthetic"])
            labels.append(0)

            # generate new unequivalent synthetic
            try:
                synth = generate_synthetic(client, current_prompt, orig)
            except Exception as e:
                print(f"  [{i}] API error: {e}")
                synth = orig  # fallback to copy — will score badly
            originals.append(orig)
            synthetics.append(synth)
            labels.append(1)

            if (i + 1) % 50 == 0:
                print(f"  generated {i+1}/{len(sample)}")

        mean_sim, pct_high, tfidf_f1, sims = compute_metrics(originals, synthetics, labels)

        metrics = {"mean_sim": mean_sim, "pct_high": pct_high, "tfidf_f1": tfidf_f1}
        print(f"\nMetrics:")
        print(f"  cosine sim mean (unequivalent): {mean_sim:.3f}  (target {TARGET_SIM_MIN}–{TARGET_MEAN_SIM})")
        print(f"  % sim > 0.9    (unequivalent): {pct_high:.1%}  (target ≤ {TARGET_PCT_HIGH:.0%})")
        print(f"  TF-IDF F1:                     {tfidf_f1:.3f}  (target ≤ {TARGET_TFIDF_F1})")

        targets_met = (TARGET_SIM_MIN <= mean_sim <= TARGET_MEAN_SIM and
                       pct_high <= TARGET_PCT_HIGH and
                       tfidf_f1 <= TARGET_TFIDF_F1)

        # save iteration
        iter_data = {
            "iteration": iteration,
            "metrics": metrics,
            "prompt": current_prompt,
            "targets_met": targets_met,
            "samples": [
                {"original": originals[i*2+1], "synthetic": synthetics[i*2+1],
                 "sim": float(sims[i*2+1])}
                for i in range(len(sample))
            ]
        }
        (out_dir / f"iter_{iteration:02d}.json").write_text(json.dumps(iter_data, indent=2))
        print(f"  saved → {out_dir}/iter_{iteration:02d}.json")

        if targets_met:
            print(f"\n✓ All targets met at iteration {iteration}. Done.")
            break

        if iteration == args.max_iter:
            print(f"\nMax iterations reached.")
            break

        # find 5 most representative bad examples:
        # if sim too low → show lowest-sim pairs (too different)
        # if sim too high → show highest-sim pairs (too similar)
        uneq_indices = [i for i, l in enumerate(labels) if l == 1]
        uneq_sims = [(sims[i], i) for i in uneq_indices]
        if mean_sim < TARGET_SIM_MIN:
            worst = sorted(uneq_sims)[:5]        # lowest sim = most different
        else:
            worst = sorted(uneq_sims, reverse=True)[:5]  # highest sim = most similar
        failure_examples = [
            {"sim": s, "original": originals[i], "synthetic": synthetics[i]}
            for s, i in worst
        ]

        print(f"\nOptimising prompt with GPT-4o...")
        current_prompt = optimize_prompt(client, current_prompt, metrics, failure_examples, args.n)
        print(f"New prompt (first 200 chars): {current_prompt[:200]}...")

        history.append({"iteration": iteration, "metrics": metrics})

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
