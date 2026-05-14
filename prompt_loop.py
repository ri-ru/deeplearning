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
TARGET_MEAN_SIM  = 0.75   # unequivalent pairs must not be too similar
TARGET_PCT_HIGH  = 0.15   # fraction with sim > 0.9
TARGET_TFIDF_F1  = 0.70   # unequiv vs equiv must not be trivially TF-IDF separable
TARGET_FLIP_RATE       = 0.90  # ≥ 90% of unequivalent pairs must actually flip the conclusion
TARGET_LEN_RATIO_MIN   = 0.5   # synth/orig word-count ratio (mean) lower bound
TARGET_LEN_RATIO_MAX   = 2.0   # synth/orig word-count ratio (mean) upper bound
TARGET_LEN_RATIO_P90   = 3.0   # 90th-percentile ratio — no extreme essays

MAX_CONSECUTIVE_API_FAILURES = 5  # bail out instead of polluting data with copies

FLIP_JUDGE_PROMPT = """\
You will see two rationales about whether a social media post is hate speech.
Answer with exactly one word: SAME or DIFFERENT.

SAME = both rationales reach the same conclusion (both say it IS hate speech, or both say it ISN'T).
DIFFERENT = the rationales reach opposite conclusions.

Rationale A: {original}
Rationale B: {synthetic}

Answer (SAME or DIFFERENT):"""

# ── initial generation prompt ─────────────────────────────────────────────────
INITIAL_PROMPT = """\
You are given a hate-speech classification rationale written by an annotator.

Your task: rewrite the rationale so that it reaches the OPPOSITE conclusion. If the original concludes the post IS hate speech, your rewrite must conclude it is NOT. If the original concludes it is NOT hate speech, your rewrite must conclude it IS.

Hard requirements:
- Match the length of the original within 50%. If the original is 2 sentences, write 2-3 sentences. Do not write paragraphs or essays.
- Match the register of the original: plain analytical prose. No "however," "therefore," "although," "moreover." No mandatory summaries.
- Reference the same post and the same key phrases the original references. The reader should be able to tell these two rationales are about the same post.
- Reach the opposite conclusion through different reasoning, not by negating individual sentences.

Original rationale:
{explanation}

Write only the rewritten rationale, nothing else."""

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

Evaluation results on current prompt ({n} samples):
  - Unequivalent cosine sim mean:   {mean_sim:.3f}  (target 0.55–0.75)
  - % unequiv pairs with sim > 0.9: {pct_high:.1%}  (target ≤ {target_pct})
  - TF-IDF classifier F1:           {tfidf_f1:.3f}  (target ≤ {target_f1})
  - Conclusion-flip rate:           {flip_rate:.1%}  (target ≥ 90%)
  - Length ratio mean (synth/orig): {length_ratio_mean:.2f}  (target 0.5–2.0)
  - Length ratio p90:               {length_ratio_p90:.2f}  (target ≤ 3.0)

Current diagnosis: {diagnosis}

Prior iteration history (oldest → newest, current iteration NOT included):
{history_block}

Anti-oscillation guidance — READ CAREFULLY:
  - If conclusion-flip rate is BELOW 90%: the unequivalent rationales are reaching the SAME
    conclusion as the originals, just with different wording. They are MISLABELED. Fix: add
    an explicit instruction that the synthetic MUST reach the opposite verdict (if original
    says "hate speech," synthetic must argue "not hate speech," and vice versa). Have the
    model state the target verdict explicitly at the start of its reasoning.
  - If length ratio mean is ABOVE 2.0: the synthetics are too long. Originals are 1–3
    sentences. Force the synthetic to be the same length. Add: "Match the length of the
    original rationale within 50%. Do not add extra paragraphs, summaries, or philosophical
    elaboration." Forbid words like "however," "therefore," "although," "moreover" if they
    are being used as paragraph-glue rather than substantive reasoning.
  - If length ratio is BELOW 0.5: the synthetics are too short and likely missing reasoning.
  - If past iterations show cosine swinging between TOO LOW (<0.55) and TOO HIGH (>0.75),
    the previous rewrites overcorrected. Make a SMALLER, more targeted edit this time.
    Do not flip the whole strategy — keep what worked in the closer-to-target iteration
    and add ONE concrete constraint to nudge the metric in the right direction.
  - If past iterations consistently miss in the same direction, your constraints are not
    strong enough. Add explicit, countable rules (e.g. "reuse at least 4 distinct nouns
    from the original").
  - Vague language ("be more creative", "use different reasoning") has already been tried
    and did not move the metric. Prefer concrete, measurable instructions.
  - DO NOT add formal-essay constraints like "use at least N sentences with 'therefore'"
    or "begin with a summary." These introduce stylistic shortcuts that TF-IDF picks up on,
    and they make the synthetic much longer than the original.

Diagnosis guidance (general):
  - cosine BELOW 0.55: synthetics drift to unrelated text. Anchor to the post's vocabulary.
  - cosine ABOVE 0.75: synthetics are near-copies. Force paraphrase of key claims.
  - cosine IN BAND but TF-IDF F1 high: a non-cosine lexical shortcut remains. Vary structure.

Here are 5 example pairs to illustrate the current issue:

{failure_examples}

Write an improved generation prompt that fixes the diagnosed problem above WHILE accounting for
the prior trajectory shown in the history. Output ONLY the improved prompt text (the {{explanation}}
placeholder must remain), nothing else. Do NOT wrap the prompt in quotes or <<<>>> markers."""


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


# ── flip-rate metric (does the unequivalent actually reach a different conclusion?) ──

def _flip_cache_key(original: str, synthetic: str) -> str:
    import hashlib
    return hashlib.md5((original + "|||" + synthetic).encode("utf-8")).hexdigest()


def load_flip_cache(out_dir: Path) -> dict[str, str]:
    path = out_dir / "flip_cache.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_flip_cache(out_dir: Path, cache: dict[str, str]) -> None:
    (out_dir / "flip_cache.json").write_text(json.dumps(cache, indent=2))


def judge_flip(client: OpenAI, original: str, synthetic: str,
               cache: dict[str, str]) -> str:
    """Return 'SAME', 'DIFFERENT', or 'OTHER' (parse failure). Cached by pair hash."""
    key = _flip_cache_key(original, synthetic)
    if key in cache:
        return cache[key]
    prompt = FLIP_JUDGE_PROMPT.format(original=original, synthetic=synthetic)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=4,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip().upper()
    except Exception as e:
        return "OTHER"
    verdict = "SAME" if raw.startswith("SAME") else ("DIFFERENT" if raw.startswith("DIFFERENT") else "OTHER")
    cache[key] = verdict
    return verdict


def compute_extra_metrics(client: OpenAI, originals: list[str], synthetics: list[str],
                          labels: list[int], cache: dict[str, str]) -> dict:
    """Compute flip_rate and length-ratio metrics on the unequivalent pairs only."""
    uneq_pairs = [(o, s) for o, s, l in zip(originals, synthetics, labels) if l == 1]

    # length ratios
    ratios = []
    for o, s in uneq_pairs:
        ow = max(len(o.split()), 1)
        sw = len(s.split())
        ratios.append(sw / ow)
    ratios_arr = np.array(ratios) if ratios else np.array([0.0])
    length_ratio_mean = float(ratios_arr.mean())
    length_ratio_p90 = float(np.percentile(ratios_arr, 90))

    # flip rate
    verdicts = []
    n_same = n_diff = n_other = 0
    for i, (o, s) in enumerate(uneq_pairs):
        v = judge_flip(client, o, s, cache)
        verdicts.append(v)
        if v == "SAME":
            n_same += 1
        elif v == "DIFFERENT":
            n_diff += 1
        else:
            n_other += 1
        if (i + 1) % 50 == 0:
            print(f"  flip-judge: {i+1}/{len(uneq_pairs)}")
    total = len(uneq_pairs)
    flip_rate = (n_diff / total) if total else 0.0

    return {
        "flip_rate": flip_rate,
        "n_same": n_same,
        "n_diff": n_diff,
        "n_other": n_other,
        "n_total_uneq": total,
        "length_ratio_mean": length_ratio_mean,
        "length_ratio_p90": length_ratio_p90,
        "verdicts": verdicts,  # parallel to uneq_pairs order
    }


def generate_synthetic(client: OpenAI, prompt_template: str, explanation: str, model="gpt-4o-mini") -> str:
    prompt = prompt_template.replace("{explanation}", explanation)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


def write_iter_csv(out_dir: Path, iteration: int, originals: list[str],
                   synthetics: list[str], labels: list[int], sims) -> None:
    """Human-readable CSV per iteration for easy skimming."""
    import csv as _csv
    path = out_dir / f"iter_{iteration:02d}_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["pair_idx", "label", "cosine_sim", "original", "synthetic"])
        for i, (o, s, l) in enumerate(zip(originals, synthetics, labels)):
            w.writerow([i, "equivalent" if l == 0 else "unequivalent",
                        f"{float(sims[i]):.4f}", o, s])


def write_summary_md(out_dir: Path, history: list[dict], final_prompt: str,
                     final_metrics: dict, final_samples: list[dict],
                     targets_met: bool, stopped_reason: str) -> None:
    eq_examples = [s for s in final_samples if s["label"] == 0][:5]
    uneq_examples = [s for s in final_samples if s["label"] == 1][:5]
    mislabeled = [s for s in final_samples
                  if s["label"] == 1 and s.get("flip_verdict") == "SAME"]

    fm = final_metrics
    lines = []
    lines.append("# Self-Improvement Loop — Summary\n")
    lines.append(f"**Status:** {stopped_reason}  ")
    lines.append(f"**Targets met:** {'yes' if targets_met else 'no'}\n")
    lines.append("## Final metrics (on final batch)\n")
    lines.append(f"- Unequivalent cosine sim mean: **{fm.get('mean_sim', 0):.3f}** "
                 f"(target {TARGET_SIM_MIN}–{TARGET_MEAN_SIM})")
    lines.append(f"- % unequiv pairs with cosine > 0.9: **{fm.get('pct_high', 0):.1%}** "
                 f"(target ≤ {TARGET_PCT_HIGH:.0%})")
    lines.append(f"- TF-IDF F1 baseline: **{fm.get('tfidf_f1', 0):.3f}** "
                 f"(target ≤ {TARGET_TFIDF_F1})")
    lines.append(f"- Conclusion-flip rate: **{fm.get('flip_rate', 0):.1%}** "
                 f"(target ≥ {TARGET_FLIP_RATE:.0%})  "
                 f"[{fm.get('n_diff', 0)} flipped / {fm.get('n_same', 0)} mislabeled / "
                 f"{fm.get('n_other', 0)} unparsable]")
    lines.append(f"- Length ratio mean (synth/orig words): **{fm.get('length_ratio_mean', 0):.2f}** "
                 f"(target {TARGET_LEN_RATIO_MIN}–{TARGET_LEN_RATIO_MAX})")
    lines.append(f"- Length ratio p90: **{fm.get('length_ratio_p90', 0):.2f}** "
                 f"(target ≤ {TARGET_LEN_RATIO_P90})\n")

    lines.append("## Iteration history\n")
    lines.append("| iter | mean_sim | pct>0.9 | tfidf_f1 | flip_rate | len_ratio | len_p90 |")
    lines.append("|------|----------|---------|----------|-----------|-----------|---------|")
    for h in history:
        m = h["metrics"]
        lines.append(
            f"| {h['iteration']} | {m['mean_sim']:.3f} | {m['pct_high']:.1%} | "
            f"{m['tfidf_f1']:.3f} | {m.get('flip_rate', 0):.1%} | "
            f"{m.get('length_ratio_mean', 0):.2f} | {m.get('length_ratio_p90', 0):.2f} |"
        )

    lines.append("\n## Final prompt used\n```")
    lines.append(final_prompt)
    lines.append("```\n")

    lines.append("## 5 equivalent samples (final batch)\n")
    for i, s in enumerate(eq_examples, 1):
        lines.append(f"### Equivalent {i} (cosine={s['sim']:.3f})")
        lines.append(f"- **ORIGINAL:** {s['original']}")
        lines.append(f"- **SYNTHETIC:** {s['synthetic']}\n")

    lines.append("## 5 unequivalent samples (final batch)\n")
    for i, s in enumerate(uneq_examples, 1):
        verdict = s.get("flip_verdict", "?")
        flag = "" if verdict == "DIFFERENT" else f" ⚠ verdict={verdict}"
        lines.append(f"### Unequivalent {i} (cosine={s['sim']:.3f}){flag}")
        lines.append(f"- **ORIGINAL:** {s['original']}")
        lines.append(f"- **SYNTHETIC:** {s['synthetic']}\n")

    lines.append("## Conclusion-flip audit\n")
    if not mislabeled:
        lines.append("No SAME-verdict pairs in the final batch — every unequivalent rationale "
                     "actually reaches a different conclusion. Clean.\n")
    else:
        lines.append(f"{len(mislabeled)} of {fm.get('n_total_uneq', 0)} unequivalent pairs were "
                     f"judged SAME (mislabeled — synthetic reaches the same conclusion as the original). "
                     f"Showing up to 5:\n")
        for i, s in enumerate(mislabeled[:5], 1):
            lines.append(f"### SAME-verdict #{i} (cosine={s['sim']:.3f})")
            lines.append(f"- **ORIGINAL:** {s['original']}")
            lines.append(f"- **SYNTHETIC:** {s['synthetic']}\n")

    # ── honest assessment, ordered by severity ──
    lines.append("## Honest assessment\n")
    fr = fm.get("flip_rate", 0)
    lr = fm.get("length_ratio_mean", 0)
    in_band = TARGET_SIM_MIN <= fm.get("mean_sim", 0) <= TARGET_MEAN_SIM
    tfidf_ok = fm.get("tfidf_f1", 1.0) <= TARGET_TFIDF_F1

    if fr < TARGET_FLIP_RATE:
        lines.append(
            f"Flip rate is **{fr:.1%}** — below the 90% target. Roughly "
            f"{fm.get('n_same', 0)} of {fm.get('n_total_uneq', 0)} 'unequivalent' pairs actually "
            f"reach the same conclusion as the original. A judge trained on this would learn to "
            f"call agreeing rationales 'unequivalent,' poisoning the supervision signal. The "
            f"prompt is not producing genuine label flips and needs to be rewritten to force "
            f"the model to commit to the opposite verdict explicitly."
        )
    elif lr > TARGET_LEN_RATIO_MAX:
        lines.append(
            f"Length ratio is **{lr:.2f}×** — synthetics are roughly that much longer than "
            f"originals. Even if cosine and flip rate look fine, the length disparity becomes a "
            f"surface shortcut: a TF-IDF classifier (and an LSTM, and a transformer with positional "
            f"signal) can separate the two by length alone. The prompt is producing essays where "
            f"the originals are short analytical notes."
        )
    elif in_band and tfidf_ok and fr >= TARGET_FLIP_RATE:
        lines.append(
            "Final batch sits in the cosine band, TF-IDF can't trivially separate, and ≥90% of "
            "unequivalent pairs actually flip the conclusion. The synthetic data looks usable for "
            "training a judge that has to learn real semantic equivalence rather than negation "
            "tokens or stylistic shortcuts."
        )
    elif fm.get("mean_sim", 0) > TARGET_MEAN_SIM:
        lines.append(
            "Cosine is too high — synthetics are near-copies of the originals. The model is "
            "hedging by changing only a few verdict words. A judge trained on this picks up "
            "negation-token heuristics rather than meaning."
        )
    elif fm.get("mean_sim", 0) < TARGET_SIM_MIN:
        lines.append(
            "Cosine is below target — synthetics drifted off-topic. The judge would learn "
            "'is this text about a different topic' rather than rationale equivalence."
        )
    else:
        lines.append(
            "Cosine band and flip rate are fine but TF-IDF F1 is still above target — there's a "
            "residual lexical shortcut. Worth eyeballing the worst-separated pairs."
        )

    (out_dir / "summary.md").write_text("\n".join(lines))


def diagnose(metrics: dict) -> str:
    """Stable one-line diagnosis used in both the optimizer and the history block.
    Priority order: flip-rate, length-ratio, cosine, TF-IDF (most severe first)."""
    mean_sim = metrics["mean_sim"]
    tfidf_f1 = metrics["tfidf_f1"]
    flip_rate = metrics.get("flip_rate")
    length_ratio_mean = metrics.get("length_ratio_mean")
    n_same = metrics.get("n_same")
    n_total = metrics.get("n_total_uneq")

    if flip_rate is not None and flip_rate < TARGET_FLIP_RATE:
        return (f"flip rate={flip_rate:.1%} TOO LOW — {n_same} of {n_total} unequivalent "
                f"pairs reach the SAME conclusion as the original. These are mislabeled.")
    if length_ratio_mean is not None and length_ratio_mean > TARGET_LEN_RATIO_MAX:
        return (f"length ratio={length_ratio_mean:.2f} TOO HIGH — synthetics are "
                f"{length_ratio_mean:.1f}× longer than originals, introducing register shortcut.")
    if length_ratio_mean is not None and length_ratio_mean < TARGET_LEN_RATIO_MIN:
        return (f"length ratio={length_ratio_mean:.2f} TOO LOW — synthetics are much "
                f"shorter than originals; likely missing reasoning.")
    if mean_sim < TARGET_SIM_MIN:
        return (f"cosine={mean_sim:.3f} TOO LOW — synthetics drift off-topic, "
                f"ignoring the original post.")
    if mean_sim > TARGET_MEAN_SIM:
        return (f"cosine={mean_sim:.3f} TOO HIGH — synthetics are near-copies with "
                f"verdict word flipped.")
    if tfidf_f1 > TARGET_TFIDF_F1:
        return (f"cosine={mean_sim:.3f} in band, but TF-IDF F1={tfidf_f1:.3f} too high — "
                f"a non-cosine lexical shortcut remains.")
    return f"all targets met (cosine={mean_sim:.3f}, flip={flip_rate or 0:.1%})."


def build_history_block(history: list[dict]) -> str:
    """Compact textual record of past iterations for the meta-prompt."""
    if not history:
        return "  (none — this is the first optimization step)"
    lines = []
    for h in history:
        m = h["metrics"]
        flip = m.get("flip_rate", float("nan"))
        lr = m.get("length_ratio_mean", float("nan"))
        lr_p90 = m.get("length_ratio_p90", float("nan"))
        lines.append(
            f"  iter {h['iteration']}: cosine={m['mean_sim']:.3f}, "
            f"pct>0.9={m['pct_high']:.1%}, tfidf_f1={m['tfidf_f1']:.3f}, "
            f"flip_rate={flip:.1%}, len_ratio={lr:.2f} (p90={lr_p90:.2f})"
        )
        lines.append(f"    diagnosis: {h['diagnosis']}")
        lines.append(f"    prompt opening: {h['prompt_head']}")
    return "\n".join(lines)


def optimize_prompt(client: OpenAI, current_prompt: str, metrics: dict,
                    failure_examples: list[dict], n: int,
                    history: list[dict]) -> str:
    examples_str = ""
    for i, ex in enumerate(failure_examples, 1):
        examples_str += (
            f"\nExample {i} (sim={ex['sim']:.3f}):\n"
            f"  ORIGINAL:  {ex['original'][:300]}\n"
            f"  SYNTHETIC: {ex['synthetic'][:300]}\n"
        )

    diagnosis = diagnose(metrics)
    history_block = build_history_block(history)

    meta = META_PROMPT.format(
        current_prompt=current_prompt,
        n=n,
        mean_sim=metrics["mean_sim"],
        pct_high=metrics["pct_high"],
        tfidf_f1=metrics["tfidf_f1"],
        flip_rate=metrics.get("flip_rate", 0.0),
        length_ratio_mean=metrics.get("length_ratio_mean", 0.0),
        length_ratio_p90=metrics.get("length_ratio_p90", 0.0),
        target_mean=TARGET_MEAN_SIM,
        target_pct=TARGET_PCT_HIGH,
        target_f1=TARGET_TFIDF_F1,
        failure_examples=examples_str,
        diagnosis=diagnosis,
        history_block=history_block,
    )
    # persist the actual meta-prompt sent (last one wins per iter) for smoke verification
    try:
        last = Path(metrics.get("_out_dir", ".")) / "last_meta_prompt.txt"
        last.write_text(meta)
    except Exception:
        pass
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        messages=[{"role": "user", "content": meta}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate_batch(client: OpenAI, prompt: str, originals_paired: pd.DataFrame,
                   max_failures: int = MAX_CONSECUTIVE_API_FAILURES,
                   tag: str = "iter") -> tuple[list[str], list[str], list[int], int]:
    """Generate one unequivalent synthetic per original. Bail on persistent API failure.

    Returns (originals, synthetics, labels, n_failures). For each input row we emit
    TWO pairs: (orig, original_equiv_synthetic, label=0) and
    (orig, new_unequivalent_synthetic, label=1).
    """
    originals, synthetics, labels = [], [], []
    consecutive_failures = 0
    total_failures = 0

    for i, row in originals_paired.iterrows():
        orig = row["explanation"]
        originals.append(orig)
        synthetics.append(row["synthetic"])
        labels.append(0)

        try:
            synth = generate_synthetic(client, prompt, orig)
            consecutive_failures = 0
        except Exception as e:
            print(f"  [{tag} {i}] API error: {e}")
            total_failures += 1
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"  ⚠ aborting batch after {consecutive_failures} consecutive failures")
                return originals, synthetics, labels, total_failures
            synth = ""  # placeholder — will be removed below if we abort, otherwise scored badly
        originals.append(orig)
        synthetics.append(synth)
        labels.append(1)

        if (i + 1) % 50 == 0:
            print(f"  {tag}: generated {i+1}/{len(originals_paired)}")

    return originals, synthetics, labels, total_failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default="reformulated_full.csv")
    p.add_argument("--n",        type=int, default=200)
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--out",      default="loop_results")
    p.add_argument("--final_n",  type=int, default=200,
                   help="size of the final clean batch generated with the best prompt")
    p.add_argument("--smoke",    action="store_true",
                   help="smoke test: n=2, max_iter=1, final_n=2, no prompt optimisation")
    args = p.parse_args()

    if args.smoke:
        args.n, args.max_iter, args.final_n = 4, 2, 2
        print("=== SMOKE MODE: n=4, max_iter=2, final_n=2 (exercises optimizer + memory) ===")

    client = OpenAI()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    eq_pool = df[df["meaning"] == "equivalent"].reset_index(drop=True)
    sample = eq_pool.sample(n=args.n, random_state=42).reset_index(drop=True)

    current_prompt = INITIAL_PROMPT
    history = []
    targets_met = False
    last_metrics = None
    stopped_reason = "completed all iterations"
    flip_cache = load_flip_cache(out_dir)

    for iteration in range(1, args.max_iter + 1):
        print(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")

        originals, synthetics, labels, n_failures = generate_batch(
            client, current_prompt, sample, tag=f"iter{iteration}"
        )
        if n_failures >= MAX_CONSECUTIVE_API_FAILURES and len(originals) < 2 * len(sample):
            stopped_reason = f"aborted at iter {iteration} after {n_failures} API failures"
            print(f"\n⚠ {stopped_reason}")
            break

        mean_sim, pct_high, tfidf_f1, sims = compute_metrics(originals, synthetics, labels)

        print(f"  computing flip-rate + length-ratio ({sum(1 for l in labels if l==1)} unequiv pairs)…")
        extra = compute_extra_metrics(client, originals, synthetics, labels, flip_cache)
        save_flip_cache(out_dir, flip_cache)

        metrics = {
            "mean_sim": mean_sim, "pct_high": pct_high, "tfidf_f1": tfidf_f1,
            "flip_rate": extra["flip_rate"],
            "n_same": extra["n_same"], "n_diff": extra["n_diff"], "n_other": extra["n_other"],
            "n_total_uneq": extra["n_total_uneq"],
            "length_ratio_mean": extra["length_ratio_mean"],
            "length_ratio_p90": extra["length_ratio_p90"],
            "_out_dir": str(out_dir),  # for last_meta_prompt.txt capture
        }
        last_metrics = metrics
        print(f"\nMetrics:")
        print(f"  cosine sim mean (unequiv): {mean_sim:.3f}  (target {TARGET_SIM_MIN}–{TARGET_MEAN_SIM})")
        print(f"  % sim > 0.9 (unequiv):     {pct_high:.1%}  (target ≤ {TARGET_PCT_HIGH:.0%})")
        print(f"  TF-IDF F1:                 {tfidf_f1:.3f}  (target ≤ {TARGET_TFIDF_F1})")
        print(f"  flip rate:                 {extra['flip_rate']:.1%}  (target ≥ {TARGET_FLIP_RATE:.0%})")
        print(f"  length ratio mean:         {extra['length_ratio_mean']:.2f}  "
              f"(target {TARGET_LEN_RATIO_MIN}–{TARGET_LEN_RATIO_MAX})")
        print(f"  length ratio p90:          {extra['length_ratio_p90']:.2f}  "
              f"(target ≤ {TARGET_LEN_RATIO_P90})")

        targets_met = (
            TARGET_SIM_MIN <= mean_sim <= TARGET_MEAN_SIM
            and pct_high <= TARGET_PCT_HIGH
            and tfidf_f1 <= TARGET_TFIDF_F1
            and extra["flip_rate"] >= TARGET_FLIP_RATE
            and TARGET_LEN_RATIO_MIN <= extra["length_ratio_mean"] <= TARGET_LEN_RATIO_MAX
            and extra["length_ratio_p90"] <= TARGET_LEN_RATIO_P90
        )

        # uneq verdicts aligned to the (orig, synth) order within unequiv pairs
        verdicts = extra["verdicts"]
        uneq_iter = iter(verdicts)
        sample_verdicts = [
            (next(uneq_iter) if labels[k] == 1 else "EQUIV_NA")
            for k in range(len(originals))
        ]

        # drop _out_dir from serialized metrics (it's a runtime helper, not data)
        serialized_metrics = {k: v for k, v in metrics.items() if k != "_out_dir"}
        iter_data = {
            "iteration": iteration,
            "metrics": serialized_metrics,
            "prompt": current_prompt,
            "targets_met": targets_met,
            "n_api_failures": n_failures,
            "samples": [
                {"original": originals[k], "synthetic": synthetics[k],
                 "label": int(labels[k]), "sim": float(sims[k]),
                 "flip_verdict": sample_verdicts[k]}
                for k in range(len(originals))
            ],
        }
        (out_dir / f"iter_{iteration:02d}.json").write_text(json.dumps(iter_data, indent=2))
        write_iter_csv(out_dir, iteration, originals, synthetics, labels, sims)
        print(f"  saved → {out_dir}/iter_{iteration:02d}.json + .csv")

        iter_diag = diagnose(metrics)
        history.append({
            "iteration": iteration,
            "metrics": serialized_metrics,
            "diagnosis": iter_diag,
            "prompt_head": current_prompt[:200].replace("\n", " ").strip() + "…",
        })

        if targets_met:
            stopped_reason = f"targets met at iter {iteration}"
            print(f"\n✓ All targets met at iteration {iteration}.")
            break

        if iteration == args.max_iter:
            stopped_reason = f"max iterations reached ({args.max_iter})"
            print(f"\nMax iterations reached.")
            break

        uneq_indices = [k for k, l in enumerate(labels) if l == 1]
        uneq_sims = [(sims[k], k) for k in uneq_indices]
        if mean_sim < TARGET_SIM_MIN:
            worst = sorted(uneq_sims)[:5]
        else:
            worst = sorted(uneq_sims, reverse=True)[:5]
        failure_examples = [
            {"sim": s, "original": originals[k], "synthetic": synthetics[k]}
            for s, k in worst
        ]

        print(f"\nOptimising prompt with GPT-4o (history={len(history)-1} prior iter(s))…")
        try:
            current_prompt = optimize_prompt(
                client, current_prompt, metrics, failure_examples, args.n,
                history=history[:-1],  # past iterations only, not the current one
            )
        except Exception as e:
            print(f"  ⚠ prompt optimiser failed: {e}")
            stopped_reason = f"prompt optimiser failed at iter {iteration}: {e}"
            break
        print(f"New prompt (first 200 chars): {current_prompt[:200]}…")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ── final clean batch with the best prompt ──────────────────────────────
    print(f"\n{'='*60}\nFINAL BATCH (n={args.final_n})\n{'='*60}")
    final_sample = eq_pool.sample(n=args.final_n, random_state=4242).reset_index(drop=True)
    f_originals, f_synthetics, f_labels, f_failures = generate_batch(
        client, current_prompt, final_sample, tag="final"
    )
    if len(f_originals) < 2:
        print("⚠ final batch failed; skipping summary metrics on it")
        final_metrics = last_metrics or {"mean_sim": 0.0, "pct_high": 0.0, "tfidf_f1": 0.0,
                                         "flip_rate": 0.0, "length_ratio_mean": 0.0,
                                         "length_ratio_p90": 0.0}
        final_metrics = {k: v for k, v in final_metrics.items() if k != "_out_dir"}
        final_samples_for_summary: list[dict] = []
    else:
        f_mean, f_pct, f_tfidf, f_sims = compute_metrics(f_originals, f_synthetics, f_labels)
        print(f"  computing flip-rate + length-ratio on final batch…")
        f_extra = compute_extra_metrics(client, f_originals, f_synthetics, f_labels, flip_cache)
        save_flip_cache(out_dir, flip_cache)
        final_metrics = {
            "mean_sim": f_mean, "pct_high": f_pct, "tfidf_f1": f_tfidf,
            "flip_rate": f_extra["flip_rate"],
            "n_same": f_extra["n_same"], "n_diff": f_extra["n_diff"], "n_other": f_extra["n_other"],
            "n_total_uneq": f_extra["n_total_uneq"],
            "length_ratio_mean": f_extra["length_ratio_mean"],
            "length_ratio_p90": f_extra["length_ratio_p90"],
        }
        f_verdicts_iter = iter(f_extra["verdicts"])
        f_sample_verdicts = [
            (next(f_verdicts_iter) if f_labels[k] == 1 else "EQUIV_NA")
            for k in range(len(f_originals))
        ]
        final_data = {
            "metrics": final_metrics,
            "prompt": current_prompt,
            "n_api_failures": f_failures,
            "samples": [
                {"original": f_originals[k], "synthetic": f_synthetics[k],
                 "label": int(f_labels[k]), "sim": float(f_sims[k]),
                 "flip_verdict": f_sample_verdicts[k]}
                for k in range(len(f_originals))
            ],
        }
        (out_dir / "final_batch.json").write_text(json.dumps(final_data, indent=2))
        write_iter_csv(out_dir, 99, f_originals, f_synthetics, f_labels, f_sims)
        # rename the dummy iter_99 csv to final_batch.csv
        (out_dir / "iter_99_samples.csv").rename(out_dir / "final_batch.csv")
        print(f"  final cosine sim mean:     {f_mean:.3f}")
        print(f"  final % sim > 0.9:         {f_pct:.1%}")
        print(f"  final TF-IDF F1:           {f_tfidf:.3f}")
        print(f"  final flip rate:           {f_extra['flip_rate']:.1%}")
        print(f"  final length ratio mean:   {f_extra['length_ratio_mean']:.2f}")
        print(f"  final length ratio p90:    {f_extra['length_ratio_p90']:.2f}")
        final_samples_for_summary = final_data["samples"]

    write_summary_md(out_dir, history, current_prompt, final_metrics,
                     final_samples_for_summary, targets_met, stopped_reason)
    print(f"\n✓ wrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
