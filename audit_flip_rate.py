"""One-off audit: how many of the existing 'unequivalent' pairs actually flip the conclusion?"""

import csv
from openai import OpenAI

FLIP_PROMPT = """You will see two rationales about whether a social media post is hate speech.
Answer with exactly one word: SAME or DIFFERENT.

SAME = both rationales reach the same conclusion (both say it IS hate speech, or both say it ISN'T).
DIFFERENT = the rationales reach opposite conclusions.

Rationale A: {original}
Rationale B: {synthetic}

Answer (SAME or DIFFERENT):"""


def judge(client: OpenAI, orig: str, synth: str) -> str:
    p = FLIP_PROMPT.format(original=orig, synthetic=synth)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4,
        messages=[{"role": "user", "content": p}],
        temperature=0,
    )
    return r.choices[0].message.content.strip().upper()


def main() -> None:
    client = OpenAI()
    with open("loop_results/final_batch.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "unequivalent"][:30]

    results = []
    for i, row in enumerate(rows, 1):
        try:
            verdict = judge(client, row["original"], row["synthetic"])
        except Exception as e:
            verdict = f"ERROR({e})"
        results.append((verdict, row["original"], row["synthetic"], row["cosine_sim"]))
        print(f"[{i:>2}/{len(rows)}] {verdict}", flush=True)

    same = [r for r in results if r[0] == "SAME"]
    diff = [r for r in results if r[0] == "DIFFERENT"]
    err = [r for r in results if r[0].startswith("ERROR")]
    other = [r for r in results if r[0] not in ("SAME", "DIFFERENT") and not r[0].startswith("ERROR")]
    flip_rate = len(diff) / len(results) if results else 0.0

    print(f"\n=== AUDIT RESULTS (n={len(results)}) ===")
    print(f"  DIFFERENT (correctly flipped): {len(diff)}")
    print(f"  SAME (mislabeled):             {len(same)}")
    print(f"  ERROR:                         {len(err)}")
    print(f"  Other:                         {len(other)} → {[r[0] for r in other]}")
    print(f"  Flip rate:                     {flip_rate:.1%}")

    print(f"\n=== 3 SAME examples ===")
    for i, (_, o, s, sim) in enumerate(same[:3], 1):
        print(f"\n--- SAME #{i} (cosine={sim}) ---")
        print(f"ORIGINAL : {o[:500]}")
        print(f"SYNTHETIC: {s[:500]}")


if __name__ == "__main__":
    main()
