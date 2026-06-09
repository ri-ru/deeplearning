import pandas as pd
import re
from pathlib import Path

REFUSAL_RE = re.compile(
    r"i cannot|i can'?t|i won'?t|i'm not able|i'm unable|i apologize|"
    r"i'm sorry|i refuse|cannot help with|cannot assist|cannot provide|"
    r"i'm designed to|my role|my purpose|i need to clarify",
    re.IGNORECASE,
)

cross = Path(r"overnight/cross_gen")
print(f"{'generator':>22} {'total':>6} {'refusal-like':>14} {'%':>6}")
for f in sorted(cross.glob("*.csv")):
    if f.name.startswith("_"): continue
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    if len(df) == 0:
        continue
    n = len(df)
    refusals = df["synthetic"].astype(str).apply(lambda s: bool(REFUSAL_RE.search(s[:200]))).sum()
    print(f"{f.stem:>22} {n:>6} {refusals:>14} {refusals/n*100:>5.1f}%")
    # break down by label for anthropic
    if "anthropic" in f.stem:
        for lbl in ("equivalent", "unequivalent"):
            sub = df[df["meaning"]==lbl]
            refs = sub["synthetic"].astype(str).apply(lambda s: bool(REFUSAL_RE.search(s[:200]))).sum()
            print(f"     - {lbl:<22} n={len(sub):>3}  refusals={refs} ({refs/max(len(sub),1)*100:.1f}%)")
