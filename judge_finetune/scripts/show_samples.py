import pandas as pd
df = pd.read_csv("../reformulated_full.csv")

print("=== 6 UNEQUIVALENT EXAMPLES ===\n")
uneq = df[df["meaning"] == "unequivalent"].sample(6, random_state=99)
for i, (_, row) in enumerate(uneq.iterrows()):
    print(f"[{i+1}] ORIGINAL:  {row['explanation'][:200]}")
    print(f"[{i+1}] SYNTHETIC: {row['synthetic'][:200]}")
    print()

print("=== 4 EQUIVALENT EXAMPLES ===\n")
eq = df[df["meaning"] == "equivalent"].sample(4, random_state=99)
for i, (_, row) in enumerate(eq.iterrows()):
    print(f"[{i+1}] ORIGINAL:  {row['explanation'][:200]}")
    print(f"[{i+1}] SYNTHETIC: {row['synthetic'][:200]}")
    print()
