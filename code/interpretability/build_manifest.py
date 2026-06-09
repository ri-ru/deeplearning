"""Build the mech-interp manifest from synthetic + manual pair CSVs.

Output: mech_interp/extracts/manifest.csv with columns
    sample_id, source, label, contains_negation_in_synthetic,
    in_synthetic_not_original, text_a, text_b

- text_a = original/model rationale (=`explanation` in synthetic, col 1 in manual)
- text_b = reformulated rationale (=`synthetic`     in synthetic, col 2 in manual)
- label  = 0 equivalent, 1 unequivalent (matches judge_finetune/src/data.py)
- contains_negation_in_synthetic = regex match on text_b
- in_synthetic_not_original      = 1 if from reformulated_full.csv else 0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

NEGATION_RE = re.compile(r"\b(not|n't|never|no|none|nothing|neither|nor)\b", re.IGNORECASE)
LABEL2ID = {"equivalent": 0, "unequivalent": 1}

REPO = Path(__file__).resolve().parents[2]
SYNTH_CSV = REPO / "reformulated_full.csv"
MANUAL_CSV = REPO / "manual labelling.csv"
DEFAULT_OUT = REPO / "mech_interp" / "extracts" / "manifest.csv"


def _has_negation(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return int(bool(NEGATION_RE.search(text)))


def load_synthetic(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["explanation", "synthetic", "meaning"]).copy()
    df["meaning"] = df["meaning"].astype(str).str.lower().str.strip()
    df = df[df["meaning"].isin(LABEL2ID)].copy()
    df["label"] = df["meaning"].map(LABEL2ID).astype(int)
    df["sample_id"] = df["uid"].astype(str)
    df["source"] = "synthetic"
    df["in_synthetic_not_original"] = 1
    df = df.rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    return df[["sample_id", "source", "label", "in_synthetic_not_original", "text_a", "text_b"]]


def load_manual(csv_path: Path) -> pd.DataFrame:
    # Numbers export with a 'Table 1' preamble row; skip it and use row 2 as the header.
    raw = pd.read_csv(csv_path, skiprows=1, header=0)
    # Expected schema: post / explanation / meaning / equivalence / 3 empty trailing cols.
    cols = list(raw.columns)
    if len(cols) < 4:
        raise ValueError(f"manual labelling.csv has unexpected schema: {cols}")
    df = raw.rename(columns={cols[0]: "post", cols[1]: "text_a", cols[2]: "text_b", cols[3]: "label_str"})
    df = df[["post", "text_a", "text_b", "label_str"]].dropna(subset=["text_a", "text_b", "label_str"]).copy()
    df["label_str"] = df["label_str"].astype(str).str.lower().str.strip()
    df = df[df["label_str"].isin(LABEL2ID)].copy()
    df["label"] = df["label_str"].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)
    df["sample_id"] = ["manual_" + str(i).zfill(3) for i in range(len(df))]
    df["source"] = "manual"
    df["in_synthetic_not_original"] = 0
    return df[["sample_id", "source", "label", "in_synthetic_not_original", "text_a", "text_b"]]


def build(out_path: Path, synth_csv: Path, manual_csv: Path) -> pd.DataFrame:
    synth = load_synthetic(synth_csv)
    manual = load_manual(manual_csv)
    manifest = pd.concat([synth, manual], ignore_index=True)
    manifest["contains_negation_in_synthetic"] = manifest["text_b"].map(_has_negation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic_csv", type=Path, default=SYNTH_CSV)
    parser.add_argument("--manual_csv", type=Path, default=MANUAL_CSV)
    args = parser.parse_args()

    manifest = build(args.out, args.synthetic_csv, args.manual_csv)

    print(f"[manifest] rows: {len(manifest)}")
    print(f"[manifest] sources: {manifest['source'].value_counts().to_dict()}")
    print(f"[manifest] label balance (overall): {manifest['label'].value_counts(normalize=True).round(4).to_dict()}")

    synth = manifest[manifest["source"] == "synthetic"]
    manual = manifest[manifest["source"] == "manual"]
    print(
        f"[manifest] synthetic negation rate: {synth['contains_negation_in_synthetic'].mean():.4f} "
        f"(n={len(synth)})"
    )
    print(
        f"[manifest] manual negation rate:    {manual['contains_negation_in_synthetic'].mean():.4f} "
        f"(n={len(manual)})"
    )

    # Prior: how strongly does the regex flag alone correlate with the label, on synthetic?
    if len(synth) > 0:
        from sklearn.metrics import roc_auc_score

        auc_prior = roc_auc_score(synth["label"], synth["contains_negation_in_synthetic"])
        print(f"[prior]   regex-only AUC vs label (synthetic): {auc_prior:.4f}")
        rate_by_label = synth.groupby("label")["contains_negation_in_synthetic"].mean()
        print(f"[prior]   negation rate by label (synthetic): {rate_by_label.to_dict()}")

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
