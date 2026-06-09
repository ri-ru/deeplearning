"""Verify the four poster (reproductibility) 
1. TF-IDF baseline F1 ≈ 0.789 on the OOD test split
2. GPT-4o-mini zero-shot F1 ≈ 0.964  
3. RoBERTa OOD F1 ≈ 0.968  
4. Cosine-sim shortcut: 21.6% of equivalent pairs and 50.9% of unequivalent pairs have sim > 0.9
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "reformulated_full.csv"

# Hard-coded so we replicate the project's split exactly.
TRAIN_DOMAINS = {"dynahate", "cad", "measure", "socialbias", "toraman"}
TEST_DOMAINS  = {"toxigen", "gab", "anatomy"}
LABEL2ID = {"equivalent": 0, "unequivalent": 1}


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["explanation", "synthetic", "meaning", "domain"])
    df["meaning"] = df["meaning"].astype(str).str.lower().str.strip()
    df = df[df["meaning"].isin(LABEL2ID)]
    df["label"] = df["meaning"].map(LABEL2ID).astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1) TF-IDF baseline
# ---------------------------------------------------------------------------
def claim_tfidf(df: pd.DataFrame) -> dict:
    train = df[df["domain"].isin(TRAIN_DOMAINS)]
    test  = df[df["domain"].isin(TEST_DOMAINS)]
    Xtr = train["explanation"].astype(str) + " [SEP] " + train["synthetic"].astype(str)
    Xte = test["explanation"].astype(str)  + " [SEP] " + test["synthetic"].astype(str)
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr_v, train["label"])
    preds = clf.predict(Xte_v)
    return {
        "f1_macro": float(f1_score(test["label"], preds, average="macro")),
        "accuracy": float(accuracy_score(test["label"], preds)),
        "n_train": int(len(train)),
        "n_test":  int(len(test)),
    }


# ---------------------------------------------------------------------------
# 2) GPT-4o-mini already on disk; just echo it
# ---------------------------------------------------------------------------
def claim_gpt4omini() -> dict:
    p = ROOT / "judge_finetune" / "runs" / "zero_shot_gpt4omini.json"
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# 3) RoBERTa OOD from saved predictions
# ---------------------------------------------------------------------------
def claim_roberta_ood() -> dict:
    p = ROOT / "judge_finetune" / "runs" / "roberta_base" / "test_predictions.csv"
    df = pd.read_csv(p)
    out = {
        "n_overall": int(len(df)),
        "accuracy": float(accuracy_score(df["label"], df["pred"])),
        "f1_macro": float(f1_score(df["label"], df["pred"], average="macro")),
        "per_domain": {},
    }
    for dom in sorted(df["domain"].unique()):
        m = df["domain"] == dom
        out["per_domain"][dom] = {
            "n":        int(m.sum()),
            "accuracy": float(accuracy_score(df.loc[m, "label"], df.loc[m, "pred"])),
            "f1_macro": float(f1_score(df.loc[m, "label"], df.loc[m, "pred"], average="macro")),
        }
    return out


# ---------------------------------------------------------------------------
# 4) Cosine similarity shortcut numbers
# ---------------------------------------------------------------------------
def claim_cosine(df: pd.DataFrame) -> dict:
    vec = TfidfVectorizer()
    vec.fit(pd.concat([df["explanation"].astype(str), df["synthetic"].astype(str)]))
    a = vec.transform(df["explanation"].astype(str))
    b = vec.transform(df["synthetic"].astype(str))
    sims = np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(a.shape[0])])
    df = df.assign(cosine_sim=sims)
    out = {}
    for name, sub in (("equivalent", df[df["label"] == 0]),
                      ("unequivalent", df[df["label"] == 1])):
        out[name] = {
            "n":      int(len(sub)),
            "mean":   float(sub["cosine_sim"].mean()),
            "median": float(sub["cosine_sim"].median()),
            "pct_gt_0_9": float((sub["cosine_sim"] > 0.9).mean() * 100),
            "pct_gt_0_8": float((sub["cosine_sim"] > 0.8).mean() * 100),
            "pct_lt_0_5": float((sub["cosine_sim"] < 0.5).mean() * 100),
        }
    return out


def main() -> int:
    df = load_df()
    print(f"[data] {len(df)} rows, label balance: {df['label'].value_counts().to_dict()}")

    results: dict = {}

    print("\n=== (1) TF-IDF baseline ===")
    results["tfidf"] = claim_tfidf(df)
    print(json.dumps(results["tfidf"], indent=2))
    print(f"  poster: 0.789  |  recomputed: {results['tfidf']['f1_macro']:.3f}")

    print("\n=== (2) GPT-4o-mini zero-shot ===")
    results["gpt4omini"] = claim_gpt4omini()
    print(f"  poster: 0.964  |  on-disk f1_macro: {results['gpt4omini']['f1_macro']:.3f}  "
          f"acc: {results['gpt4omini']['accuracy']:.3f}  (n={results['gpt4omini']['n']})")

    print("\n=== (3) RoBERTa OOD (recomputed from saved predictions) ===")
    results["roberta_ood"] = claim_roberta_ood()
    print(json.dumps(results["roberta_ood"], indent=2))
    print(f"  poster: 0.968  |  recomputed: {results['roberta_ood']['f1_macro']:.3f}")

    print("\n=== (4) Cosine-similarity shortcut ===")
    results["cosine"] = claim_cosine(df)
    print(json.dumps(results["cosine"], indent=2))
    print(f"  poster equivalent>0.9   : 21.6%  |  recomputed: {results['cosine']['equivalent']['pct_gt_0_9']:.1f}%")
    print(f"  poster unequivalent>0.9 : 50.9%  |  recomputed: {results['cosine']['unequivalent']['pct_gt_0_9']:.1f}%")

    (ROOT / "verify_poster_results.json").write_text(json.dumps(results, indent=2))
    print("\n[done] results -> verify_poster_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
