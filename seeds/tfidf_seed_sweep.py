"""TF-IDF + LogReg baseline — seed-variance sweep (final-submission-2 layout).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code" / "judge_training"))
import data as D  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
CSV = str(REPO / "data" / "reformulated_full.csv")
TEST_DOMAINS = ["anatomy", "gab", "toxigen"]
RESULTS = REPO / "seeds" / "results"


def run_one(seed: int) -> dict:
    train_ds, _v, test_ds = D.load_and_split(D.DataConfig(csv_path=CSV, seed=seed))
    tr, te = train_ds.to_pandas(), test_ds.to_pandas()
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr["text_a"] + " [SEP] " + tr["text_b"])
    Xte = vec.transform(te["text_a"] + " [SEP] " + te["text_b"])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"])
    pred = clf.predict(Xte)
    res = {"overall": {"accuracy": float(accuracy_score(te["label"], pred)),
                       "f1_macro": float(f1_score(te["label"], pred, average="macro"))}}
    for dom in TEST_DOMAINS:
        m = (te["domain"] == dom).to_numpy()
        res[dom] = {"accuracy": float(accuracy_score(te["label"][m], pred[m])),
                    "f1_macro": float(f1_score(te["label"][m], pred[m], average="macro"))}
    return res


def stats(xs):
    return {"values": [round(x, 4) for x in xs], "mean": round(statistics.mean(xs), 4),
            "std": round(statistics.pstdev(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    per = {}
    for s in SEEDS:
        print(f"[tfidf] seed {s} ...", flush=True)
        per[s] = run_one(s)
    keys = ["overall"] + TEST_DOMAINS
    summary = {"seeds": SEEDS}
    for k in keys:
        summary[k] = {"accuracy": stats([per[s][k]["accuracy"] for s in SEEDS]),
                      "f1_macro": stats([per[s][k]["f1_macro"] for s in SEEDS])}
    (RESULTS / "tfidf_seed_sweep.json").write_text(json.dumps(summary, indent=2))
    print("\n===== TF-IDF SEED SWEEP (OOD test) =====")
    for k in keys:
        s = summary[k]["f1_macro"]
        print(f"  {k:10s} f1_macro {s['mean']:.4f} +/- {s['std']:.4f}  {s['values']}")
    print(f"\nWrote {RESULTS / 'tfidf_seed_sweep.json'}")


if __name__ == "__main__":
    main()
