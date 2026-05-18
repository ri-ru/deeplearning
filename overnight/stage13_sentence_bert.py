"""Stage 13: Sentence-BERT pair-similarity baseline.

The "purpose-built" baseline for equivalence: a pretrained sentence-embedding
model + cosine similarity + a tuned threshold. No fine-tuning. If this beats
or matches our fine-tuned classifiers, we should be using it as the baseline
all along.

Uses sentence-transformers/all-mpnet-base-v2 (a strong general-purpose
sentence encoder). The model is downloaded on first use.

Pipeline:
  1. Encode both rationales of every training/eval pair via mpnet.
  2. Compute cosine similarity per pair.
  3. Sweep threshold on the multi_gen training pool to find the best one.
  4. Apply that threshold to each held-out eval (stage7, stage10, manual+claude).
  5. Also try a tiny LogReg on cosine + length difference + word overlap features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import LABEL2ID, REPO, load_human_csv, save_json  # noqa: E402

RESULTS_DIR = REPO / "overnight" / "results" / "stage13"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage13] loading {MODEL_NAME} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.train(False)
    return tok, model, device


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    s = (last_hidden_state * mask).sum(dim=1)
    n = mask.sum(dim=1).clamp(min=1e-9)
    return s / n


def encode(texts: list[str], tok, model, device, batch_size: int = 64) -> np.ndarray:
    embs = []
    for st in range(0, len(texts), batch_size):
        batch = texts[st: st + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        e = mean_pool(out.last_hidden_state, enc["attention_mask"])
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        embs.append(e.cpu().numpy())
    return np.vstack(embs)


def cosine_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=1)


def best_threshold_for_f1(sims: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """labels: 0=equivalent, 1=unequivalent. Higher cosine -> more likely equivalent.
    So we threshold sims and predict eq if sim>=t, uneq otherwise.
    """
    candidates = np.linspace(sims.min(), sims.max(), 201)
    best_t = candidates[0]; best_f1 = -1.0
    for t in candidates:
        preds = (sims < t).astype(int)  # below threshold = unequivalent
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_t = t
    return float(best_t), float(best_f1)


def evaluate_threshold(sims: np.ndarray, labels: np.ndarray, t: float) -> dict[str, Any]:
    preds = (sims < t).astype(int)
    out = {
        "n": int(len(labels)),
        "acc": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "acc_equivalent": float((preds[labels == 0] == 0).mean()) if (labels == 0).any() else None,
        "acc_unequivalent": float((preds[labels == 1] == 1).mean()) if (labels == 1).any() else None,
    }
    try:
        # Higher sim -> more "equivalent". For AUROC of "unequivalent" prediction,
        # the score is -sim (so larger score = more likely unequivalent).
        out["auroc"] = float(roc_auc_score(labels, -sims))
    except Exception:
        out["auroc"] = None
    return out


def evaluate_logreg(features_train, labels_train, features_test, labels_test) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(features_train, labels_train)
    preds = clf.predict(features_test)
    proba = clf.predict_proba(features_test)[:, 1]
    out = {
        "n": int(len(labels_test)),
        "acc": float(accuracy_score(labels_test, preds)),
        "f1_macro": float(f1_score(labels_test, preds, average="macro", zero_division=0)),
        "acc_equivalent": float((preds[labels_test == 0] == 0).mean()) if (labels_test == 0).any() else None,
        "acc_unequivalent": float((preds[labels_test == 1] == 1).mean()) if (labels_test == 1).any() else None,
    }
    try:
        out["auroc"] = float(roc_auc_score(labels_test, proba))
    except Exception:
        out["auroc"] = None
    return out


def build_features(a_emb, b_emb, text_a, text_b) -> np.ndarray:
    cos = cosine_pair(a_emb, b_emb).reshape(-1, 1)
    len_diff = np.array([abs(len(a) - len(b)) for a, b in zip(text_a, text_b)]).reshape(-1, 1)
    # token overlap (jaccard)
    def jac(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(len(sa | sb), 1)
    overlap = np.array([jac(a, b) for a, b in zip(text_a, text_b)]).reshape(-1, 1)
    return np.hstack([cos, len_diff, overlap])


def load_eval_sets() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # Stage 10 (100 Claude pairs)
    p = REPO / "overnight" / "results" / "stage10" / "claude_human_eval.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["label_verified"].isin(LABEL2ID)].copy()
        df["label"] = df["label_verified"].map(LABEL2ID).astype(int)
        df = df.rename(columns={"explanation": "text_a", "comment": "text_b"})
        out["stage10_human_n100"] = df[["text_a", "text_b", "label"]].reset_index(drop=True)

    # Stage 7 (50 GPT pairs)
    p = REPO / "overnight" / "results" / "stage7" / "harder_eval.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["label_verified"].isin(LABEL2ID)].copy()
        df["label"] = df["label_verified"].map(LABEL2ID).astype(int)
        df = df.rename(columns={"explanation": "text_a", "comment": "text_b"})
        out["stage7_gpt_n50"] = df[["text_a", "text_b", "label"]].reset_index(drop=True)

    # manual + claude (40 humans)
    rows = []
    for src in [REPO / "manual labelling.csv",
                REPO / "roberta_training" / "claude_test_labels.csv"]:
        for r in load_human_csv(src):
            rows.append({"text_a": r["explanation"], "text_b": r["comment"],
                         "label": LABEL2ID[r["label"]]})
    out["manual_plus_claude_n40"] = pd.DataFrame(rows)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tok, model, device = load_encoder()

    # Build a calibration set from the multi_gen training pool
    cross_dir = REPO / "overnight" / "cross_gen"
    train_rows = []
    for p in sorted(cross_dir.glob("*.csv")):
        if p.stem.startswith("_"):
            continue
        try:
            sub = pd.read_csv(p)
        except Exception:
            continue
        sub = sub.dropna(subset=["explanation", "synthetic", "meaning"]).copy()
        sub = sub[sub["meaning"].isin(LABEL2ID)].copy()
        if len(sub) == 0:
            continue
        sub["label"] = sub["meaning"].map(LABEL2ID).astype(int)
        sub = sub.rename(columns={"explanation": "text_a", "synthetic": "text_b"})
        train_rows.append(sub[["text_a", "text_b", "label"]])
    train_df = pd.concat(train_rows, ignore_index=True)
    print(f"[stage13] calibration pool: {len(train_df)} pairs", flush=True)

    # Encode
    train_a = encode(train_df["text_a"].astype(str).tolist(), tok, model, device, args.batch_size)
    train_b = encode(train_df["text_b"].astype(str).tolist(), tok, model, device, args.batch_size)
    train_sims = cosine_pair(train_a, train_b)
    print(f"[stage13] training cosine stats: mean={train_sims.mean():.3f} median={np.median(train_sims):.3f} "
          f"std={train_sims.std():.3f}", flush=True)

    # Find best threshold on training
    best_t, best_f1 = best_threshold_for_f1(train_sims, train_df["label"].to_numpy())
    print(f"[stage13] best cosine threshold (eq if sim >= t): t={best_t:.3f}  train f1={best_f1:.3f}", flush=True)

    # Also fit LogReg on (cos, len_diff, overlap)
    train_feat = build_features(train_a, train_b,
                                train_df["text_a"].astype(str).tolist(),
                                train_df["text_b"].astype(str).tolist())

    # Evaluate on every set
    eval_sets = load_eval_sets()
    results: dict[str, Any] = {"threshold": best_t, "train_n": int(len(train_df))}
    for name, df in eval_sets.items():
        print(f"\n[stage13] eval {name}: n={len(df)}", flush=True)
        a = encode(df["text_a"].astype(str).tolist(), tok, model, device, args.batch_size)
        b = encode(df["text_b"].astype(str).tolist(), tok, model, device, args.batch_size)
        sims = cosine_pair(a, b)
        thr_res = evaluate_threshold(sims, df["label"].to_numpy(), best_t)
        feats = build_features(a, b, df["text_a"].astype(str).tolist(), df["text_b"].astype(str).tolist())
        lr_res = evaluate_logreg(train_feat, train_df["label"].to_numpy(), feats, df["label"].to_numpy())
        results[name] = {
            "n": int(len(df)),
            "cosine_threshold": thr_res,
            "logreg_3feat": lr_res,
            "cosine_stats": {"mean": float(sims.mean()), "median": float(np.median(sims)),
                             "std": float(sims.std())},
        }
        print(f"  cosine@t={best_t:.3f}  acc={thr_res['acc']:.3f}  f1={thr_res['f1_macro']:.3f}  auroc={thr_res['auroc']:.3f}", flush=True)
        print(f"  logreg(3f)        acc={lr_res['acc']:.3f}  f1={lr_res['f1_macro']:.3f}  auroc={lr_res['auroc']:.3f}", flush=True)

    save_json(RESULTS_DIR / "sentence_bert.json", results)

    # OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {ts}  STAGE 13: Sentence-BERT pair-similarity baseline\n",
        f"Encoder: `{MODEL_NAME}`. Cosine threshold tuned on multi_gen training pool "
        f"(n={results['train_n']}), t={best_t:.3f}. Also LogReg on (cosine, len_diff, jaccard).",
        "",
        "| eval set | method | acc | f1_macro | AUROC | eq / uneq |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        if not isinstance(res, dict) or "cosine_threshold" not in res:
            continue
        for method, sub in (("cosine threshold", res["cosine_threshold"]),
                            ("logreg(3 feats)", res["logreg_3feat"])):
            lines.append(
                f"| {name} | {method} | {sub['acc']*100:.1f}% | {sub['f1_macro']:.3f} | "
                f"{sub.get('auroc',0):.3f} | "
                f"{sub.get('acc_equivalent',0)*100:.0f}%/{sub.get('acc_unequivalent',0)*100:.0f}% |"
            )
    lines.append("")
    lines.append("Reference: multi_gen on Stage 10 = 81% / 0.806 / 0.896, "
                 "poster_baseline = 59% / 0.583 / 0.745.")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[stage13] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
