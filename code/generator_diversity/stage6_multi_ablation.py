"""Stage 6: multi-direction shortcut ablation via supervised PCA.

Stage 2 ablated only the negation-direction (logreg coefficient against the
negation regex). It barely moved manual-AUC because the shortcut is encoded
in MANY parallel directions 
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_common import LABEL2ID, REPO, save_json, has_negation_flip  # noqa: E402

EXTRACTS_DIR = REPO / "overnight" / "extracts"
RESULTS_DIR = REPO / "overnight" / "results" / "stage6"


def build_proxies(manifest: pd.DataFrame) -> pd.DataFrame:
    """For every row of the manifest, compute K shortcut-proxy features.
    Returns a (N x K) dataframe of binary/continuous features.
    """
    df = manifest.copy()
    # 1. Negation flip
    df["proxy_neg_flip"] = [
        has_negation_flip(a, b) for a, b in zip(df["text_a"], df["text_b"])
    ]
    # 2. Length ratio (>1 means b longer than a)
    df["proxy_len_ratio"] = (
        df["text_b"].astype(str).str.len() / df["text_a"].astype(str).str.len().clip(lower=1)
    ).astype(float)
    # 3. text_b starts with "The post"
    df["proxy_b_starts_the_post"] = df["text_b"].astype(str).str.lower().str.startswith("the post").astype(int)
    # 4. text_b contains first-person ("I", "my", "i'd", "i'm")
    fp_re = re.compile(r"\b(i|i'd|i'm|i've|my|me|imo)\b", re.I)
    df["proxy_b_has_first_person"] = df["text_b"].astype(str).apply(
        lambda s: 1 if fp_re.search(s) else 0
    )
    # 5. Whether text_b is shorter than text_a / 2 (human rationales tend to be terse)
    df["proxy_b_much_shorter"] = (
        df["text_b"].astype(str).str.len() < df["text_a"].astype(str).str.len() / 2
    ).astype(int)
    # 6. Has explicit verdict word ("hate", "offensive", "abuse", "normal")
    verdict_re = re.compile(r"\b(hate|offensive|abuse|abusive|normal|neutral|benign)\b", re.I)
    df["proxy_b_has_verdict_word"] = df["text_b"].astype(str).apply(
        lambda s: 1 if verdict_re.search(s) else 0
    )
    cols = [c for c in df.columns if c.startswith("proxy_")]
    return df[cols]


def fit_direction(X_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray | None:
    """Return the logreg coefficient vector for predicting y from X, or
    None if y has only one class."""
    if len(np.unique(y_tr)) < 2:
        return None
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_tr, y_tr)
    v = clf.coef_[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def project_out(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the components of X along each row of the orthonormal `basis`."""
    if basis.shape[0] == 0:
        return X
    # Orthonormalize via QR
    Q, _ = np.linalg.qr(basis.T)  
    return X - X @ Q @ Q.T


def auc(y, p) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--max_K", type=int, default=6)
    parser.add_argument("--target_name", default="baseline",
                        help="subdir under overnight/extracts/")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

 
    extracts_dir = EXTRACTS_DIR / args.target_name
    meta_path = extracts_dir / "meta.json"
    if not meta_path.exists():
        print(f"[stage6] no extracts at {extracts_dir}; rerun stage 2 first.")
        return 1

    # Rebuild manifest using stage2 helper so it matches the extracts order
    sys.path.insert(0, str(REPO / "overnight"))
    from stage2_interp import build_manifest  # noqa: E402
    manifest = build_manifest()
    print(f"[stage6] manifest rows: {len(manifest)}", flush=True)
    proxies = build_proxies(manifest)
    print(f"[stage6] proxies: {list(proxies.columns)}")


    pt = torch.load(str(extracts_dir / f"layer_{args.layer:02d}.pt"), weights_only=False)
    assert pt["sample_ids"] == manifest["sample_id"].astype(str).tolist()
    X = pt["activations"].numpy()
    print(f"[stage6] layer {args.layer}  X shape: {X.shape}", flush=True)

    y_eq = manifest["label"].to_numpy().astype(int)
    is_syn = (manifest["source"] == "synthetic").to_numpy()
    is_man = (manifest["source"] == "manual").to_numpy()
    is_cla = (manifest["source"] == "claude").to_numpy()

    syn_idx = np.where(is_syn)[0]
    rng = np.random.default_rng(42)
    strata = y_eq[syn_idx] * 2 + proxies.iloc[syn_idx, 0].to_numpy()
    tr_idx, va_idx = train_test_split(syn_idx, test_size=0.2, random_state=42, stratify=strata)
    man_idx = np.where(is_man)[0]; cla_idx = np.where(is_cla)[0]

    # Fit each proxy direction once
    directions: list[tuple[str, np.ndarray]] = []
    for col in proxies.columns:
        y_proxy = proxies[col].astype(float).to_numpy()
        # binarize continuous proxies at the median
        if y_proxy.dtype != int and len(np.unique(y_proxy)) > 2:
            y_proxy = (y_proxy > np.median(y_proxy[is_syn])).astype(int)
        else:
            y_proxy = y_proxy.astype(int)
        d = fit_direction(X[tr_idx], y_proxy[tr_idx])
        if d is None:
            print(f"  proxy {col}: skipped (degenerate)")
            continue
        #sanity: how well does this direction decode the proxy on val?
        score = (X[va_idx] @ d)
        try:
            a = roc_auc_score(y_proxy[va_idx], score)
        except Exception:
            a = float("nan")
        print(f"  proxy {col}: AUC vs proxy on val = {a:.3f}")
        directions.append((col, d))

    # Sweep K from 0..len(directions). 

    rows = []
    for K in range(0, len(directions) + 1):
        basis = np.vstack([d for (_, d) in directions[:K]]) if K else np.zeros((0, X.shape[1]))
        Xp = project_out(X, basis)
        eq_clf = LogisticRegression(C=1.0, max_iter=1000)
        eq_clf.fit(Xp[tr_idx], y_eq[tr_idx])
        row = {
            "K": K,
            "directions_ablated": [n for n, _ in directions[:K]],
            "eq_val":    auc(y_eq[va_idx], eq_clf.predict_proba(Xp[va_idx])[:, 1]),
            "eq_manual": auc(y_eq[man_idx], eq_clf.predict_proba(Xp[man_idx])[:, 1]),
            "eq_claude": auc(y_eq[cla_idx], eq_clf.predict_proba(Xp[cla_idx])[:, 1]),
        }
        rows.append(row)
        print(f"  K={K}  val={row['eq_val']:.3f}  manual={row['eq_manual']:.3f}  claude={row['eq_claude']:.3f}  "
              f"  ({','.join(row['directions_ablated'])})", flush=True)

    out = {
        "target": args.target_name,
        "layer": args.layer,
        "n_syn_train": int(len(tr_idx)), "n_syn_val": int(len(va_idx)),
        
        "n_manual": int(len(man_idx)), "n_claude": int(len(cla_idx)),

        "directions_used": [n for n, _ in directions],
        "sweep": rows,
    }
    save_json(RESULTS_DIR / f"{args.target_name}_layer{args.layer}.json", out)
    print(f"[stage6] saved {args.target_name}_layer{args.layer}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
