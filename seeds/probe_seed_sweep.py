"""Probe-split seed sweep (final-submission-2 layout).

Re-fits the equivalence + negation probes on the SAME cached residuals under
N seeds. Residuals are extracted once (from the seed-42 checkpoint) into
seeds/extracts/, so this isolates probe-training (split) variance per layer.

    python seeds/probe_seed_sweep.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
EXTRACTS = REPO / "seeds" / "extracts"
RESULTS = REPO / "seeds" / "results"


def load_layer(li: int):
    pt = torch.load(EXTRACTS / f"layer_{li:02d}.pt", weights_only=True)
    return list(pt["sample_ids"]), pt["activations"].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    manifest = pd.read_csv(EXTRACTS / "manifest.csv").reset_index(drop=True)
    n_layers = sum(1 for _ in EXTRACTS.glob("layer_*.pt"))
    print(f"[sweep] manifest rows {len(manifest)}  layers {n_layers}  seeds {args.seeds}", flush=True)

    canon, _ = load_layer(0)
    if canon != manifest["sample_id"].astype(str).tolist():
        raise RuntimeError("layer-0 order != manifest order")
    is_synth = manifest["source"].to_numpy() == "synthetic"
    is_manual = manifest["source"].to_numpy() == "manual"
    y_e = manifest["label"].to_numpy().astype(int)
    y_n = manifest["contains_negation_in_synthetic"].to_numpy().astype(int)
    synth_idx, manual_idx = np.where(is_synth)[0], np.where(is_manual)[0]
    strata = y_e[synth_idx] * 2 + y_n[synth_idx]

    print("[sweep] loading layers ...", flush=True)
    layer_X = {}
    for li in range(n_layers):
        ids, X = load_layer(li)
        if ids != canon:
            raise RuntimeError(f"order mismatch layer {li}")
        layer_X[li] = X

    per = {li: {"equiv_val": [], "neg_val": [], "pearson": [], "equiv_manual": []} for li in range(n_layers)}
    for seed in args.seeds:
        tr, va = train_test_split(synth_idx, test_size=args.val_frac, random_state=seed, stratify=strata)
        for li in range(n_layers):
            X = layer_X[li]
            e = LogisticRegression(C=1.0, max_iter=1000); e.fit(X[tr], y_e[tr])
            pe_v, pe_m = e.predict_proba(X[va])[:, 1], e.predict_proba(X[manual_idx])[:, 1]
            per[li]["equiv_val"].append(float(roc_auc_score(y_e[va], pe_v)))
            if len(np.unique(y_n[tr])) == 2:
                n = LogisticRegression(C=1.0, max_iter=1000); n.fit(X[tr], y_n[tr])
                pn_v = n.predict_proba(X[va])[:, 1]
                per[li]["neg_val"].append(float(roc_auc_score(y_n[va], pn_v)))
                per[li]["pearson"].append(float(np.corrcoef(pe_v, pn_v)[0, 1]))
            if len(np.unique(y_e[manual_idx])) == 2:
                per[li]["equiv_manual"].append(float(roc_auc_score(y_e[manual_idx], pe_m)))
        print(f"[sweep] seed {seed} done", flush=True)

    def stats(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        if not xs:
            return None
        return {"mean": round(statistics.mean(xs), 4), "std": round(statistics.pstdev(xs), 4),
                "min": round(min(xs), 4), "max": round(max(xs), 4), "n": len(xs)}

    summary = {"seeds": args.seeds, "val_frac": args.val_frac,
               "per_layer": {li: {m: stats(v) for m, v in d.items()} for li, d in per.items()}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "probe_seed_sweep.json").write_text(json.dumps(summary, indent=2))
    print("\n===== PROBE-SPLIT SWEEP (per-layer val AUC, mean +/- std) =====")
    for li in range(n_layers):
        d = summary["per_layer"][li]
        def f(s): return f"{s['mean']:.4f}+/-{s['std']:.4f}" if s else "n/a"
        print(f"  L{li:02d}  equiv {f(d['equiv_val']):>16}  neg {f(d['neg_val']):>16}  rho {f(d['pearson']):>16}")
    print(f"\nWrote {RESULTS / 'probe_seed_sweep.json'}")


if __name__ == "__main__":
    main()
