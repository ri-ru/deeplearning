"""End-to-end interpretability variance across independently-trained models
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
EXTRACTS = REPO / "seeds" / "extracts"
MANIFEST = EXTRACTS / "manifest.csv"
RESULTS = REPO / "seeds" / "results"
EXTRACT_SCRIPT = REPO / "code" / "interpretability" / "extract_residuals.py"
RUNS = REPO / "seeds" / "runs"

SEEDS = [42, 1, 2]
PROBE_SPLIT_SEED = 42
VAL_FRAC = 0.2
MAX_LENGTH = 128
BATCH = 64


def find_ckpt(seed: int) -> Path | None:
    cps = sorted((RUNS / f"roberta_base_seed{seed}" / "checkpoints").glob("checkpoint-*"))
    return cps[0] if cps else None


def load_layer(d: Path, li: int):
    pt = torch.load(d / f"layer_{li:02d}.pt", weights_only=True)
    return list(pt["sample_ids"]), pt["activations"].numpy()


def probe(extract_dir: Path, manifest: pd.DataFrame) -> list[dict]:
    n_layers = sum(1 for _ in extract_dir.glob("layer_*.pt"))
    canon, _ = load_layer(extract_dir, 0)
    if canon != manifest["sample_id"].astype(str).tolist():
        raise RuntimeError("order != manifest")
    is_s = manifest["source"].to_numpy() == "synthetic"
    is_m = manifest["source"].to_numpy() == "manual"
    y_e = manifest["label"].to_numpy().astype(int)
    y_n = manifest["contains_negation_in_synthetic"].to_numpy().astype(int)
    si, mi = np.where(is_s)[0], np.where(is_m)[0]
    strata = y_e[si] * 2 + y_n[si]
    tr, va = train_test_split(si, test_size=VAL_FRAC, random_state=PROBE_SPLIT_SEED, stratify=strata)
    rows = []
    for li in range(n_layers):
        ids, X = load_layer(extract_dir, li)
        if ids != canon:
            raise RuntimeError(f"order mismatch {li}")
        e = LogisticRegression(C=1.0, max_iter=1000); e.fit(X[tr], y_e[tr])
        pe = e.predict_proba(X[va])[:, 1]; pem = e.predict_proba(X[mi])[:, 1]
        n = LogisticRegression(C=1.0, max_iter=1000); n.fit(X[tr], y_n[tr])
        pn = n.predict_proba(X[va])[:, 1]
        rows.append({"layer": li,
                     "equiv_val": float(roc_auc_score(y_e[va], pe)),
                     "neg_val": float(roc_auc_score(y_n[va], pn)),
                     "pearson": float(np.corrcoef(pe, pn)[0, 1]),
                     "equiv_manual": float(roc_auc_score(y_e[mi], pem)) if len(np.unique(y_e[mi])) == 2 else None})
    return rows


def main():
    manifest = pd.read_csv(MANIFEST).reset_index(drop=True)
    per_model = {}
    for seed in SEEDS:
        ckpt = find_ckpt(seed)
        if ckpt is None:
            print(f"[e2e] no checkpoint for seed {seed} -- skipping", flush=True)
            continue
        ed = EXTRACTS.parent / f"extracts_seed{seed}"
        ed.mkdir(parents=True, exist_ok=True)
        print(f"\n[e2e] seed {seed}: extracting from {ckpt.name} ...", flush=True)
        subprocess.run([sys.executable, str(EXTRACT_SCRIPT), "--checkpoint", str(ckpt),
                        "--out_dir", str(ed), "--manifest", str(MANIFEST),
                        "--max_length", str(MAX_LENGTH), "--batch_size", str(BATCH)], check=True)
        per_model[f"seed{seed}"] = probe(ed, manifest)
        shutil.rmtree(ed, ignore_errors=True)
        print(f"[e2e] seed {seed} done", flush=True)

    names = list(per_model)
    n_layers = len(per_model[names[0]])

    def stats(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        if not xs:
            return None
        return {"mean": round(statistics.mean(xs), 4), "std": round(statistics.pstdev(xs), 4),
                "min": round(min(xs), 4), "max": round(max(xs), 4), "values": [round(x, 4) for x in xs]}

    agg = [{"layer": li,
            "equiv_val": stats([per_model[m][li]["equiv_val"] for m in names]),
            "neg_val": stats([per_model[m][li]["neg_val"] for m in names]),
            "pearson": stats([per_model[m][li]["pearson"] for m in names])} for li in range(n_layers)]
    summary = {"models": names, "probe_split_seed": PROBE_SPLIT_SEED,
               "per_model": per_model, "aggregate_per_layer": agg}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "end2end_interp_sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"\n===== END-TO-END INTERP SWEEP ({len(names)} models) =====")
    for a in agg:
        def f(s): return f"{s['mean']:.4f}+/-{s['std']:.4f}" if s else "n/a"
        print(f"  L{a['layer']:02d}  equiv {f(a['equiv_val']):>16}  neg {f(a['neg_val']):>16}  rho {f(a['pearson']):>16}")
    print(f"\nWrote {RESULTS / 'end2end_interp_sweep.json'}")


if __name__ == "__main__":
    main()
