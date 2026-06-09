"""Per-layer linear probes on [CLS] residuals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "mech_interp" / "extracts" / "manifest.csv"
DEFAULT_EXTRACTS = REPO / "mech_interp" / "extracts"
DEFAULT_RESULTS = REPO / "mech_interp" / "results"
DEFAULT_FIGS = REPO / "mech_interp" / "figures"


def _load_layer(extracts_dir: Path, layer_idx: int) -> tuple[list[str], np.ndarray]:
    pt = torch.load(extracts_dir / f"layer_{layer_idx:02d}.pt", weights_only=False)
    return list(pt["sample_ids"]), pt["activations"].numpy()


def _align_positional(manifest: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    """Layer files were saved in manifest row order by extract_residuals.py.
    sample_id is NOT unique in the manifest (each source uid has ~3
    reformulations), so we cannot align by sample_id. Verify the layer's
    sample_id list equals the manifest's column-by-position and return
    the manifest unchanged."""
    expected = manifest["sample_id"].astype(str).tolist()
    if sample_ids != expected:
        raise RuntimeError(
            "Layer-file sample_id order does not match manifest row order. "
            "Re-run extract_residuals.py against the current manifest."
        )
    return manifest.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extracts", type=Path, default=DEFAULT_EXTRACTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    print(f"[probes] manifest rows: {len(manifest)}")
    n_layers = sum(1 for _ in args.extracts.glob("layer_*.pt"))
    if n_layers == 0:
        raise FileNotFoundError(f"No layer_*.pt files under {args.extracts}")
    print(f"[probes] found {n_layers} layer files")

 
    canon_ids, _ = _load_layer(args.extracts, 0)
    aligned = _align_positional(manifest, canon_ids)
    is_synth = aligned["source"].to_numpy() == "synthetic"
    is_manual = aligned["source"].to_numpy() == "manual"
    y_equiv = aligned["label"].to_numpy().astype(int)
    y_neg = aligned["contains_negation_in_synthetic"].to_numpy().astype(int)

    #  regex AUC vs label
    prior_auc = roc_auc_score(y_equiv[is_synth], y_neg[is_synth])
    print(f"[probes] regex-only AUC vs label (synthetic): {prior_auc:.4f}")

    # Stratified train/val split on synth index
    synth_idx = np.where(is_synth)[0]
    rng = np.random.default_rng(args.seed)
    # Stratify on (label, negation)  
    strata = y_equiv[synth_idx] * 2 + y_neg[synth_idx]
    train_idx, val_idx = train_test_split(
        synth_idx, test_size=args.val_frac, random_state=args.seed, stratify=strata,
    )
    manual_idx = np.where(is_manual)[0]
    print(f"[probes] split: train={len(train_idx)}  val={len(val_idx)}  manual={len(manual_idx)}")

    metrics: list[dict] = []
    for li in range(n_layers):
        ids_l, X = _load_layer(args.extracts, li)
        if ids_l != canon_ids:
            raise RuntimeError(f"sample_id order mismatch at layer {li}")

        X_tr = X[train_idx]
        X_va = X[val_idx]
        X_ma = X[manual_idx]
        y_tr_e, y_va_e, y_ma_e = y_equiv[train_idx], y_equiv[val_idx], y_equiv[manual_idx]
        y_tr_n, y_va_n, y_ma_n = y_neg[train_idx], y_neg[val_idx], y_neg[manual_idx]

        equiv = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=-1)
        equiv.fit(X_tr, y_tr_e)
        p_va_e = equiv.predict_proba(X_va)[:, 1]
        p_ma_e = equiv.predict_proba(X_ma)[:, 1]

        # probe_neg only makes sense if both classes are present in train.
        if len(np.unique(y_tr_n)) == 2:
            neg = LogisticRegression(C=args.C, max_iter=args.max_iter, n_jobs=-1)
            neg.fit(X_tr, y_tr_n)
            p_va_n = neg.predict_proba(X_va)[:, 1]
            p_ma_n = neg.predict_proba(X_ma)[:, 1]
            auc_va_n = roc_auc_score(y_va_n, p_va_n)
            auc_ma_n = roc_auc_score(y_ma_n, p_ma_n) if len(np.unique(y_ma_n)) == 2 else None
            pearson = float(np.corrcoef(p_va_e, p_va_n)[0, 1])
        else:
            p_va_n = p_ma_n = None
            auc_va_n = auc_ma_n = None
            pearson = None

        auc_va_e = roc_auc_score(y_va_e, p_va_e)
        auc_ma_e = roc_auc_score(y_ma_e, p_ma_e) if len(np.unique(y_ma_e)) == 2 else None

        row = {
            "layer": li,
            "probe_equiv_auc_val": float(auc_va_e),
            "probe_neg_auc_val": float(auc_va_n) if auc_va_n is not None else None,
            "probe_equiv_auc_manual": float(auc_ma_e) if auc_ma_e is not None else None,
            "probe_neg_auc_manual": float(auc_ma_n) if auc_ma_n is not None else None,
            "pred_pearson_equiv_neg_val": pearson,
        }
        metrics.append(row)
        msg = (
            f"layer {li:02d}  "
            f"equiv_val={row['probe_equiv_auc_val']:.4f}  "
            f"neg_val={row['probe_neg_auc_val']!s:>8}  "
            f"ρ={row['pred_pearson_equiv_neg_val']!s:>8}  "
            f"equiv_manual={row['probe_equiv_auc_manual']!s:>8}  "
            f"neg_manual={row['probe_neg_auc_manual']!s:>8}"
        )
        print(msg)

    payload = {
        "prior_regex_auc_synthetic": float(prior_auc),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_manual": int(len(manual_idx)),
        "metrics_by_layer": metrics,
    }
    out_json = args.results / "probe_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[probes] wrote {out_json}")

    # --- AUC sweep
    layers = [m["layer"] for m in metrics]
    auc_e = [m["probe_equiv_auc_val"] for m in metrics]
    auc_n = [m["probe_neg_auc_val"] for m in metrics]
    auc_e_man = [m["probe_equiv_auc_manual"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, auc_e, "o-", label="probe_equiv (val, synthetic)", linewidth=2)
    ax.plot(layers, auc_n, "s-", label="probe_neg (val, synthetic)", linewidth=2)
    if any(v is not None for v in auc_e_man):
        ax.plot(
            layers,
            [v if v is not None else np.nan for v in auc_e_man],
            "x--", label="probe_equiv (manual, n=20)", alpha=0.8,
        )
    ax.axhline(prior_auc, color="grey", linestyle=":", label=f"regex-only prior AUC = {prior_auc:.3f}")
    ax.axhline(0.5, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Layer index (0 = embeddings, 1..12 = transformer blocks)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-layer linear probes on [CLS] residual stream")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(layers)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = args.figures / "probe_auc_layer_sweep.png"
    fig.savefig(out_fig, dpi=140)
    plt.close(fig)
    print(f"[probes] wrote {out_fig}")

    # --- Prediction correlation
    rho = [m["pred_pearson_equiv_neg_val"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, rho, "o-", color="crimson", linewidth=2)
    ax.axhline(0.85, color="black", linestyle="--", label="H1 threshold (ρ=0.85)")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Pearson ρ (probe_equiv prob, probe_neg prob)")
    ax.set_title("Per-layer prediction correlation between probes (synthetic val)")
    ax.set_ylim(-0.1, 1.02)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_fig2 = args.figures / "probe_prediction_correlation.png"
    fig.savefig(out_fig2, dpi=140)
    plt.close(fig)
    print(f"[probes] wrote {out_fig2}")

    # --- Shortcut localization summary
    rho_vals = [r for r in rho if r is not None]
    if rho_vals:
        peak_layer = int(np.argmax([r if r is not None else -1 for r in rho]))
        peak_rho = rho[peak_layer]
        verdict = "SHORTCUT-LOCALIZED" if (peak_rho is not None and peak_rho > 0.85) else "no clear localization"
        print(f"[verdict] peak ρ at layer {peak_layer}: {peak_rho:.4f}  ({verdict})")


if __name__ == "__main__":
    main()
