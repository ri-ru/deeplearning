"""Stage 2: per-layer probe + negation-direction ablation.

For the baseline checkpoint (and, if available, the best stage-1 mitigation):
    - extract [CLS] residuals for synthetic + manual + Claude (if not cached)
    - for each layer 0..12 train two probe types: linear logreg, 2-layer MLP.
    - compute negation direction (logreg coefficient against contains_negation),
      ablate it (project it out of [CLS]), retrain the equivalence probe.
      Report the AUC drop.

"shortcut tax" per layer = val_AUC - manual_AUC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, has_negation_flip, load_full_csv, load_human_csv, save_json,
)

warnings.filterwarnings("ignore", category=FutureWarning)

EXTRACTS_DIR = REPO / "overnight" / "extracts"
RESULTS_DIR = REPO / "overnight" / "results" / "stage2"


def build_manifest() -> pd.DataFrame:
    syn = load_full_csv().rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    syn = syn[["text_a", "text_b", "label", "domain", "uid"]].copy()
    syn["source"] = "synthetic"
    syn["sample_id"] = syn["uid"].astype(str)
    syn = syn.drop(columns=["uid"])

    manual = load_human_csv(REPO / "manual labelling.csv")
    manual_df = pd.DataFrame([{
        "text_a": r["explanation"], "text_b": r["comment"],
        "label": LABEL2ID[r["label"]], "domain": "manual",
        "source": "manual", "sample_id": f"manual_{i:03d}",
    } for i, r in enumerate(manual)])

    claude = load_human_csv(REPO / "roberta_training" / "claude_test_labels.csv")
    claude_df = pd.DataFrame([{
        "text_a": r["explanation"], "text_b": r["comment"],
        "label": LABEL2ID[r["label"]], "domain": "claude",
        "source": "claude", "sample_id": f"claude_{i:03d}",
    } for i, r in enumerate(claude)])

    manifest = pd.concat([syn, manual_df, claude_df], ignore_index=True)
    manifest["neg_flip"] = [
        has_negation_flip(a, b) for a, b in zip(manifest["text_a"], manifest["text_b"])
    ]
    return manifest


def extract_residuals(checkpoint_dir: Path, manifest: pd.DataFrame, out_dir: Path,
                      max_length: int = 128, batch_size: int = 64) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and (out_dir / "layer_12.pt").exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("n_samples") == len(manifest):
            print(f"[extract] reusing cached residuals at {out_dir}", flush=True)
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), output_hidden_states=True,
    ).to(device)
    model.train(False)

    n = len(manifest)
    text_a = manifest["text_a"].astype(str).tolist()
    text_b = manifest["text_b"].astype(str).tolist()
    sample_ids = manifest["sample_id"].astype(str).tolist()

    with torch.no_grad():
        enc = tok(text_a[:min(batch_size, n)], text_b[:min(batch_size, n)],
                  padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
    n_layers = len(out.hidden_states)
    hidden = out.hidden_states[0].shape[-1]
    buffers = [torch.empty(n, hidden, dtype=torch.float32) for _ in range(n_layers)]

    cursor = 0; t0 = time.time()
    while cursor < n:
        end = min(cursor + batch_size, n)
        enc = tok(text_a[cursor:end], text_b[cursor:end],
                  padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        for li, h in enumerate(out.hidden_states):
            buffers[li][cursor:end] = h[:, 0, :].detach().to(torch.float32).cpu()
        cursor = end
        if cursor % (batch_size * 50) == 0 or cursor == n:
            rate = cursor / max(time.time() - t0, 1e-6)
            print(f"[extract] {cursor}/{n}  ({rate:.0f}/s)", flush=True)

    for li, buf in enumerate(buffers):
        torch.save({"layer": li, "sample_ids": sample_ids, "activations": buf},
                   out_dir / f"layer_{li:02d}.pt")
    meta_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_dir),
        "n_samples": n, "n_layers": n_layers, "hidden_dim": hidden,
    }, indent=2))


def project_out(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    d = direction / (np.linalg.norm(direction) + 1e-12)
    coeffs = X @ d
    return X - np.outer(coeffs, d)


def _auc_safe(y, p) -> float | None:
    try:
        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def probe_layer(X: np.ndarray, y_eq: np.ndarray, y_neg: np.ndarray,
                src_mask: dict[str, np.ndarray]) -> dict:
    is_syn = src_mask["synthetic"]
    is_man = src_mask["manual"]
    is_cla = src_mask["claude"]

    strata = y_eq[is_syn] * 2 + y_neg[is_syn]
    syn_idx = np.where(is_syn)[0]
    tr_idx, va_idx = train_test_split(syn_idx, test_size=0.2, random_state=42, stratify=strata)
    man_idx = np.where(is_man)[0]
    cla_idx = np.where(is_cla)[0]

    out: dict = {}

    lin = LogisticRegression(C=1.0, max_iter=1000)
    lin.fit(X[tr_idx], y_eq[tr_idx])
    out["linear_eq"] = {
        "val":    float(roc_auc_score(y_eq[va_idx], lin.predict_proba(X[va_idx])[:, 1])),
        "manual": _auc_safe(y_eq[man_idx], lin.predict_proba(X[man_idx])[:, 1]),
        "claude": _auc_safe(y_eq[cla_idx], lin.predict_proba(X[cla_idx])[:, 1]),
    }
    if len(np.unique(y_neg[tr_idx])) == 2:
        lin_neg = LogisticRegression(C=1.0, max_iter=1000)
        lin_neg.fit(X[tr_idx], y_neg[tr_idx])
        out["linear_neg_val"] = float(roc_auc_score(y_neg[va_idx], lin_neg.predict_proba(X[va_idx])[:, 1]))
        direction = lin_neg.coef_[0]
        X_abl = project_out(X, direction)
        lin_ab = LogisticRegression(C=1.0, max_iter=1000)
        lin_ab.fit(X_abl[tr_idx], y_eq[tr_idx])
        out["linear_eq_ablated"] = {
            "val":    float(roc_auc_score(y_eq[va_idx], lin_ab.predict_proba(X_abl[va_idx])[:, 1])),
            "manual": _auc_safe(y_eq[man_idx], lin_ab.predict_proba(X_abl[man_idx])[:, 1]),
            "claude": _auc_safe(y_eq[cla_idx], lin_ab.predict_proba(X_abl[cla_idx])[:, 1]),
        }
    else:
        out["linear_neg_val"] = None
        out["linear_eq_ablated"] = None

    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42,
                        early_stopping=True, n_iter_no_change=5)
    mlp.fit(X[tr_idx], y_eq[tr_idx])
    out["mlp_eq"] = {
        "val":    float(roc_auc_score(y_eq[va_idx], mlp.predict_proba(X[va_idx])[:, 1])),
        "manual": _auc_safe(y_eq[man_idx], mlp.predict_proba(X[man_idx])[:, 1]),
        "claude": _auc_safe(y_eq[cla_idx], mlp.predict_proba(X[cla_idx])[:, 1]),
    }

    out["shortcut_tax_linear"] = (out["linear_eq"]["val"]
                                  - (out["linear_eq"]["manual"] or 0.0))
    return out


def run_one(name: str, checkpoint_dir: Path) -> dict:
    print(f"[stage2] {name}: extracting residuals ...", flush=True)
    manifest = build_manifest()
    extract_dir = EXTRACTS_DIR / name
    extract_residuals(checkpoint_dir, manifest, extract_dir)

    y_eq = manifest["label"].to_numpy().astype(int)
    y_neg = manifest["neg_flip"].to_numpy().astype(int)
    src_mask = {
        "synthetic": (manifest["source"] == "synthetic").to_numpy(),
        "manual":    (manifest["source"] == "manual").to_numpy(),
        "claude":    (manifest["source"] == "claude").to_numpy(),
    }

    by_layer: list[dict] = []
    n_layers = len(list(extract_dir.glob("layer_*.pt")))
    for li in range(n_layers):
        pt = torch.load(str(extract_dir / f"layer_{li:02d}.pt"), weights_only=False)
        assert pt["sample_ids"] == manifest["sample_id"].astype(str).tolist(), \
            f"layer {li} sample_id order mismatch"
        X = pt["activations"].numpy()
        print(f"[stage2] {name} layer {li:02d} ...", flush=True)
        res = probe_layer(X, y_eq, y_neg, src_mask)
        res["layer"] = li
        by_layer.append(res)

    return {
        "checkpoint": str(checkpoint_dir),
        "n_synthetic": int(src_mask["synthetic"].sum()),
        "n_manual":    int(src_mask["manual"].sum()),
        "n_claude":    int(src_mask["claude"].sum()),
        "by_layer": by_layer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_best_mitigation", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    targets: list[tuple[str, Path]] = []

    baseline_ckpt = REPO / "judge_finetune" / "runs" / "roberta_base" / "checkpoints" / "checkpoint-1576"
    if baseline_ckpt.exists():
        targets.append(("baseline", baseline_ckpt))

    if args.include_best_mitigation:
        stage1_results = sorted((REPO / "overnight" / "results" / "stage1").glob("v*.json"))
        best = None; best_acc = -1.0
        for p in stage1_results:
            try:
                d = json.loads(p.read_text())
                acc = d["manual"]["accuracy"]
                if acc > best_acc:
                    best_acc = acc; best = d["_meta"]["checkpoint"]
            except Exception:
                continue
        if best is not None and Path(best).exists():
            targets.append(("best_mitigation", Path(best)))

    for name, ckpt in targets:
        result_path = RESULTS_DIR / f"{name}.json"
        if result_path.exists():
            print(f"[stage2] {name}: result exists, skipping", flush=True)
            continue
        try:
            res = run_one(name, ckpt)
            save_json(result_path, res)
            print(f"[stage2] {name}: done -> {result_path}", flush=True)
        except Exception as e:
            (RESULTS_DIR / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            print(f"[stage2] {name}: FAILED: {e}", flush=True)

    print("\n[stage2] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
