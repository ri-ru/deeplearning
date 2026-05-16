"""Attention probe over per-token residuals.

Adapted from mechhack/starter_code/train_probe.py. Trains a single-head
learned-query attention probe per chosen layer. The synthetic split trains
and validates; the manual and claude splits are held out for analysis.

For each test sample we save the per-token attention weights alpha (which
tokens the probe pooled over to make its equiv vs unequiv call). That's the
"what is RoBERTa looking at" signal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTS = REPO / "mech_interp" / "token_extracts"
DEFAULT_OUT = REPO / "mech_interp" / "probe_runs"


class AttentionProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, d), mask: (B, T) bool
        d = x.shape[-1]
        logits = (x @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", alpha, x)
        return self.head(pooled).squeeze(-1), alpha


def load_split(extracts_dir: Path):
    """Group .pt files by source and stash everything in memory."""
    by_source = {"synthetic": [], "manual": [], "claude": []}
    for p in sorted(extracts_dir.glob("*.pt")):
        d = torch.load(str(p), weights_only=False)
        by_source[d["source"]].append(d)
    return by_source


def pad_batch(samples: list[dict], layer_pos: int, device, dtype=torch.float32):
    """Pad variable-length residuals to a single (B, T_max, d) tensor."""
    B = len(samples)
    Tmax = max(s["n_tokens"] for s in samples)
    d = samples[0]["residuals"].shape[-1]
    x = torch.zeros(B, Tmax, d, dtype=dtype, device=device)
    m = torch.zeros(B, Tmax, dtype=torch.bool, device=device)
    y = torch.zeros(B, dtype=torch.float32, device=device)
    for i, s in enumerate(samples):
        T = s["n_tokens"]
        r = s["residuals"][layer_pos].to(dtype)  # (T, d)
        x[i, :T] = r.to(device)
        m[i, :T] = s["attention_mask"][:T].to(device)
        y[i] = float(s["label"])
    return x, m, y


def train_probe(train_samples, val_samples, layer_pos, d, device, seed, epochs, lr, batch_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe = AttentionProbe(d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    best_state = None
    best_auc = -1.0

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_samples))

    for epoch in range(epochs):
        rng.shuffle(idx)
        probe.train()
        epoch_loss = 0.0
        n_batches = 0
        for st in range(0, len(idx), batch_size):
            bi = idx[st: st + batch_size]
            batch = [train_samples[i] for i in bi]
            x, m, y = pad_batch(batch, layer_pos, device)
            logits, _ = probe(x, m)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        # val
        probe.eval()
        with torch.no_grad():
            x, m, y = pad_batch(val_samples, layer_pos, device)
            logits, _ = probe(x, m)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_np = y.cpu().numpy().astype(int)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_np, probs) if len(set(y_np)) > 1 else float("nan")
        except ImportError:
            auc = float("nan")
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone().cpu() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_auc


def evaluate_set(probe, samples, layer_pos, device):
    probe.eval()
    with torch.no_grad():
        x, m, y = pad_batch(samples, layer_pos, device)
        logits, alpha = probe(x, m)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs, alpha.cpu().numpy(), y.cpu().numpy().astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracts_dir", type=Path, default=DEFAULT_EXTRACTS)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"[probe] device={device}  extracts={args.extracts_dir}")

    by_source = load_split(args.extracts_dir)
    synth = by_source["synthetic"]
    manual = by_source["manual"]
    claude = by_source["claude"]

    # Stratified train/val split on synthetic
    rng = np.random.default_rng(args.seed)
    pos = [s for s in synth if s["label"] == 1]
    neg = [s for s in synth if s["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_val = int(len(pos) * args.val_frac)
    val = pos[:n_val] + neg[:n_val]
    train = pos[n_val:] + neg[n_val:]
    rng.shuffle(val)
    rng.shuffle(train)
    print(f"[probe] synthetic train={len(train)} val={len(val)} | manual={len(manual)} claude={len(claude)}")

    # Layers we extracted live at positions 0..L-1 inside each sample's
    # residuals tensor (n_layers, T, d) and correspond to layer_idxs.
    layer_idxs = synth[0]["layer_idxs"]
    d = synth[0]["residuals"].shape[-1]

    summary = {"layers": layer_idxs, "results": []}

    for layer_pos, layer_idx in enumerate(layer_idxs):
        print(f"\n[layer {layer_idx}]  position={layer_pos}/{len(layer_idxs)}")
        probe, best_val_auc = train_probe(
            train, val, layer_pos, d, device,
            seed=args.seed, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )

        # eval on each split + save per-sample alpha
        probs_val,    alpha_val,    y_val    = evaluate_set(probe, val,    layer_pos, device)
        probs_manual, alpha_manual, y_manual = evaluate_set(probe, manual, layer_pos, device)
        probs_claude, alpha_claude, y_claude = evaluate_set(probe, claude, layer_pos, device)

        from sklearn.metrics import roc_auc_score, accuracy_score
        val_auc    = roc_auc_score(y_val, probs_val)            if len(set(y_val)) > 1 else float("nan")
        manual_auc = roc_auc_score(y_manual, probs_manual)      if len(set(y_manual)) > 1 else float("nan")
        claude_auc = roc_auc_score(y_claude, probs_claude)      if len(set(y_claude)) > 1 else float("nan")
        val_acc    = accuracy_score(y_val,    (probs_val    > 0.5).astype(int))
        manual_acc = accuracy_score(y_manual, (probs_manual > 0.5).astype(int))
        claude_acc = accuracy_score(y_claude, (probs_claude > 0.5).astype(int))
        print(f"  val   AUC={val_auc:.3f}  acc={val_acc:.3f}")
        print(f"  manual AUC={manual_auc:.3f}  acc={manual_acc:.3f}")
        print(f"  claude AUC={claude_auc:.3f}  acc={claude_acc:.3f}")

        summary["results"].append({
            "layer": layer_idx,
            "val_auc":   float(val_auc),    "val_acc":   float(val_acc),
            "manual_auc": float(manual_auc), "manual_acc": float(manual_acc),
            "claude_auc": float(claude_auc), "claude_acc": float(claude_acc),
            "n_train": len(train), "n_val": len(val),
            "n_manual": len(manual), "n_claude": len(claude),
        })

        # Dump per-sample alphas for the held-out splits so we can visualize later
        layer_dir = args.out_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": probe.state_dict(),
            "d_model": d,
            "layer_idx": layer_idx,
        }, str(layer_dir / "probe.pt"))
        for name, samples, probs, alphas in (
            ("manual", manual, probs_manual, alpha_manual),
            ("claude", claude, probs_claude, alpha_claude),
            ("val",    val,    probs_val,    alpha_val),
        ):
            recs = []
            for i, s in enumerate(samples):
                T = s["n_tokens"]
                recs.append({
                    "sample_id": s["sample_id"],
                    "tokens": s["tokens"][:T],
                    "alpha": alphas[i, :T].tolist(),
                    "label": int(s["label"]),
                    "probe_prob": float(probs[i]),
                    "model_pred": s["pred"],
                    "model_prob_uneq": s["p_unequivalent"],
                })
            (layer_dir / f"{name}_alpha.json").write_text(json.dumps(recs, indent=2))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[probe] summary -> {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
