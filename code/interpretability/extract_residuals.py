"""Forward-pass every pair through the fine-tuned judge, save [CLS] residual stream per layer.

RoBERTa-base has 13 hidden states (1 embeddings + 12 blocks), hidden_dim=768.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "mech_interp" / "extracts" / "manifest.csv"
DEFAULT_OUT_DIR = REPO / "mech_interp" / "extracts"


def _find_latest_checkpoint(run_dir: Path) -> Path:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoints dir at {ckpt_root}")
    candidates = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* dirs under {ckpt_root}")
    return candidates[-1]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to a HF checkpoint dir. If omitted, latest checkpoint under runs/smoke/checkpoints/.",
    )
    parser.add_argument("--run_dir", type=Path, default=REPO / "judge_finetune" / "runs" / "smoke")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (for dev).")
    args = parser.parse_args()

    ckpt = args.checkpoint or _find_latest_checkpoint(args.run_dir)
    print(f"[extract] checkpoint = {ckpt}")

    device = _pick_device()
    print(f"[extract] device     = {device}")

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, output_hidden_states=True)
    model.to(device).eval()

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit).reset_index(drop=True)
    n = len(manifest)
    print(f"[extract] manifest rows = {n}")

    sample_ids: list[str] = manifest["sample_id"].astype(str).tolist()
    text_a = manifest["text_a"].astype(str).tolist()
    text_b = manifest["text_b"].astype(str).tolist()

    #ffp 
    with torch.no_grad():
        enc = tok(
            text_a[: min(args.batch_size, n)],
            text_b[: min(args.batch_size, n)],
            padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
    n_layers = len(out.hidden_states)
    hidden_dim = out.hidden_states[0].shape[-1]
    print(f"[extract] n_layers={n_layers}  hidden_dim={hidden_dim}")

    # Preallocate per-layer cpu  tensors
    layer_buffers = [torch.empty(n, hidden_dim, dtype=torch.float32) for _ in range(n_layers)]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    cursor = 0
    while cursor < n:
        end = min(cursor + args.batch_size, n)
        batch_a = text_a[cursor:end]
        batch_b = text_b[cursor:end]
        enc = tok(
            batch_a, batch_b,
            padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        #(B, T, H) tensors  
        for li, h in enumerate(out.hidden_states):
            cls = h[:, 0, :].detach().to(torch.float32).cpu()
            layer_buffers[li][cursor:end] = cls
        cursor = end
        if cursor % (args.batch_size * 20) == 0 or cursor == n:
            elapsed = time.time() - start
            rate = cursor / max(elapsed, 1e-6)
            eta = (n - cursor) / max(rate, 1e-6)
            print(f"[extract] {cursor:>6}/{n}  rate={rate:6.1f}/s  eta={eta:6.0f}s")

    for li, buf in enumerate(layer_buffers):
        out_path = args.out_dir / f"layer_{li:02d}.pt"
        torch.save({"layer": li, "sample_ids": sample_ids, "activations": buf}, out_path)
    print(f"[extract] wrote {n_layers} layer files to {args.out_dir}")

    meta = {
        "checkpoint": str(ckpt),
        "n_samples": n,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (args.out_dir / "extract_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[extract] meta: {meta}")


if __name__ == "__main__":
    main()
