"""Per-token residuals on a small focus set.

We already have CLS-only extracts for the full 52k corpus. Those are fine for
"does feature X linearly decode" but useless for "which tokens does the model
attend to". To get token-level attribution we need full (B, T, d) hidden
states. Going wide on 52k blows disk, so we restrict to ~few-hundred samples:

    - 20 manual labels    (the hardest test set + what the model actually fails on)
    - 22 claude rows      (the OOD voice that breaks the model)
    - 400 random synthetic rows for training the probe

Output: one .pt per sample under mech_interp/token_extracts/ + meta.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO / "sent_on_telegram" / "roberta_fewshot" / "checkpoints" / "checkpoint-1580"
DEFAULT_OUT = REPO / "mech_interp" / "token_extracts"

# layer indices into output_hidden_states (0 = embeddings, 1..12 = blocks).
# pulling early/mid/late so we can sweep without saving all 13.
DEFAULT_LAYERS = [1, 6, 9, 12]


def load_manual(path: Path) -> list[dict]:
    df = pd.read_csv(path, skiprows=1, header=0)
    df.columns = ["post", "explanation", "human_comment", "label"] + list(df.columns[4:])
    df = df[["post", "explanation", "human_comment", "label"]].dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin({"equivalent", "unequivalent"})].reset_index(drop=True)
    out = []
    for i, r in df.iterrows():
        out.append({
            "sample_id": f"manual_{i:02d}",
            "source": "manual",
            "text_a": r["explanation"],
            "text_b": r["human_comment"],
            "label": 0 if r["label"] == "equivalent" else 1,
        })
    return out


def load_claude(path: Path) -> list[dict]:
    # same shape as manual: row 0 is "Table 1", row 1 is header, then data.
    # columns 0..3: Post, Explanation, claude_comment (unnamed), label (unnamed)
    df = pd.read_csv(path, skiprows=1, header=0)
    df = df.iloc[:, :4]
    df.columns = ["post", "explanation", "claude_comment", "label"]
    df = df.dropna(subset=["explanation", "claude_comment", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin({"equivalent", "unequivalent"})].reset_index(drop=True)
    return [{
        "sample_id": f"claude_{i:02d}",
        "source": "claude",
        "text_a": r["explanation"],
        "text_b": r["claude_comment"],
        "label": 0 if r["label"] == "equivalent" else 1,
    } for i, r in df.iterrows()]


def load_synthetic(path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(path, usecols=["uid", "explanation", "synthetic", "meaning"]).dropna()
    df = df[df["meaning"].isin({"equivalent", "unequivalent"})]
    # balance the sample
    n_per = n // 2
    rng = random.Random(seed)
    eq = df[df["meaning"] == "equivalent"].sample(n=min(n_per, (df["meaning"] == "equivalent").sum()), random_state=seed)
    un = df[df["meaning"] == "unequivalent"].sample(n=min(n_per, (df["meaning"] == "unequivalent").sum()), random_state=seed + 1)
    mix = pd.concat([eq, un]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [{
        "sample_id": f"synth_{r['uid']}",
        "source": "synthetic",
        "text_a": r["explanation"],
        "text_b": r["synthetic"],
        "label": 0 if r["meaning"] == "equivalent" else 1,
    } for _, r in mix.iterrows()]


def pick_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manual_csv", type=Path,
                    default=REPO / "synthetic_data_generation_improved" / "manual_labelling.csv")
    ap.add_argument("--claude_csv", type=Path,
                    default=REPO / "roberta_training" / "claude_test_labels.csv")
    ap.add_argument("--synthetic_csv", type=Path, default=REPO / "reformulated_full.csv")
    ap.add_argument("--n_synthetic", type=int, default=400)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="for dev, cap total samples")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[token-extract] device={device}  ckpt={args.checkpoint}")

    samples = (load_manual(args.manual_csv)
               + load_claude(args.claude_csv)
               + load_synthetic(args.synthetic_csv, args.n_synthetic, args.seed))
    if args.limit:
        samples = samples[: args.limit]
    print(f"[token-extract] {len(samples)} samples "
          f"(manual={sum(1 for s in samples if s['source']=='manual')}, "
          f"claude={sum(1 for s in samples if s['source']=='claude')}, "
          f"synthetic={sum(1 for s in samples if s['source']=='synthetic')})")

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, output_hidden_states=True
    ).to(device).eval()

    t0 = time.time()
    written = 0
    for i, s in enumerate(samples):
        out_path = args.out_dir / f"{s['sample_id']}.pt"
        if out_path.exists():
            continue
        enc = tok(s["text_a"], s["text_b"], padding=False, truncation=True,
                  max_length=args.max_length, return_tensors="pt")
        # decode token strings BEFORE moving to device (MPS int64 tolist() is buggy
        # on some torch builds and produces overflowing ints).
        ids_list = enc["input_ids"][0].tolist()
        token_strs = tok.convert_ids_to_tokens(ids_list)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        T = ids.shape[1]
        stacked = torch.stack([out.hidden_states[k][0] for k in args.layers], dim=0)
        probs = torch.softmax(out.logits[0], dim=-1).cpu()
        record = {
            "sample_id": s["sample_id"],
            "source": s["source"],
            "text_a": s["text_a"],
            "text_b": s["text_b"],
            "label": int(s["label"]),
            "input_ids": torch.tensor(ids_list, dtype=torch.int32),
            "attention_mask": mask[0].to("cpu", torch.bool),
            "tokens": token_strs,
            "residuals": stacked.to("cpu", torch.float16).contiguous(),
            "layer_idxs": list(args.layers),
            "pred": int(probs.argmax().item()),
            "p_unequivalent": float(probs[1].item()),
            "n_tokens": int(T),
        }
        torch.save(record, str(out_path))
        written += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  [{i+1}/{len(samples)}]  rate={rate:.1f}/s")

    meta = {
        "checkpoint": str(args.checkpoint),
        "n_samples": len(samples),
        "n_written_now": written,
        "layers": list(args.layers),
        "max_length": args.max_length,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[token-extract] done. wrote {written} new files. elapsed={meta['elapsed_s']}s")


if __name__ == "__main__":
    main()
