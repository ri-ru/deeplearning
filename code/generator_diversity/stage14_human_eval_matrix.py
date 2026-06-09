"""Stage 14: cross-train x cross-human-eval matrix 
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, _batched_forward, _metrics_from_arrays, load_human_csv, save_json,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage14"


def resolve_ckpt(name: str) -> Path | None:
    """Map a friendly name to a checkpoint dir."""
    candidates = {
        # single-generator models
        "anthropic_haiku":      REPO / "overnight" / "runs" / "stage9" / "anthropic_haiku" / "checkpoints",
        "deepseek_v3":          REPO / "overnight" / "runs" / "stage9" / "deepseek_v3" / "checkpoints",
        "meta_llama3_8b":       REPO / "overnight" / "runs" / "stage9" / "meta_llama3_8b" / "checkpoints",
        "openai_gpt4omini":     REPO / "overnight" / "runs" / "stage9" / "openai_gpt4omini" / "checkpoints",
        "qwen25_72b":           REPO / "overnight" / "runs" / "stage9" / "qwen25_72b" / "checkpoints",
        # union + leave-one-outs
        "multi_gen":            REPO / "overnight" / "runs" / "stage9" / "multi_gen" / "checkpoints",
        "loo_drop_anthropic":   REPO / "overnight" / "runs" / "stage12" / "loo_drop_anthropic_haiku" / "checkpoints",
        "loo_drop_deepseek":    REPO / "overnight" / "runs" / "stage12" / "loo_drop_deepseek_v3" / "checkpoints",
        "loo_drop_llama":       REPO / "overnight" / "runs" / "stage12" / "loo_drop_meta_llama3_8b" / "checkpoints",
        "loo_drop_openai":      REPO / "overnight" / "runs" / "stage12" / "loo_drop_openai_gpt4omini" / "checkpoints",
        "loo_drop_qwen":        REPO / "overnight" / "runs" / "stage12" / "loo_drop_qwen25_72b" / "checkpoints",
        # poster baseline for reference
        "poster_baseline":      REPO / "judge_finetune" / "runs" / "roberta_base" / "checkpoints" / "checkpoint-1576",
    }
    ck = candidates.get(name)
    if ck is None or not ck.exists():
        return None
    if (ck / "config.json").exists():
        return ck
    subs = sorted(ck.glob("checkpoint-*"))
    return subs[-1] if subs else None


def load_eval_sets() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}


    # manual + claude_test (40 humans, never seen by any model)
    rows = []
    for src in [REPO / "manual labelling.csv",
                REPO / "roberta_training" / "claude_test_labels.csv"]:
        for r in load_human_csv(src):
            rows.append({"text_a": r["explanation"], "text_b": r["comment"],
                         "label": LABEL2ID[r["label"]]})
    out["manual_plus_claude_n40"] = pd.DataFrame(rows)

    #40 mini reader voice
    p = REPO / "overnight" / "results" / "stage7" / "harder_eval.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["label_verified"].isin(LABEL2ID)].copy()
        df["label"] = df["label_verified"].map(LABEL2ID).astype(int)
        df = df.rename(columns={"explanation": "text_a", "comment": "text_b"})
        out["stage7_gpt_n50"] = df[["text_a", "text_b", "label"]].reset_index(drop=True)

    # claude haiku
    p = REPO / "overnight" / "results" / "stage10" / "claude_human_eval.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["label_verified"].isin(LABEL2ID)].copy()
        df["label"] = df["label_verified"].map(LABEL2ID).astype(int)
        df = df.rename(columns={"explanation": "text_a", "comment": "text_b"})
        out["stage10_haiku_n100"] = df[["text_a", "text_b", "label"]].reset_index(drop=True)

    return out


def score_one(ckpt: Path, df: pd.DataFrame) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.train(False)
    pairs = list(zip(
        df["text_a"].astype(str).tolist(),
        df["text_b"].astype(str).tolist(),
        df["label"].astype(int).tolist(),
    ))
    labels, preds, probs = _batched_forward(pairs, tok, model, device)
    return _metrics_from_arrays(labels, preds, probs)


MODELS = [
    "poster_baseline",
    "anthropic_haiku",
    "openai_gpt4omini",
    "qwen25_72b",
    "meta_llama3_8b",
    "deepseek_v3",
    "loo_drop_anthropic",
    "loo_drop_deepseek",
    "loo_drop_llama",
    "loo_drop_openai",
    "loo_drop_qwen",
    "multi_gen",
]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_sets = load_eval_sets()
    eval_names = list(eval_sets.keys())
    print(f"[stage14] eval sets: {[(n, len(df)) for n, df in eval_sets.items()]}", flush=True)

    matrix: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        ck = resolve_ckpt(m)
        if ck is None:
            print(f"[stage14] {m}: no checkpoint; skipping", flush=True)
            continue
        matrix[m] = {}
        for eval_name, df in eval_sets.items():
            t0 = time.time()
            try:
                r = score_one(ck, df)
                matrix[m][eval_name] = {
                    "n": r["n"], "acc": r["accuracy"], "f1": r["f1_macro"],
                    "auroc": r.get("auroc"),
                    "acc_eq": r.get("acc_equivalent"), "acc_uneq": r.get("acc_unequivalent"),
                }
                a = r["accuracy"]; eq = r.get("acc_equivalent",0) or 0; uneq = r.get("acc_unequivalent",0) or 0
                print(f"  {m:>22}  on {eval_name:>22}  acc={a:.3f}  eq={eq*100:.0f}%/uneq={uneq*100:.0f}%  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                matrix[m][eval_name] = {"error": str(e)}
                print(f"  {m:>22}  on {eval_name:>22}  FAILED: {e}", flush=True)
        save_json(RESULTS_DIR / "matrix.json", matrix)

    # OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {ts}  STAGE 14: cross-train x cross-human-eval matrix\n",
        "Every trained checkpoint scored on every human-style eval set.\n",
        "**Accuracy (eq% / uneq%):**\n",
        "| model | manual+claude (n=40) | stage7 GPT (n=50) | stage10 Haiku (n=100) |",
        "|---|---|---|---|",
    ]
    for m, row in matrix.items():
        cells = [m]
        for e in eval_names:
            v = row.get(e, {})
            if "acc" in v:
                eq = (v.get("acc_eq") or 0) * 100
                un = (v.get("acc_uneq") or 0) * 100
                cells.append(f"{v['acc']*100:.0f}% ({eq:.0f}/{un:.0f})")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**AUROC:**")
    lines.append("")
    lines.append("| model | manual+claude | stage7 GPT | stage10 Haiku |")
    lines.append("|---|---|---|---|")
    for m, row in matrix.items():
        cells = [m]
        for e in eval_names:
            v = row.get(e, {})
            cells.append(f"{v['auroc']:.3f}" if v.get("auroc") is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[stage14] done. matrix -> {RESULTS_DIR / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
