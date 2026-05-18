"""Stage 12: leave-one-out generator ablation on multi_gen.

For each of the 5 generators, train a roberta-base on the union of the OTHER
4 generators' 160-pair train splits (= 640 pairs total). Score on every
generator's test split + the Stage 10 100-pair human eval.

Tells us which generator carries the multi_gen win:
  - If "drop qwen25_72b" causes a 15-point drop, qwen is doing the work
  - If all 5 LOO models perform similarly, the diversity itself is what matters
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, _batched_forward, _metrics_from_arrays, find_checkpoint,
    save_json,
)
from overnight.stage9_matrix import (  # noqa: E402
    load_generator_csv, make_train_test_split, tokenize,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage12"
RUNS_DIR = REPO / "overnight" / "runs" / "stage12"
STAGE10_CSV = REPO / "overnight" / "results" / "stage10" / "claude_human_eval.csv"


def train_loo(name: str, train_union: pd.DataFrame, epochs: int = 3, seed: int = 42) -> Path:
    set_seed(seed)
    run_dir = RUNS_DIR / name
    if (run_dir / "checkpoints").exists():
        try:
            return find_checkpoint(run_dir)
        except FileNotFoundError:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize(Dataset.from_pandas(train_union[["text_a", "text_b", "label"]]), tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"}, label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        save_strategy="epoch", save_total_limit=1,
        eval_strategy="no",
        bf16=torch.cuda.is_available(), fp16=False,
        logging_steps=20, report_to="none",
        dataloader_num_workers=0, seed=seed,
    )
    Trainer(model=model, args=targs,
            train_dataset=train_ds,
            processing_class=tok, data_collator=collator).train()
    return find_checkpoint(run_dir)


def score_ckpt_on_df(ckpt: Path, df: pd.DataFrame, text_a_col: str, text_b_col: str,
                     label_col: str) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    model.train(False)
    pairs = list(zip(
        df[text_a_col].astype(str).tolist(),
        df[text_b_col].astype(str).tolist(),
        df[label_col].astype(int).tolist(),
    ))
    labels, preds, probs = _batched_forward(pairs, tok, model, device)
    return _metrics_from_arrays(labels, preds, probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    cross_dir = REPO / "overnight" / "cross_gen"
    all_slugs = sorted(p.stem for p in cross_dir.glob("*.csv") if not p.stem.startswith("_"))
    train_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for s in all_slugs:
        df = load_generator_csv(s)
        if df is None:
            continue
        tr, te = make_train_test_split(df)
        train_dfs[s] = tr
        test_dfs[s] = te
    slugs = sorted(train_dfs.keys())
    print(f"[stage12] usable generators: {slugs}", flush=True)

    # Stage 10 human eval
    stage10 = pd.read_csv(STAGE10_CSV)
    stage10 = stage10[stage10["label_verified"].isin(LABEL2ID)].reset_index(drop=True)
    stage10["label_int"] = stage10["label_verified"].map(LABEL2ID).astype(int)
    print(f"[stage12] stage10 human eval: n={len(stage10)}", flush=True)

    matrix: dict[str, dict[str, Any]] = {}
    for held_out in slugs:
        name = f"loo_drop_{held_out}"
        union = pd.concat([train_dfs[s] for s in slugs if s != held_out], ignore_index=True)
        print(f"\n[loo] {name}: train on {len(union)} pairs from {[s for s in slugs if s != held_out]}", flush=True)
        t0 = time.time()
        try:
            ckpt = train_loo(name, union, epochs=args.epochs)
        except Exception as e:
            print(f"[loo] {name} train FAILED: {e}", flush=True)
            continue
        print(f"[loo] {name}: trained in {time.time()-t0:.0f}s -> {ckpt}", flush=True)

        # Score on each generator's test split
        per_gen = {}
        for te_slug, te_df in test_dfs.items():
            try:
                m = score_ckpt_on_df(ckpt, te_df, "text_a", "text_b", "label")
                per_gen[te_slug] = {"n": m["n"], "acc": m["accuracy"], "f1": m["f1_macro"],
                                    "acc_eq": m.get("acc_equivalent"),
                                    "acc_uneq": m.get("acc_unequivalent")}
                marker = " (held-out)" if te_slug == held_out else ""
                print(f"  test={te_slug:>18}{marker}  acc={m['accuracy']:.3f}  f1={m['f1_macro']:.3f}", flush=True)
            except Exception as e:
                per_gen[te_slug] = {"error": str(e)}

        # Score on Stage 10 humans
        try:
            mh = score_ckpt_on_df(ckpt, stage10, "explanation", "comment", "label_int")
            per_gen["human_stage10"] = {"n": mh["n"], "acc": mh["accuracy"], "f1": mh["f1_macro"],
                                        "acc_eq": mh.get("acc_equivalent"),
                                        "acc_uneq": mh.get("acc_unequivalent"),
                                        "auroc": mh.get("auroc")}
            print(f"  test={'human_stage10':>18}        acc={mh['accuracy']:.3f}  "
                  f"f1={mh['f1_macro']:.3f}  auroc={mh.get('auroc',0):.3f}", flush=True)
        except Exception as e:
            per_gen["human_stage10"] = {"error": str(e)}

        matrix[name] = {"held_out": held_out, "n_train_pairs": int(len(union)), "scores": per_gen}
        save_json(RESULTS_DIR / "loo.json", matrix)

    # OVERNIGHTLOG
    log_path = REPO / "OVERNIGHTLOG.md"
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## {ts}  STAGE 12: leave-one-out generator ablation\n",
        "For each of 5 generators, trained roberta-base on the union of the OTHER 4 "
        "(=640 pairs). Stage 10 humans = n=100 Claude-Haiku eval.",
        "",
        "| held-out generator | n_train | test=held-out | human (n=100) acc | human AUROC |",
        "|---|---|---|---|---|",
    ]
    for name, info in matrix.items():
        ho = info["held_out"]
        sc = info["scores"]
        ho_score = sc.get(ho, {}).get("acc")
        hu_score = sc.get("human_stage10", {}).get("acc")
        hu_auc = sc.get("human_stage10", {}).get("auroc")
        lines.append(f"| {ho} | {info['n_train_pairs']} | "
                     f"{ho_score*100:.1f}% | {hu_score*100:.1f}% | "
                     f"{hu_auc:.3f} |" if hu_auc else
                     f"| {ho} | {info['n_train_pairs']} | "
                     f"{ho_score*100:.1f}% | {hu_score*100:.1f}% | - |")
    lines.append("")
    lines.append("Reference: full multi_gen (5 generators, 800 pairs) hit 81% on Stage 10 humans.")
    lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[stage12] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
