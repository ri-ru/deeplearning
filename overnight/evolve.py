"""Self-evolving variant proposer.

Reads `overnight/results/stage1/*.json`, identifies systematic failure
patterns in the best-so-far model, and synthesizes new mitigation variants
from a parametric grid. Each iteration:

  1. Score every existing variant by `manual_acc`. Pick the best.
  2. Inspect its confusion matrix on the manual + Claude sets:
       - if it under-predicts equivalent (acc_eq low, acc_uneq high)
         → the shortcut is "be skeptical, predict unequivalent". Propose
           variants that make 'unequivalent' harder to learn cheaply.
       - if balanced but still <70% manual_acc, propose generator-diversity
         augments.
  3. Translate diagnosis to concrete grid points:
        - drop_cosine_threshold ∈ {0.75, 0.80, 0.85, 0.90}
        - drop_neg_shortcut ∈ {True, False}
        - downsample_uneq_frac ∈ {0.3, 0.5, 0.7, 1.0}
        - mix_qwen7b ∈ {True, False}
        - epochs ∈ {1, 2, 3}
  4. Skip any grid point that's already been run (cached by hash of params).
  5. Train + score it; write to overnight/results/stage1/auto_<hash>.json.
  6. Repeat until budget (--max_new) is hit or no new productive points.

This is bounded agency: we don't write new code, we sample a known schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import (  # noqa: E402
    LABEL2ID, REPO, TEST_DOMAINS, TRAIN_DOMAINS,
    find_checkpoint, has_negation_flip, load_full_csv, save_json, score_all,
)

RESULTS_DIR = REPO / "overnight" / "results" / "stage1"
RUNS_DIR = REPO / "overnight" / "runs"
EVOLVE_LOG = REPO / "overnight" / "evolve.log"


@dataclass(frozen=True)
class Variant:
    """All knobs the proposer can flip. Tuples sort+hash cleanly."""
    drop_cosine_threshold: float | None
    drop_neg_shortcut: bool
    downsample_uneq_frac: float          # 1.0 = no downsample
    upsample_eq_neg_factor: int          # 1 = no upsample
    mix_qwen7b: bool
    epochs: int

    def stable_id(self) -> str:
        d = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(d.encode()).hexdigest()[:10]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    print(line, end="", flush=True)
    EVOLVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLVE_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def _append_overnight_log(name: str, v: Variant, metrics: dict) -> None:
    log_path = REPO / "OVERNIGHTLOG.md"
    m = metrics["manual"]; o = metrics["ood_overall"]; c = metrics["claude"]
    cm = c.get("pred_balance", {})
    pred_uneq = (int(cm.get("1", 0)) / max(c.get("n", 1), 1)) * 100
    ts = time.strftime("%Y-%m-%d %H:%M")
    entry = (
        f"\n## {ts}  {name}  (evolve)\n"
        f"- recipe: {asdict(v)}\n"
        f"- OOD F1: {o['f1_macro']:.3f}  manual_acc: {m['accuracy']*100:.1f}% "
        f"(eq={m.get('acc_equivalent', 0)*100:.0f}%/uneq={m.get('acc_unequivalent', 0)*100:.0f}%)  "
        f"claude_acc: {c['accuracy']*100:.1f}% ({pred_uneq:.0f}% predicted unequivalent)\n"
        f"- interpretation: _auto-logged; review later_\n"
        f"- file: `overnight/results/stage1/{name}.json`\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


# --- data prep (kept identical to stage1_mitigations.py) -------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neg_flip"] = [
        has_negation_flip(a, b) for a, b in zip(df["explanation"], df["synthetic"])
    ]
    vec = TfidfVectorizer()
    vec.fit(pd.concat([df["explanation"].astype(str), df["synthetic"].astype(str)]))
    a = vec.transform(df["explanation"].astype(str))
    b = vec.transform(df["synthetic"].astype(str))
    sims = np.array([cosine_similarity(a[i], b[i])[0, 0] for i in range(a.shape[0])])
    df["cosine_sim"] = sims
    return df


def apply_variant(df: pd.DataFrame, v: Variant) -> pd.DataFrame:
    out = df.copy()
    if v.drop_cosine_threshold is not None:
        out = out[~((out["label"] == 1) & (out["cosine_sim"] > v.drop_cosine_threshold))]
    if v.drop_neg_shortcut:
        out = out[~((out["label"] == 1) & (out["neg_flip"] == 1))]
    if v.downsample_uneq_frac < 1.0:
        keep_uneq = out[out["label"] == 1].sample(frac=v.downsample_uneq_frac, random_state=42)
        out = pd.concat([out[out["label"] == 0], keep_uneq], ignore_index=True)
    if v.upsample_eq_neg_factor > 1:
        eq_neg = out[(out["label"] == 0) & (out["neg_flip"] == 1)]
        extras = pd.concat([eq_neg] * (v.upsample_eq_neg_factor - 1), ignore_index=True)
        out = pd.concat([out, extras], ignore_index=True)
    if v.mix_qwen7b:
        aug_rows = []
        for p in (REPO / "synthetic_data_generation_improved").glob("reformulated*.csv"):
            sub = pd.read_csv(p)
            if "explanation" not in sub.columns or "synthetic" not in sub.columns:
                continue
            sub = sub.dropna(subset=["explanation", "synthetic", "meaning", "domain"]).copy()
            sub["meaning"] = sub["meaning"].astype(str).str.lower().str.strip()
            sub = sub[sub["meaning"].isin(LABEL2ID)].copy()
            sub["label"] = sub["meaning"].map(LABEL2ID).astype(int)
            sub["neg_flip"] = [
                has_negation_flip(a, b) for a, b in zip(sub["explanation"], sub["synthetic"])
            ]
            sub["cosine_sim"] = 0.0
            aug_rows.append(sub)
        if aug_rows:
            aug = pd.concat(aug_rows, ignore_index=True)
            cols = [c for c in out.columns if c in aug.columns]
            out = pd.concat([out[cols], aug[cols]], ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def make_split(df, seed=42, val_frac=0.1):
    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    test_df = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)
    val_chunks = []
    for _, grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=val_frac, random_state=seed))
    val_df = pd.concat(val_chunks).reset_index(drop=True)
    val_idx = set(val_df.index)
    train_df = train_pool[~train_pool.index.isin(val_idx)].reset_index(drop=True)
    return train_df, val_df, test_df


def tokenize_ds(ds, tok, max_length=256):
    def _t(batch):
        return tok(batch["text_a"], batch["text_b"],
                   truncation=True, max_length=max_length, padding=False)
    cols = [c for c in ds.column_names if c not in ("label", "domain")]
    return ds.map(_t, batched=True, remove_columns=cols)


def train_variant(v: Variant, base_df_with_features: pd.DataFrame) -> Path:
    name = f"auto_{v.stable_id()}"
    run_dir = RUNS_DIR / name
    if (run_dir / "checkpoints").exists():
        try:
            return find_checkpoint(run_dir)
        except FileNotFoundError:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)

    df = apply_variant(base_df_with_features, v)
    df = df.rename(columns={"explanation": "text_a", "synthetic": "text_b"})
    train_df, val_df, _ = make_split(df)
    log(f"{name}: train={len(train_df)} val={len(val_df)}  recipe={asdict(v)}")

    tok = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = tokenize_ds(Dataset.from_pandas(train_df[["text_a", "text_b", "label", "domain"]]), tok)
    val_ds = tokenize_ds(Dataset.from_pandas(val_df[["text_a", "text_b", "label", "domain"]]), tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2,
        id2label={0: "equivalent", 1: "unequivalent"}, label2id=LABEL2ID,
    )
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=v.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        bf16=torch.cuda.is_available(), fp16=False,
        logging_steps=200, report_to="none",
        dataloader_num_workers=0, seed=42,
    )
    set_seed(42)
    Trainer(model=model, args=targs,
            train_dataset=train_ds, eval_dataset=val_ds,
            processing_class=tok, data_collator=collator).train()
    return find_checkpoint(run_dir)


# --- proposer ---------------------------------------------------------------

def load_existing() -> list[dict]:
    """All scored variants, manual and auto."""
    rows: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            rows.append({"name": p.stem, "result": d,
                         "manual_acc": d["manual"]["accuracy"],
                         "manual_acc_eq": d["manual"].get("acc_equivalent", 0.0),
                         "manual_acc_uneq": d["manual"].get("acc_unequivalent", 0.0),
                         "ood_f1": d["ood_overall"]["f1_macro"],
                         "claude_acc": d["claude"]["accuracy"]})
        except Exception:
            continue
    return rows


def diagnose(best: dict) -> dict:
    """Return a one-shot symptom dictionary used by `propose_variants`."""
    eq = best["manual_acc_eq"]
    un = best["manual_acc_uneq"]
    asymmetry = un - eq          # positive = biased toward 'unequivalent'
    return {"asymmetry": asymmetry, "manual_acc": best["manual_acc"]}


def propose_variants(symptoms: dict, already_tried: set[str], n: int) -> list[Variant]:
    """Sample up to n new grid points biased by the symptom dict."""
    candidates: list[Variant] = []

    if symptoms["asymmetry"] > 0.25:
        # Model is biased toward 'unequivalent'. Squeeze unequivalent supply
        # and inject equivalent+negation pairs.
        for cos in (0.85, 0.80, 0.75):
            for frac in (0.7, 0.5):
                for ups in (2, 3):
                    candidates.append(Variant(
                        drop_cosine_threshold=cos, drop_neg_shortcut=True,
                        downsample_uneq_frac=frac, upsample_eq_neg_factor=ups,
                        mix_qwen7b=True, epochs=2,
                    ))
    else:
        # Diversity push only
        for cos in (None, 0.85):
            for ups in (1, 2):
                for ep in (2, 3):
                    candidates.append(Variant(
                        drop_cosine_threshold=cos, drop_neg_shortcut=False,
                        downsample_uneq_frac=1.0, upsample_eq_neg_factor=ups,
                        mix_qwen7b=True, epochs=ep,
                    ))

    # Always include 1 control with very aggressive shortcut removal
    candidates.append(Variant(
        drop_cosine_threshold=0.7, drop_neg_shortcut=True,
        downsample_uneq_frac=0.5, upsample_eq_neg_factor=3,
        mix_qwen7b=True, epochs=3,
    ))

    fresh = [c for c in candidates if c.stable_id() not in already_tried]
    return fresh[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=3,
                        help="At most this many propose->train->score iterations.")
    parser.add_argument("--per_iter", type=int, default=2,
                        help="At most this many new variants per iteration.")
    parser.add_argument("--cleanup_checkpoints", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== evolve loop START  max_iters={args.max_iters}  per_iter={args.per_iter} ===")

    log("loading full csv + computing features ...")
    base_df = add_features(load_full_csv())

    for it in range(args.max_iters):
        existing = load_existing()
        if not existing:
            log("no stage1 results yet -> bail. Run stage1_mitigations.py first.")
            return 0
        best = max(existing, key=lambda r: r["manual_acc"])
        symptoms = diagnose(best)
        log(f"iter {it}: best={best['name']}  manual_acc={best['manual_acc']:.3f}  "
            f"eq={best['manual_acc_eq']:.3f}/uneq={best['manual_acc_uneq']:.3f}  "
            f"asymmetry={symptoms['asymmetry']:+.2f}")

        already_tried = set()
        for r in existing:
            if r["name"].startswith("auto_"):
                already_tried.add(r["name"].split("_", 1)[1])

        proposals = propose_variants(symptoms, already_tried, args.per_iter)
        if not proposals:
            log(f"iter {it}: no new proposals; stopping.")
            break
        log(f"iter {it}: proposing {len(proposals)} variants: {[p.stable_id() for p in proposals]}")

        for v in proposals:
            name = f"auto_{v.stable_id()}"
            result_path = RESULTS_DIR / f"{name}.json"
            if result_path.exists():
                log(f"  {name}: result exists, skipping")
                continue
            t0 = time.time()
            try:
                ckpt = train_variant(v, base_df)
                metrics = score_all(ckpt)
                metrics["_meta"] = {
                    "variant": name, "recipe": asdict(v), "epochs": v.epochs,
                    "checkpoint": str(ckpt),
                    "elapsed_sec": round(time.time() - t0, 1),
                    "proposed_by": "evolve.py", "iter": it,
                    "trigger_symptoms": symptoms,
                }
                save_json(result_path, metrics)
                log(f"  {name}: manual_acc={metrics['manual']['accuracy']:.3f}  "
                    f"ood_f1={metrics['ood_overall']['f1_macro']:.3f}  "
                    f"({metrics['_meta']['elapsed_sec']:.0f}s)")
                # Append a one-liner to OVERNIGHTLOG.md so we have a flat history
                _append_overnight_log(name, v, metrics)
                if args.cleanup_checkpoints:
                    for sub in ckpt.parent.iterdir():
                        if sub.is_dir():
                            shutil.rmtree(sub, ignore_errors=True)
            except Exception as e:
                (RESULTS_DIR / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
                log(f"  {name}: FAILED {type(e).__name__}: {e}")

    log("=== evolve loop done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
