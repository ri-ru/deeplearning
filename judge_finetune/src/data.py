"""Data loading, domain-OOD split, and tokenization for the rationale-equivalence judge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

# Domains held out for OOD evaluation. Chosen because they are stylistically
# distinct from the training domains (gab is forum-style, toxigen is LLM-generated,
# anatomy is targeted abuse patterns).
TRAIN_DOMAINS = ("dynahate", "cad", "measure", "socialbias", "toraman")
TEST_DOMAINS = ("toxigen", "gab", "anatomy")

LABEL2ID = {"equivalent": 0, "unequivalent": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class DataConfig:
    csv_path: str
    max_length: int = 128
    val_frac: float = 0.1
    seed: int = 42
    max_samples: int | None = None
    augment_csv: str | None = None   # extra CSV of hard examples to inject into train


def load_and_split(cfg: DataConfig) -> tuple[Dataset, Dataset, Dataset]:
    """Returns (train, val, test) as HF Datasets with columns:
       text_a, text_b, label, domain, post_id.
    """
    df = pd.read_csv(cfg.csv_path)

    # The CSV has both a 'meaning' column that holds the reformulated rationale
    # text AND an equivalence judgement. The schema in this dataset is:
    #   explanation -> original rationale (text_a)
    #   meaning     -> reformulated rationale (text_b) when row is judged 'equivalent',
    #                  or a *different-meaning* reformulation when 'unequivalent'.
    # The label column we care about for *our* task is the LAST column,
    # which conventionally also lives under 'meaning' but in some exports
    # under a separate column. Detect it.
    label_col = None
    for cand in ["equivalence", "judgement", "is_equivalent"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        # Fall back: in the provided CSV, the equivalence label is the value
        # in the column literally named 'meaning' when it equals one of the
        # two label strings. Otherwise the actual label column is the last one.
        last_col = df.columns[-1]
        sample = df[last_col].dropna().astype(str).str.lower().unique()[:5]
        if set(sample).issubset({"equivalent", "unequivalent"}):
            label_col = last_col

    if label_col is None:
        raise ValueError(
            "Could not identify the equivalence-label column. "
            "Expected one of: equivalence / judgement / is_equivalent, "
            "or a column whose values are 'equivalent'/'unequivalent'."
        )

    # If the label lives in the same column as the reformulated text,
    # they are actually two separate columns in the source — adjust here.
    # In Veronika's CSV: columns are
    #   id, explanation, domain, label (hate/benign/...), post, target,
    #   uid, synthetic (=reformulated text), meaning (=equivalent/unequivalent)
    text_a_col = "explanation"
    text_b_col = "synthetic" if "synthetic" in df.columns else "reformulated"
    if text_b_col not in df.columns:
        raise ValueError(
            f"Could not find reformulated-text column (looked for 'synthetic'/'reformulated'). "
            f"Have: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_a_col, text_b_col, label_col, "domain"])
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(LABEL2ID)]

    df = df.rename(
        columns={text_a_col: "text_a", text_b_col: "text_b", label_col: "label_str"}
    )
    df["label"] = df["label_str"].map(LABEL2ID).astype(int)
    df["post_id"] = df.get("uid", df.get("id", range(len(df)))).astype(str)

    # Domain split.
    train_pool = df[df["domain"].isin(TRAIN_DOMAINS)].reset_index(drop=True)
    test_df = df[df["domain"].isin(TEST_DOMAINS)].reset_index(drop=True)

    if cfg.max_samples is not None:
        train_pool = train_pool.sample(
            n=min(cfg.max_samples, len(train_pool)),
            random_state=cfg.seed,
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(cfg.max_samples // 4, len(test_df)),
            random_state=cfg.seed,
        ).reset_index(drop=True)

    # Stratified val carved from train_pool by (domain, label) to keep balance.
    val_chunks = []
    for (_dom, _lab), grp in train_pool.groupby(["domain", "label"]):
        val_chunks.append(grp.sample(frac=cfg.val_frac, random_state=cfg.seed))
    val_df = pd.concat(val_chunks, ignore_index=False)
    val_ids = set(val_df["post_id"])
    train_df = train_pool[~train_pool["post_id"].isin(val_ids)].reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Inject hard augmentation examples directly into train (never into val/test).
    if cfg.augment_csv is not None:
        aug = pd.read_csv(cfg.augment_csv)
        aug = aug[aug["quality_flags"] == "ok"].copy()
        aug["label"] = aug["label"].map(LABEL2ID).astype(int)
        aug = aug.rename(columns={"original": "text_a", "synthetic": "text_b"})
        aug["post_id"] = "aug_" + aug.index.astype(str)
        aug["domain"] = aug.get("domain", "augmented")
        aug = aug[["text_a", "text_b", "label", "domain", "post_id"]]
        train_df = pd.concat([train_df, aug], ignore_index=True)

    keep = ["text_a", "text_b", "label", "domain", "post_id"]
    return (
        Dataset.from_pandas(train_df[keep]),
        Dataset.from_pandas(val_df[keep]),
        Dataset.from_pandas(test_df[keep]),
    )


def tokenize_dataset(
    ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dataset:
    """Tokenize as a sentence pair: [CLS] text_a [SEP] text_b [SEP]."""

    def _tok(batch):
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollatorWithPadding
        )

    cols_to_remove = [c for c in ds.column_names if c not in ("label", "domain", "post_id")]
    return ds.map(_tok, batched=True, remove_columns=cols_to_remove)
