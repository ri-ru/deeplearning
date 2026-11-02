#!/usr/bin/env python3
"""Inspect SameSense's committed data and evaluations; no model or dependencies needed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = (
    ("Human-written", "stage14b_matrix.json", "manual_handwritten_n20"),
    ("Claude-written", "stage14b_matrix.json", "claude46_n20"),
    ("GPT-4o-mini", "stage14_matrix.json", "stage7_gpt_n50"),
    ("Claude Haiku", "stage14_matrix.json", "stage10_haiku_n100"),
)


def usable_pairs(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(
            1 for row in csv.DictReader(handle)
            if row.get("meaning") in {"equivalent", "unequivalent"}
            and row.get("explanation", "").strip()
            and row.get("synthetic", "").strip()
        )


def build_summary() -> dict:
    inventory = {
        path.stem: usable_pairs(path)
        for path in sorted((ROOT / "results/cross_gen").glob("*.csv"))
    }
    matrices = {
        name: json.loads((ROOT / "results/overnight" / name).read_text())
        for name in {item[1] for item in EVALUATIONS}
    }
    evaluations = []
    for name, filename, key in EVALUATIONS:
        matrix = matrices[filename]
        baseline = matrix["poster_baseline"][key]
        multi = matrix["multi_gen"][key]
        if baseline["n"] != multi["n"]:
            raise ValueError(f"Evaluation counts differ for {name}")
        evaluations.append({
            "set": name,
            "n": multi["n"],
            "single_generator_accuracy": baseline["acc"],
            "multi_generator_accuracy": multi["acc"],
            "source": f"results/overnight/{filename}",
        })
    return {
        "project": "SameSense",
        "original_usable_pairs": usable_pairs(ROOT / "data/reformulated_full.csv"),
        "cross_generator_usable_pairs": inventory,
        "cross_generator_total": sum(inventory.values()),
        "evaluations": evaluations,
        "note": "Cached results, not a new evaluation. Small sets do not establish production reliability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        summary = build_summary()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot load the committed artifacts: {error}\n")
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print("SameSense — semantic equivalence for moderation rationales\n")
    print(f"Original labelled pairs: {summary['original_usable_pairs']:,}")
    print(f"Cross-generator pairs:   {summary['cross_generator_total']:,}")
    for generator, count in summary["cross_generator_usable_pairs"].items():
        status = f"{count} usable pairs" if count else "no usable pairs (excluded)"
        print(f"  {generator:<20} {status}")
    print(f"\n{'Evaluation set':<22} {'n':>4}  {'Single-generator':>16}  {'Multi-generator':>15}")
    for row in summary["evaluations"]:
        print(f"{row['set']:<22} {row['n']:>4}  "
              f"{row['single_generator_accuracy']:>16.0%}  "
              f"{row['multi_generator_accuracy']:>15.0%}")
    print(f"\n{summary['note']}")


if __name__ == "__main__":
    main()
