#!/usr/bin/env python3
"""Check SameSense's documentation and cached artifacts without ML dependencies."""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote

from project_summary import ROOT, build_summary


def main() -> None:
    errors = []
    for path in [ROOT / "README.md", *sorted((ROOT / "guides").glob("*.md"))]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            destination = unquote(target.split("#", 1)[0])
            if not (path.parent / destination).exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    for path in [*sorted((ROOT / "results").rglob("*.json")),
                 *sorted((ROOT / "seeds/results").glob("*.json"))]:
        try:
            json.loads(path.read_text())
        except (OSError, ValueError) as error:
            errors.append(f"Invalid artifact {path.relative_to(ROOT)}: {error}")
    notebook = json.loads((ROOT / "summary_notebook.ipynb").read_text())
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        errors.append("The implementation notebook is not a non-empty v4 notebook")
    ET.parse(ROOT / "assets/samesense-banner.svg")
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        match = re.search(r"^csv_path:\s*(.+)$", path.read_text(), flags=re.MULTILINE)
        if not match or not (ROOT / "code" / match.group(1)).is_file():
            errors.append(f"Training CSV does not resolve from code/: {path.name}")
    summary = build_summary()
    if summary["original_usable_pairs"] != 52137:
        errors.append("Original pair count no longer matches the README; update the inventory")
    if summary["cross_generator_total"] != 998:
        errors.append("Cross-generator pair count no longer matches the README; update the inventory")
    readme = (ROOT / "README.md").read_text()
    for row in summary["evaluations"]:
        documented = (f"| {row['n']} | {row['single_generator_accuracy']:.0%} | "
                      f"**{row['multi_generator_accuracy']:.0%}** |")
        if documented not in readme:
            errors.append(f"README result differs from its source artifact: {row['set']}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print("Passed: local documentation links, JSON/SVG/notebook integrity, portable CSV paths,")
    print("        dataset inventory, and README metrics against their source artifacts.")


if __name__ == "__main__":
    main()
