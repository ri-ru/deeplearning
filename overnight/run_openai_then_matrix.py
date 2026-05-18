"""Wait for the main stage9 generation to finish, regenerate the failed
openai_gpt4omini.csv, then launch stage9_matrix.py to build the transfer
matrix.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CROSS = REPO / "overnight" / "cross_gen"
SLUGS_EXPECTED = {"anthropic_haiku", "meta_llama3_8b", "deepseek_v3",
                  "qwen25_72b", "mistral_7b", "openai_gpt4omini"}


def n_pairs_in(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    # Subtract 1 for header. Don't use pandas (can be slow / fragile).
    try:
        with open(path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def main() -> int:
    print(f"[wrap] waiting for primary stage9 generation to settle ...", flush=True)
    last_total = -1
    stable_loops = 0
    while True:
        # When the primary process is still generating it modifies a file every
        # second or so. We declare "settled" when total pair count is stable
        # for 6 consecutive 30s polls.
        total = sum(n_pairs_in(CROSS / f"{s}.csv") for s in SLUGS_EXPECTED if s != "openai_gpt4omini")
        if total == last_total:
            stable_loops += 1
        else:
            stable_loops = 0
        last_total = total
        if stable_loops >= 6:
            print(f"[wrap] primary settled at {total} non-openai pairs.", flush=True)
            break
        print(f"[wrap] poll: {total} non-openai pairs ({stable_loops}/6 stable polls)", flush=True)
        time.sleep(30)

    # Regenerate openai_gpt4omini
    cmd = [sys.executable, "-u", str(REPO / "overnight" / "stage9_generate.py"),
           "--only", "openai_gpt4omini"]
    print(f"[wrap] regenerating openai_gpt4omini: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[wrap] openai_gpt4omini rc={rc}", flush=True)

    # Inventory
    have = sorted(p.stem for p in CROSS.glob("*.csv") if not p.stem.startswith("_"))
    print(f"[wrap] generators on disk: {have}", flush=True)

    # Launch matrix
    cmd = [sys.executable, "-u", str(REPO / "overnight" / "stage9_matrix.py")]
    print(f"[wrap] running matrix: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[wrap] matrix rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
