"""Wait for Stage 3 timeline.json (or timeout), then run evolve.py.

Use this when the parent run_all.py was launched with the old code (before
evolve was wired in) — bolts evolution onto the back of an already-running
pipeline.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE3_TIMELINE = REPO / "overnight" / "results" / "stage3" / "timeline.json"
STAGE1_DIR = REPO / "overnight" / "results" / "stage1"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max_wait_min", type=int, default=180)
    p.add_argument("--min_stage1", type=int, default=4,
                   help="Minimum stage1 variants finished before evolve fires.")
    p.add_argument("--max_iters", type=int, default=3)
    p.add_argument("--per_iter", type=int, default=2)
    args = p.parse_args()

    start = time.time()
    print(f"[evolve-wait] waiting up to {args.max_wait_min} min for Stage 3 to finish", flush=True)

    while True:
        elapsed_min = (time.time() - start) / 60
        if elapsed_min > args.max_wait_min:
            print(f"[evolve-wait] timed out after {elapsed_min:.1f} min", flush=True)
            break
        n_stage1 = len(list(STAGE1_DIR.glob("v*.json"))) if STAGE1_DIR.exists() else 0
        if STAGE3_TIMELINE.exists() and n_stage1 >= args.min_stage1:
            print(f"[evolve-wait] preconditions met "
                  f"(stage3 timeline + {n_stage1} stage1 variants); launching evolve", flush=True)
            break
        time.sleep(60)

    cmd = [sys.executable, str(REPO / "overnight" / "evolve.py"),
           "--max_iters", str(args.max_iters), "--per_iter", str(args.per_iter)]
    print(f"[evolve-wait] running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[evolve-wait] evolve.py rc={rc}", flush=True)

    # Final summarize pass
    sumcmd = [sys.executable, str(REPO / "overnight" / "summarize.py")]
    subprocess.call(sumcmd)
    print("[evolve-wait] summarize done", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
