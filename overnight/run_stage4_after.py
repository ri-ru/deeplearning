"""Wait for the main pipeline to finish (or at least Stage 1 to be well
underway), then run Stage 4 hate-speech classifier training.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO / "overnight" / "results" / "stage1"
STAGE3_TIMELINE = REPO / "overnight" / "results" / "stage3" / "timeline.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max_wait_min", type=int, default=300)
    p.add_argument("--min_stage1", type=int, default=6,
                   help="Wait until at least this many stage1 variants exist.")
    p.add_argument("--also_wait_stage3", action="store_true",
                   help="Additionally wait for stage3 timeline.json")
    args = p.parse_args()

    start = time.time()
    print(f"[stage4-wait] waiting up to {args.max_wait_min} min for stage1 "
          f">= {args.min_stage1} and (optionally) stage3", flush=True)
    while True:
        if (time.time() - start) / 60 > args.max_wait_min:
            print("[stage4-wait] timeout; launching anyway", flush=True)
            break
        n_stage1 = len(list(STAGE1_DIR.glob("v*.json"))) if STAGE1_DIR.exists() else 0
        stage3_ok = STAGE3_TIMELINE.exists() if args.also_wait_stage3 else True
        if n_stage1 >= args.min_stage1 and stage3_ok:
            print(f"[stage4-wait] preconditions met: stage1={n_stage1}; launching", flush=True)
            break
        time.sleep(60)

    cmd = [sys.executable, str(REPO / "overnight" / "stage4_hate_classifiers.py")]
    print(f"[stage4-wait] running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[stage4-wait] rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
