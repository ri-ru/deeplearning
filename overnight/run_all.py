"""Top-level orchestrator: runs Stage 1 -> Stage 2 -> Stage 3 -> summarize.

Each stage is its own subprocess so a crash in one doesn't kill the others.
Each stage is independently resumable: re-running this script will skip any
experiment whose result JSON already exists.

Run overnight like this (PowerShell):

    $env:PYTHONIOENCODING = "utf-8"
    & "C:\\Users\\jan\\downloads\\nml_project_proposal\\.venv\\Scripts\\python.exe" `
        "C:\\Users\\jan\\Downloads\\deep learning\\overnight\\run_all.py"

The script regenerates OVERNIGHT_REPORT.md after each stage so you can peek
at partial results while it's running.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OVERNIGHT = REPO / "overnight"
LOG = OVERNIGHT / "run.log"
PYBIN = sys.executable


def stamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {msg}\n"
    print(line, end="", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)


def run_stage(name: str, script: Path, extra: list[str] | None = None) -> int:
    cmd = [PYBIN, "-u", str(script)] + (extra or [])
    log(f"START {name}: {' '.join(cmd)}")
    try:
        rc = subprocess.call(cmd)
        log(f"END {name}: rc={rc}")
        return rc
    except KeyboardInterrupt:
        log(f"INTERRUPTED {name}")
        raise
    except Exception as e:
        log(f"FAILED {name}: {type(e).__name__}: {e}")
        return 1


def summarize() -> None:
    log("Running summarize.py")
    subprocess.call([PYBIN, str(OVERNIGHT / "summarize.py")])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="", help="comma-sep stages to skip (s1,s2,s3)")
    parser.add_argument("--cleanup_checkpoints", action="store_true",
                        help="delete stage1 checkpoint files after scoring")
    args = parser.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    log(f"=== overnight run starting ===  python={PYBIN}")

    if "s1" not in skip:
        extra = ["--cleanup_checkpoints"] if args.cleanup_checkpoints else []
        run_stage("stage1", OVERNIGHT / "stage1_mitigations.py", extra)
        summarize()

    if "evolve" not in skip:
        evo_extra = ["--max_iters", "3", "--per_iter", "2"]
        if args.cleanup_checkpoints:
            evo_extra.append("--cleanup_checkpoints")
        run_stage("evolve", OVERNIGHT / "evolve.py", evo_extra)
        summarize()

    if "s2" not in skip:
        run_stage("stage2", OVERNIGHT / "stage2_interp.py", ["--include_best_mitigation"])
        summarize()

    if "s3" not in skip:
        run_stage("stage3", OVERNIGHT / "stage3_training_dynamics.py")
        summarize()

    log("=== overnight run complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
