"""Single-experiment runner for the research loop.

Thin wrapper around train.py that:
  1. Validates the variant config
  2. Checks the budget cap (RESEARCH_LOOP_BUDGET_USD) — refuses to run if
     the current total is within $5 of the cap
  3. Invokes train.py as a subprocess so each variant gets a clean process
     (Tinker SDK state, GC, etc.)
  4. Parses METRIC lines from stdout
  5. Appends the canonical result row to research_loop/results.jsonl

Usage:
    python research_loop/run_one.py --config research_loop/variant_configs/wave_001/v001.yaml

Exit codes:
    0 — completed
    1 — config or infrastructure error
    2 — training failed
    3 — budget tripwire blocked the run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RESULTS_JSONL = ROOT / "results.jsonl"
SPEND_FILE = ROOT / "spend_usd.txt"

# Heuristic cost per training run on Tinker API (Phase 1: small model, 100 steps)
# Refine after the first 10 runs with actual numbers.
PHASE1_COST_ESTIMATE_USD = 0.40


def current_spend() -> float:
    if not SPEND_FILE.exists():
        return 0.0
    try:
        return float(SPEND_FILE.read_text().strip())
    except (ValueError, OSError):
        return 0.0


def record_spend(delta_usd: float) -> None:
    total = current_spend() + delta_usd
    SPEND_FILE.write_text(f"{total:.4f}\n")


def check_budget(estimate_usd: float) -> None:
    cap_env = os.environ.get("RESEARCH_LOOP_BUDGET_USD")
    if not cap_env:
        return  # no cap configured
    try:
        cap = float(cap_env)
    except ValueError:
        return
    spent = current_spend()
    if spent + estimate_usd > cap - 5.0:
        print(
            f"BUDGET TRIPWIRE: spent ${spent:.2f} + est ${estimate_usd:.2f} "
            f"would cross ${cap:.2f} cap (within $5). Refusing to run.",
            file=sys.stderr,
        )
        sys.exit(3)


def parse_metric_lines(stdout: str) -> dict:
    metrics = {}
    for line in stdout.splitlines():
        if line.startswith("METRIC "):
            kv = line[len("METRIC "):].strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    metrics[k] = float(v)
                except ValueError:
                    metrics[k] = v
    return metrics


def append_result(row: dict) -> None:
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds per run (default 1h)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate config and budget without running")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    try:
        cfg = yaml.safe_load(args.config.read_text())
    except yaml.YAMLError as exc:
        print(f"ERROR: invalid YAML: {exc}", file=sys.stderr)
        return 1

    variant_id = cfg.get("name") or args.config.stem
    wave_dir = args.config.parent.name  # e.g., wave_001
    output_json = ROOT / "wave_briefs" / wave_dir / f"{variant_id}.result.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    check_budget(PHASE1_COST_ESTIMATE_USD)

    if args.dry_run:
        print(f"DRY RUN OK: {variant_id}")
        print(json.dumps(cfg, indent=2))
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config", str(args.config),
        "--output-json", str(output_json),
    ]
    print(f"[run_one] launching {variant_id} → {output_json}", flush=True)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        row = {
            "variant_id": variant_id,
            "config_path": str(args.config),
            "status": "timeout",
            "wall_clock_seconds": int(time.time() - t0),
        }
        append_result(row)
        record_spend(PHASE1_COST_ESTIMATE_USD * 0.5)  # partial charge
        print(f"TIMEOUT after {args.timeout}s", file=sys.stderr)
        return 2

    # Stream stdout/stderr to our console for visibility
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    metrics = parse_metric_lines(proc.stdout)
    status = metrics.get("status", "unknown")

    row = {
        "variant_id": variant_id,
        "wave": wave_dir,
        "config_path": str(args.config),
        "config": cfg,
        "metrics": metrics,
        "status": status,
        "wall_clock_seconds": int(time.time() - t0),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_result(row)
    record_spend(PHASE1_COST_ESTIMATE_USD)

    if proc.returncode != 0 or status != "completed":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
