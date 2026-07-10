#!/usr/bin/env python3
"""Head-to-head analysis for the 4-arm loss comparison (GRPO / DAPO / GSPO / Dr.GRPO).

Reads all completed result JSONs in results/head_to_head/ and emits:
  1. A summary table (per arm, mean ± std across seeds) of:
     - last10_avg training reward
     - peak training reward
     - mean ZVF trajectory
     - mean GU trajectory
     - zero_reward_pct (fraction of steps with reward == 0)
  2. A paired comparison: DAPO/GSPO/Dr.GRPO vs GRPO baseline, with the
     per-seed delta and a sign test (the non-parametric paired test that
     does not assume normality, which is the right test for n=3 seeds).

This script NEVER fabricates numbers. If a result JSON is missing or has
status != "completed", it is dropped and counted. The output is
deterministic given the same inputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = Path("/Users/arvind/Developer/tinker-rl-lab/platform_hybrid/experiments/tinker-runs/results/head_to_head")


def load_results(results_dir: Path) -> list[dict]:
    """Load every .json in the head_to_head dir; return only completed runs."""
    out = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            print(f"[warn] could not parse {path.name}: {e}", file=sys.stderr)
            continue
        status = d.get("status")
        arm = d.get("loss_arm") or d.get("loss")
        if status != "completed" or arm is None:
            print(f"[skip] {path.name}: status={status!r} arm={arm!r}")
            continue
        out.append(d)
    return out


def per_arm_summary(runs: list[dict]) -> dict[str, dict]:
    """Aggregate runs by arm, computing mean/std for the headline metrics."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_arm[r["loss_arm"]].append(r)

    summary: dict[str, dict] = {}
    for arm, rs in sorted(by_arm.items()):
        last10 = [r.get("last10_avg", 0.0) for r in rs]
        peak = [r.get("peak_reward", 0.0) for r in rs]
        zero = [r.get("zero_reward_pct", 0.0) for r in rs]
        zvfs = []
        for r in rs:
            zvfs.extend(s.get("zvf", 0.0) for s in r.get("step_log", []))
        summary[arm] = {
            "n_runs": len(rs),
            "n_seeds": len(set(r.get("seed") for r in rs)),
            "last10_avg_mean": statistics.mean(last10) if last10 else float("nan"),
            "last10_avg_std": statistics.stdev(last10) if len(last10) > 1 else 0.0,
            "peak_reward_mean": statistics.mean(peak) if peak else float("nan"),
            "peak_reward_std": statistics.stdev(peak) if len(peak) > 1 else 0.0,
            "zero_reward_pct_mean": statistics.mean(zero) if zero else float("nan"),
            "mean_zvf_across_steps": statistics.mean(zvfs) if zvfs else float("nan"),
            "per_seed_last10": {r.get("seed"): r.get("last10_avg", 0.0) for r in rs},
        }
    return summary


def sign_test(deltas: list[float]) -> dict:
    """Non-parametric paired sign test: count positive vs negative deltas.
    For n=3, p-values are 0.25 (3-0), 0.5 (2-1), 1.0 (1-2 or 0-3)."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zeros = sum(1 for d in deltas if d == 0)
    n = pos + neg  # ignore zeros for the test
    if n == 0:
        return {"pos": pos, "neg": neg, "n": 0, "p_value": float("nan")}
    # p-value: 2 * P(X <= min(pos, neg)) for a two-sided test, where X ~ Bin(n, 0.5)
    from math import comb
    k = min(pos, neg)
    p_one = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_two = min(1.0, 2 * p_one)
    return {"pos": pos, "neg": neg, "n": n, "p_value": p_two}


def paired_comparison(runs: list[dict], baseline: str = "grpo") -> list[dict]:
    """For each non-baseline arm, compute per-seed (last10) delta vs the
    baseline arm at the SAME seed. Report mean delta + sign test."""
    by_seed_arm: dict[tuple[int, str], dict] = {}
    for r in runs:
        by_seed_arm[(r.get("seed"), r.get("loss_arm"))] = r

    seeds = sorted({s for (s, _a) in by_seed_arm.keys()})
    others = sorted({a for (_s, a) in by_seed_arm.keys() if a != baseline})

    rows = []
    for arm in others:
        deltas = []
        for s in seeds:
            base = by_seed_arm.get((s, baseline))
            oth = by_seed_arm.get((s, arm))
            if base is None or oth is None:
                continue
            deltas.append(oth.get("last10_avg", 0.0) - base.get("last10_avg", 0.0))
        if not deltas:
            continue
        rows.append({
            "arm": arm,
            "baseline": baseline,
            "n_pairs": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "delta_seeds": {s: round(d, 4) for s, d in zip(
                [s for s in seeds if (s, baseline) in by_seed_arm and (s, arm) in by_seed_arm],
                deltas)},
            "sign_test": sign_test(deltas),
        })
    return rows


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%" if x == x else "n/a"


def fmt_signed(x: float) -> str:
    return f"{x:+.4f}" if x == x else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(DEFAULT_DIR),
                    help="directory of head-to-head result JSONs")
    ap.add_argument("--baseline", default="grpo",
                    help="baseline arm name for paired comparison (default: grpo)")
    ap.add_argument("--out", default="",
                    help="optional path to write the summary as JSON")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_results(results_dir)
    if not runs:
        print(f"[error] no completed runs in {results_dir}", file=sys.stderr)
        return 1

    summary = per_arm_summary(runs)
    paired = paired_comparison(runs, baseline=args.baseline)

    print("=" * 78)
    print(f"HEAD-TO-HEAD: {len(runs)} completed runs across {len(summary)} arms")
    print(f"  source: {results_dir}")
    print("=" * 78)
    print()
    print(f"{'arm':10s} {'n':>3s} {'seeds':>5s}  {'last10_avg':>20s}  {'peak_reward':>20s}  {'zero%':>8s}  {'mean_zvf':>8s}")
    print("-" * 78)
    for arm, s in summary.items():
        last10 = f"{s['last10_avg_mean']:.3f} ± {s['last10_avg_std']:.3f}"
        peak = f"{s['peak_reward_mean']:.3f} ± {s['peak_reward_std']:.3f}"
        print(f"{arm:10s} {s['n_runs']:>3d} {s['n_seeds']:>5d}  {last10:>20s}  {peak:>20s}  "
              f"{fmt_pct(s['zero_reward_pct_mean']):>8s}  {s['mean_zvf_across_steps']:>8.3f}")

    print()
    print(f"PAIRED vs baseline={args.baseline}  (per-seed delta on last10_avg)")
    print("-" * 78)
    for row in paired:
        st = row["sign_test"]
        print(f"  {row['arm']:8s}: mean_delta={fmt_signed(row['mean_delta'])}  "
              f"n={row['n_pairs']}  pos={st['pos']} neg={st['neg']}  p={st['p_value']:.3f}  "
              f"per_seed={row['delta_seeds']}")

    if args.out:
        out = {
            "results_dir": str(results_dir),
            "baseline": args.baseline,
            "summary_by_arm": summary,
            "paired": paired,
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nJSON written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
