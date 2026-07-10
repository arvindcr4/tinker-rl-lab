#!/usr/bin/env python3
"""P8 JOB A (iter 164): breakeven-tier analysis on iter-160 outputs.

Fresh vein. Iter-160 OPTIMIZED tau per (rate x fset x tier x utility) cell
and reported realized utility(τ*) with 5-seed bootstrap CI. Iter-164 answers
the OPERATIONAL follow-up:

    At each (rate x fset), what is the CHEAPEST cost tier at which VALUE-max
    utility recovers a target level (0.99, 0.95, 0.90)?  Conversely, where does
    VALUE-max degrade below the target, and how much does the deployment need
    to spend to clear the bar?

Inputs (read-only — no XGBoost retraining):
  platform_hybrid/experiments/results/p5p8/p8_iter160_opt_util_per_cell.tsv   (2000 rows)
  platform_hybrid/experiments/results/p5p8/p8_iter160_opt_tau_per_cell.tsv     (2000 rows)

Outputs:
  platform_hybrid/experiments/results/p5p8/p8_iter164_breakeven_per_cell.tsv  (60 rows: 5 rates x 4 fsets x 3 targets)
  platform_hybrid/experiments/results/p5p8/p8_iter164_tau_tier_monotone.tsv   (40 rows: 5 tiers x 4 fsets x 2 rates)
  platform_hybrid/experiments/results/p5p8/p8_iter164_summary.json            (machine-readable H1-H4 verdicts)

Stdlib only. <= 280 lines.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"

# Tier cost (matches iter-160 / p8_iter160_operating_point_utility.py)
TIERS = [
    ("cheap_heuristic", 0.0001),
    ("small_open",      0.0006),
    ("iter120_default", 0.0010),
    ("mid_tier",        0.0050),
    ("frontier_gpt4",   0.0300),
]
RATES = [1.44, 1.00, 0.50, 0.10, 0.05]
FSETS = ["20raw", "20raw+minmax", "20raw+stat", "24full"]
SEEDS = [20260706, 20260708, 20260710, 20260712, 20260714]
TARGETS = [0.99, 0.95, 0.90]


def load_tsv(path):
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        return list(rdr)


def main():
    print("Loading iter-160 outputs...", flush=True)
    util_rows = load_tsv(RES / "p8_iter160_opt_util_per_cell.tsv")
    tau_rows = load_tsv(RES / "p8_iter160_opt_tau_per_cell.tsv")
    print(f"  util_rows: {len(util_rows)}, tau_rows: {len(tau_rows)}")

    # Group VALUE-max util by (rate, fset, tier) -> list[util per seed]
    value_per_cell = defaultdict(list)
    tau_per_cell = defaultdict(list)
    for r in util_rows:
        if r["utility"] != "VALUE":
            continue
        key = (float(r["rate_pct"]), r["fset"], r["tier"])
        value_per_cell[key].append(float(r["opt_util"]))
    for r in tau_rows:
        if r["utility"] != "VALUE":
            continue
        key = (float(r["rate_pct"]), r["fset"], r["tier"])
        tau_per_cell[key].append(float(r["opt_tau"]))

    # ----- breakeven per cell -----
    breakeven_rows = []
    H1_pass = 0
    H1_total = 0
    H2_pass = 0
    H2_total = 0
    H3_pass = 0
    H3_total = 0
    H4a_pass = 0
    H4a_total = 0
    H5_pass = 0
    H5_total = 0
    for rate in RATES:
        for fset in FSETS:
            for target in TARGETS:
                # tier-by-tier mean util
                tier_means = {}
                for tier_name, tier_cost in TIERS:
                    vals = value_per_cell[(rate, fset, tier_name)]
                    tier_means[tier_name] = sum(vals) / len(vals) if vals else float("nan")
                # find cheapest tier with mean >= target
                be_tier = None
                be_cost = None
                be_util = None
                for tier_name, tier_cost in TIERS:
                    if tier_means[tier_name] >= target:
                        be_tier = tier_name
                        be_cost = tier_cost
                        be_util = tier_means[tier_name]
                        break
                breakeven_rows.append({
                    "rate_pct": rate, "fset": fset, "target": target,
                    "breakeven_tier": be_tier if be_tier else "NONE",
                    "breakeven_cost_per_call": be_cost if be_cost is not None else "",
                    "breakeven_mean_util": be_util if be_util is not None else "",
                    "mean_util_cheap": tier_means["cheap_heuristic"],
                    "mean_util_small": tier_means["small_open"],
                    "mean_util_iter120": tier_means["iter120_default"],
                    "mean_util_mid": tier_means["mid_tier"],
                    "mean_util_frontier": tier_means["frontier_gpt4"],
                })
                # H1: cheap_heuristic clears target on this cell (bar 80%)
                if tier_means["cheap_heuristic"] >= target:
                    H1_pass += 1
                H1_total += 1
                # H2: small_open clears target on this cell (bar 80%)
                if tier_means["small_open"] >= target:
                    H2_pass += 1
                H2_total += 1
                # H3: even frontier_gpt4 recovers util >= 0.95 (bar 90%)
                if tier_means["frontier_gpt4"] >= 0.95:
                    H3_pass += 1
                H3_total += 1
                # H4a: breakeven tier is the CHEAPEST tier (cheap_heuristic)
                # at this (rate x fset x target) cell
                if be_tier == "cheap_heuristic":
                    H4a_pass += 1
                H4a_total += 1
                # H5: frontier_gpt4 still clears the same target as cheap
                # (i.e., frontier does not add VALUE — value is robust)
                if tier_means["frontier_gpt4"] >= target:
                    H5_pass += 1
                H5_total += 1

    # ----- τ* monotonicity in tier -----
    tau_rows_out = []
    H4_pass = 0
    H4_total = 0
    for rate in RATES:
        for fset in FSETS:
            tier_means_tau = {}
            for tier_name, _ in TIERS:
                vals = tau_per_cell[(rate, fset, tier_name)]
                tier_means_tau[tier_name] = sum(vals) / len(vals) if vals else float("nan")
            # monotonicity: cheap <= small_open <= iter120 <= mid <= frontier
            mono = (
                tier_means_tau["cheap_heuristic"] <= tier_means_tau["small_open"] + 1e-9
                and tier_means_tau["small_open"] <= tier_means_tau["iter120_default"] + 1e-9
                and tier_means_tau["iter120_default"] <= tier_means_tau["mid_tier"] + 1e-9
                and tier_means_tau["mid_tier"] <= tier_means_tau["frontier_gpt4"] + 1e-9
            )
            if mono:
                H4_pass += 1
            H4_total += 1
            tau_rows_out.append({
                "rate_pct": rate, "fset": fset,
                "tau_cheap": tier_means_tau["cheap_heuristic"],
                "tau_small_open": tier_means_tau["small_open"],
                "tau_iter120": tier_means_tau["iter120_default"],
                "tau_mid": tier_means_tau["mid_tier"],
                "tau_frontier": tier_means_tau["frontier_gpt4"],
                "monotone_increasing": int(mono),
            })

    # ----- write outputs -----
    print("\nWriting outputs...", flush=True)
    out_be = RES / "p8_iter164_breakeven_per_cell.tsv"
    with out_be.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(breakeven_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(breakeven_rows)
    print(f"  {out_be} ({len(breakeven_rows)} rows)")

    out_tau = RES / "p8_iter164_tau_tier_monotone.tsv"
    with out_tau.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tau_rows_out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(tau_rows_out)
    print(f"  {out_tau} ({len(tau_rows_out)} rows)")

    # verdicts
    H1_verdict = "PASS" if H1_pass >= 0.80 * H1_total else "FAIL"
    H2_verdict = "PASS" if H2_pass >= 0.80 * H2_total else "FAIL"
    H3_verdict = "PASS" if H3_pass >= 0.90 * H3_total else "FAIL"
    H4_verdict = "PASS" if H4_pass >= 0.80 * H4_total else "FAIL"
    H4a_verdict = "PASS" if H4a_pass >= 0.80 * H4a_total else "FAIL"
    H5_verdict = "PASS" if H5_pass >= 0.80 * H5_total else "FAIL"

    summary = {
        "iter": 164,
        "pillar": "P8",
        "job": "A",
        "vein": "iter-161 mint vein #2 — breakeven tier extension",
        "h1_cheap_clears_target": {
            "pass": H1_verdict == "PASS",
            "n_pass": H1_pass, "n_total": H1_total,
            "fraction": round(H1_pass / max(1, H1_total), 4),
            "bar": 0.80, "verdict": H1_verdict,
        },
        "h2_small_open_clears_target": {
            "pass": H2_verdict == "PASS",
            "n_pass": H2_pass, "n_total": H2_total,
            "fraction": round(H2_pass / max(1, H2_total), 4),
            "bar": 0.80, "verdict": H2_verdict,
        },
        "h3_frontier_recovers_0p95": {
            "pass": H3_verdict == "PASS",
            "n_pass": H3_pass, "n_total": H3_total,
            "fraction": round(H3_pass / max(1, H3_total), 4),
            "bar": 0.90, "verdict": H3_verdict,
        },
        "h4_tau_monotone_in_tier": {
            "pass": H4_verdict == "PASS",
            "n_pass": H4_pass, "n_total": H4_total,
            "fraction": round(H4_pass / max(1, H4_total), 4),
            "bar": 0.80, "verdict": H4_verdict,
        },
        "h4a_breakeven_is_cheapest": {
            "pass": H4a_verdict == "PASS",
            "n_pass": H4a_pass, "n_total": H4a_total,
            "fraction": round(H4a_pass / max(1, H4a_total), 4),
            "bar": 0.80, "verdict": H4a_verdict,
        },
        "h5_frontier_matches_cheap_on_target": {
            "pass": H5_verdict == "PASS",
            "n_pass": H5_pass, "n_total": H5_total,
            "fraction": round(H5_pass / max(1, H5_total), 4),
            "bar": 0.80, "verdict": H5_verdict,
        },
    }
    out_sum = RES / "p8_iter164_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {out_sum}")
    print("\n=== Hypothesis verdicts ===")
    for k, v in summary.items():
        if k.startswith("h"):
            print(f"  {k}: {v['verdict']}  ({v['n_pass']}/{v['n_total']} = {v['fraction']*100:.1f}%)")


if __name__ == "__main__":
    main()