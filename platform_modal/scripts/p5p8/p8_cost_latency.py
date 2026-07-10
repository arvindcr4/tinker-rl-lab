#!/usr/bin/env python3
"""P8 cost-per-decision under realistic budgets + latency budgets (iter 8).

Inputs
------
This is a calculation script (no model fitting); uses the iter-4 cost
accounting baseline + a sensitivity sweep.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_cost_latency_sensitivity.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_latency_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

# Internal token prices (USD per 1k tokens), June 2026.  These are the
# canonical numbers in the iter-4 cost_accounting.tsv.
PRICE_IN_PER_1K = 0.20 / 1000.0  # input tokens, $0.20 / 1M = $0.0002 / 1k
PRICE_OUT_PER_1K = 0.60 / 1000.0  # output tokens, $0.60 / 1M = $0.0006 / 1k

# Reference tree inference time, from xgboost_results.json.
TREE_INFER_S_PER_10K = 0.006184816360473633


def write_tsv(path: Path, header, rows):
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")


def main():
    # ---- Q1: token-price sensitivity on hybrid architecture ----
    # Hybrid: 10% of traffic (the alert fraction) goes to the LLM sensor
    # while the remaining 90% is tree-scored.  Per the iter-4 baseline,
    # a per-row prompt of 120 input tokens + 5 output tokens costs
    # 120*$0.0002 + 5*$0.0006 = $0.027 / 1k rows = $0.027 per 10k LLM calls.
    # Wait: per-row cost = 120*0.0002/1000 + 5*0.0006/1000 = 0.000024 + 0.000003 = $0.000027 / row.
    # Per 10k tree-only = $1.00; per 10k LLM hybrid = 9000*tree + 1000*$0.000027 = 1000*tree + $0.027
    # That works out to ~$1.03 hybrid vs $1.00 tree. Hmm - the iter-4 TSV
    # reports $35 hybrid. That implies a different per-row price.
    # Re-examine: the iter-4 doc says "$0.0035 per 1 row" -- so the assumed
    # per-row price is much higher than current spot rates.  This iter
    # sweeps across the realistic range from $0.00001/row to $0.01/row.

    price_per_row_grid = [
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
    ]
    cost_rows = [["price_per_row_usd", "tree_only_10k_usd",
                  "llm_scorer_10k_usd", "hybrid_10pct_llm_10k_usd",
                  "hybrid_minus_tree_only"]]
    cost_json = {"tree_inference_s_per_10k": TREE_INFER_S_PER_10K,
                 "tree_cost_per_10k_usd": 1.00,
                 "tree_only_baseline_usd": 1.00,
                 "llm_coverage_frac": 0.10,
                 "sweep": []}

    for p in price_per_row_grid:
        tree_only = 1.00  # canonical from xgboost_results.json
        llm_scorer = round(p * 10_000, 3)
        hybrid = round(0.90 * tree_only + 0.10 * llm_scorer, 3)
        cost_rows.append([p, round(tree_only, 3), llm_scorer, hybrid,
                          round(hybrid - tree_only, 3)])
        cost_json["sweep"].append({
            "price_per_row_usd": p,
            "llm_scorer_10k_usd": llm_scorer,
            "hybrid_10pct_llm_10k_usd": hybrid,
            "hybrid_minus_tree_only": round(hybrid - tree_only, 3),
        })
        print(f"[p8-cost] ${p}/row  hybrid_10k=${hybrid}  "
              f"vs_tree_only=${hybrid - tree_only:+.3f}", file=sys.stderr)

    write_tsv(RES / "p8_cost_latency_sensitivity.tsv",
              cost_rows[0], cost_rows[1:])

    # ---- Q2: latency budget analysis ----
    # Card authorization deadline ~ 250ms (well-known industry estimate).
    # Tree path is ~6ms / 10k rows = ~6e-4 ms / row = 0.6us / row.
    # The LLM path latency depends on prompt size and model size; a Qwen3.5-4B
    # forward pass over 120 input tokens is roughly 50-150 ms on a single A100.
    # We sweep LLM latency from 1ms to 1000ms and report the per-row
    # authorization budget consumed.
    latency_grid = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    lat_rows = [["llm_per_row_latency_ms",
                 "tree_per_row_latency_us",
                 "tree_pct_of_budget",
                 "llm_pct_of_250ms_budget",
                 "llm_fits_in_250ms"]]
    BUDGET_MS = 250.0
    tree_us_per_row = TREE_INFER_S_PER_10K * 1e6 / 10_000  # us / row
    lat_json = {"auth_budget_ms": BUDGET_MS,
                "tree_us_per_row": round(tree_us_per_row, 3),
                "sweep": []}
    for lat in latency_grid:
        tree_pct = round(tree_us_per_row / (BUDGET_MS * 1000) * 100, 6)
        llm_pct = round(lat / BUDGET_MS * 100, 2)
        fits = "yes" if lat < BUDGET_MS else "no"
        lat_rows.append([lat, round(tree_us_per_row, 3), tree_pct, llm_pct, fits])
        lat_json["sweep"].append({
            "llm_per_row_latency_ms": lat,
            "llm_pct_of_budget": llm_pct,
            "llm_fits_in_250ms": fits,
        })
        print(f"[p8-lat] {lat}ms LLM  -> {llm_pct:.1f}% of 250ms budget  "
              f"fits={fits}", file=sys.stderr)

    write_tsv((RES / "p8_latency_budget.tsv"),
              lat_rows[0], lat_rows[1:])
    cost_json["latency"] = lat_json

    (RES / "p8_cost_latency_summary.json").write_text(
        json.dumps(cost_json, indent=2, sort_keys=True))
    print("[p8-cost-latency] done.", file=sys.stderr)


if __name__ == "__main__":
    main()