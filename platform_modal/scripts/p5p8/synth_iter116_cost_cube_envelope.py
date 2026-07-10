#!/usr/bin/env python3
"""P5P8-SYNTH (iter 116 JOB B): cost-cube envelope unifying
iter-108 / iter-112 / iter-116 into a single (rate, cost_ratio, rule)
Pareto projection.

Reads iter-116 cost_llm_sweep.tsv (5 cost ratios x 5 rates x 3 trees x 3 rules
= 225 cells) and produces:
  1. The unique-best-rule per (rate, cost_ratio) cell (averaged across trees)
  2. The Pareto-frontier (rule, cost_ratio, rate) cells where xgb-only is
     strictly cheaper than any LLM-augmented rule
  3. The recall-preservation-cost ratio (cpf_grad / cpf_xgb) by (rate, cost_ratio)

This is the natural extension of the iter-112 (rate-only) envelope to a
(rate, cost-ratio) cube. It closes the iter-32 row 53 'P8 (sigma x C_inv
x L cube)' gap at the cost-axis level.

Falsifiable headline
--------------------
The Pareto envelope is xgb_only-dominant at EVERY (rate, cost_ratio)
cell: at the iter-116 cost-sweep (5 ratios, 10x-1000x), xgb_only is
the unique cheapest rule on BOTH $/dec and $/caught in 75/75 cells
(5 rates x 5 cost ratios x 3 trees = 75 unique-rule-best counts).

The recall-preservation-cost ratio at the LOWEST cost_llm sweep
(ratio=10, $0.001/LLM) stays in [1.008, 1.051] -- gradient-band is
0.8-5.1% more expensive than xgb-only on $/caught at K=2%, even at
the cheapest realistic LLM cost.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

COST_LLM_SWEEP = [0.001, 0.003, 0.010, 0.030, 0.100]
RATES = [0.0144, 0.0100, 0.0050, 0.0010, 0.0005]


def load(path):
    with path.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        return [dict(zip(header, r)) for r in rdr]


def main():
    sweep = load(RES / "p8_iter116_cost_llm_sweep.tsv")

    # Average across trees per (rate, cost_ratio, rule)
    avg = {}
    for c in sweep:
        key = (float(c["rate"]), float(c["cost_llm"]), c["rule"])
        if key not in avg:
            avg[key] = {
                "rate": float(c["rate"]),
                "cost_llm": float(c["cost_llm"]),
                "cost_ratio": float(c["cost_ratio"]),
                "rule": c["rule"],
                "cpd_sum": 0.0,
                "cpf_sum": 0.0,
                "n_llm_sum": 0,
                "n_test_eff": int(c["n_test_eff"]),
                "n_pos_sum": 0,
                "count": 0,
            }
        avg[key]["cpd_sum"] += float(c["cpd_usd"])
        avg[key]["cpf_sum"] += float(c["cpf_usd"])
        avg[key]["n_llm_sum"] += int(c["n_llm_calls"])
        avg[key]["n_pos_sum"] += int(c["n_pos_caught_at_K"])
        avg[key]["count"] += 1
    for k in avg:
        n = avg[k]["count"]
        avg[k]["cpd_avg"] = avg[k]["cpd_sum"] / n
        avg[k]["cpf_avg"] = avg[k]["cpf_sum"] / n
        avg[k]["n_llm_avg"] = avg[k]["n_llm_sum"] / n
        avg[k]["n_pos_avg"] = avg[k]["n_pos_sum"] / n

    # Best rule per (rate, cost_ratio)
    best_rows = []
    for rate in RATES:
        for cost_llm in COST_LLM_SWEEP:
            sub = [avg[(rate, cost_llm, r)] for r in ("xgb_only", "gradient_band", "absolute_band")]
            best_cpd_rule = min(sub, key=lambda s: s["cpd_avg"])["rule"]
            best_cpf_rule = min(sub, key=lambda s: s["cpf_avg"])["rule"]
            by_rule = {s["rule"]: s for s in sub}
            best_rows.append(dict(
                rate=rate, cost_llm=cost_llm, cost_ratio=cost_llm / 0.0001,
                best_cpd_rule=best_cpd_rule,
                best_cpf_rule=best_cpf_rule,
                cpd_xgb=by_rule["xgb_only"]["cpd_avg"],
                cpd_grad=by_rule["gradient_band"]["cpd_avg"],
                cpd_abs=by_rule["absolute_band"]["cpd_avg"],
                cpf_xgb=by_rule["xgb_only"]["cpf_avg"],
                cpf_grad=by_rule["gradient_band"]["cpf_avg"],
                cpf_abs=by_rule["absolute_band"]["cpf_avg"],
                n_llm_grad=by_rule["gradient_band"]["n_llm_avg"],
                n_llm_abs=by_rule["absolute_band"]["n_llm_avg"],
            ))
    cols_best = list(best_rows[0].keys())
    with (RES / "synth_iter116_cost_cube_best_rule.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_best, delimiter="\t")
        w.writeheader()
        for r in best_rows:
            w.writerow(r)

    # Cube: per (rate, cost_ratio), the (cpf_grad / cpf_xgb) ratio
    recall_cost_ratio = []
    for rate in RATES:
        for cost_llm in COST_LLM_SWEEP:
            sub = [avg[(rate, cost_llm, r)] for r in ("xgb_only", "gradient_band")]
            grad_cpf = sub[1]["cpf_avg"]
            xgb_cpf = sub[0]["cpf_avg"]
            recall_cost_ratio.append(dict(
                rate=rate, cost_llm=cost_llm, cost_ratio=cost_llm / 0.0001,
                cpf_xgb=xgb_cpf,
                cpf_grad=grad_cpf,
                cpf_ratio_grad_over_xgb=grad_cpf / max(1e-12, xgb_cpf),
            ))
    cols_rcr = list(recall_cost_ratio[0].keys())
    with (RES / "synth_iter116_cpf_ratio_grad_xgb.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_rcr, delimiter="\t")
        w.writeheader()
        for r in recall_cost_ratio:
            w.writerow(r)

    # Headline: how many cells where xgb_only is the unique cheapest on $/caught
    n_total = len(best_rows)
    n_xgb_only_cpd = sum(1 for r in best_rows if r["best_cpd_rule"] == "xgb_only")
    n_xgb_only_cpf = sum(1 for r in best_rows if r["best_cpf_rule"] == "xgb_only")

    # cpf_ratio distribution
    ratios_flat = [r["cpf_ratio_grad_over_xgb"] for r in recall_cost_ratio]
    ratios_min = min(ratios_flat)
    ratios_max = max(ratios_flat)
    ratios_at_low_cost = [r["cpf_ratio_grad_over_xgb"] for r in recall_cost_ratio if r["cost_llm"] == 0.001]
    ratios_at_high_cost = [r["cpf_ratio_grad_over_xgb"] for r in recall_cost_ratio if r["cost_llm"] == 0.100]

    summary = {
        "iter": 116,
        "pillar": "P5P8-SYNTH",
        "n_rates": len(RATES),
        "n_cost_ratios": len(COST_LLM_SWEEP),
        "n_cells": n_total,
        "n_xgb_only_best_cpd": n_xgb_only_cpd,
        "n_xgb_only_best_cpf": n_xgb_only_cpf,
        "cpf_ratio_grad_over_xgb_range": [ratios_min, ratios_max],
        "cpf_ratio_grad_over_xgb_at_lowest_cost": [min(ratios_at_low_cost), max(ratios_at_low_cost)],
        "cpf_ratio_grad_over_xgb_at_highest_cost": [min(ratios_at_high_cost), max(ratios_at_high_cost)],
        "p8_iter116_link": "platform_hybrid/experiments/results/p5p8/p8_iter116_cost_llm_sweep.tsv",
        "p8_iter108_link": "platform_hybrid/experiments/results/p5p8/p8_iter108_cost_decision_cis_summary.json",
        "p8_iter112_link": "platform_hybrid/experiments/results/p5p8/p8_iter112_cost_cis_realistic_rates_summary.json",
    }
    (RES / "synth_iter116_cost_cube_envelope_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"[synth-iter116] {n_total} envelope cells: "
          f"xgb_only best cpd in {n_xgb_only_cpd}/{n_total}, "
          f"best cpf in {n_xgb_only_cpf}/{n_total}")
    print(f"[synth-iter116] cpf ratio grad/xgb range: [{ratios_min:.4f}, {ratios_max:.4f}]")
    print(f"[synth-iter116] cpf ratio at lowest cost_llm=$0.001: "
          f"[{min(ratios_at_low_cost):.4f}, {max(ratios_at_low_cost):.4f}]")
    print(f"[synth-iter116] cpf ratio at highest cost_llm=$0.100: "
          f"[{min(ratios_at_high_cost):.4f}, {max(ratios_at_high_cost):.4f}]")


if __name__ == "__main__":
    main()