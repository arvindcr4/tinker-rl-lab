#!/usr/bin/env python3
"""P5P8-SYNTH (iter 112): cost-vs-rate Pareto envelope unifying
iter-12 PR-AUC at 5 rates + iter-108 cost pair-CIs + iter-112 cost
realistic-rate CI.  Fresh vein, not in 117 prior ledger rows.

Falsifiable headline H1 -- the same 5 positive rates that expose
the iter-12 PR-AUC gap also expose the iter-112 cost gap; the
operational envelope (Pareto frontier across (K, rule, rate)) is
uniquely determined by the (cost, recall) projection per (rule,
rate, tree) cell.

Falsifiable headline H2 -- the rule that **lowest $/dec AND lowest
$/caught** at every rate is **xgb-only**; the rule that closes the
gradient-band / absolute-band cost gap is **gradient-band at
release rate** but **absolute-band at low (<0.10%) rate**.

Reads existing tsv artifacts -- no model retraining.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"


def load(path):
    with path.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        return [dict(zip(header, r)) for r in rdr]


def main():
    cell = load(RES / "p8_iter112_cost_per_rate_cell.tsv")
    pair = load(RES / "p8_iter112_paired_bootstrap_ci.tsv")
    # Project: average across trees per (rate, rule)
    rates = sorted({float(r["rate"]) for r in cell})
    rule_set = ["xgb_only", "gradient_band", "absolute_band"]
    avg = {}
    for r in rates:
        for rule in rule_set:
            sub = [row for row in cell if float(row["rate"]) == r and row["rule"] == rule]
            if not sub:
                continue
            avg[(r, rule)] = {
                "rate": r,
                "rule": rule,
                "mean_cpd_usd": sum(float(s["cpd_usd"]) for s in sub) / len(sub),
                "mean_cpf_usd": sum(float(s["cpf_usd"]) for s in sub) / len(sub),
                "mean_n_llm": sum(int(s["n_llm_calls"]) for s in sub) / len(sub),
                "n_trees": len(sub),
            }
    # Pareto envelope: at each rate, the rule with the LOWEST cpf
    envelope = []
    for r in rates:
        per_rate = [avg[(r, rule)] for rule in rule_set if (r, rule) in avg]
        per_rate.sort(key=lambda x: x["mean_cpf_usd"])
        envelope.append(dict(
            rate=r,
            best_cpd=min(x["mean_cpd_usd"] for x in per_rate),
            best_cpd_rule=min(per_rate, key=lambda x: x["mean_cpd_usd"])["rule"],
            best_cpf=per_rate[0]["mean_cpf_usd"],
            best_cpf_rule=per_rate[0]["rule"],
        ))
    out_path = RES / "synth_iter112_cost_rate_envelope.tsv"
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(envelope[0].keys()), delimiter="\t")
        w.writeheader()
        for r in envelope:
            w.writerow(r)

    # All-rule projected table
    proj_rows = []
    for (r, rule), v in sorted(avg.items()):
        proj_rows.append(v)
    proj_path = RES / "synth_iter112_cost_rate_projection.tsv"
    with proj_path.open("w") as f:
        cols = list(proj_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in proj_rows:
            w.writerow(r)
    summary = {
        "iter": 112,
        "pillar": "P5P8-SYNTH",
        "n_rates": len(rates),
        "rates": rates,
        "rules": rule_set,
        "envelope": envelope,
        "n_pair_cells": len(pair),
    }
    (RES / "synth_iter112_cost_rate_envelope_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[synth-iter112] wrote {out_path.name} and {proj_path.name}")
    for env in envelope:
        print(f"  rate={env['rate']:.4f} best_cpd={env['best_cpd']:.7f} ({env['best_cpd_rule']}) "
              f"best_cpf={env['best_cpf']:.5f} ({env['best_cpf_rule']})")


if __name__ == "__main__":
    main()
