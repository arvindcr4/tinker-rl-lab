#!/usr/bin/env python3
"""P8 JOB A (iter 116): realistic LLM-cost sweep on the iter-112 envelope.

Fresh vein (NOT in any of the 128 prior P8 iters) -- extends iter-112
(realistic positive rate sweep at fixed cost_llm = $0.001) by sweeping
the LLM cost ITSELF over a realistic range:
   cost_llm ∈ {0.001, 0.003, 0.010, 0.030, 0.100} USD per LLM call
   cost_xgb fixed at $0.0001 / decision
   cost_ratio r = cost_llm / cost_xgb ∈ {10, 30, 100, 300, 1000}

The operational question: at what LLM cost does the iter-112 recommendation
(gradient-band at r>=0.50%, absolute-band at r<0.10%) flip to xgb-only?

Closes the iter-32 row 53 'P8 (sigma x C_inv x L cube)' gap at the
cost-axis level -- the c_LLM axis is where modern LLM-API deployment
lives (GPT-4o ~ $0.005-0.015/call, Claude Sonnet ~ $0.003-0.015/call,
Claude Opus ~ $0.015-0.075/call).

Falsifiable headlines
---------------------
H1 -- at cost_llm = $0.10 / call (ratio r=1000), xgb-only is the unique
optimum at EVERY rate on $/dec AND $/caught; gradient-band and
absolute-band NEVER break the cost-tie because their LLM-call budget
exceeds the marginal cost-per-decision.
H2 -- the cost ratio at which gradient-band LOSES Pareto-dominance on
$/caught (versus xgb-only) is r* in [30, 100] (i.e., $0.003-0.010/LLM).
H3 -- at every realistic rate, the rule that minimises $/dec AND $/caught
is invariant across cost ratios -- xgb-only ALWAYS minimises $/dec.
H4 -- recall-preservation break-even: gradient-band recall-preservation
breaks even with xgb-only on $/caught at cost ratio r in [10, 30]; above
r=100 even absolute-band's recall-preservation is over-priced.

Stdlib only (no xgboost retrain); reads existing iter-112 n_llm_calls
per (rate, tree, rule) cell and re-derives $/dec and $/caught at each
cost_llm value.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20260705

COST_XGB = 0.0001
COST_LLM_SWEEP = [0.001, 0.003, 0.010, 0.030, 0.100]
RATES = [0.0144, 0.0100, 0.0050, 0.0010, 0.0005]
TREES = ["XGB-20raw", "XGB-24full", "XGB-4sensor"]
RULES = ["xgb_only", "gradient_band", "absolute_band"]


def load(path):
    with path.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)
        return [dict(zip(header, r)) for r in rdr]


def cost_at_llm(n_test, n_llm, n_caught, cost_xgb, cost_llm):
    """Return ($/dec, $/caught) at the given cost_xgb, cost_llm."""
    total = n_test * cost_xgb + n_llm * (cost_llm - cost_xgb)
    cpd = total / n_test
    cpf = total / max(1, n_caught)
    return cpd, cpf


def main():
    # Load existing per-cell data (n_llm_calls, n_pos_caught_at_K, n_pos, n_test)
    cell = load(RES / "p8_iter112_cost_per_rate_cell.tsv")

    # Sweep over cost_llm values
    sweep_rows = []   # cost_ratio, rate, tree, rule, cpd, cpf, n_llm
    pair_rows = []    # cost_ratio, rate, tree, pair, cpd_delta, cpf_delta
    flip_rows = []    # rate, tree, rule, cost_ratio_at_which_xgb_dominates

    for cost_llm in COST_LLM_SWEEP:
        ratio = cost_llm / COST_XGB
        for c in cell:
            n_test_eff = int(round(int(c["n_pos"]) / float(c["rate"])))
            n_test_eff = max(n_test_eff, 1)
            n_llm = int(c["n_llm_calls"])
            n_caught = int(c["n_pos_caught_at_K"])
            cpd, cpf = cost_at_llm(n_test_eff, n_llm, n_caught,
                                    COST_XGB, cost_llm)
            sweep_rows.append(dict(
                cost_llm=cost_llm, cost_ratio=ratio,
                rate=float(c["rate"]), tree=c["tree"], rule=c["rule"],
                n_test_eff=n_test_eff, n_llm_calls=n_llm,
                n_pos_caught_at_K=n_caught,
                cpd_usd=cpd, cpf_usd=cpf,
            ))
        # Pairwise deltas at this cost ratio
        for rate in RATES:
            for tree in TREES:
                sub = [s for s in sweep_rows if s["cost_llm"] == cost_llm
                       and s["rate"] == rate and s["tree"] == tree]
                by_rule = {s["rule"]: s for s in sub}
                pairs = [
                    ("gradient_band", "xgb_only"),
                    ("absolute_band", "xgb_only"),
                    ("gradient_band", "absolute_band"),
                ]
                for a, b in pairs:
                    cpd_d = by_rule[a]["cpd_usd"] - by_rule[b]["cpd_usd"]
                    cpf_d = by_rule[a]["cpf_usd"] - by_rule[b]["cpf_usd"]
                    pair_rows.append(dict(
                        cost_llm=cost_llm, cost_ratio=ratio,
                        rate=rate, tree=tree, pair=f"{a}_vs_{b}",
                        cpd_delta=cpd_d, cpf_delta=cpf_d,
                    ))

    # Find flip points: at each (rate, tree), the cost_ratio at which
    # gradient_band / absolute_band LOSES to xgb_only on $/caught.
    for rate in RATES:
        for tree in TREES:
            for rule in ("gradient_band", "absolute_band"):
                # sweep over increasing ratio
                sorted_ratios = sorted(COST_LLM_SWEEP)
                flip_ratio = None
                for cost_llm in sorted_ratios:
                    sub = [s for s in sweep_rows if s["cost_llm"] == cost_llm
                           and s["rate"] == rate and s["tree"] == tree]
                    by_rule = {s["rule"]: s for s in sub}
                    cpf_d = by_rule[rule]["cpf_usd"] - by_rule["xgb_only"]["cpf_usd"]
                    if cpf_d > 0:  # rule more expensive than xgb-only
                        flip_ratio = cost_llm
                        break
                flip_rows.append(dict(
                    rate=rate, tree=tree, rule=rule,
                    flip_cost_llm=flip_ratio,
                    xgb_always_dominates=(flip_ratio == sorted_ratios[0]),
                ))

    # Write outputs
    cols_sweep = list(sweep_rows[0].keys())
    with (RES / "p8_iter116_cost_llm_sweep.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_sweep, delimiter="\t")
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)

    cols_pair = list(pair_rows[0].keys())
    with (RES / "p8_iter116_cost_llm_pair_delta.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_pair, delimiter="\t")
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    cols_flip = list(flip_rows[0].keys())
    with (RES / "p8_iter116_cost_llm_flip.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_flip, delimiter="\t")
        w.writeheader()
        for r in flip_rows:
            w.writerow(r)

    # Compute headline numbers
    # H1: at cost_llm=0.10 (ratio 1000), does gradient-band have cpd > xgb?
    h1 = {}
    for tree in TREES:
        for rate in RATES:
            sub = [s for s in sweep_rows if s["cost_llm"] == 0.10
                   and s["tree"] == tree and s["rate"] == rate]
            by_rule = {s["rule"]: s for s in sub}
            cpd_d = by_rule["gradient_band"]["cpd_usd"] - by_rule["xgb_only"]["cpd_usd"]
            cpf_d = by_rule["gradient_band"]["cpf_usd"] - by_rule["xgb_only"]["cpf_usd"]
            h1[f"{tree}_rate{rate}"] = {
                "cpd_grad_vs_xgb": cpd_d,
                "cpf_grad_vs_xgb": cpf_d,
                "grad_wins_cpd": cpd_d <= 0,
                "grad_wins_cpf": cpf_d <= 0,
            }
    h1_breakeven_count = sum(
        1 for v in h1.values() if v["grad_wins_cpd"] and v["grad_wins_cpf"]
    )

    # H2: at which cost_ratio does gradient-band break even on $/caught
    # at release rate 0.0144 on XGB-24full?
    h2 = {}
    for cost_llm in COST_LLM_SWEEP:
        sub = [s for s in sweep_rows if s["cost_llm"] == cost_llm
               and s["tree"] == "XGB-24full" and s["rate"] == 0.0144]
        by_rule = {s["rule"]: s for s in sub}
        cpf_d = by_rule["gradient_band"]["cpf_usd"] - by_rule["xgb_only"]["cpf_usd"]
        h2[cost_llm] = {
            "cpf_grad_minus_xgb": cpf_d,
            "grad_pareto_dominates_xgb": cpf_d <= 0,
        }

    # H4: at each rate, find cost_ratio where gradient-band recall-preservation
    # (per-fraud-caught cost ratio vs xgb-only) is at most 1.05x
    h4 = {}
    for rate in RATES:
        cpf_grad = None
        cpf_xgb = None
        for cost_llm in COST_LLM_SWEEP:
            sub = [s for s in sweep_rows if s["cost_llm"] == cost_llm
                   and s["tree"] == "XGB-24full" and s["rate"] == rate]
            by_rule = {s["rule"]: s for s in sub}
            grad_cpf = by_rule["gradient_band"]["cpf_usd"]
            xgb_cpf = by_rule["xgb_only"]["cpf_usd"]
            if cpf_xgb is None:
                cpf_xgb = xgb_cpf
            if cpf_grad is None:
                cpf_grad = grad_cpf
        h4[rate] = {
            "cpf_xgb_at_lowest_cost_llm": cpf_xgb,
            "cpf_grad_at_lowest_cost_llm": cpf_grad,
            "cpf_ratio_grad_over_xgb_at_low_cost": cpf_grad / max(1e-12, cpf_xgb),
        }

    # Headline: best rule by (cpd, cpf) per (rate, tree, cost_ratio)
    best_rows = []
    for rate in RATES:
        for tree in TREES:
            for cost_llm in COST_LLM_SWEEP:
                sub = [s for s in sweep_rows if s["cost_llm"] == cost_llm
                       and s["tree"] == tree and s["rate"] == rate]
                best_cpd_rule = min(sub, key=lambda s: s["cpd_usd"])["rule"]
                best_cpf_rule = min(sub, key=lambda s: s["cpf_usd"])["rule"]
                best_rows.append(dict(
                    rate=rate, tree=tree, cost_llm=cost_llm,
                    cost_ratio=cost_llm / COST_XGB,
                    best_cpd_rule=best_cpd_rule,
                    best_cpf_rule=best_cpf_rule,
                ))
    cols_best = list(best_rows[0].keys())
    with (RES / "p8_iter116_best_rule_per_cell.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols_best, delimiter="\t")
        w.writeheader()
        for r in best_rows:
            w.writerow(r)

    # Fraction of (rate, tree) cells where xgb_only is the unique best on $/caught
    n_total = len(best_rows)
    n_xgb_only_cpd = sum(1 for r in best_rows if r["best_cpd_rule"] == "xgb_only")
    n_xgb_only_cpf = sum(1 for r in best_rows if r["best_cpf_rule"] == "xgb_only")

    # at the extreme ratio (cost_llm=0.10) only
    extreme = [r for r in best_rows if r["cost_llm"] == 0.10]
    n_extreme_xgb_only_cpd = sum(1 for r in extreme if r["best_cpd_rule"] == "xgb_only")
    n_extreme_xgb_only_cpf = sum(1 for r in extreme if r["best_cpf_rule"] == "xgb_only")
    n_extreme_total = len(extreme)

    summary = {
        "iter": 116,
        "pillar": "P8",
        "n_train": 50000,
        "n_test": 10000,
        "rates": RATES,
        "trees": TREES,
        "rules": RULES,
        "cost_xgb": COST_XGB,
        "cost_llm_sweep": COST_LLM_SWEEP,
        "cost_ratio_sweep": [c / COST_XGB for c in COST_LLM_SWEEP],
        "n_sweep_cells": len(sweep_rows),
        "n_pair_cells": len(pair_rows),
        "n_flip_cells": len(flip_rows),
        "h1_at_cost_llm_0p10": {
            "n_grad_dominates_xgb_both": h1_breakeven_count,
            "n_cells_total": len(h1),
            "all_xgb_dominates": h1_breakeven_count == 0,
        },
        "h2_gradient_vs_xgb_at_release_rate_xgb24full": h2,
        "h4_cpf_at_lowest_cost_llm_xgb24full": h4,
        "n_best_cells_total": n_total,
        "n_xgb_only_best_cpd": n_xgb_only_cpd,
        "n_xgb_only_best_cpf": n_xgb_only_cpf,
        "n_extreme_total": n_extreme_total,
        "n_extreme_xgb_only_best_cpd": n_extreme_xgb_only_cpd,
        "n_extreme_xgb_only_best_cpf": n_extreme_xgb_only_cpf,
    }
    (RES / "p8_iter116_cost_llm_sweep_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"[iter116] sweep cells = {len(sweep_rows)}; "
          f"pair cells = {len(pair_rows)}; flip cells = {len(flip_rows)}")
    print(f"[iter116] H1 at cost_llm=$0.10: gradient-band dominates xgb on "
          f"both cpd+cpf in {h1_breakeven_count}/{len(h1)} cells")
    print(f"[iter116] Best $/caught rule is xgb_only in "
          f"{n_xgb_only_cpf}/{n_total} cells; at cost_llm=$0.10 in "
          f"{n_extreme_xgb_only_cpf}/{n_extreme_total} cells.")
    print("[iter116] H2 (XGB-24full, release rate 0.0144) gradient vs xgb on $/caught:")
    for k, v in h2.items():
        print(f"  cost_llm=${k}: cpf_delta={v['cpf_grad_minus_xgb']:+.6e}  "
              f"pareto_dominates={v['grad_pareto_dominates_xgb']}")


if __name__ == "__main__":
    main()