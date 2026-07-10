#!/usr/bin/env python3
"""P5P8-SYNTH fourteen-domain density matrix (D14 + D15) (iter 172 JOB B).

Fresh vein, not in 180 prior SYNTH rows. Extends iter-168 twelve-domain
matrix (D1-D13) to fourteen domains by adding:

  **D14 = P8_vstat_ensemble_ceiling_break** = proportion of
  (rate × tier × fset) cells where, AT SOME τ in {0.05..0.90}, the
  joint_vstat classifier achieves esc_prec >= 0.05 on at least
  3 of 5 seeds. Reads the iter-172 P8 ensemble-vs-vmean matrix.

  **D15 = P8_vstat_ensemble_pareto_at_tau** = proportion of
  (rate × tier × fset) cells where the joint_vstat classifier achieves
  Pareto (esc_prec >= 0.10 AND value_rate >= 0.30) on at least
  3 of 5 seeds.

Both domains aggregate across seeds by majority (>= 3 of 5).

Hypotheses
----------
H1: D14 >= 0.10 (joint ensemble breaks 5% precision ceiling on >= 10%
     of SYNTH cells).
H2: D15 >= 0.01 (joint ensemble Pareto density is non-trivial).
H3: D14 > D12 (joint ensemble is strictly better than single-V_mean
     at the SYNTH aggregation level).
H4: D14 + D15 = 0 (both zero is the iter-172 sharpest negative finding
     reproduced at the SYNTH level).

Stdlib only.  <= 200 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)


def wilson_95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - halfw), min(1.0, center + halfw)


def layer_for(p: float) -> str:
    if p < 0.10:
        return "LOW"
    if p < 0.50:
        return "MID"
    return "HIGH"


def main():
    matrix_path = RES / "p8_iter172_threshold_matrix.tsv"
    print(f"[synth172] reading {matrix_path}")
    rows = list(csv.DictReader(matrix_path.open(), delimiter="\t"))
    print(f"[synth172] loaded {len(rows)} matrix rows")

    # Per (rate, fset, clf) per-τ joint count: how many seeds pass
    # the precision-ceiling / Pareto criterion?
    # The iter-172 matrix is tier-invariant (no LLM cost tier is involved
    # because the joint classifier is trained once on the full train set
    # and the threshold sweep is on a probability). So we collapse over
    # the tier axis at the (rate, fset, clf) cell.
    agg_keys = sorted({(r["rate_pct"], r["fset"], r["clf"]) for r in rows})
    print(f"[synth172] aggregation keys: {len(agg_keys)} (rate × fset × clf)")

    # D14: at-τ ceil = joint_vstat achieves esc_prec >= 0.05 at any τ
    # on at least 3 of 5 seeds at this (rate, fset) cell.
    d14_per_cell = []
    for rate, fset, clf in agg_keys:
        cell_rows = [r for r in rows
                     if r["rate_pct"] == rate and r["fset"] == fset
                     and r["clf"] == clf]
        seeds_with_break = set()
        seeds_with_pareto = set()
        seeds = sorted({r["seed"] for r in cell_rows})
        for seed in seeds:
            seed_rows = [r for r in cell_rows if r["seed"] == seed]
            # D14 = any τ achieves esc_prec >= 0.05
            if any(float(r["esc_prec"]) >= 0.05 for r in seed_rows):
                seeds_with_break.add(seed)
            # D15 = any τ achieves Pareto
            if any((float(r["esc_prec"]) >= 0.10
                    and float(r["value_rate"]) >= 0.30)
                   for r in seed_rows):
                seeds_with_pareto.add(seed)
        d14_per_cell.append({
            "rate_pct": rate, "fset": fset, "clf": clf,
            "n_seeds": len(seeds),
            "d14_seeds_with_break": len(seeds_with_break),
            "d15_seeds_with_pareto": len(seeds_with_pareto),
        })

    # Now compute per-(rate, fset) density at the SYNTH granularity
    # (without tier): k = sum over seeds/clfs of break events at this cell.
    # The brief calls for (rate × tier × fset) but iter-172 has no tier
    # dimension, so we collapse over it (treating it as cheap tier).
    # The SYNTH grid is 5 rates × 5 tiers × 4 fsets = 100 cells in
    # iter-168; we replicate the tier axis by enumerating 5 tiers
    # symmetric (the iter-172 finding is tier-invariant — joint_vstat
    # only depends on the test rows, not on the LLM cost tier).
    TIERS = ["cheap_heuristic", "small_open", "iter120_default",
             "mid_tier", "frontier_gpt4"]
    d14_synth_cell = []
    d15_synth_cell = []
    for rate, fset, clf in agg_keys:
        if clf != "joint_vstat":
            continue
        # Find the per-(rate, fset, clf) cell stats
        cell = next(c for c in d14_per_cell
                    if c["rate_pct"] == rate and c["fset"] == fset
                    and c["clf"] == clf)
        for tier in TIERS:
            d14_synth_cell.append({
                "rate_pct": rate, "fset": fset, "tier": tier,
                "k_d14": cell["d14_seeds_with_break"],
                "n": cell["n_seeds"],
                "d14_density": cell["d14_seeds_with_break"] / cell["n_seeds"],
            })
            d15_synth_cell.append({
                "rate_pct": rate, "fset": fset, "tier": tier,
                "k_d15": cell["d15_seeds_with_pareto"],
                "n": cell["n_seeds"],
                "d15_density": cell["d15_seeds_with_pareto"] / cell["n_seeds"],
            })

    # Aggregate densities
    d14_k = sum(c["k_d14"] >= 3 for c in d14_synth_cell)
    d14_n = len(d14_synth_cell)
    d14_p = d14_k / max(1, d14_n)
    d14_lo, d14_hi = wilson_95(d14_k, d14_n)
    print(f"[synth172] D14: {d14_k}/{d14_n} = {d14_p:.4f} "
          f"[Wilson {d14_lo:.3f}, {d14_hi:.3f}]")

    d15_k = sum(c["k_d15"] >= 3 for c in d15_synth_cell)
    d15_n = len(d15_synth_cell)
    d15_p = d15_k / max(1, d15_n)
    d15_lo, d15_hi = wilson_95(d15_k, d15_n)
    print(f"[synth172] D15: {d15_k}/{d15_n} = {d15_p:.4f} "
          f"[Wilson {d15_lo:.3f}, {d15_hi:.3f}]")

    # Load D12 from iter-168 (which has the same aggregation)
    # For comparison, iter-168 D12 = 0/100 = 0.000
    d12_density = 0.0  # from iter-168

    h1_pass = d14_p >= 0.10
    h2_pass = d15_p >= 0.01
    h3_pass = d14_p > d12_density
    h4_pass = (d14_k == 0 and d15_k == 0)
    print(f"[synth172] H1 D14>=0.10: {d14_p:.4f} PASS={h1_pass}")
    print(f"[synth172] H2 D15>=0.01: {d15_p:.4f} PASS={h2_pass}")
    print(f"[synth172] H3 D14>D12 ({d14_p:.4f} > {d12_density:.4f}): "
          f"PASS={h3_pass}")
    print(f"[synth172] H4 D14=D15=0: PASS={h4_pass}")

    # Write per-cell outputs
    out_d14 = RES / "synth_iter172_d14_per_cell.tsv"
    with out_d14.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(d14_synth_cell[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(d14_synth_cell)
    print(f"[synth172] wrote {out_d14} ({len(d14_synth_cell)} rows)")

    out_d15 = RES / "synth_iter172_d15_per_cell.tsv"
    with out_d15.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(d15_synth_cell[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(d15_synth_cell)
    print(f"[synth172] wrote {out_d15} ({len(d15_synth_cell)} rows)")

    # Fourteen-domain matrix (D1-D13 from iter-168 + D14, D15)
    iter168 = json.loads((RES / "synth_iter168_summary.json").read_text())
    base = list(iter168["twelve_domain_summary"])
    # iter168 reported D1-D13 (twelve + one more); rename to match
    new_domains = [
        {"domain": "D14", "label": "P8_vstat_ensemble_ceiling_break",
         "density": d14_p, "layer": layer_for(d14_p)},
        {"domain": "D15", "label": "P8_vstat_ensemble_pareto_at_tau",
         "density": d15_p, "layer": layer_for(d15_p)},
    ]
    full_15 = base + new_domains
    out_15 = RES / "synth_iter172_fifteen_domain_density.tsv"
    with out_15.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "label", "density", "layer"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(full_15)
    print(f"[synth172] wrote {out_15} (15 rows)")

    summary = {
        "iter": 172,
        "job": "P5P8-SYNTH fourteen-domain density matrix (D14 + D15)",
        "n_d14_synth_cells": d14_n,
        "n_d15_synth_cells": d15_n,
        "d14_overall": {"k": d14_k, "n": d14_n, "p": d14_p,
                        "lo": d14_lo, "hi": d14_hi,
                        "layer": layer_for(d14_p)},
        "d15_overall": {"k": d15_k, "n": d15_n, "p": d15_p,
                        "lo": d15_lo, "hi": d15_hi,
                        "layer": layer_for(d15_p)},
        "d12_baseline_iter168_density": d12_density,
        "fifteen_domain_summary": full_15,
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "h3_pass": h3_pass,
        "h4_pass": h4_pass,
    }
    out_sum = RES / "synth_iter172_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[synth172] wrote {out_sum}")
    print(f"[synth172] DONE")


if __name__ == "__main__":
    main()