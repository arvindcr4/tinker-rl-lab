#!/usr/bin/env python3
"""P5P8-SYNTH twelve-domain density matrix (iter 168 JOB B).

Fresh vein, not in 173 prior SYNTH rows. Extends iter-156 eleven-domain
matrix (D1-D11) to twelve domains by adding:

  D12 = P8_achievable_precision_frontier = proportion of
        (rate x tier x fset) cells where, AT SOME TAU in {0.5, 1.0,
        1.5, 2.0}, esc_prec >= 0.10 AND value_rate >= 0.30.
        D12 answers the operationally critical question: "Across the
        threshold sweep, what fraction of operational cells are
        actually reachable by the Pareto frontier?"

  D13 = P8_threshold_sweep_rescue = proportion of (rate x tier x fset)
        cells where tau=2.0 improves esc_prec by >= 5x over tau=0.0
        (the iter-156 baseline). D13 answers: "Does the threshold
        sweep materially rescue the iter-156 precision problem?"

Stdlib only.  <= 250 lines.  Reads iter-168 threshold matrix from JOB A.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)

SEEDS = [20260706, 20260708, 20260710, 20260712, 20260714]
RATES_PCT = [1.44, 1.00, 0.50, 0.10, 0.05]
TIERS = ["cheap_heuristic", "small_open", "iter120_default", "mid_tier", "frontier_gpt4"]
FSETS = ["24full", "20raw", "20raw+minmax", "20raw+stat"]
TAUS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PARETO_TAUS = [0.5, 1.0, 1.5, 2.0]  # strict thresholds considered for D12
PREC_FLOOR = 0.10
VR_FLOOR = 0.30
RESCUE_RATIO = 5.0

# Wilson 95% CI for binomial proportion (k / n)
def wilson95(k, n):
    if n == 0:
        return 0.0, 0.0, 1.0
    z = 1.96
    p = k / n
    denom = 1.0 + z*z / n
    center = (p + z*z / (2*n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z*z / (4*n*n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_matrix(path):
    """Load the iter-168 threshold matrix as list of dicts."""
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for line in rdr:
            rows.append({
                "seed": int(line["seed"]),
                "rate_pct": float(line["rate_pct"]),
                "fset": line["fset"],
                "tau_vmean": float(line["tau_vmean"]),
                "tier": line["tier"],
                "value_rate": float(line["value_rate"]),
                "esc_prec": float(line["esc_prec"]),
                "esc_cost_per_lift": float(line["esc_cost_per_lift"]),
                "n_lift": int(line["n_lift"]),
                "n_waste": int(line["n_waste"]),
                "breakeven": line["breakeven"] == "True",
            })
    return rows


def main():
    in_matrix = RES / "p8_iter168_threshold_matrix.tsv"
    if not in_matrix.exists():
        raise SystemExit(f"missing input matrix: {in_matrix} -- run JOB A first")
    print(f"[iter168-SYNTH] loading {in_matrix}")
    rows = load_matrix(in_matrix)
    print(f"[iter168-SYNTH] loaded {len(rows)} rows")

    # ----------------------------------------------------------------
    # D12: P8_achievable_precision_frontier
    # Per (rate x tier x fset) cell (100 cells): does ANY tau in
    # PARETO_TAUS achieve esc_prec >= 0.10 AND value_rate >= 0.30?
    # We aggregate across seeds by majority: cell is "achievable" if
    # >= 3 of 5 seeds find at least one tau.
    # ----------------------------------------------------------------
    d12_cells = []  # (rate, tier, fset) -> k_pareto_seeds (count out of 5)
    for rate_pct in RATES_PCT:
        for tier_name in TIERS:
            for fset_name in FSETS:
                k_pareto = 0
                best_per_seed = []
                for seed in SEEDS:
                    seed_rows = [r for r in rows
                                 if r["seed"] == seed
                                 and r["rate_pct"] == rate_pct
                                 and r["fset"] == fset_name
                                 and r["tier"] == tier_name
                                 and r["tau_vmean"] in PARETO_TAUS]
                    # Cell is Pareto-achievable on this seed if ANY tau hits
                    hits = [r for r in seed_rows
                            if r["esc_prec"] >= PREC_FLOOR
                            and r["value_rate"] >= VR_FLOOR]
                    best_prec = max((r["esc_prec"] for r in seed_rows), default=0.0)
                    best_vr = max((r["value_rate"] for r in seed_rows), default=0.0)
                    best_per_seed.append({
                        "seed": seed, "pareto_hit": len(hits) >= 1,
                        "best_prec": best_prec, "best_vr": best_vr,
                    })
                    if hits:
                        k_pareto += 1
                d12_cells.append({
                    "rate_pct": rate_pct, "tier": tier_name, "fset": fset_name,
                    "k_pareto_seeds": k_pareto,
                    "n_seeds": len(SEEDS),
                    "achievable": k_pareto >= 3,  # majority rule
                })

    # Aggregate density
    n_total = len(d12_cells)
    k_achievable = sum(1 for c in d12_cells if c["achievable"])
    d12_p, d12_lo, d12_hi = wilson95(k_achievable, n_total)
    print(f"[iter168-SYNTH] D12 overall: {k_achievable}/{n_total} = {d12_p:.4f} "
          f"[Wilson {d12_lo:.4f}, {d12_hi:.4f}]")

    # Per-tier
    d12_per_tier = {}
    for tier_name in TIERS:
        cells = [c for c in d12_cells if c["tier"] == tier_name]
        k = sum(1 for c in cells if c["achievable"])
        p, lo, hi = wilson95(k, len(cells))
        d12_per_tier[tier_name] = {"k": k, "n": len(cells), "p": p, "lo": lo, "hi": hi}
        print(f"[iter168-SYNTH] D12@{tier_name}: {k}/{len(cells)} = {p:.4f} "
              f"[Wilson {lo:.4f}, {hi:.4f}]")

    # Per-rate at cheap tier
    d12_per_rate = {}
    for rate_pct in RATES_PCT:
        cells = [c for c in d12_cells if c["tier"] == "cheap_heuristic"
                 and c["rate_pct"] == rate_pct]
        k = sum(1 for c in cells if c["achievable"])
        p, lo, hi = wilson95(k, len(cells))
        d12_per_rate[rate_pct] = {"k": k, "n": len(cells), "p": p, "lo": lo, "hi": hi}

    # Per-fset at cheap tier
    d12_per_fset = {}
    for fset_name in FSETS:
        cells = [c for c in d12_cells if c["tier"] == "cheap_heuristic"
                 and c["fset"] == fset_name]
        k = sum(1 for c in cells if c["achievable"])
        p, lo, hi = wilson95(k, len(cells))
        d12_per_fset[fset_name] = {"k": k, "n": len(cells), "p": p, "lo": lo, "hi": hi}

    # ----------------------------------------------------------------
    # D13: P8_threshold_sweep_rescue
    # Per (rate x tier x fset) cell: at tau=2.0 vs tau=0.0, does
    # esc_prec improve by >= 5x? Aggregate across seeds.
    # ----------------------------------------------------------------
    d13_cells = []
    for rate_pct in RATES_PCT:
        for tier_name in TIERS:
            for fset_name in FSETS:
                k_rescue = 0
                for seed in SEEDS:
                    tau0 = next((r for r in rows
                                 if r["seed"] == seed
                                 and r["rate_pct"] == rate_pct
                                 and r["fset"] == fset_name
                                 and r["tier"] == tier_name
                                 and r["tau_vmean"] == 0.0), None)
                    tau2 = next((r for r in rows
                                 if r["seed"] == seed
                                 and r["rate_pct"] == rate_pct
                                 and r["fset"] == fset_name
                                 and r["tier"] == tier_name
                                 and r["tau_vmean"] == 2.0), None)
                    if tau0 is None or tau2 is None:
                        continue
                    base = tau0["esc_prec"]
                    swept = tau2["esc_prec"]
                    # Rescue iff tau2 improves esc_prec >= 5x over tau0
                    # and tau2 is at least 0.05 (not just relative gain on zero)
                    ratio = (swept + 1e-9) / max(1e-9, base + 1e-9)
                    rescued = (ratio >= RESCUE_RATIO) and (swept >= 0.05)
                    if rescued:
                        k_rescue += 1
                d13_cells.append({
                    "rate_pct": rate_pct, "tier": tier_name, "fset": fset_name,
                    "k_rescue_seeds": k_rescue,
                    "n_seeds": len(SEEDS),
                    "rescued": k_rescue >= 3,
                })

    n_total13 = len(d13_cells)
    k_rescued = sum(1 for c in d13_cells if c["rescued"])
    d13_p, d13_lo, d13_hi = wilson95(k_rescued, n_total13)
    print(f"[iter168-SYNTH] D13 overall: {k_rescued}/{n_total13} = {d13_p:.4f} "
          f"[Wilson {d13_lo:.4f}, {d13_hi:.4f}]")

    # Per-tier
    d13_per_tier = {}
    for tier_name in TIERS:
        cells = [c for c in d13_cells if c["tier"] == tier_name]
        k = sum(1 for c in cells if c["rescued"])
        p, lo, hi = wilson95(k, len(cells))
        d13_per_tier[tier_name] = {"k": k, "n": len(cells), "p": p, "lo": lo, "hi": hi}
        print(f"[iter168-SYNTH] D13@{tier_name}: {k}/{len(cells)} = {p:.4f}")

    # ----------------------------------------------------------------
    # Twelve-domain density matrix (D1-D13)
    # Carry forward iter-156 D1-D11; add D12 and D13.
    # ----------------------------------------------------------------
    iter156_summary = RES / "synth_iter156_summary.json"
    if iter156_summary.exists():
        with iter156_summary.open() as f:
            base_summary = json.load(f)
        base_11 = base_summary.get("eleven_domain_summary", [])
    else:
        base_11 = []

    # Layer assignment by density (consistent with iter-156):
    # HIGH >= 0.70, MID 0.10-0.70, LOW < 0.10
    def layer_for(p):
        if p >= 0.70:
            return "HIGH"
        if p >= 0.10:
            return "MID"
        return "LOW"

    twelve = list(base_11)
    twelve.append({
        "domain": "D12",
        "label": "P8_achievable_precision_frontier",
        "density": d12_p,
        "layer": layer_for(d12_p),
    })
    twelve.append({
        "domain": "D13",
        "label": "P8_threshold_sweep_rescue",
        "density": d13_p,
        "layer": layer_for(d13_p),
    })

    # H1 PASS: D12 >= 0.10 (some achievable frontier exists)
    h1_pass = d12_p >= 0.10
    # H2 PASS: D12 monotone non-decreasing in tier affordability
    h2_pass = d12_per_tier["cheap_heuristic"]["p"] >= d12_per_tier["frontier_gpt4"]["p"]
    # H3 PASS: D13 >= 0.20 (rescue happens on at least 20% of cells)
    h3_pass = d13_p >= 0.20
    # H4 PASS: D13@cheap >= D13@frontier (cheap tier more rescuable)
    h4_pass = d13_per_tier["cheap_heuristic"]["p"] >= d13_per_tier["frontier_gpt4"]["p"]

    print(f"[iter168-SYNTH] H1 (D12>=0.10): PASS={h1_pass} ({d12_p:.4f})")
    print(f"[iter168-SYNTH] H2 (D12 cheap >= frontier): PASS={h2_pass}")
    print(f"[iter168-SYNTH] H3 (D13>=0.20): PASS={h3_pass} ({d13_p:.4f})")
    print(f"[iter168-SYNTH] H4 (D13 cheap >= frontier): PASS={h4_pass}")

    # ----------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------
    out_main = RES / "synth_iter168_twelve_domain_density.tsv"
    with out_main.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["domain", "label", "density", "layer"])
        for d in twelve:
            w.writerow([d["domain"], d["label"],
                        f"{d['density']:.4f}", d["layer"]])
    print(f"[iter168-SYNTH] wrote {out_main}")

    out_d12 = RES / "synth_iter168_d12_per_cell.tsv"
    with out_d12.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rate_pct", "tier", "fset", "k_pareto_seeds",
                    "n_seeds", "achievable"])
        for c in d12_cells:
            w.writerow([c["rate_pct"], c["tier"], c["fset"],
                        c["k_pareto_seeds"], c["n_seeds"],
                        int(c["achievable"])])
    print(f"[iter168-SYNTH] wrote {out_d12} ({len(d12_cells)} rows)")

    out_d13 = RES / "synth_iter168_d13_per_cell.tsv"
    with out_d13.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rate_pct", "tier", "fset", "k_rescue_seeds",
                    "n_seeds", "rescued"])
        for c in d13_cells:
            w.writerow([c["rate_pct"], c["tier"], c["fset"],
                        c["k_rescue_seeds"], c["n_seeds"],
                        int(c["rescued"])])
    print(f"[iter168-SYNTH] wrote {out_d13} ({len(d13_cells)} rows)")

    summary = {
        "iter": 168,
        "job": "P5P8-SYNTH twelve-domain density matrix (D12 + D13)",
        "n_d12_cells": n_total,
        "n_d13_cells": n_total13,
        "d12_overall": {"k": k_achievable, "n": n_total,
                        "p": d12_p, "lo": d12_lo, "hi": d12_hi,
                        "layer": layer_for(d12_p)},
        "d12_per_tier": d12_per_tier,
        "d12_per_rate": d12_per_rate,
        "d12_per_fset": d12_per_fset,
        "d13_overall": {"k": k_rescued, "n": n_total13,
                        "p": d13_p, "lo": d13_lo, "hi": d13_hi,
                        "layer": layer_for(d13_p)},
        "d13_per_tier": d13_per_tier,
        "twelve_domain_summary": twelve,
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "h3_pass": h3_pass,
        "h4_pass": h4_pass,
    }
    out_sum = RES / "synth_iter168_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter168-SYNTH] wrote {out_sum}")
    print(f"[iter168-SYNTH] DONE")


if __name__ == "__main__":
    main()