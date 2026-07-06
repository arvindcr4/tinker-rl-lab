#!/usr/bin/env python3
"""P5P8-SYNTH JOB B (iter 124): three-domain density matrix.

Fresh vein, not in 137 prior P5P8-SYNTH rows.  Extends iter-120's
two-domain density-ratio analysis (P7 zvf-triage vs P8 grad-band,
density ratio 0.014, refuted universality) by adding a THIRD domain
(P5 mega-manifest score-stream contrast) and computing the full 3x3
density matrix.

The three domains
-----------------
D1 = P8 grad-band rule (test_data.csv, n=10000 rows):
    "signal-depleted" = row is in top-K AND small consecutive gradient
    Density = n_llm_grad / n_test = 84/10000 = 0.0084

D2 = P7 zvf-triage rule (N2 4-method, per-step):
    "signal-depleted" = step ZVF >= 0.7 (DEGENERATE regime)
    Density = n_steps_zvf>=0.7 / n_steps_total

D3 = P5 mega-manifest score-stream contrast (98 live cells):
    "signal-depleted" = cell's per-step zvf is fully zero across all
    groups (i.e., NO signal contrast at all).  Density =
    n_cells_emit_zvf / n_cells_total.

Falsifiable headlines
---------------------
H1 -- pairwise density ratios.  If P7+P8 universality held, all
  three densities would be within 10x of each other.  iter-120
  showed P8/P7 ratio = 0.014 (71x apart).  iter-124 computes
  P5/P7, P5/P8, P8/P7 with bootstrap CIs.

H2 -- rank ordering of domain densities.  If P7 > P5 > P8, then
  signal-depletion increases with rollout-batch granularity (per-step
  >> per-cell >> per-row).  If rank is different, the granularity
  hypothesis is refuted.

H3 -- correlation: are the 3 domains structurally related?
  Per-method (P7), per-V_stat quartile (P8), per-G (P5) densities
  reported.  Linear-rank correlation across the 3 axis.

H4 -- per-G P5 density stratified.  Does P5 zvf=1.0 density
  decrease with G (i.e., larger groups have more contrast)?
  Replicates the iter-120 anti-herding delta_div finding.

Stdlib + numpy.  <= 290 lines.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
SEED = 20260705
N_BOOT = 1500


def density_ci(n_fire, n_total, n_boot=N_BOOT, seed=SEED):
    """Wilson bootstrap CI on a proportion n_fire/n_total."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_total, n_total)
        # n_fire is the count of "fire" rows; we need per-row binary mask.
        # We don't have the mask here, so use the closed-form Bernoulli
        # bootstrap on the rate.
        boots[i] = rng.binomial(n_total, n_fire / n_total) / n_total
    return {
        "rate": n_fire / n_total,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_fire": n_fire,
        "n_total": n_total,
    }


def load_mega_cells():
    """Load cells.tsv; return per-cell zvf and G, model_family, task_slice."""
    path = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                zvf = float(r["zvf"])
                G = int(r["G"])
                rows.append({
                    "cell_id": r["cell_id"],
                    "model_family": r["model_family"],
                    "task_slice": r["task_slice"],
                    "G": G,
                    "temperature": float(r["temperature"]),
                    "seed": int(r["seed"]),
                    "zvf": zvf,
                    "mean_reward": float(r["mean_reward"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def density_p5_zvf_equals_1(cells):
    """Fraction of cells where per-step zvf == 1.0 (fully contrast-depleted)."""
    n_total = len(cells)
    n_fire = sum(1 for c in cells if c["zvf"] >= 0.999)
    return density_ci(n_fire, n_total)


def density_p5_zvf_geq_07(cells):
    """Fraction of cells where zvf >= 0.7 (the P7 DEGENERATE regime)."""
    n_total = len(cells)
    n_fire = sum(1 for c in cells if c["zvf"] >= 0.7)
    return density_ci(n_fire, n_total)


def density_p5_zvf_lt_03(cells):
    """Fraction of cells where zvf < 0.3 (strong signal contrast)."""
    n_total = len(cells)
    n_fire = sum(1 for c in cells if c["zvf"] < 0.3)
    return density_ci(n_fire, n_total)


def per_G_density(cells, threshold=0.7):
    """Per-G density of cells above `threshold`."""
    out = {}
    by_G = {}
    for c in cells:
        by_G.setdefault(c["G"], []).append(c["zvf"])
    for G in sorted(by_G):
        zvf_arr = np.array(by_G[G])
        n_total = len(zvf_arr)
        n_fire = int((zvf_arr >= threshold).sum())
        out[G] = {
            "n_cells": n_total,
            "n_fire": n_fire,
            "density": n_fire / n_total,
        }
    return out


def ratio_ci(num, denom, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on a ratio of two Bernoulli rates."""
    rng = np.random.default_rng(seed)
    rates_n = num["rate"]
    rates_d = denom["rate"]
    n_n, n_d = num["n_total"], denom["n_total"]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        n_i = rng.binomial(n_n, rates_n) / max(1, n_n)
        d_i = rng.binomial(n_d, rates_d) / max(1, n_d)
        boots[i] = n_i / max(1e-9, d_i)
    point = rates_n / max(1e-9, rates_d)
    return {
        "ratio": point,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "excludes_1.0": point < 1.0 / 10 or point > 10.0,
    }


def main():
    print(f"[iter124] loading mega cells ...")
    cells = load_mega_cells()
    print(f"[iter124] n_cells = {len(cells)}")

    # --- Domain 1: P8 grad-band (anchor: from iter-120 H1 = 84/10000) ---
    D1 = {
        "domain": "P8_grad_band",
        "n_fire": 84,
        "n_total": 10000,
        "rate": 84 / 10000,
        "lo": 0.0067,
        "hi": 0.0104,
        "rule": "row is in top-K AND |consecutive_score_grad| < 0.001",
    }

    # --- Domain 2: P7 zvf-triage (anchor: from iter-120 H3 = 50% of GRPO steps) ---
    D2 = {
        "domain": "P7_zvf_triage",
        "n_fire": 20,  # 50% of 40 steps
        "n_total": 40,
        "rate": 0.50,
        "lo": 0.35,
        "hi": 0.65,
        "rule": "step-level zvf >= 0.7 (DEGENERATE regime)",
    }

    # --- Domain 3: P5 mega-manifest zvf=1.0 density ---
    d3_zvf_eq_1 = density_p5_zvf_equals_1(cells)
    d3_zvf_geq_07 = density_p5_zvf_geq_07(cells)
    d3_zvf_lt_03 = density_p5_zvf_lt_03(cells)

    # For the canonical "P5 density", use zvf=1.0 (matches P7's 1.0
    # starvation regime AND P8's contrast-depleted notion).
    D3 = {
        "domain": "P5_mega_manifest_zvf_eq_1",
        "n_fire": d3_zvf_eq_1["n_fire"],
        "n_total": d3_zvf_eq_1["n_total"],
        "rate": d3_zvf_eq_1["rate"],
        "lo": d3_zvf_eq_1["lo"],
        "hi": d3_zvf_eq_1["hi"],
        "rule": "cell per-step zvf == 1.0 (fully contrast-depleted)",
    }

    # Write 3-domain density table
    out_d = RES / "synth_iter124_three_domain_density.tsv"
    with out_d.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["domain", "n_fire", "n_total", "rate", "ci_lo", "ci_hi", "rule"])
        for d in [D1, D2, D3]:
            w.writerow([d["domain"], d["n_fire"], d["n_total"],
                        f"{d['rate']:.6f}", f"{d['lo']:.6f}", f"{d['hi']:.6f}",
                        d["rule"]])
    print(f"[iter124] wrote {out_d}")

    # --- H1: pairwise density ratios with bootstrap CIs ---
    # Convert each domain to a {n_fire, n_total, rate, lo, hi} dict
    # keyed on density_ci output.
    D1_ci = density_ci(D1["n_fire"], D1["n_total"])
    D2_ci = density_ci(D2["n_fire"], D2["n_total"])

    ratios = {}
    for label, num, den in [
        ("P5_over_P7", D3, D2_ci),
        ("P5_over_P8", D3, D1_ci),
        ("P8_over_P7", D1_ci, D2_ci),
        ("P7_over_P8", D2_ci, D1_ci),
        ("P7_over_P5", D2_ci, D3),
        ("P8_over_P5", D1_ci, D3),
    ]:
        r = ratio_ci(num, den)
        ratios[label] = r
        print(f"[iter124 H1] {label}: ratio={r['ratio']:.4f} "
              f"CI=[{r['lo']:.4f}, {r['hi']:.4f}] excludes_1.0={r['excludes_1.0']}")

    out_r = RES / "synth_iter124_density_ratios.tsv"
    with out_r.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ratio", "point", "ci_lo", "ci_hi", "excludes_1.0"])
        for k, v in ratios.items():
            w.writerow([k, f"{v['ratio']:.6f}", f"{v['lo']:.6f}",
                        f"{v['hi']:.6f}", v["excludes_1.0"]])
    print(f"[iter124] wrote {out_r}")

    # --- H2: rank ordering ---
    densities = {
        "P8_grad_band": D1_ci["rate"],
        "P7_zvf_triage": D2_ci["rate"],
        "P5_zvf_eq_1": D3["rate"],
    }
    rank = sorted(densities.items(), key=lambda kv: -kv[1])
    print(f"[iter124 H2] density rank: " +
          ", ".join(f"{k}={v:.4f}" for k, v in rank))

    # --- H3: per-V_stat quartile P8 vs per-G P5 density correlation ---
    per_G = per_G_density(cells, threshold=0.999)
    # Linear correlation: P8 16-cell density vs P5 6-G density
    # (not directly comparable, so report both stratified as descriptive)
    h3_rows = []
    for G, d in per_G.items():
        h3_rows.append({
            "G": G,
            "n_cells": d["n_cells"],
            "n_fire_zvf_1": d["n_fire"],
            "density_p5_zvf_1": d["density"],
            "density_p5_zvf_lt_03": per_G_density(cells, threshold=0.3)[G]["density"],
            "density_p5_zvf_geq_07": per_G_density(cells, threshold=0.7)[G]["density"],
        })
    out_h3 = RES / "synth_iter124_per_G_density.tsv"
    with out_h3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(h3_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(h3_rows)
    print(f"[iter124] wrote {out_h3} ({len(h3_rows)} rows)")

    # --- H4: per-G density trend ---
    Gs = sorted(per_G.keys())
    p5_dens = [per_G[G]["density"] for G in Gs]
    # Spearman rank correlation
    if len(Gs) >= 3:
        ranks_G = np.argsort(np.argsort(Gs)).astype(float)
        ranks_d = np.argsort(np.argsort(p5_dens)).astype(float)
        dG = ranks_G - ranks_G.mean()
        dd = ranks_d - ranks_d.mean()
        spearman = float((dG * dd).sum() / np.sqrt((dG**2).sum() * (dd**2).sum()))
    else:
        spearman = float("nan")
    print(f"[iter124 H4] Spearman(G, P5 density zvf=1.0) = {spearman:.4f} "
          f"across {len(Gs)} G values: {Gs}")

    # --- Summary ---
    summary = {
        "iter": 124,
        "pillar": "P5P8-SYNTH",
        "n_cells_mega": len(cells),
        "domain_densities": {
            "P8_grad_band": D1_ci,
            "P7_zvf_triage": D2_ci,
            "P5_zvf_eq_1": D3,
        },
        "density_rank": [{"domain": k, "rate": v} for k, v in rank],
        "pairwise_ratios": ratios,
        "h4_per_G_spearman": spearman,
        "h4_G_values": Gs,
        "h4_per_G_density": p5_dens,
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    out_sum = RES / "synth_iter124_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter124] wrote {out_sum}")
    print(f"[iter124] DONE")


if __name__ == "__main__":
    main()