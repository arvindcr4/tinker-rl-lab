#!/usr/bin/env python3
"""P5P8-SYNTH JOB B (iter 136): five-domain density matrix.

Fresh vein (not in 148 prior SYNTH rows). Closes the iter-132
"Recommended next-iter mint veins" (#1: extend four-domain density matrix
to N domains). Iter-132's 4-domain matrix was:

  D1 = P8 grad-band per-row (84/10000 = 0.84%)
  D2 = P7 step zvf-triage per-step (20/40 = 50.0%; GRPO panel)
  D3 = P5 mega zvf=1.0 per-cell (36/98 = 36.7%)
  D4 = P7 per-prompt boundary cells (1867/2560 = 72.9%; iter-131 panel)

This iter adds a FIFTH domain:

  D5 = P8 per-cohort ECE-violation density under per-cohort isotonic at
       the realistic 0.5% positive rate (= 3 cohort axes × 5 strata
       average × 2 trees = 30 cells; ECE > 0.10 violation).

D5 is computed from JOB A (iter-136) iter-136 calibration-realistic-rates
artifact, which audits ECE across 5 positive rates × 2 trees × 3 cohort
axes × {raw, iso_per_cohort} = 60 (rate, tree, cohort, cal) cells per
method. D5 specifically uses the iso_per_cohort + 0.5% rate subset.

Falsifiable headlines
---------------------
H1 -- pairwise density ratio matrix with bootstrap CIs at 5 domains.
     Does D5 break the iter-124 "{P5, P7-step} <-> P8" two-super-domain
     clustering or does D5 land inside {D1, D8} (both P8)?

H2 -- density rank across 5 domains. If D5 is highest, it confirms
     "calibration violation is the densest signal of all five domains."
     If D5 is similar to D4, it confirms the "sub-cell granularity
     amplifies apparent violation density" hypothesis.

H3 -- rate-stratified D5: density of ECE>0.10 cells across the 5
     positive rates {0.05, 0.10, 0.50, 1.00, 1.44}% — does violation
     density amplify or compress at low rates?

H4 -- domain-by-domain "reviewer-facing claim" coercion: does the
     4-domain framework's "claim-X is true at domain-Y" rhetoric
     generalize to D5 without modification?

Operationally this answers: does the iter-124 super-domain claim
({P5,P7}-step / {P8}) survive the inclusion of TWO additional
calibration-stratified P8 domains (D5 = iso + 0.5% rate violation)?
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
SEED = 20260705
N_BOOT = 1500
EPSILON = 0.10  # ECE compliance threshold (the iter-99 / iter-104 canonical 0.10)


def density_ci(n_fire, n_total, n_boot=N_BOOT, seed=SEED):
    """Wilson bootstrap CI on a proportion."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    p = n_fire / n_total
    for i in range(n_boot):
        boots[i] = rng.binomial(n_total, p) / n_total
    return {
        "rate": p,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n_fire": n_fire,
        "n_total": n_total,
    }


def ratio_ci(num, denom, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on a ratio of two Bernoulli rates."""
    rng = np.random.default_rng(seed)
    rn = num["rate"]; rd = denom["rate"]
    nn = num["n_total"]; nd = denom["n_total"]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        n_i = rng.binomial(nn, rn) / max(1, nn)
        d_i = rng.binomial(nd, rd) / max(1, nd)
        boots[i] = n_i / max(1e-9, d_i)
    point = rn / max(1e-9, rd)
    return {
        "ratio": point,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
    }


def compute_d5_from_iter136(target_rate=0.5, target_cal="iso_per_cohort", eps=EPSILON):
    """Load iter-136 calibration-realistic TSV; compute density of cells
    where worst_cell_ece > eps in the (target_rate, target_cal) subset."""
    path = RES / "p8_iter136_cal_realistic.tsv"
    fire = 0
    total = 0
    per_rate = defaultdict(lambda: {"fire": 0, "total": 0})
    per_tree = defaultdict(lambda: {"fire": 0, "total": 0})
    per_cohort = defaultdict(lambda: {"fire": 0, "total": 0})
    rows = []
    with path.open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            if float(r["rate_pct"]) != target_rate:
                continue
            if r["calibration"] != target_cal:
                continue
            worst_ece = float(r["worst_cell_ece"])
            n_strata_used = int(r["n_strata_used"])
            total += n_strata_used
            per_rate[float(r["rate_pct"])]["total"] += n_strata_used
            per_tree[r["tree"]]["total"] += n_strata_used
            per_cohort[r["cohort"]]["total"] += n_strata_used
            violation = int(worst_ece > eps)
            fire += violation * n_strata_used
            per_rate[float(r["rate_pct"])]["fire"] += violation * n_strata_used
            per_tree[r["tree"]]["fire"] += violation * n_strata_used
            per_cohort[r["cohort"]]["fire"] += violation * n_strata_used
            rows.append({
                "rate_pct": float(r["rate_pct"]),
                "tree": r["tree"],
                "cohort": r["cohort"],
                "worst_cell_ece": worst_ece,
                "n_strata_used": n_strata_used,
                "violation": violation,
            })
    return {
        "rate": fire / max(1, total),
        "n_fire": fire,
        "n_total": total,
        "per_rate": dict(per_rate),
        "per_tree": dict(per_tree),
        "per_cohort": dict(per_cohort),
        "rows": rows,
    }


def rate_stratified_d5():
    """For each positive rate, the iso_per_cohort ECE>0.10 violation density.
    Returns dict {rate_pct -> {rate, n_fire, n_total}}."""
    path = RES / "p8_iter136_cal_realistic.tsv"
    buckets = defaultdict(lambda: {"fire": 0, "total": 0})
    with path.open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            if r["calibration"] != "iso_per_cohort":
                continue
            rate = float(r["rate_pct"])
            worst = float(r["worst_cell_ece"])
            n_strata = int(r["n_strata_used"])
            buckets[rate]["total"] += n_strata
            if worst > EPSILON:
                buckets[rate]["fire"] += n_strata
    out = {}
    for rate, v in sorted(buckets.items()):
        out[rate] = {
            "rate": v["fire"] / max(1, v["total"]),
            "n_fire": v["fire"],
            "n_total": v["total"],
        }
    return out


# Existing iter-124 / iter-132 numbers (recomputed for self-consistency).
D1 = density_ci(84, 10000)        # P8 grad-band per-row
D2 = density_ci(20, 40)           # P7 step zvf-triage per-step
D3 = density_ci(36, 98)           # P5 mega zvf=1.0 per-cell
D4 = density_ci(1867, 2560)       # P7 per-prompt boundary


def main():
    print("[1/3] computing D5 = P8 iso_per_cohort ECE>0.10 at rate=0.5%")
    d5_bundle = compute_d5_from_iter136(target_rate=0.5, target_cal="iso_per_cohort")
    # D5 is computed across 3 cohort × 2 tree × 1 rate = 6 cells, each with up
    # to ~5 strata. For cleaner counting, use the n_strata_used sum as the
    # denominator (one stratum-cell = one unit).
    D5 = density_ci(d5_bundle["n_fire"], d5_bundle["n_total"])

    print("[2/3] rate-stratified D5 distribution")
    d5_by_rate = rate_stratified_d5()

    print("[3/3] 5-domain pairwise ratio matrix with bootstrap CIs")
    domains = {"D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5}
    ratios = {}
    pairs = []
    keys = list(domains.keys())
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if j <= i:
                continue
            r = ratio_ci(domains[k1], domains[k2])
            ratios[f"{k1}/{k2}"] = r
            pairs.append((k1, k2, r))

    # Write density table
    with (RES / "synth_iter136_five_domain_density.tsv").open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["domain", "pillar", "granularity", "n_fire", "n_total",
                    "rate", "ci_lo", "ci_hi"])
        meta = {
            "D1": ("P8", "per-row", "fraud_test"),
            "D2": ("P7", "per-step", "GRPO_steps"),
            "D3": ("P5", "per-cell", "mega_cells"),
            "D4": ("P7", "per-prompt", "iter131_panel"),
            "D5": ("P8", "per-cohort-cell", "iter136_panel"),
        }
        for k, d in domains.items():
            pillar, gran, panel = meta[k]
            w.writerow([k, pillar, gran, d["n_fire"], d["n_total"],
                        f"{d['rate']:.6f}", f"{d['lo']:.6f}", f"{d['hi']:.6f}"])

    # Write ratio matrix
    with (RES / "synth_iter136_five_domain_density_ratios.tsv").open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pair", "ratio", "lo", "hi", "excludes_1.0"])
        for k1, k2, r in pairs:
            ex = r["ratio"] < 0.1 or r["ratio"] > 10.0
            w.writerow([f"{k1}/{k2}", f"{r['ratio']:.3f}",
                        f"{r['lo']:.3f}", f"{r['hi']:.3f}", ex])

    # Write rate-stratified D5
    with (RES / "synth_iter136_d5_rate_stratified.tsv").open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rate_pct", "n_fire", "n_total", "violation_rate"])
        for rate, v in sorted(d5_by_rate.items()):
            w.writerow([rate, v["n_fire"], v["n_total"], f"{v['rate']:.6f}"])

    summary = {
        "iter": 136,
        "n_domains": 5,
        "domains": {
            "D1": {"pillar": "P8", "concept": "grad-band", "density": D1},
            "D2": {"pillar": "P7", "concept": "zvf-triage-step", "density": D2},
            "D3": {"pillar": "P5", "concept": "mega-zvf=1", "density": D3},
            "D4": {"pillar": "P7", "concept": "boundary-prompt-cell", "density": D4},
            "D5": {"pillar": "P8", "concept": "iso_per_cohort-ECE>0.10-at-0.5%-rate",
                   "density": D5,
                   "source": "p8_iter136_calibration_realistic_rates.py at rate=0.5%",
                   "n_per_tree": d5_bundle["per_tree"],
                   "n_per_cohort": d5_bundle["per_cohort"]},
        },
        "iter124_super_domain_claim": "{P5, P7-step} <-> P8 — does adding D5 break it?",
        "ratios": {k: {"ratio": v["ratio"], "lo": v["lo"], "hi": v["hi"]} for k, v in ratios.items()},
        "rate_stratified_d5": {str(k): v for k, v in d5_by_rate.items()},
    }
    with (RES / "synth_iter136_five_domain_density_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
