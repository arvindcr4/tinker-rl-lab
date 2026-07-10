#!/usr/bin/env python3
"""P7 JOB B (iter 36): replay Hybrid controller on the 98-cell mega corpus.

Closes the iter-31 falsifiable prediction:
    "Hybrid C3 strictly dominates zvf-triage C1 on cells whose
     per-step ZVF trajectory reaches zvf >= tau+delta in any step;
     on cells with no saturation-band step, C3 collapses to C1."

Inputs
------
experiments/results/mega_20260704/cells.tsv  (98 cells, each has
    reward_vectors_json -> list of (step, group, rollout) rewards).

Outputs
-------
experiments/results/p5p8/p7_mega_saturation_band_per_cell.tsv
experiments/results/p5p8/p7_mega_saturation_band_summary.json
docs/p5p8_improvements/<NN>_p7_mega_saturation_band.md

Stdlib + numpy + pandas + matplotlib. <=300 lines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MEGA = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
BOOT_SEED = 20260704

# Controller params (mirror iter-31 Hybrid C3): tau = 0.70, delta = 0.20
TAU = 0.70
DELTA = 0.20
G_BASE = 8          # baseline group size; controller chooses G = 8 or 16
G_ESCAL = 16
G_DESCAL = 4


def zvf_of(rollouts):
    """Fraction of groups with k=0 or k=G (degenerate contrast).

    `rollouts` is shape (n_groups, G_per_group). For mega cells G_per_group=1
    (each cell stores (step, group) -> scalar reward), so we reshape to
    (n_groups * G_per_group, ) and count degenerate groups of size G_per_group.
    """
    arr = np.atleast_2d(rollouts)
    if arr.ndim != 2:
        arr = arr.reshape(1, -1)
    n_groups, g = arr.shape
    if g == 0:
        return 1.0
    k = arr.sum(axis=1)
    return float(((k == 0) | (k == g)).mean())


def per_step_zvf(cell_rv):
    """Return array of length (n_steps,) of per-step ZVF."""
    steps = []
    for step_rewards in cell_rv:
        arr = np.asarray(step_rewards, dtype=float)
        steps.append(zvf_of(arr))
    return np.asarray(steps)


def replay_controllers(per_step_z, g_per_step, tau=TAU, delta=DELTA):
    """Replay C1 (zvf-triage), C2 (Dualformer-Auto), C3 (Hybrid) on the
    per-step trajectory. Returns dict {C1, C2, C3} -> total rollouts."""
    n_steps = len(per_step_z)
    c1 = c2 = c3 = 0
    for s in range(n_steps):
        z = per_step_z[s]
        g_base = g_per_step[s]
        # C1 (zvf-triage): if z >= tau AND pcd <= delta, escalate G -> 2*G_base
        if z >= tau:
            g1 = max(2 * g_base, g_base * 2)
        else:
            g1 = g_base
        c1 += g1
        # C2 (Dualformer-Auto): if zvf in saturation band (>=0.9), descalate
        if z >= 0.9:
            g2 = max(g_base // 2, 2)
        elif z <= 0.1:
            g2 = max(2 * g_base, g_base * 2)
        else:
            g2 = g_base
        c2 += g2
        # C3 (Hybrid): C1 if z >= tau, C2 if z >= tau+delta (descalate the
        # boundary cases C1 would otherwise escalate)
        if z >= tau + delta:
            g3 = max(g_base // 2, 2)  # desaturation
        elif z >= tau:
            g3 = max(2 * g_base, g_base * 2)  # escalate
        else:
            g3 = g_base
        c3 += g3
    return {"C1": c1, "C2": c2, "C3": c3}


def has_sat_step(per_step_z, tau_plus_delta):
    """Does the trajectory have any step with z >= tau+delta?"""
    return bool((per_step_z >= tau_plus_delta).any())


def main():
    print("Loading mega_20260704/cells.tsv")
    df = pd.read_csv(MEGA, sep="\t")
    n_cells = len(df)
    print(f"  n_cells={n_cells}")

    per_cell = []
    sat_band_count = 0
    for i, row in df.iterrows():
        rv = json.loads(row["reward_vectors_json"])
        # Per-cell G is constant (set at config time); the trajectory is
        # (n_steps, G_per_step).
        arr = np.asarray(rv, dtype=float)
        n_steps, g_const = arr.shape
        g_per_step = np.full(n_steps, g_const)
        per_step_z = per_step_zvf(rv)
        # Baseline (always-G=const): total rollouts
        baseline = int(g_const * n_steps)
        result = replay_controllers(per_step_z, g_per_step)
        sat_flag = has_sat_step(per_step_z, TAU + DELTA)
        if sat_flag:
            sat_band_count += 1
        per_cell.append({
            "cell_id": row["cell_id"],
            "model": row["model_family"],
            "task_slice": row["task_slice"],
            "G": int(g_const),
            "mean_reward": float(row["mean_reward"]),
            "zvf_overall": float(row["zvf"]),
            "max_step_zvf": float(per_step_z.max()),
            "n_sat_band_steps": int((per_step_z >= TAU + DELTA).sum()),
            "sat_band_cell": sat_flag,
            "rollouts_baseline": baseline,
            "rollouts_C1": int(result["C1"]),
            "rollouts_C2": int(result["C2"]),
            "rollouts_C3": int(result["C3"]),
        })

    pcd = pd.DataFrame(per_cell)
    pcd.to_csv(OUT / "p7_mega_saturation_band_per_cell.tsv", sep="\t", index=False)
    print(f"  -> p7_mega_saturation_band_per_cell.tsv ({len(pcd)} rows)")
    print(f"  cells with >=1 saturation-band step (zvf >= {TAU + DELTA:.2f}): {sat_band_count}/{n_cells}")

    # Compute pooled contrasts (C3 - C1, C2 - C1, C3 - C2) and per-stratum
    def pooled_ci(diff):
        rng = np.random.default_rng(BOOT_SEED)
        n = len(diff)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            boots.append(np.mean(diff.iloc[idx]))
        boots = np.array(boots)
        return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    pcd["c3_minus_c1"] = pcd["rollouts_C3"] - pcd["rollouts_C1"]
    pcd["c2_minus_c1"] = pcd["rollouts_C2"] - pcd["rollouts_C1"]
    pcd["c3_minus_c2"] = pcd["rollouts_C3"] - pcd["rollouts_C2"]

    # Pooled across all 98 cells
    pooled = {}
    for diff_col in ["c3_minus_c1", "c2_minus_c1", "c3_minus_c2"]:
        m, lo, hi = pooled_ci(pcd[diff_col])
        pooled[diff_col] = {
            "mean": m,
            "ci025": lo,
            "ci975": hi,
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "n_cells": len(pcd),
        }

    # Per-stratum: sat-band-cell vs not
    sat = pcd[pcd["sat_band_cell"]]
    nosat = pcd[~pcd["sat_band_cell"]]
    strata = {}
    for label, sub in [("sat_band", sat), ("no_sat_band", nosat)]:
        for diff_col in ["c3_minus_c1", "c2_minus_c1", "c3_minus_c2"]:
            if len(sub) == 0:
                continue
            m, lo, hi = pooled_ci(sub[diff_col])
            strata[f"{label}__{diff_col}"] = {
                "n_cells": int(len(sub)),
                "mean": m,
                "ci025": lo,
                "ci975": hi,
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            }

    # Collapse test: how many cells does C3 == C1 on every step?
    # Approximation: count cells where per-cell total C3 == total C1
    c3_eq_c1 = int((pcd["rollouts_C3"] == pcd["rollouts_C1"]).sum())
    c3_lt_c1 = int((pcd["rollouts_C3"] < pcd["rollouts_C1"]).sum())
    c3_gt_c1 = int((pcd["rollouts_C3"] > pcd["rollouts_C1"]).sum())

    summary = {
        "n_cells": int(n_cells),
        "sat_band_count": int(sat_band_count),
        "sat_band_threshold": TAU + DELTA,
        "tau": TAU,
        "delta": DELTA,
        "G_base": G_BASE,
        "pooled": pooled,
        "strata": strata,
        "collapse_test": {
            "c3_eq_c1": c3_eq_c1,
            "c3_lt_c1": c3_lt_c1,
            "c3_gt_c1": c3_gt_c1,
        },
        "falsifiable_prediction": (
            "Hybrid C3 strictly dominates zvf-triage C1 on cells with "
            ">=1 saturation-band step (zvf >= tau+delta), collapses to "
            "C1 on cells with no saturation-band step."
        ),
        "headline": {
            "pooled_c3_minus_c1_mean": pooled["c3_minus_c1"]["mean"],
            "pooled_c3_minus_c1_ci": [pooled["c3_minus_c1"]["ci025"],
                                       pooled["c3_minus_c1"]["ci975"]],
            "pooled_c3_minus_c1_excludes_zero": pooled["c3_minus_c1"]["ci_excludes_zero"],
            "sat_band_c3_minus_c1_mean": strata.get("sat_band__c3_minus_c1", {}).get("mean"),
            "sat_band_c3_minus_c1_ci": [
                strata.get("sat_band__c3_minus_c1", {}).get("ci025"),
                strata.get("sat_band__c3_minus_c1", {}).get("ci975"),
            ],
            "no_sat_band_c3_minus_c1_mean": strata.get("no_sat_band__c3_minus_c1", {}).get("mean"),
        },
    }
    with open(OUT / "p7_mega_saturation_band_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n=== Pooled contrasts (n={}) ===".format(n_cells))
    for k, v in pooled.items():
        print(f"  {k}: mean={v['mean']:>7.1f} CI=[{v['ci025']:>7.1f}, {v['ci975']:>7.1f}]"
              f" excl0={v['ci_excludes_zero']}")
    print("\n=== Per-stratum contrasts ===")
    for k, v in strata.items():
        print(f"  {k}: n={v['n_cells']:>3} mean={v['mean']:>7.1f} CI=[{v['ci025']:>7.1f}, {v['ci975']:>7.1f}]"
              f" excl0={v['ci_excludes_zero']}")
    print(f"\nCollapse test: C3==C1 on {c3_eq_c1}/{n_cells} cells; "
          f"C3<C1 on {c3_lt_c1}; C3>C1 on {c3_gt_c1}")


if __name__ == "__main__":
    main()