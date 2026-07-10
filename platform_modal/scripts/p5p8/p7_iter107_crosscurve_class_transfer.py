#!/usr/bin/env python3
"""
Iter 107 -- Pillar 3 (P7) PART B: cross-method (savings, fires) curve correlation
+ per-method operating-point comparison + Cohen-kappa failure-class agreement.

Uses outputs from p7_iter107_tautransfer_failure.py and adds:
  1. Savings-vs-tau curves per method on a FINE tau grid (every 0.025);
     pairwise PEARSON R between curves; bootstrap CI.
  2. Per-method operating-point costs in (tau, savings, ret) plane; identify
     whether ANY method's curve Pareto-dominates the others.
  3. Cohen-kappa per (method_a, method_b) on the failure class assignment
     (run at three canonical taus {0.70, 0.80, 0.90}).

Outputs (platform_hybrid/experiments/results/p5p8/):
  p7_iter107b_curve_table.tsv            one row per (method, tau)
  p7_iter107b_curve_correlation.tsv      Pearson r per (method_a, method_b) pair
  p7_iter107b_curve_correlation_boot.tsv bootstrap CI on r
  p7_iter107b_operating_points.tsv       per-method curve summary
  p7_iter107b_kappa_class_agreement.tsv  Cohen kappa at 3 taus x 6 method-pairs
  p7_iter107b_summary.json

Stdlib only. <= 280 LoC.
"""
from __future__ import annotations
import csv
import json
import math
import pathlib
import random
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ("grpo", "aero", "gift", "areal")
TAUS_FINE = [round(0.05 * i + 0.50, 3) for i in range(9)]  # 0.50..0.90 step 0.05
TAUS_CANONICAL = (0.70, 0.80, 0.90)
G_BASE = 8
G_POOL = (2, 4, 8, 16)
N_PROMPTS = 16
PCD_GUARD = 0.20
N_BOOT = 4000
BOOT_SEED = 20260706
EPS = 1e-9


def load_tensors():
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = (d["rewards"], {
                    "zvf": d.get("zvf", 0.0),
                    "pcd": d.get("pcd", 0.0),
                })
    return out


def per_p(prompt_rewards):
    return [sum(g) / len(g) for g in prompt_rewards]


def iid_zvf(p, g):
    pp = min(max(p, EPS), 1 - EPS)
    return (1 - pp) ** g + pp ** g


def decide_gs_c3(per_p, zvf, pcd, tau):
    gs = []
    for p in per_p:
        if p >= 0.95:
            gs.append(2)
        elif p >= 0.85:
            gs.append(4)
        elif p >= 0.70:
            gs.append(8)
        else:
            gs.append(16)
    fired = False
    if zvf >= tau and pcd <= PCD_GUARD:
        idx_map = {2: 0, 4: 1, 8: 2, 16: 3}
        bump_map = {0: 4, 1: 8, 2: 16, 3: 16}
        gs = [bump_map[idx_map[g]] for g in gs]
        fired = True
    return gs, fired


def contrast_mag(per_p, gs):
    return sum(1 - iid_zvf(p, g) for p, g in zip(per_p, gs))


def step_opportunity(per_p, g_new=16):
    return sum(1 - iid_zvf(p, g_new) for p in per_p)


def failure_class(s, tau):
    zvf, pcd, per_p = s["zvf"], s["pcd"], s["per_p"]
    opp = step_opportunity(per_p, g_new=16)
    fires_correctly = (zvf >= tau) and (pcd <= PCD_GUARD)
    if fires_correctly:
        if opp > 0.05:
            return "A_HIT"
        return "D_FP_BDRY"
    if opp <= 0.05:
        return "E_TN"
    if zvf >= tau:
        return "D_FP_BDRY"
    if opp >= 1.5:
        return "C_FN_DRIFT"
    return "B_FN_BDRY"


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def cohens_kappa(a_classes, b_classes):
    classes = sorted(set(a_classes) | set(b_classes))
    n = len(a_classes)
    if n == 0:
        return 0.0
    observed_agree = sum(1 for x, y in zip(a_classes, b_classes) if x == y)
    po = observed_agree / n
    pe = sum((a_classes.count(c) / n) * (b_classes.count(c) / n)
             for c in classes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    tensors = load_tensors()
    steps = sorted({k[1] for k in tensors if isinstance(k[1], int)})
    steps_per_method = {}
    for m in METHODS:
        sl = []
        for s in steps:
            rew, meta = tensors[(m, s)]
            sl.append({"step": s, "zvf": meta["zvf"], "pcd": meta["pcd"],
                       "per_p": per_p(rew)})
        steps_per_method[m] = sl

    # =========================================================================
    # PART 1: (tau, savings, ret) curves per method on the fine tau grid
    # =========================================================================
    curve_rows = []
    curve_savings = {m: [] for m in METHODS}   # per-tau list
    curve_ret = {m: [] for m in METHODS}
    curve_fires = {m: [] for m in METHODS}
    for m in METHODS:
        base_roll = N_PROMPTS * G_BASE * len(steps)
        base_mag = sum(contrast_mag(s["per_p"], [G_BASE] * N_PROMPTS)
                       for s in steps_per_method[m])
        for tau in TAUS_FINE:
            roll = 0
            mag = 0.0
            fires = 0
            for s in steps_per_method[m]:
                gs, fired = decide_gs_c3(s["per_p"], s["zvf"], s["pcd"], tau)
                roll += sum(gs)
                mag += contrast_mag(s["per_p"], gs)
                fires += int(fired)
            sav = (base_roll - roll) / base_roll
            ret = mag / base_mag if base_mag > 0 else 1.0
            curve_savings[m].append(sav)
            curve_ret[m].append(ret)
            curve_fires[m].append(fires)
            curve_rows.append({"method": m, "tau": tau,
                               "savings_frac": sav,
                               "magnitude_retention": ret,
                               "fires": fires})

    # Pearson r per pair on savings(τ) curves
    corr_rows = []
    for i, ma in enumerate(METHODS):
        for j, mb in enumerate(METHODS):
            if j <= i:
                continue
            r = pearson(curve_savings[ma], curve_savings[mb])
            r_ret = pearson(curve_ret[ma], curve_ret[mb])
            r_fires = pearson(curve_fires[ma], curve_fires[mb])
            corr_rows.append({"method_a": ma, "method_b": mb,
                              "r_savings_tau": r,
                              "r_retention_tau": r_ret,
                              "r_fires_tau": r_fires})

    # Bootstrap CI on Pearson r (savings): bootstrap by step-resampling
    # the per-step savings contribution within a method, then recomputing
    # the per-method savings-at-each-tau statistic and re-pearsoning.
    rng = random.Random(BOOT_SEED)
    # Per-step savings contribution by (method, tau): the savings delta if a
    # step fires (vs. doesn't).  Simple resample proxy: bootstrap the per-step
    # contributed-rollouts (sum_gs vs G_BASE*N) at each tau.
    per_step_rollout_delta = {}  # (m, tau_k) -> list of (used - G_BASE*N_PROMPTS)
    for m in METHODS:
        for ti, tau in enumerate(TAUS_FINE):
            deltas = []
            for s in steps_per_method[m]:
                gs, _ = decide_gs_c3(s["per_p"], s["zvf"], s["pcd"], tau)
                deltas.append(sum(gs) - N_PROMPTS * G_BASE)
            per_step_rollout_delta[(m, ti)] = deltas
    base_roll = N_PROMPTS * G_BASE * len(steps)
    boot_pair_corr = {}
    for c in corr_rows:
        boot_pair_corr.setdefault(
            (c["method_a"], c["method_b"]), [c["r_savings_tau"]])
    for (ma, mb), lst in list(boot_pair_corr.items()):
        n_steps = len(steps_per_method[ma])
        for _ in range(N_BOOT):
            ids = [rng.randrange(n_steps) for _ in range(n_steps)]
            xs_curve = []
            ys_curve = []
            for ti in range(len(TAUS_FINE)):
                xs_curve.append(
                    -sum(per_step_rollout_delta[(ma, ti)][k] for k in ids)
                    / base_roll)
                ys_curve.append(
                    -sum(per_step_rollout_delta[(mb, ti)][k] for k in ids)
                    / base_roll)
            r = pearson(xs_curve, ys_curve)
            lst.append(r)
    corr_boot_rows = []
    for (ma, mb), boots in boot_pair_corr.items():
        # First entry is the point estimate; rest are bootstraps.
        if len(boots) <= 1:
            continue
        point = boots[0]
        reps = sorted(boots[1:])
        lo = reps[int(0.025 * len(reps))]
        hi = reps[int(0.975 * len(reps))]
        corr_boot_rows.append({
            "method_a": ma, "method_b": mb,
            "r_savings_tau_point": point,
            "r_savings_tau_ci_lo": lo,
            "r_savings_tau_ci_hi": hi,
            "ci_excludes_zero": int(lo > 0 or hi < 0),
            "n_boot": len(reps),
        })

    # Operating-point summary: at each method, the tau that maximises
    # savings * retention -- the Pareto proxy
    op_rows = []
    for m in METHODS:
        idx_best = max(range(len(TAUS_FINE)),
                       key=lambda i: curve_savings[m][i] * curve_ret[m][i])
        tau_best = TAUS_FINE[idx_best]
        sav_best = curve_savings[m][idx_best]
        ret_best = curve_ret[m][idx_best]
        op_rows.append({"method": m, "tau_best": tau_best,
                        "savings_frac": sav_best,
                        "magnitude_retention": ret_best,
                        "sav_x_ret": sav_best * ret_best})

    # =========================================================================
    # PART 3: Cohen-kappa on failure class assignment at 3 canonical taus
    # =========================================================================
    # Class labels per (method, step) at each canonical tau
    class_at = {}
    for tau in TAUS_CANONICAL:
        for m in METHODS:
            class_at[(m, tau)] = [failure_class(s, tau)
                                   for s in steps_per_method[m]]
    kappa_rows = []
    for tau in TAUS_CANONICAL:
        for i, ma in enumerate(METHODS):
            for j, mb in enumerate(METHODS):
                if j <= i:
                    continue
                k = cohens_kappa(class_at[(ma, tau)],
                                 class_at[(mb, tau)])
                kappa_rows.append({"tau": tau, "method_a": ma,
                                   "method_b": mb,
                                   "cohens_kappa": k})

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================
    with (OUT / "p7_iter107b_curve_table.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)
    with (OUT / "p7_iter107b_curve_correlation.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in corr_rows:
            w.writerow(r)
    with (OUT / "p7_iter107b_curve_correlation_boot.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_boot_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in corr_boot_rows:
            w.writerow(r)
    with (OUT / "p7_iter107b_operating_points.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(op_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in op_rows:
            w.writerow(r)
    with (OUT / "p7_iter107b_kappa_class_agreement.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(kappa_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in kappa_rows:
            w.writerow(r)

    summary = {
        "ts": "2026-07-05",
        "iter": 107,
        "part": "B",
        "n_steps_per_method": {m: len(steps_per_method[m])
                               for m in METHODS},
        "tau_grid_fine": TAUS_FINE,
        "tau_canonical": list(TAUS_CANONICAL),
        "curve_correlations_pearson": corr_rows,
        "operating_points": op_rows,
        "kappa_table": kappa_rows,
    }
    with (OUT / "p7_iter107b_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
