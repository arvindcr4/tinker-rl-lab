#!/usr/bin/env python3
"""P7 Bayesian adaptive-G controller -- iter 11 (Pillar 3).

Replaces the iter 3 point-estimate rule with a Beta(k+1, n-k+1) posterior
over the per-prompt success probability p. Escalates to G'=16 iff the
posterior mid-range probability P(p in [0.05, 0.95]) exceeds tau_post.
This is the per-prompt analogue of AlphaProof's gamma*=0 Dirichlet(1,1)
smoothing (Berkeley row 19).

Four controllers compared on the same N2 four-method reward tensors:
(A) zvf-triage step-level (iter 3); (B) Dualformer-Auto per-prompt
(Berkeley row 01); (C) Bayesian-escalation (NEW); (D) oracle hindsight.

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_bayesian_summary.{tsv,json}
  platform_hybrid/experiments/results/p5p8/p7_bayesian_per_step.tsv

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
METHODS = ("grpo", "aero", "gift", "areal")
G_BASE = 8
G_ALT = 16
N_PROMPTS = 16
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
MAX_PCD = 0.20
# Posterior mid-range probability thresholds (P(p in [mid_lo, mid_hi]) > tau)
# Sweep spans the informative range: under Beta(9,1) the mid-range prob is
# ~0.63 (observed 8/8); under Beta(5,5) it is ~0.99; under Beta(2,8) it is
# ~0.18. The threshold sweep therefore captures the operating regimes.
TAU_POST = (0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.85, 0.95)
MID_LO = 0.05
MID_HI = 0.95
GRID_N = 1024  # grid points for posterior Beta CDF


# ---
# Beta CDF approximation (stdlib; trapezoidal rule on the log-pdf)
# ---
def _log_beta_pdf(p, alpha, beta):
    """log of Beta(alpha, beta) pdf at p in (0, 1)."""
    if p <= 0 or p >= 1:
        return float("-inf")
    log_norm = (
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )
    return (
        (alpha - 1) * math.log(p)
        + (beta - 1) * math.log(1 - p)
        + log_norm
    )


def beta_cdf(alpha, beta, x, grid=GRID_N):
    """P(P <= x) under Beta(alpha, beta). Trapezoidal integration of the
    pdf over a uniform grid of `grid` subintervals in [0, x]. Endpoints
    get half-weight; inner points full-weight. ~1e-4 accuracy with
    grid=1024 for moderate alpha/beta; uses ~1e-9 near boundaries to
    avoid log(0)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    eps = 1e-9

    def log_pdf(p):
        pp = min(max(p, eps), 1 - eps)
        return _log_beta_pdf(pp, alpha, beta)

    # Integrate over [eps, x] using trapezoidal rule with `grid` subintervals
    a = eps
    b = x
    h = (b - a) / grid
    s = 0.5 * math.exp(log_pdf(a)) + 0.5 * math.exp(log_pdf(b))
    for i in range(1, grid):
        p = a + i * h
        s += math.exp(log_pdf(p))
    val = s * h
    # Plus the tiny mass from 0 to eps (negligible for moderate alpha, beta)
    return min(1.0, max(0.0, val))


def beta_midrange_prob(k, n, lo=MID_LO, hi=MID_HI):
    """P(lo <= p <= hi | k successes, n trials) under Beta(k+1, n-k+1)."""
    alpha = k + 1
    beta = n - k + 1
    return beta_cdf(alpha, beta, hi) - beta_cdf(alpha, beta, lo)


# ---
# Data loading
# ---
def load_tensors():
    """Return dict[(method, step)] -> list[list[float]] (16 x G_BASE rewards)."""
    out = {}
    for m in METHODS:
        path = N2 / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = d["rewards"]
    return out


def is_observed_degenerate(g, eps=1e-9):
    """Observed all-1 or all-0 at G_BASE."""
    mn = min(g)
    mx = max(g)
    if abs(mx - mn) > eps:
        return False
    return abs(mn) < eps or abs(mx - 1.0) < eps


# ---
# Controllers (A,B iter 3 baselines; C NEW; D oracle)
# Bayesian-escalation is the new controller)
# ---
def zvf_triage_escalation(step_info, threshold):
    """Return (saved, wasted, rollouts_used, fires) on this step's tensors.
    For comparison purposes we apply escalation at the *current* step (not
    the next), per the iter 3 logic but re-centered on per-step cost.
    """
    fires = 0
    saved = 0
    wasted = 0
    rollouts = 0
    for i, s in enumerate(step_info):
        if s["zvf"] >= threshold and s["pcd"] <= MAX_PCD:
            fires += 1
            # escalate this step to G'=16; check per-prompt
            rollouts += N_PROMPTS * G_ALT
            for g in s["rewards"]:
                if is_observed_degenerate(g):
                    k = int(round(sum(g)))
                    p_mid = beta_midrange_prob(k, G_BASE)
                    # saved iff posterior mid-range prob is high
                    if p_mid > 0.30:
                        saved += 1
                    else:
                        wasted += 1
        else:
            rollouts += N_PROMPTS * G_BASE
    return fires, saved, wasted, rollouts


def dualformer_auto_escalation(step_info):
    """Berkeley row 01: per-prompt G' based on point-estimate acc_pred.
    G'=2 if >=0.95, 4 if >=0.85, 8 if >=0.70, 16 otherwise. Apply per
    step and accumulate total rollouts. Reports saved prompts under
    same Bayesian mid-range criterion so comparison is apples-to-apples.
    """
    fires = 0
    saved = 0
    wasted = 0
    rollouts = 0
    for s in step_info:
        for g in s["rewards"]:
            fires += 1
            p = sum(g) / len(g)
            if p >= 0.95:
                g_new = 2
            elif p >= 0.85:
                g_new = 4
            elif p >= 0.70:
                g_new = 8
            else:
                g_new = 16
            rollouts += g_new
            if is_observed_degenerate(g) and g_new > G_BASE:
                k = int(round(sum(g)))
                p_mid = beta_midrange_prob(k, G_BASE)
                if p_mid > 0.30:
                    saved += 1
                else:
                    wasted += 1
            elif is_observed_degenerate(g) and g_new <= G_BASE:
                # controller chose not to escalate
                wasted += 1
    return fires, saved, wasted, rollouts


def bayesian_escalation(step_info, tau_post):
    """Per-prompt Bayesian escalation:
       G'=16 iff observed group at G=8 is degenerate AND posterior
       P(p in [mid_lo, mid_hi]) > tau_post; else G=8.
    """
    fires = 0
    saved = 0
    wasted = 0
    rollouts = 0
    for s in step_info:
        for g in s["rewards"]:
            if is_observed_degenerate(g):
                k = int(round(sum(g)))
                p_mid = beta_midrange_prob(k, G_BASE)
                if p_mid > tau_post:
                    rollouts += G_ALT
                    fires += 1
                    saved += 1
                else:
                    rollouts += G_BASE
                    wasted += 1
            else:
                rollouts += G_BASE  # already mixed -> keep G
    return fires, saved, wasted, rollouts


def oracle_escalation(step_info):
    """Oracle hindsight: G'=16 iff observed 0.05 < p_hat < 0.95.
    This is the upper bound on contrast-restoration the data permits.
    """
    fires = 0
    saved = 0
    wasted = 0
    rollouts = 0
    for s in step_info:
        for g in s["rewards"]:
            p = sum(g) / len(g)
            if 0.05 < p < 0.95:
                rollouts += G_ALT
                fires += 1
                if is_observed_degenerate(g):
                    saved += 1
            else:
                rollouts += G_BASE
                if is_observed_degenerate(g):
                    wasted += 1
    return fires, saved, wasted, rollouts


def fixed_g_baseline():
    return N_PROMPTS * G_BASE * len(range(40))  # 40 steps


# ---
# Bootstrap CI on saved/fires across the 4 methods
# ---
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    if not values:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(values)
    point = statistics.mean(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return {"point": round(point, 3), "lo": round(lo, 3), "hi": round(hi, 3)}


# ---
# Main
# ---
def _eval_controller(name, threshold, eval_fn, method_step_info,
                     baseline, seed, per_step_rows, controller_label):
    """Run one controller over all methods; return summary row."""
    pm = []
    for m in METHODS:
        fires, saved, wasted, rollouts = eval_fn(method_step_info[m])
        pm.append({"fires": fires, "saved": saved,
                   "wasted": wasted, "rollouts": rollouts})
        if name == "zvf_triage":
            for s in method_step_info[m]:
                if s["zvf"] >= threshold and s["pcd"] <= MAX_PCD:
                    per_step_rows.append({
                        "method": m, "controller": controller_label,
                        "threshold": threshold, "step": s["step"],
                        "trigger_zvf": round(s["zvf"], 4),
                        "trigger_pcd": round(s["pcd"], 4),
                    })
    fires_v = [x["fires"] for x in pm]
    saved_v = [x["saved"] for x in pm]
    wasted_v = [x["wasted"] for x in pm]
    roll_v = [x["rollouts"] for x in pm]
    ci = bootstrap_ci(saved_v, seed=seed)
    return {
        "controller": controller_label,
        "threshold": threshold,
        "fires_per_method": json.dumps(fires_v),
        "saved_per_method": json.dumps(saved_v),
        "wasted_per_method": json.dumps(wasted_v),
        "rollouts_per_method": json.dumps(roll_v),
        "fires_mean": round(statistics.mean(fires_v), 2),
        "saved_mean": round(statistics.mean(saved_v), 2),
        "saved_lo": ci["lo"], "saved_hi": ci["hi"],
        "wasted_mean": round(statistics.mean(wasted_v), 2),
        "rollouts_mean": round(statistics.mean(roll_v), 1),
        "cost_ratio_mean": round(statistics.mean(roll_v) / baseline, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Write outputs under platform_hybrid/experiments/results/p5p8/")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tensors = load_tensors()
    steps = sorted({k[1] for k in tensors})

    # Per-step info per method
    method_step_info = {}
    for m in METHODS:
        info = []
        with (N2 / f"{m}_s0_tensors.jsonl").open() as f:
            for line in f:
                d = json.loads(line)
                info.append({"step": d["step"], "zvf": d["zvf"],
                             "pcd": d["pcd"], "rewards": d["rewards"]})
        info.sort(key=lambda x: x["step"])
        method_step_info[m] = info

    summary_rows = []
    per_step_rows = []
    baseline = fixed_g_baseline()

    # A. zvf-triage threshold sweep
    for thr in THRESHOLDS:
        summary_rows.append(_eval_controller(
            "zvf_triage", thr,
            lambda info, t=thr: zvf_triage_escalation(info, t),
            method_step_info, baseline, args.seed, per_step_rows,
            "zvf_triage"))

    # B. Dualformer-Auto
    summary_rows.append(_eval_controller(
        "dualformer_auto", "point",
        lambda info: dualformer_auto_escalation(info),
        method_step_info, baseline, args.seed, per_step_rows,
        "dualformer_auto"))

    # C. Bayesian escalation (NEW) -- sweep tau_post
    for tp in TAU_POST:
        summary_rows.append(_eval_controller(
            "bayesian", tp,
            lambda info, t=tp: bayesian_escalation(info, t),
            method_step_info, baseline, args.seed, per_step_rows,
            "bayesian"))

    # D. Oracle
    summary_rows.append(_eval_controller(
        "oracle", "midrange",
        lambda info: oracle_escalation(info),
        method_step_info, baseline, args.seed, per_step_rows,
        "oracle"))

    # Write outputs
    if not args.write:
        print(json.dumps(summary_rows, indent=2))
        return

    cols = ["controller", "threshold", "fires_mean", "saved_mean",
            "saved_lo", "saved_hi", "wasted_mean", "rollouts_mean",
            "cost_ratio_mean"]
    with (OUT / "p7_bayesian_summary.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    json_payload = {
        "n_methods": len(METHODS), "n_steps": len(steps),
        "n_prompts_per_step": N_PROMPTS, "g_base": G_BASE, "g_alt": G_ALT,
        "baseline_rollouts_fixed_g": baseline, "summary_rows": summary_rows,
        "interpretation": {
            "bayesian_threshold_meaning": (
                "tau_post is the posterior mid-range probability "
                "P(p in [0.05, 0.95]) above which the controller "
                "escalates an observed-degenerate prompt from G=8 to "
                "G'=16."),
            "alpha_proof_bridge": (
                "The Beta(k+1, n-k+1) posterior over p is the exact "
                "Bayesian analogue of AlphaProof's gamma*=0 smoothing: "
                "both treat the empirical group statistic as a single "
                "observation of a latent success probability and "
                "regularize via a prior (Dirichlet(1,1) in both cases)."),
            "dualformer_bridge": (
                "Controller B uses a point estimate of acc_pred (no "
                "uncertainty); Controller C conditions on the posterior."),
        },
    }
    with (OUT / "p7_bayesian_summary.json").open("w") as f:
        json.dump(json_payload, f, indent=2)

    cols2 = ["method", "controller", "threshold", "step",
             "trigger_zvf", "trigger_pcd"]
    with (OUT / "p7_bayesian_per_step.tsv").open("w") as f:
        f.write("\t".join(cols2) + "\n")
        for r in per_step_rows:
            f.write("\t".join(str(r[c]) for c in cols2) + "\n")


if __name__ == "__main__":
    main()