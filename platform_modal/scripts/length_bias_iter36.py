#!/usr/bin/env python3
"""
length_bias_iter36.py — Joint (L, R) saturation cross-pillar P4 x P1 analysis.

GOAL.  Prior iters (8, 12, 16, 20, 24, 28, 32) studied length and reward
trajectories *separately*.  This iter asks the joint question: do they share
a common saturation dynamic, or do they decouple?

Verbosity-trap hypothesis (Dr.GRPO / MAD GRPO literature): L grows *while*
R plateaus -> the two trajectories decouple, with R saturating FAST and L
either drifting or growing superlinearly.

Anti-trap / correction-channel hypothesis (iter 28's evidence): L *falls*
during the R-rising phase -> the two trajectories move in OPPOSITE
directions, so R saturates from below while L saturates from above.

Joint-saturation hypothesis: R and L both saturate with *similar* lambda,
same time-scale, common descriptive model.

We fit R(t) = R_max * (1 - exp(-lambda_R * t)) and
       L(t) = L_0 + (L_max - L_0) * (1 - exp(-lambda_L * t))
on each run's step log.  Then per-(task, algo, seed) we report
(lambda_R, lambda_L, t80_R, t80_L, lambda_ratio = L/R), and per-(task, algo)
we run a paired bootstrap on lambda_R - lambda_L to test the
decoupling / common-scale hypotheses.

Secondary metric: residual cross-correlation rho(eps_R(t), eps_L(t)) where
eps_X(t) = X(t) - X_fit(t).  Positive rho means common residual structure;
negative rho means the residuals move in opposite directions (one above
its fit while the other is below).

Data: platform_hybrid/experiments/results/drgrpo_vs_grpo.json (Qwen2.5-0.5B arithmetic,
  40 steps, 5 seeds per algo) and drgrpo_gsm8k_cot_full.json
  (Qwen2.5-1.5B GSM8K CoT, 30 steps, 3 seeds per algo).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "results"
DRGRPO = ROOT / "experiments" / "results" / "drgrpo_vs_grpo.json"
GSM8K = ROOT / "experiments" / "results" / "drgrpo_gsm8k_cot_full.json"


def load_runs(path):
    with open(path) as f:
        d = json.load(f)
    return d["runs"]


def fit_saturation(t, y, eps=1e-6, lam_lo=1e-4, lam_hi=10.0):
    """Fit y(t) = a + (b - a) * (1 - exp(-lambda * t)) via grid + Newton polish.

    Returns (a, b, lambda, t80 = -log(0.2)/lambda) and the in-sample fit
    values yhat(t).  Returns lambda=None if the optimum lies at the boundary
    (saturated-to-noise regime, identifiability failure).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 4:
        return None
    # --- grid search over lambda in a sensible range
    best = None
    best_sse = np.inf
    for lam in np.logspace(-2.0, 1.0, 60):
        # closed-form a, b for fixed lambda via linear regression
        X = np.column_stack([np.ones(n), 1.0 - np.exp(-lam * t)])
        # y = a + (b-a) * phi -> a + b*phi - a*phi
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, c = float(coef[0]), float(coef[1])  # c = b - a
        b = a + c
        yhat = a + c * (1.0 - np.exp(-lam * t))
        sse = float(np.sum((y - yhat) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (a, b, lam)
    a, b, lam = best
    # --- Newton polish on lambda (1-D) with bounded step
    for _ in range(8):
        dlam = 1e-3 * lam
        phi_p = 1.0 - np.exp(-(lam + dlam) * t)
        phi_m = 1.0 - np.exp(-(lam - dlam) * t)
        # refit a, c at lam+dlam and lam-dlam
        Xp = np.column_stack([np.ones(n), phi_p])
        ap, cp = np.linalg.lstsq(Xp, y, rcond=None)[0]
        Xm = np.column_stack([np.ones(n), phi_m])
        am, cm = np.linalg.lstsq(Xm, y, rcond=None)[0]
        sse_p = float(np.sum((y - ap - cp * phi_p) ** 2))
        sse_m = float(np.sum((y - am - cm * phi_m) ** 2))
        grad = (sse_p - sse_m) / (2 * dlam + eps)
        step = -1e-2 * grad / (abs(grad) + 1.0)
        lam_new = max(lam_lo, min(lam_hi, lam + step))
        if abs(lam_new - lam) < 1e-6:
            lam = lam_new
            break
        lam = lam_new
    # final refit at polished lambda
    X = np.column_stack([np.ones(n), 1.0 - np.exp(-lam * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, c = float(coef[0]), float(coef[1])
    b = a + c
    yhat = a + c * (1.0 - np.exp(-lam * t))
    t80 = float(-math.log(0.2) / max(lam, eps))
    identifiable = lam_lo < lam < lam_hi
    return dict(a=a, b=b, lam=float(lam), t80=t80, yhat=yhat, identifiable=identifiable)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def paired_bootstrap_diff(metric_per_seed_a, metric_per_seed_b, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(metric_per_seed_a, dtype=float)
    b = np.asarray(metric_per_seed_b, dtype=float)
    diffs = a - b
    obs = float(diffs.mean())
    n = len(diffs)
    if n < 2:
        return dict(obs=obs, lo=float("nan"), hi=float("nan"), p_le0=float("nan"))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_le0 = float((boots <= 0).mean())
    return dict(obs=obs, lo=float(lo), hi=float(hi), p_le0=p_le0)


def main():
    out_per_run = []
    out_summary = []
    out_cross = []
    sources = [
        ("arithmetic_easy", DRGRPO),
        ("gsm8k_cot", GSM8K),
    ]
    for task, path in sources:
        runs = load_runs(path)
        # group by algo
        per_algo_lam = {"grpo": [], "dr_grpo": []}
        per_algo_t80_R = {"grpo": [], "dr_grpo": []}
        per_algo_t80_L = {"grpo": [], "dr_grpo": []}
        per_algo_resid_rho = {"grpo": [], "dr_grpo": []}
        for run in runs:
            algo = run["algo"]
            seed = run["seed"]
            sl = run["step_log"]
            t = np.array([s["step"] for s in sl], dtype=float)
            R = np.array([s["mean_reward"] for s in sl], dtype=float)
            L = np.array([s["mean_comp_len"] for s in sl], dtype=float)
            fit_R = fit_saturation(t, R)
            fit_L = fit_saturation(t, L)
            if fit_R is None or fit_L is None:
                continue
            eps_R = R - fit_R["yhat"]
            eps_L = L - fit_L["yhat"]
            resid_rho = spearman(eps_R, eps_L)
            row = dict(
                task=task,
                algo=algo,
                seed=seed,
                n_steps=len(t),
                a_R=round(fit_R["a"], 4),
                b_R=round(fit_R["b"], 4),
                lam_R=round(fit_R["lam"], 4),
                t80_R=round(fit_R["t80"], 2),
                R_ident=int(fit_R["identifiable"]),
                a_L=round(fit_L["a"], 4),
                b_L=round(fit_L["b"], 4),
                lam_L=round(fit_L["lam"], 4),
                t80_L=round(fit_L["t80"], 2),
                L_ident=int(fit_L["identifiable"]),
                lam_ratio=round(fit_L["lam"] / fit_R["lam"], 4),
                resid_rho=round(resid_rho, 4),
            )
            out_per_run.append(row)
            per_algo_lam[algo].append(fit_L["lam"] / fit_R["lam"])
            per_algo_t80_R[algo].append(fit_R["t80"])
            per_algo_t80_L[algo].append(fit_L["t80"])
            per_algo_resid_rho[algo].append(resid_rho)
        # paired bootstrap on lambda ratio: GRPO vs Dr.GRPO (mean + median)
        boot = paired_bootstrap_diff(per_algo_lam["grpo"], per_algo_lam["dr_grpo"])
        out_cross.append(dict(
            task=task,
            metric="lam_ratio (L/R)",
            grpo_mean=round(float(np.mean(per_algo_lam["grpo"])), 4),
            grpo_median=round(float(np.median(per_algo_lam["grpo"])), 4),
            drgrpo_mean=round(float(np.mean(per_algo_lam["dr_grpo"])), 4),
            drgrpo_median=round(float(np.median(per_algo_lam["dr_grpo"])), 4),
            diff_grpo_minus_drgrpo=round(float(np.mean(per_algo_lam["grpo"]) - np.mean(per_algo_lam["dr_grpo"])), 4),
            diff_lo=round(boot["lo"], 4),
            diff_hi=round(boot["hi"], 4),
            p_le0=round(boot["p_le0"], 4),
        ))
        # per-algo summary (use median for robustness to boundary fits)
        for algo in ("grpo", "dr_grpo"):
            n = len(per_algo_lam[algo])
            if n == 0:
                continue
            # restrict to identifiable fits for the lambda summaries
            rows_a = [r for r in out_per_run if r["task"] == task and r["algo"] == algo]
            ident_R = [r for r in rows_a if r["R_ident"] == 1]
            ident_L = [r for r in rows_a if r["L_ident"] == 1]
            mean_R = float(np.median([r["lam_R"] for r in ident_R])) if ident_R else float("nan")
            mean_L = float(np.median([r["lam_L"] for r in ident_L])) if ident_L else float("nan")
            out_summary.append(dict(
                task=task,
                algo=algo,
                n_seeds=n,
                n_R_ident=len(ident_R),
                n_L_ident=len(ident_L),
                median_lam_R=round(mean_R, 4),
                median_lam_L=round(mean_L, 4),
                median_t80_R=round(float(np.median(per_algo_t80_R[algo])), 2),
                median_t80_L=round(float(np.median(per_algo_t80_L[algo])), 2),
                median_lam_ratio=round(float(np.median(per_algo_lam[algo])), 4),
                median_resid_rho=round(float(np.median(per_algo_resid_rho[algo])), 4),
            ))
    # write outputs
    p1 = OUT_DIR / "length_bias_iter36_per_run.tsv"
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_per_run[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out_per_run)
    p2 = OUT_DIR / "length_bias_iter36_summary.tsv"
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_summary[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out_summary)
    p3 = OUT_DIR / "length_bias_iter36_grpo_vs_drgrpo.tsv"
    with open(p3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_cross[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out_cross)
    # also a one-row summary table for the paper
    p4 = OUT_DIR / "length_bias_iter36_findings.tsv"
    findings = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        rows = [r for r in out_per_run if r["task"] == task]
        g = [r for r in rows if r["algo"] == "grpo"]
        d = [r for r in rows if r["algo"] == "dr_grpo"]
        # for findings use median, more robust
        ratio_g = float(np.median([r["lam_ratio"] for r in g]))
        ratio_d = float(np.median([r["lam_ratio"] for r in d]))
        rho_g = float(np.median([r["resid_rho"] for r in g]))
        rho_d = float(np.median([r["resid_rho"] for r in d]))
        # count identifiable L fits
        n_g_L = sum(1 for r in g if r["L_ident"] == 1)
        n_d_L = sum(1 for r in d if r["L_ident"] == 1)
        n_g_R = sum(1 for r in g if r["R_ident"] == 1)
        n_d_R = sum(1 for r in d if r["R_ident"] == 1)
        findings.append(dict(
            task=task,
            median_lam_ratio_GRPO=round(ratio_g, 4),
            median_lam_ratio_DrGRPO=round(ratio_d, 4),
            median_resid_rho_GRPO=round(rho_g, 4),
            median_resid_rho_DrGRPO=round(rho_d, 4),
            n_L_ident_GRPO=n_g_L,
            n_L_ident_DrGRPO=n_d_L,
            n_R_ident_GRPO=n_g_R,
            n_R_ident_DrGRPO=n_d_R,
            joint_saturation_holds=("yes" if abs(ratio_g - 1) < 0.5 and abs(ratio_d - 1) < 0.5 else "no"),
        ))
    with open(p4, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(findings[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(findings)
    print(f"Wrote {p1.name} ({len(out_per_run)} rows)")
    print(f"Wrote {p2.name} ({len(out_summary)} rows)")
    print(f"Wrote {p3.name} ({len(out_cross)} rows)")
    print(f"Wrote {p4.name} ({len(findings)} rows)")
    print()
    print("== per-(task, algo) summary ==")
    for s in out_summary:
        print(s)
    print()
    print("== GRPO vs Dr.GRPO lambda ratio ==")
    for s in out_cross:
        print(s)
    print()
    print("== findings ==")
    for f in findings:
        print(f)


if __name__ == "__main__":
    main()