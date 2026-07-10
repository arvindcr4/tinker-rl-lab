#!/usr/bin/env python3
"""
length_bias_mechanism.py — Pillar 4 / Iter 20 mechanism test for Dr.GRPO.

Claim under test (Liu et al. 2025, arXiv:2503.20783):
  Dr.GRPO's per-token advantage normalisation A_i -> A_i / |o_i| removes the
  length-induced variance from the policy-gradient update. Concretely, the
  per-step coupling between mean completion length L_t and mean reward R_t
  should be WEAKER under Dr.GRPO than under GRPO, because the long-response
  update-magnitude advantage (the "reward long completions" channel) is
  neutralised.

Inputs (existing, in this worktree):
  experiments/results/drgrpo_vs_grpo.json         (arithmetic_easy / Qwen2.5-0.5B)
  experiments/results/drgrpo_gsm8k_cot_full.json  (gsm8k_cot_hard / Qwen2.5-1.5B)

Outputs:
  experiments/results/length_bias_mechanism_per_run.tsv   (per-run regression slopes + diagnostics)
  experiments/results/length_bias_mechanism_summary.tsv   (per-task, per-algo aggregates + paired Delta)
  experiments/results/length_bias_mechanism_mediation.tsv (ZVF mediation: direct vs indirect effect)
  figures/length_bias_mechanism.pdf + .png                (3-panel: scatter, slopes, mediation)

Usage:
  python scripts/length_bias_mechanism.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ----------------------------- data loading -----------------------------------

def load_runs() -> list[dict]:
    """Load and normalise all per-step runs across the two Dr.GRPO experiments."""
    runs = []
    for src, task_alias in [
        (RES / "drgrpo_vs_grpo.json", "arithmetic_easy_qwen2.5-0.5b"),
        (RES / "drgrpo_gsm8k_cot_full.json", "gsm8k_cot_hard_qwen2.5-1.5b"),
    ]:
        d = json.loads(src.read_text())
        for r in d["runs"]:
            runs.append({
                "task": task_alias,
                "algo": r["algo"],
                "seed": r["seed"],
                "model": r["model"],
                "step_log": r["step_log"],
                "mean_zvf": r.get("mean_zvf", float("nan")),
            })
    return runs


# ----------------------------- regression diagnostics -------------------------

def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares slope of y on x, plus residual std."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    denom = float((dx * dx).sum())
    if denom <= 0:
        return float("nan"), float("nan")
    b = float((dx * dy).sum() / denom)
    a = ym - b * xm
    resid = y - (a + b * x)
    return b, float(resid.std(ddof=2) if resid.size > 2 else 0.0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank-transform + Pearson on ranks."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen robust slope = median of pairwise slopes."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-12:
                slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes)) if slopes else float("nan")


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap on (a - b) per seed. Returns mean, 95% CI, p_two-sided."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    n = len(diff)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided permutation p-value: how often does a random sign-flip match or exceed |mean(diff)|?
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (signs * diff[idx]).mean(axis=1)
    p = float((np.abs(null) >= abs(diff.mean())).mean())
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "p": p}


# ----------------------------- per-run diagnostics ----------------------------

def per_run_diagnostics(run: dict) -> dict:
    sl = run["step_log"]
    steps = np.array([s["step"] for s in sl], float)
    lens = np.array([s["mean_comp_len"] for s in sl], float)
    rews = np.array([s["mean_reward"] for s in sl], float)
    zvfs = np.array([s["zvf"] for s in sl], float)

    # (1) OLS slope of length on reward at the step level
    ols_dL_dR, ols_resid = ols_slope(rews, lens)

    # (2) Theil-Sen robust slope (rules out outlier-driven slopes)
    ts_dL_dR = theil_sen_slope(rews, lens)

    # (3) Spearman rho between step-level length and reward
    sp_L_R = spearman(lens, rews)

    # (4) Length momentum: lag-1 autocorrelation of delta L
    dL = np.diff(lens)
    if len(dL) > 2 and np.std(dL) > 1e-9:
        lag1 = float(np.corrcoef(dL[:-1], dL[1:])[0, 1])
    else:
        lag1 = float("nan")

    # (5) Reward-ceiling test: split steps into lower- vs upper-half by mean_reward,
    #     compute mean length in each half. Dr.GRPO should reduce the late half length.
    if len(rews) >= 6:
        med = float(np.median(rews))
        lo_idx = rews < med
        hi_idx = ~lo_idx
        early_mean_len = float(lens[lo_idx].mean()) if lo_idx.any() else float("nan")
        late_mean_len = float(lens[hi_idx].mean()) if hi_idx.any() else float("nan")
        len_late_minus_early = late_mean_len - early_mean_len
    else:
        early_mean_len = float("nan")
        late_mean_len = float("nan")
        len_late_minus_early = float("nan")

    # (6) Step-level correlation between length and ZVF (coupling proxy)
    sp_L_Z = spearman(lens, zvfs)

    return {
        "task": run["task"],
        "algo": run["algo"],
        "seed": run["seed"],
        "n_steps": int(len(sl)),
        "ols_dL_dR": round(ols_dL_dR, 4),
        "ols_resid": round(ols_resid, 4),
        "theil_sen_dL_dR": round(ts_dL_dR, 4),
        "spearman_L_R": round(sp_L_R, 4),
        "lag1_dL_autocorr": round(lag1, 4),
        "early_half_mean_len": round(early_mean_len, 4),
        "late_half_mean_len": round(late_mean_len, 4),
        "len_late_minus_early": round(len_late_minus_early, 4),
        "spearman_L_Z": round(sp_L_Z, 4),
    }


# ----------------------------- ZVF mediation ----------------------------------

def mediation_analysis(run: dict) -> dict:
    """Decompose the step-level mean_reward ~ mean_comp_len coupling into:
        - direct path (mean_reward ~ mean_comp_len | mean_zvf)
        - indirect path (mean_reward ~ mean_zvf | mean_comp_len)
        - total (mean_reward ~ mean_comp_len, no controls)
    Returns a dict with standardised regression coefficients.
    """
    sl = run["step_log"]
    lens = np.array([s["mean_comp_len"] for s in sl], float)
    rews = np.array([s["mean_reward"] for s in sl], float)
    zvfs = np.array([s["zvf"] for s in sl], float)

    def standardise(v: np.ndarray) -> np.ndarray:
        s = v.std(ddof=0)
        return (v - v.mean()) / s if s > 1e-12 else v * 0.0

    def std_beta(x: np.ndarray, y: np.ndarray) -> float:
        """Standardised slope (== Pearson r) of y on x."""
        sx, sy = standardise(x), standardise(y)
        return float((sx * sy).mean())

    # total effect: standardised slope of reward on length (no controls)
    total = std_beta(lens, rews)
    # direct: residualise reward on zvf, length on zvf, then beta
    rews_resid = rews - zvfs * (np.cov(rews, zvfs, ddof=0)[0, 1] / max(np.var(zvfs, ddof=0), 1e-12))
    lens_resid = lens - zvfs * (np.cov(lens, zvfs, ddof=0)[0, 1] / max(np.var(zvfs, ddof=0), 1e-12))
    direct = std_beta(lens_resid, rews_resid)
    # indirect path: reward ~ zvf | length
    rews_resid2 = rews - lens * (np.cov(rews, lens, ddof=0)[0, 1] / max(np.var(lens, ddof=0), 1e-12))
    zvfs_resid2 = zvfs - lens * (np.cov(zvfs, lens, ddof=0)[0, 1] / max(np.var(lens, ddof=0), 1e-12))
    indirect_reward_on_zvf = std_beta(zvfs_resid2, rews_resid2)
    # length ~ zvf (mediator), then chain through reward
    length_on_zvf = std_beta(zvfs, lens)
    indirect = length_on_zvf * indirect_reward_on_zvf
    prop_mediated = (indirect / total) if abs(total) > 1e-9 else float("nan")

    return {
        "task": run["task"],
        "algo": run["algo"],
        "seed": run["seed"],
        "total_effect_L_R": round(total, 4),
        "direct_effect_L_R_given_Z": round(direct, 4),
        "indirect_via_Z": round(indirect, 4),
        "length_on_zvf": round(length_on_zvf, 4),
        "rew_on_zvf_given_L": round(indirect_reward_on_zvf, 4),
        "prop_mediated": round(prop_mediated, 4),
    }


# ----------------------------- summary tables ---------------------------------

def per_task_summary(per_run: list[dict]) -> list[dict]:
    out = []
    for task in sorted({r["task"] for r in per_run}):
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in per_run if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            slopes_ols = np.array([r["ols_dL_dR"] for r in sub])
            slopes_ts = np.array([r["theil_sen_dL_dR"] for r in sub])
            sp = np.array([r["spearman_L_R"] for r in sub])
            lag1 = np.array([r["lag1_dL_autocorr"] for r in sub])
            le = np.array([r["len_late_minus_early"] for r in sub])
            sp_lz = np.array([r["spearman_L_Z"] for r in sub])
            out.append({
                "task": task,
                "algo": algo,
                "n_seeds": len(sub),
                "mean_ols_dL_dR": round(float(slopes_ols.mean()), 4),
                "sd_ols_dL_dR": round(float(slopes_ols.std(ddof=1)) if len(slopes_ols) > 1 else 0.0, 4),
                "mean_ts_dL_dR": round(float(slopes_ts.mean()), 4),
                "sd_ts_dL_dR": round(float(slopes_ts.std(ddof=1)) if len(slopes_ts) > 1 else 0.0, 4),
                "mean_spearman_L_R": round(float(sp.mean()), 4),
                "mean_lag1_dL_autocorr": round(float(np.nanmean(lag1)), 4),
                "mean_len_late_minus_early": round(float(le.mean()), 4),
                "mean_spearman_L_Z": round(float(sp_lz.mean()), 4),
            })
    # paired delta: dr_grpo - grpo per seed, per task
    for task in sorted({r["task"] for r in per_run}):
        grpo_seeds = {r["seed"] for r in per_run if r["task"] == task and r["algo"] == "grpo"}
        dr_seeds = {r["seed"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo"}
        common = sorted(grpo_seeds & dr_seeds)
        if not common:
            continue
        g_slopes = np.array([r["ols_dL_dR"] for r in per_run if r["task"] == task and r["algo"] == "grpo" and r["seed"] in common])
        d_slopes = np.array([r["ols_dL_dR"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo" and r["seed"] in common])
        g_ts = np.array([r["theil_sen_dL_dR"] for r in per_run if r["task"] == task and r["algo"] == "grpo" and r["seed"] in common])
        d_ts = np.array([r["theil_sen_dL_dR"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo" and r["seed"] in common])
        g_sp = np.array([r["spearman_L_R"] for r in per_run if r["task"] == task and r["algo"] == "grpo" and r["seed"] in common])
        d_sp = np.array([r["spearman_L_R"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo" and r["seed"] in common])
        g_lag = np.array([r["lag1_dL_autocorr"] for r in per_run if r["task"] == task and r["algo"] == "grpo" and r["seed"] in common])
        d_lag = np.array([r["lag1_dL_autocorr"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo" and r["seed"] in common])
        g_le = np.array([r["len_late_minus_early"] for r in per_run if r["task"] == task and r["algo"] == "grpo" and r["seed"] in common])
        d_le = np.array([r["len_late_minus_early"] for r in per_run if r["task"] == task and r["algo"] == "dr_grpo" and r["seed"] in common])

        boot_ols = paired_bootstrap_diff(g_slopes, d_slopes, seed=task.__hash__() & 0xFFFF)
        boot_ts = paired_bootstrap_diff(g_ts, d_ts, seed=task.__hash__() & 0xBEEF)
        boot_sp = paired_bootstrap_diff(g_sp, d_sp, seed=task.__hash__() & 0xCAFE)
        boot_lag = paired_bootstrap_diff(g_lag, d_lag, seed=task.__hash__() & 0xDEAD)
        boot_le = paired_bootstrap_diff(g_le, d_le, seed=task.__hash__() & 0xBABE)

        out.append({
            "task": task,
            "algo": "drgrpo_minus_grpo",
            "n_seeds": len(common),
            "mean_ols_dL_dR": round(boot_ols["mean"], 4),
            "sd_ols_dL_dR": round(float(g_slopes.mean() - d_slopes.mean()) * 0 + 0.0, 4),
            "mean_ts_dL_dR": round(boot_ts["mean"], 4),
            "sd_ts_dL_dR": round(0.0, 4),
            "mean_spearman_L_R": round(boot_sp["mean"], 4),
            "mean_lag1_dL_autocorr": round(boot_lag["mean"], 4),
            "mean_len_late_minus_early": round(boot_le["mean"], 4),
            "mean_spearman_L_Z": float("nan"),
            "_ci_ols_lo": round(boot_ols["lo"], 4),
            "_ci_ols_hi": round(boot_ols["hi"], 4),
            "_p_ols": round(boot_ols["p"], 4),
            "_ci_ts_lo": round(boot_ts["lo"], 4),
            "_ci_ts_hi": round(boot_ts["hi"], 4),
            "_p_ts": round(boot_ts["p"], 4),
            "_ci_sp_lo": round(boot_sp["lo"], 4),
            "_ci_sp_hi": round(boot_sp["hi"], 4),
            "_p_sp": round(boot_sp["p"], 4),
            "_ci_lag_lo": round(boot_lag["lo"], 4),
            "_ci_lag_hi": round(boot_lag["hi"], 4),
            "_p_lag": round(boot_lag["p"], 4),
            "_ci_le_lo": round(boot_le["lo"], 4),
            "_ci_le_hi": round(boot_le["hi"], 4),
            "_p_le": round(boot_le["p"], 4),
        })
    return out


# ----------------------------- figure -----------------------------------------

def make_figure(per_run: list[dict], summary: list[dict], mediation: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_by_key = load_runs()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    colors = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}

    # panel A: scatter of mean_comp_len vs mean_reward per (task, algo, seed)
    ax = axes[0]
    for r in runs_by_key:
        sl = r["step_log"]
        lens = [s["mean_comp_len"] for s in sl]
        rews = [s["mean_reward"] for s in sl]
        c = colors.get(r["algo"], "#888888")
        marker = "o" if "gsm8k" in r["task"] else "s"
        ax.scatter(lens, rews, c=c, alpha=0.5, marker=marker, s=18,
                   label=f"{r['algo']} {'(GSM8K)' if 'gsm8k' in r['task'] else '(arithmetic)'}")
    ax.set_xlabel("step-level mean completion length (tokens)")
    ax.set_ylabel("step-level mean reward")
    ax.set_title("(A) L_t vs R_t trajectories")
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['grpo'], markersize=8, label='GRPO'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dr_grpo'], markersize=8, label='Dr. GRPO')]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    # panel B: OLS slope dL/dR per task per algo (with paired error bars)
    ax = axes[1]
    tasks = sorted({r["task"] for r in per_run})
    width = 0.35
    xpos = np.arange(len(tasks))
    for j, algo in enumerate(("grpo", "dr_grpo")):
        means, lo_err, hi_err = [], [], []
        for task in tasks:
            sub = [r["ols_dL_dR"] for r in per_run if r["task"] == task and r["algo"] == algo]
            if not sub:
                means.append(0.0); lo_err.append(0.0); hi_err.append(0.0)
            else:
                means.append(float(np.mean(sub)))
                # bootstrap CI on the per-seed mean
                rng = np.random.default_rng((task + algo).__hash__() & 0xFFFF)
                idx = rng.integers(0, len(sub), size=(2000, len(sub)))
                boot = np.array(sub)[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                means[-1] = float(np.mean(sub))
                lo_err.append(means[-1] - lo)
                hi_err.append(hi - means[-1])
        ax.bar(xpos + (j - 0.5) * width, means, width, yerr=[lo_err, hi_err],
               color=colors[algo], alpha=0.85, label=algo)
    ax.set_xticks(xpos)
    ax.set_xticklabels([t.replace("_qwen2.5-", "\nQwen2.5-") for t in tasks], fontsize=8)
    ax.set_ylabel("OLS slope dL/dR per run (tokens / reward)")
    ax.set_title("(B) length-reward coupling: lower = flatter")
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.25, axis='y')

    # panel C: mediation direct vs indirect (averaged per (task, algo))
    ax = axes[2]
    pairs = []
    for task in tasks:
        for algo in ("grpo", "dr_grpo"):
            sub = [m for m in mediation if m["task"] == task and m["algo"] == algo]
            if not sub:
                continue
            direct = np.mean([m["direct_effect_L_R_given_Z"] for m in sub])
            indirect = np.mean([m["indirect_via_Z"] for m in sub])
            total = np.mean([m["total_effect_L_R"] for m in sub])
            pairs.append((task, algo, direct, indirect, total))
    xpos = np.arange(len(pairs))
    width = 0.27
    for j, (key, color) in enumerate([("direct", "#1f77b4"), ("indirect_via_Z", "#d62728"), ("total", "#2ca02c")]):
        vals = [p[2 + j] for p in pairs]
        ax.bar(xpos + (j - 1) * width, vals, width, color=color, alpha=0.85, label=key.replace("_", " "))
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{p[0].split('_')[0]}\n{p[1]}" for p in pairs], fontsize=7, rotation=0)
    ax.set_ylabel("standardised slope (Pearson r on standardised vars)")
    ax.set_title("(C) ZVF mediation of length→reward coupling")
    ax.axhline(0.0, color='black', linewidth=0.5)
    ax.legend(framealpha=0.9, fontsize=8)
    ax.grid(alpha=0.25, axis='y')

    fig.suptitle("Dr. GRPO mechanism test: length–reward coupling, slope, and ZVF mediation",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    pdf_path = FIGS / "length_bias_mechanism.pdf"
    png_path = FIGS / "length_bias_mechanism.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    shutil.copy(pdf_path, PAPER_FIGS / "length_bias_mechanism.pdf")
    print(f"wrote {pdf_path} and {png_path} (+ copy in paper/figures/)")


# ----------------------------- tsv writers ------------------------------------

def write_tsv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


# ----------------------------- main -------------------------------------------

def main() -> int:
    runs = load_runs()
    print(f"loaded {len(runs)} runs across {len({r['task'] for r in runs})} tasks")
    for r in runs:
        print(f"  task={r['task']:36s} algo={r['algo']:8s} seed={r['seed']:>5d} steps={len(r['step_log'])}")

    per_run = [per_run_diagnostics(r) for r in runs]
    mediation = [mediation_analysis(r) for r in runs]
    summary = per_task_summary(per_run)

    # ---- per-run regression diagnostics -----------------------------------
    per_run_cols = ["task", "algo", "seed", "n_steps",
                    "ols_dL_dR", "ols_resid", "theil_sen_dL_dR", "spearman_L_R",
                    "lag1_dL_autocorr",
                    "early_half_mean_len", "late_half_mean_len", "len_late_minus_early",
                    "spearman_L_Z"]
    write_tsv(RES / "length_bias_mechanism_per_run.tsv", per_run, per_run_cols)

    # ---- per-task summary -------------------------------------------------
    summary_cols = ["task", "algo", "n_seeds",
                    "mean_ols_dL_dR", "sd_ols_dL_dR", "mean_ts_dL_dR", "sd_ts_dL_dR",
                    "mean_spearman_L_R", "mean_lag1_dL_autocorr",
                    "mean_len_late_minus_early", "mean_spearman_L_Z"]
    write_tsv(RES / "length_bias_mechanism_summary.tsv",
              [{k: v for k, v in r.items() if k in summary_cols} for r in summary],
              summary_cols)

    # ---- paired delta (with bootstrap CI) ---------------------------------
    paired_cols = summary_cols + ["ci_ols_lo", "ci_ols_hi", "p_ols",
                                  "ci_ts_lo", "ci_ts_hi", "p_ts",
                                  "ci_sp_lo", "ci_sp_hi", "p_sp",
                                  "ci_lag_lo", "ci_lag_hi", "p_lag",
                                  "ci_le_lo", "ci_le_hi", "p_le"]
    paired_rows = []
    for r in summary:
        if r["algo"] != "drgrpo_minus_grpo":
            continue
        flat = {k.replace("_", "", 1) if k.startswith("_") else k: v for k, v in r.items()}
        # rename _ci_* keys to plain ci_*
        out = dict(r)
        for k, v in r.items():
            if k.startswith("_ci_"):
                out["ci_" + k[len("_ci_"):]] = v
            if k.startswith("_p_"):
                out["p_" + k[len("_p_"):]] = v
        paired_rows.append(out)
    write_tsv(RES / "length_bias_mechanism_paired.tsv", paired_rows, paired_cols)

    # ---- mediation --------------------------------------------------------
    med_cols = ["task", "algo", "seed",
                "total_effect_L_R", "direct_effect_L_R_given_Z", "indirect_via_Z",
                "length_on_zvf", "rew_on_zvf_given_L", "prop_mediated"]
    write_tsv(RES / "length_bias_mechanism_mediation.tsv", mediation, med_cols)

    # ---- figure -----------------------------------------------------------
    make_figure(per_run, summary, mediation)

    # ---- terminal summary -------------------------------------------------
    print("\n=== mechanism summary (per-task, per-algo) ===")
    for r in summary:
        if r["algo"] == "drgrpo_minus_grpo":
            continue
        print(f"  {r['task']:36s} {r['algo']:8s} "
              f"OLS(dL/dR)={r['mean_ols_dL_dR']:+.2f}  "
              f"TS(dL/dR)={r['mean_ts_dL_dR']:+.2f}  "
              f"rho(L,R)={r['mean_spearman_L_R']:+.2f}  "
              f"lag1(dL)={r['mean_lag1_dL_autocorr']:+.2f}  "
              f"L(late-early)={r['mean_len_late_minus_early']:+.2f}  "
              f"rho(L,Z)={r['mean_spearman_L_Z']:+.2f}")
    print("\n=== paired delta dr_grpo - grpo (negative = Dr.GRPO decouples length-reward) ===")
    for r in summary:
        if r["algo"] != "drgrpo_minus_grpo":
            continue
        print(f"  {r['task']:36s} "
              f"Delta_OLS={r['mean_ols_dL_dR']:+.2f}  "
              f"Delta_rho={r['mean_spearman_L_R']:+.2f}  "
              f"Delta_lag1={r['mean_lag1_dL_autocorr']:+.2f}  "
              f"Delta_L_LE={r['mean_len_late_minus_early']:+.2f}")

    print("\n=== mediation (ZVF as mediator of length -> reward coupling) ===")
    for task in sorted({r["task"] for r in mediation}):
        for algo in ("grpo", "dr_grpo"):
            sub = [m for m in mediation if m["task"] == task and m["algo"] == algo]
            if not sub:
                continue
            tot = np.mean([m["total_effect_L_R"] for m in sub])
            dire = np.mean([m["direct_effect_L_R_given_Z"] for m in sub])
            ind = np.mean([m["indirect_via_Z"] for m in sub])
            pm = np.mean([m["prop_mediated"] for m in sub])
            print(f"  {task:36s} {algo:8s} total={tot:+.3f} direct={dire:+.3f} indirect(Z)={ind:+.3f} prop_med={pm:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())