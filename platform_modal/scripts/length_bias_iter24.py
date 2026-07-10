#!/usr/bin/env python3
"""
length_bias_iter24.py — Pillar 4 / Iter 24: drift & forecast test of the
Dr.GRPO verbosity-trap signature.

Hypothesis under test (Liu et al. 2025, arXiv:2503.20783):
  The Dr.GRPO verbosity trap should manifest as a *drift*: ρ(len_t, R_t)
  goes from negative (compression regime) toward zero or positive as
  training proceeds, because the model's within-step length distribution
  collapses onto a single verbose template and length ceases to be
  diagnostic of correctness.

We measure FOUR drift/forecast signals purely from per-step aggregates
(mean_reward_t, mean_comp_len_t) that already exist in the worktree:

  S1. Sliding-window Spearman ρ_w(len, R) for windows w=8, 10, 12 steps.
      Report (mean, sd) over seeds for [early, mid, late] windows.

  S2. Sign-flip test: fraction of late-window ρ_w that are >= 0.
      Under H0 (no drift), the fraction should equal 0.50 ± sampling noise.

  S3. Forecast horizon: linear extrapolation of length vs step — report
      the step t* at which len(t*) returns to len(0) (i.e. predicted
      length inflation horizon). If the linear fit gives a NEGATIVE t*,
      compression is unbounded; report it explicitly.

  S4. First-difference coupling: ρ(ΔL_t, ΔR_t) within-run. This detrends
      the joint trajectory and exposes the local feedback channel:
      does a length INCREASE follow a reward INCREASE (herding) or
      precede a reward DECREASE (compression)?

Inputs (existing):
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json         (40 steps × 10 runs)
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json  (30 steps × 6 runs)

Outputs (new):
  platform_hybrid/experiments/results/length_bias_iter24_windows.tsv     (S1: per-window ρ)
  platform_hybrid/experiments/results/length_bias_iter24_signflip.tsv    (S2: drift statistics)
  platform_hybrid/experiments/results/length_bias_iter24_forecast.tsv    (S3: t* predictions)
  platform_hybrid/experiments/results/length_bias_iter24_diffcorr.tsv    (S4: ΔL-ΔR coupling)
  platform_hybrid/experiments/results/length_bias_iter24_summary.tsv     (per-task-algo aggregates)
  figures/length_bias_iter24.pdf + .png                  (4-panel: windows / sign / forecast / diff)

Cite: tong2025drgrpo (Liu et al. 2025, arXiv:2503.20783)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


# ----------------------------- data loading ---------------------------------

def load_runs() -> list[dict]:
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
                "model": r.get("model", ""),
                "step_log": r["step_log"],
            })
    return runs


# ----------------------------- metrics --------------------------------------

def windowed_spearman(x: list[float], y: list[float], win: int) -> list[float]:
    """Sliding-window Spearman ρ(len, R) for windows of size `win`."""
    out = []
    for i in range(0, len(x) - win + 1):
        xs = x[i : i + win]
        ys = y[i : i + win]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        rho, _ = stats.spearmanr(xs, ys)
        if not np.isnan(rho):
            out.append(float(rho))
    return out


def forecast_horizon(steps: list[int], lens: list[float]) -> dict:
    """Linear-extrapolation horizon: at which step does len(t) = len(0)?

    Returns the predicted step t* if slope > 0, else a sentinel indicating
    that compression is unbounded (slope <= 0).
    """
    x = np.asarray(steps, float)
    y = np.asarray(lens, float)
    if len(x) < 2 or np.std(x) == 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "t_star": float("nan"), "predicted": "insufficient"}
    b, a = np.polyfit(x, y, 1)
    y0 = y[0]
    if b > 0:
        t_star = (y0 - a) / b
        return {"slope": float(b), "intercept": float(a),
                "t_star": float(t_star), "predicted": "horizon"}
    return {"slope": float(b), "intercept": float(a),
            "t_star": float("inf"), "predicted": "no_horizon"}


def first_diff_corr(steps: list[int], lens: list[float],
                    rewards: list[float]) -> tuple[float, float, int]:
    """ρ(ΔL, ΔR) per run, where ΔL_t = L_t - L_{t-1}, ΔR_t = R_t - R_{t-1}."""
    if len(steps) < 3:
        return float("nan"), float("nan"), 0
    dL = np.diff(np.asarray(lens, float))
    dR = np.diff(np.asarray(rewards, float))
    if np.std(dL) == 0 or np.std(dR) == 0:
        return float("nan"), float("nan"), int(len(dL))
    rho, p = stats.spearmanr(dL, dR)
    return float(rho), float(p), int(len(dL))


# ----------------------------- main pipeline --------------------------------

def main() -> int:
    runs = load_runs()

    # S1: per-window ρ
    windows = [8, 10, 12]
    s1_rows = []
    for r in runs:
        steps = [s["step"] for s in r["step_log"]]
        rewards = [float(s["mean_reward"]) for s in r["step_log"]]
        lens = [float(s["mean_comp_len"]) for s in r["step_log"]]
        for w in windows:
            rhos = windowed_spearman(lens, rewards, w)
            if not rhos:
                continue
            n = len(rhos)
            third = max(1, n // 3)
            early = rhos[:third]
            mid = rhos[third : 2 * third]
            late = rhos[2 * third :]
            s1_rows.append({
                "task": r["task"],
                "algo": r["algo"],
                "seed": r["seed"],
                "model": r["model"],
                "win": w,
                "n_windows": n,
                "mean_rho_all": round(float(np.mean(rhos)), 4),
                "mean_rho_early": round(float(np.mean(early)), 4) if early else float("nan"),
                "mean_rho_mid": round(float(np.mean(mid)), 4) if mid else float("nan"),
                "mean_rho_late": round(float(np.mean(late)), 4) if late else float("nan"),
                "frac_late_nonneg": round(float(np.mean([1 if x >= 0 else 0 for x in late])), 4) if late else float("nan"),
            })

    s1_cols = list(s1_rows[0].keys())
    s1_tsv = RES / "length_bias_iter24_windows.tsv"
    with s1_tsv.open("w") as f:
        f.write("\t".join(s1_cols) + "\n")
        for r in s1_rows:
            f.write("\t".join(str(r[c]) for c in s1_cols) + "\n")
    print(f"wrote {s1_tsv} ({len(s1_rows)} rows)")

    # S2: sign-flip test (per (task, algo, win))
    s2_rows = []
    keys = sorted({(r["task"], r["algo"], r["win"]) for r in s1_rows})
    for (task, algo, w) in keys:
        sub = [r for r in s1_rows if r["task"] == task and r["algo"] == algo and r["win"] == w]
        late_nonneg = [r["frac_late_nonneg"] for r in sub if not math.isnan(r["frac_late_nonneg"])]
        late_rhos = [r["mean_rho_late"] for r in sub if not math.isnan(r["mean_rho_late"])]
        early_rhos = [r["mean_rho_early"] for r in sub if not math.isnan(r["mean_rho_early"])]
        # paired sign-flip test: under H0 mean(late_nonneg) = 0.5
        # one-sample t on (frac_late_nonneg - 0.5)
        diffs = [v - 0.5 for v in late_nonneg]
        if len(diffs) >= 2:
            t_stat, p_two = stats.ttest_1samp(diffs, 0.0)
        else:
            t_stat, p_two = float("nan"), float("nan")
        # drift = mean(late) - mean(early) per seed
        drift_per_seed = []
        for r in sub:
            if not math.isnan(r["mean_rho_early"]) and not math.isnan(r["mean_rho_late"]):
                drift_per_seed.append(r["mean_rho_late"] - r["mean_rho_early"])
        if len(drift_per_seed) >= 2:
            mean_drift = float(np.mean(drift_per_seed))
            # paired t-test that drift != 0
            t_drift, p_drift = stats.ttest_1samp(drift_per_seed, 0.0)
        else:
            mean_drift = float("nan")
            t_drift, p_drift = float("nan"), float("nan")
        s2_rows.append({
            "task": task,
            "algo": algo,
            "win": w,
            "n_seeds": len(sub),
            "mean_late_nonneg_frac": round(float(np.mean(late_nonneg)), 4) if late_nonneg else float("nan"),
            "t_vs_half": round(float(t_stat), 4) if not math.isnan(t_stat) else float("nan"),
            "p_vs_half": round(float(p_two), 4) if not math.isnan(p_two) else float("nan"),
            "mean_rho_early": round(float(np.mean(early_rhos)), 4) if early_rhos else float("nan"),
            "mean_rho_late": round(float(np.mean(late_rhos)), 4) if late_rhos else float("nan"),
            "mean_drift_late_minus_early": round(float(mean_drift), 4) if not math.isnan(mean_drift) else float("nan"),
            "t_drift": round(float(t_drift), 4) if not math.isnan(t_drift) else float("nan"),
            "p_drift": round(float(p_drift), 4) if not math.isnan(p_drift) else float("nan"),
        })

    s2_cols = list(s2_rows[0].keys())
    s2_tsv = RES / "length_bias_iter24_signflip.tsv"
    with s2_tsv.open("w") as f:
        f.write("\t".join(s2_cols) + "\n")
        for r in s2_rows:
            f.write("\t".join(str(r[c]) for c in s2_cols) + "\n")
    print(f"wrote {s2_tsv} ({len(s2_rows)} rows)")

    # S3: forecast horizon
    s3_rows = []
    for r in runs:
        steps = [s["step"] for s in r["step_log"]]
        lens = [float(s["mean_comp_len"]) for s in r["step_log"]]
        f = forecast_horizon(steps, lens)
        s3_rows.append({
            "task": r["task"],
            "algo": r["algo"],
            "seed": r["seed"],
            "model": r["model"],
            "n_steps": len(steps),
            "len_first": round(float(lens[0]), 4),
            "len_last": round(float(lens[-1]), 4),
            "len_slope_per_step": round(f["slope"], 6) if not math.isnan(f["slope"]) else float("nan"),
            "predicted_t_star": round(f["t_star"], 2) if not (math.isnan(f["t_star"]) or math.isinf(f["t_star"])) else str(f["predicted"]),
            "horizon_class": f["predicted"],
        })

    s3_cols = list(s3_rows[0].keys())
    s3_tsv = RES / "length_bias_iter24_forecast.tsv"
    with s3_tsv.open("w") as f:
        f.write("\t".join(s3_cols) + "\n")
        for r in s3_rows:
            f.write("\t".join(str(r[c]) for c in s3_cols) + "\n")
    print(f"wrote {s3_tsv} ({len(s3_rows)} rows)")

    # S4: first-difference coupling
    s4_rows = []
    for r in runs:
        steps = [s["step"] for s in r["step_log"]]
        rewards = [float(s["mean_reward"]) for s in r["step_log"]]
        lens = [float(s["mean_comp_len"]) for s in r["step_log"]]
        rho, p, n = first_diff_corr(steps, lens, rewards)
        s4_rows.append({
            "task": r["task"],
            "algo": r["algo"],
            "seed": r["seed"],
            "model": r["model"],
            "n_diffs": n,
            "rho_dL_dR": round(rho, 4) if not math.isnan(rho) else float("nan"),
            "p_dL_dR": round(p, 4) if not math.isnan(p) else float("nan"),
        })

    s4_cols = list(s4_rows[0].keys())
    s4_tsv = RES / "length_bias_iter24_diffcorr.tsv"
    with s4_tsv.open("w") as f:
        f.write("\t".join(s4_cols) + "\n")
        for r in s4_rows:
            f.write("\t".join(str(r[c]) for c in s4_cols) + "\n")
    print(f"wrote {s4_tsv} ({len(s4_rows)} rows)")

    # Summary: per (task, algo) aggregates
    summary_rows = []
    for task in sorted({r["task"] for r in runs}):
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in runs if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            # collect dr_grpo_minus_grpo diff for the per-seed metrics
            sub_s4 = [r for r in s4_rows if r["task"] == task and r["algo"] == algo]
            dL_dR = [r["rho_dL_dR"] for r in sub_s4 if not math.isnan(r["rho_dL_dR"])]
            sub_s3 = [r for r in s3_rows if r["task"] == task and r["algo"] == algo]
            slopes = [r["len_slope_per_step"] for r in sub_s3 if not isinstance(r["len_slope_per_step"], str) and not math.isnan(r["len_slope_per_step"])]
            n_with_horizon = sum(1 for r in sub_s3 if r["horizon_class"] == "horizon")
            # mean late nonneg frac at win=10 (canonical window)
            sub_s2_w10 = [r for r in s2_rows if r["task"] == task and r["algo"] == algo and r["win"] == 10]
            late_nn = [r["mean_late_nonneg_frac"] for r in sub_s2_w10 if not math.isnan(r["mean_late_nonneg_frac"])]
            drift_w10 = [r["mean_drift_late_minus_early"] for r in sub_s2_w10 if not math.isnan(r["mean_drift_late_minus_early"])]
            summary_rows.append({
                "task": task,
                "algo": algo,
                "n_seeds": len(sub),
                "mean_rho_dL_dR": round(float(np.mean(dL_dR)), 4) if dL_dR else float("nan"),
                "sd_rho_dL_dR": round(float(np.std(dL_dR, ddof=1)) if len(dL_dR) > 1 else 0.0, 4),
                "mean_len_slope_per_step": round(float(np.mean(slopes)), 6) if slopes else float("nan"),
                "n_with_horizon": n_with_horizon,
                "mean_late_nonneg_w10": round(float(np.mean(late_nn)), 4) if late_nn else float("nan"),
                "mean_drift_w10": round(float(np.mean(drift_w10)), 4) if drift_w10 else float("nan"),
            })

    s_cols = list(summary_rows[0].keys())
    sum_tsv = RES / "length_bias_iter24_summary.tsv"
    with sum_tsv.open("w") as f:
        f.write("\t".join(s_cols) + "\n")
        for r in summary_rows:
            f.write("\t".join(str(r[c]) for c in s_cols) + "\n")
    print(f"wrote {sum_tsv} ({len(summary_rows)} rows)")

    # ---- figure (4 panels) ----
    make_figure(runs, s1_rows, s2_rows, s3_rows, s4_rows)

    # ---- terminal summary ----
    print()
    print("=== iter24 drift/forecast summary ===")
    for r in summary_rows:
        print(f"  {r['task']:36s} {r['algo']:8s} "
              f"rho_dL_dR={r['mean_rho_dL_dR']:+.3f} "
              f"slope={r['mean_len_slope_per_step']:+.5f} "
              f"n_horizon={r['n_with_horizon']}/{r['n_seeds']} "
              f"late_nonneg_w10={r['mean_late_nonneg_w10']:.3f} "
              f"drift_w10={r['mean_drift_w10']:+.3f}")
    print()
    print("=== sign-flip test (S2): win=10 ===")
    for r in [r for r in s2_rows if r["win"] == 10]:
        print(f"  {r['task']:36s} {r['algo']:8s} "
              f"late_nonneg={r['mean_late_nonneg_frac']:.3f} "
              f"(H0=0.5, p={r['p_vs_half']:.3f}) "
              f"drift={r['mean_drift_late_minus_early']:+.3f} "
              f"(p_drift={r['p_drift']:.3f})")

    return 0


# ----------------------------- figure ---------------------------------------

def make_figure(runs, s1_rows, s2_rows, s3_rows, s4_rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: mean late-window ρ for w=10, per (task, algo) with error bar
    ax = axes[0, 0]
    s1_w10 = [r for r in s1_rows if r["win"] == 10]
    grp = {}
    for r in s1_w10:
        key = (r["task"], r["algo"])
        grp.setdefault(key, {"early": [], "mid": [], "late": []})
        if not math.isnan(r["mean_rho_early"]):
            grp[key]["early"].append(r["mean_rho_early"])
        if not math.isnan(r["mean_rho_mid"]):
            grp[key]["mid"].append(r["mean_rho_mid"])
        if not math.isnan(r["mean_rho_late"]):
            grp[key]["late"].append(r["mean_rho_late"])
    keys = sorted(grp.keys())
    x_pos = np.arange(len(keys))
    width = 0.27
    early_means = [float(np.mean(grp[k]["early"])) if grp[k]["early"] else 0 for k in keys]
    early_sds = [float(np.std(grp[k]["early"], ddof=1)) if len(grp[k]["early"]) > 1 else 0 for k in keys]
    mid_means = [float(np.mean(grp[k]["mid"])) if grp[k]["mid"] else 0 for k in keys]
    mid_sds = [float(np.std(grp[k]["mid"], ddof=1)) if len(grp[k]["mid"]) > 1 else 0 for k in keys]
    late_means = [float(np.mean(grp[k]["late"])) if grp[k]["late"] else 0 for k in keys]
    late_sds = [float(np.std(grp[k]["late"], ddof=1)) if len(grp[k]["late"]) > 1 else 0 for k in keys]
    ax.bar(x_pos - width, early_means, width, yerr=early_sds, label="early windows", color="#88c4f5", capsize=3)
    ax.bar(x_pos, mid_means, width, yerr=mid_sds, label="mid windows", color="#5e9bd5", capsize=3)
    ax.bar(x_pos + width, late_means, width, yerr=late_sds, label="late windows", color="#1f4e9d", capsize=3)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xticks(x_pos)
    short = [f"{k[0].split('_')[0][:6]}\n{k[0].split('_')[-1]}\n{k[1]}" for k in keys]
    ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel(r"windowed Spearman $\rho_w(\mathrm{len}, R)$  (w=10)")
    ax.set_title("A. windowed ρ(len, R) drift (early→mid→late)")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: sign-flip — late_nonneg_frac per (task, algo)
    ax = axes[0, 1]
    s2_w10 = [r for r in s2_rows if r["win"] == 10]
    keys2 = sorted({(r["task"], r["algo"]) for r in s2_w10})
    x_pos2 = np.arange(len(keys2))
    vals = [r["mean_late_nonneg_frac"] for r in s2_w10 if (r["task"], r["algo"]) in keys2]
    vals = [v for v in vals if not math.isnan(v)]
    colors = []
    for r in s2_w10:
        if (r["task"], r["algo"]) not in keys2:
            continue
        colors.append("#d62728" if r["p_vs_half"] < 0.05 else "#7f7f7f")
    ax.bar(x_pos2, vals[: len(colors)], color=colors[: len(x_pos2)])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="H0: 0.5 (no drift)")
    ax.set_xticks(x_pos2)
    ax.set_xticklabels([f"{k[0].split('_')[0][:6]}\n{k[0].split('_')[-1]}\n{k[1]}" for k in keys2], fontsize=7)
    ax.set_ylabel("frac of late windows with ρ ≥ 0")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("B. sign-flip test (S2) — drift toward positive coupling")
    ax.legend(fontsize=8, loc="upper right")

    # Panel C: forecast horizon (mean slope per (task, algo))
    ax = axes[1, 0]
    grp3 = {}
    for r in s3_rows:
        key = (r["task"], r["algo"])
        grp3.setdefault(key, []).append(r["len_slope_per_step"])
    keys3 = sorted(grp3.keys())
    x_pos3 = np.arange(len(keys3))
    means3 = [float(np.mean([s for s in grp3[k] if not (isinstance(s, str) or math.isnan(s))])) for k in keys3]
    sds3 = [float(np.std([s for s in grp3[k] if not (isinstance(s, str) or math.isnan(s))], ddof=1)) if len([s for s in grp3[k] if not (isinstance(s, str) or math.isnan(s))]) > 1 else 0 for k in keys3]
    ax.bar(x_pos3, means3, yerr=sds3, color=["#ff7f0e" if m > 0 else "#2ca02c" for m in means3], capsize=4)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xticks(x_pos3)
    ax.set_xticklabels([f"{k[0].split('_')[0][:6]}\n{k[0].split('_')[-1]}\n{k[1]}" for k in keys3], fontsize=7)
    ax.set_ylabel("len slope (tokens / step)")
    ax.set_title("C. forecast horizon — linear len-vs-step slope")

    # Panel D: first-difference ρ(ΔL, ΔR) per (task, algo)
    ax = axes[1, 1]
    grp4 = {}
    for r in s4_rows:
        key = (r["task"], r["algo"])
        grp4.setdefault(key, []).append(r["rho_dL_dR"])
    keys4 = sorted(grp4.keys())
    x_pos4 = np.arange(len(keys4))
    means4 = [float(np.mean([s for s in grp4[k] if not math.isnan(s)])) for k in keys4]
    sds4 = [float(np.std([s for s in grp4[k] if not math.isnan(s)], ddof=1)) if len([s for s in grp4[k] if not math.isnan(s)]) > 1 else 0 for k in keys4]
    colors4 = ["#d62728" if m > 0 else "#1f77b4" for m in means4]
    ax.bar(x_pos4, means4, yerr=sds4, color=colors4, capsize=4)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xticks(x_pos4)
    ax.set_xticklabels([f"{k[0].split('_')[0][:6]}\n{k[0].split('_')[-1]}\n{k[1]}" for k in keys4], fontsize=7)
    ax.set_ylabel(r"first-diff Spearman $\rho(\Delta L, \Delta R)$")
    ax.set_title("D. detrended coupling — does ΔL follow ΔR?")

    fig.suptitle("Pillar 4 / Iter 24 — drift & forecast test of the Dr.GRPO verbosity-trap signature", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf = FIGS / "length_bias_iter24.pdf"
    png = FIGS / "length_bias_iter24.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    shutil.copyfile(pdf, PAPER_FIGS / "length_bias_iter24.pdf")
    print(f"wrote {pdf} and {png} (copied to {PAPER_FIGS / 'length_bias_iter24.pdf'})")


if __name__ == "__main__":
    sys.exit(main())