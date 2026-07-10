#!/usr/bin/env python3
"""
length_bias_iter32_fig.py — 4-panel synthesis figure for iter 32.

Panel A: rho(L_t, R_{t+k}) for k in [-3,-2,-1,0,1,2,3], per (task, algo).
Panel B: per-seed dominant_lag distribution (histogram).
Panel C: per-run CV_L per (task, algo) with bootstrap 95% CIs.
Panel D: per-(task, algo) rho(k+1) - rho(k-1) point with bootstrap CI,
        demonstrating whether L leads R (positive) or R leads L (negative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"

LAGS = [-3, -2, -1, 0, 1, 2, 3]
COLORS = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}


def _agg_CV(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, algo), g in profile.groupby(["task", "algo"]):
        cvs = g["CV_L"].dropna().values
        if len(cvs) < 2:
            rows.append({"task": task, "algo": algo, "n": len(cvs),
                         "CV_mean": float(np.mean(cvs)),
                         "CV_lo": float(np.mean(cvs)),
                         "CV_hi": float(np.mean(cvs))})
            continue
        rng = np.random.default_rng(20260702)
        boots = np.array([np.mean(rng.choice(cvs, size=len(cvs), replace=True))
                          for _ in range(2000)])
        rows.append({"task": task, "algo": algo, "n": int(len(cvs)),
                     "CV_mean": float(np.mean(cvs)),
                     "CV_lo": float(np.percentile(boots, 2.5)),
                     "CV_hi": float(np.percentile(boots, 97.5))})
    return pd.DataFrame(rows).sort_values(["task", "algo"]).reset_index(drop=True)


def _lead_lag_pt(profile: pd.DataFrame) -> pd.DataFrame:
    """Per (task, algo) point estimate of rho(k+1)-rho(k-1)."""
    rows = []
    for (task, algo), g in profile.groupby(["task", "algo"]):
        diffs = (g["rho_k+1"] - g["rho_k-1"]).dropna().values
        if len(diffs) < 2:
            rows.append({"task": task, "algo": algo,
                         "mean": float(np.mean(diffs)) if len(diffs) else np.nan,
                         "lo": float(np.mean(diffs)) if len(diffs) else np.nan,
                         "hi": float(np.mean(diffs)) if len(diffs) else np.nan,
                         "n": int(len(diffs))})
            continue
        rng = np.random.default_rng(20260702)
        boots = np.array([np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                          for _ in range(2000)])
        rows.append({"task": task, "algo": algo,
                     "mean": float(np.mean(diffs)),
                     "lo": float(np.percentile(boots, 2.5)),
                     "hi": float(np.percentile(boots, 97.5)),
                     "n": int(len(diffs))})
    return pd.DataFrame(rows).sort_values(["task", "algo"]).reset_index(drop=True)


def main():
    profile = pd.read_csv(RESULTS / "length_bias_iter32_lag_profile.tsv", sep="\t")
    summary = pd.read_csv(RESULTS / "length_bias_iter32_lag_summary.tsv", sep="\t")
    cv = _agg_CV(profile)
    ll = _lead_lag_pt(profile)

    cv.to_csv(RESULTS / "length_bias_iter32_cv.tsv", sep="\t", index=False)
    ll.to_csv(RESULTS / "length_bias_iter32_lead_lag.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: rho(L_t, R_{t+k}) vs k
    ax = axes[0, 0]
    keys = sorted(set(zip(profile["task"], profile["algo"])))
    for (task, algo) in keys:
        g = profile[(profile["task"] == task) & (profile["algo"] == algo)]
        means = [g[f"rho_k{k:+d}"].mean() for k in LAGS]
        ax.plot(LAGS, means, marker="o", color=COLORS[algo],
                label=f"{task}/{algo}")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("lag k  (R_{t+k} vs L_t; +ve k ⇒ L leads R)")
    ax.set_ylabel("mean rho(L_t, R_{t+k})")
    ax.set_title("(A) Lag-k Spearman profile per (task, algo)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: dominant_lag distribution
    ax = axes[0, 1]
    bins = np.arange(-3.5, 4.5, 1)
    for (task, algo) in keys:
        g = profile[(profile["task"] == task) & (profile["algo"] == algo)]
        ax.hist(g["dominant_lag"], bins=bins, alpha=0.45,
                color=COLORS[algo], label=f"{task}/{algo} (n={len(g)})",
                edgecolor="black")
    ax.set_xlabel("dominant lag (argmax_k |rho(L, R_{t+k})|)")
    ax.set_ylabel("# seeds")
    ax.set_title("(B) Dominant lag per seed by cell")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: per-run CV_L with bootstrap CI
    ax = axes[1, 0]
    cell_order = sorted(cv["task"].unique())
    xpos = np.arange(len(cell_order) * 2)
    labels = []
    pos = []
    i = 0
    for task in cell_order:
        for algo in ["grpo", "dr_grpo"]:
            sub = cv[(cv["task"] == task) & (cv["algo"] == algo)]
            if len(sub) == 0:
                continue
            ax.errorbar(i, sub["CV_mean"].iloc[0],
                        yerr=[[sub["CV_mean"].iloc[0] - sub["CV_lo"].iloc[0]],
                              [sub["CV_hi"].iloc[0] - sub["CV_mean"].iloc[0]]],
                        fmt="o", color=COLORS[algo], capsize=4)
            labels.append(f"{task}/{algo}")
            pos.append(i)
            i += 1
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CV_L = std(L)/mean(L)")
    ax.set_title("(C) Length stability per cell (bootstrap 95% CI)")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel D: rho(k+1)-rho(k-1)
    ax = axes[1, 1]
    i = 0
    for task in cell_order:
        for algo in ["grpo", "dr_grpo"]:
            sub = ll[(ll["task"] == task) & (ll["algo"] == algo)]
            if len(sub) == 0:
                continue
            ax.errorbar(i, sub["mean"].iloc[0],
                        yerr=[[sub["mean"].iloc[0] - sub["lo"].iloc[0]],
                              [sub["hi"].iloc[0] - sub["mean"].iloc[0]]],
                        fmt="o", color=COLORS[algo], capsize=4)
            i += 1
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rho(L_t, R_{t+1}) - rho(L_t, R_{t-1})  (>0 ⇒ L leads R)")
    ax.set_title("(D) Lead-lag point estimate with bootstrap 95% CI")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Iter 32 — Length-vs-Reward temporal lag cross-correlation\n"
                 "$\\rho(L_t, R_{t+k})$ for k in [-3, 3]; Dr.GRPO relabelled red",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = FIGURES / "length_bias_iter32.pdf"
    out_png = FIGURES / "length_bias_iter32.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print()
    print("=== CV ===")
    print(cv.to_string(index=False))
    print()
    print("=== lead-lag ===")
    print(ll.to_string(index=False))


if __name__ == "__main__":
    main()
