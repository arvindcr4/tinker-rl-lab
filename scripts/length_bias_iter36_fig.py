#!/usr/bin/env python3
"""
length_bias_iter36_fig.py — 4-panel synthesis figure for iter36 joint saturation.

Reads the iter36 TSVs and produces figures/length_bias_iter36.{pdf,png}.
Panel layout (4 panels in a 2x2 grid):
  TL: arithmetic — overlay (L, R) saturation fits for one seed (GRPO + Dr.GRPO)
  TR: gsm8k_cot — same overlay (illustrates L non-saturation)
  BL: lam_ratio bar plot per (task, algo) with bootstrap CI
  BR: resid_rho per (task, algo) showing anti-trap on GSM8K
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "figures"
DRGRPO = ROOT / "experiments" / "results" / "drgrpo_vs_grpo.json"
GSM8K = ROOT / "experiments" / "results" / "drgrpo_gsm8k_cot_full.json"
PER_RUN = ROOT / "experiments" / "results" / "length_bias_iter36_per_run.tsv"
GR_VS_DR = ROOT / "experiments" / "results" / "length_bias_iter36_grpo_vs_drgrpo.tsv"


def load_runs(path):
    with open(path) as f:
        return json.load(f)["runs"]


def saturation(t, a, b, lam):
    return a + (b - a) * (1.0 - np.exp(-lam * np.asarray(t, dtype=float)))


def main():
    runs_a = load_runs(DRGRPO)
    runs_g = load_runs(GSM8K)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # ===== TL: arithmetic overlay =====
    ax = axes[0, 0]
    seed = 42
    for algo, color, marker in [("grpo", "tab:blue", "o"), ("dr_grpo", "tab:red", "s")]:
        runs = [r for r in runs_a if r["algo"] == algo and r["seed"] == seed]
        if not runs:
            continue
        sl = runs[0]["step_log"]
        t = np.array([s["step"] for s in sl])
        R = np.array([s["mean_reward"] for s in sl])
        L = np.array([s["mean_comp_len"] for s in sl])
        ax.plot(t, R, color=color, marker=marker, ms=3, alpha=0.7, label=f"{algo} R")
        ax.plot(t, L, color=color, ls="--", alpha=0.5, label=f"{algo} L (right axis)")
        # find fit params from per-run TSV
    ax.set_xlabel("step")
    ax.set_ylabel("reward (left axis)")
    ax2 = ax.twinx()
    for algo, color in [("grpo", "tab:blue"), ("dr_grpo", "tab:red")]:
        runs = [r for r in runs_a if r["algo"] == algo and r["seed"] == seed]
        if not runs:
            continue
        sl = runs[0]["step_log"]
        t = np.array([s["step"] for s in sl])
        L = np.array([s["mean_comp_len"] for s in sl])
        ax2.plot(t, L, color=color, ls="--", alpha=0.6)
    ax2.set_ylabel("mean_comp_len (right axis)")
    ax.set_title(f"Arithmetic (Qwen2.5-0.5B, seed={seed}): R (solid) and L (dashed) trajectories")
    ax.legend(loc="upper left", fontsize=7)

    # ===== TR: gsm8k overlay =====
    ax = axes[0, 1]
    seed = 456
    for algo, color, marker in [("grpo", "tab:blue", "o"), ("dr_grpo", "tab:red", "s")]:
        runs = [r for r in runs_g if r["algo"] == algo and r["seed"] == seed]
        if not runs:
            continue
        sl = runs[0]["step_log"]
        t = np.array([s["step"] for s in sl])
        R = np.array([s["mean_reward"] for s in sl])
        L = np.array([s["mean_comp_len"] for s in sl])
        ax.plot(t, R, color=color, marker=marker, ms=3, alpha=0.7, label=f"{algo} R")
        ax2 = ax.twinx()
        ax2.plot(t, L, color=color, ls="--", alpha=0.6, label=f"{algo} L")
    ax.set_xlabel("step")
    ax.set_title(f"GSM8K CoT (Qwen2.5-1.5B, seed={seed})")
    ax.legend(loc="upper right", fontsize=7)

    # ===== BL: lam_ratio bar plot =====
    ax = axes[1, 0]
    gr_vs_dr = []
    with open(GR_VS_DR) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split("\t")
            gr_vs_dr.append(dict(
                task=parts[0], grpo_mean=float(parts[2]), drgrpo_mean=float(parts[4]),
                diff=float(parts[6]), diff_lo=float(parts[7]), diff_hi=float(parts[8]),
            ))
    x = np.arange(len(gr_vs_dr))
    width = 0.35
    ax.bar(x - width / 2, [r["grpo_mean"] for r in gr_vs_dr], width,
           color="tab:blue", label="GRPO", alpha=0.8)
    ax.bar(x + width / 2, [r["drgrpo_mean"] for r in gr_vs_dr], width,
           color="tab:red", label="Dr.GRPO", alpha=0.8)
    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in gr_vs_dr], rotation=10)
    ax.set_ylabel("$\\lambda_L / \\lambda_R$")
    ax.set_title("Saturation-rate ratio (1.0 = joint saturation)")
    ax.legend(fontsize=8)

    # ===== BR: resid_rho per (task, algo) =====
    ax = axes[1, 1]
    per_run = []
    with open(PER_RUN) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split("\t")
            per_run.append(dict(
                task=parts[0], algo=parts[1],
                resid_rho=float(parts[14]),
            ))
    # aggregate
    tasks = ["arithmetic_easy", "gsm8k_cot"]
    algos = ["grpo", "dr_grpo"]
    grid = np.zeros((len(tasks), len(algos)))
    for i, task in enumerate(tasks):
        for j, algo in enumerate(algos):
            vals = [r["resid_rho"] for r in per_run if r["task"] == task and r["algo"] == algo]
            grid[i, j] = float(np.median(vals)) if vals else 0.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(algos)):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Median residual cross-correlation $\\rho(\\epsilon_R, \\epsilon_L)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_pdf = FIG_DIR / "length_bias_iter36.pdf"
    out_png = FIG_DIR / "length_bias_iter36.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()