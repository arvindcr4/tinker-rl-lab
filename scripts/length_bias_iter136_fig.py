#!/usr/bin/env python3
"""Iter 136 -- Figure: 4-panel summary of step-level trajectory coupling.

Panel A: Per-task, per-hypothesis effect size (paired Cohen's d, DR - GR)
         with 95% bootstrap CI.  Sign convention: positive = DR favourable.
Panel B: Sign-test summary: 8/8 paired tests in predicted direction,
         global binomial p=0.0039.
Panel C: Late-training efficiency (H3) per (algo, seed, task) bar plot
         with task split.
Panel D: ZVF-length co-movement (H4) per (algo, seed, task) scatter
         of rho(ΔZ, ΔL) with task split.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "figures")


def read_tsv(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def bootstrap_paired_diff_ci(gr: np.ndarray, dr: np.ndarray, n_boot: int = 5000,
                            rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap 95% CI for paired mean(DR - GR)."""
    if rng is None:
        rng = np.random.default_rng(20260704)
    delta = dr - gr
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def main():
    meta = json.load(open(os.path.join(RES, "length_bias_iter136_meta.json")))
    per_run = read_tsv(os.path.join(RES, "length_bias_iter136_step_coupling.tsv"))
    paired = read_tsv(os.path.join(RES, "length_bias_iter136_paired_tests.tsv"))

    tasks = ["arithmetic_easy", "gsm8k_cot"]
    task_to_seeds = {t: sorted({int(r["seed"]) for r in per_run if r["task"] == t})
                     for t in tasks}

    # Build per-task, per-hypothesis arrays
    hyp_to_keys = {
        "H1_ΔR-ΔL coupling": ("abs_rho_dR_dL", "smaller"),
        "H2_length trendiness": ("abs_rho_len_lag1", "smaller"),
        "H3_late efficiency": ("late_eff", "larger"),
        "H4_ZVF-length link": ("rho_dZ_dL", "lessneg"),
    }
    by_seed: dict[tuple[str, str, str], tuple[float, float]] = {}
    for r in per_run:
        for h_name, (k, _) in hyp_to_keys.items():
            by_seed[(r["task"], r["algo"], h_name)] = float(r[k])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel A: effect sizes (paired Cohen's d) per task per hypothesis ---
    ax = axes[0, 0]
    bar_x = np.arange(len(hyp_to_keys))
    width = 0.35
    rng = np.random.default_rng(20260704)
    for i, (task, color) in enumerate(zip(tasks, ["#1f77b4", "#d62728"])):
        effects = []
        ci_lo = []
        ci_hi = []
        for h_name, _ in hyp_to_keys.items():
            seeds = task_to_seeds[task]
            gr = np.array([by_seed[(task, "grpo", h_name)] for s in seeds], dtype=float)
            dr = np.array([by_seed[(task, "dr_grpo", h_name)] for s in seeds], dtype=float)
            delta = dr - gr
            d = float(delta.mean() / (np.std(delta, ddof=1) + 1e-9))
            lo, hi = bootstrap_paired_diff_ci(gr, dr, rng=rng)
            effects.append(d)
            # CI on the raw delta in standardized units
            d_lo = lo / (np.std(delta, ddof=1) + 1e-9)
            d_hi = hi / (np.std(delta, ddof=1) + 1e-9)
            ci_lo.append(d_lo)
            ci_hi.append(d_hi)
        ax.bar(bar_x + (i - 0.5) * width, effects, width=width, color=color, alpha=0.7,
               label=task)
        # Add CI as error bars
        err_lo = [e - lo for e, lo in zip(effects, ci_lo)]
        err_hi = [hi - e for e, hi in zip(effects, ci_hi)]
        ax.errorbar(bar_x + (i - 0.5) * width, effects,
                    yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(list(hyp_to_keys.keys()), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Paired Cohen's d (DR - GR)")
    ax.set_title("A. Effect size per (task, hypothesis); error bars = 95% boot CI")
    ax.legend(loc="upper right", fontsize=9)

    # --- Panel B: sign-test summary bar ---
    ax = axes[0, 1]
    summary = read_tsv(os.path.join(RES, "length_bias_iter136_cross_task.tsv"))
    n_in_dir = int(next(r["n_dir_positive"] for r in summary
                        if r["hypothesis"] == "GLOBAL_sign_test_over_all_8_paired"))
    n_total = 8
    ax.bar(["favours Dr.GR", "favours GR"], [n_in_dir, n_total - n_in_dir],
           color=["#d62728", "#1f77b4"], alpha=0.7)
    ax.set_ylabel("count of paired tests")
    ax.set_ylim(0, n_total + 1)
    ax.set_title(f"B. Global sign test: {n_in_dir}/{n_total} in predicted direction; "
                 f"one-sided binom p=0.0039")
    for i, v in enumerate([n_in_dir, n_total - n_in_dir]):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=11, fontweight="bold")

    # --- Panel C: H3 late efficiency per (algo, seed, task) ---
    ax = axes[1, 0]
    h_name = "H3_late efficiency"
    bar_x = np.arange(2)
    width = 0.18
    colors = ["#1f77b4", "#d62728"]
    for ti, task in enumerate(tasks):
        seeds = task_to_seeds[task]
        for ai, algo in enumerate(["grpo", "dr_grpo"]):
            vals = [by_seed[(task, algo, h_name)] for s in seeds]
            x = bar_x[ti] + (ai - 0.5) * width
            for j, v in enumerate(vals):
                ax.scatter(x, v, color=colors[ai], alpha=0.6, s=30, zorder=3)
            ax.bar(x, float(np.mean(vals)), width=width * 1.2, color=colors[ai], alpha=0.3,
                   edgecolor=colors[ai])
    ax.set_xticks(bar_x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("late-training efficiency = $\\Delta r / (|\\Delta L|+1)$")
    ax.set_title("C. H3 late-training efficiency (GR blue, Dr.GR red); dots = seeds")
    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#1f77b4", alpha=0.5, label="GRPO"),
                       Patch(facecolor="#d62728", alpha=0.5, label="Dr.GRPO")],
              loc="upper right", fontsize=9)

    # --- Panel D: H4 ZVF-length scatter ---
    ax = axes[1, 1]
    h_name = "H4_ZVF-length link"
    rng_d = np.random.default_rng(20260704)
    for ti, task in enumerate(tasks):
        seeds = task_to_seeds[task]
        for ai, algo in enumerate(["grpo", "dr_grpo"]):
            vals = [by_seed[(task, algo, h_name)] for s in seeds]
            x_jitter = ti + (ai - 0.5) * 0.18
            for v in vals:
                ax.scatter(x_jitter + rng_d.uniform(-0.04, 0.04),
                           v, color=colors[ai], alpha=0.7, s=40, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(tasks)
    ax.set_ylabel(r"$\rho(\Delta Z_t, \Delta L_t)$ over consecutive steps")
    ax.set_title("D. H4 ZVF-length co-movement; >0 = ZVF falls with length shrink")

    fig.suptitle("Iter 136 — Step-level trajectory coupling (Δreward, Δlength): "
                 "8/8 paired tests in predicted direction, global p=0.0039",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_pdf = os.path.join(FIG, "length_bias_iter136.pdf")
    out_png = os.path.join(FIG, "length_bias_iter136.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()