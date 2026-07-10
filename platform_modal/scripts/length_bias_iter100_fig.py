#!/usr/bin/env python3
"""Iter 100 -- figure: bivariate-VAR(1) structural summary, GRPO vs Dr.GRPO.

Two-panel figure:
  Panel A:  cross-equation coefficient matrix  K  (GRPO | Dr.GRPO)
            with per-element 95% bootstrap CIs overlaid.
  Panel B:  forecast-error variance decomposition  FEVD(L->R | R->L)
            at horizons h = 1, 4, 8, mean per group.

Saves to ``figures/length_bias_iter100_var.pdf`` and PNG.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import length_bias_iter100 as ib100  # type: ignore


def main() -> None:
    out_dir = "experiments/results"
    out_pdf = "figures/length_bias_iter100_var.pdf"
    out_png = "figures/length_bias_iter100_var.png"
    os.makedirs("figures", exist_ok=True)

    # Collect per-run VAR metrics, grouped by (task, algo).
    runs = []
    runs += [(r, "arithmetic_easy") for r in ib100.load_step_log(ib100.DRGRPO_VS_GRPO_PATH)]
    runs += [(r, "gsm8k_cot") for r in ib100.load_step_log(ib100.DRGRPO_GSM8K_PATH)]
    by_task: dict[str, list[dict]] = {}
    for r, task in runs:
        by_task.setdefault(task, []).append(r)

    per_task_metrics: dict[str, dict[str, dict[str, list[float]]]] = {}
    for task, runs_t in by_task.items():
        for r in ib100.analyze_runs(runs_t):
            algo = r["algo"]
            per_task_metrics.setdefault(task, {}).setdefault(algo, {})
            for k in ("phi_LL", "phi_LR", "phi_RL", "phi_RR",
                      "cumul_impulse_L_to_R", "cumul_impulse_R_to_L",
                      "fevd_R_from_L_h1", "fevd_R_from_L_h4", "fevd_R_from_L_h8",
                      "fevd_L_from_R_h1", "fevd_L_from_R_h4", "fevd_L_from_R_h8"):
                per_task_metrics[task][algo].setdefault(k, []).append(float(r[k]))

    # --- Panel A: per-task VAR coefficient matrix ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    fig.suptitle("Iter 100 — Bivariate VAR(1) structural coupling (Pillar 4 / Length Bias)")
    for row, task in enumerate(["arithmetic_easy", "gsm8k_cot"]):
        ax = axes[row, 0]
        titles = ["phi_LL", "phi_LR", "phi_RL", "phi_RR"]
        x = np.arange(4)
        width = 0.35
        for j, algo in enumerate(["grpo", "dr_grpo"]):
            ys = [np.mean(per_task_metrics[task][algo][t]) for t in titles]
            errs = [1.96 * np.std(per_task_metrics[task][algo][t]) /
                    max(np.sqrt(len(per_task_metrics[task][algo][t])), 1)
                    for t in titles]
            ax.bar(x + (j - 0.5) * width, ys, width, yerr=errs,
                   label={"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}[algo], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=0, fontsize=8)
        ax.set_title(f"{task}  K coefficients", fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # --- Panel B: per-task FEVD at h=1,4,8 ----------
    for col, task in enumerate(["arithmetic_easy", "gsm8k_cot"]):
        ax = axes[col, 1]
        hs = [1, 4, 8]
        for algo, marker in [("grpo", "o"), ("dr_grpo", "s")]:
            yR = [np.mean(per_task_metrics[task][algo][f"fevd_R_from_L_h{h}"]) for h in hs]
            ax.plot(hs, yR, marker=marker, label=f"R-variance from L ({algo})")
        ax.set_xticks(hs)
        ax.set_xlabel("horizon h")
        ax.set_ylabel("FEVD share")
        ax.set_title(f"{task}  FEVD", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
