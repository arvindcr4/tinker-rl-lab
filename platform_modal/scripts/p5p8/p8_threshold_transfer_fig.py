#!/usr/bin/env python3
"""P8 threshold-transfer figure (companion to p8_threshold_transfer.py).

Reads the .tsv files produced by p8_threshold_transfer.py and renders
the two-panel transfer-gap figure (per-model mean ± bootstrap CI as a
function of ρ). Stdlib + matplotlib + pandas only.

Outputs
-------
experiments/results/p5p8/figures/p8_threshold_transfer.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "p5p8"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

RATES = ["release", "1.00%", "0.50%", "0.10%", "0.05%"]


def main():
    df = pd.read_csv(OUT / "p8_threshold_transfer.tsv", sep="\t")
    dfb = pd.read_csv(OUT / "p8_threshold_transfer_boot.tsv", sep="\t")
    colors = {"XGB-20raw": "#1f77b4", "XGB-24full": "#2ca02c",
              "XGB-4sensor": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax_idx, panel_rhos in enumerate([
        [10, 50, 100], [100, 200, 500]
    ]):
        ax = axes[ax_idx]
        for m in ["XGB-20raw", "XGB-24full", "XGB-4sensor"]:
            xs, means, los, his = [], [], [], []
            for r in panel_rhos:
                sub_b = dfb[(dfb["rho"] == r) & (dfb["model"] == m)]
                xs.append(r)
                means.append(sub_b["mean"].mean())
                los.append(sub_b["ci_lo"].mean())
                his.append(sub_b["ci_hi"].mean())
            ax.errorbar(xs, means,
                        yerr=[np.array(means) - np.array(los),
                              np.array(his) - np.array(means)],
                        marker="o", capsize=4, label=m, color=colors[m])
        ax.set_xscale("log")
        ax.set_xticks(panel_rhos)
        ax.set_xticklabels([str(r) for r in panel_rhos])
        ax.set_xlabel(r"$\rho = L / c_\mathrm{inv}$")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_title(r"$\rho \in$ {" + ", ".join(str(r) for r in panel_rhos) + "}")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Transfer gap (USD/dec)")
        if ax_idx == 1:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(r"P8 threshold-transfer gap: $\tau^*(\mathrm{train})$ cost on test "
                 r"$-$ $\tau^*(\mathrm{test})$ cost, $\sigma=0$")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "p8_threshold_transfer.png", dpi=140)
    fig.savefig(FIGS / "p8_threshold_transfer.pdf")
    print("wrote:", (FIGS / "p8_threshold_transfer.png").name,
          (FIGS / "p8_threshold_transfer.pdf").name)


if __name__ == "__main__":
    main()