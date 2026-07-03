#!/usr/bin/env python3
"""Iter 34 — ZVF cross-pillar phase discrimination figure.

Two-panel PDF that visualises the iter34 findings:

    Panel A: per-phase ZVF-feature profile (zvf_level,
             zvf_direction, zvf_discriminator) with mean +- sd.
    Panel B: 4x4 LOO confusion matrix as a heatmap with the
             per-class recovery annotated in each cell.

Inputs:
    experiments/results/zvf_iter34_discriminant.tsv
    experiments/results/zvf_iter34_confusion.tsv
    experiments/results/zvf_iter34_summary.json

Output:
    figures/zvf_iter34.pdf
    figures/zvf_iter34.png

Source: scripts/zvf_iter34_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results"
FIG = ROOT / "figures"


PHASE_COLOURS = {
    "plateau": "#4C72B0",
    "saturation": "#55A868",
    "drift": "#C44E52",
    "collapse": "#8172B3",
}


def panel_a(ax, disc: pd.DataFrame) -> None:
    features = ["zvf_level", "zvf_direction", "zvf_discriminator"]
    labels = ["ZVF level", "ZVF direction", "ZVF discriminator"]
    phases = ["plateau", "saturation", "drift", "collapse"]
    x = np.arange(len(features))
    width = 0.2
    for i, phase in enumerate(phases):
        sub = disc[disc["phase"] == phase].set_index("feature")
        means = [sub.loc[f, "mean"] for f in features]
        sds = [sub.loc[f, "sd"] for f in features]
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width=width,
            yerr=sds,
            color=PHASE_COLOURS[phase],
            label=phase,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("feature value")
    ax.set_title("Per-phase ZVF-feature profile (mean +- sd)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def panel_b(ax, confusion: pd.DataFrame) -> None:
    cm = confusion.to_numpy()
    total = cm.sum(axis=1, keepdims=True)
    pct = np.where(total > 0, cm / total, 0.0)
    n = cm.shape[0]
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(confusion.columns, rotation=30, ha="right")
    ax.set_yticklabels(confusion.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("LOO confusion (row-normalised)")
    for i in range(n):
        for j in range(n):
            colour = "white" if pct[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct[i, j]:.0%})",
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    disc = pd.read_csv(OUT / "zvf_iter34_discriminant.tsv", sep="\t")
    confusion = pd.read_csv(OUT / "zvf_iter34_confusion.tsv", sep="\t", index_col=0)
    summary = json.loads((OUT / "zvf_iter34_summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel_a(axes[0], disc)
    panel_b(axes[1], confusion)
    fig.suptitle(
        f"ZVF cross-pillar phase discrimination (n=12 anchors, "
        f"LOO acc={summary['loo_accuracy']:.3f}, chance=0.250)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "zvf_iter34.pdf")
    fig.savefig(FIG / "zvf_iter34.png", dpi=130)
    print(f"wrote {FIG / 'zvf_iter34.pdf'} and .png")


if __name__ == "__main__":
    main()