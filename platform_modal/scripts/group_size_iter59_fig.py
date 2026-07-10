#!/usr/bin/env python3
"""Iter 59 figure — Pillar 3: Equivalence-Region Estimation and Decomposition.

Two-panel PDF (~842x595 pt, A4 landscape):

Panel (a) --- Retention curve R(T) = acc(G=4)/acc(G=32) on the iso-token grid,
    with three horizontal threshold lines at R=0.85 (pragmatic equivalence),
    R=0.75 (operational equivalence), and R=0.70 (hard divergence).  Marker
    shading indicates region membership at each (T, R) cell.

Panel (b) --- Multiplicative decomposition at each budget: structural upper
    bound R_struct = (steps4/steps32) * (Y4/Y32) = 4.28 (a constant on the
    iso-token grid), empirical R(T) bars, and the implied noise residual
    R_emp/R_struct - 1.

Inputs:
  experiments/results/group_size_iter59_equivalence.tsv
  experiments/results/group_size_iter59_min_tokens.tsv
  experiments/results/group_size_iter59_decomp.tsv

Outputs:
  figures/group_size_iter59.{pdf,png}
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_equivalence():
    obs_rows = []
    with open(RES / "group_size_iter59_equivalence.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row["threshold_name"].startswith("R_observed"):
                obs_rows.append(
                    {
                        "T_M": int(row["T_min_grid_M"]),
                        "R": float(row["threshold_R"]),
                    }
                )
    return obs_rows


def load_decomp():
    out = []
    with open(RES / "group_size_iter59_decomp.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            try:
                T = int(row["T_tokens"])
            except (ValueError, TypeError):
                continue
            out.append(
                {
                    "T_M": T // 1_000_000,
                    "R_emp": float(row["empirical_R"]),
                    "R_struct": float(row["structural_R_Y_times_steps"]),
                    "noise": float(row["noise_residual"]),
                }
            )
    return out


def panel_a(ax, obs):
    Ts = [r["T_M"] for r in obs]
    Rs = [r["R"] for r in obs]
    # Plot retention curve
    ax.plot(Ts, Rs, "o-", color="#1f77b4", linewidth=2.2, markersize=10,
            label=r"Empirical $R(T) = \mathrm{acc}(G{=}4)/\mathrm{acc}(G{=}32)$")
    # Threshold lines
    thresholds = [
        (0.85, "Pragmatic equiv.", "#2ca02c", "-"),
        (0.75, "Operational equiv.", "#ff7f0e", "--"),
        (0.70, "Hard divergence", "#d62728", ":"),
    ]
    for thr, lab, col, ls in thresholds:
        ax.axhline(thr, color=col, linestyle=ls, linewidth=1.4, alpha=0.85,
                   label=f"{lab} ($R \\geq {thr:.2f}$)")
    # Shade cells above/below thresholds (vertical bands)
    for i, (T, R) in enumerate(zip(Ts, Rs)):
        col = "#2ca02c" if R >= 0.85 else ("#ff7f0e" if R >= 0.75 else
                                            ("#d62728" if R < 0.70 else "#bcbd22"))
        ax.scatter([T], [R], s=220, color=col, edgecolor="black",
                   linewidth=0.8, zorder=5)
        # Annotate R value above each point
        ax.annotate(f"{R:.3f}", xy=(T, R), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="black")
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([f"{t}M" for t in Ts], fontsize=10)
    ax.set_xlabel(r"Iso-token budget $T$ (tokens, log scale)", fontsize=11)
    ax.set_ylabel(r"Retention $R = \mathrm{acc}(G{=}4) / \mathrm{acc}(G{=}32)$",
                  fontsize=11)
    ax.set_title("(a) Equivalence regions in the $(T, R)$ plane",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)


def panel_b(ax, decomp):
    Ts = [r["T_M"] for r in decomp]
    R_emp = [r["R_emp"] for r in decomp]
    R_struct = [r["R_struct"] for r in decomp]
    noise = [r["noise"] for r in decomp]

    width = 0.32
    x = np.arange(len(Ts))

    # Bars: empirical R vs structural upper bound R_struct
    b1 = ax.bar(x - width / 2, R_emp, width, color="#1f77b4", alpha=0.85,
                edgecolor="black", linewidth=0.6,
                label=r"Empirical $R(T)$")
    b2 = ax.bar(x + width / 2, R_struct, width, color="#ff7f0e", alpha=0.55,
                edgecolor="black", linewidth=0.6,
                label=r"Structural $R_{\mathrm{struct}} = (Y_4/Y_{32}) \cdot (s_4/s_{32})$")

    # Annotate empirical R values on top of each bar
    for i, v in enumerate(R_emp):
        ax.annotate(f"{v:.3f}", xy=(x[i] - width / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)

    # Annotate the noise residual at each budget (below the x-axis or as text)
    for i, n in enumerate(noise):
        ax.annotate(f"noise={n:+.3f}", xy=(x[i], -0.06),
                    ha="center", va="top", fontsize=8.5, color="#d62728")

    # Horizontal line at R=1 for reference
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(0.02, 1.02, "$R=1$ (no retention loss)", transform=ax.get_yaxis_transform(),
            fontsize=8, color="black", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}M" for t in Ts], fontsize=10)
    ax.set_xlabel(r"Iso-token budget $T$", fontsize=11)
    ax.set_ylabel(r"Retention", fontsize=11)
    ax.set_title("(b) Multiplicative decomposition of $R(T)$",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.10, 5.0)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)


def main():
    obs = load_equivalence()
    decomp = load_decomp()
    print(f"Loaded {len(obs)} retention points, {len(decomp)} decomposition rows.")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.0, 5.4))
    panel_a(ax_a, obs)
    panel_b(ax_b, decomp)
    fig.suptitle(
        "Iter 59 — Pillar 3: Equivalence regions and multiplicative decomposition of $G{=}4/G{=}32$ retention",
        fontsize=12.5, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_pdf = FIG / "group_size_iter59.pdf"
    out_png = FIG / "group_size_iter59.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()