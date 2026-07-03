#!/usr/bin/env python3
"""Iteration 31 — Pillar 3 figure: 4-panel G=4 vs G=32 cross-pillar audit.

Panel A: Iso-token retention vs T (Qwen3-8B / GSM8K, G=4 vs G=32) with
         Wu 97.6% constant band; markers colored by TOST verdict.
Panel B: ZVF(p, G) theoretical curves at p=0.86 (illustrative) — shows
         the ZVF collapse from 0.55 at G=4 to 0.008 at G=32.
Panel C: Easy-vs-hard regime bar plot of retention ranges.
Panel D: Predicted vs measured retention scatter (iter19 fit residual).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"

WU_RETENTION = 0.976


def panel_a(ax) -> None:
    iso = pd.read_csv(RES / "group_size_iter31_iso_token.tsv", sep="\t")
    T_M = iso["T_tokens"] / 1e6
    R = iso["retention"]
    R_lo = iso["retention_ci_low"]
    R_hi = iso["retention_ci_high"]
    equiv = iso["tost_equivalent"].astype(bool)
    above = iso["above_wu_97_6pct"].astype(bool)
    colors = []
    for e, a in zip(equiv, above):
        if e:
            colors.append("#2ca02c")  # TOST equiv = green
        elif a:
            colors.append("#1f77b4")  # above Wu = blue
        else:
            colors.append("#d62728")  # well below Wu = red
    ax.errorbar(T_M, R, yerr=[R - R_lo, R_hi - R], fmt="o",
                capsize=4, markersize=10, linewidth=1.5, color="black",
                ecolor="black", alpha=0.4)
    for i, T in enumerate(T_M):
        ax.scatter(T, R.iloc[i], color=colors[i], s=110, zorder=5,
                   edgecolors="black", linewidth=0.6)
    ax.axhline(WU_RETENTION, color="gray", linestyle="--", linewidth=1.4,
               label=f"Wu 2025 retention = {WU_RETENTION:.3f}")
    ax.axhspan(0.956, 0.996, color="gray", alpha=0.10,
               label="TOST equivalence band (eps=0.02)")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget T (M)")
    ax.set_ylabel("Retention R = acc(G=4) / acc(G=32)")
    ax.set_title("(A) Iso-token retention: G=4 vs G=32 (GSM8K)")
    ax.set_ylim(0.65, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def panel_b(ax) -> None:
    # Theoretical ZVF(p, G) = p^G + (1-p)^G for several p
    G = np.arange(2, 65)
    p_list = [0.50, 0.70, 0.86, 0.95]
    cmap = plt.cm.viridis
    for i, p in enumerate(p_list):
        zvf = p ** G + (1 - p) ** G
        ax.semilogy(G, zvf, "-", color=cmap(i / max(1, len(p_list) - 1)),
                    linewidth=1.8, label=f"p={p}")
    ax.axvline(4, color="#1f77b4", linestyle=":", linewidth=1.5,
               label="G=4")
    ax.axvline(32, color="#d62728", linestyle=":", linewidth=1.5,
               label="G=32")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("Group size G")
    ax.set_ylabel("ZVF = $p^G + (1-p)^G$ (log)")
    ax.set_title("(B) ZVF collapses with G at fixed difficulty p")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def panel_c(ax) -> None:
    audit = pd.read_csv(RES / "group_size_iter31_wu_audit.tsv", sep="\t")
    easy = audit[audit["regime"].str.startswith("easy")].iloc[0]
    hard = audit[audit["regime"].str.startswith("hard")].iloc[0]
    labels = ["Easy arithmetic\n(Qwen2.5-0.5B,\nG=2..16, 3 seeds)",
              "Hard GSM8K\n(Qwen3-8B,\nG=4 vs G=32, 4 budgets)"]
    mins = [float(easy["retention_min"]), float(hard["retention_min"])]
    maxs = [float(easy["retention_max"]), float(hard["retention_max"])]
    means = [(a + b) / 2 for a, b in zip(mins, maxs)]
    errs_lo = [m - mn for m, mn in zip(means, mins)]
    errs_hi = [mx - m for mx, m in zip(maxs, means)]
    xs = [0, 1]
    ax.errorbar(xs, means, yerr=[errs_lo, errs_hi], fmt="s",
                color="black", markersize=14, capsize=8, linewidth=2)
    ax.axhline(WU_RETENTION, color="gray", linestyle="--", linewidth=1.4,
               label=f"Wu 2025 retention = {WU_RETENTION:.3f}")
    ax.fill_between([-0.5, 1.5], 0.956, 0.996, color="gray",
                    alpha=0.10, label="TOST equivalence band")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Retention R (range across G and T)")
    ax.set_title("(C) Easy vs hard regime")
    ax.set_ylim(0.65, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)


def panel_d(ax) -> None:
    zvf = pd.read_csv(RES / "group_size_iter31_zvf_coupling.tsv", sep="\t")
    R_meas = zvf["retention_measured"].to_numpy()
    R_pred = zvf["retention_pred_iter19_fit"].to_numpy()
    ax.scatter(R_pred, R_meas, s=110, color="#1f77b4",
               edgecolors="black", linewidth=0.6, zorder=5)
    for i, T in enumerate(zvf["T_tokens"]):
        ax.annotate(f"T={int(T)//1_000_000}M",
                    (R_pred[i], R_meas[i]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#1f77b4")
    lo = float(min(R_pred.min(), R_meas.min())) - 0.02
    hi = float(max(R_pred.max(), R_meas.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2,
            label="y = x (perfect prediction)")
    ax.axhline(WU_RETENTION, color="gray", linestyle="--", linewidth=1.4,
               label=f"Wu 2025 = {WU_RETENTION:.3f}")
    ax.set_xlabel("Predicted retention R (iter19 saturating fit)")
    ax.set_ylabel("Measured retention R (Qwen3-8B / GSM8K)")
    ax.set_title("(D) Predicted vs measured retention (G=4 vs G=32)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.suptitle(
        "Iter 31 — Pillar 3: G=4 vs G=32 at broader scale — "
        "Wu 2025 'It Takes Two' audit",
        y=1.005, fontsize=11,
    )
    fig.tight_layout()
    out_pdf = FIG / "group_size_iter31.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(FIG / "group_size_iter31.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()