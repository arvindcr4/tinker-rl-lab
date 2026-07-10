#!/usr/bin/env python3
"""Iter 47 figure: 3-panel summary of T* and monotonicity results.

  (a) Critical-budget T*(Ga, R): horizontal bar chart of T* at R=0.976 for
      G_a in {4, 8, 16, 32} with bootstrap 95% CI whiskers.
  (b) Retention vs log(T) for G_a in {4, 8, 16, 32}, with the 0.976 Wu
      threshold drawn as a red dashed horizontal line and the T* markers
      as vertical lines.
  (c) Per-difficulty T*(R=0.976) horizontal bars with bootstrap.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"


def read_tsv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    crit = read_tsv(RES / "group_size_iter47_critical_T.tsv")
    mono = read_tsv(RES / "group_size_iter47_monotonicity.tsv")
    diff = read_tsv(RES / "group_size_iter47_diff_Tstar.tsv")
    raw = read_tsv(RES / "group_size_token_normalized.tsv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- (a) Bar chart: T* at R=0.976 per Ga ---
    ax = axes[0]
    rows_a = [r for r in crit if float(r["R_target"]) == 0.976]
    Gas = [int(r["G_a"]) for r in rows_a]
    Ts_star = [float(r["T_star_M_tokens"]) if r["T_star_M_tokens"] not in ("inf", "no_crossover") else np.nan
                for r in rows_a]
    lo = [float(r["T_star_ci_low_M"]) if r["T_star_ci_low_M"] else np.nan for r in rows_a]
    hi = [float(r["T_star_ci_high_M"]) if r["T_star_ci_high_M"] else np.nan for r in rows_a]
    err_lo = [t - l if not (np.isnan(t) or np.isnan(l)) else 0 for t, l in zip(Ts_star, lo)]
    err_hi = [h - t if not (np.isnan(t) or np.isnan(h)) else 0 for t, h in zip(Ts_star, hi)]
    ax.bar(range(len(Gas)), [t if not np.isnan(t) else 0 for t in Ts_star],
            yerr=[err_lo, err_hi], color="#5fa8d3", edgecolor="black", capsize=4)
    ax.set_xticks(range(len(Gas)))
    ax.set_xticklabels([f"G={g}" for g in Gas])
    ax.set_ylabel(r"$T^*$ (M tokens) at $R = 0.976$ (Wu threshold)")
    ax.set_title(r"(a) Critical budget $T^*$ per $G_a$")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="T = 1M (smallest budget)")
    ax.axhline(4.0, color="orange", linestyle="--", linewidth=1, label="T = 4M (first scale-up)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- (b) Retention vs log(T) per Ga ---
    ax = axes[1]
    Ts = sorted({int(r["budget_tokens"]) for r in raw})
    G_set = sorted({int(r["G"]) for r in raw})
    anchor = {T: max(float(r["heldout_acc_mean"]) for r in raw if int(r["budget_tokens"]) == T)
               for T in Ts}
    cmap = plt.cm.viridis
    for j, G in enumerate(G_set):
        Rs = []
        for T in Ts:
            acc = next((float(r["heldout_acc_mean"]) for r in raw
                          if int(r["budget_tokens"]) == T and int(r["G"]) == G), np.nan)
            Rs.append(acc / anchor[T])
        ax.plot(Ts, Rs, "-o", color=cmap(j / len(G_set)), label=f"G={G}")
    ax.axhline(0.976, color="red", linestyle="--", linewidth=1.5, label="Wu 2025 R=0.976")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget T (M)")
    ax.set_ylabel("Retention  $R(G_a, T) = \\mathrm{acc}(G_a, T) / \\mathrm{acc}(G_{\\max}, T)$")
    ax.set_title("(b) Retention decays with $T$ for every $G_a$")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # --- (c) Per-bin T* at R=0.976 ---
    ax = axes[2]
    rows_c = [r for r in diff if float(r["R_target"]) == 0.976
               and r["T_star_M_tokens"] not in ("n/a", "no_crossover")]
    bins = [r["bin"] for r in rows_c]
    Ts_b = [float(r["T_star_M_tokens"]) for r in rows_c]
    ax.bar(range(len(bins)), Ts_b, color="#a4d29e", edgecolor="black")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}" for b in bins])
    ax.set_ylabel(r"$T^*$ (M tokens) at $R = 0.976$")
    ax.set_title(r"(c) Per-difficulty $T^*$ (Wu retention crossover)")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="T = 1M")
    ax.axhline(4.0, color="orange", linestyle="--", linewidth=1, label="T = 4M")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Iter 47 -- Pillar 3: Wu Retention Critical-Budget $T^*$ Decomposition", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "group_size_iter47.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(f"Wrote {out} and PNG sibling.")


if __name__ == "__main__":
    main()
