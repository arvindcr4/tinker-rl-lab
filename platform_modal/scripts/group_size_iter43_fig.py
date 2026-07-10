#!/usr/bin/env python3
"""Iter 43 Pillar 3 figure: 3-panel ZVF-decomposed + cost-adjusted equivalence.

Panel (a): Measured retention R(G, T) for all 5 G x 4 T cells, with the
           Wu 2025 97.6% threshold as a horizontal red dashed line.
Panel (b): ZVF-implied retention vs measured retention (scatter); points
           below the diagonal are cells where ZVF alone OVER-predicts
           retention (i.e., the residual gap is from non-ZVF factors).
Panel (c): TOST p-value curves across epsilons, one line per (G_a, G_b)
           pair at T=64M, showing the Wu claim's verification threshold.
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
FIG.mkdir(parents=True, exist_ok=True)

WU = 0.976
COLORS_G = {4: "#e41a1c", 8: "#377eb8", 16: "#4daf4a", 32: "#984ea3", 64: "#ff7f00"}
MARKERS_G = {4: "o", 8: "s", 16: "^", 32: "D", 64: "v"}


def read_tsv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def panel_a(ax, eff_rows: list[dict]) -> None:
    """Measured retention R(G, T) with Wu threshold line."""
    by_g: dict[int, list[dict]] = {}
    for r in eff_rows:
        G = int(r["G"])
        by_g.setdefault(G, []).append(r)
    for G in sorted(by_g):
        rows = sorted(by_g[G], key=lambda r: int(r["T_tokens"]))
        Ts = [int(r["T_tokens"]) / 1e6 for r in rows]
        Rs = [float(r["retention_vs_max_G"]) for r in rows]
        ax.plot(Ts, Rs, color=COLORS_G[G], marker=MARKERS_G[G],
                label=f"G={G}", linewidth=2.0, markersize=7)
    ax.axhline(WU, color="red", linestyle="--", linewidth=1.5, label=f"Wu 2025 ({WU})")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget $T$ (M, log scale)")
    ax.set_ylabel("Retention $R$ (vs max-G at same $T$)")
    ax.set_title("(a) Retention $R(G, T)$ vs Wu threshold")
    ax.set_ylim(0.5, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def panel_b(ax, eff_rows: list[dict]) -> None:
    """ZVF-implied vs measured retention."""
    Rs = [float(r["retention_vs_max_G"]) for r in eff_rows]
    Rs_zvf = [float(r["zvf_implied_retention"]) for r in eff_rows]
    Gs = [int(r["G"]) for r in eff_rows]
    for G in sorted(set(Gs)):
        idx = [i for i, g in enumerate(Gs) if g == G]
        ax.scatter([Rs_zvf[i] for i in idx], [Rs[i] for i in idx],
                   color=COLORS_G[G], marker=MARKERS_G[G], s=70,
                   edgecolors="black", linewidths=0.5, label=f"G={G}")
    ax.plot([0.5, 1.05], [0.5, 1.05], "k--", alpha=0.5, label="Identity")
    ax.set_xlabel("ZVF-implied retention (1 - ZVF/$ZVF_{max}$)")
    ax.set_ylabel("Measured retention $R$")
    ax.set_title("(b) ZVF-mechanistic decomposition")
    ax.set_xlim(0.7, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def panel_c(ax, tost_rows: list[dict]) -> None:
    """TOST equivalence curves at T=64M for each (G_a, G_b)."""
    eps_levels = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    # Filter to T=64M.
    rows_t64 = [r for r in tost_rows if int(r["T_tokens"]) == 64000000]
    pairs = sorted({(int(r["G_a"]), int(r["G_b"])) for r in rows_t64})
    for G_a, G_b in pairs:
        sub = next(r for r in rows_t64 if int(r["G_a"]) == G_a and int(r["G_b"]) == G_b)
        diff = float(sub["diff"])
        diff_lo = float(sub["diff_ci_low"])
        diff_hi = float(sub["diff_ci_high"])
        se = max((diff_hi - diff_lo) / (2 * 1.96), 1e-6)
        from math import erf, sqrt
        def norm_cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))
        # TOST p = max(p_lower, p_upper)
        p_tost = []
        for eps in eps_levels:
            p_lower = 1 - norm_cdf((-eps - diff) / se)
            p_upper = norm_cdf((diff - eps) / se)
            p_tost.append(max(p_lower, p_upper))
        ax.plot(eps_levels, p_tost,
                marker="o", markersize=4,
                label=f"G={G_a} vs G={G_b}", linewidth=1.5)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="$p=0.05$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Equivalence margin $\\epsilon$ (log scale)")
    ax.set_ylabel("TOST $p$-value (log scale)")
    ax.set_title("(c) Cost-adjusted TOST at $T{=}64$M")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=7)


def main() -> None:
    eff_rows = read_tsv(RES / "group_size_iter43_eff_zvf.tsv")
    tost_rows = read_tsv(RES / "group_size_iter43_flop_tost.tsv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_a(axes[0], eff_rows)
    panel_b(axes[1], eff_rows)
    panel_c(axes[2], tost_rows)
    fig.suptitle("Iter 43 Pillar 3: G=4 vs G=32 ZVF-Decomposed Equivalence",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = FIG / "group_size_iter43.pdf"
    out_png = FIG / "group_size_iter43.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[iter43/fig] Wrote {out_pdf}")
    print(f"[iter43/fig] Wrote {out_png}")


if __name__ == "__main__":
    main()