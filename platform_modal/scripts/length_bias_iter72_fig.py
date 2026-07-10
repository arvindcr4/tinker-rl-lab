"""length_bias_iter72_fig.py — 3-panel matplotlib PDF for iter72 persistence finding.

Three panels:
  (a) AR(1) phi per (experiment, algo) boxplots
  (b) Residual-variance ratio (Dr.GRPO / GRPO), with bootstrap CI
  (c) Lag-2 cross-correlation bar pairs
"""
from __future__ import annotations

import csv
import json
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG_DIR = os.path.join(ROOT, "figures")
PAPER_FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def load_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def f(v, default=float("nan")):
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (ValueError, TypeError):
        return default


def main():
    summary = load_tsv(os.path.join(RES, "length_bias_iter72_summary.tsv"))
    persist = load_tsv(os.path.join(RES, "length_bias_iter72_persistence.tsv"))
    residvar = load_tsv(os.path.join(RES, "length_bias_iter72_residvar.tsv"))
    lagcorr = load_tsv(os.path.join(RES, "length_bias_iter72_lagcorr.tsv"))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    gsm_persist = [r for r in persist if r["experiment"] == "drgrpo_gsm8k_cot"]
    arith_persist = [r for r in persist if r["experiment"] == "drgrpo_vs_grpo"]
    gsm_lag = [r for r in lagcorr if r["experiment"] == "drgrpo_gsm8k_cot" and r["lag"] == "2"]
    arith_lag = [r for r in lagcorr if r["experiment"] == "drgrpo_vs_grpo" and r["lag"] == "2"]

    # Panel (a): AR(1) phi boxplots
    ax = axes[0]
    for i, (data, label, color) in enumerate([
        (gsm_persist, "GSM8K CoT\n(3 seeds)", "#ffe0b2"),
        (arith_persist, "Arith-easy\n(5 seeds)", "#c8e6c9"),
    ]):
        gr_phis = [f(r["phi"]) for r in data if r["algo"] == "grpo"]
        dr_phis = [f(r["phi"]) for r in data if r["algo"] == "drgrpo"]
        positions = [i * 4 + 1, i * 4 + 2]
        vals = [gr_phis, dr_phis]
        bp = ax.boxplot(vals, positions=positions, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=color),
                        medianprops=dict(color="black"))
        for j, vlist in enumerate(vals):
            xs = [positions[j] + (k - (len(vlist) - 1) / 2) * 0.06 for k in range(len(vlist))]
            ax.scatter(xs, vlist, color="black", s=22, zorder=4)
        ax.set_xticks([i * 4 + 1.5])
        ax.set_xticklabels([label], fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_ylabel(r"AR(1) $\phi$ of $\Delta L$ on $\Delta R_t, \Delta L_{t-1}$")
    ax.set_title(r"(a) Persistence coefficient $\phi$" + "\n($\phi<0$: anti-persistent)")
    ax.set_ylim(-0.85, 0.05)

    # Panel (b): Residual variance ratio with CI
    ax = axes[1]
    for i, s in enumerate(summary):
        ratio = f(s["resid_var_ratio"])
        lo = f(s["resid_var_lo"])
        hi = f(s["resid_var_hi"])
        # plot on log scale
        ax.errorbar(i, math.log(ratio),
                    yerr=[[math.log(ratio) - lo], [hi - math.log(ratio)]],
                    fmt="o", markersize=10, color="#b71c1c",
                    capsize=6, linewidth=2)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.axhline(math.log(1.0), color="black", linewidth=1.0)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([r["experiment"] for r in summary], fontsize=9)
    ax.set_ylabel(r"$\log(\sigma^2_{\varepsilon,\mathrm{Dr}} / \sigma^2_{\varepsilon,\mathrm{GR}})$")
    ax.set_title("(b) Residual-variance ratio\n($<0$: Dr.GRPO less noisy after AR(1)+dR fit)")

    # Panel (c): Lag-2 cross-correlation paired bars
    ax = axes[2]
    width = 0.35
    for i, (data, label, color) in enumerate([
        (gsm_lag, "GSM8K CoT\nlag-2 corr", "#ffe0b2"),
        (arith_lag, "Arith-easy\nlag-2 corr", "#c8e6c9"),
    ]):
        gr_vals = [f(r["r_corr"]) for r in data if r["algo"] == "grpo"]
        dr_vals = [f(r["r_corr"]) for r in data if r["algo"] == "drgrpo"]
        x = i * 1.5
        ax.bar(x - width / 2, np.mean(gr_vals), width, color="#90a4ae", label="GRPO")
        ax.bar(x + width / 2, np.mean(dr_vals), width, color=color, edgecolor="black",
               label="Dr.GRPO")
        # paired scatter
        for k, (g, d) in enumerate(zip(gr_vals, dr_vals)):
            ax.scatter([x - width / 2], [g], color="black", s=18, zorder=5)
            ax.scatter([x + width / 2], [d], color="black", s=18, zorder=5)
            ax.plot([x - width / 2, x + width / 2], [g, d], color="grey",
                    linewidth=0.6, zorder=3)
        ax.text(x, max(gr_vals + dr_vals) + 0.04, label, ha="center",
                fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xticks([])
    ax.set_ylabel(r"Pearson $r(dL_t, dR_{t-2})$")
    ax.set_title("(c) Lag-2 dR->dL correlation\n(positive = dR leads dL by 2 steps)")

    fig.suptitle("Iter 72 — Length-shock persistence: AR(1) of $\\Delta L$ on $\\Delta R, \\Delta L_{t-1}$",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(FIG_DIR, "length_bias_iter72_persistence.pdf")
    out_png = os.path.join(FIG_DIR, "length_bias_iter72_persistence.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Also copy into paper/figures for the LaTeX build
    import shutil
    for ext in ("pdf", "png"):
        src = os.path.join(FIG_DIR, f"length_bias_iter72_persistence.{ext}")
        dst = os.path.join(PAPER_FIG_DIR, f"length_bias_iter72_persistence.{ext}")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
