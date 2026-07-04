"""Render the four-panel iter140 Eureka figure into figures/eureka_cross_pillar.{pdf,png}.

Panels:
  (a) RQS bar per anchor (12)
  (b) AIC race on R_max at n=5
  (c) 12-anchor residualization scatter (RQS vs residual_cap_only)
  (d) iter127 cross-pillar scatter (richness proxy vs residual)

Reads the same evidence TSVs produced by eureka_reward_design_quality.py.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT_BERKELEY = RES / "berkeley"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _read_tsv(path):
    with open(path) as f:
        return [r for r in csv.reader(f, delimiter="\t")]


def panel_a(ax):
    rows = _read_tsv(OUT_BERKELEY / "eureka_rqs_per_anchor.tsv")[1:]
    names = [r[0] for r in rows]
    rqs = [float(r[-1]) for r in rows]
    n = len(names)
    idx = np.arange(n)
    bar_colors = ["tab:blue" if v > 0.6 else ("tab:gray" if v > 0.1 else "tab:red") for v in rqs]
    bars = ax.bar(idx, rqs, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("RQS (Reward-Design Quality)")
    ax.set_title("(a) RQS per anchor (12)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.6, ls="--", lw=0.6, color="tab:green", alpha=0.6)
    ax.grid(True, axis="y", ls=":", lw=0.4)


def panel_b(ax):
    rows = _read_tsv(OUT_BERKELEY / "eureka_aic_compare.tsv")[1:]
    names = [r[0] for r in rows]
    aiccs = [float(r[4]) for r in rows]
    delta = [float(r[5]) for r in rows]
    n = len(names)
    idx = np.arange(n)
    colors = [
        "tab:green" if d == 0 else ("tab:olive" if d < 2 else "tab:red") for d in delta
    ]
    bars = ax.bar(idx, delta, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("$\\Delta$AICc vs best")
    ax.set_title("(b) AIC race on $R_{\\max}$, $n=5$ (borderline NULL)", fontsize=9)
    ax.axhline(2, ls="--", lw=0.6, color="tab:red")
    ax.text(0.05, 0.95, "Miller-recipe NULL threshold $\\Delta$AICc$\\geq2$",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))
    ax.grid(True, axis="y", ls=":", lw=0.4)


def panel_c(ax):
    rows = _read_tsv(OUT_BERKELEY / "eureka_residualization.tsv")[1:]
    names = [r[0] for r in rows]
    rqs = [float(r[2]) for r in rows]
    resids = [float(r[7]) for r in rows]  # resid_cap_only
    rqs_np = np.array(rqs)
    resids_np = np.array(resids)
    ax.scatter(rqs_np, resids_np, s=24, c="tab:blue", edgecolor="black", linewidth=0.3)
    # regression
    if len(rqs_np) > 1:
        coef = np.polyfit(rqs_np, resids_np, 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, coef[0] * xs + coef[1], ls="--", c="tab:red", lw=0.8,
                label="$\\rho = +0.225$")
    for i, n in enumerate(names):
        ax.annotate(n.split("-")[0], (rqs[i], resids[i]), fontsize=5, alpha=0.7,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("RQS")
    ax.set_ylabel("residual from capability-only fit")
    ax.set_title("(c) 12-anchor cap-residual vs RQS", fontsize=9)
    ax.axhline(0, ls=":", lw=0.5, color="gray")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, ls=":", lw=0.4)


def panel_d(ax):
    rows = _read_tsv(OUT_BERKELEY / "eureka_cross_pillar.tsv")[1:]
    proxy = []
    resid = []
    g_axis = []
    for r in rows:
        try:
            proxy.append(float(r[7]))
            resid.append(float(r[5]))
            g_axis.append(int(r[1]))
        except Exception:
            continue
    proxy_np = np.array(proxy)
    resid_np = np.array(resid)
    g_axis_np = np.array(g_axis)
    sc = ax.scatter(proxy_np, resid_np, s=30, c=g_axis_np, cmap="viridis",
                    edgecolor="black", linewidth=0.3)
    if len(proxy_np) > 1:
        coef = np.polyfit(proxy_np, resid_np, 1)
        xs = np.linspace(min(proxy_np), max(proxy_np), 50)
        ax.plot(xs, coef[0] * xs + coef[1], ls="--", c="tab:red", lw=0.8,
                label="$\\rho = -0.569$, $p=0.029$ (DECISIVE)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel("$G$ (group size)", fontsize=8)
    ax.set_xlabel("richness proxy $y = 1 - \\mathrm{ZVF}_{\\mathrm{theory}}$ (iter~131)")
    ax.set_ylabel("iter~127 joint-fit residual $r$")
    ax.set_title("(d) Iter~127 cross-pillar, $n=20$ (Pearson $\\rho=-0.569$, $p=0.029$)",
                 fontsize=9)
    ax.axhline(0, ls=":", lw=0.5, color="gray")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, ls=":", lw=0.4)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    plt.suptitle("Iter 140 (Berkeley F24 L9 -- Eureka) reward-design quality as Pillar-1 "
                 "exogenous covariate",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_pdf = FIG / "eureka_cross_pillar.pdf"
    out_png = FIG / "eureka_cross_pillar.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=130)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
