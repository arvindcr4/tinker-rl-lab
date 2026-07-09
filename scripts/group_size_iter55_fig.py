#!/usr/bin/env python3
"""Iter 55 — figure: 2-panel group_size_iter55 visualization.

Panel (a): d_eff(G) and contrastive yield Y(G) — shows that even though
Y increases sublinearly with G, d_eff = G * Y grows near-linearly.
Panel (b): empirical vs predicted argmax G at each budget — 3/4 within
2x; the T=1M under-prediction is the cleanest single-cell evidence of
budget-dependent per-token optimizer work.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
PAPER_FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)
PAPER_FIG.mkdir(exist_ok=True)


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in f]


def main():
    de = read_tsv(RES / "group_size_iter55_d_eff.tsv")
    coup = read_tsv(RES / "group_size_iter55_coupling.tsv")
    step = read_tsv(RES / "group_size_iter55_stepaware.tsv")
    wu = read_tsv(RES / "group_size_iter55_wu_model.tsv")

    # ---- Panel (a): d_eff vs G, Y vs G ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    Gs = [int(r["G"]) for r in de]
    Ys = [float(r["Y_obs"]) for r in de]
    des = [float(r["d_eff_obs"]) for r in de]
    Ys_iid = [float(r["Y_iid"]) for r in de]
    des_iid = [float(r["d_eff_iid"]) for r in de]

    ax = axes[0]
    ax.plot(Gs, Ys, "o-", color="C0", label=r"$Y_{\mathrm{obs}}=1-\mathrm{ZVF}_{\mathrm{obs}}$")
    ax.plot(Gs, Ys_iid, "s--", color="C0", alpha=0.5, label=r"$Y_{\mathrm{iid}}$ (i.i.d. baseline)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Group size $G$")
    ax.set_ylabel("Contrastive yield $Y$", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2 = ax.twinx()
    ax2.plot(Gs, des, "o-", color="C3", label=r"$d_{\mathrm{eff}}=G\cdot Y_{\mathrm{obs}}$")
    ax2.plot(Gs, des_iid, "s--", color="C3", alpha=0.5, label=r"$d_{\mathrm{eff,iid}}$")
    ax2.set_ylabel("Effective samples $d_{\\mathrm{eff}}$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax.set_title("(a) Contrastive yield and effective sample size\nat the small-scale zvf sweep ($p\\approx 0.86$)")
    ax.grid(True, alpha=0.3)
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # ---- Panel (b): empirical vs predicted argmax G; budget-aware retention fit ----
    ax = axes[1]
    Ts = [int(r["T_tokens"]) / 1e6 for r in coup]
    emp = [int(r["empirical_argmax_G"]) for r in coup]
    pred = [int(r["predicted_argmax_G"]) for r in coup]
    within = [r["within_factor_2"] for r in coup]
    ax.plot(Ts, emp, "o-", color="C0", label="Empirical argmax $G$", markersize=8)
    ax.plot(Ts, pred, "s--", color="C2", label="Iso-yield prediction", markersize=8)
    # Mark within 2x
    for i, w in enumerate(within):
        if w == "yes":
            ax.scatter([Ts[i]], [emp[i]], s=200, facecolors="none", edgecolors="green", linewidths=2, zorder=5)
        else:
            ax.scatter([Ts[i]], [emp[i]], s=200, facecolors="none", edgecolors="red", linewidths=2, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Iso-token budget $T$ (M)")
    ax.set_ylabel("Argmax $G$")
    ax.set_yticks([4, 8, 16, 32, 64])
    ax.set_yticklabels(["4", "8", "16", "32", "64"])
    ax.set_title("(b) Argmax $G$ per budget: empirical vs iso-yield\n(green=within 2x, red=out of 2x)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Iter 55: Theory-coupled $G{=}4$ vs $G{=}32$", fontsize=12)
    fig.tight_layout()
    out_pdf = FIG / "group_size_iter55.pdf"
    out_png = FIG / "group_size_iter55.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    print(f"Wrote {out_pdf} and {out_png}")
    # Mirror to paper/figures
    import shutil
    shutil.copy(out_pdf, PAPER_FIG / "group_size_iter55.pdf")
    print(f"Mirrored to {PAPER_FIG / 'group_size_iter55.pdf'}")


if __name__ == "__main__":
    main()