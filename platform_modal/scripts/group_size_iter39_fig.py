"""Iter 39 figure: 3-panel synthesis of G=4 vs G=32 critical-budget analysis."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("platform_hybrid/experiments/results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    # Load iter39 outputs.
    rows = read_tsv(RESULTS / "group_size_iter39_t_critical.tsv")
    claim = {r["claim"]: r for r in read_tsv(RESULTS / "group_size_iter39_claim_strength.tsv")}

    # ----- Panel 1: Retention curve + exponential fit + T* -----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    iter31_row = next(r for r in rows if r["pair"] == "G=4 vs G=32" and r["source"] == "iter31_iso_token")
    T = np.array([1.0, 4.0, 16.0, 64.0])  # M tokens
    R_obs = np.array([0.9762, 0.8333, 0.75, 0.7273])
    R_lo = np.array([0.8444, 0.7536, 0.6897, 0.6703])
    R_hi = np.array([1.1282, 0.9206, 0.8148, 0.7882])
    T_dense = np.linspace(0.5, 64, 200)
    R_inf = float(iter31_row["R_inf_exp"])
    Tau = float(iter31_row["Tau_M"])
    R_fit = R_inf + (1 - R_inf) * np.exp(-T_dense / Tau)
    T_star = float(iter31_row["T_star_M"])
    T_star_lo = float(iter31_row["T_star_lo_M"])
    T_star_hi = float(iter31_row["T_star_hi_M"])
    ax.fill_between(T_dense, R_fit, alpha=0.20, color="C0")
    ax.plot(T_dense, R_fit, color="C0", label=f"Exp fit: $R_\\infty={R_inf:.3f}$, $\\tau={Tau:.2f}$M")
    ax.errorbar(T, R_obs, yerr=[R_obs - R_lo, R_hi - R_obs], fmt="o", color="C0",
                capsize=3, label="Measured (iter 31)")
    ax.axhline(0.976, color="C3", linestyle="--", label="Wu 2025 97.6% threshold")
    ax.axvline(T_star, color="C2", linestyle=":", label=f"$T^*={T_star:.2f}$M ({T_star_lo:.2f}-{T_star_hi:.2f}M)")
    ax.fill_betweenx([0, 1.2], T_star_lo, T_star_hi, color="C2", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 80)
    ax.set_ylim(0.6, 1.2)
    ax.set_xlabel("Token budget $T$ (M)")
    ax.set_ylabel("Retention $R$ vs $G{=}32$")
    ax.set_title("(a) $G{=}4$ vs $G{=}32$ retention curve")
    ax.legend(loc="lower left", fontsize=8)

    # ----- Panel 2: T* across (G_a, G_b) pairs -----
    ax = axes[1]
    pairs_iter35 = [r for r in rows if r["source"] == "iter35_pair_sweep"]
    pairs_iter35_sorted = sorted(pairs_iter35, key=lambda r: float(r["T_star_M"]))
    labels = [r["pair"] for r in pairs_iter35_sorted]
    T_stars = [float(r["T_star_M"]) for r in pairs_iter35_sorted]
    T_lows = [float(r["T_star_lo_M"]) for r in pairs_iter35_sorted]
    T_highs = [float(r["T_star_hi_M"]) for r in pairs_iter35_sorted]
    y = np.arange(len(labels))
    ax.barh(y, T_stars, xerr=[np.array(T_stars) - np.array(T_lows), np.array(T_highs) - np.array(T_stars)],
            color="C0", alpha=0.7, error_kw=dict(ecolor="C0", capsize=3))
    ax.axvline(64, color="C3", linestyle="--", label="Max measured $T$=64M")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Critical budget $T^*$ (M tokens)")
    ax.set_title("(b) $T^*$ at which $R < 0.976$ across all pairs")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    # ----- Panel 3: Claim strength audit -----
    ax = axes[2]
    n_pass = int(claim["n_cells_above_wu_in_CI"]["value"])
    n_total = int(claim["n_cells_above_wu_in_CI"]["denominator"])
    n_tost = int(claim["n_cells_tost_equivalent_eps0.02"]["value"])
    ci_lo = float(claim["pass_fraction_bootstrap_CI_low"]["value"])
    ci_hi = float(claim["pass_fraction_bootstrap_CI_high"]["value"])
    cats = ["Wu 97.6%\nin 95% CI", "TOST equiv.\n($\\epsilon{=}0.02$)", "Worst cell\npasses Wu?"]
    vals = [n_pass / n_total, n_tost / n_total,
            1.0 if claim["worst_cell_upper_CI_excludes_Wu"]["value"] == "True" else 0.0]
    colors = ["C0", "C1", "C2" if vals[2] == 1.0 else "C3"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.7)
    ax.axhline(0.5, color="grey", linestyle=":", label="50% reference")
    # Annotate counts
    for i, (b, v) in enumerate(zip(bars, vals)):
        if i == 0:
            txt = f"{n_pass}/{n_total} ({ci_lo:.2f}-{ci_hi:.2f})"
        elif i == 1:
            txt = f"{n_tost}/{n_total}"
        else:
            txt = "NO" if v == 1.0 else "yes"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, txt,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Claim-strength audit (iter 35 sweep)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_pdf = FIG_DIR / "group_size_iter39.pdf"
    out_png = FIG_DIR / "group_size_iter39.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[iter39] Figure written: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()