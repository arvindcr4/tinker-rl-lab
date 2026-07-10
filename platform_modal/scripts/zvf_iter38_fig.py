#!/usr/bin/env python3
"""Iter 38 figure: 3-panel cross-library ZVF / Iso-Yield / Classifier.

Panel A: Iso-Yield curves G(y) for 5 representative libraries (grpo, aero,
        gift, scafgrpo, mcgrpo), iid baseline (dashed) and empirical-
        corrected (solid) using the tinker_gsm8k delta_div = +0.122.
Panel B: Per-library cost savings at y_target=0.80 (G_iid*G*K_prompts*L_bar
        vs G_emp*...), as a horizontal bar chart.
Panel C: Failure-mode LOO confusion heatmap on the 14-pool rows.

Inputs:
    platform_hybrid/experiments/results/zvf_iter38_isoyield.tsv
    platform_hybrid/experiments/results/zvf_iter38_classifier.tsv

Output:
    figures/zvf_iter38.pdf  (matplotlib fallback).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "figures"


def load_tsv(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def main() -> None:
    iso_rows = load_tsv(RESULTS / "zvf_iter38_isoyield.tsv")
    # Header: library, model, p_x, y_target, G_ref, delta_div_lib, G_iid,
    #         G_empirical, G_savings, cost_iid, cost_empirical, cost_savings_frac
    iso_hdr = iso_rows[0]
    idx = {c: i for i, c in enumerate(iso_hdr)}
    iso_libs = sorted({r[idx["library"]] for r in iso_rows[1:]})

    cls_rows = load_tsv(RESULTS / "zvf_iter38_classifier.tsv")
    # Filter out the confusion block (lines that don't start with a library name)
    LIB_NAMES = {
        "grpo", "aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift",
        "areal", "es", "gsm8k_real", "arithmetic_groupsize", "tool_use",
        "scaling_law", "drgrpo_vs_grpo", "samestack_ppo_grpo",
    }
    cls_data = [r for r in cls_rows if r[0] in LIB_NAMES]
    # Header: library, model, mean_zvf, mean_last10, mean_peak, true, pred, correct
    cls_hdr = cls_data[0]

    # ------------------------------------------------------------------
    # Panel A: Iso-Yield curves for 5 representative libraries
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    rep_libs = ["grpo", "aero", "scafgrpo", "gift", "mcgrpo"]
    p_for_lib = {}
    for r in iso_rows[1:]:
        p_for_lib[r[idx["library"]]] = float(r[idx["p_x"]])

    y_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    delta_div = 0.122  # tinker_gsm8k per-prompt anti-herding
    for lib in rep_libs:
        p = p_for_lib[lib]
        g_iid = []
        g_emp = []
        for y in y_grid:
            # iid
            G = 1
            if 1.0 - 2 * p * (1 - p) < y:
                for Gc in range(2, 65):
                    if 1.0 - p ** Gc - (1 - p) ** Gc >= y:
                        G = Gc
                        break
            g_iid.append(G)
            # empirical
            Ge = 1
            if 1.0 - max(0.0, 2 * p * (1 - p) - delta_div) < y:
                for Gc in range(2, 65):
                    z_eff = max(0.0, min(1.0, p ** Gc + (1 - p) ** Gc - delta_div))
                    if 1.0 - z_eff >= y:
                        Ge = Gc
                        break
            g_emp.append(Ge)
        axes[0].plot(y_grid, g_iid, "--", alpha=0.5, lw=1)
        axes[0].plot(y_grid, g_emp, "-", lw=2, label=f"{lib} (p={p:.2f})")
    axes[0].set_xlabel("Target yield $Y_{target}$")
    axes[0].set_ylabel("Minimum $G$ required")
    axes[0].set_title("(A) Iso-Yield curves: iid (dashed) vs empirical (solid)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel B: per-library cost savings at y_target=0.80
    # ------------------------------------------------------------------
    savings_by_lib = {}
    for r in iso_rows[1:]:
        if abs(float(r[idx["y_target"]]) - 0.80) < 1e-6:
            savings_by_lib[r[idx["library"]]] = float(r[idx["cost_savings_frac"]])
    libs = list(savings_by_lib.keys())
    saves = [savings_by_lib[l] for l in libs]
    axes[1].barh(libs, saves, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Cost savings (frac) at $Y_{target}=0.80$")
    axes[1].set_title("(B) Iso-Yield cost savings, iid → empirical")
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].axvline(0.0, color="black", lw=0.5)

    # ------------------------------------------------------------------
    # Panel C: failure-mode confusion heatmap
    # ------------------------------------------------------------------
    classes = ["collapse", "drift", "plateau", "converged"]
    conf = {c: {c2: 0 for c2 in classes} for c in classes}
    for r in cls_data[1:]:
        # library, model, mean_zvf, mean_last10, mean_peak, true, pred, correct
        truth = r[5]
        pred = r[6]
        if truth in conf and pred in conf[truth]:
            conf[truth][pred] += 1
    mat = [[conf[c][c2] for c2 in classes] for c in classes]
    im = axes[2].imshow(mat, cmap="Blues", aspect="auto")
    axes[2].set_xticks(range(len(classes)))
    axes[2].set_xticklabels(classes, rotation=45, ha="right")
    axes[2].set_yticks(range(len(classes)))
    axes[2].set_yticklabels(classes)
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")
    axes[2].set_title("(C) Failure-mode LOO confusion (k=3)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if mat[i][j] > 5 else "black"
            axes[2].text(
                j, i, str(mat[i][j]), ha="center", va="center", color=color
            )
    plt.colorbar(im, ax=axes[2], fraction=0.04)

    plt.tight_layout()
    out = FIG_DIR / "zvf_iter38.pdf"
    plt.savefig(out)
    out_png = FIG_DIR / "zvf_iter38.png"
    plt.savefig(out_png, dpi=120)
    print(f"Wrote {out} and {out_png}")


if __name__ == "__main__":
    main()