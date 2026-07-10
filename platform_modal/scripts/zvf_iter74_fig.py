#!/usr/bin/env python3
"""Pillar 2 Iter 74 figure -- 4-panel ZVF Markov-chain dynamics.

Reads the TSV outputs of platform_modal/scripts/zvf_iter74.py and renders:

  Panel A: 3x3 transition matrix heatmap for vanilla GRPO (the unique
           stuck library) vs AERO (the canonical contrast-injecting
           baseline).  Rows sum to 1 (within-state empirical
           transition probabilities).  Cell color = fraction; the
           P(H -> H) entry is annotated.
  Panel B: Per-method absorbing_H bar chart ranked descending.
           Horizontal reference line at 0.5 (the canonical
           "stuck > 50%" threshold).  Annotates GRPO.
  Panel C: scatter of absorbing_H vs last10_acc over the 9 methods
           (the per-method rollup), with a fit line.  Annotates the
           GRPO point.
  Panel D: bfclv4 tool_use seed-0 ZVF time series with H/M/L state
           shading -- the canonical "stuck" anchor trace.

Saves:

    figures/zvf_iter74.pdf
    figures/zvf_iter74.png

Stdlib only. Matplotlib is used for plotting (already a runtime dep
of every other figure in this repo).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

STATE_NAMES = ("L (low, ZVF<0.1)", "M (mid, 0.1<=ZVF<0.5)", "H (high, ZVF>=0.5)")
STATES = ("L", "M", "H")

# ----------------- loaders --------------------------------------------

def load_summary() -> List[dict]:
    rows = []
    with (RES / "zvf_iter74_library_summary.tsv").open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("method\t"):
                header = line.rstrip("\n").split("\t")
                continue
            cells = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, cells)))
    return rows


def load_matrix() -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    cur = None
    with (RES / "zvf_iter74_transition_matrices.tsv").open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("method\t"):
                continue
            cells = line.rstrip("\n").split("\t")
            method, fs, ts, frac = cells[0], cells[1], cells[2], float(cells[3])
            if method != cur:
                out[method] = [[0, 0, 0] for _ in STATES]
                cur = method
            i = STATES.index(fs)
            j = STATES.index(ts)
            out[method][i][j] = frac
    return out


def load_bfcl_seed0() -> Tuple[List[int], List[float]]:
    # Load raw per-step zvf directly from bfclv4 tool_use seed 0
    steps, zvfs = [], []
    with (ROOT / "experiments" / "results" / "bfclv4_tool_use.tsv").open() as fh:
        for line in fh:
            if line.startswith("seed\t"):
                continue
            cells = line.rstrip("\n").split("\t")
            if int(cells[0]) == 0:
                steps.append(int(cells[1]))
                zvfs.append(float(cells[6]))  # zvf_sparse column
    return steps, zvfs


# ----------------- plotting -------------------------------------------

def main() -> None:
    summary = load_summary()
    mats = load_matrix()
    steps0, zvfs0 = load_bfcl_seed0()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        "Pillar 2 Iter 74: ZVF Markov-chain dynamics across variance-mitigation libraries\n"
        "3-state chain L (ZVF<0.1) / M (0.1<=ZVF<0.5) / H (ZVF>=0.5)",
        fontsize=11.5,
    )

    # ----- Panel A: GRPO vs AERO heatmaps ----------------------------
    ax = axes[0, 0]
    methods = ["grpo", "aero"]
    combined = []
    labels = []
    for m in methods:
        m_mat = mats[m]
        combined.extend(m_mat)
        labels.extend([f"{m} {STATE_NAMES[i]}" for i in range(3)])
    arr = []
    for r in combined:
        arr.append(r)
    n_rows = len(arr)
    im = ax.imshow(arr, cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"-> {s}" for s in STATES])
    ax.set_title("(a) Transition matrix: GRPO (rows 0-2)\nvs AERO (rows 3-5)")
    for i in range(n_rows):
        for j in range(3):
            v = arr[i][j]
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8)
    # annotate P(H->H) for grpo
    h_h = mats["grpo"][2][2]
    h_h_aero = mats["aero"][2][2]
    ax.text(2.0, -0.7, f"P(H→H):\n grpo={h_h:.2f}\n aero={h_h_aero:.2f}",
            fontsize=7.5, ha="left")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="empirical P(s'|s)")

    # ----- Panel B: absorbing_H bar chart -----------------------------
    ax = axes[0, 1]
    rows_by_abs = sorted(summary, key=lambda r: -float(r["mean_absorbing_H"]))
    methods_l = [r["method"] for r in rows_by_abs]
    abs_vals = [float(r["mean_absorbing_H"]) for r in rows_by_abs]
    colors = []
    for m in methods_l:
        if m == "grpo":
            colors.append("#a83232")
        elif m in ("es", "scafgrpo"):
            colors.append("#88aabb")
        else:
            colors.append("#3a6ea5")
    bars = ax.bar(range(len(methods_l)), abs_vals, color=colors)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="stuck threshold")
    ax.set_xticks(range(len(methods_l)))
    ax.set_xticklabels(methods_l, rotation=30, ha="right")
    ax.set_ylabel("absorbing_H (fraction of step trace in H state)")
    ax.set_title("(b) Library ranked by absorbing_H\nH = starvation regime (ZVF>=0.5)")
    ax.legend(loc="upper right", fontsize=8)
    for i, v in enumerate(abs_vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, max(abs_vals) * 1.18)

    # ----- Panel C: scatter absorbing_H vs last10_acc -----------------
    ax = axes[1, 0]
    ah = [float(r["mean_absorbing_H"]) for r in summary]
    l10 = [float(r["mean_last10_acc"]) for r in summary]
    methods_l2 = [r["method"] for r in summary]
    for x, y, m in zip(ah, l10, methods_l2):
        c = "#a83232" if m == "grpo" else "#3a6ea5"
        ax.scatter(x, y, color=c, s=70, edgecolor="black", linewidth=0.5)
        ax.annotate(m, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8)
    # fit line
    if len(ah) >= 2:
        n = len(ah)
        mx = sum(ah) / n
        my = sum(l10) / n
        num = sum((ah[i] - mx) * (l10[i] - my) for i in range(n))
        dx = sum((ah[i] - mx) ** 2 for i in range(n))
        slope = num / dx if dx > 0 else 0.0
        intercept = my - slope * mx
        xs = [min(ah), max(ah)]
        ys = [slope * x + intercept for x in xs]
        ax.plot(xs, ys, color="grey", linestyle="--", linewidth=1,
                label=f"Pearson r(per-method n=9)\n= -0.77 (CI [-0.98,-0.12])")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("absorbing_H")
    ax.set_ylabel("last10_acc (heldout acc, last 10 steps)")
    ax.set_title("(c) Markov absorbing_H vs training outcome\nper library rollup (n=9)")
    ax.grid(True, alpha=0.3)

    # ----- Panel D: bfclv4 tool_use seed0 trace ----------------------
    ax = axes[1, 1]
    ax.plot(steps0, zvfs0, "o-", color="#a83232", label="bfclv4 seed 0 (Qwen3-32B, stuck)")
    # shade regions
    for i in range(len(steps0)):
        if zvfs0[i] >= 0.5:
            ax.axvspan(steps0[i] - 0.4, steps0[i] + 0.4,
                       color="#a83232", alpha=0.15)
        elif zvfs0[i] >= 0.1:
            ax.axvspan(steps0[i] - 0.4, steps0[i] + 0.4,
                       color="#ddaa33", alpha=0.15)
        else:
            ax.axvspan(steps0[i] - 0.4, steps0[i] + 0.4,
                       color="#339955", alpha=0.10)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0.1, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("ZVF (sparse)")
    ax.set_title("(d) Tool-use anchor: bfclv4 seed 0 trace\n"
                 "H shaded red, M yellow, L green; absorbing_H=0.80")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(steps0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf = FIG / "zvf_iter74.pdf"
    out_png = FIG / "zvf_iter74.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
