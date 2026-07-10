"""Iter 44 figure: 4-panel synthesis of conditional quantile regression results.

Panel A: per-quantile slope boxplot across (task, algo) — shows the slope of
         q_tau(R|L) for tau in {0.1, 0.25, 0.5, 0.75, 0.9}, GRPO vs Dr.GRPO.
Panel B: asymmetry delta (slope_q90 - slope_q10) per seed with bootstrap CI
         bars (Dr.GRPO - GRPO).
Panel C: ZVF-binned ols_slope heatmap (4 rows task x algo, 3 cols zvf_bin).
Panel D: condvar std(R|L) vs L_bin_center overlaid for the two algos on the
         easy task (most data).
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)


def read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_float(x: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def main() -> None:
    qs = read_tsv(RES / "length_bias_iter44_quantile_slopes.tsv")
    asym = read_tsv(RES / "length_bias_iter44_asymmetry.tsv")
    zvf = read_tsv(RES / "length_bias_iter44_zvf_binned.tsv")
    cv = read_tsv(RES / "length_bias_iter44_condvar.tsv")
    paired = read_tsv(RES / "length_bias_iter44_grpo_vs_drgrpo.tsv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # ---- Panel A: quantile slope boxplot ----
    ax = axes[0, 0]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for r in qs:
        key = (r["task"], r["algo"], r["tau"])
        grouped.setdefault(key, []).append(to_float(r["slope"]))
    labels = []
    data = []
    colors = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for algo in ("grpo", "dr_grpo"):
            for tau in TAUS:
                key = (task, algo, tau)
                v = grouped.get(key, [])
                labels.append(f"{task[:4]}/{algo[:4]}/q{int(float(tau)*100)}")
                data.append(v if v else [0.0])
                colors.append("#377eb8" if algo == "grpo" else "#e41a1c")
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black"})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=70, fontsize=6)
    ax.set_ylabel("slope of q_tau(R) on L")
    ax.set_title("A. quantile slope of R|L (per task/algo/tau)")
    ax.grid(True, alpha=0.3)

    # ---- Panel B: asymmetry delta per (task, algo) ----
    ax = axes[0, 1]
    grp = {(r["task"], r["algo"]): [] for r in asym}
    for r in asym:
        grp[(r["task"], r["algo"])].append(to_float(r["asymmetry_delta"]))
    tasks = ("arithmetic_easy", "gsm8k_cot")
    algos = ("grpo", "dr_grpo")
    x = np.arange(len(tasks))
    width = 0.35
    means_g = [np.mean(grp[(t, "grpo")]) for t in tasks]
    means_d = [np.mean(grp[(t, "dr_grpo")]) for t in tasks]
    ax.bar(x - width/2, means_g, width, color="#377eb8", label="GRPO", alpha=0.7)
    ax.bar(x + width/2, means_d, width, color="#e41a1c", label="Dr.GRPO", alpha=0.7)
    for i, t in enumerate(tasks):
        for j, vals in enumerate((grp[(t, "grpo")], grp[(t, "dr_grpo")])):
            for k, v in enumerate(vals):
                ax.scatter(i + (-width/2 if j == 0 else width/2),
                           v, color="black", s=18, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([t[:10] for t in tasks])
    ax.set_ylabel("asymmetry delta = slope_q90 - slope_q10")
    ax.set_title("B. asymmetry delta (dots=seeds)")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # ---- Panel C: ZVF-binned ols_slope heatmap ----
    ax = axes[1, 0]
    rows = []
    for task in tasks:
        for algo in algos:
            row = []
            for zb in ("low", "mid", "high"):
                v = next((to_float(r["ols_slope"]) for r in zvf
                          if r["task"] == task and r["algo"] == algo
                          and r["zvf_bin"] == zb), float("nan"))
                row.append(v)
            rows.append(row)
    row_labels = [f"{t[:4]}/{a[:4]}" for t in tasks for a in algos]
    M = np.array(rows)
    vmax = np.nanmax(np.abs(M)) or 1e-6
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(("low\nzvf<0.2", "mid\n0.2-0.5", "high\n>=0.5"))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.5 else "black",
                        fontsize=8)
    ax.set_title("C. ols_slope by ZVF bin (task/algo)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="ols_slope R|L")

    # ---- Panel D: condvar std(R|L) vs L_bin (arithmetic_easy only) ----
    ax = axes[1, 1]
    for algo, color, label in (("grpo", "#377eb8", "GRPO"),
                                ("dr_grpo", "#e41a1c", "Dr.GRPO")):
        xs_all, ys_all = [], []
        for r in cv:
            if r["task"] == "arithmetic_easy" and r["algo"] == algo:
                xs_all.append(to_float(r["L_bin_center"]))
                ys_all.append(to_float(r["std_R_given_L"]))
        ax.scatter(xs_all, ys_all, color=color, label=label, alpha=0.5, s=14)
    ax.set_xlabel("L bin center (completion-length tokens, arithmetic_easy)")
    ax.set_ylabel("std(R | L)")
    ax.set_title("D. conditional variance std(R|L) on arithmetic_easy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Iter 44 conditional quantile regression on R|L — anti-trap at upper "
        "quantile, Dr.GRPO strengthens asymmetry",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        out = FIG / f"length_bias_iter44.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()