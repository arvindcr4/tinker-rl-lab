#!/usr/bin/env python3
"""Iter 128 -- Pillar 4 figure: Length-Efficiency Frontier.

4-panel PDF/PNG:
  A. (dL_within, dR_within) scatter, GR vs DR, arithmetic_easy
     Pareto frontier overlaid (lower-left is better)
  B. Per-seed efficiency ratio (DR/GR) bar plot, both tasks
  C. Signed CCF |bwd_signed| per (task, algo), mean +/- 95% bootstrap CI
  D. GSM8K heldout (dL, dacc) scatter, GR vs DR

Reads platform_hybrid/experiments/results/length_bias_iter128_*.tsv + meta.
Writes  figures/length_bias_iter128_efficiency_frontier.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

ALGO_COLORS = {"grpo": "#d62728", "dr_grpo": "#1f77b4"}
ALGO_MARKER = {"grpo": "o", "dr_grpo": "s"}


def load_tsv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path) as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    return rows[0], rows[1:]


def to_float(rows: list[list[str]], col: str,
             hdr: list[str]) -> list[float]:
    j = hdr.index(col)
    out = []
    for r in rows:
        try:
            out.append(float(r[j]))
        except (ValueError, IndexError):
            out.append(float("nan"))
    return out


def bootstrap_ci(values: list[float], B: int = 10000,
                 seed: int = 20260703) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    n = arr.size
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    return float(arr.mean()), float(np.quantile(means, 0.025)), \
        float(np.quantile(means, 0.975))


def main() -> None:
    meta = json.load(open(os.path.join(RES, "length_bias_iter128_meta.json")))

    # ------------------ load ------------------
    hdr, rows = load_tsv(os.path.join(RES,
                                       "length_bias_iter128_efficiency_frontier.tsv"))
    task = rows and [r[hdr.index("task")] for r in rows] or []
    algo = [r[hdr.index("algo")] for r in rows]
    seed = [int(r[hdr.index("seed")]) for r in rows]
    dL = to_float(rows, "dL_within", hdr)
    dR = to_float(rows, "dR_within", hdr)
    dacc = to_float(rows, "dacc_heldout", hdr)
    eff = to_float(rows, "efficiency", hdr)

    # signed CCF
    hdr2, rows2 = load_tsv(os.path.join(RES,
                                         "length_bias_iter128_signed_ccf.tsv"))
    abs_bwd = to_float(rows2, "abs_bwd_signed", hdr2)

    # ------------------ figure ------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))

    # --- A. arithmetic_easy Pareto (dL, dR) ---
    ax = axes[0, 0]
    for a, mk, c in [("grpo", "o", "#d62728"),
                     ("dr_grpo", "s", "#1f77b4")]:
        xs = [dL[i] for i in range(len(rows))
              if task[i] == "arithmetic_easy" and algo[i] == a]
        ys = [dR[i] for i in range(len(rows))
              if task[i] == "arithmetic_easy" and algo[i] == a]
        ax.scatter(xs, ys, s=80, marker=mk, c=c, alpha=0.85,
                   edgecolors="black", linewidths=0.7,
                   label=f"{a} (arith_easy)")
    # Pareto frontier: lower-left (less length, more reward) is better
    for a, c in [("grpo", "#d62728"), ("dr_grpo", "#1f77b4")]:
        pts = [(dL[i], dR[i]) for i in range(len(rows))
               if task[i] == "arithmetic_easy" and algo[i] == a]
        # Sort by dL ascending and keep only non-dominated (highest dR for given dL)
        pts.sort()
        best = []
        max_r = -1e9
        for x, y in pts:
            if y > max_r:
                best.append((x, y))
                max_r = y
        if best:
            xs, ys = zip(*best)
            ax.plot(xs, ys, "--", c=c, alpha=0.5, linewidth=1.5,
                    label=f"{a} Pareto")
    ax.set_xlabel("within-run length contraction  (dL = L_first5 - L_last5)")
    ax.set_ylabel("within-run reward gain  (dR = R_last5 - R_first5)")
    ax.set_title("(A) arithmetic_easy: (dL, dR) Pareto  "
                 "(lower-left = better)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- B. efficiency ratio (DR/GR) per task ---
    # Per-task normalization so 1.0 == GRPO efficiency (Dr.GR/GR)
    ax = axes[0, 1]
    h1 = meta["h1_paired_by_task"]
    tasks = sorted(h1.keys())
    width = 0.35
    xpos = np.arange(len(tasks))
    norm_gr = [1.0 for _ in tasks]
    norm_dr = [h1[t]["mean_eff_dr"] / h1[t]["mean_eff_gr"]
               if h1[t]["mean_eff_gr"] != 0 else float("nan")
               for t in tasks]
    ax.bar(xpos - width / 2, norm_gr, width, color="#d62728", alpha=0.85,
           label="GRPO (normalised)")
    ax.bar(xpos + width / 2, norm_dr, width, color="#1f77b4", alpha=0.85,
           label="Dr.GRPO (Dr/GR ratio)")
    for i, t in enumerate(tasks):
        ratio = norm_dr[i]
        ax.text(i + width / 2, ratio + 0.02,
                f"x{ratio:.2f}", ha="center", fontsize=10, fontweight="bold",
                color="#1f77b4")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
               label="GRPO baseline")
    ax.set_xticks(xpos)
    ax.set_xticklabels([t for t in tasks], rotation=15)
    ax.set_ylabel("efficiency ratio  (Dr.GRPO / GRPO)")
    ax.set_title("(B) efficiency ratio per task  "
                 "(>1 = Dr.GRPO Pareto-dominant)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, max(norm_dr + [1.5]) * 1.15)

    # --- C. |bwd_signed| per (task, algo) with bootstrap CI ---
    ax = axes[1, 0]
    hdr2, rows2 = load_tsv(os.path.join(RES,
                                         "length_bias_iter128_signed_ccf.tsv"))
    task2 = [r[hdr2.index("task")] for r in rows2]
    algo2 = [r[hdr2.index("algo")] for r in rows2]
    abs_bwd = to_float(rows2, "abs_bwd_signed", hdr2)
    sub_tasks = sorted(set(task2))
    pos = []
    labels = []
    vals = []
    colors = []
    i = 0
    for t in sub_tasks:
        for a in ("grpo", "dr_grpo"):
            xs = [abs_bwd[j] for j in range(len(rows2))
                  if task2[j] == t and algo2[j] == a]
            m, lo, hi = bootstrap_ci(xs)
            pos.append(i)
            labels.append(f"{t[:6]}\n{a.replace('_', '.')}")
            vals.append((m, lo, hi))
            colors.append(ALGO_COLORS[a])
            i += 1
        i += 0.4  # gap between tasks
    pos = np.asarray(pos)
    means = np.asarray([v[0] for v in vals])
    los = np.asarray([v[1] for v in vals])
    his = np.asarray([v[2] for v in vals])
    yerr = np.vstack([means - los, his - means])
    ax.bar(pos, means, color=colors, alpha=0.85,
           yerr=yerr, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("|bwd_signed|  (per-window signed CCF magnitude)")
    ax.set_title("(C) per-window |bwd_signed|  "
                 "(Dr.GRPO tighter coupling)")
    ax.grid(True, axis="y", alpha=0.3)

    # --- D. GSM8K heldout frontier (dL, dacc) ---
    ax = axes[1, 1]
    for a, mk, c in [("grpo", "o", "#d62728"),
                     ("dr_grpo", "s", "#1f77b4")]:
        xs = [dL[i] for i in range(len(rows))
              if task[i] == "gsm8k_cot" and algo[i] == a]
        ys = [dacc[i] for i in range(len(rows))
              if task[i] == "gsm8k_cot" and algo[i] == a]
        ax.scatter(xs, ys, s=80, marker=mk, c=c, alpha=0.85,
                   edgecolors="black", linewidths=0.7,
                   label=f"{a} (gsm8k_cot)")
    ax.set_xlabel("within-run length contraction  (dL)")
    ax.set_ylabel("heldout accuracy gain  (dacc)")
    ax.set_title("(D) GSM8K: (dL, dacc) heldout frontier  "
                 "(Dr.GRPO upper-left = better trade-off)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Iter 128 -- Pillar 4: Length-Efficiency Frontier  "
                 "(Dr.GRPO Pareto-dominance)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_pdf = os.path.join(FIG, "length_bias_iter128_efficiency_frontier.pdf")
    out_png = out_pdf.replace(".pdf", ".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()