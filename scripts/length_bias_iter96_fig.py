#!/usr/bin/env python3
"""Iter 96 -- Pillar 4 (Length Bias / Dr.GRPO): 4-panel figure for ICCA.

Panel A : Per-task seed-averaged CCF(+/-k) bars (lags -3..+3) for GRPO
          vs Dr.GRPO on GSM8K CoT (the hard task).
Panel B : Same as A for arithmetic (the saturated control task).
Panel C : Per-seed asymmetry index AI, paired by seed, on GSM8K CoT.
          Dr.GRPO severs R->L feedback => AI rises (less negative).
Panel D : Forward vs Backward innovation coupling scatter, GSM8K CoT
          seeds; arrows show Dr.GRPO's effect.

USAGE
-----
python3 scripts/length_bias_iter96_fig.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIG_DIR = os.path.join(ROOT, "figures")
PAPER_FIG_DIR = os.path.join(ROOT, "paper", "figures")

PERRUN = os.path.join(RESULTS, "length_bias_iter96_perrun.tsv")
SUMMARY = os.path.join(RESULTS, "length_bias_iter96_summary.tsv")
OUT_PDF = os.path.join(FIG_DIR, "length_bias_iter96.pdf")
OUT_PNG = os.path.join(FIG_DIR, "length_bias_iter96.png")


def load_perrun() -> list[dict[str, float]]:
    rows = []
    with open(PERRUN) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            out = {}
            for k, v in row.items():
                if k in ("task", "algo"):
                    out[k] = v
                elif v in ("", None):
                    out[k] = 0.0
                else:
                    out[k] = float(v)
            out["seed"] = int(out["seed"])
            rows.append(out)
    return rows


def load_summary() -> list[dict[str, float]]:
    rows = []
    with open(SUMMARY) as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            out = {}
            for k, v in row.items():
                if k in ("task", "algo"):
                    out[k] = v
                elif v in ("", None):
                    out[k] = 0.0
                else:
                    out[k] = float(v)
            rows.append(out)
    return rows


def panel_a_b(ax, rows, task: str, k_range=range(-3, 4)):
    """CCF bar pair per lag, GRPO vs Dr.GRPO on a given task."""
    sub = [r for r in rows if r["task"] == task]
    by_algo = defaultdict(list)
    for r in sub:
        by_algo[r["algo"]].append(r)
    width = 0.38
    x = np.array(list(k_range), dtype=np.float64)
    for j, (algo, col) in enumerate([("grpo", "#4c72b0"), ("dr_grpo", "#c44e52")]):
        recs = by_algo.get(algo, [])
        if not recs:
            continue
        ccf = np.array([[r[f"ccf_k={k:+d}"] for k in k_range] for r in recs])
        m = ccf.mean(axis=0)
        s = ccf.std(axis=0)
        offset = (-width / 2) if j == 0 else (width / 2)
        ax.bar(x + offset, m, yerr=s, width=width, color=col, alpha=0.85,
               label=f"{algo} (n={len(recs)})", edgecolor="black", linewidth=0.4,
               capsize=2)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xticks(list(k_range))
    ax.set_xticklabels([f"{k:+d}" for k in k_range])
    ax.set_xlabel("lag k (k>0: e_L leads e_R)")
    ax.set_ylabel(r"CCF$(e_L, e_R; k)$")
    ax.set_title(f"{task}: innovation cross-correlation", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def panel_c(ax, rows):
    """Per-seed AI paired plot, GSM8K CoT only."""
    sub = [r for r in rows if r["task"] == "gsm8k_cot"]
    by_algo = defaultdict(list)
    for r in sub:
        by_algo[r["algo"]].append(r)
    seeds_grpo = {int(r["seed"]): r["ai"] for r in by_algo["grpo"]}
    seeds_drgrpo = {int(r["seed"]): r["ai"] for r in by_algo["dr_grpo"]}
    common = sorted(set(seeds_grpo) & set(seeds_drgrpo))
    if not common:
        ax.set_title("GSM8K CoT: no paired seeds")
        return
    x = np.arange(len(common))
    gv = [seeds_grpo[s] for s in common]
    dv = [seeds_drgrpo[s] for s in common]
    for i, s in enumerate(common):
        ax.plot([i, i], [gv[i], dv[i]], color="grey", linewidth=1, alpha=0.6)
        ax.scatter([i], [gv[i]], color="#4c72b0", s=55, zorder=3, edgecolor="black",
                   linewidth=0.4, label="GRPO" if i == 0 else None)
        ax.scatter([i], [dv[i]], color="#c44e52", s=55, zorder=3, edgecolor="black",
                   linewidth=0.4, label="Dr.GRPO" if i == 0 else None)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in common])
    ax.set_xlabel("seed (paired)")
    ax.set_ylabel("Asymmetry Index AI")
    ax.set_title("GSM8K CoT: AI per seed (AI>0 = L leads R)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def panel_d(ax, rows):
    """Forward vs Backward coupling magnitude scatter (GSM8K CoT)."""
    sub = [r for r in rows if r["task"] == "gsm8k_cot"]
    by_algo = defaultdict(list)
    for r in sub:
        by_algo[r["algo"]].append(r)
    seeds_grpo = {int(r["seed"]): r for r in by_algo["grpo"]}
    seeds_drgrpo = {int(r["seed"]): r for r in by_algo["dr_grpo"]}
    common = sorted(set(seeds_grpo) & set(seeds_drgrpo))
    for s in common:
        g = seeds_grpo[s]
        d = seeds_drgrpo[s]
        ax.scatter([g["bwd"]], [g["fwd"]], color="#4c72b0", s=70,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.scatter([d["bwd"]], [d["fwd"]], color="#c44e52", s=70,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.annotate("", xy=(d["bwd"], d["fwd"]), xytext=(g["bwd"], g["fwd"]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.1,
                                    alpha=0.75))
    # Reference y=x line (forward = backward => AI = 0)
    lim_max = max([r["bwd"] for r in sub] + [r["fwd"] for r in sub]) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="grey", linestyle="--",
            linewidth=0.7, label="F = B")
    ax.set_xlabel(r"Backward $|CCF|$ sum  (R leads L)")
    ax.set_ylabel(r"Forward $|CCF|$ sum  (L leads R)")
    ax.set_title("GSM8K CoT: F vs B (arrows: Dr.GRPO moves)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> int:
    rows = load_perrun()
    summary = load_summary()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panel_a_b(axes[0, 0], rows, "gsm8k_cot")
    panel_a_b(axes[0, 1], rows, "arithmetic_easy")
    panel_c(axes[1, 0], rows)
    panel_d(axes[1, 1], rows)
    fig.suptitle(
        "Iter 96 — Innovation Cross-Correlation Asymmetry\n"
        r"After marginal AR(1) filtering of $(L_t, R_t)$" +
        "\n" + r"Dr.GRPO severs R$\to$L innovation feedback on GSM8K CoT",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(os.path.join(PAPER_FIG_DIR, "length_bias_iter96.pdf"))
    fig.savefig(os.path.join(PAPER_FIG_DIR, "length_bias_iter96.png"), dpi=150)
    print(f"[iter96-fig] wrote {OUT_PDF}")
    print(f"[iter96-fig] wrote {OUT_PNG}")
    print(f"[iter96-fig] wrote {os.path.join(PAPER_FIG_DIR, 'length_bias_iter96.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())