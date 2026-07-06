#!/usr/bin/env python3
"""Iter 124 -- figure: 4-panel layout.

Panel (A): scatter of severship vs BASELINE dL/dt
Panel (B): pooled Spearman bar chart H1 / H2 with bootstrap CI
Panel (C): per-window Spearman(severship, dL/dt)
Panel (D): cross-task envelope (mean dL/dt vs mean severship)

Reads the iter124 TSV artefacts, mirrors to figures/ and
paper/figures/, prints the headline and exits.
"""
from __future__ import annotations

import json
import os
import sys
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "figures")
PFIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PFIG, exist_ok=True)

TASKS = ["arithmetic_easy", "gsm8k_cot"]
COL = {"arithmetic_easy": "#1f77b4", "gsm8k_cot": "#d62728"}


def _read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


per_window = _read_tsv(os.path.join(RES, "length_bias_iter124_per_window.tsv"))
envelope = _read_tsv(os.path.join(RES, "length_bias_iter124_envelope.tsv"))
pooled_h1 = _read_tsv(os.path.join(RES, "length_bias_iter124_pooled_h1_sever_vs_dL_GR.tsv"))
pooled_h2 = _read_tsv(os.path.join(RES, "length_bias_iter124_pooled_h2_sever_vs_dR_GR.tsv"))
boot = _read_tsv(os.path.join(RES, "length_bias_iter124_rho_bootstrap.tsv"))

per_seed = {}
for r in per_window:
    per_seed.setdefault(r["task"], []).append(r)
boot_by_h = {}
for r in boot:
    boot_by_h.setdefault(r["hypothesis"], []).append(r)

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))

ax = axes[0]
for task in TASKS:
    rows = per_seed[task]
    xs = [float(r["mean_dL_GR"]) for r in rows]
    ys = [float(r["mean_sever"]) for r in rows]
    ax.scatter(xs, ys, color=COL[task], s=42, alpha=0.7,
               edgecolor="black", linewidth=0.6, label=task)
    ax.scatter(float(np.mean(xs)), float(np.mean(ys)),
               color=COL[task], marker="*", s=240, edgecolor="black",
               linewidth=0.8)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel(r"baseline GR  $\,dL/dt$  (tokens / window)")
ax.set_ylabel(r"severship  $-\delta\mathrm{bwd}$")
ax.set_title(r"(A)  sever vs baseline $dL/dt$")
ax.legend(loc="best", fontsize=8)

ax = axes[1]
labels = [r"H1: sever $\to dL^{\mathrm{GR}}/dt$",
          r"H2: sever $\to dR^{\mathrm{GR}}/dt$"]
rho_h1 = float(pooled_h1[0]["spearman_sever_vs_dL_GR"]) if pooled_h1[0]["task"] == TASKS[0] \
    else float(pooled_h1[1]["spearman_sever_vs_dL_GR"])
rho_h2 = float(pooled_h2[0]["spearman_sever_vs_dR_GR"]) if pooled_h2[0]["task"] == TASKS[0] \
    else float(pooled_h2[1]["spearman_sever_vs_dR_GR"])
ci_lo_h1 = float(boot_by_h["H1_sever_vs_dL_GR"][0]["ci_lo"])
ci_hi_h1 = float(boot_by_h["H1_sever_vs_dL_GR"][0]["ci_hi"])
ci_lo_h2 = float(boot_by_h["H2_sever_vs_dR_GR"][0]["ci_lo"])
ci_hi_h2 = float(boot_by_h["H2_sever_vs_dR_GR"][0]["ci_hi"])
rhos = [rho_h1, rho_h2]
err_lo = [rho_h1 - ci_lo_h1, rho_h2 - ci_lo_h2]
err_hi = [ci_hi_h1 - rho_h1, ci_hi_h2 - rho_h2]
xs = np.arange(2)
ax.bar(xs, rhos, yerr=[err_lo, err_hi], capsize=5, color=["#7f7f7f", "#bcbd22"])
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"pooled Spearman  $\rho$")
ax.set_title("(B)  pooled sever-vs-velocity")
ax.set_ylim(-0.6, 0.6)

ax = axes[2]
W = sorted({int(r["window"]) for r in per_window})
for task in TASKS:
    rhos = []
    for w in W:
        match = [r for r in per_window
                 if r["task"] == task and int(r["window"]) == w]
        if match:
            rhos.append(float(match[0]["rho_sever_vs_dL_GR"]))
        else:
            rhos.append(np.nan)
    ax.plot(W, rhos, marker="o", color=COL[task], label=task)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xticks(W)
ax.set_xlabel("window index")
ax.set_ylabel(r"per-window $\rho(\mathrm{sever},\,dL^{\mathrm{GR}}/dt)$")
ax.set_title("(C)  per-window sever-vs-dL/dt")
ax.legend(fontsize=8)

ax = axes[3]
xs = [float(r["mean_dL_GR"]) for r in envelope]
ys = [float(r["mean_sever"]) for r in envelope]
labels = [r["task"] for r in envelope]
for x, y, lbl in zip(xs, ys, labels):
    ax.scatter(x, y, s=180, color=COL[lbl],
               edgecolor="black", linewidth=0.8, label=lbl)
ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
            arrowprops=dict(arrowstyle="->", color="black",
                            lw=1.0, ls="--"))
ax.set_xscale("symlog")
ax.set_xlabel(r"task mean  $\,dL^{\mathrm{GR}}/dt$")
ax.set_ylabel(r"task mean severship  $-\delta\mathrm{bwd}$")
ax.set_title("(D)  cross-task envelope")
ax.axhline(0, color="grey", lw=0.5, ls="--")

plt.tight_layout()
out = os.path.join(FIG, "length_bias_iter124_static_vs_dynamic.pdf")
plt.savefig(out, dpi=160)
plt.savefig(os.path.join(PFIG, "length_bias_iter124_static_vs_dynamic.pdf"),
            dpi=160)
plt.savefig(out.replace(".pdf", ".png"), dpi=160)
print(f"[iter124 fig] wrote {out}")
print(f"[iter124 fig] wrote {out.replace('.pdf', '.png')}")