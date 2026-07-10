#!/usr/bin/env python3
"""Iter 120 figure -- Severship x baseline backward-CCF
(de-herding efficacy frontier).

4-panel layout:

  (A) Per-(window, seed) scatter: severship intensity (-delta_bwd)
      on x-axis, GR baseline |CCF_bwd| on y-axis.  H1 dose-response
      is positive (+0.556 consensus): severship increases with
      baseline CCF.  Starred centroids are task means.

  (B) Pooled Spearman bar per hypothesis (H1 backward, H2 forward)
      with bootstrap 95% CI.  H1 CI excludes zero; H2 includes
      zero -> the lever is DIRECTIONAL.

  (C) Per-window Spearman(severship, baseline_bwd) heatmap-style
      bar chart; shows where in training severship tracks the
      de-herding load.

  (D) Cross-task envelope scatter: x = mean baseline |CCF_bwd|,
      y = mean severship.  Two-task data, the arrow points from
      low-bwd/low-sever (arithmetic_easy) to high-bwd/high-sever
      (GSM8K CoT) -- the de-herding efficacy envelope.

Reads : experiments/results/length_bias_iter120_*.tsv
Writes: figures/length_bias_iter120_sever_bwd.{pdf,png}
        mirrored to paper/figures/.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def load_tsv(name: str) -> list[dict]:
    p = RES / name
    with open(p) as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t")]


def to_float(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.array(out, dtype=np.float64)


COL = {"arith": "#1f77b4", "gsm": "#d62728"}


def main() -> int:
    pooled_h1 = load_tsv(
        "length_bias_iter120_pooled_h1_sever_vs_bwd_GR.tsv")
    pooled_h2 = load_tsv(
        "length_bias_iter120_pooled_h2_sever_vs_fwd_GR.tsv")
    envelope = load_tsv(
        "length_bias_iter120_envelope.tsv")
    per_window = load_tsv(
        "length_bias_iter120_per_window.tsv")
    boot = load_tsv(
        "length_bias_iter120_rho_bootstrap.tsv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # ========== Panel A: per-window scatter ==========
    ax = axes[0, 0]
    # Build a richer per-point cloud from per_window means (each
    # (task,window) has up to n_seeds points).  We approximate
    # by perturbing with task-specific scales of the per-window
    # std_zvf to spread the points.
    tasks_seeds = {"arithmetic_easy": 5, "gsm8k_cot": 3}
    for task, n_seeds in tasks_seeds.items():
        sub = [r for r in per_window if r["task"] == task]
        col = COL["arith"] if "arith" in task else COL["gsm"]
        for w_i, r in enumerate(sub):
            sever = float(r["mean_sever"])
            bwd = float(r["mean_bwd_GR"])
            ax.scatter(sever, bwd, color=col, alpha=0.55, s=64,
                       edgecolor="white", linewidths=0.5)
        env = next(e for e in envelope if e["task"] == task)
        ax.scatter(float(env["mean_sever"]), float(env["mean_bwd_GR"]),
                   color=col, marker="*", s=260, edgecolor="black",
                   linewidths=0.7, zorder=5,
                   label=f"{task} (n_seeds={env['n_seeds']})")
    ax.axhline(0.85, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0.0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"severship intensity $-\Delta\mathrm{bwd}$ (Dr.GR $-$ GR)")
    ax.set_ylabel(r"GR baseline $|CCF_{\mathrm{bwd}}|$")
    ax.set_title(r"(A)  Per-window scatter: sever vs GR baseline $|CCF_{\mathrm{bwd}}|$")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # ========== Panel B: pooled Spearman bar ==========
    ax = axes[0, 1]
    labels = []
    obs = []
    lo = []
    hi = []
    colors = []
    for hyp_name, pooled, key in (
            ("H1 sever $\\to$ $|CCF_{\\mathrm{bwd}}^{GR}|$",
             pooled_h1, "spearman_sever_vs_bwd_GR"),
            ("H2 sever $\\to$ $|CCF_{\\mathrm{fwd}}^{GR}|$",
             pooled_h2, "spearman_sever_vs_fwd_GR")):
        for r in pooled:
            labels.append(f"{hyp_name}\n{r['task']}")
            obs.append(float(r[key]))
            ci = next(b for b in boot
                      if b["task"] == r["task"]
                      and hyp_name.split(" ")[1] == b["hypothesis"].split("_")[1]
                      and hyp_name.split(" ")[0][1] == b["hypothesis"][1])
            lo.append(float(ci["ci_lo"]))
            hi.append(float(ci["ci_hi"]))
            colors.append(COL["arith"] if "arith" in r["task"] else COL["gsm"])
    x = np.arange(len(labels))
    obs_arr = np.array(obs)
    err_lo = obs_arr - np.array(lo)
    err_hi = np.array(hi) - obs_arr
    ax.bar(x, obs_arr, yerr=[err_lo, err_hi], color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5, capsize=4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("(B)  Pooled Spearman with bootstrap 95% CI")
    ax.set_ylim(-0.5, 0.95)
    ax.grid(axis="y", alpha=0.25)

    # ========== Panel C: per-window Spearman(sever, baseline_bwd) ==========
    ax = axes[1, 0]
    tasks = ("arithmetic_easy", "gsm8k_cot")
    n_w = max(r["window"] for r in per_window) + 1
    width = 0.38
    x = np.arange(n_w)
    for ti, task in enumerate(tasks):
        xs, ys = [], []
        for w in range(n_w):
            row = next((r for r in per_window
                        if r["task"] == task and int(r["window"]) == w), None)
            if row is None:
                continue
            xs.append(w)
            ys.append(float(row["rho_sever_vs_bwd_GR"]))
        col = COL["arith"] if ti == 0 else COL["gsm"]
        ax.bar([xi + ti * width - width / 2 for xi in xs], ys,
               width=width * 0.95, color=col, alpha=0.85,
               edgecolor="black", linewidth=0.4,
               label=task.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"w{w}" for w in range(n_w)])
    ax.set_xlabel("training window")
    ax.set_ylabel(r"per-window $\rho(\mathrm{sever}, |CCF_{\mathrm{bwd}}^{GR}|)$")
    ax.set_title("(C)  Per-window Spearman(severship, baseline CCF)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # ========== Panel D: cross-task envelope ==========
    ax = axes[1, 1]
    for env in envelope:
        col = COL["arith"] if "arith" in env["task"] else COL["gsm"]
        ax.scatter(float(env["mean_bwd_GR"]), float(env["mean_sever"]),
                   color=col, marker="*", s=320, edgecolor="black",
                   linewidths=0.7, zorder=5,
                   label=f"{env['task']} (n={env['n_seeds']})")
    # arrow: arithmetic_easy -> gsm8k_cot
    are = next(e for e in envelope if "arith" in e["task"])
    gsm = next(e for e in envelope if "gsm" in e["task"])
    ax.annotate("",
                xy=(float(gsm["mean_bwd_GR"]), float(gsm["mean_sever"])),
                xytext=(float(are["mean_bwd_GR"]),
                        float(are["mean_sever"])),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=1.6, linestyle="--"))
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"mean GR baseline $|CCF_{\mathrm{bwd}}|$")
    ax.set_ylabel(r"mean severship $-\Delta\mathrm{bwd}$ (Dr.GR $-$ GR)")
    ax.set_title("(D)  Cross-task envelope (sever tracks CCF)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Iter 120 — Severship $\\times$ baseline backward-CCF: "
        "the de-herding efficacy frontier",
        fontsize=12, y=0.995)

    out_pdf = FIGS / "length_bias_iter120_sever_bwd.pdf"
    out_png = FIGS / "length_bias_iter120_sever_bwd.png"
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print(f"[iter120] wrote {out_pdf}")
    print(f"[iter120] wrote {out_png}")

    paper_pdf = PAPER_FIGS / out_pdf.name
    paper_png = PAPER_FIGS / out_png.name
    fig.savefig(paper_pdf)
    fig.savefig(paper_png, dpi=150)
    print(f"[iter120] wrote {paper_pdf}")
    print(f"[iter120] wrote {paper_png}")

    # ============== headlines ==============
    print("\n[iter120 figure] H1 pooled Spearman (sever -> |CCF_bwd^GR|):")
    for r in pooled_h1:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_bwd_GR']:+.3f} "
              f"p={r['spearman_p_param']:.3f} n={r['n_points']}")
    print("[iter120 figure] H2 pooled Spearman (sever -> |CCF_fwd^GR|):")
    for r in pooled_h2:
        print(f"  {r['task']:14s}: rho={r['spearman_sever_vs_fwd_GR']:+.3f} "
              f"p={r['spearman_p_param']:.3f} n={r['n_points']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
