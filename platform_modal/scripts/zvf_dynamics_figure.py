#!/usr/bin/env python3
"""Render two figures for iter14 ZVF dynamics:

  paper/figures/zvf_dynamics_phase.pdf
      Late-phase-mean ZVF minus early-phase-mean ZVF for every method.
      A positive bar means the method monotonically starves its training
      signal over the run (signal-degeneracy drift).

  paper/figures/zvf_dynamics_leadtime.pdf
      Per-(method, seed) ZVF first-passage-vs-collapse lead time scatter
      for the variance-mitigation collapse-flagged runs.

Inputs: experiments/results/zvf_dynamics_phase.tsv,
        experiments/results/zvf_dynamics_leadtime.tsv
        experiments/results/zvf_dynamics_summary.tsv
Outputs: paper/figures/zvf_dynamics_{phase,leadtime}.pdf and .png
"""
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required for figure rendering", file=sys.stderr)
    raise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "experiments", "results")
FIGS = os.path.join(REPO, "paper", "figures")


def _read_tsv(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    hdr = lines[0].rstrip("\n").split("\t")
    return [dict(zip(hdr, l.rstrip("\n").split("\t"))) for l in lines[1:]]


def _phase_chart():
    rows = _read_tsv(os.path.join(RESULTS, "zvf_dynamics_phase.tsv"))
    rows = [r for r in rows if r["kind"] == "variance_mitigation"]
    methods = [r["method"] for r in rows]
    p1 = [float(r["zvf_phase1_mean"]) for r in rows]
    p2 = [float(r["zvf_phase2_mean"]) for r in rows]
    p3 = [float(r["zvf_phase3_mean"]) for r in rows]
    drift = [a - b for a, b in zip(p3, p1)]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = list(range(len(methods)))
    width = 0.27
    ax.bar([xi - width for xi in x], p1, width, label="early (1/3 steps)", color="#7fbf7b")
    ax.bar(x, p2, width, label="mid (2/3 steps)", color="#fdc086")
    ax.bar([xi + width for xi in x], p3, width, label="late (3/3 steps)", color="#be353c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("phase-mean ZVF", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-method ZVF phase decomposition (5 seeds pooled)\nmonotonic late>early implies training-time signal starvation", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    for xi, d in zip(x, drift):
        ax.text(xi, 1.02, f"Δ={d:+.2f}", ha="center", fontsize=7, color="#333")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "zvf_dynamics_phase.pdf"))
    fig.savefig(os.path.join(FIGS, "zvf_dynamics_phase.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {FIGS}/zvf_dynamics_phase.pdf")


def _leadtime_chart():
    rows = _read_tsv(os.path.join(RESULTS, "zvf_dynamics_leadtime.tsv"))
    rows = [r for r in rows if r["kind"] == "variance_mitigation"]
    if not rows:
        print("  (no lead events; skipping lead-time chart)")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    methods = sorted(set(r["method"] for r in rows))
    cmap = {m: c for m, c in zip(methods, ["#be353c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])}
    for r in rows:
        ax.scatter(int(r["first_pass_step"]), int(r["first_collapse_step"]),
                   color=cmap[r["method"]], s=80, edgecolors="black", linewidth=0.5,
                   label=r["method"] if r is rows[0] else None)
        ax.plot([int(r["first_pass_step"]), int(r["first_pass_step"])],
                [int(r["first_pass_step"]), int(r["first_collapse_step"])],
                color=cmap[r["method"]], alpha=0.4, lw=1.5)
    lim = max(int(r["first_collapse_step"]) for r in rows) + 5
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=1, label="y=x (instant)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("first step at which ZVF crosses θ", fontsize=9)
    ax.set_ylabel("first collapse-flag step", fontsize=9)
    ax.set_title(f"ZVF first-passage vs collapse lead\n({len(rows)} (method,seed) events; variance-mitigation runs)", fontsize=9)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap[m], markersize=8, label=m) for m in methods]
    handles.append(plt.Line2D([0], [0], color="grey", ls="--", lw=1, label="y=x"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "zvf_dynamics_leadtime.pdf"))
    fig.savefig(os.path.join(FIGS, "zvf_dynamics_leadtime.png"), dpi=150)
    plt.close(fig)
    print(f"  -> {FIGS}/zvf_dynamics_leadtime.pdf")


def main():
    os.makedirs(FIGS, exist_ok=True)
    _phase_chart()
    _leadtime_chart()


if __name__ == "__main__":
    main()
