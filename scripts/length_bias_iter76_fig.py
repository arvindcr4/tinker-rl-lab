"""length_bias_iter76_fig.py — 3-panel diagnostic figure.

Renders:
  figures/length_bias_iter76.{pdf,png}
Mirrors to paper/figures/length_bias_iter76.{pdf,png}.

Panels:
  A. Half-life τ (steps) per (algo, task), conditioned on shock sign.
  B. Damping ratio ζ per (algo, task).  ζ=1 horizontal reference.
  C. Phase-plane loop area ∮R dL per (algo, task).
"""
from __future__ import annotations
import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "figures")
PAPER_FIG = os.path.join(ROOT, "paper", "figures")


def load_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    half = load_tsv(os.path.join(RES, "length_bias_iter76_halflife.tsv"))
    damp = load_tsv(os.path.join(RES, "length_bias_iter76_damping.tsv"))
    loop = load_tsv(os.path.join(RES, "length_bias_iter76_looparea.tsv"))

    tasks = ["gsm8k_cot", "arith_easy"]
    algo_order = ["grpo", "dr_grpo"]
    algo_label = {"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}
    color = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}

    def bytask(rows, key):
        out = {(t, a): [] for t in tasks for a in algo_order}
        for r in rows:
            v = r.get(key)
            if v in (None, "", "nan"):
                continue
            out[(r["task"], r["algo"])].append(float(v))
        return out

    hl_up = bytask(half, "tau_up")
    hl_dn = bytask(half, "tau_dn")
    hl_all = bytask(half, "tau_all")
    zt = bytask(damp, "zeta_all")
    lp = bytask(loop, "loop_area")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: half-life bar chart, grouped by task, with up/dn conditioning
    ax = axes[0]
    width = 0.18
    x = np.arange(len(tasks))
    for j, cond in enumerate(("up", "dn")):
        d = hl_up if cond == "up" else hl_dn
        for k, a in enumerate(algo_order):
            offset = (j * 2 + k - 1.5) * width
            vals = d.get((tasks[0], a)) if cond == "up" else d.get((tasks[0], a))
            # use the same lookup for both task rows
            grp = []
            for t in tasks:
                vals = d.get((t, a), [])
                grp.append(float(np.mean(vals)) if vals else np.nan)
            ax.bar(x + offset, grp, width, color=color[a],
                   alpha=0.45 if cond == "dn" else 0.95,
                   label=f"{algo_label[a]} ({'+' if cond=='up' else '−'}dR)"
                   if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in tasks])
    ax.set_ylabel(r"Half-life $\tau$ (steps)")
    ax.set_title("A. Reward-shock half-life")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: damping ratio
    ax = axes[1]
    x = np.arange(len(tasks))
    width = 0.32
    for k, a in enumerate(algo_order):
        grp = []
        for t in tasks:
            vals = zt.get((t, a), [])
            grp.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (k - 0.5) * width, grp, width, color=color[a],
               label=algo_label[a])
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label=r"$\zeta = 1$ (crit.)")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in tasks])
    ax.set_ylabel(r"Damping ratio $\zeta$")
    ax.set_title("B. 2nd-order damping")
    ax.legend(fontsize=8)

    # Panel C: loop area
    ax = axes[2]
    x = np.arange(len(tasks))
    width = 0.32
    for k, a in enumerate(algo_order):
        grp = []
        for t in tasks:
            vals = lp.get((t, a), [])
            grp.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (k - 0.5) * width, grp, width, color=color[a],
               label=algo_label[a])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in tasks])
    ax.set_ylabel(r"Loop area $\oint R\,dL$")
    ax.set_title("C. Phase-plane loop area")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Iter 76 — Reward-shock mean-reversion & phase-plane dissipativity",
        fontsize=12, y=1.02)
    fig.tight_layout()

    out_pdf = os.path.join(FIG, "length_bias_iter76.pdf")
    out_png = os.path.join(FIG, "length_bias_iter76.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=110)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

    # mirror to paper/figures
    for src, dst in [(out_pdf, os.path.join(PAPER_FIG, "length_bias_iter76.pdf")),
                     (out_png, os.path.join(PAPER_FIG, "length_bias_iter76.png"))]:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
        print(f"mirrored to {dst}")


if __name__ == "__main__":
    main()