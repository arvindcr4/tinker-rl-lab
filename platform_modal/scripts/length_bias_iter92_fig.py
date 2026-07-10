#!/usr/bin/env python3
"""Iter 92 figure script: 2x2 panel of TE / asymmetry statistics.

Panels:
  (a) arithmetic_easy: TE_{L->R} vs TE_{R->L} scatter, GRPO open circles
       vs Dr.GRPO filled circles, with the y=x reference line and a
       per-task legend showing mean (+/- SE) of the asymmetry
       Delta_TE = TE_LR - TE_RL for both algos.
  (b) gsm8k_cot: same layout as (a).
  (c) arithmetic_easy: bar chart of the time-reversal asymmetry
       A_lr and A_rl (with sign of A indicating direction).
  (d) gsm8k_cot: same layout as (c).

Outputs: figures/length_bias_iter92.{pdf,png} and mirror to
paper/figures/.
Stdlib + numpy + matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
FIG = os.path.join(W, "figures")
PAPERFIG = os.path.join(W, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PAPERFIG, exist_ok=True)


def load_perrun():
    rows = []
    with open(os.path.join(RES, "length_bias_iter92_perrun.tsv")) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            d = dict(zip(header, parts))
            for k in ("te_lr", "te_rl", "delta", "ratio", "a_lr", "a_rl"):
                d[k] = float(d[k]) if d[k] != "nan" else float("nan")
            rows.append(d)
    return rows


def by_task_algo(rows, task, algo):
    out = dict(te_lr=[], te_rl=[], delta=[], ratio=[], a_lr=[], a_rl=[])
    for r in rows:
        if r["task"] == task and r["algo"] == algo:
            for k in out:
                v = r[k]
                if not np.isnan(v):
                    out[k].append(v)
    return out


def scatter_panel(ax, rows, task, title):
    grpo = by_task_algo(rows, task, "grpo")
    drgrpo = by_task_algo(rows, task, "dr_grpo")
    ax.scatter(grpo["te_rl"], grpo["te_lr"], s=70, marker="o",
               facecolors="none", edgecolors="C0", linewidths=1.6,
               label="GRPO")
    ax.scatter(drgrpo["te_rl"], drgrpo["te_lr"], s=70, marker="o",
               facecolors="C1", edgecolors="C1", linewidths=1.6,
               label="Dr.GRPO")
    lo = min(min(grpo["te_lr"] + drgrpo["te_lr"] + grpo["te_rl"] + drgrpo["te_rl"]),
             0.0)
    hi = max(max(grpo["te_lr"] + drgrpo["te_lr"] + grpo["te_rl"] + drgrpo["te_rl"]),
             0.5)
    pad = 0.05 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], ls=":", color="grey", lw=1.0,
            label="y = x (TE$_{L\\to R}$ = TE$_{R\\to L}$)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("TE$_{R\\to L}$  [bits]")
    ax.set_ylabel("TE$_{L\\to R}$  [bits]")
    ax.set_title(f"({title})  {task}")
    ax.grid(alpha=0.25)
    g_mean = np.mean(grpo["delta"]) if grpo["delta"] else float("nan")
    d_mean = np.mean(drgrpo["delta"]) if drgrpo["delta"] else float("nan")
    g_se = (np.std(grpo["delta"], ddof=1) / np.sqrt(len(grpo["delta"]))
            if len(grpo["delta"]) > 1 else 0.0)
    d_se = (np.std(drgrpo["delta"], ddof=1) / np.sqrt(len(drgrpo["delta"]))
            if len(drgrpo["delta"]) > 1 else 0.0)
    ax.text(0.04, 0.96,
            f"GRPO   $\\Delta_{{TE}}$ = {g_mean:+.3f} $\\pm$ {g_se:.3f}\n"
            f"Dr.GRPO $\\Delta_{{TE}}$ = {d_mean:+.3f} $\\pm$ {d_se:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8.5)


def bar_panel(ax, rows, task, title):
    grpo = by_task_algo(rows, task, "grpo")
    drgrpo = by_task_algo(rows, task, "dr_grpo")
    metrics = [("a_lr", "A$_{L\\to R}$"),
               ("a_rl", "A$_{R\\to L}$")]
    x = np.arange(len(metrics))
    w = 0.35
    g_means = [np.mean(grpo[k]) if grpo[k] else 0.0 for k, _ in metrics]
    g_ses = [(np.std(grpo[k], ddof=1) / np.sqrt(len(grpo[k])))
             if len(grpo[k]) > 1 else 0.0 for k, _ in metrics]
    d_means = [np.mean(drgrpo[k]) if drgrpo[k] else 0.0 for k, _ in metrics]
    d_ses = [(np.std(drgrpo[k], ddof=1) / np.sqrt(len(drgrpo[k])))
             if len(drgrpo[k]) > 1 else 0.0 for k, _ in metrics]
    ax.bar(x - w/2, g_means, w, yerr=g_ses, color="C0",
           edgecolor="black", label="GRPO", capsize=3, alpha=0.6)
    ax.bar(x + w/2, d_means, w, yerr=d_ses, color="C1",
           edgecolor="black", label="Dr.GRPO", capsize=3, alpha=0.85)
    ax.axhline(0.0, color="grey", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylabel("time-reversal asymmetry $A$")
    ax.set_title(f"({title})  {task}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    ymax = max(0.05, max(g_means + d_means) + max(g_ses + d_ses) + 0.05)
    ymin = min(-0.05, min(g_means + d_means) - max(g_ses + d_ses) - 0.05)
    ax.set_ylim(ymin, ymax)


def main():
    rows = load_perrun()
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    scatter_panel(axs[0, 0], rows, "arithmetic_easy", "a")
    scatter_panel(axs[0, 1], rows, "gsm8k_cot", "b")
    bar_panel(axs[1, 0], rows, "arithmetic_easy", "c")
    bar_panel(axs[1, 1], rows, "gsm8k_cot", "d")
    fig.suptitle("Iter 92 -- Pillar 4: transfer-entropy decomposition "
                 "(TE$_{L\\to R}$ vs TE$_{R\\to L}$, time-reversal $A$)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "length_bias_iter92.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    out2 = os.path.join(PAPERFIG, "length_bias_iter92.pdf")
    fig.savefig(out2)
    print(f"wrote {out} and {out2}")


if __name__ == "__main__":
    main()
