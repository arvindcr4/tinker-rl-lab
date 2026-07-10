#!/usr/bin/env python3
"""Iter 104 figure -- Conditional ICCA by reward quantile.

3-panel plot:
  (A) backward |CCF| absolute value as a function of reward quantile,
      GRPO vs Dr.GRPO, color-coded by algorithm.  Task-split.
  (B) Share shift: ratio (q_high / q_low) of the backward |CCF|, on a
      log2 axis.  A negative bar = "Dr.GRPO up-weights low-reward regimes";
      positive bar = "Dr.GRPO up-weights high-reward regimes".
  (C) Per-quantile Dr.GRPO - GRPO paired delta with bootstrap 95% CI,
      sub-panelled per task.

Reads:  experiments/results/length_bias_iter104_summary.tsv
        experiments/results/length_bias_iter104_paired.tsv
Writes: figures/length_bias_iter104_qreg.{pdf,png}
        paper/figures/length_bias_iter104_qreg.{pdf,png}
"""
from __future__ import annotations
import os, json, sys
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


def load_summary():
    rows = []
    with open(RES / "length_bias_iter104_summary.tsv") as fh:
        header = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            rows.append(dict(zip(header, f)))
    return rows


def load_paired():
    rows = []
    with open(RES / "length_bias_iter104_paired.tsv") as fh:
        header = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            rec = dict(zip(header, f))
            for k in ["q", "n", "delta", "ci_lo", "ci_hi", "p"]:
                rec[k] = float(rec[k])
            rows.append(rec)
    return rows


def main() -> int:
    summary = load_summary()
    paired = load_paired()

    TASKS = ["arithmetic_easy", "gsm8k_cot"]
    N_Q = 5
    q_centres = np.arange(N_Q) + 0.5

    # -- panel A: backward |CCF| per quantile per algo
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.4, 1, 1.6]})

    axA = axes[0]
    width = 0.36
    colors = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}
    task_centres = np.arange(len(TASKS)) * (N_Q + 1.5)
    for ti, task in enumerate(TASKS):
        sub_g = [r for r in summary if r["task"] == task and r["algo"] == "grpo"]
        sub_d = [r for r in summary if r["task"] == task and r["algo"] == "dr_grpo"]
        sub_g = sorted(sub_g, key=lambda r: int(r["q"]))
        sub_d = sorted(sub_d, key=lambda r: int(r["q"]))
        base = task_centres[ti]
        xs_g = base + (np.arange(N_Q) - (N_Q - 1) / 2.0) - width / 2.0
        xs_d = base + (np.arange(N_Q) - (N_Q - 1) / 2.0) + width / 2.0
        bwd_g = np.array([float(r["bwd_abs_mean"]) for r in sub_g])
        bwd_d = np.array([float(r["bwd_abs_mean"]) for r in sub_d])
        axA.bar(xs_g, bwd_g, width=width, color=colors["grpo"], alpha=0.7,
                label=("GRPO" if ti == 0 else None))
        axA.bar(xs_d, bwd_d, width=width, color=colors["dr_grpo"], alpha=0.7,
                label=("Dr.GRPO" if ti == 0 else None))
        task_tag = task.replace("_", r"\\_")
        axA.text(base, axA.get_ylim()[1] * 0.97 if False else 0,
                 task_tag, ha="center", va="top", fontsize=10,
                 transform=axA.get_transform())
    # x-axis: show q labels centered under each task group
    xticks, xticklabels = [], []
    for ti in range(len(TASKS)):
        for q in range(N_Q):
            xticks.append(task_centres[ti] + (q - (N_Q - 1) / 2.0))
            xticklabels.append(f"q{q}")
    axA.set_xticks(xticks)
    axA.set_xticklabels(xticklabels)
    # task separators / labels
    for ti, t in enumerate(TASKS):
        axA.axvline(task_centres[ti], color="grey", lw=0.3, ymin=0.05, ymax=0.95)
        axA.text(task_centres[ti], 1.02, t.replace("_", r"\\_"),
                 transform=axA.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=10)
    axA.set_xlim(task_centres[0] - 3, task_centres[-1] + 3)
    axA.set_xlabel("reward quantile bin")
    axA.set_ylabel(r"$\sum_{k<0}|\mathrm{CCF}(e_L,e_R;k)|$  (per run)")
    axA.set_title("(A) backward innovation CCF by reward quintile",
                  fontsize=11, loc="left")
    axA.legend(frameon=False, loc="upper left")

    # -- panel B: log2 share ratio q=4 / q=0
    axB = axes[1]
    share = {(t, a): np.zeros(N_Q) for t in TASKS for a in ["grpo", "dr_grpo"]}
    for r in summary:
        share[(r["task"], r["algo"])][int(r["q"])] = float(r["bwd_ratio"])
    bar_x = np.arange(len(TASKS))
    bar_w = 0.32
    log_ratios = {a: [] for a in ["grpo", "dr_grpo"]}
    for ti, t in enumerate(TASKS):
        for ai, a in enumerate(["grpo", "dr_grpo"]):
            s = share[(t, a)]
            eps = 1e-4
            lr = float(np.log2((s[4] + eps) / (s[0] + eps)))
            log_ratios[a].append(lr)
            x = ti - 0.5 * bar_w + ai * bar_w
            axB.bar(x, lr, width=bar_w, color=colors[a], alpha=0.7,
                    label=("GRPO" if ti == 0 else None))
            axB.text(x, lr + (0.05 if lr >= 0 else -0.07), f"{lr:+.2f}",
                     ha="center", va=("bottom" if lr >= 0 else "top"), fontsize=9)
    axB.set_xticks(bar_x)
    axB.set_xticklabels([t.replace("_", r"\\_") for t in TASKS])
    axB.set_ylabel(r"$\log_2(bwd\_share_{q=4}/bwd\_share_{q=0})$")
    axB.axhline(0, color="k", lw=0.5)
    axB.set_title("(B) high-R / low-R backward share ratio",
                  fontsize=11, loc="left")
    axB.legend(frameon=False, loc="lower right")
    print("[iter104 fig] log2(q4/q0) bwd share:")
    for ti, t in enumerate(TASKS):
        print(f"  {t}: GRPO={log_ratios['grpo'][ti]:+.3f}, "
              f"Dr.GRPO={log_ratios['dr_grpo'][ti]:+.3f}, "
              f"diff={log_ratios['dr_grpo'][ti] - log_ratios['grpo'][ti]:+.3f}")

    # -- panel C: paired delta Dr.GRPO - GRPO at each q for bwd
    axC = axes[2]
    sub = [r for r in paired if r["side"] == "bwd"]
    sub_by_task = {t: sorted([r for r in sub if r["task"] == t],
                              key=lambda r: int(r["q"]))
                   for t in TASKS}
    width = 0.35
    for ti, t in enumerate(TASKS):
        rows = sub_by_task[t]
        xs = np.arange(N_Q) + (ti - 0.5) * (2 * width + 0.05)
        deltas = np.array([r["delta"] for r in rows])
        ci_lo = np.array([r["ci_lo"] for r in rows])
        ci_hi = np.array([r["ci_hi"] for r in rows])
        colors_t = ["#117733" if ti == 0 else "#882211"]
        axC.bar(xs, deltas, width=width * 1.4, color=colors_t[0],
                alpha=0.6,
                label=(t.replace("_", r"\\_") if False else None))
        # CI
        axC.errorbar(xs, deltas,
                     yerr=[deltas - ci_lo, ci_hi - deltas],
                     fmt="none", ecolor="black", elinewidth=1, capsize=3)
        axC.axhline(0, color="grey", lw=0.5)
        # Mark significance
        for x, d, lo, hi, p in zip(xs, deltas, ci_lo, ci_hi, [r["p"] for r in rows]):
            if (lo > 0) or (hi < 0):
                axC.text(x, d + (0.05 if d >= 0 else -0.07), "*",
                         ha="center", va=("bottom" if d >= 0 else "top"),
                         fontsize=14)
    # Reposition: one task per sub-axis (we'll just show them side by side)
    axC.set_xticks(np.arange(N_Q))
    axC.set_xticklabels([f"q{q}" for q in range(N_Q)])
    axC.set_xlabel("reward quantile bin")
    axC.set_ylabel(r"$\Delta$ backward |CCF|  (Dr.GRPO - GRPO)")
    axC.set_title("(C) paired delta  (CI: 95% paired bootstrap B=2000)",
                  fontsize=11, loc="left")

    # Add task legend manually
    from matplotlib.patches import Patch
    handles = [Patch(color="#117733", alpha=0.6, label="arithmetic\\_easy"),
               Patch(color="#882211", alpha=0.6, label="gsm8k\\_cot")]
    axC.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = FIGS / "length_bias_iter104_qreg.pdf"
    out_png = FIGS / "length_bias_iter104_qreg.png"
    fig.savefig(out)
    fig.savefig(out_png, dpi=130)
    print(f"[iter104 fig] wrote {out}")
    # mirror to paper/figures/
    import shutil
    for ext in ("pdf", "png"):
        shutil.copy(out.with_suffix(f".{ext}"), PAPER_FIGS / out.with_suffix(f".{ext}").name)
    print(f"[iter104 fig] mirrored -> paper/figures/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
