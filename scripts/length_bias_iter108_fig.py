#!/usr/bin/env python3
"""Iter 108 figure -- Progress-window + Length-quantile ICCA.

6-panel plot:
  Row 1 (progress-window decomposition):
    (A) backward |CCF| per training-progress window, GRPO vs Dr.GRPO, 2 tasks
    (B) forward |CCF| per training-progress window, GRPO vs Dr.GRPO, 2 tasks
    (C) paired Dr.GRPO - GRPO delta per window (bwd) with 95% bootstrap CI

  Row 2 (length-quantile decomposition):
    (D) backward |CCF| per L-quantile, GRPO vs Dr.GRPO, 2 tasks
    (E) log2 share ratio (q=4 / q=0) per (task, algo) for L-quantile bwd
    (F) paired Dr.GRPO - GRPO delta per L-quantile (bwd) with 95% CI

Reads:  experiments/results/length_bias_iter108_perrun_progress.tsv
        experiments/results/length_bias_iter108_paired_progress.tsv
        experiments/results/length_bias_iter108_summary_length.tsv
        experiments/results/length_bias_iter108_paired_length.tsv
        experiments/results/length_bias_iter108_trend_length.tsv
        experiments/results/length_bias_iter108_trend_progress.tsv
Writes: figures/length_bias_iter108_progress_lquant.{pdf,png}
        paper/figures/length_bias_iter108_progress_lquant.{pdf,png}
"""
from __future__ import annotations
import csv
import os
import sys
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

TASKS = ["arithmetic_easy", "gsm8k_cot"]
COLORS = {"grpo": "#1f77b4", "dr_grpo": "#d62728"}


def load_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            rows.append(dict(zip(header, f)))
    return rows


def to_float(row: dict, *keys: str) -> dict:
    out = dict(row)
    for k in keys:
        if k in out and out[k] != "":
            out[k] = float(out[k])
    return out


def main() -> int:
    perrun_prog = load_tsv(RES / "length_bias_iter108_perrun_progress.tsv")
    paired_prog = [to_float(r, "delta", "ci_lo", "ci_hi", "p")
                   for r in load_tsv(RES / "length_bias_iter108_paired_progress.tsv")]
    trend_prog = [to_float(r, "vals_gr_q0", "vals_gr_q_last",
                            "vals_dr_q0", "vals_dr_q_last",
                            "delta_q0", "delta_q_last",
                            "spearman_rho_delta_vs_window",
                            "spearman_p_delta_vs_window",
                            "spearman_rho_grpo", "spearman_p_grpo",
                            "spearman_rho_drgrpo", "spearman_p_drgrpo")
                  for r in load_tsv(RES / "length_bias_iter108_trend_progress.tsv")]
    summary_len = load_tsv(RES / "length_bias_iter108_summary_length.tsv")
    paired_len = [to_float(r, "delta", "ci_lo", "ci_hi", "p")
                  for r in load_tsv(RES / "length_bias_iter108_paired_length.tsv")]
    trend_len = [to_float(r, "share_bwd_q0", "share_bwd_q4",
                           "share_fwd_q0", "share_fwd_q4", "log2_bwd_q4_q0")
                 for r in load_tsv(RES / "length_bias_iter108_trend_length.tsv")]

    n_w = 4
    n_q = 5
    w_centres = np.arange(n_w) + 0.5
    q_centres = np.arange(n_q) + 0.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # ----------------- ROW 1: PROGRESS WINDOWS -----------------
    # Aggregate per-run per-(task, algo, window) -> mean across seeds
    agg_prog: dict[tuple, dict] = {}
    for r in perrun_prog:
        key = (r["task"], r["algo"], int(r["window"]))
        agg_prog.setdefault(key, []).append(r)
    def mean_window(task, algo, w_idx, field):
        vals = [float(x[field]) for x in agg_prog.get((task, algo, w_idx), [])]
        return float(np.mean(vals)) if vals else 0.0

    # (A) backward |CCF| per progress window
    ax = axes[0, 0]
    width = 0.18
    for ti, task in enumerate(TASKS):
        base = ti * (n_w + 1.2)
        for ai, algo in enumerate(["grpo", "dr_grpo"]):
            ys = [mean_window(task, algo, w, "bwd") for w in range(n_w)]
            xs = base + (np.arange(n_w) - (n_w - 1) / 2.0) + (ai - 0.5) * width
            ax.bar(xs, ys, width=width, color=COLORS[algo], alpha=0.75,
                   label=(algo.upper().replace("_", ".") if ti == 0 else None))
        ax.text(base, ax.get_ylim()[1] * 0.97 if False else 1.05,
                task.replace("_", r"\\_"),
                ha="center", va="bottom", fontsize=10,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(n_w))
    ax.set_xticklabels([f"w{w}" for w in range(n_w)])
    ax.set_xlabel("training-progress window (early → late)")
    ax.set_ylabel(r"$\sum_{k<0}|\mathrm{CCF}(e_L,e_R;k)|$  (per run)")
    ax.set_title("(A) backward innovation $|CCF|$ per progress window",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (B) forward |CCF| per progress window
    ax = axes[0, 1]
    for ti, task in enumerate(TASKS):
        base = ti * (n_w + 1.2)
        for ai, algo in enumerate(["grpo", "dr_grpo"]):
            ys = [mean_window(task, algo, w, "fwd") for w in range(n_w)]
            xs = base + (np.arange(n_w) - (n_w - 1) / 2.0) + (ai - 0.5) * width
            ax.bar(xs, ys, width=width, color=COLORS[algo], alpha=0.75)
        ax.text(base, 1.05, task.replace("_", r"\\_"),
                ha="center", va="bottom", fontsize=10,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(n_w))
    ax.set_xticklabels([f"w{w}" for w in range(n_w)])
    ax.set_xlabel("training-progress window (early → late)")
    ax.set_ylabel(r"$\sum_{k>0}|\mathrm{CCF}(e_L,e_R;k)|$  (per run)")
    ax.set_title("(B) forward innovation $|CCF|$ per progress window",
                 fontsize=10.5, loc="left")

    # (C) paired Dr.GR - GR delta per window (bwd), with bootstrap CI
    ax = axes[0, 2]
    width = 0.32
    for ti, task in enumerate(TASKS):
        rows = sorted([r for r in paired_prog
                        if r["task"] == task and r["side"] == "bwd"],
                       key=lambda r: int(r["window"]))
        deltas = np.array([r["delta"] for r in rows])
        ci_lo = np.array([r["ci_lo"] for r in rows])
        ci_hi = np.array([r["ci_hi"] for r in rows])
        xs = np.arange(n_w) + (ti - 0.5) * (2 * width + 0.05)
        col = "#117733" if ti == 0 else "#882211"
        ax.bar(xs, deltas, width=width * 1.4, color=col, alpha=0.65)
        ax.errorbar(xs, deltas, yerr=[deltas - ci_lo, ci_hi - deltas],
                     fmt="none", ecolor="black", elinewidth=1, capsize=3)
        # mark significance
        for x, d, lo, hi, p in zip(xs, deltas, ci_lo, ci_hi,
                                   [r["p"] for r in rows]):
            if (lo > 0) or (hi < 0):
                ax.text(x, d + (0.04 if d >= 0 else -0.07), "*",
                         ha="center", va=("bottom" if d >= 0 else "top"),
                         fontsize=14)
        # annotate Spearman rho from trend table
        tr = next(t for t in trend_prog if t["task"] == task and t["side"] == "bwd")
        ax.text(xs[-1] + 0.45, deltas[-1],
                 f"ρ={tr['spearman_rho_delta_vs_window']:+.2f}\np={tr['spearman_p_delta_vs_window']:.2f}",
                 fontsize=8, va="center", ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(np.arange(n_w))
    ax.set_xticklabels([f"w{w}" for w in range(n_w)])
    ax.set_xlabel("training-progress window")
    ax.set_ylabel(r"$\Delta$ backward $|CCF|$  (Dr.GRPO - GRPO)")
    ax.set_title("(C) paired Δ by window  (CI: 95%, B=10⁴)",
                 fontsize=10.5, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#117733", alpha=0.65, label="arithmetic_easy"),
                       Patch(color="#882211", alpha=0.65, label="gsm8k_cot")],
              frameon=False, fontsize=8, loc="lower right")

    # ----------------- ROW 2: LENGTH QUANTILES -----------------
    # (D) backward |CCF| per L-quantile
    ax = axes[1, 0]
    width = 0.16
    for ti, task in enumerate(TASKS):
        base = ti * (n_q + 1.2)
        for ai, algo in enumerate(["grpo", "dr_grpo"]):
            # group summary by (task, algo)
            sub = [r for r in summary_len if r["task"] == task and r["algo"] == algo]
            assert len(sub) >= 3, f"expected >=3 seeds per (task,algo), got {len(sub)} for {task},{algo}"
            # average across seeds (rows are individual runs; summary already has per-quantile fields)
            ys = []
            for q in range(n_q):
                vals = [float(s[f"q{q}_bwd"]) for s in sub if s[f"q{q}_bwd"] != ""]
                ys.append(float(np.mean(vals)))
            xs = base + (np.arange(n_q) - (n_q - 1) / 2.0) + (ai - 0.5) * width
            ax.bar(xs, ys, width=width, color=COLORS[algo], alpha=0.75,
                   label=(algo.upper().replace("_", ".") if ti == 0 else None))
        ax.text(base, 1.05, task.replace("_", r"\\_"),
                ha="center", va="bottom", fontsize=10,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(np.arange(n_q))
    ax.set_xticklabels([f"q{q}" for q in range(n_q)])
    ax.set_xlabel("length quintile (low → high)")
    ax.set_ylabel(r"$\sum_{k<0}|\mathrm{CCF}(e_L,e_R;k)|$  (per run)")
    ax.set_title("(D) backward $|CCF|$ per L-quantile",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # (E) log2 share ratio (q=4 / q=0) per (task, algo)
    ax = axes[1, 1]
    bar_x = np.arange(len(TASKS))
    bar_w = 0.32
    log_ratios = {a: [] for a in ["grpo", "dr_grpo"]}
    for ti, task in enumerate(TASKS):
        for ai, a in enumerate(["grpo", "dr_grpo"]):
            r = next(t for t in trend_len if t["task"] == task and t["algo"] == a)
            lr = float(r["log2_bwd_q4_q0"])
            log_ratios[a].append(lr)
            x = ti - 0.5 * bar_w + ai * bar_w
            ax.bar(x, lr, width=bar_w, color=COLORS[a], alpha=0.75,
                   label=(a.upper().replace("_", ".") if ti == 0 else None))
            ax.text(x, lr + (0.05 if lr >= 0 else -0.07), f"{lr:+.2f}",
                     ha="center", va=("bottom" if lr >= 0 else "top"),
                     fontsize=9)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([t.replace("_", r"\\_") for t in TASKS])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"$\log_2(\mathrm{bwd\ share}_{q=4}\,/\,\mathrm{bwd\ share}_{q=0})$  (L-quantile)")
    ax.set_title("(E) high-L / low-L backward share ratio",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # (F) paired Dr.GR - GR delta per L-quantile (bwd) with CI
    ax = axes[1, 2]
    width = 0.32
    for ti, task in enumerate(TASKS):
        rows = sorted([r for r in paired_len
                        if r["task"] == task and r["side"] == "bwd"],
                       key=lambda r: int(r["q"]))
        deltas = np.array([r["delta"] for r in rows])
        ci_lo = np.array([r["ci_lo"] for r in rows])
        ci_hi = np.array([r["ci_hi"] for r in rows])
        xs = np.arange(n_q) + (ti - 0.5) * (2 * width + 0.05)
        col = "#117733" if ti == 0 else "#882211"
        ax.bar(xs, deltas, width=width * 1.4, color=col, alpha=0.65)
        ax.errorbar(xs, deltas, yerr=[deltas - ci_lo, ci_hi - deltas],
                     fmt="none", ecolor="black", elinewidth=1, capsize=3)
        for x, d, lo, hi in zip(xs, deltas, ci_lo, ci_hi):
            if (lo > 0) or (hi < 0):
                ax.text(x, d + (0.04 if d >= 0 else -0.07), "*",
                         ha="center", va=("bottom" if d >= 0 else "top"),
                         fontsize=14)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(np.arange(n_q))
    ax.set_xticklabels([f"q{q}" for q in range(n_q)])
    ax.set_xlabel("length quintile")
    ax.set_ylabel(r"$\Delta$ backward $|CCF|$  (Dr.GRPO - GRPO)")
    ax.set_title("(F) paired Δ by L-quantile  (CI: 95%, B=10⁴)",
                 fontsize=10.5, loc="left")
    ax.legend(handles=[Patch(color="#117733", alpha=0.65, label="arithmetic_easy"),
                       Patch(color="#882211", alpha=0.65, label="gsm8k_cot")],
              frameon=False, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = FIGS / "length_bias_iter108_progress_lquant.pdf"
    out_png = FIGS / "length_bias_iter108_progress_lquant.png"
    fig.savefig(out)
    fig.savefig(out_png, dpi=130)
    print(f"[iter108 fig] wrote {out}")
    import shutil
    for ext in ("pdf", "png"):
        shutil.copy(out.with_suffix(f".{ext}"),
                    PAPER_FIGS / out.with_suffix(f".{ext}").name)
    print(f"[iter108 fig] mirrored -> paper/figures/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())