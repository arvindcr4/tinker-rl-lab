"""Iter 48 figure: rises-vs-plateau slopes side-by-side bar chart.

Reads experiments/results/length_bias_iter48_summary.tsv and writes
figures/length_plateau_slopes.pdf with two panels:
  (a) arithmetic_easy: plat_slope bar (GRPO negative, Dr.GRPO positive)
  (b) gsm8k_cot: rise_slope bar (GRPO much more negative than Dr.GRPO)

Bars carry 95% bootstrap CI whiskers from the paired diff columns. Headline
annotation in panel (a) reports the diff + CI; panel (b) reports the same.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_TSV = RES / "length_bias_iter48_summary.tsv"
PAIRED_TSV = RES / "length_bias_iter48_grpo_vs_drgrpo.tsv"

OUT_PDF = FIG_DIR / "length_plateau_slopes.pdf"


def read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    summary = read_tsv(SUMMARY_TSV)
    paired = read_tsv(PAIRED_TSV)

    # Panel A: arithmetic_easy plat_slope
    a_plat = next(r for r in paired
                  if r["task"] == "arithmetic_easy" and r["metric"] == "plat_slope")
    a_sig = next(r for r in paired
                 if r["task"] == "arithmetic_easy" and r["metric"] == "signature")
    # Panel B: gsm8k_cot rise_slope
    b_rise = next(r for r in paired
                  if r["task"] == "gsm8k_cot" and r["metric"] == "rise_slope")
    b_sig = next(r for r in paired
                 if r["task"] == "gsm8k_cot" and r["metric"] == "signature")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel A: post-plateau drift (the "Dr.GRPO inflation" signature)
    ax = axes[0]
    labels = ["GRPO", "Dr.GRPO"]
    vals = [float(a_plat["mean_grpo"]), float(a_plat["mean_drgrpo"])]
    # whiskers = paired diff CI on (drgrpo - grpo) centered on the drgrpo bar
    diff = float(a_plat["mean_diff"])
    ci_lo = float(a_plat["ci_lo"])
    ci_hi = float(a_plat["ci_hi"])
    # asymmetric whiskers: GRPO has no CI (we only diff'd); Dr.GRPO error = CI shifted by -diff
    err_lo = max(0.0, vals[1] - (diff - ci_lo))
    err_hi = max(0.0, (diff + ci_hi) - vals[1])
    errs = [[0.0, 0.0], [err_lo, err_hi]]
    colors = ["#4C72B0", "#DD8452"]
    ax.bar(labels, vals, color=colors, yerr=errs, capsize=6)
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_ylabel("dL/dt in plateau phase")
    ax.set_title(
        "arithmetic_easy: post-plateau length drift\n"
        f"diff = {diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )
    for i, v in enumerate(vals):
        ax.text(i, v + (err_hi if i == 1 else 0.0) + 0.0003,
                f"{v:+.4f}", ha="center", fontsize=9)

    # Panel B: rising-phase compression (the "Dr.GRPO stays longer" signature)
    ax = axes[1]
    vals = [float(b_rise["mean_grpo"]), float(b_rise["mean_drgrpo"])]
    diff = float(b_rise["mean_diff"])
    ci_lo = float(b_rise["ci_lo"])
    ci_hi = float(b_rise["ci_hi"])
    err_lo = max(0.0, vals[1] - (diff - ci_lo))
    err_hi = max(0.0, (diff + ci_hi) - vals[1])
    errs = [[0.0, 0.0], [err_lo, err_hi]]
    ax.bar(["GRPO", "Dr.GRPO"], vals, color=colors, yerr=errs, capsize=6)
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_ylabel("dL/dt in rising phase (more negative = more compression)")
    ax.set_title(
        "gsm8k_cot: pre-plateau compression rate\n"
        f"diff = {diff:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )
    for i, v in enumerate(vals):
        ax.text(i, v - (err_lo if i == 1 else 0.0) - 0.03,
                f"{v:+.3f}", ha="center", fontsize=9)

    fig.suptitle("Iter 48 — Plateau-Anchored Length-Bias Decomposition (Dr.GRPO vs GRPO)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()