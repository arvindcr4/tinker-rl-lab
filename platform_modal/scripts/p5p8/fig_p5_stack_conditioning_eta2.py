#!/usr/bin/env python3
"""
P5 stack-conditioning eta^2 figure: three horizontal bar panels
(method / step / prompt) with point estimate + bootstrap 95% CI whisker.
Reads p5_stack_conditioning_eta2_boot.tsv, writes figures/p5_stack_conditioning_eta2.{png,pdf}.
"""
import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "experiments/results/p5p8"
FIG  = ROOT / "experiments/results/p5p8/figures"
FIG.mkdir(parents=True, exist_ok=True)

rows = []
with (RES / "p5_stack_conditioning_eta2_boot.tsv").open() as fh:
    rdr = csv.DictReader(fh, delimiter="\t")
    for r in rdr:
        rows.append({"axis": r["axis"], "mean": float(r["mean"]),
                     "lo": float(r["lo"]), "hi": float(r["hi"])})

fig, ax = plt.subplots(figsize=(6.0, 3.0))
labels = ["method (k=4)", "step (k=40)", "prompt (k=16)"]
means  = [r["mean"] for r in rows]
los    = [r["lo"] for r in rows]
his    = [r["hi"] for r in rows]
y      = list(range(3))
colors = ["#2c7fb8", "#7fcdbb", "#edf8b1"]
ax.barh(y, means, color=colors, edgecolor="black", height=0.55)
for i, (m, lo, hi) in enumerate(zip(means, los, his)):
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none",
                ecolor="black", capsize=4, lw=1.4)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.axvline(0.10, ls="--", lw=1.0, color="red", label="P5 threshold 0.10")
ax.set_xlabel("eta^2 (cell-mean reward)  [95% bootstrap CI]")
ax.set_title("Stack-Conditioning Quantification (N2 four-method same-stack)\nP5: algorithm label explains <0.1% of variance on a fixed stack")
ax.legend(loc="lower right", framealpha=0.9)
for i, m in enumerate(means):
    ax.text(m + 0.002, i, f"{m:.4f}", va="center", fontsize=8.5)
fig.tight_layout()
out_png = FIG / "p5_stack_conditioning_eta2.png"
out_pdf = FIG / "p5_stack_conditioning_eta2.pdf"
fig.savefig(out_png, dpi=140)
fig.savefig(out_pdf)
print(f"[OK] wrote {out_png} and {out_pdf}")
