#!/usr/bin/env python3
"""Figure for P7 unified controller bank (iter 51).
Pareto frontier: mean_savings_n10 vs seed_CV_total_G,
highlighting the best calibrated theta (0.65, 0.70) and the best
savings theta (0.55, 0.60)."""
from __future__ import annotations
import csv
import json
import pathlib

ROOT = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
OUT = ROOT / "platform_hybrid/experiments/results/p5p8/figures"
OUT.mkdir(parents=True, exist_ok=True)

summary = list(csv.DictReader(open(ROOT / "platform_hybrid/experiments/results/p5p8/p7_unified_controller_summary.tsv"), delimiter="\t"))
for r in summary:
    r["mean_savings_n10"] = float(r["mean_savings_n10"])
    r["seed_cv_total_G"] = float(r["seed_cv_total_G"])
    r["tau_esc"] = float(r["tau_esc"])
    r["tau_des"] = float(r["tau_des"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.6, 4.4))
# Pareto points (all 37 in this sweep have headroom=0 and CV<=0.10)
xs = [r["seed_cv_total_G"] for r in summary]
ys = [r["mean_savings_n10"] for r in summary]
labels = [r["tau_esc"] for r in summary]
ax.scatter(xs, ys, c="#1f77b4", alpha=0.55, s=42, edgecolor="white",
           linewidth=0.6, label="Pareto (CV ≤ 0.10, headroom=0)")

best = next(r for r in summary if r["tau_esc"] == 0.55 and r["tau_des"] == 0.60)
ax.scatter([best["seed_cv_total_G"]], [best["mean_savings_n10"]],
           c="#2ca02c", s=130, marker="*", zorder=5, label="Best savings (0.55,0.60): +0.30")
cal = next(r for r in summary if r["tau_esc"] == 0.65 and r["tau_des"] == 0.70)
ax.scatter([cal["seed_cv_total_G"]], [cal["mean_savings_n10"]],
           c="#d62728", s=130, marker="D", zorder=5, label="Calibrated (0.65,0.70): +0.14, CV=0.051")

# annotate Pareto-frontier (greedy by savings)
ordered = sorted(summary, key=lambda r: r["mean_savings_n10"], reverse=True)
fx = [r["seed_cv_total_G"] for r in ordered]
fy = [r["mean_savings_n10"] for r in ordered]
ax.plot(fx, fy, c="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.55)

ax.set_xlabel("Seed CV of total compute (5 GRPO seeds)")
ax.set_ylabel("Mean savings (vs always-G=8)")
ax.set_title("Unified Adaptive-G Controller Bank — Pareto frontier\n(N10 5-seed panel; headroom-bad = 0 throughout)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = OUT / "p7_unified_controller_bank"
fig.savefig(f"{out}.png", dpi=160)
fig.savefig(f"{out}.pdf")
print(f"Wrote {out}.png and {out}.pdf")
