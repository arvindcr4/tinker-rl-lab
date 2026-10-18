"""Re-plot P2 Fig. 'Five-anchor ZVF-proxy diagnostic' from the checked-in TSV (Deep Review finding 122:
negative tick labels lost their minus signs; the original PDF output is git-ignored and absent from clean checkouts).

Input:  platform_hybrid/experiments/results/zvf_scaling_cross_pillar.tsv
Output: platform_hybrid/paper/figures/zvf_scaling_cross_pillar.png
Plot logic follows platform_modal/scripts/zvf_scaling_coupling.py (figure block only; that script also rewrites TSVs/sections).
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
ROOT = pathlib.Path(__file__).resolve().parents[2]
rows = list(
    csv.DictReader(
        open(ROOT / "platform_hybrid/experiments/results/zvf_scaling_cross_pillar.tsv"),
        delimiter="\t",
    )
)
col = ["#d62728" if r["is_collapse"] == "True" else "#1f77b4" for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
ax.barh([r["model"] for r in rows], [float(r["zvf_proxy"]) for r in rows], color=col)
ax.set_xlabel("ZVF-proxy = (frac_below_0p1 + zero_fraction) / 2")
ax.set_title("(A) ZVF-proxy per anchor (red = collapse)")
ax.grid(alpha=0.3, axis="x")
ax = axes[1]
for r, c in zip(rows, col):
    x, y = float(r["frac_below_0p1"]), float(r["late_minus_peak"])
    ax.scatter(x, y, s=120, c=c, edgecolors="black", linewidths=0.5)
    off = {
        "Qwen3.5-4B": (8, 6),
        "Llama-3.1-8B-Instruct": (8, -4),
        "DeepSeek-V3.1": (8, -12),
        "Nemotron-120B": (-70, 8),
    }
    ax.annotate(
        r["model"],
        (x, y),
        textcoords="offset points",
        xytext=off.get(r["model"], (8, 5)),
        fontsize=8,
    )
ax.set_xlabel("frac_below_0p1 (steps with held-out < 0.1)")
ax.set_ylabel("late_minus_peak (negative = collapse)")
ax.set_title("(B) ZVF-proxy vs collapse slope")
ax.axhline(0, color="grey", linewidth=0.5)
ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "platform_hybrid/paper/figures/zvf_scaling_cross_pillar.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
