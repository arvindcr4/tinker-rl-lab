#!/usr/bin/env python3
"""Figure for iter58 signed ZVF decomposition.
Left: stacked ZVF-/ZVF+ per run, sorted by raw ZVF, annotated by outcome.
Right: ZVF- vs raw ZVF as failure separators (unhealthy vs healthy strips).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RES = "platform_hybrid/experiments/results"
rows = []
with open(f"{RES}/zvf_signed_summary.tsv") as fh:
    for line in fh:
        if line.startswith("#") or line.startswith("source"):
            continue
        p = line.rstrip("\n").split("\t")
        rows.append(dict(source=p[0], label=p[1], acc=float(p[3]), raw=float(p[4]),
                         neg=float(p[5]), pos=float(p[6]), outcome=p[8]))

OC = {"converged": "#2ca02c", "plateau": "#ff7f0e", "drift": "#d62728", "collapse": "#8b0000"}
rows.sort(key=lambda r: r["raw"])
labels = [f"{r['source'][:4]}:{r['label']}" for r in rows]
neg = [r["neg"] for r in rows]
pos = [r["pos"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
y = range(len(rows))
ax1.barh(y, neg, color="#c0392b", label=r"ZVF$^-$ (all-wrong, starvation)", edgecolor="k", lw=.3)
ax1.barh(y, pos, left=neg, color="#2980b9", label=r"ZVF$^+$ (all-correct, saturation)", edgecolor="k", lw=.3)
ax1.set_yticks(list(y))
ax1.set_yticklabels(labels, fontsize=7)
for i, r in enumerate(rows):
    ax1.plot(r["raw"] + 0.012, i, "D", color=OC[r["outcome"]], ms=4)
ax1.set_xlabel("Zero-variance fraction (signed decomposition)")
ax1.set_title(r"Raw ZVF $=$ ZVF$^-$ (pathological) $+$ ZVF$^+$ (benign)", fontsize=9)
ax1.legend(loc="lower right", fontsize=7.5)
ax1.set_xlim(0, 1.06)
oc_handles = [Patch(color=c, label=o) for o, c in OC.items()]
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + oc_handles, fontsize=6.6, loc="lower right", ncol=1)

# right: separator strips
healthy = [r for r in rows if r["outcome"] == "converged"]
unhealthy = [r for r in rows if r["outcome"] != "converged"]
for xi, feat, name in [(0, "raw", "raw ZVF"), (1, "neg", r"ZVF$^-$")]:
    hx = [xi - 0.12] * len(healthy)
    ux = [xi + 0.12] * len(unhealthy)
    ax2.scatter(hx, [r[feat] for r in healthy], c="#2ca02c", s=26, label="healthy" if xi == 0 else None, zorder=3)
    ax2.scatter(ux, [r[feat] for r in unhealthy], c="#c0392b", marker="s", s=26, label="unhealthy" if xi == 0 else None, zorder=3)
# perfect separation line for ZVF-
maxh = max(r["neg"] for r in healthy)
minu = min(r["neg"] for r in unhealthy)
thr = (maxh + minu) / 2
ax2.hlines(thr, 0.7, 1.3, color="k", ls="--", lw=1)
ax2.text(1.32, thr, f"perfect\nsplit\n{thr:.2f}", fontsize=6.5, va="center")
ax2.set_xticks([0, 1])
ax2.set_xticklabels([f"raw ZVF\nAUC=0.40", r"ZVF$^-$" + "\nAUC=1.00"], fontsize=8)
ax2.set_ylabel("value")
ax2.set_title("Failure separability", fontsize=9)
ax2.legend(fontsize=7, loc="upper left")
ax2.set_ylim(-0.03, 1.05)
fig.tight_layout()
fig.savefig("figures/zvf_signed_decomposition.pdf", bbox_inches="tight")
fig.savefig("figures/zvf_signed_decomposition.png", dpi=130, bbox_inches="tight")
print("wrote figures/zvf_signed_decomposition.{pdf,png}")
