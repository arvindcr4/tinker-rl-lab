#!/usr/bin/env python3
"""Figure for iter 53 P5 sub-field audit + MVE. Two-panel figure:

  (a) per-sub-field coverage & entropy bars  (left, 22 sub-fields ranked by
      coverage then entropy)
  (b) MVE lift waterfall  (right, top-N extension fields by delta_distinct)

Saves PNG + PDF under experiments/results/p5p8/figures/.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUB_TSV = OUT_DIR / "p5_minreport_subfield_audit.tsv"
MVE_TSV = OUT_DIR / "p5_minreport_mve.tsv"

# ---- panel (a): sub-field coverage + H ----
with open(SUB_TSV) as f:
    sub_rows = list(csv.DictReader(f, delimiter="\t"))

# order: zero-coverage first, then vacuous (covered + H=0), then informative
sub_rows.sort(key=lambda r: (float(r["coverage_pct"]) == 0, float(r["H_bits"]) == 0,
                              -float(r["H_bits"]), r["sub_field"]))
sub_labels = [r["sub_field"].replace("_", "\n") for r in sub_rows]
sub_cov = [float(r["coverage_pct"]) for r in sub_rows]
sub_H = [float(r["H_bits"]) for r in sub_rows]
sub_vacuous = [r["vacuous"] == "True" for r in sub_rows]

# color by vacuous state
colors_a = []
for v in sub_vacuous:
    if v:
        colors_a.append("#cccccc")
    elif sub_cov[sub_rows.index(next(r for r in sub_rows if r["vacuous"] == str(v)))] < 1.0:
        colors_a.append("#888888")
    else:
        colors_a.append("#1b7837")

# ---- panel (b): MVE lift waterfall ----
with open(MVE_TSV) as f:
    mve_rows = list(csv.DictReader(f, delimiter="\t"))
# sort by rank ascending (already sorted)
mve_rows.sort(key=lambda r: int(r["rank_by_delta_distinct"]))
mve_labels = [r["extension_field"] for r in mve_rows]
mve_delta_distinct = [int(r["delta_distinct"]) for r in mve_rows]
mve_n_distinct_after = [int(r["n_distinct_with_extension"]) for r in mve_rows]
mve_cumul_lift_pct = [100.0 * (int(r["n_distinct_with_extension"]) - 15) / 98
                       for r in mve_rows]

# color top contributors
colors_b = ["#1b7837" if d >= 30 else "#5aae61" if d >= 10 else "#c7e0b4"
            for d in mve_delta_distinct]

# ---- plot ----
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

# panel (a)
y_pos = np.arange(len(sub_labels))
ax_a.barh(y_pos, sub_cov, color=colors_a, edgecolor="black", linewidth=0.4)
for i, (cov, H, vac) in enumerate(zip(sub_cov, sub_H, sub_vacuous)):
    if not vac:
        ax_a.text(cov + 1, i, f"H={H:.2f}", va="center", fontsize=7)
    elif cov == 100.0:
        ax_a.text(cov - 5, i, "VACUOUS", va="center", ha="right",
                  color="white", fontsize=6, fontweight="bold")
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(sub_labels, fontsize=6.5)
ax_a.set_xlim(0, 115)
ax_a.set_xlabel("Coverage %", fontsize=9)
ax_a.set_title("Per-sub-field coverage & entropy\n(n=98 manifests; vacuous in grey/green)",
               fontsize=9)
ax_a.invert_yaxis()
ax_a.grid(axis="x", alpha=0.3)

# panel (b)
x_pos = np.arange(len(mve_labels))
ax_b.bar(x_pos, mve_delta_distinct, color=colors_b, edgecolor="black", linewidth=0.4)
for i, (d, n, p) in enumerate(zip(mve_delta_distinct, mve_n_distinct_after,
                                   mve_cumul_lift_pct)):
    ax_b.text(i, d + 1.5, f"→ {n}/98", ha="center", fontsize=7)
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([l.replace("_", "\n") for l in mve_labels],
                     fontsize=7, rotation=0)
ax_b.set_ylabel("Δ distinct profiles vs MIN-REPORT-alone", fontsize=9)
ax_b.set_title("MVE: extension-field lift in distinct profiles\n"
                "(content baseline = 15/98; n=98 cells)",
                fontsize=9)
ax_b.grid(axis="y", alpha=0.3)
ax_b.axhline(y=49, color="red", linestyle="--", linewidth=1, alpha=0.6,
             label="n/2 threshold")
ax_b.legend(loc="upper right", fontsize=8)

plt.tight_layout()
out_png = FIG_DIR / "p5_minreport_subfield_audit.png"
out_pdf = FIG_DIR / "p5_minreport_subfield_audit.pdf"
plt.savefig(out_png, dpi=160, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"  -> {out_png.relative_to(ROOT)}")
print(f"  -> {out_pdf.relative_to(ROOT)}")
