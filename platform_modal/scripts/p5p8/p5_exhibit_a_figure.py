#!/usr/bin/env python3
"""P5 Exhibit A: per-MIN-REPORT-item coverage figure (JOB B SYNTH, iter 20).

The iter-1 audit produced `minreport_field_coverage.tsv` (one row per
MIN-REPORT item) and the iter-9 audit produced
`minreport_extended_coverage.tsv` (one row per item split by sub-corpus).
The numbers have lived in TSV/JSON for 19 iterations; this script
turns them into a publication-quality bar chart and migrates a per-item
TSV into Figure 1 of P5 paper (Exhibit A: per-MIN-REPORT-item validation
rate at the live 98-cell corpus).

Why drive item 05 to validated now:
- Long-standing proposed item from iter 1 (T5 presentation).
- All evidence is already on disk; this iter is the figure render step.
- Adds a new P5 paper figure that the reviewer can read at a glance:
  "How rigorously does this benchmark's 98 cells validate each MIN-REPORT item?"

Inputs
------
platform_hybrid/experiments/results/p5p8/minreport_field_coverage.tsv       (per-item, n=98)
platform_hybrid/experiments/results/p5p8/minreport_extended_coverage.tsv    (per-item × sub)
platform_hybrid/experiments/results/p5p8/minreport_extended_n10.tsv         (n=6 n10 records)

Outputs
-------
platform_hybrid/experiments/results/p5p8/figures/p5_minreport_per_item.png
platform_hybrid/experiments/results/p5p8/figures/p5_minreport_per_item.pdf
platform_hybrid/experiments/results/p5p8/p5_exhibit_a_data.tsv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results" / "p5p8"
FIGS = RES / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Use stdlib csv to avoid pandas dependency
import csv


def read_tsv(path):
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    return rows


def main():
    field = read_tsv(RES / "minreport_field_coverage.tsv")
    extended = read_tsv(RES / "minreport_extended_coverage.tsv")

    # ---- per-item validation rate for the 98-cell corpus ----------------
    per_item = []
    for r in field:
        try:
            total = int(r["cells_total"])
            validated = int(r["validated"])
            missing = int(r["missing"])
            na = int(r["na_or_empty"])
        except (KeyError, ValueError):
            continue
        per_item.append({
            "item_no": int(r["item_no"]),
            "item_name": r["item_name"],
            "total": total,
            "validated": validated,
            "missing": missing,
            "na_or_empty": na,
            "pct_validated": (validated / total * 100.0) if total else 0.0,
            "pct_missing": (missing / total * 100.0) if total else 0.0,
        })

    # sort by item_no for canonical ordering
    per_item.sort(key=lambda r: r["item_no"])

    # write the data behind the figure as a fresh TSV
    with (RES / "p5_exhibit_a_data.tsv").open("w") as f:
        f.write("item_no\titem_name\ttotal\tvalidated\tmissing\tna_or_empty\t"
                "pct_validated\tpct_missing\n")
        for r in per_item:
            f.write(f"{r['item_no']}\t{r['item_name']}\t{r['total']}\t"
                    f"{r['validated']}\t{r['missing']}\t{r['na_or_empty']}\t"
                    f"{r['pct_validated']:.2f}\t{r['pct_missing']:.2f}\n")

    # ---- figure: per-item validated vs missing vs n/a -------------------
    items = [f"{r['item_no']}: {r['item_name']}" for r in per_item]
    validated = [r["pct_validated"] for r in per_item]
    missing = [r["pct_missing"] for r in per_item]
    # na_pct = (na / total * 100)
    na_pct = [(r["na_or_empty"] / r["total"] * 100.0) if r["total"] else 0
              for r in per_item]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(items))
    w = 0.27
    ax.bar(x - w, validated, w, color="#2ca02c", label="Validated",
           edgecolor="#1f4f1f", linewidth=0.4)
    ax.bar(x, missing, w, color="#d62728", label="Missing",
           edgecolor="#7a1f1f", linewidth=0.4)
    ax.bar(x + w, na_pct, w, color="#7f7f7f", label="n/a declared",
           edgecolor="#3a3a3a", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(items, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Share of 98 cells (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "P5 Exhibit A: Per-MIN-REPORT-item validation rate "
        "(n=98 live mega cells, 2026-07-04)"
    )
    ax.legend(loc="upper right", ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", color="#cccccc", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    png = FIGS / "p5_minreport_per_item.png"
    pdf = FIGS / "p5_minreport_per_item.pdf"
    plt.savefig(png, dpi=200)
    plt.savefig(pdf)
    plt.close(fig)

    # headline numbers
    summary = {
        "n_items": len(per_item),
        "n_cells": per_item[0]["total"] if per_item else 0,
        "fully_validated_items": [r["item_no"] for r in per_item
                                  if r["pct_validated"] >= 99.0],
        "fully_missing_items": [r["item_no"] for r in per_item
                               if r["pct_missing"] >= 99.0],
        "any_validation_items": [r["item_no"] for r in per_item
                                if 0 < r["pct_validated"] < 99.0],
        "items": per_item,
    }
    (RES / "p5_exhibit_a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    print("wrote:", png.name, "and", pdf.name)
    print("wrote: p5_exhibit_a_data.tsv")
    print("wrote: p5_exhibit_a_summary.json")
    print("fully validated items:", summary["fully_validated_items"])
    print("fully missing items:  ", summary["fully_missing_items"])
    print("partial items:        ", summary["any_validation_items"])


if __name__ == "__main__":
    main()
