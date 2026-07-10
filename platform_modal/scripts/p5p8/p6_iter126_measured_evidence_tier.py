#!/usr/bin/env python3
"""
Iter-126 P6 audit: per-delta measured-evidence tier classification.

For each delta_*.json registry entry:
  - n_measured_rows
  - n_significant_rows
  - pct_significant
  - median_ci_width   (mean of ci_high - ci_low)
  - n_zero_ci         (CI overlaps 0)
  - direction_spread  (max - min of delta)
  - n_panels          (unique measured.panel values)
  - n_metrics         (unique measured.metric values)
  - median_n_per_row
  - evidence_tier     A/B/C/D (A=strongest)

Outputs:
  experiments/results/p5p8/p6_iter126_measured_evidence_tier.tsv   (15 rows: per delta)
  experiments/results/p5p8/p6_iter126_measured_evidence_tier.json  (summary + ranking)

No network; stdlib only; deterministic.
"""
import json, os, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = ROOT / "registry" / "entries"
OUT_TSV = ROOT / "experiments" / "results" / "p5p8" / "p6_iter126_measured_evidence_tier.tsv"
OUT_JSON = ROOT / "experiments" / "results" / "p5p8" / "p6_iter126_measured_evidence_tier.json"


def ci_overlaps_zero(row):
    try:
        lo, hi = row.get("ci_low"), row.get("ci_high")
        if lo is None or hi is None:
            return None
        return lo <= 0 <= hi
    except Exception:
        return None


def ci_width(row):
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if lo is None or hi is None:
        return None
    return hi - lo


def tier(n_sig, n_total, n_panels):
    if n_total == 0:
        return "D"
    if n_sig >= 3 and n_panels >= 2:
        return "A"
    if n_sig >= 1:
        return "B"
    return "C"


def audit():
    rows = []
    files = sorted(p for p in ENTRIES.glob("delta_*.json"))
    for fp in files:
        d = json.load(open(fp))
        measured = d.get("measured") or []
        if not isinstance(measured, list):
            measured = []
        n_total = len(measured)
        sig_rows = [r for r in measured if r.get("significant") is True]
        n_sig = len(sig_rows)
        pct_sig = round(100.0 * n_sig / max(1, n_total), 1)
        widths = [ci_width(r) for r in measured if ci_width(r) is not None]
        med_w = round(statistics.median(widths), 4) if widths else None
        n_zero = sum(1 for r in measured if ci_overlaps_zero(r) is True)
        deltas = [r.get("delta") for r in measured if isinstance(r.get("delta"), (int, float))]
        spread = round(max(deltas) - min(deltas), 4) if deltas else None
        panels = sorted({r.get("panel") for r in measured if r.get("panel")})
        metrics = sorted({r.get("metric") for r in measured if r.get("metric")})
        ns = [r.get("n") for r in measured if isinstance(r.get("n"), (int, float))]
        med_n = round(statistics.median(ns), 1) if ns else None
        t = tier(n_sig, n_total, len(panels))
        rows.append({
            "delta_id": d.get("id", fp.stem),
            "name": d.get("name", ""),
            "base": d.get("base", ""),
            "n_measured_rows": n_total,
            "n_significant_rows": n_sig,
            "pct_significant": pct_sig,
            "median_ci_width": med_w,
            "n_zero_ci": n_zero,
            "direction_spread": spread,
            "n_panels": len(panels),
            "panels": ";".join(panels),
            "n_metrics": len(metrics),
            "metrics": ";".join(metrics),
            "median_n_per_row": med_n,
            "evidence_tier": t,
            "citation_arxiv": (d.get("citation") or {}).get("arxiv", ""),
        })

    # Sort: tier A first, then n_sig desc, then n_total desc
    tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows.sort(key=lambda r: (tier_order[r["evidence_tier"]], -r["n_significant_rows"], -r["n_measured_rows"]))

    # Write TSV
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "delta_id", "name", "base", "n_measured_rows", "n_significant_rows",
        "pct_significant", "median_ci_width", "n_zero_ci", "direction_spread",
        "n_panels", "panels", "n_metrics", "metrics", "median_n_per_row",
        "evidence_tier", "citation_arxiv",
    ]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    # Summary
    counts_by_tier = {}
    for r in rows:
        counts_by_tier[r["evidence_tier"]] = counts_by_tier.get(r["evidence_tier"], 0) + 1
    total_measured_rows = sum(r["n_measured_rows"] for r in rows)
    total_sig = sum(r["n_significant_rows"] for r in rows)
    d_tier = sum(1 for r in rows if r["evidence_tier"] == "D")
    summary = {
        "n_deltas": len(rows),
        "tier_counts": counts_by_tier,
        "n_total_measured_rows": total_measured_rows,
        "n_total_significant_rows": total_sig,
        "pct_d_overall": round(100 * d_tier / max(1, len(rows)), 1),
        "d_tier_ids": [r["delta_id"] for r in rows if r["evidence_tier"] == "D"],
        "a_tier_ids": [r["delta_id"] for r in rows if r["evidence_tier"] == "A"],
        "ranking_top10": [
            {"delta_id": r["delta_id"], "tier": r["evidence_tier"], "n_sig": r["n_significant_rows"], "n_total": r["n_measured_rows"]}
            for r in rows[:10]
        ],
        "ranking_bottom5": [
            {"delta_id": r["delta_id"], "tier": r["evidence_tier"], "n_sig": r["n_significant_rows"], "n_total": r["n_measured_rows"]}
            for r in rows[-5:]
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print(f"=== iter-126 P6 measured-evidence tier audit ===")
    print(f"deltas audited: {len(rows)}")
    print(f"total measured rows: {total_measured_rows}")
    print(f"total significant rows: {total_sig}")
    print(f"tier distribution: {counts_by_tier}")
    print()
    print("rank | tier | delta_id                | n_sig/n_total | n_panels")
    print("-----+------+-------------------------+---------------+---------")
    for i, r in enumerate(rows, 1):
        print(f"{i:4d} | {r['evidence_tier']:4s} | {r['delta_id']:23s} | {r['n_significant_rows']:2d}/{r['n_measured_rows']:2d} ({r['pct_significant']:.0f}%) | {r['n_panels']}")


if __name__ == "__main__":
    audit()
