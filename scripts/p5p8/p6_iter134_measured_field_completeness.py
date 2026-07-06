#!/usr/bin/env python3
"""P6 iter-134 — measured-row field-completeness audit.

Fresh vein (a)+(b) on the brief: per-row field-completeness audit of the
measured[] array on every variant_delta entry. Iter-130 covered schema-level
parse_ok / schema_ok / stale_method_rows. Iter-134 goes one level deeper:
for each of 15 variant_delta entries x 11+ measured fields, classify each
field as PRESENT / NULL / MISSING; aggregate per-field and per-entry;
verify provenance (source file exists on disk) and CI consistency
(ci_low <= delta <= ci_high).

Output:
  experiments/results/p5p8/p6_iter134_per_row.tsv         (one row per measured row)
  experiments/results/p5p8/p6_iter134_per_entry.tsv      (one row per entry)
  experiments/results/p5p8/p6_iter134_per_field.tsv      (one row per field)
  experiments/results/p5p8/p6_iter134_summary.json

Stdlib only.
"""
import csv
import json
import pathlib
import sys
from collections import defaultdict, Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
REG = ROOT / "registry/entries"
OUT = ROOT / "experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FIELDS = [
    "metric", "panel", "base", "delta", "ci_low", "ci_high",
    "n", "significant", "ci_method", "source", "note",
]
EXTRA_FIELDS = [
    "window_sensitivity", "robust_panel",
    "welch_ci_low", "welch_ci_high", "welch_sig",
    "sig_robust_bootstrap_and_welch", "sig_robust_note",
    "xpanel_verdict", "xpanel_method",
]
ALL_FIELDS = BASE_FIELDS + EXTRA_FIELDS


def classify(field, value):
    """Classify a field's presence on a measured row."""
    if value is None:
        return "NULL"
    if isinstance(value, str) and value.strip() == "":
        return "NULL"
    if isinstance(value, (list, dict)) and len(value) == 0:
        return "NULL"
    return "PRESENT"


def audit_entry(entry_path):
    """Yield per-row audit records for one entry."""
    rec = json.loads(entry_path.read_text())
    eid = rec.get("id", entry_path.stem)
    rt = rec.get("record_type")
    measured = rec.get("measured", []) or []
    rows = []
    for i, m in enumerate(measured):
        row = {"id": eid, "record_type": rt, "row_idx": i}
        # Per-field classification
        for f in ALL_FIELDS:
            row[f + "_state"] = classify(f, m.get(f))
        # CI consistency
        try:
            d = float(m.get("delta"))
            cl = float(m.get("ci_low"))
            ch = float(m.get("ci_high"))
            row["ci_consistent"] = "yes" if (cl <= d <= ch) else "no"
        except (TypeError, ValueError):
            row["ci_consistent"] = "n/a"
        # Provenance (source file exists on disk?)
        src = m.get("source")
        if src is None or (isinstance(src, str) and src.strip() == ""):
            row["src_exists"] = "n/a"
            row["src_state"] = "NULL"
        else:
            # source is repo-relative
            sp = ROOT / src
            row["src_exists"] = "yes" if sp.exists() else "no"
            row["src_state"] = "PRESENT"
        rows.append(row)
    return rows


def main():
    per_row = []
    per_entry = []
    for p in sorted(REG.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("record_type") != "variant_delta":
            continue
        rs = audit_entry(p)
        per_row.extend(rs)
        # Per-entry aggregation
        if not rs:
            per_entry.append({
                "id": rec.get("id"),
                "record_type": "variant_delta",
                "n_measured_rows": 0,
                **{f + "_present": 0 for f in ALL_FIELDS},
                **{f + "_null": 0 for f in ALL_FIELDS},
                "n_ci_consistent": 0,
                "n_ci_inconsistent": 0,
                "n_src_present": 0,
                "n_src_missing": 0,
            })
            continue
        n = len(rs)
        agg = {"id": rec.get("id"), "record_type": "variant_delta",
               "n_measured_rows": n}
        for f in ALL_FIELDS:
            pres = sum(1 for r in rs if r[f + "_state"] == "PRESENT")
            agg[f + "_present"] = pres
            agg[f + "_null"] = n - pres
        agg["n_ci_consistent"] = sum(1 for r in rs if r["ci_consistent"] == "yes")
        agg["n_ci_inconsistent"] = sum(1 for r in rs if r["ci_consistent"] == "no")
        agg["n_src_present"] = sum(1 for r in rs if r["src_exists"] == "yes")
        agg["n_src_missing"] = sum(1 for r in rs if r["src_exists"] == "no")
        per_entry.append(agg)

    # Per-field aggregation
    per_field = []
    for f in ALL_FIELDS:
        states = Counter(r[f + "_state"] for r in per_row)
        per_field.append({
            "field": f,
            "n_rows": len(per_row),
            "n_present": states.get("PRESENT", 0),
            "n_null": states.get("NULL", 0),
            "present_frac": round(states.get("PRESENT", 0) / max(len(per_row), 1), 4),
        })
    # CI consistency & provenance roll-ups
    n_rows = len(per_row)
    n_ci_cons = sum(1 for r in per_row if r["ci_consistent"] == "yes")
    n_ci_incon = sum(1 for r in per_row if r["ci_consistent"] == "no")
    n_src_yes = sum(1 for r in per_row if r["src_exists"] == "yes")
    n_src_no = sum(1 for r in per_row if r["src_exists"] == "no")

    # Significant-fraction roll-up
    n_sig = sum(1 for r in per_row
                if (per_row[0] and r.get("id") and
                    any(rr["id"] == r["id"] and rr["row_idx"] == r["row_idx"]
                        for rr in per_row) and
                    r.get("significant_state") == "PRESENT"))
    # Recompute sig cleanly
    n_sig = 0
    for p in sorted(REG.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("record_type") != "variant_delta":
            continue
        for m in (rec.get("measured") or []):
            if m.get("significant") is True:
                n_sig += 1
            elif m.get("significant") == False:
                pass

    summary = {
        "n_entries": len(per_entry),
        "n_measured_rows": n_rows,
        "n_ci_consistent": n_ci_cons,
        "n_ci_inconsistent": n_ci_incon,
        "n_src_present": n_src_yes,
        "n_src_missing": n_src_no,
        "n_significant_true": n_sig,
        "per_field_coverage": per_field,
    }

    # Write TSVs
    if per_row:
        with (OUT / "p6_iter134_per_row.tsv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(per_row)
    if per_entry:
        with (OUT / "p6_iter134_per_entry.tsv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_entry[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(per_entry)
    with (OUT / "p6_iter134_per_field.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_field[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_field)
    (OUT / "p6_iter134_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"WROTE per_row={len(per_row)} per_entry={len(per_entry)} per_field={len(per_field)}")
    print(f"CI consistent: {n_ci_cons}/{n_rows}, src present: {n_src_yes}/{n_rows}, sig=true: {n_sig}")
    # Per-field quick view
    for r in per_field:
        if r["field"] in ("metric", "panel", "base", "delta", "ci_low", "ci_high",
                          "n", "significant", "ci_method", "source", "note"):
            print(f"  {r['field']:>20s} present={r['n_present']:>2d}/{r['n_rows']:<2d} ({r['present_frac']*100:.1f}%)")


if __name__ == "__main__":
    main()