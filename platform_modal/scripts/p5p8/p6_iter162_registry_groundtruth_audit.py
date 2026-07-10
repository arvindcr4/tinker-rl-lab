#!/usr/bin/env python3
"""
Iter-162 — P6 registry ground-truth audit (citation resolution, source-path
existence, zvf130 value integrity, per-entry integrity score).

The P6 GRPO-Registry encodes three classes of external references that, if
silently stale, undermine the catalog's machine-readability claim:
  (1) citation.bibkey — a string that should resolve to a real entry in
      paper/references.bib. The catalog asserts "verified" in free-text notes
      but does not machine-check this.
  (2) measured[].source — a relative path that should exist on disk. A
      measured row whose source file has been moved or deleted is silently
      orphaned; the audit's "source" field reads as authoritative but the
      number behind it is not recoverable.
  (3) zvf130_*.json outcomes.zvf_risk_mean / zvf_risk_sd — values that should
      match the ground-truth zvf_iter130_method_risk.tsv within tolerance.

Iter-162 audits the JOIN between each entry's claims and the worktree's
ground-truth at these three layers, surfaces the gap count per layer and per
entry, and computes a per-entry registry_integrity_score (fraction of the
applicable ground-truth checks that pass).

Inputs : registry/entries/*.json
         paper/references.bib
         experiments/results/zvf_iter130_method_risk.tsv
Outputs: experiments/results/p5p8/p6_iter162_per_entry.tsv
         experiments/results/p5p8/p6_iter162_per_layer_summary.tsv
         experiments/results/p5p8/p6_iter162_per_cell.tsv
         experiments/results/p5p8/p6_iter162_summary.json
"""
import csv
import json
import os
import re
from collections import defaultdict

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENT_DIR = os.path.join(WORKTREE, "registry", "entries")
BIB_PATH = os.path.join(WORKTREE, "paper", "references.bib")
GTT_PATH = os.path.join(WORKTREE, "experiments", "results",
                        "zvf_iter130_method_risk.tsv")
OUT_DIR = os.path.join(WORKTREE, "experiments", "results", "p5p8")
os.makedirs(OUT_DIR, exist_ok=True)

# Tolerance for floating-point equality on zvf_risk_mean values carried in
# stack entries: 1e-4 absolute is tighter than the 5-seed bootstrap CI half-
# width (typically ≥1e-3) so we will not flag pure sampling-noise differences.
ZVF_TOL = 1e-4


def load_bibkeys():
    """Parse every @TYPE{key, ...} line from references.bib and return the
    set of declared bibkeys. The optional `[DEDUP] skipped duplicate` comment
    is filtered out (it is metadata, not a fresh key)."""
    keys = set()
    pat = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,")
    with open(BIB_PATH) as f:
        for line in f:
            if line.startswith("%"):
                continue
            m = pat.match(line.strip())
            if m:
                keys.add(m.group(1))
    return keys


def load_groundtruth():
    """Parse zvf_iter130_method_risk.tsv; return {method -> {col: value}}."""
    rows = {}
    with open(GTT_PATH) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows[r["method"]] = r
    return rows


def load_entries():
    out = []
    for f in sorted(os.listdir(ENT_DIR)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(ENT_DIR, f)
        d = json.load(open(path))
        d["_fname"] = f
        d["_path"] = path
        out.append(d)
    return out


def main():
    bibkeys = load_bibkeys()
    groundtruth = load_groundtruth()
    entries = load_entries()

    per_entry_rows = []
    per_layer_counts = defaultdict(int)
    per_cell_rows = []
    failed_cites = []
    missing_paths = []
    value_diffs = []

    n_delta = 0
    n_stack = 0
    n_total = len(entries)

    for d in entries:
        eid = d.get("id", d.get("_fname", "?"))
        rtype = d.get("record_type", "?")
        if rtype == "variant_delta":
            n_delta += 1
        elif rtype == "stack":
            n_stack += 1

        # Layer 1: citation resolution
        cite_ok = None
        cite_key = None
        if rtype == "variant_delta":
            cite = d.get("citation") or {}
            cite_key = cite.get("bibkey")
            if cite_key is None:
                cite_ok = None  # not applicable for this record
            else:
                cite_ok = cite_key in bibkeys
                per_layer_counts["citation_total"] += 1
                if cite_ok:
                    per_layer_counts["citation_pass"] += 1
                else:
                    per_layer_counts["citation_fail"] += 1
                    failed_cites.append({
                        "entry_id": eid, "bibkey": cite_key,
                        "fname": d["_fname"],
                    })
            per_cell_rows.append({
                "entry_id": eid,
                "layer": "citation",
                "field": "citation.bibkey",
                "value": cite_key or "",
                "ok": "PASS" if cite_ok else ("FAIL" if cite_ok is False else "NA"),
                "note": "" if cite_ok else "bibkey not in paper/references.bib",
            })

        # Layer 2: source-path existence on every measured[] row
        measured = d.get("measured") or []
        n_measured = len(measured)
        n_measured_ok = 0
        for i, m in enumerate(measured):
            src = m.get("source")
            ok = None
            note = ""
            if src is None:
                ok = False
                note = "source field is null"
            else:
                full = os.path.join(WORKTREE, src) if not os.path.isabs(src) else src
                if os.path.exists(full):
                    ok = True
                else:
                    ok = False
                    note = f"path not found: {src}"
            per_layer_counts["source_total"] += 1
            if ok:
                per_layer_counts["source_pass"] += 1
                n_measured_ok += 1
            else:
                per_layer_counts["source_fail"] += 1
                missing_paths.append({
                    "entry_id": eid, "source": src,
                    "fname": d["_fname"],
                })
            per_cell_rows.append({
                "entry_id": eid,
                "layer": "source_path",
                "field": f"measured[{i}].source",
                "value": src or "",
                "ok": "PASS" if ok else "FAIL",
                "note": note,
            })

        # Layer 3: zvf130 value integrity (stack entries only)
        zvf_match = None
        zvf_diff = None
        if rtype == "stack" and eid.startswith("zvf130_"):
            method = eid[len("zvf130_"):]
            outcomes = d.get("outcomes") or {}
            stored = outcomes.get("zvf_risk_mean")
            stored_sd = outcomes.get("zvf_risk_sd")
            gt = groundtruth.get(method)
            if gt is None:
                zvf_match = None
                note = f"method '{method}' not in ground-truth TSV"
            elif stored is None:
                zvf_match = None
                note = "outcomes.zvf_risk_mean is null"
            else:
                gt_val = float(gt["zvf_risk_mean"])
                zvf_diff = abs(float(stored) - gt_val)
                zvf_match = zvf_diff <= ZVF_TOL
                note = f"|stored - gt|={zvf_diff:.6g}"
                # Also check sd if present
                if stored_sd is not None and gt.get("zvf_risk_sd"):
                    gt_sd = float(gt["zvf_risk_sd"])
                    sd_diff = abs(float(stored_sd) - gt_sd)
                    sd_match = sd_diff <= ZVF_TOL
                    note += f", |sd_diff|={sd_diff:.6g}"
                else:
                    sd_match = None
            per_layer_counts["zvf_value_total"] += 1
            if zvf_match is True:
                per_layer_counts["zvf_value_pass"] += 1
            elif zvf_match is False:
                per_layer_counts["zvf_value_fail"] += 1
                value_diffs.append({
                    "entry_id": eid,
                    "stored": stored,
                    "groundtruth": gt.get("zvf_risk_mean") if gt else None,
                    "diff": zvf_diff,
                    "fname": d["_fname"],
                })
            per_cell_rows.append({
                "entry_id": eid,
                "layer": "zvf_value",
                "field": "outcomes.zvf_risk_mean",
                "value": f"{stored:.6g}" if stored is not None else "null",
                "ok": ("PASS" if zvf_match else "FAIL") if zvf_match is not None else "NA",
                "note": note,
            })
            # Track sd separately
            if stored_sd is not None and gt and gt.get("zvf_risk_sd"):
                per_layer_counts["zvf_sd_total"] += 1
                gt_sd = float(gt["zvf_risk_sd"])
                sd_diff = abs(float(stored_sd) - gt_sd)
                sd_match = sd_diff <= ZVF_TOL
                if sd_match:
                    per_layer_counts["zvf_sd_pass"] += 1
                else:
                    per_layer_counts["zvf_sd_fail"] += 1
                    value_diffs.append({
                        "entry_id": eid,
                        "stored": stored_sd,
                        "groundtruth": gt.get("zvf_risk_sd"),
                        "diff": sd_diff,
                        "fname": d["_fname"],
                    })
                per_cell_rows.append({
                    "entry_id": eid,
                    "layer": "zvf_value_sd",
                    "field": "outcomes.zvf_risk_sd",
                    "value": f"{stored_sd:.6g}",
                    "ok": "PASS" if sd_match else "FAIL",
                    "note": f"|sd - gt_sd|={sd_diff:.6g}",
                })

        # Per-entry integrity score: applicable checks
        applicable = 0
        passed = 0
        if cite_ok is not None:
            applicable += 1
            if cite_ok:
                passed += 1
        for _ in range(n_measured):
            applicable += 1
        for i, m in enumerate(measured):
            src = m.get("source")
            if src is not None:
                full = os.path.join(WORKTREE, src) if not os.path.isabs(src) else src
                if os.path.exists(full):
                    passed += 1
        if zvf_match is not None:
            applicable += 1
            if zvf_match:
                passed += 1
        integrity_score = passed / applicable if applicable else None

        per_entry_rows.append({
            "entry_id": eid,
            "record_type": rtype,
            "fname": d["_fname"],
            "n_citation_check": int(cite_ok is not None),
            "citation_ok": "PASS" if cite_ok else ("FAIL" if cite_ok is False else "NA"),
            "bibkey": cite_key or "",
            "n_measured": n_measured,
            "n_measured_ok": n_measured_ok,
            "zvf_match": ("PASS" if zvf_match else "FAIL") if zvf_match is not None else "NA",
            "zvf_diff": "" if zvf_diff is None else f"{zvf_diff:.6g}",
            "applicable_checks": applicable,
            "passed_checks": passed,
            "integrity_score": "" if integrity_score is None else f"{integrity_score:.4f}",
        })

    # Sort per-entry by integrity score ascending so biggest gaps appear first
    sorted_rows = sorted(
        per_entry_rows,
        key=lambda r: (r["integrity_score"] == "", r["integrity_score"] or "0.0"),
    )

    # Write per-entry TSV
    pe_path = os.path.join(OUT_DIR, "p6_iter162_per_entry.tsv")
    with open(pe_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_entry_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in sorted_rows:
            w.writerow(r)

    # Per-layer summary
    layer_rows = []
    for layer, total_key, pass_key, fail_key in [
        ("citation", "citation_total", "citation_pass", "citation_fail"),
        ("source_path", "source_total", "source_pass", "source_fail"),
        ("zvf_value", "zvf_value_total", "zvf_value_pass", "zvf_value_fail"),
        ("zvf_value_sd", "zvf_sd_total", "zvf_sd_pass", "zvf_sd_fail"),
    ]:
        total = per_layer_counts.get(total_key, 0)
        passed = per_layer_counts.get(pass_key, 0)
        failed = per_layer_counts.get(fail_key, 0)
        layer_rows.append({
            "layer": layer,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": "" if total == 0 else f"{passed/total:.4f}",
        })

    pl_path = os.path.join(OUT_DIR, "p6_iter162_per_layer_summary.tsv")
    with open(pl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in layer_rows:
            w.writerow(r)

    # Per-cell TSV (sorted by entry then layer)
    cell_path = os.path.join(OUT_DIR, "p6_iter162_per_cell.tsv")
    with open(cell_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cell_rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in sorted(per_cell_rows, key=lambda r: (r["entry_id"], r["layer"])):
            w.writerow(r)

    # Headline numbers
    n_total_checks = sum(r["applicable_checks"] for r in per_entry_rows)
    n_total_passed = sum(r["passed_checks"] for r in per_entry_rows)
    headline = {
        "n_entries": n_total,
        "n_delta_entries": n_delta,
        "n_stack_entries": n_stack,
        "n_citation_checks": per_layer_counts.get("citation_total", 0),
        "n_citation_fail": per_layer_counts.get("citation_fail", 0),
        "n_source_checks": per_layer_counts.get("source_total", 0),
        "n_source_fail": per_layer_counts.get("source_fail", 0),
        "n_zvf_value_checks": per_layer_counts.get("zvf_value_total", 0),
        "n_zvf_value_fail": per_layer_counts.get("zvf_value_fail", 0),
        "n_zvf_sd_checks": per_layer_counts.get("zvf_sd_total", 0),
        "n_zvf_sd_fail": per_layer_counts.get("zvf_sd_fail", 0),
        "pct_citation_pass": (
            100.0 * per_layer_counts.get("citation_pass", 0) /
            max(per_layer_counts.get("citation_total", 0), 1)
        ),
        "pct_source_pass": (
            100.0 * per_layer_counts.get("source_pass", 0) /
            max(per_layer_counts.get("source_total", 0), 1)
        ),
        "pct_zvf_value_pass": (
            100.0 * per_layer_counts.get("zvf_value_pass", 0) /
            max(per_layer_counts.get("zvf_value_total", 0), 1)
        ),
        "pct_zvf_sd_pass": (
            100.0 * per_layer_counts.get("zvf_sd_pass", 0) /
            max(per_layer_counts.get("zvf_sd_total", 0), 1)
        ),
        "overall_integrity": (
            100.0 * n_total_passed / max(n_total_checks, 1)
        ),
        "n_failed_citations": len(failed_cites),
        "n_missing_source_paths": len(missing_paths),
        "n_zvf_value_diffs": len(value_diffs),
    }

    summary = {
        "n_bibkeys_in_references_bib": len(bibkeys),
        "n_groundtruth_methods_in_tsv": len(groundtruth),
        "n_entries": n_total,
        "n_delta_entries": n_delta,
        "n_stack_entries": n_stack,
        "per_layer_counts": dict(per_layer_counts),
        "headline": headline,
        "failed_citations": failed_cites,
        "missing_source_paths": missing_paths,
        "zvf_value_diffs": value_diffs,
    }
    sum_path = os.path.join(OUT_DIR, "p6_iter162_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Stdout
    print(f"n_bibkeys_in_references_bib={len(bibkeys)} "
          f"n_groundtruth_methods_in_tsv={len(groundtruth)}")
    print(f"n_entries={n_total} (delta={n_delta}, stack={n_stack})")
    print("Per-layer pass rate:")
    for r in layer_rows:
        print(f"  {r['layer']:20s} total={r['total']:3d} "
              f"passed={r['passed']:3d} failed={r['failed']:3d} "
              f"pass_rate={r['pass_rate']}")
    print(f"Overall integrity: {n_total_passed}/{n_total_checks} "
          f"= {headline['overall_integrity']:.2f}%")
    print(f"Failed citations ({len(failed_cites)}):")
    for f in failed_cites:
        print(f"  {f['fname']:50s} bibkey={f['bibkey']}")
    print(f"Missing source paths ({len(missing_paths)}):")
    for m in missing_paths[:10]:
        print(f"  {m['fname']:50s} source={m['source']}")
    if len(missing_paths) > 10:
        print(f"  ... ({len(missing_paths) - 10} more)")
    print(f"ZVF value diffs ({len(value_diffs)}):")
    for v in value_diffs:
        print(f"  {v['fname']:50s} diff={v['diff']:.6g} "
              f"(stored={v['stored']}, gt={v['groundtruth']})")
    print(f"\nFiles:\n  {pe_path}\n  {pl_path}\n  {cell_path}\n  {sum_path}")


if __name__ == "__main__":
    main()