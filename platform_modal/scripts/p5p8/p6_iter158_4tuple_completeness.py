#!/usr/bin/env python3
"""
Iter-158 — P6 registry 4-tuple completeness audit (claims vs expected_effects
vs measured vs claim_validation) at the (entry x metric x panel) granularity.

For every delta_*.json entry, the registry exposes FOUR per-(metric, panel)
provenance tuples:
  * deltas[]           : prose-side mechanism description (component, field, change)
  * expected_effects[] : forward reference, predicted_sign the prose implies for
                         a (metric, panel) pair  (iter-46 extension)
  * measured[]         : empirical value, paired bootstrap CI  (iter-34/iter-122)
  * claim_validation[] : machine-generated verdict matching each (measured,
                         expected_effects) pair to SUPPORTS / NEUTRAL /
                         CONTRADICTS / UNCLAIMED  (iter-46)

Iter-158 audits the JOIN COVERAGE among these four sets at the (entry x metric
x panel) granularity, surfaces the four coverage gaps, and computes a per-entry
``registry_completeness`` score.

Inputs : registry/entries/delta_*.json
Outputs: platform_hybrid/experiments/results/p5p8/p6_iter158_per_cell.tsv
         platform_hybrid/experiments/results/p5p8/p6_iter158_per_entry.tsv
         platform_hybrid/experiments/results/p5p8/p6_iter158_coverage_gaps.tsv
         platform_hybrid/experiments/results/p5p8/p6_iter158_summary.json
"""
import csv
import json
import os
from collections import defaultdict

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENT_DIR  = os.path.join(WORKTREE, "registry", "entries")
OUT_DIR  = os.path.join(WORKTREE, "experiments", "results", "p5p8")
os.makedirs(OUT_DIR, exist_ok=True)


def load_entries():
    out = []
    for f in sorted(os.listdir(ENT_DIR)):
        if not f.startswith("delta_") or not f.endswith(".json"):
            continue
        path = os.path.join(ENT_DIR, f)
        d = json.load(open(path))
        d["_fname"] = f
        out.append(d)
    return out


def key(metric, panel):
    return f"{metric}|{panel}"


def main():
    entries = load_entries()
    # Per (entry, metric, panel) cell across the 4 tuples
    cells = []  # one row per cell with membership flags
    per_entry_stats = {}  # entry_id -> counts
    coverage_gaps = []  # one row per gap

    for d in entries:
        eid = d.get("id", d.get("_fname", "?"))
        deltas = d.get("deltas", []) or []
        exp_eff = d.get("expected_effects", []) or []
        measured = d.get("measured", []) or []
        claim_val = d.get("claim_validation", []) or []

        # Index by (metric, panel)
        exp_idx = {(e["metric"], e["panel"]): e for e in exp_eff}
        meas_idx = {(e["metric"], e["panel"]): e for e in measured}
        cv_idx = {(e["metric"], e["panel"]): e for e in claim_val}

        all_keys = set(exp_idx) | set(meas_idx) | set(cv_idx)

        # Per-entry stats
        n_deltas = len(deltas)
        n_exp = len(exp_eff)
        n_meas = len(measured)
        n_cv = len(claim_val)
        n_keys = len(all_keys)
        # joined keys: present in all three (exp, meas, cv)
        joined = set(exp_idx) & set(meas_idx) & set(cv_idx)
        n_joined = len(joined)
        # completeness = fraction of (metric,panel) tau that has all three
        completeness = n_joined / n_keys if n_keys else None

        per_entry_stats[eid] = {
            "n_deltas": n_deltas,
            "n_expected_effects": n_exp,
            "n_measured": n_meas,
            "n_claim_validation": n_cv,
            "n_distinct_keys": n_keys,
            "n_joined": n_joined,
            "completeness": completeness,
        }

        # Emit per-cell rows for all distinct (metric, panel) pairs
        for k in sorted(all_keys):
            m, p = k
            in_exp = k in exp_idx
            in_meas = k in meas_idx
            in_cv = k in cv_idx
            # join classification
            if in_exp and in_meas and in_cv:
                classification = "FULL"
            elif in_exp and in_meas and not in_cv:
                classification = "EXP_MEAS_NO_CV"
            elif in_exp and in_cv and not in_meas:
                classification = "EXP_CV_NO_MEAS"
            elif in_meas and in_cv and not in_exp:
                classification = "MEAS_CV_NO_EXP"
            elif in_exp and not in_meas and not in_cv:
                classification = "EXP_ONLY"
            elif in_meas and not in_exp and not in_cv:
                classification = "MEAS_ONLY"
            elif in_cv and not in_exp and not in_meas:
                classification = "CV_ONLY"
            else:
                classification = "?"
            # verdict of the claim_validation row (when present + machine-generated)
            verdict = cv_idx[k].get("verdict") if in_cv else None
            obs_delta = meas_idx[k].get("delta") if in_meas else None
            cells.append({
                "entry_id": eid,
                "metric": m,
                "panel": p,
                "in_expected_effects": int(in_exp),
                "in_measured": int(in_meas),
                "in_claim_validation": int(in_cv),
                "classification": classification,
                "verdict": verdict if verdict is not None else "",
                "obs_delta": "" if obs_delta is None else f"{obs_delta:.6f}",
            })

        # Coverage gaps: which (metric, panel) have prose but no claim_validation?
        for k in sorted(set(exp_idx) - set(cv_idx)):
            m, p = k
            coverage_gaps.append({
                "entry_id": eid,
                "gap_type": "EXPECTED_WITHOUT_CV",
                "metric": m, "panel": p,
                "predicted_sign": exp_idx[k].get("predicted_sign"),
                "rationale": (exp_idx[k].get("rationale") or "")[:120],
                "severity": "high",  # forward claim without machine validation
            })
        # Which measured lack expected_effects (no forward claim)
        for k in sorted(set(meas_idx) - set(exp_idx)):
            m, p = k
            coverage_gaps.append({
                "entry_id": eid,
                "gap_type": "MEASURED_WITHOUT_EXPECTED",
                "metric": m, "panel": p,
                "predicted_sign": None,
                "rationale": f"measured delta={meas_idx[k].get('delta')}",
                "severity": "medium",  # measurement without prose prediction
            })
        # Which prose-deltas have NO expected_effects at all
        if n_deltas > 0 and n_exp == 0:
            for di, dl in enumerate(deltas):
                coverage_gaps.append({
                    "entry_id": eid,
                    "gap_type": "DELTAS_WITHOUT_EXPECTED",
                    "metric": "?", "panel": "?",
                    "predicted_sign": None,
                    "rationale": (dl.get("component") or "?") + ": " + (dl.get("change") or "")[:80],
                    "severity": "high",
                })
        # Which expected_effects lack measured
        for k in sorted(set(exp_idx) - set(meas_idx)):
            m, p = k
            coverage_gaps.append({
                "entry_id": eid,
                "gap_type": "EXPECTED_WITHOUT_MEASURED",
                "metric": m, "panel": p,
                "predicted_sign": exp_idx[k].get("predicted_sign"),
                "rationale": (exp_idx[k].get("rationale") or "")[:120],
                "severity": "medium",
            })

    # Write per-cell TSV
    cell_path = os.path.join(OUT_DIR, "p6_iter158_per_cell.tsv")
    with open(cell_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()), delimiter="\t")
        w.writeheader()
        for c in cells:
            w.writerow(c)

    # Verdict distribution on FULL cells (the SUPPORTS/NEUTRAL/CONTRADICTS/UNCLAIMED tally)
    full_verdict_counts = defaultdict(int)
    for c in cells:
        if c["classification"] == "FULL":
            v = c.get("verdict") or "?"
            full_verdict_counts[v] += 1

    # Per-entry TSV (sorted by completeness ascending — biggest gaps first)
    entry_path = os.path.join(OUT_DIR, "p6_iter158_per_entry.tsv")
    sorted_eids = sorted(per_entry_stats.keys(),
                         key=lambda e: (per_entry_stats[e]["completeness"]is None,
                                        per_entry_stats[e]["completeness"] or 0))
    with open(entry_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entry_id", "n_deltas", "n_expected_effects", "n_measured",
                    "n_claim_validation", "n_distinct_keys", "n_joined",
                    "registry_completeness"])
        for eid in sorted_eids:
            s = per_entry_stats[eid]
            w.writerow([eid, s["n_deltas"], s["n_expected_effects"],
                        s["n_measured"], s["n_claim_validation"],
                        s["n_distinct_keys"], s["n_joined"],
                        f"{s['completeness']:.4f}" if s["completeness"] is not None else "NA"])

    # Coverage gaps TSV
    gap_path = os.path.join(OUT_DIR, "p6_iter158_coverage_gaps.tsv")
    with open(gap_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_gaps[0].keys()), delimiter="\t")
        w.writeheader()
        for g in coverage_gaps:
            w.writerow(g)

    # Classification histogram
    classification_counts = defaultdict(int)
    for c in cells:
        classification_counts[c["classification"]] += 1

    # Severity histogram
    severity_counts = defaultdict(int)
    for g in coverage_gaps:
        severity_counts[g["severity"]] += 1
    severity_counts_total = sum(severity_counts.values())

    # Per-entry gap counts
    per_entry_gap_counts = defaultdict(int)
    for g in coverage_gaps:
        per_entry_gap_counts[g["entry_id"]] += 1

    summary = {
        "n_entries": len(entries),
        "n_cells": len(cells),
        "n_gaps": len(coverage_gaps),
        "n_joined_cells": sum(1 for c in cells if c["classification"] == "FULL"),
        "classification_counts": dict(classification_counts),
        "severity_counts": dict(severity_counts),
        "full_cell_verdict_counts": dict(full_verdict_counts),
        "per_entry_completeness": {e: per_entry_stats[e] for e in sorted_eids},
        "per_entry_gap_count": dict(per_entry_gap_counts),
        "headline": {
            "n_entries_with_deltas_but_no_expected_effects": sum(
                1 for eid, s in per_entry_stats.items()
                if s["n_deltas"] > 0 and s["n_expected_effects"] == 0
            ),
            "n_entries_with_expected_but_no_measured": sum(
                1 for eid, s in per_entry_stats.items()
                if s["n_expected_effects"] > 0 and s["n_measured"] < s["n_expected_effects"]
            ),
            "n_entries_with_expected_but_no_cv": sum(
                1 for eid, s in per_entry_stats.items()
                if s["n_expected_effects"] > 0 and s["n_claim_validation"] < s["n_expected_effects"]
            ),
            "pct_full_cells": round(100.0 * sum(1 for c in cells if c["classification"] == "FULL") / max(len(cells), 1), 2),
            "pct_high_severity_gaps": round(100.0 * severity_counts.get("high", 0) / max(severity_counts_total, 1), 2),
        }
    }

    sum_path = os.path.join(OUT_DIR, "p6_iter158_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Stdout summary
    print(f"n_entries={len(entries)} n_cells={len(cells)} n_gaps={len(coverage_gaps)}")
    print(f"classification_counts={dict(classification_counts)}")
    print(f"severity_counts={dict(severity_counts)}")
    print(f"headline={summary['headline']}")
    print(f"per_entry_completeness (sorted ascending, NA last):")
    for eid in sorted_eids:
        s = per_entry_stats[eid]
        comp = f"{s['completeness']:.3f}" if s["completeness"] is not None else "  NA"
        print(f"  {eid:50s} comp={comp}  deltas={s['n_deltas']:2d} "
              f"exp={s['n_expected_effects']:2d} meas={s['n_measured']:2d} "
              f"cv={s['n_claim_validation']:2d}  "
              f"gaps={per_entry_gap_counts.get(eid, 0)}")
    print(f"\nFiles: {cell_path}\n        {entry_path}\n        {gap_path}\n        {sum_path}")


if __name__ == "__main__":
    main()
