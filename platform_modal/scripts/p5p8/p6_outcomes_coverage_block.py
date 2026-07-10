#!/usr/bin/env python3
"""
Iter-62 P6 (Pillar 2) — `outcomes.coverage` self-report block on all 31 entries.

Vein: iter-60 row 71 recommendation ("schema patch needed: `outcomes.coverage`
block per #36 pattern"), combined with iter-61 row 72 ("manifest auditor has
98/98 honest-but-vacuous cells — discriminative range needs widening on the
manifest emitter, but the registry surface itself has the same issue: 0/31
entries disclose their MIN-REPORT compliance"). This script:

  (1) Computes, per-entry, four coverage surfaces:
        a) min_report_coverage   — fraction of 7 MIN-REPORT items with at
                                   least one sub-field populated (non-null)
        b) declared_deltas_coverage — fraction of declared
                                   variant_deltas_applied[*] with status
                                   == "implemented" or "surrogate"
                                   (informative status)
        c) measured_coverage     — for delta records: ratio of measured rows
                                   to declared delta components; for stack
                                   records: ratio of measured+ci_method
                                   populated fields to total measured fields
        d) ci_method_present     — bool: outcomes.ci_method is non-null

  (2) Writes `platform_hybrid/experiments/results/p5p8/p6_outcomes_coverage_audit.tsv` with one
      row per registry entry.

  (3) Writes `platform_hybrid/experiments/results/p5p8/p6_outcomes_coverage_summary.json` with
      cross-table aggregate stats.

  (4) Patches `registry/schema.json` additively to add the `coverage` object
      as an OPTIONAL property inside `outcomes`. The patch is idempotent —
      if `coverage` is already defined, this is a no-op.

  (5) Patches all 31 entries to add `outcomes.coverage` if absent. If already
      present, this is a no-op (we preserve a prior manual entry).

  (6) Validates the result via the existing `python3 registry/query.py
      validate` machinery.

Stdlib only. ≤300 LoC. Run from the worktree root:

    python3 platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REGISTRY = ROOT / "registry"
SCHEMA = REGISTRY / "schema.json"
ENTRIES_DIR = REGISTRY / "entries"
RESULTS = ROOT / "experiments" / "results" / "p5p8"

MIN_REPORT_ITEMS = (
    "loss_form", "reference_kl", "sampler_backend", "telemetry",
    "group_size_schedule", "heldout_split", "decontamination",
)


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def dump_json(path: Path, obj, *, indent: int = 2) -> None:
    path.write_text(json.dumps(obj, indent=indent, sort_keys=False) + "\n")


def is_populated(value) -> bool:
    """JSON null counts as UNPOPULATED; everything else counts as populated.

    Empty string and empty array also count as UNPOPULATED (degenerate).
    """
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def coverage_min_report(entry: dict) -> tuple[float, int, int]:
    """Return (fraction, n_populated, n_total) for the 7 MIN-REPORT items."""
    mr = entry.get("min_report", {})
    if not isinstance(mr, dict):
        return 0.0, 0, len(MIN_REPORT_ITEMS)
    n_pop = 0
    for item in MIN_REPORT_ITEMS:
        sub = mr.get(item)
        if not isinstance(sub, dict):
            continue
        # At least one leaf is non-null and non-empty
        if any(is_populated(v) for v in sub.values()):
            n_pop += 1
    return n_pop / len(MIN_REPORT_ITEMS), n_pop, len(MIN_REPORT_ITEMS)


def coverage_declared_deltas(entry: dict) -> tuple[float, int, int, int]:
    """Return (fraction_informative, n_informative, n_total_declared, n_implemented).

    Stacks only — delta records don't carry variant_deltas_applied.
    """
    vda = entry.get("variant_deltas_applied")
    if not isinstance(vda, list):
        return 0.0, 0, 0, 0
    n_total = len(vda)
    n_inform = sum(1 for d in vda if isinstance(d, dict)
                   and d.get("status") in ("implemented", "surrogate"))
    n_impl = sum(1 for d in vda if isinstance(d, dict)
                 and d.get("status") == "implemented")
    return (n_inform / n_total if n_total else 0.0), n_inform, n_total, n_impl


def coverage_measured(entry: dict) -> tuple[float, int, int]:
    """For delta records: measured rows / declared components.
    For stack records: outcomes fields populated / 5 (mean_last10, mean_zvf,
    heldout_delta, rollouts, ci_method).
    """
    rt = entry.get("record_type")
    if rt == "variant_delta":
        measured = entry.get("measured") or []
        deltas = entry.get("deltas") or []
        n_total = max(1, len(deltas))
        n_pop = sum(1 for m in measured if isinstance(m, dict)
                    and is_populated(m.get("metric"))
                    and is_populated(m.get("delta")))
        return n_pop / n_total, n_pop, n_total
    # stack record
    out = entry.get("outcomes") or {}
    fields = ("mean_last10_train_reward", "mean_zvf", "heldout_delta",
              "rollouts", "ci_method")
    n_pop = sum(1 for f in fields if is_populated(out.get(f)))
    return n_pop / len(fields), n_pop, len(fields)


def ci_method_present(entry: dict) -> bool:
    out = entry.get("outcomes")
    if not isinstance(out, dict):
        return False
    cm = out.get("ci_method")
    return isinstance(cm, dict) and is_populated(cm.get("method"))


def compute_entry_coverage(entry: dict) -> dict:
    rt = entry.get("record_type")
    min_rate, min_pop, min_tot = coverage_min_report(entry)
    declared_rate, n_inform, n_decl, n_impl = coverage_declared_deltas(entry)
    meas_rate, meas_pop, meas_tot = coverage_measured(entry)
    ci_present = ci_method_present(entry)
    return {
        "record_type": rt,
        "min_report_coverage": round(min_rate, 4),
        "min_report_pop": min_pop,
        "min_report_total": min_tot,
        "declared_deltas_coverage": round(declared_rate, 4),
        "declared_deltas_informative": n_inform,
        "declared_deltas_total": n_decl,
        "declared_deltas_implemented": n_impl,
        "measured_coverage": round(meas_rate, 4),
        "measured_pop": meas_pop,
        "measured_total": meas_tot,
        "ci_method_present": ci_present,
    }


def patch_schema() -> bool:
    """Add `outcomes.coverage` as an OPTIONAL additive property.

    Returns True if a change was made.
    """
    schema = load_json(SCHEMA)
    defs = schema.get("$defs", {})
    stack_record = defs.get("stack_record", {})
    outcomes = stack_record.get("properties", {}).get("outcomes", {})
    if not isinstance(outcomes, dict):
        print("ERROR: schema stack_record.properties.outcomes not a dict", file=sys.stderr)
        sys.exit(1)
    props = outcomes.setdefault("properties", {})
    if "coverage" in props:
        return False  # already defined
    # Add the optional coverage block. All fields nullable per the iter-28
    # `ci_method` pattern; the registry convention is null = unreported.
    props["coverage"] = {
        "type": ["object", "null"],
        "description": (
            "Iter-62: self-reported coverage of the entry's reporting blocks. "
            "All fields nullable; null = not yet audited. min_report_coverage "
            "is the fraction of7 MIN-REPORT items with at least one sub-field "
            "populated. declared_deltas_coverage is the fraction of "
            "variant_deltas_applied entries with status in {implemented, "
            "surrogate}. measured_coverage is variant: measured_rows / "
            "delta_components, or stack: outcomes_fields_populated / 5. "
            "ci_method_present mirrors outcomes.ci_method being non-null. "
            "audit_source is the script path that produced the block."
        ),
        "properties": {
            "min_report_coverage": {"type": ["number", "null"]},
            "declared_deltas_coverage": {"type": ["number", "null"]},
            "measured_coverage": {"type": ["number", "null"]},
            "ci_method_present": {"type": ["boolean", "null"]},
            "audit_source": {"type": ["string", "null"]},
            "audit_date": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }
    schema["$defs"] = defs
    dump_json(SCHEMA, schema)
    return True


def patch_entry(entry_path: Path) -> bool:
    """Add outcomes.coverage to a single entry if not present.

    Returns True if the file was modified.
    """
    entry = load_json(entry_path)
    if entry.get("record_type") != "stack":
        return False  # coverage block is on stack_record only (variant_delta
                      # does not carry `outcomes` — see schema defs)
    out = entry.get("outcomes")
    if not isinstance(out, dict):
        out = {}
        entry["outcomes"] = out
    if "coverage" in out:
        return False
    cov = compute_entry_coverage(entry)
    # Restrict to the schema-bounded subset (avoid "additionalProperties: false")
    cov_schema_bounded = {
        "min_report_coverage": cov["min_report_coverage"],
        "declared_deltas_coverage": cov["declared_deltas_coverage"],
        "measured_coverage": cov["measured_coverage"],
        "ci_method_present": cov["ci_method_present"],
        "audit_source": "platform_modal/scripts/p5p8/p6_outcomes_coverage_block.py",
        "audit_date": "2026-07-05",
    }
    out["coverage"] = cov_schema_bounded
    dump_json(entry_path, entry)
    return True


def main():
    schema_changed = patch_schema()
    print(f"schema patch: {'applied' if schema_changed else 'noop (already patched)'}")

    n_patched = 0
    per_entry = []
    for entry_path in sorted(ENTRIES_DIR.glob("*.json")):
        entry = load_json(entry_path)
        cov = compute_entry_coverage(entry)
        cov_row = {"entry_id": entry_path.stem, **cov}
        per_entry.append(cov_row)
        if patch_entry(entry_path):
            n_patched += 1
    print(f"entries patched: {n_patched}/31")

    # Aggregate stats
    stacks = [r for r in per_entry if r["record_type"] == "stack"]
    deltas = [r for r in per_entry if r["record_type"] == "variant_delta"]
    mean_min = sum(r["min_report_coverage"] for r in stacks) / len(stacks)
    mean_decl = sum(r["declared_deltas_coverage"] for r in stacks) / len(stacks)
    mean_meas_s = sum(r["measured_coverage"] for r in stacks) / len(stacks)
    mean_meas_d = sum(r["measured_coverage"] for r in deltas) / len(deltas)
    n_ci = sum(1 for r in stacks if r["ci_method_present"])
    # Cross-table consistency check: a stack with variant_deltas_applied
    # pointing to a delta that has 0 measured rows is a claim-without-evidence.
    delta_measured = {r["entry_id"]: r["measured_pop"] for r in deltas}
    claim_without_evidence = []
    for r in stacks:
        entry = load_json(ENTRIES_DIR / f"{r['entry_id']}.json")
        for d in (entry.get("variant_deltas_applied") or []):
            if not isinstance(d, dict):
                continue
            did = d.get("delta_id")
            st = d.get("status")
            if st in ("implemented", "surrogate"):
                rows = delta_measured.get(did, 0)
                claim_without_evidence.append({
                    "stack": r["entry_id"], "delta": did,
                    "status": st, "delta_measured_rows": rows,
                })
    n_cwe = len(claim_without_evidence)
    n_cwe_with_rows = sum(1 for c in claim_without_evidence if c["delta_measured_rows"] > 0)
    n_cwe_without_rows = n_cwe - n_cwe_with_rows

    # Write audit TSV
    RESULTS.mkdir(parents=True, exist_ok=True)
    tsv_path = RESULTS / "p6_outcomes_coverage_audit.tsv"
    fields = ["entry_id", "record_type", "min_report_coverage",
              "min_report_pop", "min_report_total",
              "declared_deltas_coverage", "declared_deltas_informative",
              "declared_deltas_total", "declared_deltas_implemented",
              "measured_coverage", "measured_pop", "measured_total",
              "ci_method_present"]
    with tsv_path.open("w") as fh:
        fh.write("\t".join(fields) + "\n")
        for r in per_entry:
            fh.write("\t".join(str(r.get(f, "")) for f in fields) + "\n")
    print(f"wrote {tsv_path} ({len(per_entry)} rows)")

    # Write claim-without-evidence audit
    cwe_path = RESULTS / "p6_outcomes_coverage_claim_evidence.tsv"
    with cwe_path.open("w") as fh:
        fh.write("stack\tdelta\tstatus\tdelta_measured_rows\n")
        for c in claim_without_evidence:
            fh.write(f"{c['stack']}\t{c['delta']}\t{c['status']}\t{c['delta_measured_rows']}\n")
    print(f"wrote {cwe_path} ({len(claim_without_evidence)} rows)")

    # Summary JSON
    summary = {
        "n_entries": len(per_entry),
        "n_stacks": len(stacks),
        "n_deltas": len(deltas),
        "schema_patched": schema_changed,
        "n_entries_patched": n_patched,
        "stack_means": {
            "min_report_coverage": round(mean_min, 4),
            "declared_deltas_coverage": round(mean_decl, 4),
            "measured_coverage": round(mean_meas_s, 4),
            "n_with_ci_method": n_ci,
        },
        "delta_means": {
            "measured_coverage": round(mean_meas_d, 4),
        },
        "claim_without_evidence": {
            "n_total": n_cwe,
            "n_with_measured_rows": n_cwe_with_rows,
            "n_without_measured_rows": n_cwe_without_rows,
        },
    }
    sum_path = RESULTS / "p6_outcomes_coverage_summary.json"
    dump_json(sum_path, summary)
    print(f"wrote {sum_path}")
    print(json.dumps(summary, indent=2))

    # CI-style validate
    print("\n=== Running registry/query.py validate ===")
    rc = subprocess.run(
        [sys.executable, str(REGISTRY / "query.py"), "validate"],
        cwd=str(ROOT), check=False,
    )
    print(f"validate exit: {rc.returncode}")
    if rc.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()