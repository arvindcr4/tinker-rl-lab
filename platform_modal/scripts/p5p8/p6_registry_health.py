#!/usr/bin/env python3
"""P6 iter-50: registry health audit + CI-style schema validator.

Combines brief veins (b) coverage audit + (c) schema validation script.
For every entry in registry/entries/ compute:
  - schema validity (jsonschema.validate against registry/schema.json)
  - MIN-REPORT badge per entry (same formula as query.py::badge)
  - framework × method coverage grid (which framework×method cells
    have at least one entry)
  - per-leaf null-rate across all stack records (how often a MIN-REPORT
    field is reported-as-null vs reported-as-value)
  - per-delta measured[], expected_effects[], claim_validation[]
    presence + verdict distribution (extends iter-46 audit)
  - cross-delta verdict signature: per delta, the verdict string of
    every claim_validation row, hashed to a short tag for comparison

Writes three artifacts to platform_hybrid/experiments/results/p5p8/:
  - p6_registry_health.tsv         (one row per entry; long format)
  - p6_registry_health_coverage.tsv (framework × method grid)
  - p6_registry_health_summary.json  (single object with headline stats)

The script is stdlib + jsonschema only. Exit code is 0 if every entry
passes schema validation, 1 otherwise (CI-friendly).
"""
import hashlib
import json
import pathlib
import sys

try:
    import jsonschema  # type: ignore
except ImportError:
    print("FATAL: jsonschema not installed; pip install jsonschema",
          file=sys.stderr)
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
REG = ROOT / "registry"
SCHEMA = json.load(open(REG / "schema.json"))
OUT = ROOT / "experiments" / "results" / "p5p8"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Load + classify
# -------------------------------------------------------------------------
stacks = {}
deltas = {}
for p in sorted((REG / "entries").glob("*.json")):
    rec = json.loads(p.read_text())
    if rec["record_type"] == "stack":
        stacks[rec["id"]] = rec
    elif rec["record_type"] == "variant_delta":
        deltas[rec["id"]] = rec
    else:
        print(f"WARN: unknown record_type={rec.get('record_type')} in {p.name}",
              file=sys.stderr)

ITEMS = ["loss_form", "reference_kl", "sampler_backend", "telemetry",
         "group_size_schedule", "heldout_split", "decontamination"]


def leaf_null_rate(d):
    """Fraction of leaves in a MIN-REPORT item that are reported-as-null."""
    if not isinstance(d, dict):
        return None
    leaves = list(d.values())
    if not leaves:
        return 0.0
    return sum(v is None for v in leaves) / len(leaves)


def leaf_coverage(d):
    if not isinstance(d, dict):
        return None
    leaves = list(d.values())
    if not leaves:
        return 0.0
    return sum(v is not None for v in leaves) / len(leaves)


def badge(rec):
    scores = {it: leaf_coverage(rec["min_report"][it]) for it in ITEMS}
    pct = round(100 * sum(scores.values()) / len(ITEMS))
    return pct, scores


# -------------------------------------------------------------------------
# 2. Schema validation pass
# -------------------------------------------------------------------------
schema_errors = []
for rid, rec in {**stacks, **deltas}.items():
    try:
        jsonschema.validate(rec, SCHEMA)
    except jsonschema.ValidationError as e:
        schema_errors.append((rid, str(e.message)[:160]))

# -------------------------------------------------------------------------
# 3. Per-entry health row (long format)
# -------------------------------------------------------------------------
rows = []
for rid, rec in stacks.items():
    pct, scores = badge(rec)
    null_rates = {it: round(leaf_null_rate(rec["min_report"][it]), 4)
                  for it in ITEMS}
    has_outcomes = any(v is not None for v in (rec.get("outcomes") or {}).values())
    has_ci = bool((rec.get("outcomes") or {}).get("ci_method"))
    n_variants = len(rec.get("variant_deltas_applied") or [])
    rows.append({
        "entry_id": rid,
        "record_type": "stack",
        "framework": rec["framework"]["name"],
        "openness": rec["framework"]["openness"],
        "label_claimed": rec["label_claimed"],
        "min_report_badge": pct,
        **{f"score_{it}": round(scores[it], 4) for it in ITEMS},
        **{f"null_rate_{it}": null_rates[it] for it in ITEMS},
        "n_variant_deltas": n_variants,
        "has_outcomes": has_outcomes,
        "has_ci_method": has_ci,
        "schema_pass": rid not in {eid for eid, _ in schema_errors},
        "schema_err": next((m for eid, m in schema_errors if eid == rid), ""),
    })

for rid, rec in deltas.items():
    measured = rec.get("measured") or []
    expected = rec.get("expected_effects") or []
    cv = rec.get("claim_validation") or []
    verdict_counts = {"SUPPORTS": 0, "NEUTRAL": 0, "CONTRADICTS": 0, "UNCLAIMED": 0}
    for row in cv:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    rows.append({
        "entry_id": rid,
        "record_type": "variant_delta",
        "framework": "",
        "openness": "",
        "label_claimed": "",
        "min_report_badge": "",
        **{f"score_{it}": "" for it in ITEMS},
        **{f"null_rate_{it}": "" for it in ITEMS},
        "n_variant_deltas": len(rec.get("deltas") or []),
        "has_outcomes": len(measured) > 0,
        "has_ci_method": any(
            (m.get("ci_method") or {}).get("method") not in (None, "", "point_no_perseed_sd")
            for m in measured),
        "schema_pass": rid not in {eid for eid, _ in schema_errors},
        "schema_err": next((m for eid, m in schema_errors if eid == rid), ""),
        "n_measured": len(measured),
        "n_expected_effects": len(expected),
        "n_claim_validation": len(cv),
        "n_supports": verdict_counts["SUPPORTS"],
        "n_neutral": verdict_counts["NEUTRAL"],
        "n_contradicts": verdict_counts["CONTRADICTS"],
        "n_unclaimed": verdict_counts["UNCLAIMED"],
    })

# Write per-entry TSV (long format)
cols = ["entry_id", "record_type", "framework", "openness", "label_claimed",
        "min_report_badge"] + [f"score_{it}" for it in ITEMS] + \
       [f"null_rate_{it}" for it in ITEMS] + \
       ["n_variant_deltas", "has_outcomes", "has_ci_method",
        "schema_pass", "schema_err",
        "n_measured", "n_expected_effects", "n_claim_validation",
        "n_supports", "n_neutral", "n_contradicts", "n_unclaimed"]
with open(OUT / "p6_registry_health.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

# -------------------------------------------------------------------------
# 4. Framework × method coverage grid
# -------------------------------------------------------------------------
frameworks = sorted({r["framework"]["name"] for r in stacks.values()})
methods = sorted({r["label_claimed"] for r in stacks.values()} | {d["id"] for d in deltas.values()})
grid = {}
for fw in frameworks:
    for m in methods:
        hits = []
        for rid, r in stacks.items():
            if r["framework"]["name"] == fw and r["label_claimed"] == m:
                hits.append(rid)
        for rid, r in deltas.items():
            if rid == m:
                hits.append(f"delta:{rid}")
        grid[(fw, m)] = hits

with open(OUT / "p6_registry_health_coverage.tsv", "w") as f:
    f.write("framework\tmethod\tn_entries\tentry_ids\n")
    for fw in frameworks:
        for m in methods:
            hits = grid[(fw, m)]
            f.write(f"{fw}\t{m}\t{len(hits)}\t{','.join(hits)}\n")

# -------------------------------------------------------------------------
# 5. Headline summary
# -------------------------------------------------------------------------
stack_null_rates = {it: [] for it in ITEMS}
for r in stacks.values():
    for it in ITEMS:
        stack_null_rates[it].append(leaf_null_rate(r["min_report"][it]))
mean_null_rate_per_item = {it: round(sum(v) / len(v), 4) if v else 0.0
                           for it, v in stack_null_rates.items()}

badges = [badge(r)[0] for r in stacks.values()]
overall_badge = round(sum(badges) / len(badges), 2) if badges else 0.0

cells_with_entry = sum(1 for v in grid.values() if v)
cells_total = len(frameworks) * len(methods)

all_verdicts = []
for rec in deltas.values():
    for row in rec.get("claim_validation") or []:
        all_verdicts.append(row["verdict"])
verdict_dist = {v: all_verdicts.count(v) for v in
                ("SUPPORTS", "NEUTRAL", "CONTRADICTS", "UNCLAIMED")}

# Cross-delta verdict signature (short hash)
def sig(rec):
    s = "|".join(f"{r['metric']}/{r['panel']}={r['verdict']}"
                 for r in (rec.get("claim_validation") or []))
    return hashlib.sha1(s.encode()).hexdigest()[:10]

sigs = {did: sig(d) for did, d in deltas.items()}
sig_clusters = {}
for did, s in sigs.items():
    sig_clusters.setdefault(s, []).append(did)

# Null-rate ranking (top 5 fields with highest null-rate across stacks)
field_null_pairs = [(it, mean_null_rate_per_item[it]) for it in ITEMS]
field_null_pairs.sort(key=lambda x: -x[1])

summary = {
    "n_entries_total": len(stacks) + len(deltas),
    "n_stack_records": len(stacks),
    "n_delta_records": len(deltas),
    "schema_pass_count": len(stacks) + len(deltas) - len(schema_errors),
    "schema_fail_count": len(schema_errors),
    "schema_fail_ids": [eid for eid, _ in schema_errors],
    "frameworks_count": len(frameworks),
    "methods_count": len(methods),
    "framework_x_method_cells": cells_total,
    "cells_with_entry": cells_with_entry,
    "coverage_rate": round(cells_with_entry / cells_total, 4) if cells_total else 0.0,
    "mean_min_report_badge": overall_badge,
    "badge_min": min(badges) if badges else 0,
    "badge_max": max(badges) if badges else 0,
    "mean_null_rate_per_item": mean_null_rate_per_item,
    "top5_null_fields": [{"item": it, "mean_null_rate": v}
                         for it, v in field_null_pairs[:5]],
    "stacks_with_outcomes": sum(1 for r in stacks.values()
                                if any(v is not None
                                       for v in (r.get("outcomes") or {}).values())),
    "stacks_with_ci_method": sum(
        1 for r in stacks.values()
        if (r.get("outcomes") or {}).get("ci_method")),
    "claim_validation_total": len(all_verdicts),
    "claim_validation_distribution": verdict_dist,
    "claim_validation_significant_share": round(
        (verdict_dist["SUPPORTS"] + verdict_dist["CONTRADICTS"]) / len(all_verdicts), 4)
        if all_verdicts else 0.0,
    "verdict_signature_clusters": {s: ids for s, ids in sig_clusters.items()
                                   if len(ids) > 1},
    "frameworks": frameworks,
    "methods": methods,
}

with open(OUT / "p6_registry_health_summary.json", "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

# -------------------------------------------------------------------------
# 6. Stdout report (CI-friendly, also useful for human inspection)
# -------------------------------------------------------------------------
print(f"P6 registry health audit — iter 50")
print(f"  entries: {len(stacks)} stack + {len(deltas)} delta "
      f"= {len(stacks) + len(deltas)} total")
print(f"  schema: {summary['schema_pass_count']}/{summary['n_entries_total']} PASS, "
      f"{summary['schema_fail_count']} FAIL")
print(f"  frameworks: {len(frameworks)}  methods: {len(methods)}  "
      f"cells: {cells_with_entry}/{cells_total} populated "
      f"({summary['coverage_rate']*100:.1f}%)")
print(f"  mean MIN-REPORT badge: {overall_badge}/100 "
      f"(min={summary['badge_min']}, max={summary['badge_max']})")
print(f"  null-rate per MIN-REPORT item:")
for it in ITEMS:
    print(f"    {it:24s} {mean_null_rate_per_item[it]*100:5.1f}%")
print(f"  claim_validation total: {len(all_verdicts)} rows; "
      f"distribution: {verdict_dist}")
print(f"  significant share (SUPPORTS+CONTRADICTS): "
      f"{summary['claim_validation_significant_share']*100:.1f}%")
print(f"  verdict-signature clusters (>1 delta sharing a signature):")
for s, ids in sig_clusters.items():
    if len(ids) > 1:
        print(f"    {s}: {ids}")
print()
print(f"  outputs:")
print(f"    {OUT / 'p6_registry_health.tsv'}")
print(f"    {OUT / 'p6_registry_health_coverage.tsv'}")
print(f"    {OUT / 'p6_registry_health_summary.json'}")

# Exit non-zero if any schema failure
sys.exit(1 if schema_errors else 0)