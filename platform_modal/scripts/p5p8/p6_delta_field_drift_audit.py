#!/usr/bin/env python3
"""Iter 42 (P6 / Pillar 2) — Delta `field:` self-claim ↔ schema MIN-REPORT
drift audit.

For every (delta_id, component) pair the catalog defines, the audit
parses the component's `field:` self-claim and classifies it against the
actual schema MIN-REPORT block. Verdicts:

    OK                       — `field:` names a real min_report.{item}.{leaf}
                               path AND the leaf is populate-able (not $ref
                               to a primitive nullable).
    BLOCK_NOT_IN_MIN_REPORT  — `field:` names a block that the schema does
                               not expose (e.g. 'sampling.*', 'reward.*').
                               Captured by iter-30's surface map.
    LEAF_NOT_IN_SCHEMA       — `field:` names a block that exists but the
                               given leaf is not present (legacy name from
                               before an iter-41 schema bump, or a typo).
    AMBIGUOUS_REFERENCE      — `field:` names a block but no leaf
                               ('reference_kl' or 'loss_form.clip' alone).
                               Honest move: pin to the canonical leaf.
    SEE_CITATION             — component defers to the source paper; field
                               is intentionally unset.

Inputs:
  registry/schema.json
  registry/entries/delta_*.json

Outputs:
  platform_hybrid/experiments/results/p5p8/p6_delta_field_drift.tsv          (per-row)
  platform_hybrid/experiments/results/p5p8/p6_delta_field_drift_summary.json  (machine-readable)
"""
from __future__ import annotations

import json, pathlib, datetime, warnings
from collections import defaultdict, Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = pathlib.Path("registry")
OUT = pathlib.Path("platform_hybrid/experiments/results/p5p8")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# (1) Extract the schema's real MIN-REPORT leaf surface.
# ------------------------------------------------------------------
schema = json.load(open(ROOT / "schema.json"))
defs = schema.get("$defs", {})


def deref(node):
    """Resolve local $refs (we never need network resolution here)."""
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if ref.startswith("#/$defs/"):
                return deref(defs[ref.split("/")[-1]])
        return {k: deref(v) for k, v in node.items()}
    if isinstance(node, list):
        return [deref(x) for x in node]
    return node


stack_def = deref(schema["$defs"]["stack_record"])
min_report = stack_def["properties"]["min_report"]["properties"]

# Build {item: set_of_leaf_names} from the schema.
block_leaves: dict[str, set[str]] = {}
for item_name, item_def in min_report.items():
    out: set[str] = set()

    def walk(o):
        """Recurse into nested `properties` blocks, collecting leaf names."""
        if not isinstance(o, dict):
            return
        for k, v in o.items():
            if k == "properties" and isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and "properties" in vv:
                        walk(vv)
                    elif isinstance(vv, dict) and "$ref" in vv:
                        out.add(kk)
                    elif isinstance(vv, dict):
                        out.add(kk)
                    else:
                        out.add(kk)

    walk(item_def)
    block_leaves[item_name] = out


# ------------------------------------------------------------------
# (2) Parse each delta entry's (component, field) claim.
# ------------------------------------------------------------------
def classify(field: str) -> tuple[str, str]:
    """Return (verdict, block_or_reason)."""
    if not field:
        return ("EMPTY", "no field declared")
    if field == "see delta-list and citation":
        return ("SEE_CITATION", "deferred to source paper")
    if "." not in field:
        return ("AMBIGUOUS_REFERENCE", f"block-only reference: {field!r}")
    block, leaf = field.split(".", 1)
    if block not in block_leaves:
        return ("BLOCK_NOT_IN_MIN_REPORT", f"block {block!r} not in MIN-REPORT")
    if leaf not in block_leaves[block]:
        return ("LEAF_NOT_IN_SCHEMA",
                f"leaf {leaf!r} missing from MIN-REPORT.{block}")
    return ("OK", f"{block}.{leaf}")


rows = []
counts = Counter()
per_delta_drift = defaultdict(list)
for f in sorted((ROOT / "entries").glob("delta_*.json")):
    d = json.load(open(f))
    delta_id = d["id"]
    for c in d.get("deltas", []):
        comp, field = c["component"], c.get("field", "")
        verdict, reason = classify(field)
        counts[verdict] += 1
        if verdict != "OK" and verdict != "SEE_CITATION":
            per_delta_drift[delta_id].append((comp, field, verdict, reason))
        rows.append({
            "delta_id": delta_id,
            "component": comp,
            "field_claim": field,
            "verdict": verdict,
            "reason": reason,
            "change_text": c.get("change", "")[:120],
        })

# ------------------------------------------------------------------
# (3) Write TSV + JSON summary.
# ------------------------------------------------------------------
tsv = OUT / "p6_delta_field_drift.tsv"
with open(tsv, "w") as fh:
    cols = ["delta_id", "component", "field_claim", "verdict", "reason", "change_text"]
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ")
                          for c in cols) + "\n")
print(f"wrote {tsv}  ({len(rows)} rows)")

# Repair proposals (the canonical iter-41 mapping for known drifts).
# (delta_id, component) -> proposed repair.
# 1. iter-41 schema-bump closed four DAPO gaps; the delta entries' `field:`
#    claims still name the pre-bump paths.
# 2. kl_removed is block-level — pin to a canonical leaf.
# 3. gspo's sequence-level clip has no MIN-REPORT leaf (no schema bump yet);
#    honestly mark it as SEE_CITATION (the deferred-surface pattern).
repair_table: dict[tuple[str, str], str] = {
    ("delta_dapo", "dynamic_sampling"):         "loss_form.sampling_dynamic_filter",
    ("delta_dapo", "token_level_loss"):         "loss_form.token_aggregation",
    ("delta_dapo", "overlong_reward_shaping"):  "loss_form.reward_shaping_type",
    ("delta_dapo", "kl_removed"):               "reference_kl.kl_beta",
    ("delta_gspo", "sequence_level_clip"):      "see delta-list and citation",
}

# JSON summary
summary = {
    "date": str(datetime.date.today()),
    "rows": len(rows),
    "verdict_counts": dict(counts),
    "per_delta_drift": {k: v for k, v in per_delta_drift.items()},
    "repair_proposals": [
        {"delta_id": di, "component": co,
         "current_field": next(
             (r["field_claim"] for r in rows
              if r["delta_id"] == di and r["component"] == co), ""),
         "proposed_field": pf, "rationale": "iter-42 drift audit"}
        for (di, co), pf in repair_table.items()
    ],
    "schema_block_leaf_counts": {k: len(v) for k, v in block_leaves.items()},
}
out_json = OUT / "p6_delta_field_drift_summary.json"
json.dump(summary, open(out_json, "w"), indent=2, sort_keys=True)
print(f"wrote {out_json}")

# ------------------------------------------------------------------
# (4) Console headline.
# ------------------------------------------------------------------
ok = counts.get("OK", 0)
see_cit = counts.get("SEE_CITATION", 0)
drift = sum(v for k, v in counts.items()
            if k not in ("OK", "SEE_CITATION", "EMPTY"))
total = ok + see_cit + drift
print()
print(f"=== Iter 42 — Delta field: drift audit ===")
print(f"Total (delta, component) pairs: {total}")
print(f"  OK (schema-anchored):        {ok}")
print(f"  SEE_CITATION (deferred):     {see_cit}")
print(f"  DRIFT (needs repair):        {drift}")
print(f"  drift rate:                  {drift/max(1, total):.3f}")
print()
print("Per-verdict breakdown:")
for v, n in sorted(counts.items()):
    print(f"  {v:24s} {n}")
