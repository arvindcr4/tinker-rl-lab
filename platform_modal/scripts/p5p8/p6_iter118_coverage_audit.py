#!/usr/bin/env python3
"""P6 iter-118 coverage audit.

Cross-cuts every (framework x method) cell the brief lists, scoring
whether the registry carries a stack record and what its MIN-REPORT
badge is. Emits:

    experiments/results/p5p8/p6_iter118_coverage_audit.tsv

with columns: framework, method, has_stack_entry, stack_id,
              schema_pass, badge, variant_deltas_resolved,
              n_orphan_deltas, n_missing_source, source_paths
"""
import csv
import json
import pathlib
import sys

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
REGISTRY = WORKTREE / "registry"
RESULTS = WORKTREE / "experiments" / "results" / "p5p8"

# Frameworks in repo (as listed in the brief) + 9 zvf130 methods.
FRAMEWORKS = ["tinker", "openrlhf", "verl", "trl",
              "atropos", "skyrl", "colab-open"]
METHODS = ["grpo", "aero", "gift", "areal", "ngrpo", "cppo",
           "mcgrpo", "es", "scafgrpo"]

# Stack entries that match a (framework, method).
STACKS = sorted(p for p in (REGISTRY / "entries").glob("*.json")
                if json.loads(p.read_text()).get("record_type") == "stack")

DELTAS = {p.stem for p in (REGISTRY / "entries").glob("delta_*.json")}


def match_stack(framework, method, stack):
    """Heuristic: framework matches first token before -/_ ;
    method matches label_claimed OR a delta id parsed from id."""
    rec = json.loads(stack.read_text())
    rid = rec.get("id", "")
    label = rec.get("label_claimed", "")
    fname = stack.stem.lower()
    if framework == "colab-open":
        # colab-open entries are named like colab-open_grpo_e3
        if not fname.startswith("colab-open"):
            return False
    else:
        # other frameworks: name begins with framework or framework_model
        if not fname.startswith(framework):
            return False
    if method == "grpo-adaptiveg":
        return "adaptiveg" in fname
    return method in fname or label == method


def leaves(d):
    """Yield every leaf key path in a (possibly nested) dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            for k2 in leaves(v):
                yield f"{k}.{k2}"
        else:
            yield k


def badge(rec):
    """MIN-REPORT badge as a 0-100 integer fraction of non-null leaves."""
    mr = rec.get("min_report")
    if not mr:
        return None
    n_nonull = 0
    n_total = 0
    for k in leaves(mr):
        v = mr
        for seg in k.split("."):
            v = v[seg]
        n_total += 1
        if v is not None:
            n_nonull += 1
    if n_total == 0:
        return None
    return round(100 * n_nonull / n_total)


def source_check(rec):
    """Return number of measured-row source files that exist on disk."""
    n_ok = 0
    n_miss = 0
    for m in rec.get("measured", []):
        src = m.get("source")
        if src and (WORKTREE / src).exists():
            n_ok += 1
        else:
            n_miss += 1
    return n_ok, n_miss


def orphan_deltas(rec):
    """Return orphan delta_ids in variant_deltas_applied."""
    orphans = []
    for vda in rec.get("variant_deltas_applied", []):
        did = vda.get("delta_id")
        if did and did not in DELTAS:
            orphans.append(did)
    return orphans


def main():
    rows = []
    for framework in FRAMEWORKS:
        for method in METHODS:
            matched = [s for s in STACKS if match_stack(framework, method, s)]
            if not matched:
                rows.append({
                    "framework": framework, "method": method,
                    "has_stack_entry": "no",
                    "stack_id": "",
                    "schema_pass": "",
                    "badge": "",
                    "variant_deltas_resolved": "",
                    "n_orphan_deltas": "",
                    "n_measured_src_ok": "",
                    "n_measured_src_missing": "",
                    "stack_ids": "",
                    "note": "no matching entry",
                })
                continue
            ids = ";".join(s.stem for s in matched)
            for s in matched:
                rec = json.loads(s.read_text())
                try:
                    schema = json.loads((REGISTRY / "schema.json").read_text())
                    import jsonschema
                    jsonschema.validate(rec, schema)
                    ok = "PASS"
                except Exception:
                    ok = "FAIL"
                vdas = [v.get("delta_id") for v in rec.get("variant_deltas_applied", [])]
                vd_resolved = sum(1 for d in vdas if d in DELTAS)
                orphans = orphan_deltas(rec)
                nok, nmiss = source_check(rec)
                rows.append({
                    "framework": framework, "method": method,
                    "has_stack_entry": "yes",
                    "stack_id": s.stem,
                    "schema_pass": ok,
                    "badge": badge(rec) or 0,
                    "variant_deltas_resolved": f"{vd_resolved}/{len(vdas)}",
                    "n_orphan_deltas": len(orphans),
                    "n_measured_src_ok": nok,
                    "n_measured_src_missing": nmiss,
                    "stack_ids": ids,
                    "note": "",
                })

    cols = ["framework", "method", "has_stack_entry", "stack_id",
            "schema_pass", "badge", "variant_deltas_resolved",
            "n_orphan_deltas", "n_measured_src_ok",
            "n_measured_src_missing", "stack_ids", "note"]
    out = RESULTS / "p6_iter118_coverage_audit.tsv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Frameworks with at least one entry
    fw_present = sorted({r["framework"] for r in rows if r["has_stack_entry"] == "yes"})
    fw_absent = sorted({r["framework"] for r in rows if r["has_stack_entry"] == "no"})
    n_ok = sum(r["schema_pass"] == "PASS" for r in rows)
    n_total = len(rows)
    avg_badge = sum(int(r["badge"]) for r in rows if r["badge"]) / max(1, sum(1 for r in rows if r["badge"]))
    orphan_total = sum(int(r["n_orphan_deltas"] or 0) for r in rows)
    miss_src_total = sum(int(r["n_measured_src_missing"] or 0) for r in rows)
    print(f"# coverage audit: {n_total} (fw x method) cells")
    print(f"  frameworks present: {fw_present}")
    print(f"  frameworks absent:  {fw_absent}")
    print(f"  schema pass:        {n_ok}/{n_total}")
    print(f"  average badge:      {avg_badge:.1f}")
    print(f"  orphan deltas:      {orphan_total}")
    print(f"  missing sources:    {miss_src_total}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
