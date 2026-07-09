#!/usr/bin/env python3
"""P6 — Delta-Implementation Cross-Reference Matrix (iter 38)

Builds a (delta, component) × stack cross-reference matrix:
  - For each of 11 `delta_*.json` records, enumerate every (delta, component) pair.
  - For each of the 20 stack records, find every applied (delta, component) and
    classify by status: implemented / surrogate / absent / unknown / not_applicable.
  - Quantify three angles:
      (a) per-(delta, component) claim count, by status
      (b) per-stack claim count, by status
      (c) the "registry gap": which (delta, component) pairs are NEVER claimed?
  - Cross-link to iter-34's measured block: for variants with measured deltas,
    is the registry claim rate consistent with measurement (i.e., is a
    `surrogate` claim backed by a measured surrogate delta)?

Inputs
------
- registry/entries/{stack,delta}_*.json (20 + 11 = 31 entries)
- registry/entries/delta_{aero,areal,gift,cppo,ngrpo,mcgrpo,es,scafgrpo}.json
  (iter-34 measured block on `measured[]`)
- experiments/results/p5p8/p6_measured_delta_block.tsv (iter-34 measured TSV)

Outputs
-------
- experiments/results/p5p8/p6_delta_implementation_matrix.tsv
  (one row per (delta_id, component, stack_id); full cross-product)
- experiments/results/p5p8/p6_delta_implementation_matrix_summary.json
- prints a headline report

Stdlib only. ≤300 LoC.
"""
from __future__ import annotations
import csv, json, pathlib, sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root
ENTRIES = HERE / "registry" / "entries"
OUTDIR = HERE / "experiments" / "results" / "p5p8"
OUTDIR.mkdir(parents=True, exist_ok=True)
MEASURED_TSV = OUTDIR / "p6_measured_delta_block.tsv"

VALID_STATUSES = ("implemented", "surrogate", "absent", "unknown")


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def load_entries():
    stacks, deltas = {}, {}
    for p in sorted(ENTRIES.glob("*.json")):
        r = load_json(p)
        rid = r["id"]
        if r["record_type"] == "stack":
            stacks[rid] = r
        elif r["record_type"] == "variant_delta":
            deltas[rid] = r
        else:
            raise SystemExit(f"unknown record_type in {p.name}")
    return stacks, deltas


def load_measured_block():
    """Read iter-34 TSV and return {(delta_id, panel): [metric, delta, ci_low, ci_high, n]}."""
    out = defaultdict(list)
    if not MEASURED_TSV.exists():
        return out
    with MEASURED_TSV.open() as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            key = (row["delta_id"], row["panel"])
            out[key].append(row)
    return out


def build_matrix(stacks, deltas):
    """Return list of rows: (delta_id, component, stack_id, status, label_claimed)."""
    # First, enumerate all (delta, component) from the 11 delta records.
    delta_components = []
    for did, drec in sorted(deltas.items()):
        for comp in drec["deltas"]:
            delta_components.append((did, comp["component"]))
    # Index stacks' claims by (delta_id, component) -> status
    stack_claims = defaultdict(dict)  # (did, comp) -> {stack_id: status}
    for sid, srec in stacks.items():
        for vd in srec.get("variant_deltas_applied", []):
            stack_claims[(vd["delta_id"], vd["component"])][sid] = vd["status"]
    # Build full cross product
    rows = []
    for did, comp in delta_components:
        # Stacks that *could* claim this: those whose label is the variant OR
        # that have at least one applied_delta for this variant.
        label = deltas[did].get("base", "grpo")  # delta_records inherit from grpo
        for sid in sorted(stacks):
            srec = stacks[sid]
            claimed_status = stack_claims.get((did, comp), {}).get(sid)
            if claimed_status is None:
                status = "not_applicable"
            elif claimed_status not in VALID_STATUSES:
                status = "unknown_status"
            else:
                status = claimed_status
            rows.append({
                "delta_id": did,
                "component": comp,
                "stack_id": sid,
                "framework": srec["framework"]["name"],
                "openness": srec["framework"]["openness"],
                "label_claimed": srec["label_claimed"],
                "status": status,
            })
    return rows, delta_components


def summarise(rows, delta_components, stacks, measured):
    """Compute aggregate statistics."""
    by_dc = defaultdict(lambda: defaultdict(int))  # (did,comp) -> status -> count
    by_stack = defaultdict(lambda: defaultdict(int))  # stack -> status -> count
    by_delta = defaultdict(lambda: defaultdict(int))  # delta -> status -> count
    for r in rows:
        by_dc[(r["delta_id"], r["component"])][r["status"]] += 1
        by_stack[r["stack_id"]][r["status"]] += 1
        by_delta[r["delta_id"]][r["status"]] += 1
    # Gap: which (did, comp) have ZERO claims
    gap = []
    for did, comp in delta_components:
        counts = by_dc[(did, comp)]
        n_claims = sum(counts[s] for s in VALID_STATUSES)
        if n_claims == 0:
            gap.append((did, comp))
    # Headline: registry claim rate = claimed cells / (claimed + not_applicable).
    total_cells = len(rows)
    n_not_applicable = sum(1 for r in rows if r["status"] == "not_applicable")
    n_applicable = total_cells - n_not_applicable
    n_implemented = sum(1 for r in rows if r["status"] == "implemented")
    n_surrogate = sum(1 for r in rows if r["status"] == "surrogate")
    n_absent = sum(1 for r in rows if r["status"] == "absent")
    n_unknown = sum(1 for r in rows if r["status"] == "unknown")
    n_unknown_status = sum(1 for r in rows if r["status"] == "unknown_status")
    # Measured linkage
    measured_linkage = []
    for r in rows:
        if r["status"] in ("implemented", "surrogate"):
            for panel in ("n2_same_stack_last10", "zvf130_5seed"):
                for m in measured.get((r["delta_id"], panel), []):
                    measured_linkage.append({
                        "delta_id": r["delta_id"], "panel": panel,
                        "stack_status": r["status"], "stack_id": r["stack_id"],
                        "metric": m["metric"], "delta": m["delta"],
                        "ci_low": m["ci_low"], "ci_high": m["ci_high"],
                        "n": m["n"], "significant": m["significant"],
                    })
    summary = {
        "n_delta_records": len({did for did, _ in delta_components}),
        "n_delta_components": len(delta_components),
        "n_stacks": len(stacks),
        "n_cells": total_cells,
        "n_applicable_cells": n_applicable,
        "n_not_applicable_cells": n_not_applicable,
        "n_implemented": n_implemented,
        "n_surrogate": n_surrogate,
        "n_absent": n_absent,
        "n_unknown": n_unknown,
        "n_unknown_status": n_unknown_status,
        "registry_gap_pairs": gap,
        "n_registry_gap_pairs": len(gap),
        "claim_rate_implemented": round(n_implemented / n_applicable, 4) if n_applicable else 0,
        "claim_rate_surrogate": round(n_surrogate / n_applicable, 4) if n_applicable else 0,
        "claim_rate_absent": round(n_absent / n_applicable, 4) if n_applicable else 0,
        "claim_rate_unknown": round(n_unknown / n_applicable, 4) if n_applicable else 0,
        "by_dc": {f"{did}|{comp}": dict(v) for (did, comp), v in by_dc.items()},
        "by_stack": {sid: dict(v) for sid, v in by_stack.items()},
        "by_delta": {did: dict(v) for did, v in by_delta.items()},
        "n_measured_linkage_rows": len(measured_linkage),
    }
    return summary, measured_linkage


def main():
    stacks, deltas = load_entries()
    measured = load_measured_block()
    rows, delta_components = build_matrix(stacks, deltas)
    summary, measured_linkage = summarise(rows, delta_components, stacks, measured)
    # Write TSV
    out_tsv = OUTDIR / "p6_delta_implementation_matrix.tsv"
    fields = ["delta_id", "component", "stack_id", "framework", "openness",
              "label_claimed", "status"]
    with out_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    # Measured linkage TSV
    out_link = OUTDIR / "p6_delta_implementation_matrix_measured_linkage.tsv"
    if measured_linkage:
        with out_link.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(measured_linkage[0].keys()),
                               delimiter="\t")
            w.writeheader()
            for r in measured_linkage:
                w.writerow(r)
    # Summary JSON
    out_json = OUTDIR / "p6_delta_implementation_matrix_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    # Headline report
    print(f"[iter 38 P6] delta-implementation matrix")
    print(f"  delta records: {summary['n_delta_records']}, "
          f"(delta, component) pairs: {summary['n_delta_components']}, "
          f"stacks: {summary['n_stacks']}")
    print(f"  cells (cross product): {summary['n_cells']}")
    print(f"  applicable cells: {summary['n_applicable_cells']}  "
          f"not_applicable cells: {summary['n_not_applicable_cells']}")
    print(f"  implemented={summary['n_implemented']}  "
          f"surrogate={summary['n_surrogate']}  "
          f"absent={summary['n_absent']}  "
          f"unknown={summary['n_unknown']}")
    print(f"  registry-gap pairs (zero claims): "
          f"{summary['n_registry_gap_pairs']}/{summary['n_delta_components']}")
    print(f"  measured-linkage rows: {summary['n_measured_linkage_rows']}")
    print(f"  claim rate (impl/appl): "
          f"{summary['claim_rate_implemented']:.4f}")
    print(f"  wrote: {out_tsv.relative_to(HERE)}  "
          f"({out_tsv.stat().st_size}B)")
    print(f"  wrote: {out_json.relative_to(HERE)}  "
          f"({out_json.stat().st_size}B)")
    if measured_linkage:
        print(f"  wrote: {out_link.relative_to(HERE)}  "
              f"({out_link.stat().st_size}B)")


if __name__ == "__main__":
    main()