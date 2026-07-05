#!/usr/bin/env python3
"""P6 iter-122 cross-entry consistency check.

The registry has two record types (stack + variant_delta) and many entries.
iter-118 verified that every (delta_id, component) reference resolves and
every measured source path exists on disk. iter-122 closes brief vein (b) at
the cross-entry layer: when 2+ stacks both claim the same (delta_id,
component), do they AGREE on the implementation status?

The check surfaces three classes of inconsistency:

  H1  STATUS_CONFLICT: 2+ stacks reference the same (delta_id, component)
      but disagree on the status (e.g. implemented vs surrogate, or
      implemented vs absent). When one is a managed surrogate and another
      is the canonical open implementation, the registry should report
      both transparently rather than collapse them.

  H2  STATUS_HOMOGENEITY: when 2+ stacks reference the same (delta_id,
      component) and ALL agree on the status, emit a single HOMOGENEOUS
      row (sanity check -- confirms the registry is internally consistent).

  H3  EVIDENCE_MISSING: a stack's variant_deltas_applied references a
      component that is NOT in the named delta_*.json record's
      `deltas[].component` set. This catches stack entries that drift
      from the variant-delta definition (e.g. typo'd component name).

Output: experiments/results/p5p8/p6_iter122_cross_entry_consistency.tsv
(per (delta_id, component) row with N stacks, N statuses, verdict).
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
REGISTRY = WORKTREE / "registry"
RESULTS = WORKTREE / "experiments" / "results" / "p5p8"


def load():
    stacks = {}
    deltas = {}
    for p in sorted((REGISTRY / "entries").glob("*.json")):
        r = json.loads(p.read_text())
        if r["record_type"] == "stack":
            stacks[p.stem] = r
        else:
            deltas[p.stem] = r
    return stacks, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="Persist TSV to experiments/results/p5p8/")
    args = ap.parse_args()
    stacks, deltas = load()
    # index of declared (delta_id, component) -> delta_name
    declared_components = defaultdict(set)
    for did, d in deltas.items():
        for comp in d.get("deltas", []):
            declared_components[did].add(comp["component"])
    # index of (delta_id, component) -> [(stack_id, status, note)]
    refs = defaultdict(list)
    for sid, s in stacks.items():
        for vda in s.get("variant_deltas_applied", []):
            refs[(vda["delta_id"], vda["component"])].append(
                (sid, vda["status"], vda.get("note", "")))
    rows = []
    # Iterate declared (delta_id, component) pairs that have at least one
    # stack reference; this catches the common case where stack and delta
    # definitions agree.
    referenced_keys = sorted(refs.keys())
    n_conflict = 0
    n_homogeneous = 0
    n_evidence_missing = 0
    for did, comp in referenced_keys:
        # Check EVIDENCE_MISSING
        declared = did in deltas and comp in declared_components[did]
        if did not in deltas:
            # The delta_*.json record itself is missing -- iter-118 orphan
            # check already catches this; record for cross-reference.
            pass
        elif not declared:
            n_evidence_missing += 1
        # Check STATUS_HOMOGENEITY vs STATUS_CONFLICT
        statuses = sorted({st for _, st, _ in refs[(did, comp)]})
        stacks_list = ";".join(sorted(sid for sid, _, _ in refs[(did, comp)]))
        statuses_str = ";".join(statuses)
        verdict = ("CONFLICT" if len(statuses) > 1 else "HOMOGENEOUS")
        if verdict == "CONFLICT":
            n_conflict += 1
        else:
            n_homogeneous += 1
        rows.append({
            "delta_id": did,
            "component": comp,
            "n_stacks": len(refs[(did, comp)]),
            "n_unique_statuses": len(statuses),
            "statuses": statuses_str,
            "stacks": stacks_list,
            "delta_has_component": declared,
            "verdict": verdict,
        })
    # Write outputs
    if args.write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / "p6_iter122_cross_entry_consistency.tsv"
        with out.open("w") as fh:
            cols = ["delta_id", "component", "n_stacks", "n_unique_statuses",
                    "statuses", "stacks", "delta_has_component", "verdict"]
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    # Print summary
    print(f"# iter-122 cross-entry consistency: {len(rows)} (delta, component) cells")
    print(f"  HOMOGENEOUS       {n_homogeneous}")
    print(f"  CONFLICT          {n_conflict}")
    print(f"  evidence_missing  {n_evidence_missing}")
    if n_conflict > 0:
        print(f"--- CONFLICTING (delta, component) ---")
        for r in rows:
            if r["verdict"] == "CONFLICT":
                print(f"  {r['delta_id']}:{r['component']}: "
                      f"{r['statuses']}  on stacks=[{r['stacks']}]")
    if n_evidence_missing > 0:
        print(f"--- evidence_missing ---")
        for r in rows:
            if not r["delta_has_component"]:
                print(f"  {r['delta_id']}:{r['component']}: "
                      f"referenced by [{r['stacks']}] but delta entry does not declare this component")
    return 0


if __name__ == "__main__":
    sys.exit(main())