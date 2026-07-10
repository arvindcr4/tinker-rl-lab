#!/usr/bin/env python3
"""
Iter-146 P6 auto-patch: apply the iter-146 fix_plan.tsv (which lists 13 rows
where the stored delta is correct but the `source` path is misattributed).

For each row in p6_iter146_fix_plan.tsv:
  1. Read the corresponding registry entry.
  2. Walk its `measured[]` list; find the row whose (metric, panel) match.
  3. Update the `source` field to the new path.
  4. Append a `provenance_fix` note documenting the iter-146 audit.
  5. Write the patched entry back in place.
  6. Re-validate the registry post-patch via jsonschema (Draft 2020-12).

Output: platform_hybrid/experiments/results/p5p8/p6_iter146_patch_log.tsv
        platform_hybrid/experiments/results/p5p8/p6_iter146_post_patch_summary.json
"""
import csv
import json
import glob
import os
import sys

WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG_DIR = os.path.join(WORKTREE, "registry", "entries")
FIX_TSV = os.path.join(WORKTREE, "experiments", "results", "p5p8",
                       "p6_iter146_fix_plan.tsv")
OUT_DIR = os.path.join(WORKTREE, "experiments", "results", "p5p8")


def load_fix_plan():
    rows = []
    with open(FIX_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append(r)
    return rows


def main():
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema not installed", file=sys.stderr)
        return 1

    plan = load_fix_plan()
    log = []
    patched = 0
    for fr in plan:
        eid = fr["entry_id"]
        metric = fr["metric"]
        panel = fr["panel"]
        new_src = fr["new_source"]
        fpath = os.path.join(REG_DIR, eid + ".json")
        if not os.path.exists(fpath):
            log.append({"entry": eid, "status": "MISSING_FILE"})
            continue
        d = json.load(open(fpath))
        changed = False
        for m in (d.get("measured") or []):
            if m.get("metric") == metric and m.get("panel") == panel:
                old = m.get("source", "")
                m["source"] = new_src
                # Append fix note (preserving any prior note)
                prior = m.get("note", "")
                fix_note = ("iter-146 provenance-recompute audit: stored delta "
                            f"matches recompute from {new_src}; source path was "
                            f"misattributed to {old}; patched in iter-146")
                if prior:
                    m["note"] = prior + " | " + fix_note
                else:
                    m["note"] = fix_note
                changed = True
                log.append({
                    "entry": eid,
                    "metric": metric,
                    "panel": panel,
                    "old_source": old,
                    "new_source": new_src,
                    "stored_delta": m.get("delta"),
                    "status": "PATCHED",
                })
                patched += 1
        if changed:
            with open(fpath, "w") as f:
                json.dump(d, f, indent=2)
                f.write("\n")

    # re-validate
    schema = json.load(open(os.path.join(WORKTREE, "registry", "schema.json")))
    n_pass = 0
    n_fail = 0
    fails = []
    for fpath in sorted(glob.glob(os.path.join(REG_DIR, "*.json"))):
        try:
            jsonschema.validate(json.load(open(fpath)), schema)
            n_pass += 1
        except jsonschema.ValidationError as e:
            n_fail += 1
            fails.append({"file": os.path.basename(fpath), "error": str(e)[:120]})

    out_log = os.path.join(OUT_DIR, "p6_iter146_patch_log.tsv")
    with open(out_log, "w") as f:
        f.write("entry\tmetric\tpanel\told_source\tnew_source\t"
                "stored_delta\tstatus\n")
        for r in log:
            f.write("\t".join(str(r.get(c, "")) for c in
                              ["entry", "metric", "panel", "old_source",
                               "new_source", "stored_delta", "status"]) + "\n")

    out_json = os.path.join(OUT_DIR, "p6_iter146_post_patch_summary.json")
    summary = {
        "patched_rows": patched,
        "schema_pass": n_pass,
        "schema_fail": n_fail,
        "fails": fails,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())