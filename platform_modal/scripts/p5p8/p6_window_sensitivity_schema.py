#!/usr/bin/env python3
"""P6 JOB B / SYNTH (iter 84) — window-sensitivity registry schema backfill.

Closes the iter-82 row 97 mint recommendation:
  "add a new schema-optional field `window_sensitivity` to the
   `measured[]` record (allowed values STABLE-DIRECTION-MAG-SHIFT,
   FRAGILE-SIGN-FLIP, STABLE; default STABLE-DIRECTION-MAG-SHIFT),
   and a companion `robust_panel` string recording the most generous
   panel under which the effect remains significant."

Schema extension (additive, optional, nullable):
  - measured_delta.window_sensitivity : string enum
  - measured_delta.robust_panel        : string enum

Classification rule (per method x metric cell on the n2 same-stack corpus,
reading platform_hybrid/experiments/results/p5p8/p6_n2_window_deltas.tsv):

  STABLE
    -> same sign at full40 / last10 / last5 AND all three panels are SIG,
       AND |delta_full40 - delta_last5| / |delta_full40| < 0.30
  FRAGILE-SIGN-FLIP
    -> sign(d_full40) != sign(d_last10) OR sign(d_full40) != sign(d_last5)
  STABLE-DIRECTION-MAG-SHIFT
    -> everything else (default; matches the iter-82 row 97 default)

robust_panel:
  -> the largest-window panel (in {full40, last20, last10, last5}) at
     which the cell is significant AND matches the sign at full40;
     "none" if no panel is significant (vacuous cell).
"""
from __future__ import annotations
import csv
import json
import os
import re
import sys
from copy import deepcopy

import jsonschema

ROOT = "/home/claude/tinker-rl-lab-minimax"
SCHEMA = f"{ROOT}/registry/schema.json"
ENTRIES = f"{ROOT}/registry/entries"
ITER_DATA = f"{ROOT}/platform_hybrid/experiments/results/p5p8/p6_n2_window_deltas.tsv"
SUMMARY = f"{ROOT}/platform_hybrid/experiments/results/p5p8/p6_window_sensitivity_backfill.json"
LOG = f"{ROOT}/platform_hybrid/experiments/results/p5p8/p6_window_sensitivity_backfill.log"

WINDOW_ORDER = ["full40", "last20", "last10", "last5", "early10"]
SIG_PANELS = ["full40", "last20", "last10", "last5"]  # ignore early10 (low-signal)
DELTA_ENTRY_IDS = ["delta_aero", "delta_gift", "delta_areal"]

ALLOWED = ["STABLE", "STABLE-DIRECTION-MAG-SHIFT", "FRAGILE-SIGN-FLIP"]


def load_window_deltas() -> dict:
    """Return {(method, metric): {panel: (delta, ci_lo, ci_hi, sig)}}."""
    out = {}
    with open(ITER_DATA) as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        cols = rdr.fieldnames
        for row in rdr:
            metric = row["metric"]
            window = row["window"]
            for method in ["aero", "gift", "areal"]:
                delta = float(row[f"{method}_delta"])
                ci_lo = float(row[f"{method}_ci_lo"])
                ci_hi = float(row[f"{method}_ci_hi"])
                sig_s = row.get(f"{method}_sig", "True" if ci_lo > 0 or ci_hi < 0 else "False")
                sig = sig_s.lower() == "true"
                out.setdefault((method, metric), {})[window] = {
                    "delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi, "sig": sig,
                }
    return out


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def classify_cell(method: str, metric: str, panels: dict) -> tuple[str, str]:
    """Classify (method, metric) -> (window_sensitivity, robust_panel).

    Sign comparisons use a noise floor of |delta| > 0.005 so that
    full40-delta=0.0 (a degenerate 'no measurable effect' measurement,
    not a real sign) does not flip the verdict.
    """
    full40 = panels.get("full40")
    last10 = panels.get("last10")
    last5 = panels.get("last5")

    if full40 is None or last10 is None or last5 is None:
        return "STABLE-DIRECTION-MAG-SHIFT", "none"

    SIG_FLOOR = 0.005
    d_f = full40["delta"]
    d_10 = last10["delta"]
    d_5 = last5["delta"]
    s_f = sign(d_f) if abs(d_f) > SIG_FLOOR else 0
    s_10 = sign(d_10) if abs(d_10) > SIG_FLOOR else 0
    s_5 = sign(d_5) if abs(d_5) > SIG_FLOOR else 0

    # FRAGILE-SIGN-FLIP: only count sign disagreement when both panels
    # have non-trivial measured delta (i.e., NOT a "no effect at full40"
    # degenerate case).
    if (s_f != 0 and s_5 != 0 and s_f != s_5) or (s_f != 0 and s_10 != 0 and s_f != s_10):
        return "FRAGILE-SIGN-FLIP", _robust_panel(panels)

    # STABLE: all three sig AND same sign AND magnitude consistent (<=30% rel diff)
    if full40["sig"] and last10["sig"] and last5["sig"] and abs(d_f) > SIG_FLOOR:
        if abs((d_f - d_5) / d_f) < 0.30:
            return "STABLE", "full40"

    # Default (covers magnitude-shifts and not-significant full40)
    return "STABLE-DIRECTION-MAG-SHIFT", _robust_panel(panels)


def _robust_panel(panels: dict) -> str:
    """Largest-window panel where sig holds and sign matches full40; else 'none'."""
    full40 = panels.get("full40")
    s_f = sign(full40["delta"]) if full40 else 0
    for panel in SIG_PANELS:  # full40 first, narrowest as fallback
        p = panels.get(panel)
        if p is None:
            continue
        if p["sig"] and sign(p["delta"]) == s_f:
            return panel
    return "none"


# ---------- main ----------

def main():
    print("[1/5] Loading iter-82 window delta data ...")
    by_cell = load_window_deltas()
    print(f"  loaded {len(by_cell)} (method, metric) cells")
    for k, v in sorted(by_cell.items()):
        print(f"  {k}: {len(v)} windows")

    print("[2/5] Classifying each (method, metric) cell ...")
    classifications = {}
    for (method, metric), panels in by_cell.items():
        ws, rp = classify_cell(method, metric, panels)
        classifications[(method, metric)] = {
            "window_sensitivity": ws, "robust_panel": rp,
            "panels": {w: v for w, v in panels.items()},
        }
    for k, v in sorted(classifications.items()):
        print(f"  {k}: {v['window_sensitivity']:<28}  robust_panel={v['robust_panel']}")

    print("[3/5] Patching registry schema (additive optional fields) ...")
    with open(SCHEMA) as fh:
        schema = json.load(fh)
    md = schema["$defs"]["measured_delta"]
    if "window_sensitivity" not in md["properties"]:
        md["properties"]["window_sensitivity"] = {
            "type": "string",
            "enum": ALLOWED,
            "description": "iter-84: window-sensitivity verdict of this (metric, panel) row under remeasurement on full40 / last10 / last5 (paired bootstrap B=2000, seed 20260705); see docs/p5p8_improvements/97_p6_n2_window_sensitivity.md.",
        }
    if "robust_panel" not in md["properties"]:
        md["properties"]["robust_panel"] = {
            "type": "string",
            "enum": ["full40", "last20", "last10", "last5", "none"],
            "description": "iter-84: largest-window panel at which the same-sign significance holds; 'none' if no panel is significant.",
        }
    schema["$defs"]["measured_delta"] = md
    with open(SCHEMA, "w") as fh:
        json.dump(schema, fh, indent=2, sort_keys=False)

    print("[4/5] Backfilling 3 delta entries ...")
    summary = {"entries_patched": [], "cells_patched": []}
    for eid in DELTA_ENTRY_IDS:
        path = f"{ENTRIES}/{eid}.json"
        with open(path) as fh:
            data = json.load(fh)
        n_patched = 0
        measured = data.get("measured", [])
        for row in measured:
            panel = row.get("panel", "")
            metric = row.get("metric", "")
            if not panel.startswith("n2_same_stack_"):
                continue
            cell = classifications.get((eid.replace("delta_", ""), metric))
            if cell is None:
                continue
            row["window_sensitivity"] = cell["window_sensitivity"]
            row["robust_panel"] = cell["robust_panel"]
            n_patched += 1
            summary["cells_patched"].append({
                "entry": eid, "metric": metric, "panel": panel,
                "window_sensitivity": cell["window_sensitivity"],
                "robust_panel": cell["robust_panel"],
            })
        # append an inline notes addendumto make the relation explicit
        prior_notes = data.get("notes", "")
        addendum = (
            "  iter-84: window-sensitivity backfill on every measured[] row of "
            "panel=n2_same_stack_*: STABLE / FRAGILE-SIGN-FLIP / "
            "STABLE-DIRECTION-MAG-SHIFT classification + robust_panel recorded "
            "based on the iter-82 re-measurement "
            "(platform_hybrid/experiments/results/p5p8/p6_n2_window_deltas.tsv + "
            "p6_n2_window_sensitivity.json, B=2000 paired-step bootstrap)."
        )
        if "iter-84: window-sensitivity backfill" not in prior_notes:
            data["notes"] = (prior_notes + addendum).strip()
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        summary["entries_patched"].append({"entry": eid, "rows_patched": n_patched})
        print(f"  {eid}: {n_patched} measured[] rows backfilled")

    print("[5/5] jsonschema validation (post-bump) ...")
    n_pass, n_fail = 0, 0
    fails = []
    for fname in sorted(os.listdir(ENTRIES)):
        if not fname.endswith(".json"):
            continue
        path = f"{ENTRIES}/{fname}"
        with open(path) as fh:
            doc = json.load(fh)
        try:
            jsonschema.validate(doc, schema)
            n_pass += 1
        except jsonschema.ValidationError as exc:
            n_fail += 1
            fails.append({"entry": fname, "msg": str(exc)[:120]})

    summary["validation"] = {
        "n_pass": n_pass, "n_fail": n_fail,
        "fails": fails,
    }
    with open(SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(LOG, "w") as fh:
        fh.write(f"P6 window-sensitivity schema backfill (iter 84 SYNTH)\n")
        fh.write(f"validation pass={n_pass} fail={n_fail}\n")
        for f in fails:
            fh.write(f"  FAIL {f['entry']}: {f['msg']}\n")

    print(f"  pass={n_pass} fail={n_fail}")
    for f in fails:
        print(f"   FAIL {f['entry']}: {f['msg']}")

    # Output compact summary
    print()
    print("=== SUMMARY ===")
    print(f"Schema patched: 2 new optional fields (`window_sensitivity`, `robust_panel`) on `measured_delta`.")
    print(f"Entries backfilled: {len(summary['entries_patched'])}")
    print(f"Cells backfilled: {len(summary['cells_patched'])}")
    n_stable = sum(1 for c in summary['cells_patched'] if c['window_sensitivity'] == 'STABLE')
    n_shift = sum(1 for c in summary['cells_patched'] if c['window_sensitivity'] == 'STABLE-DIRECTION-MAG-SHIFT')
    n_frag = sum(1 for c in summary['cells_patched'] if c['window_sensitivity'] == 'FRAGILE-SIGN-FLIP')
    print(f"  STABLE:           {n_stable}/{len(summary['cells_patched'])}")
    print(f"  STABLE-DIR-MAG:   {n_shift}/{len(summary['cells_patched'])}")
    print(f"  FRAGILE-SIGN-FLIP:{n_frag}/{len(summary['cells_patched'])}")
    print(f"  Schema validate:  {n_pass}/{n_pass+n_fail} entries PASS")


if __name__ == "__main__":
    main()
