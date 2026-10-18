#!/usr/bin/env python3
"""Stage complete review17-v2 result objects; validate before publishing proposals.

Never overwrite historical registry entries. Applying a reviewed migration is a
separate operation. Old summaries without verified sample/provenance are rejected.
"""

import math
import copy
import json
from pathlib import Path
import sys
import jsonschema

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT
from review17_contract import VERSION, validate_interval

ROOT = REPO_ROOT
RES = ROOT / "platform_hybrid/experiments/results/p5p8/review17_v2"
DELTA_IDS = [
    f"delta_{m}"
    for m in ("aero", "areal", "cppo", "es", "gift", "grpo", "mcgrpo", "ngrpo", "scafgrpo")
]


def proposed_entry(entry, rec):
    result = rec["result"]
    if (
        result.get("analysis_version") != VERSION
        or rec.get("metric") != "mean_zvf"
        or not rec.get("source_sha256")
        or not validate_interval(result["ci_low"], result["ci_high"])
        or not math.isfinite(result["delta"])
        or not result.get("ci_method")
        or result["n"] < 2
        or result["n"] != len(set(result["sample_ids"]))
    ):
        raise ValueError("Incomplete or invalid recomputation provenance")
    if result["significant"] != (result["ci_low"] > 0 or result["ci_high"] < 0):
        raise ValueError("Inconsistent significance")
    entry = copy.deepcopy(entry)
    matches = [
        r
        for r in entry.get("measured", [])
        if r.get("metric") == "mean_zvf" and r.get("panel") == "zvf130_5seed"
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one measurement identity")
    matches[0].update(copy.deepcopy(result))
    matches[0].update(source_sha256=rec["source_sha256"], units=rec["units"], base="grpo")
    return entry


def main():
    summary = json.loads((RES / "p6_iter128_recompute_point_only_summary.json").read_text())
    if summary.get("analysis_version") != VERSION:
        raise ValueError("Historical recomputation is not safe to migrate")
    records = summary["recomputed"]
    by_id = {r["delta_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(DELTA_IDS):
        raise ValueError("Incomplete/duplicate recomputation panel")
    registry = ROOT / "platform_hybrid/registry"
    schema = json.loads((registry / "schema.json").read_text())
    proposed = {}
    for eid in DELTA_IDS:
        entry = json.loads((registry / "entries" / f"{eid}.json").read_text())
        proposed[eid] = proposed_entry(entry, by_id[eid])
        jsonschema.validate(proposed[eid], schema)
    (RES / "p6_iter128_proposed_entries.json").write_text(
        json.dumps(proposed, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
