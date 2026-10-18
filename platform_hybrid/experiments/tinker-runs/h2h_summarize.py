"""Build a descriptive H2H summary from explicit campaign membership.

No import-time I/O and no fixed scientific conclusions. Unverified or conflicting
results are rejected, missing measurements remain missing, and surrogate IDs
are used in place of requested canonical labels.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from platform_hybrid.experiments.framework_contracts import require_completed, validate_resume


def summarize(records, manifest):
    expected = manifest["cells"]  # tag -> immutable specification
    selected = {}
    for record in records:
        tag = record.get("tag")
        if tag not in expected:
            continue
        require_completed(record)
        validate_resume(expected[tag], record)
        if tag in selected and selected[tag] != record:
            raise ValueError(f"Conflicting duplicate result: {tag}")
        selected[tag] = record
    arms = defaultdict(list)
    for record in selected.values():
        objective = record.get("objective_id")
        if not objective:
            raise ValueError("Implemented objective identity required")
        arms[objective].append(record)
    summary = {
        "cells_requested": len(expected),
        "cells_observed": len(selected),
        "missing_cells": sorted(set(expected) - set(selected)),
        "by_objective": {},
        "comparison_contract_satisfied": False,
        "interpretation": "Descriptive recipe results; framework equivalence is unverified",
    }
    for arm, rows in arms.items():
        values = {
            "last10_avg": [r.get("last10_avg") for r in rows],
            "peak_reward": [r.get("peak_reward") for r in rows],
            "zvf": [
                statistics.mean(
                    [s["zvf"] for s in r.get("step_log", []) if s.get("zvf") is not None]
                )
                if any(s.get("zvf") is not None for s in r.get("step_log", []))
                else None
                for r in rows
            ],
        }
        summary["by_objective"][arm] = {
            k: {"mean": statistics.mean(v) if v else None, "n": len(v)}
            for k, raw in values.items()
            for v in [[x for x in raw if x is not None]]
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        [json.loads(p.read_text()) for p in args.results.glob("*.json")],
        json.loads(args.manifest.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
