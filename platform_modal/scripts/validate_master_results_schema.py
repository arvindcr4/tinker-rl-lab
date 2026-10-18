#!/usr/bin/env python3
"""
Validate master_results.json schema consistency.

This script checks that all rows in master_results.json conform to a canonical
schema and reports any inconsistencies.

Usage:
    python3 platform_modal/scripts/validate_master_results_schema.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Base directory
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
MASTER_RESULTS = BASE_DIR / "experiments" / "master_results.json"

# Canonical schema
CANONICAL_SCHEMA = {
    "run_id": str,
    "paper_claim_ids": list,
    "model": str,
    "model_short": str,
    "method": str,
    "platform": str,
    "task": str,
    "split": str,
    "seed": int,
    "steps_completed": int,
    "group_size": int,
    "learning_rate": float,
    "metric_type": str,
    "peak_reward": float,
    "last10_avg": float,
    "reward_trace": list,
    "status": str,
    "source_files": list,
}

# Field aliases (old names -> new names)
FIELD_ALIASES = {
    "experiment_id": "run_id",
    "experiment": "run_id",
    "name": "run_id",
    "model_id": "model",
    "algorithm": "method",
    "task_name": "task",
    "steps": "steps_completed",
    "last_10_avg": "last10_avg",
    "last-10": "last10_avg",
    "url": "source_files",
    "wandb_url": "source_files",
    "wandb_run_url": "source_files",
}


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def normalize_row(row: Dict) -> Dict:
    """Normalize a row using aliases."""
    normalized = {}
    for key, value in row.items():
        new_key = FIELD_ALIASES.get(key, key)
        if new_key == "source_files" and key != "source_files" and isinstance(value, str):
            value = [value]
        if new_key in normalized and normalized[new_key] != value:
            raise ValueError(f"conflicting aliases for {new_key}")
        normalized[new_key] = value
    return normalized


NULLABLE_WHEN_UNAVAILABLE = {"peak_reward", "last10_avg", "reward_trace"}


def validate_row(row: Dict, row_num: int) -> List[str]:
    """Validate a single row against the schema. Returns list of issues."""
    issues = []
    if not isinstance(row, dict):
        return [f"Row {row_num}: expected object"]
    for alias in FIELD_ALIASES:
        if alias in row:
            issues.append(f"Row {row_num}: legacy alias {alias}; use explicit migrate command")

    # Check required fields
    for field, expected_type in CANONICAL_SCHEMA.items():
        if field not in row:
            issues.append(f"Row {row_num}: Missing field '{field}'")

    # Check type mismatches
    for field, expected_type in CANONICAL_SCHEMA.items():
        if field in row:
            value = row[field]
            if value is None:
                if field in NULLABLE_WHEN_UNAVAILABLE and row.get("status") == "unavailable":
                    continue
                issues.append(f"Row {row_num}: null required field {field}")
                continue
            valid_type = (
                (not isinstance(value, bool) and isinstance(value, (int, float)))
                if expected_type is float
                else (
                    type(value) is int if expected_type is int else isinstance(value, expected_type)
                )
            )
            if not valid_type:
                issues.append(
                    f"Row {row_num}: Field '{field}' has type {type(value).__name__}, "
                    f"expected {expected_type.__name__}"
                )

    for field in ("paper_claim_ids", "source_files"):
        value = row.get(field)
        if isinstance(value, list) and any(not isinstance(x, str) or not x.strip() for x in value):
            issues.append(f"Row {row_num}: invalid {field} member")
    for field in ("peak_reward", "last10_avg", "learning_rate"):
        value = row.get(field)
        if isinstance(value, bool) or isinstance(value, (int, float)) and not math.isfinite(value):
            issues.append(f"Row {row_num}: nonfinite/boolean {field}")
    for field, minimum in (("seed", 0), ("steps_completed", 0), ("group_size", 1)):
        if type(row.get(field)) is int and row[field] < minimum:
            issues.append(f"Row {row_num}: invalid range {field}")
    if isinstance(row.get("learning_rate"), (int, float)) and row["learning_rate"] <= 0:
        issues.append(f"Row {row_num}: learning_rate must be positive")
    if row.get("source_files") == []:
        issues.append(f"Row {row_num}: source_files must identify evidence")
    if row.get("status") == "unavailable" and (
        row.get("steps_completed") != 0
        or any(row.get(k) is not None for k in NULLABLE_WHEN_UNAVAILABLE)
    ):
        issues.append(f"Row {row_num}: unavailable requires zero steps and null trace/summaries")
    trace = row.get("reward_trace")
    if isinstance(trace, list):
        if not trace or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in trace
        ):
            issues.append(f"Row {row_num}: invalid reward_trace")
        else:
            if row.get("steps_completed") != len(trace):
                issues.append(f"Row {row_num}: trace/step count mismatch")
            for key, actual in (
                ("peak_reward", max(trace)),
                ("last10_avg", sum(trace[-10:]) / len(trace[-10:])),
            ):
                value = row.get(key)
                if isinstance(value, (int, float)) and not math.isclose(
                    value, actual, rel_tol=1e-6, abs_tol=1e-9
                ):
                    issues.append(f"Row {row_num}: {key} inconsistent with trace")
    for field in ("run_id", "model", "method", "task", "split", "status"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            issues.append(f"Row {row_num}: empty identity {field}")
    return issues


def analyze_schema(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the schema of all rows."""
    field_counts = {}
    type_variations = {}

    for _i, row in enumerate(data):
        normalized = row

        for key, value in normalized.items():
            if key not in field_counts:
                field_counts[key] = 0
                type_variations[key] = set()

            field_counts[key] += 1
            type_variations[key].add(type(value).__name__)

    return {
        "field_counts": field_counts,
        "type_variations": type_variations,
        "n_rows": len(data),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Strict validation; migration is an explicit separate operation"
    )
    parser.add_argument("command", choices=("validate", "migrate"), nargs="?", default="validate")
    parser.add_argument("--input", type=Path, default=MASTER_RESULTS)
    parser.add_argument(
        "--output", type=Path, help="Required new migration destination; never overwrites input"
    )
    args = parser.parse_args(argv)
    try:
        data = load_json(args.input)
        rows = data.get("experiments", [data]) if isinstance(data, dict) else data
        if not isinstance(rows, list) or not rows:
            raise ValueError("expected nonempty list of rows")
        if args.command == "migrate":
            if (
                not args.output
                or args.output.resolve() == args.input.resolve()
                or args.output.exists()
            ):
                raise ValueError("migration requires a new distinct --output file")
            rows = [normalize_row(row) for row in rows]
        issues = [issue for i, row in enumerate(rows, 1) for issue in validate_row(row, i)]
        ids = [r.get("run_id") for r in rows if isinstance(r, dict)]
        if any(not isinstance(i, str) for i in ids) or len(
            set(i for i in ids if isinstance(i, str))
        ) != len(ids):
            issues.append("invalid or duplicate run identities")
        if issues:
            print("\n".join(issues))
            return 1
        if args.command == "migrate":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as fh:
                json.dump(rows, fh, indent=2, allow_nan=False)
                fh.write("\n")
        print(f"Validated {len(rows)} canonical rows; review27-v3")
        return 0
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
