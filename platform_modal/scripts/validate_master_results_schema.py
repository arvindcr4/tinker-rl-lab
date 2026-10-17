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
from typing import Dict, List, Tuple, Any
import pprint

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
    with open(path, 'r') as f:
        return json.load(f)


def normalize_row(row: Dict) -> Dict:
    """Normalize a row using aliases."""
    normalized = {}
    for key, value in row.items():
        new_key = FIELD_ALIASES.get(key, key)
        if new_key == "source_files" and key != "source_files" and isinstance(value,str):
            value=[value]
        if new_key in normalized and normalized[new_key] != value:
            raise ValueError(f"conflicting aliases for {new_key}")
        normalized[new_key] = value
    return normalized


def validate_row(row: Dict, row_num: int) -> List[str]:
    """Validate a single row against the schema. Returns list of issues."""
    issues = []
    
    # Check required fields
    for field, expected_type in CANONICAL_SCHEMA.items():
        if field not in row:
            issues.append(f"Row {row_num}: Missing field '{field}'")
    
    # Check type mismatches
    for field, expected_type in CANONICAL_SCHEMA.items():
        if field in row:
            value = row[field]
            if value is None:
                issues.append(f"Row {row_num}: null required field {field}")
                continue
            if not isinstance(value, expected_type):
                issues.append(
                    f"Row {row_num}: Field '{field}' has type {type(value).__name__}, "
                    f"expected {expected_type.__name__}"
                )
    
    for field in ("paper_claim_ids","source_files"):
        value=row.get(field)
        if isinstance(value,list) and any(not isinstance(x,str) or not x.strip() for x in value):
            issues.append(f"Row {row_num}: invalid {field} member")
    for field in ("peak_reward","last10_avg","learning_rate"):
        value=row.get(field)
        if isinstance(value,bool) or isinstance(value,(int,float)) and not math.isfinite(value):
            issues.append(f"Row {row_num}: nonfinite/boolean {field}")
    trace=row.get("reward_trace")
    if isinstance(trace,list):
        if not trace or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in trace):
            issues.append(f"Row {row_num}: invalid reward_trace")
        else:
            if row.get("steps_completed") != len(trace):
                issues.append(f"Row {row_num}: trace/step count mismatch")
            for key,actual in (("peak_reward",max(trace)),("last10_avg",sum(trace[-10:])/len(trace[-10:]))):
                value=row.get(key)
                if isinstance(value,(int,float)) and not math.isclose(value,actual,rel_tol=1e-6,abs_tol=1e-9):
                    issues.append(f"Row {row_num}: {key} inconsistent with trace")
    for field in ("run_id","model","method","task","split","status"):
        if not isinstance(row.get(field),str) or not row[field].strip():
            issues.append(f"Row {row_num}: empty identity {field}")
    return issues


def analyze_schema(data: List[Dict]) -> Dict[str, Any]:
    """Analyze the schema of all rows."""
    field_counts = {}
    type_variations = {}
    
    for i, row in enumerate(data):
        normalized = normalize_row(row)
        
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


def main():
    if not MASTER_RESULTS.exists():
        print(f"ERROR: {MASTER_RESULTS} not found")
        return 1
    
    print(f"Loading {MASTER_RESULTS}...")
    
    try:
        data = load_json(MASTER_RESULTS)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return 1
    
    # Handle different JSON structures
    if isinstance(data, dict):
        if "experiments" in data:
            rows = data["experiments"]
        else:
            rows = [data]
    else:
        rows = data
    
    print(f"Found {len(rows)} rows")
    
    # Analyze schema
    try:
        normalized_rows=[normalize_row(row) for row in rows]
        ids=[row.get("run_id") for row in normalized_rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError("empty dataset or duplicate run identities")
        analysis = analyze_schema(rows)
    except (ValueError,TypeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    
    print("\n" + "="*60)
    print("SCHEMA ANALYSIS")
    print("="*60)
    print(f"\nTotal rows: {analysis['n_rows']}")
    
    print("\nField usage:")
    for field, count in sorted(analysis['field_counts'].items(), 
                                 key=lambda x: -x[1]):
        types = analysis['type_variations'][field]
        print(f"  {field}: {count} rows, types: {', '.join(types)}")
    
    # Validate each row
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    all_issues = []
    for i, row in enumerate(rows):
        normalized = normalize_row(row)
        issues = validate_row(normalized, i + 1)
        all_issues.extend(issues)
    
    if all_issues:
        print(f"\nFound {len(all_issues)} issues:")
        for issue in all_issues[:20]:  # Limit output
            print(f"  - {issue}")
        if len(all_issues) > 20:
            print(f"  ... and {len(all_issues) - 20} more")
    else:
        print("\nAll rows conform to canonical schema!")
    
    # Check for common issues
    print("\n" + "="*60)
    print("COMMON ISSUE CHECKS")
    print("="*60)
    
    # Check for name vs experiment_id inconsistency
    names = set()
    experiment_ids = set()
    for row in rows:
        if "name" in row:
            names.add(row["name"])
        if "experiment_id" in row:
            experiment_ids.add(row["experiment_id"])
    
    if names and experiment_ids:
        overlap = names & experiment_ids
        if not overlap:
            print("\nWARNING: 'name' and 'experiment_id' fields have no overlap")
            print("  This suggests inconsistent identification across rows")
    
    # Check for PPO/GRPO rows
    methods = {}
    for row in rows:
        method = row.get("method", row.get("algorithm", "unknown"))
        methods[method] = methods.get(method, 0) + 1
    
    print("\nMethod distribution:")
    for method, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")
    
    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())
