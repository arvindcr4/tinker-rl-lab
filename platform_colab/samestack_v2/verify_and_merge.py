"""Validate staged bundles without retroactively inventing legacy provenance.

A successful merge requires strict JSON, semantic filename/run identity, the
versioned shared schema, complete artifact hashes and no duplicate identities.
Legacy or incomplete artifacts are reported unverified and are never rewritten.
Empty staging is not success. The corrected protocol has its own namespace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from platform_hybrid.experiments.samestack_contract import (
    completion_path, strict_json_load, validate_completion, validate_run,
)
from platform_hybrid.experiments.samestack_protocol import PROTOCOL_HASH, PROTOCOL_ID

STAGE = ROOT / "platform_hybrid/experiments/results" / (PROTOCOL_ID + "_colab")
CANON = ROOT / "platform_hybrid/experiments/results" / PROTOCOL_ID
OUT = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/colab_runs_check.json"


def load(path):
    return strict_json_load(path)


def run_paths(directory):
    """Recognize flat historical records and promoted per-run bundles, not partials."""
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*_s*.json")
                  if not path.name.endswith(".complete.json")
                  and not any(part.startswith(".") for part in path.relative_to(directory).parts)
                  and "adapters" not in path.relative_to(directory).parts)


def verify_stage(stage, canonical=None):
    stage = Path(stage)
    paths = run_paths(stage)
    report = {"protocol_id": PROTOCOL_ID, "runs": {}, "staged_count": len(paths),
              "all_verified": False, "errors": []}
    seen, run_ids = set(), set()
    canonical_identities = {}
    if canonical is not None:
        for existing in run_paths(canonical):
            try:
                existing_record = validate_completion(existing)
                manifest = existing_record["manifest"]
                identity = (manifest["environment"], manifest["protocol_hash"],
                            manifest["ordered_test_item_hash"], existing_record["arm"], existing_record["seed"])
                if identity in canonical_identities:
                    report["errors"].append("canonical directory already has duplicate run identities")
                canonical_identities[identity] = existing
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report["errors"].append(f"canonical artifact is unverified: {existing.name}: {exc}")
    for path in paths:
        label = str(path.relative_to(stage))
        try:
            record = load(path)
            text_path = path.with_name(path.stem + "_texts.json.gz")
            texts = load(text_path) if text_path.exists() else None
            checks = validate_run(record, filename=path.name, texts=texts, seen=seen,
                                  expected={"environment": "colab", "protocol_id": PROTOCOL_ID,
                                            "protocol_hash": PROTOCOL_HASH})
            if record.get("smoke") is not False:
                checks["errors"].append("smoke runs are ineligible for the full-run staging cohort")
            manifest = record.get("manifest", {})
            run_id = manifest.get("run_id") if isinstance(manifest, dict) else None
            if run_id and run_id in run_ids:
                checks["errors"].append("duplicate run_id")
            if run_id:
                run_ids.add(run_id)
            if not completion_path(path).exists():
                checks["unverified"].append("artifact completion marker unavailable")
            else:
                try:
                    validate_completion(path, expected={"environment": "colab", "protocol_id": PROTOCOL_ID,
                                                        "protocol_hash": PROTOCOL_HASH})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    checks["errors"].append(f"artifact completion failed: {exc}")
            if canonical is not None:
                destination = Path(canonical) / path.stem / path.name
                duplicate = canonical_identities.get(checks["identity"])
                if duplicate is not None and duplicate != destination:
                    checks["errors"].append("duplicate run identity already exists elsewhere in canonical directory")
                if destination.parent.exists():
                    try:
                        validate_completion(destination)
                        if load(completion_path(destination)) != load(completion_path(path)):
                            checks["errors"].append("canonical run identity already exists with different artifacts")
                    except (OSError, ValueError, KeyError, TypeError):
                        checks["errors"].append("canonical destination already contains unverified artifacts")
            checks["valid"] = not checks["errors"]
            checks["verified"] = checks["valid"] and not checks["unverified"]
            checks["status"] = "verified" if checks["verified"] else "invalid" if not checks["valid"] else "unverified"
            report["runs"][label] = checks
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["runs"][label] = {"valid": False, "verified": False, "status": "invalid",
                                     "errors": [str(exc)], "unverified": [], "metrics": {}}
    if not paths:
        report["errors"].append("no staged runs; empty staging is not verified success")
    report["all_verified"] = bool(paths) and not report["errors"] and all(row["verified"] for row in report["runs"].values())
    return report


def merge_verified(stage, canonical):
    """Copy only independently verified inventories; preserve original staged files."""
    stage, canonical = Path(stage), Path(canonical)
    report = verify_stage(stage, canonical)
    if not report["all_verified"]:
        raise ValueError("not merging: staged runs are empty, invalid, or unverified")
    canonical.mkdir(parents=True, exist_ok=True)
    for path in run_paths(stage):
        marker = load(completion_path(path))
        destination = canonical / path.stem
        if destination.exists():
            continue  # verify_stage already established byte-identical inventory.
        temporary = Path(tempfile.mkdtemp(prefix=f".{path.stem}.merge-", dir=canonical))
        for relative in marker["files"]:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path.parent / relative, target)
        shutil.copy2(completion_path(path), temporary / completion_path(path).name)
        validate_completion(temporary / path.name)
        os.replace(temporary, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--canonical", type=Path, default=CANON)
    parser.add_argument("--report", type=Path, default=OUT)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    report = verify_stage(args.stage, args.canonical)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, row in report["runs"].items():
        print(f"{row['status'].upper()} {name}: " + "; ".join(row["errors"] + row["unverified"]))
    if args.merge:
        if not report["all_verified"]:
            parser.exit(1, "not merging: staged runs are empty, invalid, or unverified\n")
        merge_verified(args.stage, args.canonical)
        print(f"merged {report['staged_count']} verified bundles")
    return 0 if report["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
