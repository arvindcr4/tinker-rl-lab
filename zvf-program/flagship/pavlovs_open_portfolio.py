#!/usr/bin/env python3
"""Build and fail-closed validate the open-source Pavlov evaluation portfolio."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from flagship.pavlovs_domain_contract import (
    CONTRACT_PATH,
    coverage_report,
    load_contract,
    validate_contract,
)


OVERLAY_PATH = Path(__file__).with_name("pavlovs_open_portfolio_overrides.json")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_overlay(path: Path = OVERLAY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        overlay = json.load(handle)
    if not isinstance(overlay, dict):
        raise ValueError("open portfolio overlay root must be a JSON object")
    return overlay


def build_open_contract(base_contract: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(base_contract)
    registry = contract["suite_registry"]
    for old_suite_id, replacement in overlay.get("replacements", {}).items():
        registry.pop(old_suite_id, None)
        new_suite_id = replacement["new_suite_id"]
        registry[new_suite_id] = {
            key: value
            for key, value in replacement.items()
            if key
            not in {
                "new_suite_id",
                "runner_status",
                "fully_public_evaluation",
                "requires_private_assets",
                "source_revision",
                "code_license",
                "dataset_license",
            }
        }
    contract["schema_version"] = "pavlovs-domain-contract-open-v1"
    contract["status"] = "open-source-portfolio-defined-not-executed"
    contract["open_portfolio"] = {
        "overlay_schema_version": overlay.get("schema_version"),
        "claim_boundary": overlay.get("claim_boundary"),
        "replacement_count": len(overlay.get("replacements", {})),
    }
    return contract


def validate_open_portfolio(base_contract: dict[str, Any], overlay: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    registry = base_contract.get("suite_registry", {})
    replacements = overlay.get("replacements", {})
    retained = overlay.get("retained_primary_suites", {})
    allowed_licenses = set(overlay.get("license_allowlist", []))

    if overlay.get("schema_version") != "pavlovs-open-portfolio-overrides-v1":
        errors.append("unsupported open portfolio overlay schema")
    if not overlay.get("claim_boundary"):
        errors.append("open portfolio claim boundary must be explicit")
    if not isinstance(replacements, dict) or not replacements:
        errors.append("at least one replacement is required")

    new_ids: list[str] = []
    for old_suite_id, replacement in replacements.items():
        old = registry.get(old_suite_id)
        if old is None:
            errors.append(f"{old_suite_id}: replacement source suite is missing")
        elif old.get("role") != "primary_eval":
            errors.append(f"{old_suite_id}: only primary suites may be replaced")
        new_suite_id = replacement.get("new_suite_id")
        if not isinstance(new_suite_id, str) or not new_suite_id:
            errors.append(f"{old_suite_id}: new_suite_id is required")
        else:
            new_ids.append(new_suite_id)
        if replacement.get("role") != "primary_eval":
            errors.append(f"{old_suite_id}: replacement must remain primary_eval")
        if replacement.get("fully_public_evaluation") is not True:
            errors.append(f"{old_suite_id}: evaluation is not fully public")
        if replacement.get("requires_private_assets") is not False:
            errors.append(f"{old_suite_id}: private assets are not allowed")
        for field in ("code_license", "dataset_license"):
            if replacement.get(field) not in allowed_licenses:
                errors.append(f"{old_suite_id}: {field} is not in the explicit allowlist")
        if not SHA40.fullmatch(str(replacement.get("source_revision", ""))):
            errors.append(f"{old_suite_id}: source_revision must be a 40-hex pin")
        if replacement.get("runner_status") not in {
            "NOT_YET_INTEGRATED",
            "CANARY_VERIFIED",
            "FULLY_RUN",
        }:
            errors.append(f"{old_suite_id}: unknown runner_status")

    if len(new_ids) != len(set(new_ids)):
        errors.append("replacement suite IDs must be unique")

    for suite_id, metadata in retained.items():
        suite = registry.get(suite_id)
        if suite is None or suite.get("role") != "primary_eval":
            errors.append(f"{suite_id}: retained primary suite is missing")
        if metadata.get("fully_public_evaluation") is not True:
            errors.append(f"{suite_id}: retained evaluation is not fully public")
        if metadata.get("requires_private_assets") is not False:
            errors.append(f"{suite_id}: retained suite requires private assets")
        for field in ("code_license", "dataset_license"):
            if metadata.get(field) not in allowed_licenses:
                errors.append(f"{suite_id}: retained {field} is not allowed")
        if not SHA40.fullmatch(str(metadata.get("source_revision", ""))):
            errors.append(f"{suite_id}: retained source_revision must be pinned")

    original_primary_ids = {
        suite_id for suite_id, suite in registry.items() if suite.get("role") == "primary_eval"
    }
    accounted = set(replacements) | set(retained)
    unaccounted = sorted(original_primary_ids - accounted)
    if unaccounted:
        errors.append(f"primary suites lack open disposition: {unaccounted}")

    effective = build_open_contract(base_contract, overlay)
    errors.extend(f"effective contract: {error}" for error in validate_contract(effective))
    if coverage_report(effective)["primary_eval_suites"] != 14:
        errors.append("effective open portfolio must contain exactly 14 primary suites")
    return errors


def report(base_contract: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    effective = build_open_contract(base_contract, overlay)
    errors = validate_open_portfolio(base_contract, overlay)
    replacements = overlay["replacements"]
    return {
        "schema_version": overlay["schema_version"],
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "replacement_count": len(replacements),
        "retained_primary_count": len(overlay["retained_primary_suites"]),
        "primary_eval_suites": sorted(
            suite_id
            for suite_id, suite in effective["suite_registry"].items()
            if suite.get("role") == "primary_eval"
        ),
        "runner_status_counts": {
            status: sum(entry["runner_status"] == status for entry in replacements.values())
            for status in ("NOT_YET_INTEGRATED", "CANARY_VERIFIED", "FULLY_RUN")
        },
        "score": None,
        "claim_boundary": overlay["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--overlay", type=Path, default=OVERLAY_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = report(load_contract(args.contract), load_overlay(args.overlay))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["status"])
        for error in result["errors"]:
            print(f"ERROR: {error}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
