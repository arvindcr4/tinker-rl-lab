#!/usr/bin/env python3
"""Resolve the Pavlov suites that have runnable, receipt-backed inputs.

The portfolio split registry proves the immutable split metadata.  This module
adds the second, local gate: every suite selected for execution must have a
runtime receipt bundle whose references and hashes agree with the registry.
Only receipt references, revisions, task digests, and other metadata leave this
module.  Prompts, targets, datasets, network clients, and job launchers are not
part of the resolver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Package import.
    from .pavlov_portfolio_split_registry import (
        EXPECTED_PRIMARY_EVAL_SUITE_IDS,
        EXPECTED_TRAIN_SUITE_IDS,
        PINNED_REVISION,
        PRIMARY_EVAL_ROLE,
        PortfolioSplitRegistryError,
        ROLES,
        TASK_DIGEST,
        TRAIN_ROLE,
        verify_registry,
    )
except ImportError:  # Direct execution from the flagship directory.
    from pavlov_portfolio_split_registry import (  # type: ignore[no-redef]
        EXPECTED_PRIMARY_EVAL_SUITE_IDS,
        EXPECTED_TRAIN_SUITE_IDS,
        PINNED_REVISION,
        PRIMARY_EVAL_ROLE,
        PortfolioSplitRegistryError,
        ROLES,
        TASK_DIGEST,
        TRAIN_ROLE,
        verify_registry,
    )


SCHEMA_VERSION = "pavlov-runnable-suite-resolver-v1"
XLAM_SUITE_ID = "pavlov_xlam"
HELD_OUT_ROLE = "held_out"
PRIVATE_ROLE = "private"
EXPECTED_SUITE_IDS = {
    TRAIN_ROLE: tuple(EXPECTED_TRAIN_SUITE_IDS),
    PRIMARY_EVAL_ROLE: tuple(EXPECTED_PRIMARY_EVAL_SUITE_IDS),
}
_RAW_CONTENT_KEYS = {"prompt", "prompts", "target", "targets"}
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "nil",
    "pending",
    "placeholder",
    "todo",
    "tbd",
    "unset",
    "unrecorded",
    "unknown",
    "latest",
    "master",
    "head",
    "tip",
    "descriptive",
    "not provided",
    "not_provided",
    "not applicable",
    "n/a",
}
_MUTABLE_MARKERS = {
    "latest",
    "main",
    "master",
    "head",
    "tip",
    "pending",
    "current",
    "branch",
}
_REQUIRED_RUNTIME_RECEIPTS = (
    "revision",
    "license",
    "container",
    "decontamination",
    "verifier",
    "split_manifest",
    "runtime",
)


class RunnableSuiteResolverError(ValueError):
    """Raised when the registry or runtime receipt bundle is malformed."""


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnableSuiteResolverError(f"cannot read JSON {path!s}: {exc}") from exc


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_metadata_only(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_CONTENT_KEYS:
                raise RunnableSuiteResolverError(
                    f"{path} contains raw {str(key).lower()} content; resolver is metadata-only"
                )
            _assert_metadata_only(child, f"{path}.{key}")
    elif _is_sequence(value):
        for index, child in enumerate(value):
            _assert_metadata_only(child, f"{path}[{index}]")


def _looks_placeholder(value: Any) -> bool:
    if value is None or not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if normalized in _PLACEHOLDERS:
        return True
    return any(token in normalized for token in ("placeholder", "unrecorded", "not provided"))


def _receipt_is_proof(value: Any) -> bool:
    """Reject prose and mutable aliases while accepting local receipt URIs."""

    if _looks_placeholder(value):
        return False
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    lowered = normalized.lower()
    if any(marker in {part for part in re.split(r"[/:?#._-]+", lowered) if part} for marker in _MUTABLE_MARKERS):
        return False
    if any(character.isspace() for character in normalized):
        return False
    if lowered.startswith("sha256:"):
        return bool(_DIGEST.fullmatch(normalized[7:]))
    if lowered.startswith(("receipt://", "receipt:", "urn:receipt:", "file://")):
        return len(normalized.split(":", 1)[1].strip("/")) >= 2
    if lowered.startswith(("http://", "https://")):
        # A web URL is acceptable only when it binds to an immutable digest.
        return bool(re.search(r"(?:sha256[:/_-])?[0-9a-f]{40}(?:[0-9a-f]{24})?", lowered))
    # Opaque IDs are accepted only when they identify themselves as receipts.
    return "receipt" in lowered and len(normalized) >= 8


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _receipt_source(record: Mapping[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {}
    for field in ("receipt_refs", "provenance_receipts", "receipts", "runtime_receipts"):
        value = record.get(field)
        if isinstance(value, Mapping):
            source.update(value)
    source.update({key: value for key, value in record.items() if key.endswith("_receipt_ref")})
    return source


def _extract_receipts(record: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    source = _receipt_source(record)
    aliases = {
        "revision": ("revision_receipt_ref", "revision_ref", "revision"),
        "license": ("license_receipt_ref", "license_ref", "license"),
        "container": (
            "container_receipt_ref",
            "container_runtime_receipt_ref",
            "container_ref",
            "container",
        ),
        "decontamination": (
            "decontamination_receipt_ref",
            "decontamination_ref",
            "decontamination",
        ),
        "verifier": (
            "verifier_receipt_ref",
            "verifier_ref",
            "verification_receipt_ref",
            "verifier",
        ),
        "split_manifest": (
            "split_manifest_receipt_ref",
            "split_manifest_ref",
            "task_hash_receipt_ref",
            "task_id_hash_receipt_ref",
            "split_manifest",
        ),
        "runtime": (
            "runtime_receipt_ref",
            "runtime_ref",
            "execution_receipt_ref",
            "execution_ref",
            "runtime_receipt",
            "runtime",
        ),
        "held_out": (
            "held_out",
            "held_out_receipt_ref",
            "heldout_receipt_ref",
            "private_receipt_ref",
            "held_out_receipt",
            "heldout_receipt",
        ),
    }
    receipts: dict[str, str] = {}
    errors: list[str] = []
    for name, keys in aliases.items():
        value = _first(source, keys)
        if value is None:
            record_keys = keys if name != "held_out" else tuple(key for key in keys if key != "held_out")
            value = _first(record, record_keys)
        if value is None:
            continue
        if not _receipt_is_proof(value):
            errors.append(f"{name} receipt is missing, mutable, placeholder, or descriptive")
            continue
        receipts[name] = value.strip()
    return receipts, errors


def _revision(record: Mapping[str, Any]) -> str | None:
    values = [
        record[key]
        for key in ("revision", "dataset_revision", "immutable_revision")
        if key in record
    ]
    if not values:
        return None
    if any(not isinstance(value, str) for value in values) or len(set(values)) != 1:
        return None
    value = values[0]
    return value if PINNED_REVISION.fullmatch(value) else None


def _role(record: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    for key in ("role", "suite_role", "split_role", "portfolio_split_role"):
        value = record.get(key)
        if isinstance(value, str):
            values.append(value)
    if isinstance(record.get("split_roles"), Mapping):
        values.extend(str(value) for value in record["split_roles"].values() if value is not None)
    if not values:
        return None
    unique = set(values)
    if len(unique) == 1:
        return values[0]
    if unique <= {PRIMARY_EVAL_ROLE, HELD_OUT_ROLE, PRIVATE_ROLE}:
        return HELD_OUT_ROLE
    return None


def _task_hashes(record: Mapping[str, Any], role: str) -> list[str] | None:
    value = record.get("task_hashes")
    hashes: Any = None
    if isinstance(value, Mapping):
        for key in (role, "test" if role == PRIMARY_EVAL_ROLE else "train"):
            if key in value:
                hashes = value[key]
                break
    elif _is_sequence(value):
        hashes = value
    if hashes is None:
        hashes = _first(
            record,
            ("primary_eval_task_hashes", "test_task_hashes")
            if role == PRIMARY_EVAL_ROLE
            else ("train_task_hashes",),
        )
    if not _is_sequence(hashes) or not hashes:
        return None
    if any(not isinstance(value, str) or not TASK_DIGEST.fullmatch(value) for value in hashes):
        return None
    if len(set(hashes)) != len(hashes):
        return None
    return list(hashes)


def _aggregate(record: Mapping[str, Any], role: str, hashes: Sequence[str]) -> str:
    expected = _sha256("\n".join(hashes))
    declared: Any = None
    values = record.get("aggregate_hashes")
    if isinstance(values, Mapping):
        declared = values.get(role)
        if declared is None and role == PRIMARY_EVAL_ROLE:
            declared = values.get("test")
    elif isinstance(values, str):
        declared = values
    if declared is None:
        declared = _first(
            record,
            ("aggregate_sha256", f"{role}_aggregate_sha256", "task_hash_aggregate"),
        )
    if declared is not None and declared != expected:
        return ""
    return expected


def _registry_digest(registry: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in registry.items() if key != "registry_sha256"}
    return _sha256(_stable_json(payload))


def _load_mapping(value: Mapping[str, Any] | str | Path, label: str) -> Mapping[str, Any]:
    loaded = _read_json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(loaded, Mapping):
        raise RunnableSuiteResolverError(f"{label} must be a JSON object")
    _assert_metadata_only(loaded, label)
    return loaded


def _suite_records(registry: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    raw = registry.get("suites")
    if not _is_sequence(raw):
        return {}, ["portfolio registry is missing suites"]
    records: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    expected_by_id = {
        suite_id: role for role, ids in EXPECTED_SUITE_IDS.items() for suite_id in ids
    }
    for item in raw:
        if not isinstance(item, Mapping):
            errors.append("portfolio registry contains a non-object suite")
            continue
        suite_id = item.get("suite_id", item.get("id"))
        if not isinstance(suite_id, str) or not suite_id:
            errors.append("portfolio registry suite is missing suite_id")
            continue
        if suite_id in records:
            errors.append(f"duplicate suite ID in portfolio registry: {suite_id}")
            continue
        if suite_id not in expected_by_id:
            errors.append(f"extra suite ID in portfolio registry: {suite_id}")
            continue
        role = item.get("role", item.get("suite_role", item.get("split_role")))
        if role != expected_by_id[suite_id]:
            errors.append(f"{suite_id} role does not match contract role {expected_by_id[suite_id]!r}")
            continue
        records[suite_id] = item
    missing = sorted(set(expected_by_id) - set(records))
    if missing:
        errors.append("missing suite IDs in portfolio registry: " + ", ".join(missing))
    return records, errors


def _runtime_records(bundle: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None, list[str]]:
    if isinstance(bundle, (str, Path)):
        bundle = _read_json(bundle)
    if not isinstance(bundle, Mapping) and not _is_sequence(bundle):
        return [], None, ["runtime receipt bundle must be a JSON object or list"]
    _assert_metadata_only(bundle, "runtime_receipt_bundle")
    xlam: Mapping[str, Any] | None = None
    if isinstance(bundle, Mapping):
        for key in ("xlam_component", "xlam_component_preflight", "xlam"):
            if isinstance(bundle.get(key), Mapping):
                xlam = bundle[key]
                break
        value: Any = _first(bundle, ("runtime_receipts", "receipts", "suites", "suite_receipts"))
        if value is None:
            # An ID-keyed mapping is also a useful local fixture format.
            value = [dict(item, suite_id=str(key)) for key, item in bundle.items() if isinstance(item, Mapping)]
        elif isinstance(value, Mapping):
            value = [dict(item, suite_id=str(key)) for key, item in value.items() if isinstance(item, Mapping)]
    else:
        value = bundle
    if not _is_sequence(value):
        return [], xlam, ["runtime receipt bundle is missing runtime_receipts"]
    records = [item for item in value if isinstance(item, Mapping)]
    errors = ["runtime receipt bundle contains a non-object receipt"] if any(
        not isinstance(item, Mapping) for item in value
    ) else []
    return records, xlam, errors


def _held_out_state(registry_record: Mapping[str, Any]) -> tuple[bool, str | None]:
    proven = bool(registry_record.get("held_out_receipt_proven"))
    refs = registry_record.get("receipt_refs")
    held_ref = refs.get("held_out") if isinstance(refs, Mapping) else None
    if proven and isinstance(held_ref, str) and _receipt_is_proof(held_ref):
        return True, held_ref
    return False, None


def _resolve_one(
    suite_id: str,
    registry_record: Mapping[str, Any],
    runtime_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    role = str(registry_record.get("role"))
    held_out_proven, held_out_ref = _held_out_state(registry_record)
    blockers: list[str] = []
    if runtime_record is None:
        blockers.append("missing runtime receipt")
        return {
            "suite_id": suite_id,
            "role": role,
            "status": "BLOCKED",
            "runnable": False,
            "held_out_private": False,
            "blockers": blockers,
        }
    runtime_id = runtime_record.get("suite_id", runtime_record.get("id"))
    if runtime_id != suite_id:
        blockers.append("runtime receipt suite_id mismatch")
    runtime_role = _role(runtime_record)
    declared_kind = runtime_role
    if runtime_role in (HELD_OUT_ROLE, PRIVATE_ROLE):
        if role != PRIMARY_EVAL_ROLE:
            blockers.append("heldout/private runtime role is not allowed for a training suite")
        if not held_out_proven:
            blockers.append("heldout/private runtime claim lacks registry-proven receipt")
        runtime_held_ref = _extract_receipts(runtime_record)[0].get("held_out")
        if runtime_held_ref != held_out_ref:
            blockers.append("heldout/private receipt does not match registry proof")
        runtime_role = PRIMARY_EVAL_ROLE
    elif runtime_role != role:
        blockers.append("runtime receipt role mismatch")
    revision = _revision(runtime_record)
    registry_revision = registry_record.get("revision")
    if revision is None:
        blockers.append("runtime receipt revision is missing or mutable")
    elif revision != registry_revision:
        blockers.append("runtime receipt revision mismatch")
    receipts, receipt_errors = _extract_receipts(runtime_record)
    blockers.extend(receipt_errors)
    registry_refs = registry_record.get("receipt_refs")
    if not isinstance(registry_refs, Mapping):
        blockers.append("registry suite receipt_refs are missing")
        registry_refs = {}
    for key in _REQUIRED_RUNTIME_RECEIPTS:
        if key not in receipts:
            blockers.append(f"missing {key} runtime receipt")
    for key in ("revision", "license", "container", "decontamination", "verifier", "split_manifest"):
        expected = registry_refs.get(key)
        if expected is not None and receipts.get(key) != expected:
            blockers.append(f"{key} receipt mismatch")
    registry_hashes = _task_hashes(registry_record, role)
    runtime_hashes = _task_hashes(runtime_record, role)
    if registry_hashes is None:
        blockers.append("registry task hashes are missing")
    if runtime_hashes is not None and registry_hashes != runtime_hashes:
        blockers.append("runtime task hashes mismatch registry order")
    runtime_aggregate = _aggregate(runtime_record, role, runtime_hashes or []) if runtime_hashes else ""
    registry_aggregate = _aggregate(registry_record, role, registry_hashes or []) if registry_hashes else ""
    if not registry_aggregate:
        blockers.append("registry aggregate hash is invalid")
    if runtime_hashes is None and runtime_record.get("aggregate_sha256") is None:
        blockers.append("runtime task hash or aggregate receipt is missing")
    declared_runtime_aggregate = runtime_record.get("aggregate_sha256")
    if declared_runtime_aggregate is None and isinstance(runtime_record.get("aggregate_hashes"), Mapping):
        declared_runtime_aggregate = runtime_record["aggregate_hashes"].get(role)
    if declared_runtime_aggregate is not None and declared_runtime_aggregate != registry_aggregate:
        blockers.append("runtime aggregate hash mismatch")
    if runtime_hashes is not None and runtime_aggregate != registry_aggregate:
        blockers.append("runtime aggregate does not match registry")
    if runtime_record.get("registry_sha256") is not None and runtime_record["registry_sha256"] != registry_record.get("registry_sha256"):
        blockers.append("runtime registry digest mismatch")
    runtime_status = runtime_record.get("status", runtime_record.get("state"))
    if runtime_status is not None and str(runtime_status).lower() not in {"ready", "verified", "complete", "completed", "runnable"}:
        blockers.append("runtime receipt is not verified-ready")
    explicit_held = bool(
        runtime_record.get("held_out")
        or runtime_record.get("is_held_out")
        or runtime_record.get("private")
        or runtime_role in (HELD_OUT_ROLE, PRIVATE_ROLE)
        or "held_out" in receipts
    )
    if explicit_held and not held_out_proven:
        blockers.append("heldout/private classification is not receipt-proven")
    held_private = explicit_held and held_out_proven and not blockers
    return {
        "suite_id": suite_id,
        "role": role,
        "status": "READY" if not blockers else "BLOCKED",
        "runnable": not blockers,
        "held_out_private": held_private,
        "held_out_receipt_proven": held_out_proven,
        "revision": registry_revision,
        "runtime_receipt_ref": receipts.get("runtime"),
        "aggregate_sha256": registry_aggregate,
        "task_hash_count": len(registry_hashes or []),
        "blockers": sorted(set(blockers)),
    }


def _registry_global_blockers(registry: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    gate = registry.get("contract_gate")
    if isinstance(gate, Mapping) and _is_sequence(gate.get("blockers")):
        blockers.extend(str(item) for item in gate["blockers"])
    status = str(registry.get("status", "")).upper()
    raw_blockers = registry.get("blockers")
    if _is_sequence(raw_blockers):
        for item in raw_blockers:
            text = str(item)
            if "xLAM" not in text and "xlam" not in text:
                blockers.append("portfolio registry blocker: " + text)
    if status not in {"READY", "BLOCKED"}:
        blockers.append("portfolio registry status is not recognized")
    return sorted(set(blockers))


def _xlam_resolution(component: Mapping[str, Any] | None) -> dict[str, Any]:
    if component is None:
        return {
            "suite_id": XLAM_SUITE_ID,
            "component": "xLAM",
            "portfolio_suite": False,
            "status": "BLOCKED",
            "runnable": False,
            "blockers": ["xLAM runtime receipt missing; component is not part of the 26-suite count"],
        }
    receipts, errors = _extract_receipts(component)
    blockers = list(errors)
    revision = _revision(component)
    if revision is None:
        blockers.append("xLAM runtime revision is missing or mutable")
    if "runtime" not in receipts:
        blockers.append("xLAM runtime receipt missing")
    if component.get("status") is not None and str(component["status"]).upper() not in {"READY", "VERIFIED", "COMPLETE", "COMPLETED", "RUNNABLE"}:
        blockers.append("xLAM component preflight is not verified-ready")
    return {
        "suite_id": str(component.get("suite_id", XLAM_SUITE_ID)),
        "component": "xLAM",
        "portfolio_suite": False,
        "status": "READY" if not blockers else "BLOCKED",
        "runnable": not blockers,
        "revision": revision,
        "runtime_receipt_ref": receipts.get("runtime"),
        "blockers": sorted(set(blockers)),
    }


def resolve_runnable_suites(
    registry: Mapping[str, Any] | str | Path,
    runtime_receipts: Any,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Return the exact receipt-backed runnable suite IDs.

    Invalid or missing per-suite runtime receipts are represented as blocked
    suite records rather than being silently treated as runnable.  A malformed
    portfolio registry is a hard resolver error because no suite can then be
    trusted.
    """

    registry_value = _load_mapping(registry, "portfolio_registry")
    registry_records, registry_errors = _suite_records(registry_value)
    if registry_errors:
        raise RunnableSuiteResolverError("; ".join(registry_errors))
    recorded_digest = registry_value.get("registry_sha256")
    if not isinstance(recorded_digest, str) or recorded_digest != _registry_digest(registry_value):
        raise RunnableSuiteResolverError("portfolio registry digest is missing or mismatched")
    try:
        verify_registry(registry_value, contract=contract)
    except (PortfolioSplitRegistryError, ValueError) as exc:
        raise RunnableSuiteResolverError("invalid portfolio split registry: " + str(exc)) from exc
    runtime_records, xlam_record, bundle_errors = _runtime_records(runtime_receipts)
    if bundle_errors:
        raise RunnableSuiteResolverError("; ".join(bundle_errors))
    by_id: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in runtime_records:
        suite_id = record.get("suite_id", record.get("id"))
        if not isinstance(suite_id, str) or not suite_id:
            errors.append("runtime receipt is missing suite_id")
            continue
        if suite_id in by_id:
            errors.append(f"duplicate runtime receipt suite ID: {suite_id}")
            continue
        by_id[suite_id] = record
    if xlam_record is None and XLAM_SUITE_ID in by_id:
        xlam_record = by_id.pop(XLAM_SUITE_ID)
    unknown = sorted(set(by_id) - set(registry_records) - {XLAM_SUITE_ID})
    if unknown:
        errors.append("extra runtime receipt suite IDs: " + ", ".join(unknown))
    global_blockers = _registry_global_blockers(registry_value)
    resolutions = [
        _resolve_one(suite_id, registry_records[suite_id], by_id.get(suite_id))
        for role in ROLES
        for suite_id in EXPECTED_SUITE_IDS[role]
    ]
    if errors:
        global_blockers.extend(errors)
        for resolution in resolutions:
            resolution["runnable"] = False
            resolution["status"] = "BLOCKED"
            resolution["blockers"] = sorted(set(resolution["blockers"] + errors))
    runnable = {
        role: sorted(
            item["suite_id"]
            for item in resolutions
            if item["role"] == role and item["runnable"]
        )
        for role in ROLES
    }
    if global_blockers:
        for resolution in resolutions:
            resolution["runnable"] = False
            resolution["status"] = "BLOCKED"
            resolution["blockers"] = sorted(set(resolution["blockers"] + global_blockers))
        runnable = {role: [] for role in ROLES}
    held_private = sorted(
        item["suite_id"]
        for item in resolutions
        if item["runnable"] and item.get("held_out_private")
    )
    primary_only = sorted(
        item["suite_id"]
        for item in resolutions
        if item["runnable"] and item["role"] == PRIMARY_EVAL_ROLE and not item.get("held_out_private")
    )
    xlam = _xlam_resolution(xlam_record)
    blocked = any(not item["runnable"] for item in resolutions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED" if blocked or global_blockers else "READY",
        "launch_authorized": False,
        "launches_any_job": False,
        "registry_sha256": registry_value["registry_sha256"],
        "expected_suite_counts": {role: len(EXPECTED_SUITE_IDS[role]) for role in ROLES},
        "runnable_suite_counts": {role: len(runnable[role]) for role in ROLES},
        "runnable_suite_ids": runnable,
        "runnable_primary_eval_suite_ids": runnable[PRIMARY_EVAL_ROLE],
        "primary_eval_only_suite_ids": primary_only,
        "held_out_private_suite_ids": held_private,
        "blocked_suite_ids": {
            role: sorted(set(EXPECTED_SUITE_IDS[role]) - set(runnable[role])) for role in ROLES
        },
        "suite_resolutions": resolutions,
        "xlam_component": xlam,
        "global_blockers": sorted(set(global_blockers)),
    }
    result["resolution_sha256"] = _sha256(
        _stable_json({key: value for key, value in result.items() if key != "resolution_sha256"})
    )
    return result


resolve = resolve_runnable_suites
build_resolution = resolve_runnable_suites
resolve_portfolio = resolve_runnable_suites


def validate_resolution(
    resolution: Mapping[str, Any] | str | Path,
    registry: Mapping[str, Any] | str | Path,
    runtime_receipts: Any,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    try:
        value = _load_mapping(resolution, "resolution")
        expected = resolve_runnable_suites(registry, runtime_receipts, contract=contract)
        if value != expected:
            return ["resolution metadata drift or non-canonical fields"]
        return []
    except RunnableSuiteResolverError as exc:
        return [str(exc)]


def verify_resolution(
    resolution: Mapping[str, Any] | str | Path,
    registry: Mapping[str, Any] | str | Path,
    runtime_receipts: Any,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
) -> bool:
    errors = validate_resolution(resolution, registry, runtime_receipts, contract=contract)
    if errors:
        raise RunnableSuiteResolverError("invalid runnable-suite resolution: " + "; ".join(errors))
    return True


verify = verify_resolution


def _write_json(value: Any, path: str | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path:
        Path(path).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    resolve_parser = subparsers.add_parser("resolve", help="resolve runnable suites")
    resolve_parser.add_argument("--registry", required=True)
    resolve_parser.add_argument("--runtime-receipts", required=True)
    resolve_parser.add_argument("--contract")
    resolve_parser.add_argument("--out")
    verify_parser = subparsers.add_parser("verify", help="verify an existing resolution")
    verify_parser.add_argument("--resolution", required=True)
    verify_parser.add_argument("--registry", required=True)
    verify_parser.add_argument("--runtime-receipts", required=True)
    verify_parser.add_argument("--contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {"resolve", "verify", "-h", "--help"}:
        args_list.insert(0, "resolve")
    parser = _parser()
    args = parser.parse_args(args_list)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        if args.command == "resolve":
            result = resolve_runnable_suites(
                args.registry,
                args.runtime_receipts,
                contract=args.contract,
            )
            _write_json(result, args.out)
            return 0
        verify_resolution(
            args.resolution,
            args.registry,
            args.runtime_receipts,
            contract=args.contract,
        )
        return 0
    except RunnableSuiteResolverError as exc:
        print(json.dumps({"status": "INVALID", "errors": [str(exc)]}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
