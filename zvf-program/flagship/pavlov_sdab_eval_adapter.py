#!/usr/bin/env python3
"""Offline boundary adapter for the SDAB primary evaluation suite.

This module defines *what must be pinned before an SDAB evaluation can count*;
it does not download SDAB, launch an environment, call Tinker, or invent a
result.  The source identity is the Pavlov contract's ``sdab_eval`` entry and
the official Emulated benchmark page.  A caller supplies the immutable source
revision, license receipt, task identifiers, split manifest, native
environment receipt, artifact receipt, verifier receipt, and backend run
receipt.  The adapter only canonicalizes and hashes those inputs.

``primary_eval`` is a role, not evidence that a split is held out.  A
receipt-proven held-out claim requires immutable split, license, task,
container, and decontamination receipts, plus a completed result receipt.
Related benchmarks and xLAM are never accepted as SDAB substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_ID = "sdab_eval"
ROLE = "primary_eval"
BENCHMARK_ID = "sdab"
BENCHMARK_NAME = "Software Development Automation Benchmark"
CANONICAL_URL = "https://emulated.so/sdab"
PROVIDER = "Emulated, Inc."
OFFICIAL_TASK_COUNT = 80
OFFICIAL_CATEGORIES = (
    "infrastructure_debugging",
    "migrations_and_upgrades",
    "ci_cd_and_deployment",
    "observability_and_incident_response",
    "distributed_systems",
)

REQUIRED_HELDOUT_RECEIPTS = (
    "split",
    "license",
    "task",
    "container",
    "decontamination",
)
RESULT_BACKENDS = ("wandb", "tinker", "hf")
PLACEHOLDER_VALUES = {
    "",
    "latest",
    "main",
    "master",
    "head",
    "todo",
    "unknown",
    "unresolved",
    "to_be_pinned_before_paid_runs",
}
SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")

# Any identifier carrying this marker is a locally generated harness fixture.
# It exercises the ingestion plumbing and must never reach an authoritative
# boundary, a runtime manifest, or a score.
SYNTHETIC_MARKER = "SYNTHETIC-NOT-SDAB"
_SYNTHETIC_TOKENS = ("synthetic", "fixture-not-real", "not-sdab")

RELATED_SUBSTITUTES = {
    "xlam",
    "pavlov_xlam",
    "xlam_eval",
    "swe_bench_pro_eval",
    "frontier_swe_eval",
    "swe_bench",
    "frontier_swe",
    "banker_toolbench_eval",
    "bankertoolbench",
    "apex_agents_eval",
    "apex_agents",
    "webbench_eval",
    "webbench",
}

NATIVE_ENVIRONMENT_CONTRACT = {
    "execution_mode": "native",
    "stateful": True,
    "artifact_or_side_effect": True,
    "deterministic_seed_required": True,
    "policy_visible_surfaces": (
        "workspace",
        "running_infrastructure",
        "operational_tooling",
        "traffic_generator",
    ),
    "grading_harness": "outside_policy_visible_workspace",
}

ARTIFACT_CONTRACT = {
    "required": True,
    "native": True,
    "required_fields": (
        "artifact_id",
        "artifact_type",
        "artifact_digest",
        "state_digest",
    ),
}

VERIFIER_CONTRACT = {
    "required": True,
    "native": True,
    "required_fields": (
        "verifier_id",
        "verifier_revision",
        "verifier_digest",
        "behavioral_tests_digest",
        "rubric_digest",
        "state_validation_digest",
        "hidden_tests_digest",
    ),
    "hidden_tests_outside_policy_workspace": True,
}

COMMON_RESULT_FIELDS = (
    "receipt_id",
    "status",
    "started_at",
    "metrics",
    "model_revision",
    "adapter_revision",
    "environment_receipt",
    "artifact_receipt",
    "verifier_receipt",
)


class SdabBoundaryError(ValueError):
    """Raised when an SDAB boundary or result receipt is unsafe to accept."""


def _required_string(mapping: Mapping[str, Any], key: str, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SdabBoundaryError(f"{field}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: Mapping[str, Any], key: str, field: str) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _required_string(mapping, key, field)


def _immutable_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SdabBoundaryError(f"{field} must be a non-placeholder string")
    normalised = value.strip()
    if normalised.casefold() in PLACEHOLDER_VALUES:
        raise SdabBoundaryError(f"{field} must be pinned; placeholder {normalised!r} is not allowed")
    return normalised


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SdabBoundaryError(f"{field} must be a sha256 digest")
    match = SHA256_RE.fullmatch(value.strip())
    if match is None:
        raise SdabBoundaryError(f"{field} must be sha256:<64 lowercase hexadecimal characters>")
    return "sha256:" + match.group(1).lower()


def _canonical(value: Any, field: str = "value") -> Any:
    """Return a JSON-canonical value without lossy numeric conversion."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise SdabBoundaryError(f"{field} mapping keys must be strings")
            result[key] = _canonical(value[key], f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SdabBoundaryError(f"{field} contains a non-finite float")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise SdabBoundaryError(f"{field} contains a non-JSON value")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for receipt and split hashing."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_provider(value: Any, field: str) -> str:
    provider = _required_string({"value": value}, "value", field)
    if provider.casefold() not in {"emulated", "emulated, inc.", "emulated inc"}:
        raise SdabBoundaryError(f"{field} must identify Emulated, Inc.")
    return PROVIDER


def _looks_synthetic(value: Any) -> bool:
    """Return True when a value is marked as a locally generated fixture."""

    if not isinstance(value, str):
        return False
    normalised = value.strip().casefold().replace("_", "-").replace(" ", "-")
    return any(token in normalised for token in _SYNTHETIC_TOKENS)


def _reject_synthetic(value: Any, field: str) -> None:
    if _looks_synthetic(value):
        raise SdabBoundaryError(
            f"{field}={value!r} is a synthetic harness fixture and can never be "
            "used as authoritative SDAB evidence"
        )


def _reject_substitute(value: Any, field: str) -> None:
    if value is None:
        return
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    for item in values:
        if not isinstance(item, str):
            raise SdabBoundaryError(f"{field} must contain only benchmark identifiers")
        candidate = item.strip().casefold().replace("-", "_").replace(" ", "_")
        if "xlam" in candidate or candidate in RELATED_SUBSTITUTES:
            raise SdabBoundaryError(
                f"{field}={item!r} cannot substitute for the authoritative {SUITE_ID} suite"
            )


def _source_identity(spec: Mapping[str, Any]) -> dict[str, str]:
    suite_id = _required_string(spec, "suite_id", "boundary")
    if suite_id != SUITE_ID:
        _reject_substitute(suite_id, "boundary.suite_id")
        raise SdabBoundaryError(f"boundary.suite_id must be exactly {SUITE_ID!r}")
    role = _required_string(spec, "role", "boundary")
    if role != ROLE:
        raise SdabBoundaryError(f"{SUITE_ID} must have role {ROLE!r}, not {role!r}")

    raw = spec.get("source_identity")
    if not isinstance(raw, Mapping):
        raise SdabBoundaryError("boundary.source_identity must be an object")
    benchmark_id = _required_string(raw, "benchmark_id", "boundary.source_identity")
    if benchmark_id.casefold() != BENCHMARK_ID:
        _reject_substitute(benchmark_id, "boundary.source_identity.benchmark_id")
        raise SdabBoundaryError("boundary.source_identity.benchmark_id must be 'sdab'")
    benchmark_name = _required_string(raw, "benchmark_name", "boundary.source_identity")
    if benchmark_name.casefold() != BENCHMARK_NAME.casefold():
        _reject_substitute(benchmark_name, "boundary.source_identity.benchmark_name")
        raise SdabBoundaryError(
            f"boundary.source_identity.benchmark_name must be {BENCHMARK_NAME!r}"
        )
    url = _required_string(raw, "canonical_url", "boundary.source_identity")
    if url.rstrip("/") != CANONICAL_URL:
        raise SdabBoundaryError(
            f"boundary.source_identity.canonical_url must be {CANONICAL_URL!r}"
        )
    provider = _normalise_provider(raw.get("provider"), "boundary.source_identity.provider")
    # Reject conflicting top-level identity aliases instead of silently
    # canonicalizing a record that also names another benchmark.
    if "benchmark_id" in spec:
        top_level_id = _required_string(spec, "benchmark_id", "boundary")
        if top_level_id.casefold() != BENCHMARK_ID:
            _reject_substitute(top_level_id, "boundary.benchmark_id")
            raise SdabBoundaryError("boundary.benchmark_id must identify SDAB")
    if "benchmark_name" in spec:
        top_level_name = _required_string(spec, "benchmark_name", "boundary")
        if top_level_name.casefold() != BENCHMARK_NAME.casefold():
            _reject_substitute(top_level_name, "boundary.benchmark_name")
            raise SdabBoundaryError("boundary.benchmark_name must identify SDAB")
    if "canonical_url" in spec and _required_string(spec, "canonical_url", "boundary").rstrip("/") != CANONICAL_URL:
        raise SdabBoundaryError("boundary.canonical_url must identify the authoritative SDAB source")
    if "provider" in spec:
        _normalise_provider(spec["provider"], "boundary.provider")
    _reject_substitute(spec.get("substitute_for"), "boundary.substitute_for")
    for key in ("benchmark", "benchmark_alias", "related_benchmark", "evaluation_substitute"):
        _reject_substitute(spec.get(key), f"boundary.{key}")
    return {
        "provider": provider,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_name": BENCHMARK_NAME,
        "canonical_url": CANONICAL_URL,
        "suite_id": SUITE_ID,
    }


def _license_receipt(spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_license = spec.get("license")
    if isinstance(raw_license, str):
        license_info: dict[str, Any] = {"name": _immutable_string(raw_license, "boundary.license")}
    elif isinstance(raw_license, Mapping):
        license_info = dict(raw_license)
        identity_keys = ("name", "spdx_id", "identifier")
        if not any(
            isinstance(license_info.get(key), str) and license_info[key].strip()
            for key in identity_keys
        ):
            raise SdabBoundaryError("boundary.license needs a name or SPDX identifier")
        for key in identity_keys:
            if key in license_info and license_info[key] is not None:
                _immutable_string(license_info[key], f"boundary.license.{key}")
        license_info = _canonical(license_info, "boundary.license")
    else:
        raise SdabBoundaryError("boundary.license must be a string or object")
    raw_receipt = spec.get("license_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise SdabBoundaryError("boundary.license_receipt must be an immutable receipt object")
    receipt = _immutable_receipt(raw_receipt, "boundary.license_receipt")
    return license_info, receipt


def _immutable_receipt(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SdabBoundaryError(f"{field} must be an immutable receipt object")
    receipt_id = _immutable_string(value.get("receipt_id"), f"{field}.receipt_id")
    digest = _digest(value.get("digest"), f"{field}.digest")
    reference_values = [
        value[key]
        for key in ("reference", "uri", "url")
        if key in value and value[key] is not None
    ]
    if not reference_values:
        raise SdabBoundaryError(f"{field}.reference must be a non-placeholder string")
    if not all(isinstance(item, str) and item.strip() for item in reference_values):
        raise SdabBoundaryError(f"{field}.reference must be a non-placeholder string")
    if len({item.strip() for item in reference_values}) != 1:
        raise SdabBoundaryError(f"{field}.reference aliases disagree")
    reference = _immutable_string(reference_values[0], f"{field}.reference")
    result = dict(_canonical(value, field))
    result.update({"receipt_id": receipt_id, "digest": digest, "reference": reference})
    return result


def _task_ids(spec: Mapping[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    raw_tasks = spec.get("tasks")
    task_records: list[dict[str, Any]] = []
    if raw_tasks is not None:
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise SdabBoundaryError("boundary.tasks must be a non-empty list")
        for index, raw in enumerate(raw_tasks):
            if not isinstance(raw, Mapping):
                raise SdabBoundaryError(f"boundary.tasks[{index}] must be an object")
            task_id = _immutable_string(raw.get("task_id"), f"boundary.tasks[{index}].task_id")
            record = dict(_canonical(raw, f"boundary.tasks[{index}]"))
            record["task_id"] = task_id
            category = record.get("category")
            if category is not None:
                if not isinstance(category, str) or not category.strip():
                    raise SdabBoundaryError(f"boundary.tasks[{index}].category must be non-empty")
                if category.strip() not in OFFICIAL_CATEGORIES:
                    raise SdabBoundaryError(
                        f"boundary.tasks[{index}].category is not an SDAB category"
                    )
                record["category"] = category.strip()
            task_records.append(record)
    supplied_ids = spec.get("task_ids")
    if supplied_ids is None:
        ids = [record["task_id"] for record in task_records]
    else:
        if not isinstance(supplied_ids, (list, tuple)) or not supplied_ids:
            raise SdabBoundaryError("boundary.task_ids must be a non-empty list")
        ids = [_immutable_string(value, f"boundary.task_ids[{index}]") for index, value in enumerate(supplied_ids)]
    if len(set(ids)) != len(ids) or len({item.casefold() for item in ids}) != len(ids):
        raise SdabBoundaryError("boundary.task_ids must be unique without case-fold collisions")
    for index, task_id in enumerate(ids):
        _reject_synthetic(task_id, f"boundary.task_ids[{index}]")
    task_record_ids = [record["task_id"] for record in task_records]
    if len(set(task_record_ids)) != len(task_record_ids) or len(
        {item.casefold() for item in task_record_ids}
    ) != len(task_record_ids):
        raise SdabBoundaryError("boundary.tasks must be unique without case-fold collisions")
    if task_records and set(ids) != {record["task_id"] for record in task_records}:
        raise SdabBoundaryError("boundary.tasks and boundary.task_ids do not describe the same tasks")
    if not task_records:
        task_records = [{"task_id": task_id} for task_id in ids]
    ordered_ids = sorted(ids)
    supplied_hash = spec.get("task_id_hash", spec.get("task_ids_hash"))
    task_hash = sha256_digest(ordered_ids)
    if supplied_hash is not None and _digest(supplied_hash, "boundary.task_id_hash") != task_hash:
        raise SdabBoundaryError("boundary.task_id_hash does not match the deterministic task IDs")
    return ordered_ids, sorted(task_records, key=lambda record: record["task_id"])


def _split_manifest(
    spec: Mapping[str, Any], task_ids: Sequence[str], source_revision_digest: str
) -> tuple[str, dict[str, Any]]:
    split = _required_string(spec, "split", "boundary")
    if split != "evaluation":
        raise SdabBoundaryError("sdab_eval requires the evaluation split")
    raw = spec.get("split_manifest")
    if not isinstance(raw, Mapping):
        raise SdabBoundaryError("boundary.split_manifest must be an immutable object")
    name = _required_string(raw, "name", "boundary.split_manifest")
    if name != "evaluation":
        raise SdabBoundaryError("boundary.split_manifest.name must be 'evaluation'")
    revision = _immutable_string(raw.get("revision"), "boundary.split_manifest.revision")
    manifest_digest = _digest(raw.get("digest"), "boundary.split_manifest.digest")
    manifest_ids = raw.get("task_ids")
    if not isinstance(manifest_ids, (list, tuple)):
        raise SdabBoundaryError("boundary.split_manifest.task_ids must be a list")
    normalised_manifest_ids = sorted(
        _immutable_string(value, f"boundary.split_manifest.task_ids[{index}]")
        for index, value in enumerate(manifest_ids)
    )
    if normalised_manifest_ids != list(task_ids):
        raise SdabBoundaryError("boundary.split_manifest.task_ids do not match boundary.task_ids")
    canonical = {
        "name": name,
        "revision": revision,
        "digest": manifest_digest,
        "task_ids": normalised_manifest_ids,
        "source_revision_digest": source_revision_digest,
    }
    split_hash = sha256_digest(canonical)
    supplied_hash = spec.get("split_manifest_hash", spec.get("split_hash"))
    if supplied_hash is not None and _digest(supplied_hash, "boundary.split_manifest_hash") != split_hash:
        raise SdabBoundaryError("boundary.split_manifest_hash does not match the split manifest")
    return split_hash, canonical


def _validate_contract_overrides(spec: Mapping[str, Any]) -> None:
    for key, expected in (
        ("native_environment_contract", NATIVE_ENVIRONMENT_CONTRACT),
        ("artifact_contract", ARTIFACT_CONTRACT),
        ("verifier_contract", VERIFIER_CONTRACT),
    ):
        override = spec.get(key)
        if override is None:
            continue
        if not isinstance(override, Mapping):
            raise SdabBoundaryError(f"boundary.{key} must be an object")
        for required_key, expected_value in expected.items():
            if required_key not in override:
                raise SdabBoundaryError(f"boundary.{key} cannot omit {required_key!r}")
            actual = override[required_key]
            if isinstance(expected_value, tuple):
                if not isinstance(actual, (list, tuple, set)) or not set(expected_value).issubset(set(actual)):
                    raise SdabBoundaryError(f"boundary.{key}.{required_key} weakens the native contract")
            elif actual != expected_value:
                raise SdabBoundaryError(f"boundary.{key}.{required_key} must remain {expected_value!r}")


def build_sdab_boundary(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize and validate an offline SDAB primary-evaluation boundary."""

    if not isinstance(spec, Mapping):
        raise SdabBoundaryError("SDAB boundary must be an object")
    if spec.get("synthetic") or spec.get("is_synthetic") or spec.get("fixture"):
        raise SdabBoundaryError(
            "boundary.synthetic is set; a harness fixture can never become an "
            "authoritative SDAB boundary"
        )
    identity = _source_identity(spec)
    source_revision = _immutable_string(spec.get("source_revision"), "boundary.source_revision")
    _reject_synthetic(source_revision, "boundary.source_revision")
    source_revision_digest = _digest(
        spec.get("source_revision_digest"), "boundary.source_revision_digest"
    )
    supplied_container_keys = tuple(
        key for key in ("container_digest", "container_image_digest") if key in spec and spec[key] is not None
    )
    container_digest = (
        _required_digest_alias(spec, supplied_container_keys, "boundary.container_digest")
        if supplied_container_keys
        else None
    )
    license_info, license_receipt = _license_receipt(spec)
    task_ids, tasks = _task_ids(spec)
    split_hash, split_manifest = _split_manifest(spec, task_ids, source_revision_digest)
    _validate_contract_overrides(spec)
    supplied_task_count = spec.get("task_count")
    if supplied_task_count is not None:
        if not isinstance(supplied_task_count, int) or isinstance(supplied_task_count, bool):
            raise SdabBoundaryError("boundary.task_count must be an integer")
        if supplied_task_count != len(task_ids):
            raise SdabBoundaryError("boundary.task_count does not match task_ids")

    task_hash = sha256_digest(task_ids)
    return {
        "schema_version": "pavlov-sdab-eval-boundary-v1",
        "suite_id": SUITE_ID,
        "role": ROLE,
        "primary_eval": True,
        "source_identity": identity,
        "source_revision": source_revision,
        "source_revision_digest": source_revision_digest,
        # The source revision is the dataset revision named by the Pavlov
        # contamination contract.  Keep both names visible at the boundary.
        "dataset_revision": source_revision,
        "dataset_revision_digest": source_revision_digest,
        "container_digest": container_digest,
        "license": license_info,
        "license_receipt": license_receipt,
        "split": "evaluation",
        "tasks": tasks,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "official_task_count": OFFICIAL_TASK_COUNT,
        "task_id_hash": task_hash,
        "task_id_hashes": {"all": task_hash},
        "split_manifest": split_manifest,
        "split_manifest_hash": split_hash,
        "native_environment_contract": dict(NATIVE_ENVIRONMENT_CONTRACT),
        "artifact_contract": dict(ARTIFACT_CONTRACT),
        "verifier_contract": dict(VERIFIER_CONTRACT),
        "official_metadata": {
            "provider": PROVIDER,
            "benchmark_name": BENCHMARK_NAME,
            "canonical_url": CANONICAL_URL,
            "task_count": OFFICIAL_TASK_COUNT,
            "categories": list(OFFICIAL_CATEGORIES),
        },
        "allowed_result_backends": list(RESULT_BACKENDS),
        "related_benchmarks_are_not_substitutes": True,
        "receipt_proven_heldout": False,
        "heldout_claim_requested": False,
        "heldout_missing_receipts": list(REQUIRED_HELDOUT_RECEIPTS),
    }


def validate_sdab_boundary(spec: Mapping[str, Any]) -> list[str]:
    """Return validation errors without throwing for preflight callers."""

    try:
        build_sdab_boundary(spec)
    except SdabBoundaryError as exc:
        return [str(exc)]
    return []


def _timestamp(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SdabBoundaryError(f"{field} must be a timezone-aware ISO timestamp")
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SdabBoundaryError(f"{field} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise SdabBoundaryError(f"{field} must include a timezone offset")
    return value.strip()


def _required_digest_alias(mapping: Mapping[str, Any], keys: Sequence[str], field: str) -> str:
    values = [mapping[key] for key in keys if key in mapping and mapping[key] is not None]
    if not values:
        raise SdabBoundaryError(f"{field} is missing")
    digests = {_digest(value, field) for value in values}
    if len(digests) != 1:
        raise SdabBoundaryError(f"{field} aliases disagree")
    return next(iter(digests))


def _environment_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SdabBoundaryError("result.environment_receipt must be an object")
    _immutable_string(value.get("environment_id"), "result.environment_receipt.environment_id")
    seed = value.get("seed")
    if not isinstance(seed, (str, int)) or isinstance(seed, bool):
        raise SdabBoundaryError("result.environment_receipt.seed must be a deterministic identifier")
    if isinstance(seed, str) and not seed.strip():
        raise SdabBoundaryError("result.environment_receipt.seed must be a deterministic identifier")
    workspace_digest = _required_digest_alias(
        value, ("workspace_digest",), "result.environment_receipt.workspace_digest"
    )
    state_before_digest = _required_digest_alias(
        value, ("state_before_digest",), "result.environment_receipt.state_before_digest"
    )
    state_after_digest = _required_digest_alias(
        value, ("state_after_digest",), "result.environment_receipt.state_after_digest"
    )
    container_digest = _required_digest_alias(
        value,
        ("container_digest", "container_image_digest"),
        "result.environment_receipt.container_digest",
    )
    environment_digest = _required_digest_alias(
        value, ("environment_digest", "digest"), "result.environment_receipt.environment_digest"
    )
    if value.get("native") is not True:
        raise SdabBoundaryError("result.environment_receipt.native must be true")
    result = dict(_canonical(value, "result.environment_receipt"))
    result["workspace_digest"] = workspace_digest
    result["state_before_digest"] = state_before_digest
    result["state_after_digest"] = state_after_digest
    result["container_digest"] = container_digest
    result["environment_digest"] = environment_digest
    return result


def _artifact_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SdabBoundaryError("result.artifact_receipt must be an object")
    for key in ("artifact_id", "artifact_type"):
        _immutable_string(value.get(key), f"result.artifact_receipt.{key}")
    state_digest = _required_digest_alias(
        value, ("state_digest",), "result.artifact_receipt.state_digest"
    )
    artifact_digest = _required_digest_alias(
        value, ("artifact_digest", "digest"), "result.artifact_receipt.artifact_digest"
    )
    if value.get("native") is not True:
        raise SdabBoundaryError("result.artifact_receipt.native must be true")
    result = dict(_canonical(value, "result.artifact_receipt"))
    result["state_digest"] = state_digest
    result["artifact_digest"] = artifact_digest
    return result


def _verifier_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SdabBoundaryError("result.verifier_receipt must be an object")
    for key in ("verifier_id", "verifier_revision"):
        _immutable_string(value.get(key), f"result.verifier_receipt.{key}")
    digests: dict[str, str] = {}
    for key in (
        "verifier_digest",
        "behavioral_tests_digest",
        "rubric_digest",
        "state_validation_digest",
        "hidden_tests_digest",
    ):
        digests[key] = _required_digest_alias(value, (key,), f"result.verifier_receipt.{key}")
    if value.get("native") is not True:
        raise SdabBoundaryError("result.verifier_receipt.native must be true")
    if value.get("hidden_tests_outside_policy_workspace") is not True:
        raise SdabBoundaryError(
            "result.verifier_receipt.hidden_tests_outside_policy_workspace must be true"
        )
    result = dict(_canonical(value, "result.verifier_receipt"))
    result.update(digests)
    return result


def _backend_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SdabBoundaryError("result.backend must be wandb, tinker, or hf")
    normalised = value.strip().casefold().replace("-", "_")
    aliases = {
        "wandb": "wandb",
        "weights_and_biases": "wandb",
        "tinker": "tinker",
        "hf": "hf",
        "huggingface": "hf",
        "hugging_face": "hf",
    }
    if normalised not in aliases:
        raise SdabBoundaryError(f"unsupported result backend {value!r}")
    return aliases[normalised]


def _normalise_tinker_provider(value: Any, field: str) -> str:
    provider = _required_string({"value": value}, "value", field)
    if provider.casefold() not in {"tinker", "tinker api", "tinker, inc.", "tinker inc"}:
        raise SdabBoundaryError(f"{field} must identify Tinker")
    return "Tinker"


def _backend_fields(result: Mapping[str, Any], backend: str) -> dict[str, str]:
    requirements: dict[str, dict[str, tuple[str, ...]]] = {
        "wandb": {
            "entity": ("wandb_entity", "entity"),
            "project": ("wandb_project", "project"),
            "run_id": ("wandb_run_id", "run_id"),
            "run_url": ("wandb_run_url", "run_url"),
            "summary_digest": ("wandb_summary_digest", "summary_digest"),
        },
        "tinker": {
            "run_id": ("tinker_run_id", "run_id"),
            "job_id": ("tinker_job_id", "job_id"),
            "provider": ("tinker_provider", "provider"),
        },
        "hf": {
            "run_id": ("hf_run_id", "run_id"),
            "repo_id": ("hf_repo_id", "repo_id"),
            "revision": ("hf_revision", "revision"),
        },
    }[backend]
    if backend == "tinker":
        requirements["receipt_id"] = (
            ("tinker_receipt_id", "billing_receipt_id")
            if any(key in result for key in ("tinker_receipt_id", "billing_receipt_id"))
            else ("receipt_id",)
        )
    elif backend == "hf":
        requirements["receipt_id"] = (
            ("hf_receipt_id",)
            if "hf_receipt_id" in result
            else ("receipt_id",)
        )
    fields: dict[str, str] = {}
    for canonical, aliases in requirements.items():
        values = [result[key] for key in aliases if key in result and result[key] is not None]
        if not values:
            raise SdabBoundaryError(f"result.{backend} receipt is missing {canonical}")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise SdabBoundaryError(f"result.{backend}.{canonical} must be non-empty")
        if len({value.strip() for value in values}) != 1:
            raise SdabBoundaryError(f"result.{backend}.{canonical} aliases disagree")
        fields[canonical] = values[0].strip()
    for canonical in fields:
        fields[canonical] = _immutable_string(fields[canonical], f"result.{backend}.{canonical}")
    if canonical := fields.get("summary_digest"):
        fields["summary_digest"] = _digest(canonical, f"result.{backend}.summary_digest")
    if backend == "tinker":
        fields["provider"] = _normalise_tinker_provider(fields["provider"], f"result.{backend}.provider")
    if backend == "hf":
        fields["revision"] = _immutable_string(fields["revision"], f"result.{backend}.revision")
    return fields


def _claim_requested(result: Mapping[str, Any]) -> bool:
    aliases = [
        result[key]
        for key in ("heldout_claim_requested", "claim_heldout")
        if key in result and result[key] is not None
    ]
    if not aliases:
        return False
    parsed: list[bool] = []
    for value in aliases:
        if isinstance(value, bool):
            parsed.append(value)
        elif isinstance(value, str) and value.strip().casefold() in {
            "true",
            "1",
            "yes",
            "heldout",
            "held-out",
        }:
            parsed.append(True)
        elif isinstance(value, str) and value.strip().casefold() in {"false", "0", "no", ""}:
            parsed.append(False)
        else:
            raise SdabBoundaryError("result.heldout_claim_requested must be boolean")
    if len(set(parsed)) != 1:
        raise SdabBoundaryError("result.heldout_claim_requested aliases disagree")
    return parsed[0]


def _heldout_receipts(result: Mapping[str, Any], completed: bool) -> tuple[dict[str, Any], list[str]]:
    heldout_raw = result.get("heldout_receipts")
    immutable_raw = result.get("immutable_receipts")
    if heldout_raw is not None and immutable_raw is not None:
        if not isinstance(heldout_raw, Mapping) or not isinstance(immutable_raw, Mapping):
            raise SdabBoundaryError("result.heldout_receipts aliases must be objects")
        if canonical_json(heldout_raw) != canonical_json(immutable_raw):
            raise SdabBoundaryError("result.heldout_receipts aliases disagree")
    raw = heldout_raw if heldout_raw is not None else immutable_raw
    if raw is None:
        missing = list(REQUIRED_HELDOUT_RECEIPTS)
        if not completed:
            missing.append("completed_result")
        return {}, missing
    if not isinstance(raw, Mapping):
        raise SdabBoundaryError("result.heldout_receipts must be an object")
    unknown = set(raw) - set(REQUIRED_HELDOUT_RECEIPTS)
    if unknown:
        _reject_substitute(sorted(unknown), "result.heldout_receipts")
        raise SdabBoundaryError(
            "result.heldout_receipts contains unsupported requirement(s): "
            + ",".join(sorted(str(item) for item in unknown))
        )
    normalised: dict[str, Any] = {}
    missing: list[str] = []
    for name in REQUIRED_HELDOUT_RECEIPTS:
        if name not in raw:
            missing.append(name)
            continue
        normalised[name] = _immutable_receipt(raw[name], f"result.heldout_receipts.{name}")
    receipt_ids = [receipt["receipt_id"] for receipt in normalised.values()]
    if len(set(receipt_ids)) != len(receipt_ids):
        raise SdabBoundaryError("result.heldout_receipts must use a distinct immutable receipt for each requirement")
    receipt_fingerprints = [
        (receipt["digest"], receipt["reference"]) for receipt in normalised.values()
    ]
    if len(set(receipt_fingerprints)) != len(receipt_fingerprints):
        raise SdabBoundaryError("result.heldout_receipts must not reuse one immutable receipt")
    if not completed:
        missing.append("completed_result")
    return normalised, missing


def build_result_receipt(
    boundary: Mapping[str, Any], result: Mapping[str, Any], backend: str | None = None
) -> dict[str, Any]:
    """Validate one W&B, Tinker, or HF receipt against the SDAB boundary."""

    canonical_boundary = build_sdab_boundary(boundary)
    if not isinstance(result, Mapping):
        raise SdabBoundaryError("SDAB result receipt must be an object")
    if backend is None:
        backend = result.get("backend")
    backend_name = _backend_name(backend)
    if backend_name not in RESULT_BACKENDS:
        raise SdabBoundaryError(f"unsupported result backend {backend!r}")
    if result.get("backend") is not None and _backend_name(result["backend"]) != backend_name:
        raise SdabBoundaryError("result.backend does not match the selected backend")
    _reject_substitute(result.get("suite_id"), "result.suite_id")
    for key in ("benchmark_id", "benchmark", "evaluation_suite", "substitute_for", "related_benchmark"):
        _reject_substitute(result.get(key), f"result.{key}")
    if result.get("suite_id") not in (None, SUITE_ID):
        raise SdabBoundaryError("result.suite_id must match sdab_eval")
    if result.get("role") not in (None, ROLE):
        raise SdabBoundaryError("result.role must remain primary_eval")
    if result.get("benchmark_id") is not None and str(result["benchmark_id"]).casefold() != BENCHMARK_ID:
        raise SdabBoundaryError("result.benchmark_id must identify the authoritative SDAB benchmark")
    if result.get("benchmark") is not None:
        benchmark = result["benchmark"]
        if not isinstance(benchmark, str) or benchmark.strip().casefold() not in {
            BENCHMARK_ID,
            BENCHMARK_NAME.casefold(),
        }:
            raise SdabBoundaryError("result.benchmark must identify the authoritative SDAB benchmark")
    if result.get("evaluation_suite") is not None and str(result["evaluation_suite"]).casefold() != SUITE_ID:
        raise SdabBoundaryError("result.evaluation_suite must identify sdab_eval")
    if result.get("benchmark_name") is not None:
        if (
            not isinstance(result["benchmark_name"], str)
            or result["benchmark_name"].strip().casefold() != BENCHMARK_NAME.casefold()
        ):
            _reject_substitute(result["benchmark_name"], "result.benchmark_name")
            raise SdabBoundaryError("result.benchmark_name must identify the authoritative SDAB benchmark")
    if result.get("canonical_url") is not None and (
        not isinstance(result["canonical_url"], str)
        or result["canonical_url"].strip().rstrip("/") != CANONICAL_URL
    ):
        raise SdabBoundaryError("result.canonical_url must identify the authoritative SDAB source")
    if result.get("provider") is not None:
        _normalise_provider(result["provider"], "result.provider")
    if "source_identity" in result:
        result_identity = _source_identity(
            {
                "suite_id": result.get("suite_id", SUITE_ID),
                "role": result.get("role", ROLE),
                "source_identity": result["source_identity"],
            }
        )
        if result_identity != canonical_boundary["source_identity"]:
            raise SdabBoundaryError("result.source_identity does not match the SDAB boundary")
    if "source_revision" in result and _immutable_string(
        result["source_revision"], "result.source_revision"
    ) != canonical_boundary["source_revision"]:
        raise SdabBoundaryError("result.source_revision does not match the SDAB boundary")
    for key, expected in (
        ("source_revision_digest", canonical_boundary["source_revision_digest"]),
        ("dataset_revision_digest", canonical_boundary["source_revision_digest"]),
        ("task_id_hash", canonical_boundary["task_id_hash"]),
        ("split_manifest_hash", canonical_boundary["split_manifest_hash"]),
    ):
        if key in result and _digest(result[key], f"result.{key}") != expected:
            raise SdabBoundaryError(f"result.{key} does not match the SDAB boundary")
    if "task_id_hashes" in result:
        supplied_hashes = result["task_id_hashes"]
        if not isinstance(supplied_hashes, Mapping):
            raise SdabBoundaryError("result.task_id_hashes must be an object")
        if "all" not in supplied_hashes or _digest(
            supplied_hashes["all"], "result.task_id_hashes.all"
        ) != canonical_boundary["task_id_hash"]:
            raise SdabBoundaryError("result.task_id_hashes does not match the SDAB boundary")
    if "dataset_revision" in result and _immutable_string(
        result["dataset_revision"], "result.dataset_revision"
    ) != canonical_boundary["dataset_revision"]:
        raise SdabBoundaryError("result.dataset_revision does not match the SDAB boundary")
    if "container_digest" in result and canonical_boundary["container_digest"] is not None:
        if _digest(result["container_digest"], "result.container_digest") != canonical_boundary["container_digest"]:
            raise SdabBoundaryError("result.container_digest does not match the SDAB boundary")

    receipt_id = _immutable_string(result.get("receipt_id"), "result.receipt_id")
    status = _required_string(result, "status", "result").casefold()
    if status not in {"completed", "failed", "running", "pending", "rejected"}:
        raise SdabBoundaryError(f"result.status {status!r} is not a supported receipt state")
    started_at = _timestamp(result.get("started_at"), "result.started_at")
    completed_at = _timestamp(
        result.get("completed_at"), "result.completed_at", required=status in {"completed", "failed", "rejected"}
    )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SdabBoundaryError("result.metrics must be an object")
    if status == "completed" and not metrics:
        raise SdabBoundaryError("completed SDAB results must include metrics; no result may be fabricated")
    metrics_digest = _required_digest_alias(result, ("metrics_digest",), "result.metrics_digest")
    expected_metrics_digest = sha256_digest(metrics)
    if metrics_digest != expected_metrics_digest:
        raise SdabBoundaryError("result.metrics_digest does not match result.metrics")
    model_revision = _immutable_string(result.get("model_revision"), "result.model_revision")
    adapter_revision = _immutable_string(result.get("adapter_revision"), "result.adapter_revision")
    environment = _environment_receipt(result.get("environment_receipt"))
    if "container_digest" in result:
        if _digest(result["container_digest"], "result.container_digest") != environment["container_digest"]:
            raise SdabBoundaryError("result.container_digest does not match the environment receipt")
    if (
        canonical_boundary["container_digest"] is not None
        and environment["container_digest"] != canonical_boundary["container_digest"]
    ):
        raise SdabBoundaryError("result.environment_receipt.container_digest does not match the SDAB boundary")
    artifact = _artifact_receipt(result.get("artifact_receipt"))
    verifier = _verifier_receipt(result.get("verifier_receipt"))
    backend_fields = _backend_fields(result, backend_name)
    heldout, missing = _heldout_receipts(result, status == "completed")
    claim_requested = _claim_requested(result)
    if claim_requested and missing:
        raise SdabBoundaryError(
            "held-out claim requires immutable split/license/task/container/decontamination "
            "receipts and a completed result; missing=" + ",".join(missing)
        )
    heldout_proven = not missing
    receipt = {
        "schema_version": "pavlov-sdab-eval-result-receipt-v1",
        "receipt_id": receipt_id,
        "backend": backend_name,
        "backend_fields": backend_fields,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "primary_eval": True,
        "source_identity": canonical_boundary["source_identity"],
        "source_revision": canonical_boundary["source_revision"],
        "source_revision_digest": canonical_boundary["source_revision_digest"],
        "dataset_revision": canonical_boundary["dataset_revision"],
        "dataset_revision_digest": canonical_boundary["dataset_revision_digest"],
        "container_digest": environment["container_digest"],
        "license": canonical_boundary["license"],
        "license_receipt": canonical_boundary["license_receipt"],
        "split": canonical_boundary["split"],
        "task_count": canonical_boundary["task_count"],
        "official_task_count": canonical_boundary["official_task_count"],
        "task_id_hash": canonical_boundary["task_id_hash"],
        "task_id_hashes": canonical_boundary["task_id_hashes"],
        "split_manifest": canonical_boundary["split_manifest"],
        "split_manifest_hash": canonical_boundary["split_manifest_hash"],
        "model_revision": model_revision,
        "adapter_revision": adapter_revision,
        "metrics": dict(_canonical(metrics, "result.metrics")),
        "metrics_digest": metrics_digest,
        "environment_receipt": environment,
        "artifact_receipt": artifact,
        "verifier_receipt": verifier,
        "heldout_receipts": heldout,
        "heldout_claim_requested": claim_requested,
        "receipt_proven_heldout": heldout_proven,
        "heldout_missing_receipts": missing,
        "related_benchmarks_are_not_substitutes": True,
    }
    supplied_receipt_digest = result.get("receipt_digest")
    receipt_digest = sha256_digest(receipt)
    if supplied_receipt_digest is not None and _digest(
        supplied_receipt_digest, "result.receipt_digest"
    ) != receipt_digest:
        raise SdabBoundaryError("result.receipt_digest does not match the canonical receipt")
    receipt["receipt_digest"] = receipt_digest
    return receipt


def validate_result_receipt(
    boundary: Mapping[str, Any], result: Mapping[str, Any], backend: str | None = None
) -> list[str]:
    try:
        build_result_receipt(boundary, result, backend)
    except SdabBoundaryError as exc:
        return [str(exc)]
    return []


# Compatibility names for callers that use shorter adapter/preflight verbs.
build_boundary = build_sdab_boundary
validate_boundary = validate_sdab_boundary
adapt_result = build_result_receipt
validate_result = validate_result_receipt


# ---------------------------------------------------------------------------
# Provider bundle ingestion
#
# Everything below is ready *before* provider access exists.  It turns a
# provider-issued 80-task SDAB evaluation bundle into (a) a validated bundle
# report, (b) a split manifest, (c) a train/eval disjointness proof, (d) an
# authoritative boundary, and (e) the metadata-only runtime manifest that
# ``flagship.eval_pavlov_sdab`` accepts.  None of it can invent a score.
# ---------------------------------------------------------------------------

BUNDLE_SCHEMA_VERSION = "pavlov-sdab-eval-bundle-v1"
INGEST_SCHEMA_VERSION = "pavlov-sdab-eval-bundle-ingest-v1"
DISJOINTNESS_SCHEMA_VERSION = "pavlov-sdab-eval-disjointness-v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = "pavlov-sdab-e3-runtime-manifest-v1"

INGEST_MODES = ("authoritative", "harness_validation")

# Keys that may carry raw benchmark content.  They are permitted inside the
# provider bundle (that is the data) but are stripped from every derived
# artifact so raw prompts/targets never cross a receipt boundary.
RAW_CONTENT_KEYS = (
    "answer",
    "answers",
    "gold",
    "gold_patch",
    "hidden_tests",
    "instructions",
    "patch",
    "prompt",
    "prompts",
    "rubric",
    "solution",
    "solutions",
    "target",
    "targets",
    "test",
    "tests",
    "trajectory",
    "trajectories",
)


class SdabBundleError(SdabBoundaryError):
    """Raised when a provider bundle is unsafe to ingest."""


def newline_task_id_sha256(task_ids: Sequence[str]) -> str:
    """Hash ordered task IDs the way ``flagship.eval_pavlov_sdab`` does.

    The runtime manifest gate hashes ``"\\n".join(task_ids)`` and returns bare
    hex.  The boundary hashes the canonical JSON array and returns a
    ``sha256:`` digest.  Both are emitted so the two receipt families stay
    reconcilable instead of silently disagreeing.
    """

    ids = [_immutable_string(value, f"task_ids[{index}]") for index, value in enumerate(task_ids)]
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _bundle_synthetic_flag(bundle: Mapping[str, Any]) -> bool:
    flags = [
        bundle[key]
        for key in ("synthetic", "is_synthetic", "fixture")
        if key in bundle and bundle[key] is not None
    ]
    parsed = set()
    for value in flags:
        if isinstance(value, bool):
            parsed.add(value)
        elif isinstance(value, str):
            parsed.add(value.strip().casefold() in {"true", "1", "yes", "synthetic"})
        else:
            raise SdabBundleError("bundle.synthetic must be boolean")
    if len(parsed) > 1:
        raise SdabBundleError("bundle synthetic flags disagree")
    return bool(parsed and next(iter(parsed)))


def _strip_raw_content(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    stripped = sorted(key for key in record if key.casefold() in RAW_CONTENT_KEYS)
    metadata = {key: record[key] for key in record if key.casefold() not in RAW_CONTENT_KEYS}
    return metadata, stripped


def validate_task_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_task_count: int = OFFICIAL_TASK_COUNT,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Validate a provider-issued SDAB evaluation bundle and hash its task IDs.

    Returns a metadata-only bundle report.  Raw task content present in the
    bundle is recorded by key name and dropped, never copied forward.
    """

    if not isinstance(bundle, Mapping):
        raise SdabBundleError("SDAB bundle must be an object")
    synthetic = _bundle_synthetic_flag(bundle)
    if synthetic and not allow_synthetic:
        raise SdabBundleError(
            "bundle is marked synthetic; ingest it with mode='harness_validation' "
            "or supply the authentic provider bundle"
        )

    benchmark_id = _required_string(bundle, "benchmark_id", "bundle")
    if benchmark_id.casefold() != BENCHMARK_ID:
        _reject_substitute(benchmark_id, "bundle.benchmark_id")
        raise SdabBundleError("bundle.benchmark_id must be 'sdab'")
    for key in ("benchmark", "related_benchmark", "substitute_for", "evaluation_substitute"):
        _reject_substitute(bundle.get(key), f"bundle.{key}")
    if "benchmark_name" in bundle:
        name = _required_string(bundle, "benchmark_name", "bundle")
        if name.casefold() != BENCHMARK_NAME.casefold():
            _reject_substitute(name, "bundle.benchmark_name")
            raise SdabBundleError(f"bundle.benchmark_name must be {BENCHMARK_NAME!r}")
    if "canonical_url" in bundle and _required_string(
        bundle, "canonical_url", "bundle"
    ).rstrip("/") != CANONICAL_URL:
        raise SdabBundleError(f"bundle.canonical_url must be {CANONICAL_URL!r}")
    provider = _normalise_provider(
        bundle.get("provider", PROVIDER), "bundle.provider"
    )

    split = _required_string(bundle, "split", "bundle")
    if split != "evaluation":
        raise SdabBundleError("sdab_eval ingests only the evaluation split")

    revision_keys = [key for key in ("revision", "source_revision") if key in bundle]
    if not revision_keys:
        raise SdabBundleError("bundle.revision is required and must be immutable")
    revisions = {_immutable_string(bundle[key], f"bundle.{key}") for key in revision_keys}
    if len(revisions) != 1:
        raise SdabBundleError("bundle revision aliases disagree")
    revision = next(iter(revisions))
    if not synthetic:
        _reject_synthetic(revision, "bundle.revision")

    license_identity: dict[str, Any]
    raw_license = bundle.get("license", bundle.get("license_id"))
    if isinstance(raw_license, str):
        license_identity = {"name": _immutable_string(raw_license, "bundle.license")}
    elif isinstance(raw_license, Mapping):
        license_identity = _canonical(raw_license, "bundle.license")
        if not any(
            isinstance(license_identity.get(key), str) and license_identity[key].strip()
            for key in ("name", "spdx_id", "identifier")
        ):
            raise SdabBundleError("bundle.license needs a name or SPDX identifier")
    else:
        raise SdabBundleError("bundle.license must be a string or object")
    raw_license_receipt = bundle.get("license_receipt")
    if not isinstance(raw_license_receipt, Mapping):
        raise SdabBundleError("bundle.license_receipt must be an immutable receipt object")
    license_receipt = _immutable_receipt(raw_license_receipt, "bundle.license_receipt")

    raw_tasks = bundle.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SdabBundleError("bundle.tasks must be a non-empty list")
    if expected_task_count is not None and len(raw_tasks) != expected_task_count:
        raise SdabBundleError(
            f"bundle.tasks must contain exactly {expected_task_count} tasks; got {len(raw_tasks)}"
        )

    ordered_ids: list[str] = []
    records: list[dict[str, Any]] = []
    stripped_keys: set[str] = set()
    categories: dict[str, int] = {}
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise SdabBundleError(f"bundle.tasks[{index}] must be an object")
        task_id = _immutable_string(raw.get("task_id"), f"bundle.tasks[{index}].task_id")
        if synthetic:
            if not _looks_synthetic(task_id):
                raise SdabBundleError(
                    f"bundle.tasks[{index}].task_id must carry the {SYNTHETIC_MARKER} "
                    "marker in a synthetic bundle"
                )
        else:
            _reject_synthetic(task_id, f"bundle.tasks[{index}].task_id")
        metadata, stripped = _strip_raw_content(raw)
        stripped_keys.update(stripped)
        record = dict(_canonical(metadata, f"bundle.tasks[{index}]"))
        record["task_id"] = task_id
        category = record.get("category")
        if category is not None:
            if not isinstance(category, str) or category.strip() not in OFFICIAL_CATEGORIES:
                raise SdabBundleError(
                    f"bundle.tasks[{index}].category is not an SDAB category"
                )
            record["category"] = category.strip()
            categories[record["category"]] = categories.get(record["category"], 0) + 1
        ordered_ids.append(task_id)
        records.append(record)

    if len(set(ordered_ids)) != len(ordered_ids) or len(
        {item.casefold() for item in ordered_ids}
    ) != len(ordered_ids):
        raise SdabBundleError("bundle task IDs must be unique without case-fold collisions")

    sorted_ids = sorted(ordered_ids)
    report = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "synthetic": synthetic,
        "authoritative": not synthetic,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_name": BENCHMARK_NAME,
        "canonical_url": CANONICAL_URL,
        "provider": provider,
        "revision": revision,
        "split": "evaluation",
        "license": license_identity,
        "license_receipt": license_receipt,
        "task_count": len(sorted_ids),
        "expected_task_count": expected_task_count,
        "task_ids": sorted_ids,
        "task_order": ordered_ids,
        "tasks": sorted(records, key=lambda item: item["task_id"]),
        "categories": dict(sorted(categories.items())),
        # Both hash schemes, so the runtime manifest and the boundary receipt
        # can be reconciled against one another.
        "task_id_sha256": newline_task_id_sha256(sorted_ids),
        "task_id_sha256_supplied_order": newline_task_id_sha256(ordered_ids),
        "task_id_digest": sha256_digest(sorted_ids),
        "raw_content_keys_stripped": sorted(stripped_keys),
    }
    report["bundle_digest"] = sha256_digest(
        {key: value for key, value in report.items() if key != "bundle_digest"}
    )
    return report


def build_split_manifest(
    bundle_report: Mapping[str, Any],
    *,
    revision: str | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    """Construct the evaluation split manifest from a validated bundle."""

    task_ids = list(bundle_report["task_ids"])
    manifest_revision = _immutable_string(
        revision if revision is not None else f"{bundle_report['revision']}#evaluation-split",
        "split_manifest.revision",
    )
    manifest_digest = (
        _digest(digest, "split_manifest.digest")
        if digest is not None
        else sha256_digest({"revision": manifest_revision, "task_ids": task_ids})
    )
    return {
        "name": "evaluation",
        "revision": manifest_revision,
        "digest": manifest_digest,
        "task_ids": task_ids,
    }


def prove_split_disjointness(
    eval_task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Prove that evaluation and training task IDs do not overlap.

    Overlap is checked case-insensitively: a training ID differing only in case
    is still contamination.  With ``strict`` the function fails closed.
    """

    eval_ids = sorted(
        _immutable_string(value, f"eval_task_ids[{index}]")
        for index, value in enumerate(eval_task_ids)
    )
    train_ids = sorted(
        _immutable_string(value, f"train_task_ids[{index}]")
        for index, value in enumerate(train_task_ids)
    )
    if not eval_ids:
        raise SdabBundleError("disjointness proof needs a non-empty evaluation split")
    exact = sorted(set(eval_ids) & set(train_ids))
    folded = sorted(
        task_id
        for task_id in eval_ids
        if task_id.casefold() in {item.casefold() for item in train_ids}
    )
    eval_hash = newline_task_id_sha256(eval_ids)
    train_hash = newline_task_id_sha256(train_ids) if train_ids else None
    disjoint = not exact and not folded and (train_hash is None or train_hash != eval_hash)
    proof = {
        "schema_version": DISJOINTNESS_SCHEMA_VERSION,
        "method": "case-folded set intersection over immutable task IDs",
        "eval_task_count": len(eval_ids),
        "train_task_count": len(train_ids),
        "eval_task_id_sha256": eval_hash,
        "train_task_id_sha256": train_hash,
        "eval_task_id_digest": sha256_digest(eval_ids),
        "train_task_id_digest": sha256_digest(train_ids) if train_ids else None,
        "intersection_count": len(exact),
        "casefold_intersection_count": len(folded),
        "overlapping_task_ids": exact,
        "disjoint": disjoint,
        "train_split_supplied": bool(train_ids),
    }
    proof["proof_digest"] = sha256_digest(proof)
    if strict and not disjoint:
        raise SdabBundleError(
            "train/eval task IDs are not disjoint; overlap="
            + (",".join(exact or folded) or "task_id_hash_collision")
        )
    if strict and not train_ids:
        raise SdabBundleError(
            "a disjointness proof requires the provider's training task-ID list; "
            "none was supplied"
        )
    return proof


def build_boundary_spec_from_bundle(
    bundle_report: Mapping[str, Any],
    *,
    source_revision_digest: str,
    split_manifest: Mapping[str, Any] | None = None,
    container_digest: str | None = None,
) -> dict[str, Any]:
    """Derive the offline boundary spec that :func:`build_sdab_boundary` accepts."""

    manifest = dict(split_manifest) if split_manifest is not None else build_split_manifest(bundle_report)
    spec: dict[str, Any] = {
        "suite_id": SUITE_ID,
        "role": ROLE,
        "source_identity": {
            "provider": bundle_report["provider"],
            "benchmark_id": BENCHMARK_ID,
            "benchmark_name": BENCHMARK_NAME,
            "canonical_url": CANONICAL_URL,
        },
        "source_revision": bundle_report["revision"],
        "source_revision_digest": _digest(source_revision_digest, "bundle.source_revision_digest"),
        "license": bundle_report["license"],
        "license_receipt": bundle_report["license_receipt"],
        "split": "evaluation",
        "tasks": [dict(record) for record in bundle_report["tasks"]],
        "task_ids": list(bundle_report["task_ids"]),
        "task_count": bundle_report["task_count"],
        "split_manifest": manifest,
    }
    if container_digest is not None:
        spec["container_digest"] = _digest(container_digest, "bundle.container_digest")
    return spec


def build_runtime_manifest(
    ingest_report: Mapping[str, Any],
    *,
    container_digest: str,
    environment_digest: str,
    verifier_sha256: str,
    verifier_identity: str,
    adapter_entrypoint: str,
    disjointness_receipt: str,
    license_id: str | None = None,
    license_receipt: str | None = None,
) -> dict[str, Any]:
    """Emit the metadata-only runtime manifest ``eval_pavlov_sdab`` accepts.

    Raw task content never reaches this manifest: only task IDs, hashes, and
    provider digests.  A synthetic ingest is refused outright, so the fixture
    can never launch a run.
    """

    if ingest_report.get("synthetic"):
        raise SdabBundleError(
            "a synthetic harness fixture can never produce an SDAB runtime manifest"
        )
    bundle = ingest_report["bundle"]
    proof = ingest_report["disjointness_proof"]
    if not proof.get("disjoint"):
        raise SdabBundleError("runtime manifest requires a proven disjoint split")
    train_hash = proof.get("train_task_id_sha256")
    if not isinstance(train_hash, str) or not train_hash:
        raise SdabBundleError("runtime manifest requires the training task-ID hash")
    task_ids = list(bundle["task_ids"])
    resolved_license = license_id
    if resolved_license is None:
        license_identity = bundle["license"]
        resolved_license = (
            license_identity
            if isinstance(license_identity, str)
            else license_identity.get("spdx_id") or license_identity.get("name")
        )
    manifest = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "benchmark_name": BENCHMARK_NAME,
        "provider": "Emulated",
        "official_url": CANONICAL_URL,
        "benchmark_revision": _immutable_string(
            bundle["revision"], "runtime_manifest.benchmark_revision"
        ),
        "license_id": _immutable_string(resolved_license, "runtime_manifest.license_id"),
        "license_receipt": _immutable_string(
            license_receipt
            if license_receipt is not None
            else bundle["license_receipt"]["reference"],
            "runtime_manifest.license_receipt",
        ),
        "split": "evaluation",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "task_id_sha256": newline_task_id_sha256(task_ids),
        "train_task_id_sha256": train_hash,
        "disjointness_receipt": _immutable_string(
            disjointness_receipt, "runtime_manifest.disjointness_receipt"
        ),
        "container_digest": _digest(container_digest, "runtime_manifest.container_digest"),
        "environment_digest": _digest(environment_digest, "runtime_manifest.environment_digest"),
        "verifier_sha256": _immutable_string(verifier_sha256, "runtime_manifest.verifier_sha256"),
        "verifier_identity": _immutable_string(
            verifier_identity, "runtime_manifest.verifier_identity"
        ),
        "adapter_entrypoint": _immutable_string(
            adapter_entrypoint, "runtime_manifest.adapter_entrypoint"
        ),
        "native_verifier": True,
        "stateful": True,
        "artifact_or_side_effect": True,
    }
    forbidden = sorted(set(manifest) & set(RAW_CONTENT_KEYS))
    if forbidden:
        raise SdabBundleError(
            "runtime manifest must be metadata-only; found " + ",".join(forbidden)
        )
    return manifest


def ingest_task_bundle(
    bundle: Mapping[str, Any],
    *,
    train_task_ids: Sequence[str] = (),
    mode: str = "authoritative",
    source_revision_digest: str | None = None,
    container_digest: str | None = None,
    split_manifest_revision: str | None = None,
    split_manifest_digest: str | None = None,
    expected_task_count: int = OFFICIAL_TASK_COUNT,
) -> dict[str, Any]:
    """Ingest an SDAB evaluation bundle end to end.

    ``mode='authoritative'`` requires an authentic provider bundle and builds
    the canonical boundary.  ``mode='harness_validation'`` accepts a synthetic
    fixture, exercises exactly the same plumbing, and proves the fixture is
    rejected by the authoritative boundary.  Neither mode produces a score.
    """

    if mode not in INGEST_MODES:
        raise SdabBundleError(f"unsupported ingest mode {mode!r}")
    harness_only = mode == "harness_validation"
    bundle_report = validate_task_bundle(
        bundle, expected_task_count=expected_task_count, allow_synthetic=harness_only
    )
    if harness_only and not bundle_report["synthetic"]:
        raise SdabBundleError(
            "mode='harness_validation' is only for synthetic fixtures; an authentic "
            "bundle must be ingested with mode='authoritative'"
        )
    split_manifest = build_split_manifest(
        bundle_report, revision=split_manifest_revision, digest=split_manifest_digest
    )
    proof = prove_split_disjointness(
        bundle_report["task_ids"], train_task_ids, strict=not harness_only
    )

    report: dict[str, Any] = {
        "schema_version": INGEST_SCHEMA_VERSION,
        "mode": mode,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "synthetic": bundle_report["synthetic"],
        "authoritative": not harness_only,
        "bundle": bundle_report,
        "split_manifest": split_manifest,
        "split_manifest_hash": None,
        "disjointness_proof": proof,
        "boundary": None,
        "boundary_rejection": None,
        # A bundle ingest describes plumbing, never performance.
        "score": None,
        "is_model_score": False,
        "evidence_kind": "harness_validation" if harness_only else "boundary_pin",
        "related_benchmarks_are_not_substitutes": True,
    }

    spec = build_boundary_spec_from_bundle(
        bundle_report,
        source_revision_digest=source_revision_digest
        or sha256_digest({"revision": bundle_report["revision"], "bundle_digest": bundle_report["bundle_digest"]}),
        split_manifest=split_manifest,
        container_digest=container_digest,
    )
    if harness_only:
        # The fixture must be refused by the authoritative path.  If it is not,
        # the guard has regressed and ingestion fails closed.
        errors = validate_sdab_boundary(spec)
        if not errors:
            raise SdabBundleError(
                "synthetic fixture was accepted by the authoritative boundary; "
                "refusing to continue"
            )
        report["boundary_rejection"] = errors
    else:
        boundary = build_sdab_boundary(spec)
        report["boundary"] = boundary
        report["split_manifest_hash"] = boundary["split_manifest_hash"]
    report["ingest_digest"] = sha256_digest(report)
    return report


def build_ingest_receipt(
    ingest_report: Mapping[str, Any],
    *,
    status: str,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit a receipt for a bundle ingest.  Never carries a score."""

    if status not in {"READY", "BLOCKED", "PARTIAL"}:
        raise SdabBundleError(f"unsupported ingest receipt status {status!r}")
    receipt = {
        "schema_version": "pavlov-sdab-eval-ingest-receipt-v1",
        "suite_id": SUITE_ID,
        "role": ROLE,
        "status": status,
        "mode": ingest_report["mode"],
        "synthetic": ingest_report["synthetic"],
        "authoritative": ingest_report["authoritative"],
        "score": None,
        "is_model_score": False,
        "evidence_kind": ingest_report["evidence_kind"],
        "task_count": ingest_report["bundle"]["task_count"],
        "task_id_sha256": ingest_report["bundle"]["task_id_sha256"],
        "task_id_digest": ingest_report["bundle"]["task_id_digest"],
        "split_manifest": ingest_report["split_manifest"],
        "split_manifest_hash": ingest_report["split_manifest_hash"],
        "disjointness_proof": ingest_report["disjointness_proof"],
        "boundary_rejection": ingest_report["boundary_rejection"],
        "ingest_digest": ingest_report["ingest_digest"],
        "blockers": list(blockers),
    }
    receipt["receipt_digest"] = sha256_digest(receipt)
    return receipt


def _load_json(path: Path, field: str) -> Any:
    if not path.is_file():
        raise SdabBoundaryError(f"{field} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SdabBoundaryError(f"{field} is malformed: {path}") from exc


def _display(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_display(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "boundary", type=Path, nargs="?", help="offline SDAB boundary JSON"
    )
    parser.add_argument("--result", type=Path, help="optional offline backend result receipt JSON")
    parser.add_argument("--backend", choices=RESULT_BACKENDS)
    parser.add_argument(
        "--bundle",
        type=Path,
        help="provider-issued SDAB evaluation bundle JSON to ingest",
    )
    parser.add_argument(
        "--train-task-ids",
        type=Path,
        help="JSON list (or object with task_ids) of the training split task IDs",
    )
    parser.add_argument(
        "--mode",
        choices=INGEST_MODES,
        default="authoritative",
        help="'harness_validation' accepts a synthetic fixture and never scores",
    )
    parser.add_argument("--source-revision-digest")
    parser.add_argument("--container-digest")
    parser.add_argument("--out", type=Path, help="write the ingest receipt to this path")
    args = parser.parse_args(argv)
    if args.bundle is None and args.boundary is None:
        print(
            json.dumps(
                {"status": "ERROR", "errors": ["either a boundary path or --bundle is required"]},
                indent=2,
            )
        )
        return 1
    try:
        output: dict[str, Any] = {}
        if args.bundle is not None:
            train_ids: list[str] = []
            if args.train_task_ids is not None:
                raw_train = _load_json(args.train_task_ids, "training task IDs")
                if isinstance(raw_train, Mapping):
                    raw_train = raw_train.get("task_ids")
                if not isinstance(raw_train, list):
                    raise SdabBoundaryError("training task IDs must be a JSON list")
                train_ids = [str(item) for item in raw_train]
            ingest = ingest_task_bundle(
                _load_json(args.bundle, "bundle"),
                train_task_ids=train_ids,
                mode=args.mode,
                source_revision_digest=args.source_revision_digest,
                container_digest=args.container_digest,
            )
            receipt = build_ingest_receipt(
                ingest,
                status="BLOCKED" if ingest["synthetic"] else "READY",
                blockers=(
                    ["synthetic fixture: harness validation only, never a model score"]
                    if ingest["synthetic"]
                    else []
                ),
            )
            output["ingest"] = ingest
            output["ingest_receipt"] = receipt
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(_display(receipt), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        if args.boundary is not None:
            boundary = build_sdab_boundary(_load_json(args.boundary, "boundary"))
            output["boundary"] = boundary
            if args.result is not None:
                if args.backend is None:
                    raise SdabBoundaryError("--backend is required with --result")
                output["result_receipt"] = build_result_receipt(
                    boundary, _load_json(args.result, "result receipt"), args.backend
                )
        print(json.dumps(_display(output), indent=2, sort_keys=True))
        return 0
    except SdabBoundaryError as exc:
        print(json.dumps({"status": "ERROR", "errors": [str(exc)]}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
