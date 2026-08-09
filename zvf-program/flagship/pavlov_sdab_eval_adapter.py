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
    identity = _source_identity(spec)
    source_revision = _immutable_string(spec.get("source_revision"), "boundary.source_revision")
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
    parser.add_argument("boundary", type=Path, help="offline SDAB boundary JSON")
    parser.add_argument("--result", type=Path, help="optional offline backend result receipt JSON")
    parser.add_argument("--backend", choices=RESULT_BACKENDS)
    args = parser.parse_args(argv)
    try:
        boundary = build_sdab_boundary(_load_json(args.boundary, "boundary"))
        output: dict[str, Any] = {"boundary": boundary}
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
