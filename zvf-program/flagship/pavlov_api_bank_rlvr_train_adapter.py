#!/usr/bin/env python3
"""Define the offline T11 API-Bank RLVR training boundary.

This module is a metadata and receipt gate.  It does not download API-Bank,
contact a finance API, launch Tinker, write W&B/HF results, or infer a score.
The only task information that can leave the boundary is a deterministic
SHA-256 digest.  A train boundary is READY only when its source, license,
train task hashes, finance sandbox/native verifier, artifacts, result receipts,
and budget authorization are all pinned and mutually consistent.

API-Bank RLVR is deliberately distinct from the E4 BankerToolBench primary
evaluation.  A caller must provide the BankerToolBench held-out task-hash
exclusion receipt, and any overlap is a hard failure.  xLAM and other related
benchmarks are never accepted as substitutes for this source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pavlov-api-bank-rlvr-train-boundary-v1"
SUITE_ID = "api_bank_rlvr_train"
ROLE = "train"
TRAIN_ROLE = ROLE
SPLIT_DESCRIPTION = "train-only API-Bank RLVR episodes"
DATASET_ID = "Simu-Env/API-Bank-RLVR"
SOURCE_DATASET_ID = DATASET_ID
AUTHORITATIVE_DATASET_ID = DATASET_ID
API_BANK_DATASET_ID = DATASET_ID
API_BANK_RLVR_DATASET_ID = DATASET_ID
SOURCE_URL = "https://huggingface.co/datasets/Simu-Env/API-Bank-RLVR"
SOURCE_KIND = "huggingface_dataset"
EXPECTED_DOMAINS = ("enterprise", "finance", "long_horizon", "tool_use")
AUTHORITATIVE_SOURCE_IDENTITY = {
    "kind": SOURCE_KIND,
    "dataset_id": DATASET_ID,
    "url": SOURCE_URL,
}

# The T11 train split is disjoint from E4's primary evaluation split.  xLAM is
# a component preflight elsewhere in the portfolio, never a suite substitute.
EXCLUDED_SUITE_ID = "banker_toolbench_eval"
EXCLUDED_DATASET_ID = "handshake-ai-research/bankertoolbench"
E4_SUITE_ID = EXCLUDED_SUITE_ID
E4_DATASET_ID = EXCLUDED_DATASET_ID
BANKER_TOOLBENCH_SUITE_ID = EXCLUDED_SUITE_ID
BANKER_TOOLBENCH_DATASET_ID = EXCLUDED_DATASET_ID
XLAM_SUITE_ID = "pavlov_xlam"
XLAM_COMPONENT = XLAM_SUITE_ID
XLAM_DATASET_ID = "Salesforce/xlam-function-calling-60k"

_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
PINNED_REVISION = _PINNED_REVISION
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
TASK_DIGEST = _DIGEST
_DIGEST_TOKEN = re.compile(r"[0-9a-f]{64}")
_HEX_REVISION_IN_URI = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])")
_DIGEST_IN_URI = re.compile(r"(?:^|[@/#:])sha256:[0-9a-f]{64}(?:$|[/?#])", re.IGNORECASE)
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
    "main",
    "master",
    "head",
    "tip",
    "current",
    "descriptive",
    "not provided",
    "not_provided",
    "not available",
    "not_available",
    "n/a",
    "na",
    "to be pinned",
    "to_be_pinned_before_paid_runs",
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
    "mutable",
}
_RAW_CONTENT_KEYS = {"prompt", "prompts", "target", "targets"}

RESULT_RECEIPT_FIELDS = {
    "wandb": (
        "run_id",
        "entity",
        "project",
        "run_url",
        "summary_sha256",
        "history_sha256",
        "config_sha256",
    ),
    "tinker": (
        "job_id",
        "model_revision",
        "adapter_revision",
        "sampling_receipt_sha256",
        "result_receipt_sha256",
    ),
    "hf": (
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_revision",
        "artifact_url",
        "artifact_sha256",
        "result_receipt_sha256",
    ),
}
_RESULT_PROVIDER_ALIASES = {"huggingface": "hf", "hugging_face": "hf", "weights": "hf"}


class ApiBankRLVRTrainBoundaryError(ValueError):
    """Raised when a strict API-Bank RLVR train boundary cannot be built."""


# Friendly aliases used by adjacent local manifest callers.
ApiBankBoundaryError = ApiBankRLVRTrainBoundaryError
ApiBankRLVRBoundaryError = ApiBankRLVRTrainBoundaryError


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_id_hash(task_id: str) -> str:
    """Return a non-reversible digest for one authoritative task identifier."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise ApiBankRLVRTrainBoundaryError("task IDs must be non-empty strings")
    return _sha256(task_id.strip())


task_hash = task_id_hash


def split_manifest_hash(task_id_hashes: Sequence[str]) -> str:
    """Hash the canonical train role and ordered task-ID-hash list."""

    return _sha256(
        _canonical_json(
            {
                "suite_id": SUITE_ID,
                "role": ROLE,
                "task_id_hashes": list(task_id_hashes),
            }
        )
    )


def exclusion_manifest_hash(task_id_hashes: Sequence[str]) -> str:
    """Hash the E4 exclusion role and its ordered task-ID hashes."""

    return _sha256(
        _canonical_json(
            {
                "suite_id": EXCLUDED_SUITE_ID,
                "dataset_id": EXCLUDED_DATASET_ID,
                "role": "primary_eval",
                "task_id_hashes": list(task_id_hashes),
            }
        )
    )


def _read_json(value: str | Path) -> Any:
    try:
        return json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiBankRLVRTrainBoundaryError(f"cannot read JSON {value!s}: {exc}") from exc


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _assert_metadata_only(value: Any, path: str = "input") -> None:
    """Reject raw prompt/target fields recursively, including nested records."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_CONTENT_KEYS:
                raise ApiBankRLVRTrainBoundaryError(
                    f"{path} contains raw {str(key).lower()} content; boundary is metadata-only"
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
    return any(
        token in normalized
        for token in ("placeholder", "unrecorded", "not provided", "to be pinned")
    )


def _immutable_reference(value: Any) -> bool:
    """Accept only receipt references carrying an immutable digest/revision."""

    if _looks_placeholder(value) or not isinstance(value, str):
        return False
    reference = value.strip()
    lowered = reference.lower()
    if any(character.isspace() for character in reference):
        return False
    parts = {part for part in re.split(r"[/:?#._@-]+", lowered) if part}
    if parts.intersection(_MUTABLE_MARKERS):
        return False
    if re.fullmatch(r"sha256:[0-9a-f]{64}", lowered):
        return True
    if _DIGEST_IN_URI.search(lowered):
        return True
    # Local receipt URIs may bind to a bare SHA-256 or immutable commit.  A
    # descriptive URI without such a binding is deliberately not evidence.
    if lowered.startswith(("receipt://", "receipt:", "urn:receipt:")):
        tail = lowered.split(":", 1)[1]
        return bool(re.search(r"[0-9a-f]{64}", tail) or _HEX_REVISION_IN_URI.search(tail))
    if lowered.startswith(("hf://", "oci://", "git://")):
        return bool(_DIGEST_TOKEN.search(lowered) or _HEX_REVISION_IN_URI.search(lowered))
    if lowered.startswith(("http://", "https://")):
        return bool(_DIGEST_TOKEN.search(lowered) or _HEX_REVISION_IN_URI.search(lowered))
    return False


def _contains_substitute(value: Any, path: str = "input") -> None:
    """Reject E4/xLAM identity metadata when supplied as a purported source."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {
                "suite_id",
                "dataset_id",
                "source_dataset_id",
                "source_id",
                "benchmark",
                "related_benchmark",
                "substitute_for",
            } and isinstance(child, str):
                lowered = child.lower()
                if any(
                    token in lowered
                    for token in (
                        EXCLUDED_SUITE_ID,
                        "bankertoolbench",
                        EXCLUDED_DATASET_ID.lower(),
                        "xlam",
                        "function-calling-60k",
                    )
                ):
                    raise ApiBankRLVRTrainBoundaryError(
                        f"{path}.{key} names E4 BankerToolBench or xLAM; it is not the authoritative API-Bank source"
                    )
            _contains_substitute(child, f"{path}.{key}")
    elif _is_sequence(value):
        for index, child in enumerate(value):
            _contains_substitute(child, f"{path}[{index}]")


def _require_exact_source(source_identity: Mapping[str, Any] | None) -> None:
    if source_identity is None:
        return
    if not isinstance(source_identity, Mapping):
        raise ApiBankRLVRTrainBoundaryError("source_identity must be a JSON object")
    _assert_metadata_only(source_identity, "source_identity")
    _contains_substitute(source_identity, "source_identity")
    expected = {"kind": SOURCE_KIND, "dataset_id": DATASET_ID, "url": SOURCE_URL}
    for key, expected_value in expected.items():
        if source_identity.get(key) != expected_value:
            raise ApiBankRLVRTrainBoundaryError(
                f"source_identity.{key} must identify the authoritative API-Bank RLVR source"
            )
    for key in ("substitute_for", "related_benchmark", "benchmark"):
        if source_identity.get(key):
            raise ApiBankRLVRTrainBoundaryError(
                "related benchmarks and substitutes are not the authoritative API-Bank source"
            )


def _normalise_hashes(
    values: Sequence[str] | None,
    *,
    label: str,
) -> tuple[list[str], list[str]]:
    if values is None:
        return [], [f"{label} are required"]
    if not _is_sequence(values):
        return [], [f"{label} must be a list of SHA-256 digests"]
    raw = list(values)
    if not raw:
        return [], [f"{label} must be non-empty"]
    if any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in raw):
        return [], [f"{label} must contain lower-case SHA-256 digests"]
    errors: list[str] = []
    if len(set(raw)) != len(raw):
        errors.append(f"{label} contain duplicates")
    return sorted(raw), errors


def _extract_task_hashes(
    tasks: Iterable[Any] | Mapping[str, Any] | str | Path | None,
    task_id_hashes: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if tasks is not None:
        _assert_metadata_only(tasks, "tasks")
        _contains_substitute(tasks, "tasks")
    if tasks is not None and task_id_hashes is not None:
        return [], ["supply tasks or task_id_hashes, not both"]
    if isinstance(tasks, (str, Path)):
        tasks = _read_json(tasks)
        _assert_metadata_only(tasks, "tasks")
        _contains_substitute(tasks, "tasks")
    if task_id_hashes is not None:
        return _normalise_hashes(task_id_hashes, label="task_id_hashes")
    if tasks is None:
        return [], ["authoritative train task IDs are required"]
    if isinstance(tasks, Mapping):
        if "task_ids" in tasks:
            tasks = tasks["task_ids"]
        elif "tasks" in tasks:
            tasks = tasks["tasks"]
        else:
            tasks = list(tasks.values())
    if not _is_sequence(tasks):
        return [], ["tasks must be a list of metadata records or task IDs"]
    ids: list[str] = []
    for index, item in enumerate(tasks):
        if isinstance(item, str):
            value: Any = item.strip()
        elif isinstance(item, Mapping):
            value = item.get("task_id", item.get("id"))
            value = value.strip() if isinstance(value, str) else value
        else:
            value = None
        if not isinstance(value, str) or not value:
            errors.append(f"task at index {index} is missing a deterministic task_id")
            continue
        ids.append(value)
    if not ids:
        errors.append("authoritative train task IDs are required")
        return [], errors
    if len(ids) != len(set(ids)):
        errors.append("duplicate authoritative train task ID")
    ids = sorted(ids)
    return [task_id_hash(value) for value in ids], errors


def _extract_exclusion_hashes(
    *,
    heldout_task_id_hashes: Sequence[str] | None,
    heldout_task_hashes: Sequence[str] | None,
    held_out_task_id_hashes: Sequence[str] | None,
    excluded_task_id_hashes: Sequence[str] | None,
    excluded_task_hashes: Sequence[str] | None,
    banker_toolbench_task_id_hashes: Sequence[str] | None,
    banker_toolbench_task_hashes: Sequence[str] | None,
    exclusion_manifest: Mapping[str, Any] | None,
) -> tuple[list[str], str | None, list[str]]:
    """Collect one canonical, receipt-backed E4 exclusion hash list."""

    errors: list[str] = []
    candidates: list[tuple[str, Sequence[str]]] = []
    for name, value in (
        ("heldout_task_id_hashes", heldout_task_id_hashes),
        ("heldout_task_hashes", heldout_task_hashes),
        ("held_out_task_id_hashes", held_out_task_id_hashes),
        ("excluded_task_id_hashes", excluded_task_id_hashes),
        ("excluded_task_hashes", excluded_task_hashes),
        ("banker_toolbench_task_id_hashes", banker_toolbench_task_id_hashes),
        ("banker_toolbench_task_hashes", banker_toolbench_task_hashes),
    ):
        if value is not None:
            candidates.append((name, value))

    manifest_receipt: str | None = None
    if exclusion_manifest is not None:
        if not isinstance(exclusion_manifest, Mapping):
            errors.append("exclusion_manifest must be a JSON object")
        else:
            _assert_metadata_only(exclusion_manifest, "exclusion_manifest")
            # The exclusion manifest is expected to name E4 BankerToolBench;
            # it is proof of disjointness, not an API-Bank source substitute.
            if exclusion_manifest.get("suite_id") not in (None, EXCLUDED_SUITE_ID):
                errors.append("exclusion_manifest must identify banker_toolbench_eval")
            if exclusion_manifest.get("dataset_id") not in (None, EXCLUDED_DATASET_ID):
                errors.append("exclusion_manifest must identify BankerToolBench dataset")
            if exclusion_manifest.get("role") not in (None, "primary_eval"):
                errors.append("exclusion_manifest role must remain primary_eval")
            manifest_values = exclusion_manifest.get(
                "task_id_hashes", exclusion_manifest.get("task_hashes")
            )
            if manifest_values is not None:
                candidates.append(("exclusion_manifest.task_id_hashes", manifest_values))
            manifest_receipt = exclusion_manifest.get(
                "receipt_ref",
                exclusion_manifest.get(
                    "heldout_exclusion_receipt_ref",
                    exclusion_manifest.get(
                        "held_out_receipt_ref", exclusion_manifest.get("heldout_receipt_ref")
                    ),
                ),
            )
            held_out = exclusion_manifest.get("held_out")
            if manifest_receipt is None and isinstance(held_out, Mapping):
                manifest_receipt = held_out.get("receipt_ref")

    if not candidates:
        return [], manifest_receipt, [
            "held-out BankerToolBench task hashes are required to prove disjointness"
        ] + errors
    first = list(candidates[0][1]) if _is_sequence(candidates[0][1]) else candidates[0][1]
    if not isinstance(first, list):
        return [], manifest_receipt, ["excluded task hashes must be a list of SHA-256 digests"] + errors
    for name, value in candidates[1:]:
        other = list(value) if _is_sequence(value) else value
        comparable_first = sorted(first) if all(isinstance(item, str) for item in first) else first
        comparable_other = sorted(other) if isinstance(other, list) and all(isinstance(item, str) for item in other) else other
        if comparable_other != comparable_first:
            errors.append(f"conflicting exclusion hash lists ({candidates[0][0]} and {name})")
    hashes, hash_errors = _normalise_hashes(first, label="excluded BankerToolBench task hashes")
    errors.extend(hash_errors)
    return hashes, manifest_receipt, errors


def _validate_environment(contract: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(contract, Mapping):
        return {}, ["environment contract is required"]
    _assert_metadata_only(contract, "environment")
    normalized = dict(contract)
    errors: list[str] = []
    for key in (
        "environment_id",
        "environment_revision",
        "native",
        "stateful",
        "sandboxed",
        "network_access",
        "reset_protocol",
    ):
        if key not in contract or contract[key] in (None, "", []):
            errors.append(f"environment contract is missing {key}")
    revision = contract.get("environment_revision")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        errors.append("environment_revision must be an immutable 40-character SHA")
    if contract.get("native") is not True:
        errors.append("environment contract must declare native=true")
    if contract.get("stateful") is not True:
        errors.append("environment contract must declare stateful=true")
    if contract.get("sandboxed") is not True:
        errors.append("environment contract must declare sandboxed=true")
    if contract.get("network_access") is not False:
        errors.append("finance API sandbox must declare network_access=false")
    api_surface = contract.get("api_surface", contract.get("tool_api"))
    if not isinstance(api_surface, str) or not api_surface.strip():
        errors.append("environment contract must name a finance API sandbox surface")
    elif "finance" not in api_surface.lower() and "api" not in api_surface.lower():
        errors.append("environment api_surface must identify the finance API sandbox")
    if "finance_api_sandbox" in contract and contract.get("finance_api_sandbox") is not True:
        errors.append("environment contract must declare finance_api_sandbox=true")
    return normalized, errors


def _validate_artifact(contract: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(contract, Mapping):
        return {}, ["artifact contract is required"]
    _assert_metadata_only(contract, "artifact")
    normalized = dict(contract)
    errors: list[str] = []
    for key in ("required", "artifact_types", "artifact_receipt_ref", "side_effect_receipt_required"):
        if key not in contract or contract[key] in (None, "", []):
            errors.append(f"artifact contract is missing {key}")
    if contract.get("required") is not True:
        errors.append("artifact contract must declare required=true")
    if not _is_sequence(contract.get("artifact_types")) or not contract.get("artifact_types"):
        errors.append("artifact_types must be a non-empty list")
    if not _immutable_reference(contract.get("artifact_receipt_ref")):
        errors.append("artifact_receipt_ref must be immutable and non-placeholder")
    if contract.get("side_effect_receipt_required") is not True:
        errors.append("artifact contract must require side_effect_receipt_required=true")
    return normalized, errors


def _validate_verifier(contract: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(contract, Mapping):
        return {}, ["verifier contract is required"]
    _assert_metadata_only(contract, "verifier")
    normalized = dict(contract)
    errors: list[str] = []
    for key in (
        "verifier_id",
        "verifier_revision",
        "deterministic",
        "checks",
        "finance_api_sandbox",
        "verifier_receipt_ref",
    ):
        if key not in contract or contract[key] in (None, "", []):
            errors.append(f"verifier contract is missing {key}")
    revision = contract.get("verifier_revision")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        errors.append("verifier_revision must be an immutable 40-character SHA")
    if contract.get("deterministic") is not True:
        errors.append("verifier contract must declare deterministic=true")
    if contract.get("finance_api_sandbox") is not True:
        errors.append("verifier contract must declare finance_api_sandbox=true")
    if not _is_sequence(contract.get("checks")) or not contract.get("checks"):
        errors.append("verifier checks must be a non-empty list")
    native = contract.get("native", contract.get("native_verifier"))
    identifier = str(contract.get("verifier_id", "")).lower()
    if native is False or contract.get("native_verifier") is False:
        errors.append("verifier contract must identify a native verifier")
    elif native is not True and "native" not in identifier:
        errors.append("verifier contract must identify a native verifier")
    if not _immutable_reference(contract.get("verifier_receipt_ref")):
        errors.append("verifier_receipt_ref must be immutable and non-placeholder")
    return normalized, errors


def _validate_budget(budget_gate: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(budget_gate, Mapping):
        return {
            "status": "UNRECORDED",
            "launch_authorized": False,
            "launches_any_job": False,
        }, ["budget gate is required; no paid job may launch"]
    _assert_metadata_only(budget_gate, "budget_gate")
    normalized = dict(budget_gate)
    errors: list[str] = []
    if budget_gate.get("launch_authorized") is True or budget_gate.get("launches_any_job") is True:
        errors.append("budget gate cannot authorize or launch a paid job from this adapter")
    if budget_gate.get("status") != "AUTHORIZED_TINKER_ONLY":
        errors.append("budget gate status must be AUTHORIZED_TINKER_ONLY")
    if budget_gate.get("provider") != "Tinker":
        errors.append("budget gate provider must be Tinker")
    if budget_gate.get("paid_jobs_may_launch") is not True:
        errors.append("budget gate must explicitly authorize paid_jobs_may_launch")
    for key in ("maximum_usd", "operational_cap_usd", "safety_reserve_usd", "authorized_at"):
        if key not in budget_gate or budget_gate[key] in (None, "", []):
            errors.append(f"budget gate is missing {key}")
    maximum = budget_gate.get("maximum_usd")
    operational = budget_gate.get("operational_cap_usd")
    reserve = budget_gate.get("safety_reserve_usd")
    if not isinstance(maximum, (int, float)) or isinstance(maximum, bool) or maximum <= 0:
        errors.append("budget maximum_usd must be positive")
    elif maximum > 18.0:
        errors.append("budget maximum_usd exceeds the authorized 18.0 USD cap")
    if not isinstance(operational, (int, float)) or isinstance(operational, bool) or operational < 0:
        errors.append("budget operational_cap_usd must be non-negative")
    if not isinstance(reserve, (int, float)) or isinstance(reserve, bool) or reserve < 0:
        errors.append("budget safety_reserve_usd must be non-negative")
    if isinstance(maximum, (int, float)) and isinstance(operational, (int, float)) and operational > maximum:
        errors.append("budget operational_cap_usd exceeds maximum_usd")
    if (
        isinstance(maximum, (int, float))
        and isinstance(operational, (int, float))
        and isinstance(reserve, (int, float))
        and operational + reserve > maximum
    ):
        errors.append("budget operational cap plus safety reserve exceeds maximum_usd")
    if _looks_placeholder(budget_gate.get("authorized_at")):
        errors.append("budget authorized_at is placeholder or unrecorded")
    normalized["launch_authorized"] = False
    normalized["launches_any_job"] = False
    return normalized, errors


def _result_receipt_fields(
    result_receipts: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    required_shape = {provider: list(fields) for provider, fields in RESULT_RECEIPT_FIELDS.items()}
    if result_receipts is None:
        return {
            "status": "UNRECORDED",
            "recorded": False,
            "required_providers": ["wandb", "tinker", "hf"],
            "required_fields": required_shape,
        }, ["W&B, Tinker, and HF result receipts are unrecorded"]
    if not isinstance(result_receipts, Mapping):
        return {"status": "INVALID", "recorded": False, "required_fields": required_shape}, [
            "result_receipts must be a JSON object"
        ]
    _assert_metadata_only(result_receipts, "result_receipts")
    errors: list[str] = []
    normalized: dict[str, Any] = {
        "status": "RECORDED",
        "recorded": True,
        "required_providers": ["wandb", "tinker", "hf"],
        "required_fields": required_shape,
        "providers": {},
    }
    for provider, required_fields in RESULT_RECEIPT_FIELDS.items():
        source = result_receipts.get(provider)
        if source is None:
            for alias, canonical in _RESULT_PROVIDER_ALIASES.items():
                if canonical == provider and alias in result_receipts:
                    source = result_receipts[alias]
                    break
        if not isinstance(source, Mapping):
            errors.append(f"missing {provider} result receipt")
            continue
        provider_value: dict[str, Any] = {}
        for field in required_fields:
            value = source.get(field)
            if value in (None, "") or _looks_placeholder(value):
                errors.append(f"{provider} result receipt missing {field}")
                continue
            if field.endswith("_sha256"):
                if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                    errors.append(f"{provider}.{field} must be a SHA-256 digest")
                    continue
            if field.endswith("_revision"):
                if not isinstance(value, str) or not _PINNED_REVISION.fullmatch(value):
                    errors.append(f"{provider}.{field} must be an immutable revision")
                    continue
            provider_value[field] = value if isinstance(value, (int, float, bool)) else str(value)
        if provider_value:
            normalized["providers"][provider] = provider_value
    if errors:
        normalized["status"] = "INVALID"
        normalized["recorded"] = False
    return normalized, errors


def _normalise_exclusion(
    hashes: Sequence[str],
    receipt_ref: str | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not hashes:
        errors.append("held-out BankerToolBench task hashes are required to prove disjointness")
    if not _immutable_reference(receipt_ref):
        errors.append("held-out exclusion receipt_ref must be immutable and non-placeholder")
    aggregate = _sha256("\n".join(hashes)) if hashes else None
    manifest = exclusion_manifest_hash(hashes) if hashes else None
    return {
        "suite_id": EXCLUDED_SUITE_ID,
        "role": "primary_eval",
        "dataset_id": EXCLUDED_DATASET_ID,
        "task_count": len(hashes),
        "task_id_hashes": list(hashes),
        "task_hashes": list(hashes),
        "task_id_aggregate_sha256": aggregate,
        "task_ids_sha256": aggregate,
        "split_manifest_sha256": manifest,
        "receipt_ref": receipt_ref.strip() if isinstance(receipt_ref, str) else receipt_ref,
        "receipt_proven": bool(hashes and _immutable_reference(receipt_ref)),
    }, errors


def _boundary_errors(boundary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if boundary.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the API-Bank RLVR train boundary schema")
    if boundary.get("boundary_status") != boundary.get("status"):
        errors.append("boundary_status must match status")
    expected_source = {"kind": SOURCE_KIND, "dataset_id": DATASET_ID, "url": SOURCE_URL}
    if boundary.get("source_identity") != expected_source:
        errors.append("source_identity is not the authoritative API-Bank RLVR source")
    if boundary.get("suite_id") != SUITE_ID:
        errors.append("suite_id must be api_bank_rlvr_train")
    if boundary.get("role") != ROLE:
        errors.append("API-Bank RLVR boundary role must be train")
    if boundary.get("split") != SPLIT_DESCRIPTION:
        errors.append("split must remain train-only API-Bank RLVR")
    if boundary.get("domains") != list(EXPECTED_DOMAINS):
        errors.append("domains must match the API-Bank RLVR contract")
    semantics = boundary.get("split_semantics")
    expected_semantics = {
        "role": ROLE,
        "train_only": True,
        "primary_eval": False,
        "held_out": False,
        "private": False,
    }
    if semantics != expected_semantics:
        errors.append("split_semantics must identify a train-only, non-held-out boundary")
    if boundary.get("held_out") not in (None, False):
        errors.append("train boundary cannot claim held-out/private semantics")
    if boundary.get("primary_eval") is True:
        errors.append("train boundary cannot claim primary_eval")
    revision = boundary.get("dataset_revision")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        errors.append("dataset_revision must be an immutable 40-character lower-case SHA")
    if "dataset_revision_receipt_ref" in boundary and boundary.get("dataset_revision_receipt_ref") is not None:
        if not _immutable_reference(boundary.get("dataset_revision_receipt_ref")):
            errors.append("dataset revision receipt_ref must be immutable and non-placeholder")
    license_info = boundary.get("license")
    if not isinstance(license_info, Mapping):
        errors.append("license metadata is missing")
    else:
        if _looks_placeholder(license_info.get("id")):
            errors.append("license id is placeholder or unrecorded")
        if not _immutable_reference(license_info.get("receipt_ref")):
            errors.append("license receipt_ref must be immutable and non-placeholder")

    task_hashes = boundary.get("task_id_hashes")
    valid_task_hashes: list[str] = []
    if not _is_sequence(task_hashes) or not task_hashes:
        errors.append("task_id_hashes must be a non-empty ordered list")
    else:
        valid_task_hashes = [
            value for value in task_hashes if isinstance(value, str) and _DIGEST.fullmatch(value)
        ]
        if len(valid_task_hashes) != len(task_hashes):
            errors.append("task_id_hashes must contain lower-case SHA-256 digests")
        if len(set(valid_task_hashes)) != len(valid_task_hashes):
            errors.append("task_id_hashes contain duplicates")
        if len(valid_task_hashes) == len(task_hashes):
            if boundary.get("task_count") != len(task_hashes):
                errors.append("task_count does not match task_id_hashes")
            aggregate = _sha256("\n".join(task_hashes))
            if boundary.get("task_id_aggregate_sha256") != aggregate:
                errors.append("task_id_aggregate_sha256 does not match ordered task hashes")
            if boundary.get("task_ids_sha256") != aggregate:
                errors.append("task_ids_sha256 does not match ordered task hashes")
            if boundary.get("task_hashes") != list(task_hashes):
                errors.append("task_hashes does not match task_id_hashes")
            if boundary.get("split_manifest_sha256") != split_manifest_hash(task_hashes):
                errors.append("split_manifest_sha256 does not match the boundary")

    exclusion = boundary.get("exclusion_boundary")
    excluded_hashes: list[str] = []
    if not isinstance(exclusion, Mapping):
        errors.append("exclusion_boundary is required for BankerToolBench disjointness")
    else:
        if exclusion.get("suite_id") != EXCLUDED_SUITE_ID:
            errors.append("exclusion_boundary must identify banker_toolbench_eval")
        if exclusion.get("dataset_id") != EXCLUDED_DATASET_ID:
            errors.append("exclusion_boundary must identify BankerToolBench dataset")
        if exclusion.get("role") != "primary_eval":
            errors.append("BankerToolBench exclusion role must be primary_eval")
        raw_excluded = exclusion.get("task_id_hashes")
        if not _is_sequence(raw_excluded) or not raw_excluded:
            errors.append("excluded BankerToolBench task hashes are required")
        else:
            excluded_hashes = [
                value for value in raw_excluded if isinstance(value, str) and _DIGEST.fullmatch(value)
            ]
            if len(excluded_hashes) != len(raw_excluded):
                errors.append("excluded task hashes must contain lower-case SHA-256 digests")
            if len(set(excluded_hashes)) != len(excluded_hashes):
                errors.append("excluded task hashes contain duplicates")
            if len(excluded_hashes) == len(raw_excluded):
                if list(raw_excluded) != sorted(raw_excluded):
                    errors.append("excluded task hashes must be canonically sorted")
                aggregate = _sha256("\n".join(raw_excluded))
                if exclusion.get("task_count") != len(raw_excluded):
                    errors.append("excluded task_count does not match excluded hashes")
                if exclusion.get("task_hashes") != list(raw_excluded):
                    errors.append("excluded task_hashes does not match task_id_hashes")
                if exclusion.get("task_id_aggregate_sha256") != aggregate:
                    errors.append("excluded task_id_aggregate_sha256 does not match hashes")
                if exclusion.get("task_ids_sha256") != aggregate:
                    errors.append("excluded task_ids_sha256 does not match hashes")
                if exclusion.get("split_manifest_sha256") != exclusion_manifest_hash(raw_excluded):
                    errors.append("excluded split_manifest_sha256 does not match hashes")
        if not _immutable_reference(exclusion.get("receipt_ref")):
            errors.append("held-out exclusion receipt_ref must be immutable and non-placeholder")
        if exclusion.get("receipt_proven") is not True:
            errors.append("BankerToolBench exclusion receipt is not proven")
    if valid_task_hashes and excluded_hashes and set(valid_task_hashes).intersection(excluded_hashes):
        errors.append("API-Bank train task hashes overlap BankerToolBench held-out task IDs")
    if boundary.get("excluded_task_id_hashes") != excluded_hashes:
        errors.append("excluded_task_id_hashes does not match exclusion_boundary")
    if boundary.get("excluded_task_ids_sha256") != (
        _sha256("\n".join(excluded_hashes)) if excluded_hashes else None
    ):
        errors.append("excluded_task_ids_sha256 does not match exclusion hashes")
    expected_disjoint = {
        "suite_id": EXCLUDED_SUITE_ID,
        "dataset_id": EXCLUDED_DATASET_ID,
        "role": "primary_eval",
        "task_hashes_only": True,
        "exclusion_receipt_ref": exclusion.get("receipt_ref") if isinstance(exclusion, Mapping) else None,
    }
    if boundary.get("disjoint_from") != expected_disjoint:
        errors.append("disjoint_from metadata does not match BankerToolBench exclusion")
    if boundary.get("rejected_substitutes") != [EXCLUDED_SUITE_ID, XLAM_SUITE_ID]:
        errors.append("rejected_substitutes must include E4 BankerToolBench and xLAM")

    for name, validator in (
        ("environment_contract", _validate_environment),
        ("artifact_contract", _validate_artifact),
        ("verifier_contract", _validate_verifier),
    ):
        _, contract_errors = validator(boundary.get(name))
        errors.extend(contract_errors)
    budget, budget_errors = _validate_budget(boundary.get("budget_gate"))
    errors.extend(budget_errors)
    if boundary.get("launch_authorized") is not False or boundary.get("launches_any_job") is not False:
        errors.append("adapter must never authorize or launch a paid job")
    if boundary.get("budget_gate") != budget:
        # The builder adds the two explicit non-launch fields to the normalized
        # budget record; mutation of either must be visible to verification.
        errors.append("budget_gate metadata is not normalized")

    result = boundary.get("result_receipts")
    if not isinstance(result, Mapping):
        errors.append("result_receipts metadata is missing")
    elif result.get("recorded") is not True:
        errors.append("W&B, Tinker, and HF result receipts are unrecorded")
    else:
        if result.get("status") != "RECORDED":
            errors.append("result_receipts.status must be RECORDED when providers are recorded")
        expected_result_fields = {
            provider: list(fields) for provider, fields in RESULT_RECEIPT_FIELDS.items()
        }
        if result.get("required_providers") != ["wandb", "tinker", "hf"]:
            errors.append("result_receipts.required_providers metadata is incomplete")
        if result.get("required_fields") != expected_result_fields:
            errors.append("result_receipts.required_fields metadata is incomplete")
        providers = result.get("providers")
        _, result_errors = _result_receipt_fields(providers)
        errors.extend(result_errors)
        if isinstance(providers, Mapping):
            hf = providers.get("hf")
            if isinstance(hf, Mapping):
                if hf.get("dataset_id") != DATASET_ID:
                    errors.append("hf result receipt dataset_id must identify API-Bank RLVR")
                if hf.get("dataset_revision") != revision:
                    errors.append("hf result receipt dataset_revision must match dataset_revision")
    if boundary.get("result_claims") is not None:
        errors.append("result_claims must remain null until externally verified")

    expected_errors = sorted(set(errors))
    recorded_errors = boundary.get("errors")
    if recorded_errors != expected_errors:
        errors.append("errors field does not match boundary validation errors")
    return sorted(set(errors))


def build_api_bank_rlvr_train_boundary(
    revision: str,
    license_id: str,
    license_receipt_ref: str,
    tasks: Iterable[Any] | Mapping[str, Any] | str | Path | None = None,
    *,
    task_ids: Sequence[str] | None = None,
    task_id_hashes: Sequence[str] | None = None,
    heldout_task_id_hashes: Sequence[str] | None = None,
    heldout_task_hashes: Sequence[str] | None = None,
    held_out_task_id_hashes: Sequence[str] | None = None,
    excluded_task_id_hashes: Sequence[str] | None = None,
    excluded_task_hashes: Sequence[str] | None = None,
    banker_toolbench_task_id_hashes: Sequence[str] | None = None,
    banker_toolbench_task_hashes: Sequence[str] | None = None,
    exclusion_manifest: Mapping[str, Any] | None = None,
    banker_toolbench_manifest: Mapping[str, Any] | None = None,
    heldout_exclusion_receipt_ref: str | None = None,
    heldout_receipt_ref: str | None = None,
    held_out_receipt_ref: str | None = None,
    exclusion_receipt_ref: str | None = None,
    environment_contract: Mapping[str, Any] | None = None,
    artifact_contract: Mapping[str, Any] | None = None,
    verifier_contract: Mapping[str, Any] | None = None,
    budget_gate: Mapping[str, Any] | None = None,
    result_receipts: Mapping[str, Any] | None = None,
    revision_receipt_ref: str | None = None,
    dataset_revision_receipt_ref: str | None = None,
    source_identity: Mapping[str, Any] | None = None,
    role: str = ROLE,
    related_benchmark: str | None = None,
    substitute_for: str | None = None,
) -> dict[str, Any]:
    """Build T11 from caller-supplied metadata only; never launch training."""

    _require_exact_source(source_identity)
    if related_benchmark or substitute_for:
        raise ApiBankRLVRTrainBoundaryError(
            "related benchmarks and substitutes are not the authoritative API-Bank source"
        )
    if role != ROLE:
        raise ApiBankRLVRTrainBoundaryError("API-Bank RLVR boundary role must be train")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        raise ApiBankRLVRTrainBoundaryError(
            "revision must be an immutable 40-character lower-case commit SHA"
        )
    if _looks_placeholder(license_id):
        raise ApiBankRLVRTrainBoundaryError("license_id must be recorded, not a placeholder")
    if not _immutable_reference(license_receipt_ref):
        raise ApiBankRLVRTrainBoundaryError(
            "license_receipt_ref must be immutable and non-placeholder"
        )
    if (
        revision_receipt_ref is not None
        and dataset_revision_receipt_ref is not None
        and revision_receipt_ref != dataset_revision_receipt_ref
    ):
        raise ApiBankRLVRTrainBoundaryError("conflicting dataset revision receipt references")
    revision_receipt = dataset_revision_receipt_ref or revision_receipt_ref
    revision_receipt_errors: list[str] = []
    if revision_receipt is not None and not _immutable_reference(revision_receipt):
        revision_receipt_errors.append(
            "dataset revision receipt_ref must be immutable and non-placeholder"
        )
    if task_ids is not None:
        if tasks is not None or task_id_hashes is not None:
            raise ApiBankRLVRTrainBoundaryError("supply task_ids, tasks, or task_id_hashes, not multiple")
        tasks = task_ids

    if banker_toolbench_manifest is not None:
        if exclusion_manifest is not None and exclusion_manifest != banker_toolbench_manifest:
            raise ApiBankRLVRTrainBoundaryError("conflicting BankerToolBench exclusion manifests")
        exclusion_manifest = banker_toolbench_manifest

    exclusion_ref = heldout_exclusion_receipt_ref
    for alias_name, alias_value in (
        ("heldout_receipt_ref", heldout_receipt_ref),
        ("held_out_receipt_ref", held_out_receipt_ref),
    ):
        if alias_value is not None:
            if exclusion_ref is not None and exclusion_ref != alias_value:
                raise ApiBankRLVRTrainBoundaryError(
                    f"conflicting exclusion receipt references ({alias_name})"
                )
            exclusion_ref = alias_value
    if exclusion_receipt_ref is not None:
        if exclusion_ref is not None and exclusion_ref != exclusion_receipt_ref:
            raise ApiBankRLVRTrainBoundaryError("conflicting exclusion receipt references")
        exclusion_ref = exclusion_receipt_ref
    excluded_hashes, manifest_receipt, exclusion_errors = _extract_exclusion_hashes(
        heldout_task_id_hashes=heldout_task_id_hashes,
        heldout_task_hashes=heldout_task_hashes,
        held_out_task_id_hashes=held_out_task_id_hashes,
        excluded_task_id_hashes=excluded_task_id_hashes,
        excluded_task_hashes=excluded_task_hashes,
        banker_toolbench_task_id_hashes=banker_toolbench_task_id_hashes,
        banker_toolbench_task_hashes=banker_toolbench_task_hashes,
        exclusion_manifest=exclusion_manifest,
    )
    if exclusion_ref is None:
        exclusion_ref = manifest_receipt
    exclusion_receipt_conflict = (
        manifest_receipt is not None
        and exclusion_ref is not None
        and manifest_receipt != exclusion_ref
    )

    task_hash_values, task_errors = _extract_task_hashes(tasks, task_id_hashes)
    environment, environment_errors = _validate_environment(environment_contract)
    artifact, artifact_errors = _validate_artifact(artifact_contract)
    verifier, verifier_errors = _validate_verifier(verifier_contract)
    budget, budget_errors = _validate_budget(budget_gate)
    receipts, receipt_errors = _result_receipt_fields(result_receipts)
    exclusion, exclusion_ref_errors = _normalise_exclusion(excluded_hashes, exclusion_ref)
    errors = [
        *task_errors,
        *exclusion_errors,
        *exclusion_ref_errors,
        *environment_errors,
        *artifact_errors,
        *verifier_errors,
        *budget_errors,
        *receipt_errors,
        *revision_receipt_errors,
    ]
    if exclusion_receipt_conflict:
        errors.append("exclusion_manifest receipt conflicts with exclusion receipt")
    providers = receipts.get("providers")
    if isinstance(providers, Mapping):
        hf = providers.get("hf")
        if isinstance(hf, Mapping):
            if hf.get("dataset_id") != DATASET_ID:
                errors.append("hf result receipt dataset_id must identify API-Bank RLVR")
            if hf.get("dataset_revision") != revision:
                errors.append("hf result receipt dataset_revision must match dataset_revision")
    if task_hash_values and excluded_hashes and set(task_hash_values).intersection(excluded_hashes):
        errors.append("API-Bank train task hashes overlap BankerToolBench held-out task IDs")

    aggregate = _sha256("\n".join(task_hash_values)) if task_hash_values else None
    boundary = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not errors else "BLOCKED",
        "boundary_status": "READY" if not errors else "BLOCKED",
        "launch_authorized": False,
        "launches_any_job": False,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "split": SPLIT_DESCRIPTION,
        "split_semantics": {
            "role": ROLE,
            "train_only": True,
            "primary_eval": False,
            "held_out": False,
            "private": False,
        },
        "domains": list(EXPECTED_DOMAINS),
        "source_identity": {
            "kind": SOURCE_KIND,
            "dataset_id": DATASET_ID,
            "url": SOURCE_URL,
        },
        "dataset_revision": revision,
        "dataset_revision_receipt_ref": revision_receipt,
        "license": {"id": str(license_id), "receipt_ref": license_receipt_ref.strip()},
        "task_count": len(task_hash_values),
        "task_id_hashes": task_hash_values,
        "task_hashes": task_hash_values,
        "task_id_aggregate_sha256": aggregate,
        "task_ids_sha256": aggregate,
        "split_manifest_sha256": split_manifest_hash(task_hash_values) if task_hash_values else None,
        "exclusion_boundary": exclusion,
        "excluded_task_id_hashes": excluded_hashes,
        "excluded_task_ids_sha256": (
            _sha256("\n".join(excluded_hashes)) if excluded_hashes else None
        ),
        "disjoint_from": {
            "suite_id": EXCLUDED_SUITE_ID,
            "dataset_id": EXCLUDED_DATASET_ID,
            "role": "primary_eval",
            "task_hashes_only": True,
            "exclusion_receipt_ref": exclusion.get("receipt_ref"),
        },
        "rejected_substitutes": [EXCLUDED_SUITE_ID, XLAM_SUITE_ID],
        "environment_contract": environment,
        "artifact_contract": artifact,
        "verifier_contract": verifier,
        "budget_gate": budget,
        "result_receipts": receipts,
        "result_claims": None,
        "training_claims": None,
        "errors": sorted(set(errors)),
        "evidence_scope": "training boundary and receipts only; no paid run or result is asserted",
    }
    return boundary


# Common local manifest names.
build_boundary = build_api_bank_rlvr_train_boundary
build_manifest = build_api_bank_rlvr_train_boundary
generate_boundary = build_api_bank_rlvr_train_boundary
build_api_bank_manifest = build_api_bank_rlvr_train_boundary
build_api_bank_rlvr_manifest = build_api_bank_rlvr_train_boundary
build_api_bank_rlvr_train_manifest = build_api_bank_rlvr_train_boundary
build_api_bank_rlvr_train_adapter = build_api_bank_rlvr_train_boundary


def validate_boundary(boundary: Mapping[str, Any]) -> list[str]:
    if not isinstance(boundary, Mapping):
        return ["boundary must be a JSON object"]
    try:
        _assert_metadata_only(boundary)
    except ApiBankRLVRTrainBoundaryError as exc:
        return [str(exc)]
    try:
        return _boundary_errors(boundary)
    except (TypeError, ValueError, AttributeError) as exc:
        # Verification is a fail-closed diagnostic path: malformed mutation
        # yields a concrete error, never an accidental traceback or READY.
        return [f"malformed boundary metadata: {exc}"]


def verify_boundary(boundary: Mapping[str, Any] | str | Path) -> bool:
    if isinstance(boundary, (str, Path)):
        boundary = _read_json(boundary)
    errors = validate_boundary(boundary)
    if errors:
        raise ApiBankRLVRTrainBoundaryError(
            "invalid API-Bank RLVR train boundary: " + "; ".join(errors)
        )
    if boundary.get("status") != "READY":
        raise ApiBankRLVRTrainBoundaryError("API-Bank RLVR train boundary is blocked")
    return True


validate_manifest = validate_boundary
verify_manifest = verify_boundary
verify_api_bank_rlvr_train = verify_boundary
verify_api_bank_rlvr_train_boundary = verify_boundary
verify_api_bank_rlvr_manifest = verify_boundary
ApiBankRLVRTrainAdapterError = ApiBankRLVRTrainBoundaryError


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--license-id")
    parser.add_argument("--license-receipt-ref")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-id-hash", action="append")
    parser.add_argument("--heldout-task-id-hash", action="append")
    parser.add_argument("--heldout-exclusion-receipt-ref")
    parser.add_argument("--environment-contract", type=Path)
    parser.add_argument("--artifact-contract", type=Path)
    parser.add_argument("--verifier-contract", type=Path)
    parser.add_argument("--budget-gate", type=Path)
    parser.add_argument("--result-receipts", type=Path)
    parser.add_argument("--exclusion-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "verify":
            if args.boundary is None:
                parser.error("verify requires --boundary")
            verify_boundary(args.boundary)
            print(json.dumps({"verified": True, "boundary": str(args.boundary)}, sort_keys=True))
            return 0
        if not args.revision or not args.license_id or not args.license_receipt_ref:
            parser.error("generate requires --revision, --license-id, and --license-receipt-ref")
        environment = _read_json(args.environment_contract) if args.environment_contract else None
        artifact = _read_json(args.artifact_contract) if args.artifact_contract else None
        verifier = _read_json(args.verifier_contract) if args.verifier_contract else None
        budget = _read_json(args.budget_gate) if args.budget_gate else None
        results = _read_json(args.result_receipts) if args.result_receipts else None
        exclusion_manifest = _read_json(args.exclusion_manifest) if args.exclusion_manifest else None
        boundary = build_api_bank_rlvr_train_boundary(
            args.revision,
            args.license_id,
            args.license_receipt_ref,
            args.task_id if args.task_id else None,
            task_id_hashes=args.task_id_hash,
            heldout_task_id_hashes=args.heldout_task_id_hash,
            heldout_exclusion_receipt_ref=args.heldout_exclusion_receipt_ref,
            exclusion_manifest=exclusion_manifest,
            environment_contract=environment,
            artifact_contract=artifact,
            verifier_contract=verifier,
            budget_gate=budget,
            result_receipts=results,
        )
        _write_json(boundary, args.out)
        return 0 if boundary["status"] == "READY" else 2
    except (ApiBankRLVRTrainBoundaryError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
