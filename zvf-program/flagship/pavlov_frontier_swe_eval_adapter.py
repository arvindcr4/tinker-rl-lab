#!/usr/bin/env python3
"""Offline FrontierSWE E2 boundary and result-receipt contract.

This module deliberately does not run FrontierSWE, launch Tinker, initialize
W&B, upload to Hugging Face, or fetch benchmark tasks.  It makes the exact
boundary explicit and validates locally supplied manifests and receipts.  The
source is a primary-evaluation component; a held-out claim is admissible only
when an immutable exclusion receipt is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITE_ID = "frontier_swe_eval"
BENCHMARK_ID = "frontierswe"
BENCHMARK_FAMILY = "FrontierSWE"
DOMAINS = ("code", "ml", "long_horizon")

# The repository and commit were read from the official GitHub repository
# metadata/refs.  The repository metadata reports no declared license, so no
# license identifier is invented here; a boundary must carry an explicit,
# content-hashed license receipt before it can pass.
AUTHORITATIVE_REPOSITORY = "Proximal-Labs/frontier-swe"
AUTHORITATIVE_SOURCE_URL = "https://github.com/Proximal-Labs/frontier-swe"
OFFICIAL_METADATA_URL = "https://api.github.com/repos/Proximal-Labs/frontier-swe"
AUTHORITATIVE_DEFAULT_BRANCH = "main"
PINNED_SOURCE_REVISION = "422b9bb95deb8efe436becb0ed3c44be23611e10"
OFFICIAL_LICENSE_IDENTIFIER: str | None = None
OFFICIAL_LICENSE_STATUS = "not_declared_by_official_metadata"

PRIMARY_EVAL = "primary_eval"
RECEIPT_PROVEN_HELDOUT = "receipt_proven_heldout"
PRIMARY_EVAL_CLAIM = (
    "FrontierSWE is primary_eval only; held-out status is not claimed without "
    "an immutable receipt."
)

BOUNDARY_SCHEMA_VERSION = "pavlov-frontier-swe-boundary-v1"
RECEIPT_SCHEMA_VERSION = "pavlov-frontier-swe-result-receipt-v1"
NATIVE_ENVIRONMENT_ID = "frontier_swe_native"
NATIVE_ARTIFACT_MODE = "native_artifact_or_side_effect"
NATIVE_VERIFIER_ID = "native_frontier_swe_verifier"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$", re.IGNORECASE)
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_URL_PREFIXES = ("https://", "http://")
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "to_be_pinned",
    "to-be-pinned",
    "tbd",
    "todo",
    "changeme",
    "replace_me",
    "replace-me",
)
_SUBSTITUTE_MARKERS = ("xlam", "swebench", "swe-bench", "swe_bench")
_RAW_RESULT_KEYS = {"raw_response", "response_text", "completion", "trajectory"}


class FrontierSWEEvaluationError(ValueError):
    """Raised by require_* helpers when an E2 contract is inadmissible."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_canonical(value: Any) -> str:
    """Hash JSON with stable key/order/number rules."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    lowered = value.strip().lower()
    if not lowered or lowered in {"none", "null", "na", "n/a", "unknown", "unset"}:
        return True
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True
    candidate = lowered.removeprefix("sha256:")
    return len(candidate) in {40, 64} and set(candidate) == {"0"}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _text(value: Any, path: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or _is_placeholder(value):
        errors.append(f"{path} must be a non-placeholder string")
        return None
    return value.strip()


def _sha256(value: Any, path: str, errors: list[str]) -> str | None:
    text = _text(value, path, errors)
    if text is not None and not _SHA256_RE.fullmatch(text):
        errors.append(f"{path} must be a 64-hex SHA256")
        return text
    return text.lower() if text is not None else None


def _digest(value: Any, path: str, errors: list[str]) -> str | None:
    text = _text(value, path, errors)
    if text is not None and not _DIGEST_RE.fullmatch(text):
        errors.append(f"{path} must be sha256:<64-hex>")
        return text
    return text.lower() if text is not None else None


def _revision(value: Any, path: str, errors: list[str]) -> str | None:
    text = _text(value, path, errors)
    if text is not None and not _REVISION_RE.fullmatch(text):
        errors.append(f"{path} must be a 40-hex immutable Git revision")
        return text
    return text.lower() if text is not None else None


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _same_json(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _reject_raw_result_fields(value: Any, path: str, errors: list[str]) -> None:
    """Reject raw model responses wherever they appear in a receipt payload."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = key.lower() if isinstance(key, str) else ""
            if key_text in _RAW_RESULT_KEYS:
                errors.append(f"{path}.{key_text} must not contain raw result data")
            _reject_raw_result_fields(nested, f"{path}.{key_text or '<key>'}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_raw_result_fields(nested, f"{path}[{index}]", errors)


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _validate_task_ids(task_ids: Any, path: str, errors: list[str]) -> list[str]:
    if not isinstance(task_ids, list) or not task_ids:
        errors.append(f"{path} must be a non-empty ordered list")
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for index, task_id in enumerate(task_ids):
        if not isinstance(task_id, str) or _is_placeholder(task_id) or task_id.strip() != task_id:
            errors.append(f"{path}[{index}] must be a non-placeholder task ID")
            continue
        if task_id in seen:
            errors.append(f"{path} contains duplicate task ID {task_id!r}")
        seen.add(task_id)
        normalized.append(task_id)
    return normalized


def deterministic_split_manifest(
    task_ids: Sequence[str],
    split_id: str,
    *,
    source_revision: str = PINNED_SOURCE_REVISION,
) -> dict[str, Any]:
    """Build the deterministic task-ID and split-manifest hashes."""

    errors: list[str] = []
    normalized = _validate_task_ids(list(task_ids), "task_ids", errors)
    normalized_split_id = _text(split_id, "split_id", errors)
    if normalized_split_id is not None and normalized_split_id != split_id:
        errors.append("split_id must not have leading or trailing whitespace")
    revision = _revision(source_revision, "source_revision", errors)
    if errors:
        raise FrontierSWEEvaluationError("; ".join(errors))
    task_ids_sha256 = sha256_canonical(normalized)
    manifest_core = {
        "source_repo": AUTHORITATIVE_REPOSITORY,
        "source_revision": revision,
        "split_id": normalized_split_id,
        "task_count": len(normalized),
        "task_ids": normalized,
        "task_ids_sha256": task_ids_sha256,
    }
    return {
        **manifest_core,
        "split_manifest_sha256": sha256_canonical(manifest_core),
    }


def native_environment_contract(
    *,
    environment_digest: str,
    runner_entrypoint: str,
    network_policy: str = "declared_native_only",
) -> dict[str, Any]:
    """Return the required native environment contract without running it."""

    return {
        "backend_id": NATIVE_ENVIRONMENT_ID,
        "source_repo": AUTHORITATIVE_REPOSITORY,
        "source_revision": PINNED_SOURCE_REVISION,
        "task_id_namespace": "frontier-swe",
        "stateful": True,
        "artifact_or_side_effect": True,
        "environment_digest": environment_digest,
        "runner_entrypoint": runner_entrypoint,
        "network_policy": network_policy,
    }


def native_artifact_contract(
    *, manifest_sha256: str, paths: Sequence[str]
) -> dict[str, Any]:
    return {
        "required": True,
        "mode": NATIVE_ARTIFACT_MODE,
        "manifest_sha256": manifest_sha256,
        "paths": list(paths),
        "verifier_reads_artifact": True,
    }


def native_verifier_contract(
    *, verifier_sha256: str, entrypoint: str
) -> dict[str, Any]:
    return {
        "verifier_id": NATIVE_VERIFIER_ID,
        "entrypoint": entrypoint,
        "source_sha256": verifier_sha256,
        "checks_environment_state": True,
        "checks_artifacts": True,
    }


def build_boundary(
    *,
    task_ids: Sequence[str],
    split_id: str,
    environment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    verifier: Mapping[str, Any],
    license_identifier: str | None = None,
    license_receipt: Mapping[str, Any] | None = None,
    source_revision: str = PINNED_SOURCE_REVISION,
    evaluation_role: str = PRIMARY_EVAL,
    heldout_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a JSON boundary; callers still must run :func:`validate_boundary`."""

    split = deterministic_split_manifest(
        task_ids, split_id, source_revision=source_revision
    )
    claim = {
        "evaluation_role": evaluation_role,
        "heldout_status": (
            "receipt_proven" if evaluation_role == RECEIPT_PROVEN_HELDOUT else "not_proven"
        ),
        "claim_text": PRIMARY_EVAL_CLAIM,
        "heldout_proof": dict(heldout_proof) if heldout_proof is not None else None,
    }
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_family": BENCHMARK_FAMILY,
        "domains": list(DOMAINS),
        "source": {
            "repository": AUTHORITATIVE_REPOSITORY,
            "url": AUTHORITATIVE_SOURCE_URL,
            "official_metadata_url": OFFICIAL_METADATA_URL,
            "default_branch": AUTHORITATIVE_DEFAULT_BRANCH,
            "authoritative": True,
            "revision": source_revision,
            "revision_kind": "git_commit",
            "license_status": OFFICIAL_LICENSE_STATUS,
            "license_identifier": license_identifier,
            "license_receipt": dict(license_receipt) if license_receipt is not None else None,
        },
        "split": split,
        "environment": dict(environment),
        "artifact": dict(artifact),
        "verifier": dict(verifier),
        "claim": claim,
    }


def _validate_license(source: Mapping[str, Any], errors: list[str]) -> None:
    identifier = _text(
        source.get("license_identifier"), "source.license_identifier", errors
    )
    receipt = _mapping(source.get("license_receipt"), "source.license_receipt", errors)
    if receipt is None:
        return
    if receipt.get("status") != "verified":
        errors.append("source.license_receipt.status must be 'verified'")
    if identifier is not None and receipt.get("identifier") != identifier:
        errors.append("source.license_receipt.identifier must match license_identifier")
    url = receipt.get("source_url")
    if not isinstance(url, str) or not url.startswith(_URL_PREFIXES):
        errors.append("source.license_receipt.source_url must be http(s)")
    elif not (
        url == OFFICIAL_METADATA_URL
        or url.startswith(f"{AUTHORITATIVE_SOURCE_URL}/")
    ):
        errors.append("source.license_receipt.source_url must be an official FrontierSWE URL")
    _sha256(receipt.get("content_sha256"), "source.license_receipt.content_sha256", errors)
    _text(receipt.get("observed_at"), "source.license_receipt.observed_at", errors)


def _validate_source(boundary: Mapping[str, Any], errors: list[str]) -> Mapping[str, Any] | None:
    source = _mapping(boundary.get("source"), "source", errors)
    if source is None:
        return None
    expected = {
        "repository": AUTHORITATIVE_REPOSITORY,
        "url": AUTHORITATIVE_SOURCE_URL,
        "official_metadata_url": OFFICIAL_METADATA_URL,
        "default_branch": AUTHORITATIVE_DEFAULT_BRANCH,
        "authoritative": True,
        "revision_kind": "git_commit",
        "license_status": OFFICIAL_LICENSE_STATUS,
    }
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            errors.append(f"source.{key} does not match the authoritative FrontierSWE source")
    revision = _revision(source.get("revision"), "source.revision", errors)
    if revision is not None and revision != PINNED_SOURCE_REVISION:
        errors.append("source.revision is not the pinned official FrontierSWE revision")
    _validate_license(source, errors)
    return source


def _validate_split(boundary: Mapping[str, Any], source: Mapping[str, Any], errors: list[str]) -> Mapping[str, Any] | None:
    split = _mapping(boundary.get("split"), "split", errors)
    if split is None:
        return None
    task_ids = _validate_task_ids(split.get("task_ids"), "split.task_ids", errors)
    split_id = _text(split.get("split_id"), "split.split_id", errors)
    revision = source.get("revision")
    if split.get("source_repo") != AUTHORITATIVE_REPOSITORY:
        errors.append("split.source_repo must be the authoritative FrontierSWE repository")
    if split.get("source_revision") != revision:
        errors.append("split.source_revision must match source.revision")
    if split.get("task_count") != len(task_ids):
        errors.append("split.task_count must equal the deterministic task-ID count")
    expected_task_hash = sha256_canonical(task_ids) if task_ids else None
    if split.get("task_ids_sha256") != expected_task_hash:
        errors.append("split.task_ids_sha256 does not match the ordered task IDs")
    core = {
        "source_repo": split.get("source_repo"),
        "source_revision": split.get("source_revision"),
        "split_id": split_id,
        "task_count": split.get("task_count"),
        "task_ids": task_ids,
        "task_ids_sha256": split.get("task_ids_sha256"),
    }
    if split.get("split_manifest_sha256") != sha256_canonical(core):
        errors.append("split.split_manifest_sha256 does not match the immutable manifest")
    _sha256(split.get("task_ids_sha256"), "split.task_ids_sha256", errors)
    _sha256(split.get("split_manifest_sha256"), "split.split_manifest_sha256", errors)
    return split


def _validate_native_contracts(boundary: Mapping[str, Any], errors: list[str]) -> None:
    environment = _mapping(boundary.get("environment"), "environment", errors)
    if environment is not None:
        source = boundary.get("source")
        source_revision = source.get("revision") if isinstance(source, Mapping) else None
        expected = {
            "backend_id": NATIVE_ENVIRONMENT_ID,
            "source_repo": AUTHORITATIVE_REPOSITORY,
            "source_revision": source_revision,
            "task_id_namespace": "frontier-swe",
            "stateful": True,
            "artifact_or_side_effect": True,
        }
        for key, expected_value in expected.items():
            if environment.get(key) != expected_value:
                errors.append(f"environment.{key} is not the native FrontierSWE contract")
        _digest(environment.get("environment_digest"), "environment.environment_digest", errors)
        _text(environment.get("runner_entrypoint"), "environment.runner_entrypoint", errors)
        _text(environment.get("network_policy"), "environment.network_policy", errors)

    artifact = _mapping(boundary.get("artifact"), "artifact", errors)
    if artifact is not None:
        if artifact.get("required") is not True:
            errors.append("artifact.required must be true")
        if artifact.get("mode") != NATIVE_ARTIFACT_MODE:
            errors.append("artifact.mode must require native artifacts or side effects")
        _sha256(artifact.get("manifest_sha256"), "artifact.manifest_sha256", errors)
        paths = artifact.get("paths")
        if not isinstance(paths, list) or not paths or not all(
            isinstance(path, str) and path.strip() for path in paths
        ):
            errors.append("artifact.paths must be a non-empty list of paths")
        if artifact.get("verifier_reads_artifact") is not True:
            errors.append("artifact.verifier_reads_artifact must be true")

    verifier = _mapping(boundary.get("verifier"), "verifier", errors)
    if verifier is not None:
        if verifier.get("verifier_id") != NATIVE_VERIFIER_ID:
            errors.append("verifier.verifier_id must be the native FrontierSWE verifier")
        _text(verifier.get("entrypoint"), "verifier.entrypoint", errors)
        _sha256(verifier.get("source_sha256"), "verifier.source_sha256", errors)
        for key in ("checks_environment_state", "checks_artifacts"):
            if verifier.get(key) is not True:
                errors.append(f"verifier.{key} must be true")


def _validate_claim(boundary: Mapping[str, Any], split: Mapping[str, Any] | None, errors: list[str]) -> None:
    claim = _mapping(boundary.get("claim"), "claim", errors)
    if claim is None:
        return
    role = claim.get("evaluation_role")
    if role not in {PRIMARY_EVAL, RECEIPT_PROVEN_HELDOUT}:
        errors.append("claim.evaluation_role must be primary_eval or receipt_proven_heldout")
    expected_status = "receipt_proven" if role == RECEIPT_PROVEN_HELDOUT else "not_proven"
    if claim.get("heldout_status") != expected_status:
        errors.append("claim.heldout_status does not match evaluation_role")
    claim_text = claim.get("claim_text")
    if not isinstance(claim_text, str) or not claim_text.strip():
        errors.append("claim.claim_text must be non-empty")
    elif role == PRIMARY_EVAL and claim_text != PRIMARY_EVAL_CLAIM:
        errors.append("primary_eval claim text must not overclaim held-out evidence")
    proof = claim.get("heldout_proof")
    if role == PRIMARY_EVAL:
        if proof not in (None, {}):
            errors.append("primary_eval cannot carry a held-out proof")
        return
    proof_mapping = _mapping(proof, "claim.heldout_proof", errors)
    if proof_mapping is None or split is None:
        return
    if proof_mapping.get("status") != "verified":
        errors.append("claim.heldout_proof.status must be 'verified'")
    for key in ("task_ids_sha256", "split_manifest_sha256"):
        if proof_mapping.get(key) != split.get(key):
            errors.append(f"claim.heldout_proof.{key} must match the immutable split")
    _sha256(
        proof_mapping.get("training_exclusion_sha256"),
        "claim.heldout_proof.training_exclusion_sha256",
        errors,
    )
    _sha256(
        proof_mapping.get("proof_artifact_sha256"),
        "claim.heldout_proof.proof_artifact_sha256",
        errors,
    )
    _text(
        proof_mapping.get("decontamination_receipt"),
        "claim.heldout_proof.decontamination_receipt",
        errors,
    )


def _reject_substitutes(boundary: Mapping[str, Any], errors: list[str]) -> None:
    exact_values = (
        boundary.get("suite_id"),
        boundary.get("benchmark_id"),
        boundary.get("benchmark_family"),
        boundary.get("source", {}).get("repository")
        if isinstance(boundary.get("source"), Mapping)
        else None,
    )
    if boundary.get("suite_id") != SUITE_ID:
        errors.append("suite_id must be frontier_swe_eval; related benchmarks and xLAM are rejected")
    if boundary.get("benchmark_id") != BENCHMARK_ID:
        errors.append("benchmark_id must be frontierswe; related benchmarks and xLAM are rejected")
    if boundary.get("benchmark_family") != BENCHMARK_FAMILY:
        errors.append("benchmark_family must be FrontierSWE")
    for value in exact_values:
        if isinstance(value, str) and any(marker in value.lower() for marker in _SUBSTITUTE_MARKERS):
            if value not in {AUTHORITATIVE_REPOSITORY, AUTHORITATIVE_SOURCE_URL}:
                errors.append("related benchmark or xLAM identity cannot substitute for FrontierSWE")
    for key in ("substitute_benchmark", "related_benchmark", "replacement_suite_id"):
        if boundary.get(key) not in (None, "", []):
            errors.append(f"{key} is not an admissible FrontierSWE source")
    for key in ("dataset_id", "source_dataset", "related_suite_id"):
        value = boundary.get(key)
        if isinstance(value, str) and any(marker in value.lower() for marker in _SUBSTITUTE_MARKERS):
            errors.append(f"{key} is rejected as an xLAM or related-benchmark substitute")


def validate_boundary(boundary: Mapping[str, Any]) -> list[str]:
    """Return all local boundary violations; never performs external I/O."""

    errors: list[str] = []
    if not isinstance(boundary, Mapping):
        return ["boundary must be a JSON object"]
    if boundary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        errors.append("boundary.schema_version is unsupported")
    _reject_substitutes(boundary, errors)
    if boundary.get("domains") != list(DOMAINS):
        errors.append(f"domains must be exactly {list(DOMAINS)!r}")
    source = _validate_source(boundary, errors)
    split = _validate_split(boundary, source or {}, errors) if source is not None else None
    _validate_native_contracts(boundary, errors)
    _validate_claim(boundary, split, errors)
    return errors


def _validate_service_receipts(receipt: Mapping[str, Any], errors: list[str]) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    wandb = _mapping(receipt.get("wandb"), "wandb", errors)
    tinker = _mapping(receipt.get("tinker"), "tinker", errors)
    huggingface = _mapping(
        receipt.get("huggingface", receipt.get("hf")), "huggingface", errors
    )
    if wandb is not None:
        _text(_first(wandb, "run_id", "id"), "wandb.run_id", errors)
        for key in ("url", "entity", "project", "group", "config_sha256", "artifact_name", "artifact_sha256"):
            if key.endswith("sha256"):
                _sha256(wandb.get(key), f"wandb.{key}", errors)
            else:
                _text(wandb.get(key), f"wandb.{key}", errors)
        if wandb.get("mode") != "online":
            errors.append("wandb.mode must be online")
        if wandb.get("artifact_acknowledged") is not True:
            errors.append("wandb.artifact_acknowledged must be true")
        if isinstance(wandb.get("url"), str) and not wandb["url"].startswith(_URL_PREFIXES):
            errors.append("wandb.url must be http(s)")
    if tinker is not None:
        _text(_first(tinker, "run_id", "id"), "tinker.run_id", errors)
        _text(_first(tinker, "model_id", "model"), "tinker.model_id", errors)
        _revision(
            _first(tinker, "base_model_revision", "model_revision"),
            "tinker.base_model_revision",
            errors,
        )
        _revision(tinker.get("adapter_revision"), "tinker.adapter_revision", errors)
        _text(tinker.get("service_client_status"), "tinker.service_client_status", errors)
        if tinker.get("service_client_status") != "verified":
            errors.append("tinker.service_client_status must be verified")
    if huggingface is not None:
        _text(_first(huggingface, "repo_id", "repository"), "huggingface.repo_id", errors)
        _revision(_first(huggingface, "revision", "commit"), "huggingface.revision", errors)
        _sha256(huggingface.get("artifact_sha256"), "huggingface.artifact_sha256", errors)
        _text(huggingface.get("artifact_path"), "huggingface.artifact_path", errors)
        _text(huggingface.get("upload_status"), "huggingface.upload_status", errors)
        if huggingface.get("upload_status") != "acknowledged":
            errors.append("huggingface.upload_status must be acknowledged")
    return wandb, tinker, huggingface


def _validate_result_rows(
    expected_task_ids: Sequence[str], results: Mapping[str, Any], errors: list[str]
) -> None:
    rows = results.get("task_rows")
    if not isinstance(rows, list) or len(rows) != len(expected_task_ids):
        errors.append("results.task_rows must contain exactly one row per deterministic task ID")
        return
    observed_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"results.task_rows[{index}] must be an object")
            continue
        task_id = row.get("task_id")
        observed_ids.append(task_id if isinstance(task_id, str) else "")
        if task_id != expected_task_ids[index]:
            errors.append(f"results.task_rows[{index}].task_id does not match the frozen order")
        for key in ("success", "state_integrity", "artifact_integrity"):
            if not isinstance(row.get(key), bool):
                errors.append(f"results.task_rows[{index}].{key} must be boolean")
        if row.get("status") not in {"passed", "failed"}:
            errors.append(f"results.task_rows[{index}].status must be passed or failed")
        if row.get("verifier_status") not in {"pass", "fail"}:
            errors.append(f"results.task_rows[{index}].verifier_status must be pass or fail")
        for key in _RAW_RESULT_KEYS:
            if key in row:
                errors.append(f"results.task_rows[{index}] must not contain raw result field {key}")
    if observed_ids != list(expected_task_ids):
        errors.append("results.task_rows task IDs are not an exact ordered match")
    task_count = len(rows)
    expected_counts = {
        "task_count": task_count,
        "successful_tasks": sum(bool(row.get("success")) for row in rows if isinstance(row, Mapping)),
        "state_integrity_tasks": sum(bool(row.get("state_integrity")) for row in rows if isinstance(row, Mapping)),
        "artifact_integrity_tasks": sum(bool(row.get("artifact_integrity")) for row in rows if isinstance(row, Mapping)),
        "verifier_pass_tasks": sum(row.get("verifier_status") == "pass" for row in rows if isinstance(row, Mapping)),
    }
    for key, expected in expected_counts.items():
        if results.get(key) != expected:
            errors.append(f"results.{key} does not match task rows")
    expected_rates = {
        "task_success_rate": expected_counts["successful_tasks"] / task_count,
        "state_integrity_rate": expected_counts["state_integrity_tasks"] / task_count,
        "artifact_integrity_rate": expected_counts["artifact_integrity_tasks"] / task_count,
        "verifier_pass_rate": expected_counts["verifier_pass_tasks"] / task_count,
    }
    for key, expected in expected_rates.items():
        if not _is_finite_number(results.get(key)) or results.get(key) != expected:
            errors.append(f"results.{key} does not match task rows exactly")


def build_result_receipt(
    boundary: Mapping[str, Any],
    *,
    task_rows: Sequence[Mapping[str, Any]],
    wandb: Mapping[str, Any],
    tinker: Mapping[str, Any],
    huggingface: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a receipt envelope from already-observed local result rows."""

    rows = [dict(row) for row in task_rows]
    total = len(rows)
    if total <= 0:
        raise FrontierSWEEvaluationError("task_rows must be non-empty")
    counts = {
        "task_count": total,
        "successful_tasks": sum(bool(row.get("success")) for row in rows),
        "state_integrity_tasks": sum(bool(row.get("state_integrity")) for row in rows),
        "artifact_integrity_tasks": sum(bool(row.get("artifact_integrity")) for row in rows),
        "verifier_pass_tasks": sum(row.get("verifier_status") == "pass" for row in rows),
    }
    results = {
        **counts,
        "task_success_rate": counts["successful_tasks"] / total,
        "state_integrity_rate": counts["state_integrity_tasks"] / total,
        "artifact_integrity_rate": counts["artifact_integrity_tasks"] / total,
        "verifier_pass_rate": counts["verifier_pass_tasks"] / total,
        "task_rows": rows,
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_family": BENCHMARK_FAMILY,
        "domains": list(DOMAINS),
        "boundary": dict(boundary),
        "source": dict(boundary.get("source", {})),
        "split": dict(boundary.get("split", {})),
        "environment": dict(boundary.get("environment", {})),
        "artifact": dict(boundary.get("artifact", {})),
        "verifier": dict(boundary.get("verifier", {})),
        "claim": dict(boundary.get("claim", {})),
        "results": results,
        "wandb": dict(wandb),
        "tinker": dict(tinker),
        "huggingface": dict(huggingface),
        "provenance": {
            "source_revision": boundary.get("source", {}).get("revision"),
            "license_identifier": boundary.get("source", {}).get("license_identifier"),
            "task_ids_sha256": boundary.get("split", {}).get("task_ids_sha256"),
            "split_manifest_sha256": boundary.get("split", {}).get("split_manifest_sha256"),
            "environment_digest": boundary.get("environment", {}).get("environment_digest"),
            "verifier_sha256": boundary.get("verifier", {}).get("source_sha256"),
            "base_model_revision": _first(tinker, "base_model_revision", "model_revision"),
            "adapter_revision": tinker.get("adapter_revision"),
        },
    }


def validate_result_receipt(
    receipt: Mapping[str, Any], boundary: Mapping[str, Any] | None = None
) -> list[str]:
    """Return local result-receipt violations; no services are contacted."""

    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt must be a JSON object"]
    _reject_raw_result_fields(receipt, "receipt", errors)
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("receipt.schema_version is unsupported")
    if receipt.get("status") != "completed":
        errors.append("receipt.status must be completed")
    if receipt.get("suite_id") != SUITE_ID:
        errors.append("receipt.suite_id must be frontier_swe_eval")
    if receipt.get("benchmark_id") != BENCHMARK_ID:
        errors.append("receipt.benchmark_id must be frontierswe")
    if receipt.get("benchmark_family") != BENCHMARK_FAMILY:
        errors.append("receipt.benchmark_family must be FrontierSWE")
    if receipt.get("domains") != list(DOMAINS):
        errors.append(f"receipt.domains must be exactly {list(DOMAINS)!r}")
    for key in ("dataset_id", "source_dataset", "related_suite_id"):
        value = receipt.get(key)
        if isinstance(value, str) and any(marker in value.lower() for marker in _SUBSTITUTE_MARKERS):
            errors.append(f"receipt.{key} is rejected as an xLAM or related-benchmark substitute")
    nested_boundary = _mapping(receipt.get("boundary"), "receipt.boundary", errors)
    if nested_boundary is None:
        return errors
    errors.extend(f"boundary: {error}" for error in validate_boundary(nested_boundary))
    if boundary is not None and not _same_json(boundary, nested_boundary):
        errors.append("receipt.boundary does not equal the supplied immutable boundary")
    split = _mapping(nested_boundary.get("split"), "receipt.boundary.split", errors)
    for key in ("source", "split", "environment", "artifact", "verifier", "claim"):
        if not _same_json(receipt.get(key), nested_boundary.get(key)):
            errors.append(f"receipt.{key} does not match receipt.boundary.{key}")
    results = _mapping(receipt.get("results"), "results", errors)
    if results is not None and split is not None:
        _validate_result_rows(split.get("task_ids", []), results, errors)
    wandb, tinker, huggingface = _validate_service_receipts(receipt, errors)
    provenance = _mapping(receipt.get("provenance"), "provenance", errors)
    if provenance is not None and tinker is not None:
        expected_provenance = {
            "source_revision": (
                nested_boundary.get("source", {}).get("revision")
                if isinstance(nested_boundary.get("source"), Mapping)
                else None
            ),
            "license_identifier": (
                nested_boundary.get("source", {}).get("license_identifier")
                if isinstance(nested_boundary.get("source"), Mapping)
                else None
            ),
            "task_ids_sha256": split.get("task_ids_sha256") if split is not None else None,
            "split_manifest_sha256": split.get("split_manifest_sha256") if split is not None else None,
            "environment_digest": (
                nested_boundary.get("environment", {}).get("environment_digest")
                if isinstance(nested_boundary.get("environment"), Mapping)
                else None
            ),
            "verifier_sha256": (
                nested_boundary.get("verifier", {}).get("source_sha256")
                if isinstance(nested_boundary.get("verifier"), Mapping)
                else None
            ),
            "base_model_revision": _first(tinker, "base_model_revision", "model_revision"),
            "adapter_revision": tinker.get("adapter_revision"),
        }
        for key, expected in expected_provenance.items():
            if provenance.get(key) != expected:
                errors.append(f"provenance.{key} does not match the immutable boundary/receipt")
    if huggingface is not None and tinker is not None:
        if _first(huggingface, "revision", "commit") != tinker.get("adapter_revision"):
            errors.append("huggingface.revision must equal tinker.adapter_revision")
    if wandb is not None and results is not None:
        artifact = nested_boundary.get("artifact")
        expected_artifact = artifact.get("manifest_sha256") if isinstance(artifact, Mapping) else None
        if wandb.get("artifact_sha256") != expected_artifact:
            errors.append("wandb.artifact_sha256 must acknowledge the boundary artifact manifest")
    return errors


def require_valid_boundary(boundary: Mapping[str, Any]) -> Mapping[str, Any]:
    errors = validate_boundary(boundary)
    if errors:
        raise FrontierSWEEvaluationError("boundary blocked: " + "; ".join(errors))
    return boundary


def require_valid_receipt(
    receipt: Mapping[str, Any], boundary: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    errors = validate_result_receipt(receipt, boundary)
    if errors:
        raise FrontierSWEEvaluationError("receipt blocked: " + "; ".join(errors))
    return receipt


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = load_json(args.receipt)
        boundary = load_json(args.boundary) if args.boundary is not None else None
        errors = validate_result_receipt(receipt, boundary)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors = [f"receipt load/parse failure: {exc}"]
    result = {
        "schema_version": "pavlov-frontier-swe-validation-v1",
        "status": "PASS" if not errors else "BLOCKED",
        "suite_id": SUITE_ID,
        "evaluation_role": PRIMARY_EVAL,
        "errors": errors,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
