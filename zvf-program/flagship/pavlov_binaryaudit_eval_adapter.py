"""Offline evaluation-boundary adapter for the BinaryAudit primary suite.

This adapter is a contract validator, not a benchmark runner.  It does not
download BinaryAudit, execute binaries, contact W&B/Hugging Face/Tinker, or
invent a result.  A boundary is admissible only when it identifies the
authoritative source and immutable license/revision, hashes deterministic task
and split manifests, and pins a native environment, artifact contract, and
verifier.  A result receipt must carry the same identity and complete tracking
receipts.

``primary_eval`` is evidence-bearing only after a completed result receipt has
passed validation.  ``receipt_proven_heldout`` proves split provenance and
receipt completeness, but is intentionally *not* a substitute for a primary
evaluation result.  xLAM, BFCL, AgentHarm, WebBench, and every other related
benchmark are rejected when presented as a BinaryAudit substitute.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any


AUTHORITATIVE_SOURCE_ID = "QuesmaOrg/BinaryAudit"
AUTHORITATIVE_SOURCE_URL = "https://github.com/QuesmaOrg/BinaryAudit"
SUITE_ID = "binaryaudit_eval"
BENCHMARK_ID = "binaryaudit"
RESULT_SCHEMA_VERSION = "binaryaudit-result-v1"
PRIMARY_EVAL = "primary_eval"
RECEIPT_PROVEN_HELDOUT = "receipt_proven_heldout"
SPLIT_KEYS = ("train", PRIMARY_EVAL, RECEIPT_PROVEN_HELDOUT)
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_SUBSTITUTE_WORDS = {
    "xlam",
    "bfcl",
    "agentharm",
    "webbench",
    "bankertoolbench",
    "apex_agents",
    "swe_bench",
    "frontierswe",
}
_SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded"}


class BinaryAuditBoundaryError(ValueError):
    """Raised when a BinaryAudit contract or receipt fails closed."""

    def __init__(self, diagnostics: Iterable[str]):
        unique = tuple(dict.fromkeys(str(item) for item in diagnostics if str(item)))
        self.diagnostics = unique or ("BinaryAudit boundary validation failed",)
        super().__init__("; ".join(self.diagnostics))


BoundaryValidationError = BinaryAuditBoundaryError


def _successful_status(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _SUCCESS_STATUSES


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON so task/split receipts are reproducible."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BinaryAuditBoundaryError((f"{name} must be a non-empty string",))
    return value.strip()


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BinaryAuditBoundaryError((f"{name} must be a finite real number",))
    result = float(value)
    if not math.isfinite(result):
        raise BinaryAuditBoundaryError((f"{name} must be finite",))
    return result


def _hash64(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise BinaryAuditBoundaryError((f"{name} must be a 64-character hexadecimal hash",))
    return value.lower()


def _immutable_revision(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BinaryAuditBoundaryError((f"{name} must be an immutable revision",))
    revision = value.strip()
    if not (_HEX40.fullmatch(revision) or _DIGEST.fullmatch(revision)):
        raise BinaryAuditBoundaryError(
            (f"{name} must be a 40-character commit or sha256 digest, not a mutable tag/branch",)
        )
    return revision.lower()


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BinaryAuditBoundaryError((f"{name} must be a mapping",))
    return value


def _ids(name: str, values: Any, *, require_nonempty: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise BinaryAuditBoundaryError((f"{name} must be an iterable of task IDs",))
    materialized = tuple(values)
    if require_nonempty and not materialized:
        raise BinaryAuditBoundaryError((f"{name} must be non-empty",))
    normalized: list[str] = []
    for index, value in enumerate(materialized):
        try:
            normalized.append(_string(f"{name}[{index}]", value))
        except BinaryAuditBoundaryError as exc:
            raise exc
    if len(set(normalized)) != len(normalized):
        raise BinaryAuditBoundaryError((f"{name} must not contain duplicate IDs",))
    if tuple(normalized) != tuple(sorted(normalized)):
        raise BinaryAuditBoundaryError((f"{name} must be lexically sorted for deterministic hashing",))
    return tuple(normalized)


def task_id_manifest_sha256(task_ids: Iterable[str]) -> str:
    """Return the canonical hash for an ordered, unique task-ID list."""

    return canonical_sha256(list(_ids("task_ids", task_ids)))


def split_manifest_sha256(split_manifest: Mapping[str, Iterable[str]]) -> str:
    """Return the canonical hash for the complete disjoint split manifest."""

    normalized = _normalize_split_manifest(split_manifest)
    return canonical_sha256({key: list(normalized[key]) for key in SPLIT_KEYS})


def deterministic_task_id(
    raw_task_id: str,
    source_revision: str,
    *,
    source_id: str = AUTHORITATIVE_SOURCE_ID,
) -> str:
    """Derive a stable task identity bound to the pinned source revision."""

    raw_task_id = _string("raw_task_id", raw_task_id)
    source_id = _string("source_id", source_id)
    source_revision = _immutable_revision("source_revision", source_revision)
    payload = f"{source_id}\0{source_revision}\0{raw_task_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_mapping(source: Mapping[str, Any] | Path | str, name: str) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, (Path, str)):
        raise BinaryAuditBoundaryError((f"{name} must be a mapping or JSON path",))
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BinaryAuditBoundaryError((f"could not read {name}: {exc}",)) from exc
    return _mapping(name, value)


def _normalize_split_manifest(value: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise BinaryAuditBoundaryError(("split_manifest must be a mapping",))
    if set(value) != set(SPLIT_KEYS):
        raise BinaryAuditBoundaryError(
            (f"split_manifest keys must be exactly {list(SPLIT_KEYS)}",)
        )
    normalized = {
        key: _ids(f"split_manifest.{key}", value[key]) for key in SPLIT_KEYS
    }
    seen: set[str] = set()
    for key in SPLIT_KEYS:
        overlap = seen.intersection(normalized[key])
        if overlap:
            raise BinaryAuditBoundaryError(
                (f"split_manifest has overlapping IDs in {key}: {len(overlap)}",)
            )
        seen.update(normalized[key])
    return normalized


def _reject_substitute(value: Any, name: str, diagnostics: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip().lower().replace("-", "_").replace(" ", "_")
        if text and text not in {BENCHMARK_ID, SUITE_ID}:
            diagnostics.append(f"{name} is not BinaryAudit; related benchmarks cannot substitute")
        if any(word in text for word in _SUBSTITUTE_WORDS):
            diagnostics.append(f"{name} names a rejected related benchmark/xLAM substitute")
        return
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = tuple(value)
        if values:
            diagnostics.append(f"{name} must not list a substitute benchmark")
        return
    diagnostics.append(f"{name} must not identify a related benchmark substitute")


def _validate_source(boundary: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    source = boundary.get("source_identity", boundary.get("source"))
    if not isinstance(source, Mapping):
        diagnostics.append("source_identity is required")
        return {}
    source_id = source.get("id", source.get("source_id"))
    source_url = source.get("url", source.get("source_url"))
    license_record = source.get("license")
    if not isinstance(license_record, Mapping):
        license_record = {}
    if source_id != AUTHORITATIVE_SOURCE_ID:
        diagnostics.append(f"source identity must be {AUTHORITATIVE_SOURCE_ID!r}")
    if source_url != AUTHORITATIVE_SOURCE_URL:
        diagnostics.append(f"source URL must be {AUTHORITATIVE_SOURCE_URL!r}")
    try:
        revision = _immutable_revision(
            "source_identity.revision",
            source.get("revision", source.get("immutable_revision")),
        )
        license_spdx = _string(
            "source_identity.license_spdx",
            source.get("license_spdx", license_record.get("spdx")),
        )
        license_hash = _hash64(
            "source_identity.license_text_sha256",
            source.get("license_text_sha256", license_record.get("text_sha256")),
        )
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        revision = ""
        license_spdx = ""
        license_hash = ""
    license_url = source.get("license_url")
    if license_url is not None and (not isinstance(license_url, str) or not license_url.strip()):
        diagnostics.append("source_identity.license_url must be non-empty when present")
    return {
        "id": AUTHORITATIVE_SOURCE_ID,
        "url": AUTHORITATIVE_SOURCE_URL,
        "revision": revision,
        "license_spdx": license_spdx,
        "license_text_sha256": license_hash,
        **({"license_url": license_url} if license_url is not None else {}),
    }


def _validate_environment(boundary: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    environment = boundary.get("native_environment", boundary.get("environment"))
    if not isinstance(environment, Mapping):
        diagnostics.append("native_environment is required")
        return {}
    mode = environment.get("mode", environment.get("environment_mode"))
    if mode != "native":
        diagnostics.append("native_environment.mode must be 'native'")
    try:
        container = environment.get("container_digest")
        if not isinstance(container, str) or not _DIGEST.fullmatch(container):
            raise BinaryAuditBoundaryError(("native_environment.container_digest must be sha256:<64 hex>",))
        environment_hash = _hash64(
            "native_environment.environment_manifest_sha256",
            environment.get("environment_manifest_sha256", environment.get("manifest_sha256")),
        )
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        container = ""
        environment_hash = ""

    artifact = environment.get("artifact_contract")
    if not isinstance(artifact, Mapping):
        diagnostics.append("native_environment.artifact_contract is required")
        artifact = {}
    if artifact.get("required") is not True:
        diagnostics.append("artifact_contract.required must be true")
    try:
        artifact_hash = _hash64(
            "artifact_contract.manifest_sha256",
            artifact.get("manifest_sha256", artifact.get("artifact_manifest_sha256")),
        )
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        artifact_hash = ""
    artifact_types = artifact.get("types", artifact.get("artifact_types"))
    try:
        if isinstance(artifact_types, (str, bytes, bytearray)) or not artifact_types:
            raise BinaryAuditBoundaryError(("artifact_contract.types must be non-empty",))
        artifact_types = tuple(_string("artifact_contract.types[]", item) for item in artifact_types)
    except (BinaryAuditBoundaryError, TypeError) as exc:
        diagnostics.extend(
            exc.diagnostics if isinstance(exc, BinaryAuditBoundaryError) else ("artifact_contract.types must be iterable",)
        )
        artifact_types = ()

    verifier = environment.get("verifier_contract", environment.get("verifier"))
    if not isinstance(verifier, Mapping):
        diagnostics.append("native_environment.verifier_contract is required")
        verifier = {}
    verifier_name = verifier.get("name")
    if not isinstance(verifier_name, str) or not verifier_name.strip():
        diagnostics.append("verifier_contract.name must be non-empty")
        verifier_name = ""
    try:
        verifier_revision = _immutable_revision(
            "verifier_contract.revision", verifier.get("revision")
        )
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        verifier_revision = ""
    receipt_schema = verifier.get("receipt_schema")
    if not isinstance(receipt_schema, str) or not receipt_schema.strip():
        diagnostics.append("verifier_contract.receipt_schema must be non-empty")
        receipt_schema = ""
    checks = verifier.get("checks")
    if isinstance(checks, (str, bytes, bytearray)) or not checks:
        diagnostics.append("verifier_contract.checks must be non-empty")
        checks = ()
    else:
        try:
            checks = tuple(_string("verifier_contract.checks[]", item) for item in checks)
        except (BinaryAuditBoundaryError, TypeError) as exc:
            diagnostics.extend(
                exc.diagnostics if isinstance(exc, BinaryAuditBoundaryError) else ("verifier_contract.checks must be iterable",)
            )
            checks = ()
    return {
        "mode": "native",
        "container_digest": str(container).lower(),
        "environment_manifest_sha256": environment_hash,
        "artifact_contract": {
            "required": True,
            "manifest_sha256": artifact_hash,
            "types": list(artifact_types),
        },
        "verifier_contract": {
            "name": str(verifier_name),
            "revision": verifier_revision,
            "receipt_schema": str(receipt_schema),
            "checks": list(checks),
        },
    }


def validate_binaryaudit_boundary(
    source: Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    """Validate and normalize a BinaryAudit boundary manifest."""

    boundary = _read_mapping(source, "boundary")
    diagnostics: list[str] = []
    if boundary.get("suite_id") != SUITE_ID:
        diagnostics.append(f"suite_id must be {SUITE_ID!r}")
    if boundary.get("benchmark_id") != BENCHMARK_ID:
        diagnostics.append(f"benchmark_id must be {BENCHMARK_ID!r}; related benchmarks are not substitutes")
    for key in ("related_benchmark", "substitute_benchmark", "substitutes"):
        _reject_substitute(boundary.get(key), key, diagnostics)
    source_identity = _validate_source(boundary, diagnostics)

    try:
        task_ids = _ids("task_ids", boundary.get("task_ids"))
        task_hash = _hash64("task_id_manifest_sha256", boundary.get("task_id_manifest_sha256"))
        expected_task_hash = task_id_manifest_sha256(task_ids)
        if task_hash != expected_task_hash:
            diagnostics.append("task_id_manifest_sha256 does not match task_ids")
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        task_ids = ()
        task_hash = ""
    try:
        split = _normalize_split_manifest(boundary.get("split_manifest"))
        split_hash = _hash64("split_manifest_sha256", boundary.get("split_manifest_sha256"))
        expected_split_hash = split_manifest_sha256(split)
        if split_hash != expected_split_hash:
            diagnostics.append("split_manifest_sha256 does not match split_manifest")
        if set(task_ids) != set().union(*(set(split[key]) for key in SPLIT_KEYS)):
            diagnostics.append("task_ids must equal the complete split-manifest ID union")
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        split = {key: () for key in SPLIT_KEYS}
        split_hash = ""
    environment = _validate_environment(boundary, diagnostics)
    if diagnostics:
        raise BinaryAuditBoundaryError(diagnostics)
    return {
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "source_identity": source_identity,
        "task_ids": list(task_ids),
        "task_id_manifest_sha256": task_hash,
        "split_manifest": {key: list(split[key]) for key in SPLIT_KEYS},
        "split_manifest_sha256": split_hash,
        "native_environment": environment,
        "substitutes_rejected": True,
        "boundary_status": "validated_contract_only",
    }


def _validate_tracking(receipt: Mapping[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    tracking: dict[str, Any] = {}
    wandb = receipt.get("wandb")
    if not isinstance(wandb, Mapping):
        diagnostics.append("missing W&B receipt")
    else:
        for field in ("run_id", "url"):
            if not isinstance(wandb.get(field), str) or not wandb[field].strip():
                diagnostics.append(f"W&B receipt missing {field}")
        if wandb.get("mode") != "online":
            diagnostics.append("W&B receipt mode must be online")
        try:
            config_hash = _hash64("W&B config_sha256", wandb.get("config_sha256"))
            metrics_hash = _hash64(
                "W&B metrics_receipt_sha256", wandb.get("metrics_receipt_sha256")
            )
        except BinaryAuditBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            config_hash = metrics_hash = ""
        if wandb.get("metrics_logged") is not True:
            diagnostics.append("W&B receipt metrics_logged must be true")
        tracking["wandb"] = {
            "run_id": str(wandb.get("run_id", "")),
            "url": str(wandb.get("url", "")),
            "mode": "online",
            "config_sha256": config_hash,
            "metrics_receipt_sha256": metrics_hash,
        }

    tinker = receipt.get("tinker")
    if not isinstance(tinker, Mapping):
        diagnostics.append("missing Tinker receipt")
    else:
        if not isinstance(tinker.get("run_id"), str) or not tinker["run_id"].strip():
            diagnostics.append("Tinker receipt missing run_id")
        if not _successful_status(tinker.get("status")):
            diagnostics.append("Tinker receipt status must be completed/success")
        try:
            config_hash = _hash64("Tinker config_sha256", tinker.get("config_sha256"))
            cost = _finite("Tinker cost_usd", tinker.get("cost_usd"))
            if cost < 0:
                raise BinaryAuditBoundaryError(("Tinker cost_usd must be non-negative",))
        except BinaryAuditBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            config_hash = ""
            cost = 0.0
        tracking["tinker"] = {
            "run_id": str(tinker.get("run_id", "")),
            "status": str(tinker.get("status", "")),
            "config_sha256": config_hash,
            "cost_usd": cost,
        }

    hf = receipt.get("hf")
    if not isinstance(hf, Mapping):
        diagnostics.append("missing Hugging Face receipt")
    else:
        for field in ("repo",):
            if not isinstance(hf.get(field), str) or not hf[field].strip():
                diagnostics.append(f"Hugging Face receipt missing {field}")
        try:
            revision = _immutable_revision("Hugging Face revision", hf.get("revision", hf.get("commit")))
            artifact_hash = _hash64(
                "Hugging Face artifact_manifest_sha256", hf.get("artifact_manifest_sha256")
            )
        except BinaryAuditBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
            revision = artifact_hash = ""
        if hf.get("visibility") != "private":
            diagnostics.append("Hugging Face checkpoint visibility must be private")
        tracking["hf"] = {
            "repo": str(hf.get("repo", "")),
            "revision": revision,
            "visibility": "private",
            "artifact_manifest_sha256": artifact_hash,
        }
    return tracking


def _validate_metric_mapping(value: Any, diagnostics: list[str]) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        diagnostics.append("primary result metrics must be a non-empty mapping")
        return {}
    metrics: dict[str, float] = {}
    for key, metric in value.items():
        if not isinstance(key, str) or not key.strip():
            diagnostics.append("metric names must be non-empty strings")
            continue
        try:
            metrics[key] = _finite(f"metrics.{key}", metric)
        except BinaryAuditBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)
    return metrics


def validate_binaryaudit_result_receipt(
    boundary_source: Mapping[str, Any] | Path | str,
    receipt_source: Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    """Validate one completed BinaryAudit receipt against its exact boundary."""

    boundary = validate_binaryaudit_boundary(boundary_source)
    receipt = _read_mapping(receipt_source, "result receipt")
    diagnostics: list[str] = []
    if receipt.get("schema_version") != RESULT_SCHEMA_VERSION:
        diagnostics.append(f"schema_version must be {RESULT_SCHEMA_VERSION!r}")
    if receipt.get("suite_id") != SUITE_ID or receipt.get("benchmark_id") != BENCHMARK_ID:
        diagnostics.append("result receipt must identify BinaryAudit, not a related benchmark")
    for key in ("related_benchmark", "substitute_benchmark", "substitutes"):
        _reject_substitute(receipt.get(key), key, diagnostics)
    role = receipt.get("evaluation_role", receipt.get("role"))
    if role not in {PRIMARY_EVAL, RECEIPT_PROVEN_HELDOUT}:
        diagnostics.append("evaluation_role must be primary_eval or receipt_proven_heldout")
        role = ""

    source = receipt.get("source_identity", receipt.get("source"))
    if not isinstance(source, Mapping):
        diagnostics.append("result receipt source_identity is required")
    else:
        result_license = source.get("license")
        if not isinstance(result_license, Mapping):
            result_license = {}
        result_source_fields = {
            "id": source.get("id", source.get("source_id")),
            "url": source.get("url", source.get("source_url")),
            "revision": source.get("revision", source.get("immutable_revision")),
            "license_spdx": source.get("license_spdx", result_license.get("spdx")),
            "license_text_sha256": source.get(
                "license_text_sha256", result_license.get("text_sha256")
            ),
        }
        for field in ("id", "url", "revision", "license_spdx", "license_text_sha256"):
            if result_source_fields[field] != boundary["source_identity"].get(field):
                diagnostics.append(f"result source_identity.{field} differs from boundary")
        try:
            _immutable_revision("result source_identity.revision", result_source_fields["revision"])
            _hash64(
                "result source_identity.license_text_sha256",
                result_source_fields["license_text_sha256"],
            )
        except BinaryAuditBoundaryError as exc:
            diagnostics.extend(exc.diagnostics)

    try:
        task_ids = _ids("result task_ids", receipt.get("task_ids"))
        task_hash = _hash64("result task_id_manifest_sha256", receipt.get("task_id_manifest_sha256"))
        if task_hash != task_id_manifest_sha256(task_ids):
            diagnostics.append("result task_id_manifest_sha256 does not match result task_ids")
        expected = tuple(boundary["split_manifest"].get(role, ()))
        if tuple(task_ids) != expected:
            diagnostics.append(f"result task_ids must exactly match the {role} split")
    except BinaryAuditBoundaryError as exc:
        diagnostics.extend(exc.diagnostics)
        task_ids = ()
        task_hash = ""
    if receipt.get("split_manifest_sha256") != boundary["split_manifest_sha256"]:
        diagnostics.append("result split_manifest_sha256 differs from boundary")

    environment = receipt.get("native_environment_receipt", receipt.get("environment_receipt"))
    expected_environment = boundary["native_environment"]
    if not isinstance(environment, Mapping):
        diagnostics.append("native_environment_receipt is required")
    else:
        for field in ("container_digest", "environment_manifest_sha256"):
            if environment.get(field) != expected_environment.get(field):
                diagnostics.append(f"environment receipt {field} differs from boundary")

    artifact = receipt.get("artifact_receipt")
    expected_artifact = expected_environment.get("artifact_contract", {})
    if not isinstance(artifact, Mapping):
        diagnostics.append("artifact_receipt is required")
    else:
        if artifact.get("manifest_sha256", artifact.get("artifact_manifest_sha256")) != expected_artifact.get("manifest_sha256"):
            diagnostics.append("artifact receipt manifest differs from artifact contract")
        paths = artifact.get("paths", artifact.get("artifacts"))
        if isinstance(paths, (str, bytes, bytearray)) or not paths:
            diagnostics.append("artifact_receipt.paths must be non-empty")

    verifier = receipt.get("verifier_receipt", receipt.get("verifier"))
    expected_verifier = expected_environment.get("verifier_contract", {})
    if not isinstance(verifier, Mapping):
        diagnostics.append("verifier_receipt is required")
    else:
        for field in ("name", "revision", "receipt_schema"):
            if verifier.get(field) != expected_verifier.get(field):
                diagnostics.append(f"verifier receipt {field} differs from boundary")

    tracking = _validate_tracking(receipt, diagnostics)
    if tracking.get("hf", {}).get("artifact_manifest_sha256") != expected_artifact.get(
        "manifest_sha256"
    ):
        diagnostics.append("Hugging Face artifact manifest differs from artifact contract")
    metrics: dict[str, float] = {}
    if role == PRIMARY_EVAL:
        if not _successful_status(receipt.get("status")):
            diagnostics.append("primary_eval receipt status must be completed/success")
        metrics = _validate_metric_mapping(receipt.get("metrics"), diagnostics)
    elif role == RECEIPT_PROVEN_HELDOUT:
        proof = receipt.get("heldout_proof")
        if not isinstance(proof, Mapping):
            diagnostics.append("receipt_proven_heldout requires heldout_proof")
        else:
            expected_selection_hash = task_id_manifest_sha256(boundary["split_manifest"]["train"])
            expected_heldout_hash = task_id_manifest_sha256(boundary["split_manifest"][RECEIPT_PROVEN_HELDOUT])
            if proof.get("selection_task_id_manifest_sha256") != expected_selection_hash:
                diagnostics.append("heldout_proof selection task-ID hash differs from boundary")
            if proof.get("heldout_task_id_manifest_sha256") != expected_heldout_hash:
                diagnostics.append("heldout_proof heldout task-ID hash differs from boundary")
            for field in ("disjoint", "selection_locked", "not_used_for_selection"):
                if proof.get(field) is not True:
                    diagnostics.append(f"heldout_proof.{field} must be true")
        if "metrics" in receipt:
            metrics = _validate_metric_mapping(receipt.get("metrics"), diagnostics)
    if diagnostics:
        raise BinaryAuditBoundaryError(diagnostics)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "admissible_primary_eval" if role == PRIMARY_EVAL else "receipt_proven_heldout",
        "evaluation_role": role,
        "primary_eval": role == PRIMARY_EVAL,
        "receipt_proven_heldout": role == RECEIPT_PROVEN_HELDOUT,
        "scientific_evidence": role == PRIMARY_EVAL,
        "claim_scope": "binaryaudit_security_component",
        "substitute_rejected": True,
        "source_identity": boundary["source_identity"],
        "task_ids": list(task_ids),
        "task_id_manifest_sha256": task_hash,
        "split_manifest_sha256": boundary["split_manifest_sha256"],
        "native_environment": boundary["native_environment"],
        "tracking_receipts": tracking,
        "metrics": metrics,
        "primary_eval_required": True,
        "portfolio_claim_permitted": False,
        "company_claim_permitted": False,
    }


validate_binaryaudit_receipt = validate_binaryaudit_result_receipt


__all__ = [
    "AUTHORITATIVE_SOURCE_ID",
    "AUTHORITATIVE_SOURCE_URL",
    "BENCHMARK_ID",
    "BoundaryValidationError",
    "BinaryAuditBoundaryError",
    "PRIMARY_EVAL",
    "RECEIPT_PROVEN_HELDOUT",
    "RESULT_SCHEMA_VERSION",
    "SUITE_ID",
    "canonical_sha256",
    "deterministic_task_id",
    "split_manifest_sha256",
    "task_id_manifest_sha256",
    "validate_binaryaudit_boundary",
    "validate_binaryaudit_receipt",
    "validate_binaryaudit_result_receipt",
]
