#!/usr/bin/env python3
"""Define the offline evaluation boundary for ``banker_toolbench_eval``.

This adapter is a contract and receipt gate, not an evaluator.  It accepts
already supplied metadata only: a pinned source revision/license, deterministic
task-ID hashes, native environment/artifact/verifier contracts, and optional
result receipts.  It never downloads BankerToolBench, calls W&B/Tinker/HF, or
fabricates a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pavlov-banker-toolbench-eval-boundary-v1"
SUITE_ID = "banker_toolbench_eval"
ROLE = "primary_eval"
SPLIT_DESCRIPTION = "held-out evaluation tasks"
DATASET_ID = "handshake-ai-research/bankertoolbench"
SOURCE_URL = "https://huggingface.co/datasets/handshake-ai-research/bankertoolbench"
SOURCE_KIND = "huggingface_dataset"
EXPECTED_DOMAINS = ("enterprise", "finance", "long_horizon", "tool_use")
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = {
    "",
    "none",
    "null",
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
    "not provided",
}
_MUTABLE_MARKERS = {"latest", "main", "master", "head", "tip", "current", "branch", "pending"}
_IMMUTABLE_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMMUTABLE_DIGEST_URI = re.compile(
    r"(?:^|[@/#:])sha256:[0-9a-f]{64}(?:$|[/?#])",
    re.IGNORECASE,
)
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


class BankerToolBenchBoundaryError(ValueError):
    """Raised when an E4 boundary cannot be used as a strict gate."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_id_hash(task_id: str) -> str:
    """Hash one authoritative task ID without retaining the ID in the boundary."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise BankerToolBenchBoundaryError("task IDs must be non-empty strings")
    return _sha256(task_id.strip())


def split_manifest_hash(task_id_hashes: Sequence[str]) -> str:
    """Hash the canonical role and ordered task-ID-hash list."""

    return _sha256(
        _canonical_json(
            {
                "suite_id": SUITE_ID,
                "role": ROLE,
                "task_id_hashes": list(task_id_hashes),
            }
        )
    )


def _read_json(value: str | Path) -> Any:
    try:
        return json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BankerToolBenchBoundaryError(f"cannot read JSON {value!s}: {exc}") from exc


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _assert_metadata_only(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_CONTENT_KEYS:
                raise BankerToolBenchBoundaryError(
                    f"{path} contains raw {str(key).lower()} content; E4 boundary is metadata-only"
                )
            _assert_metadata_only(child, f"{path}.{key}")
    elif _is_sequence(value):
        for index, child in enumerate(value):
            _assert_metadata_only(child, f"{path}[{index}]")


def _looks_placeholder(value: Any) -> bool:
    if value is None or not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDER or any(
        token in normalized
        for token in ("placeholder", "unrecorded", "not provided", "pending", "todo", "unset")
    )


def _immutable_reference(value: Any) -> bool:
    if _looks_placeholder(value) or not isinstance(value, str):
        return False
    reference = value.strip()
    lowered = reference.lower()
    if any(character.isspace() for character in reference):
        return False
    parts = {part for part in re.split(r"[/:?#._@-]+", lowered) if part}
    if parts.intersection(_MUTABLE_MARKERS):
        return False
    if _IMMUTABLE_SHA256.fullmatch(lowered) or _IMMUTABLE_DIGEST_URI.search(lowered):
        return True
    return lowered.startswith(
        (
            "receipt://",
            "receipt:",
            "urn:receipt:",
            "hf://",
            "oci://",
            "git://",
            "https://",
            "http://",
        )
    ) and len(reference.split(":", 1)[1].strip("/")) >= 2


def _require_exact_source(source_identity: Mapping[str, Any] | None) -> None:
    if source_identity is None:
        return
    if not isinstance(source_identity, Mapping):
        raise BankerToolBenchBoundaryError("source_identity must be a JSON object")
    _assert_metadata_only(source_identity, "source_identity")
    expected = {
        "kind": SOURCE_KIND,
        "dataset_id": DATASET_ID,
        "url": SOURCE_URL,
    }
    for key, expected_value in expected.items():
        if source_identity.get(key) != expected_value:
            raise BankerToolBenchBoundaryError(
                f"source_identity.{key} must identify authoritative BankerToolBench source"
            )
    substitute = source_identity.get("substitute_for", source_identity.get("related_benchmark"))
    if substitute:
        raise BankerToolBenchBoundaryError("related benchmarks and substitutes are not BankerToolBench")


def _extract_task_ids(
    tasks: Iterable[Any] | Mapping[str, Any] | str | Path | None,
    task_id_hashes: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if tasks is not None:
        _assert_metadata_only(tasks, "tasks")
    if tasks is not None and task_id_hashes is not None:
        errors.append("supply tasks or task_id_hashes, not both")
        return [], errors
    if isinstance(tasks, (str, Path)):
        tasks = _read_json(tasks)
        _assert_metadata_only(tasks, "tasks")
    if task_id_hashes is not None:
        raw_hashes = list(task_id_hashes)
        if not raw_hashes:
            return [], ["task_id_hashes must be non-empty"]
        if any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in raw_hashes):
            return [], ["task_id_hashes must contain ordered SHA-256 digests"]
        if len(set(raw_hashes)) != len(raw_hashes):
            errors.append("duplicate task ID hash")
        return sorted(raw_hashes), errors
    if tasks is None:
        return [], ["authoritative task IDs are required"]
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
            value = item.strip()
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
        errors.append("authoritative task IDs are required")
        return [], errors
    if len(ids) != len(set(ids)):
        errors.append("duplicate authoritative task ID")
    ids = sorted(ids)
    return [task_id_hash(value) for value in ids], errors


def _validate_contract(
    contract: Mapping[str, Any] | None,
    *,
    name: str,
    required: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(contract, Mapping):
        return {}, [f"{name} contract is required"]
    errors: list[str] = []
    _assert_metadata_only(contract, name)
    normalized = dict(contract)
    for key in required:
        if key not in contract or contract[key] in (None, "", []):
            errors.append(f"{name} contract is missing {key}")
    if name == "environment":
        if contract.get("native") is not True:
            errors.append("environment contract must declare native=true")
        if contract.get("stateful") is not True:
            errors.append("environment contract must declare stateful=true")
        revision = contract.get("environment_revision")
        if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
            errors.append("environment_revision must be an immutable 40-character SHA")
    if name == "artifact":
        if contract.get("required") is not True:
            errors.append("artifact contract must declare required=true")
        if not _immutable_reference(contract.get("artifact_receipt_ref")):
            errors.append("artifact_receipt_ref must be immutable and non-placeholder")
    if name == "verifier":
        if contract.get("deterministic") is not True:
            errors.append("verifier contract must declare deterministic=true")
        revision = contract.get("verifier_revision")
        if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
            errors.append("verifier_revision must be an immutable 40-character SHA")
        if not _immutable_reference(contract.get("verifier_receipt_ref")):
            errors.append("verifier_receipt_ref must be immutable and non-placeholder")
    return normalized, errors


def _result_receipt_fields(
    result_receipts: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    required_shape = {
        provider: list(fields) for provider, fields in RESULT_RECEIPT_FIELDS.items()
    }
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
    for raw_provider, required_fields in RESULT_RECEIPT_FIELDS.items():
        source = result_receipts.get(raw_provider)
        if source is None:
            for alias, canonical in _RESULT_PROVIDER_ALIASES.items():
                if canonical == raw_provider and alias in result_receipts:
                    source = result_receipts[alias]
                    break
        if not isinstance(source, Mapping):
            errors.append(f"missing {raw_provider} result receipt")
            continue
        provider_value: dict[str, Any] = {}
        for field in required_fields:
            value = source.get(field)
            if value in (None, "") or _looks_placeholder(value):
                errors.append(f"{raw_provider} result receipt missing {field}")
                continue
            if field.endswith("_sha256"):
                if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                    errors.append(f"{raw_provider}.{field} must be a SHA-256 digest")
                    continue
            if field.endswith("_revision"):
                if not isinstance(value, str) or not _PINNED_REVISION.fullmatch(value):
                    errors.append(f"{raw_provider}.{field} must be an immutable revision")
                    continue
            provider_value[field] = str(value) if not isinstance(value, (int, float, bool)) else value
        if provider_value:
            normalized["providers"][raw_provider] = provider_value
    if errors:
        normalized["status"] = "INVALID"
        normalized["recorded"] = False
    return normalized, errors


def _boundary_errors(boundary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if boundary.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the BankerToolBench boundary schema")
    source = boundary.get("source_identity")
    expected_source = {"kind": SOURCE_KIND, "dataset_id": DATASET_ID, "url": SOURCE_URL}
    if source != expected_source:
        errors.append("source_identity is not the authoritative BankerToolBench source")
    if boundary.get("suite_id") != SUITE_ID:
        errors.append("suite_id must be banker_toolbench_eval")
    if boundary.get("role") != ROLE:
        errors.append("evaluation role must remain primary_eval")
    revision = boundary.get("dataset_revision")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        errors.append("dataset_revision must be immutable")
    license_info = boundary.get("license")
    if not isinstance(license_info, Mapping):
        errors.append("license metadata is missing")
    else:
        if _looks_placeholder(license_info.get("id")):
            errors.append("license id is placeholder or unrecorded")
        if not _immutable_reference(license_info.get("receipt_ref")):
            errors.append("license receipt_ref must be immutable and non-placeholder")
    held = boundary.get("held_out")
    if not isinstance(held, Mapping):
        errors.append("held_out semantics are missing")
    else:
        ref = held.get("receipt_ref")
        proven = held.get("receipt_proven") is True
        if proven != bool(ref):
            errors.append("held-out proof status must match its receipt reference")
        if ref is not None and not _immutable_reference(ref):
            errors.append("held-out receipt_ref must be immutable and non-placeholder")
    task_hashes = boundary.get("task_id_hashes")
    if not _is_sequence(task_hashes) or not task_hashes:
        errors.append("task_id_hashes must be a non-empty ordered list")
    else:
        valid_task_hashes = [
            value for value in task_hashes if isinstance(value, str) and _DIGEST.fullmatch(value)
        ]
        if len(valid_task_hashes) != len(task_hashes):
            errors.append("task_id_hashes must contain SHA-256 digests")
        if len(set(valid_task_hashes)) != len(valid_task_hashes):
            errors.append("task_id_hashes contain duplicates")
        if len(valid_task_hashes) == len(task_hashes):
            if boundary.get("task_count") != len(task_hashes):
                errors.append("task_count does not match task_id_hashes")
            if boundary.get("task_id_aggregate_sha256") != _sha256("\n".join(task_hashes)):
                errors.append("task_id_aggregate_sha256 does not match ordered task hashes")
            if boundary.get("task_ids_sha256") != _sha256("\n".join(task_hashes)):
                errors.append("task_ids_sha256 does not match ordered task hashes")
            if boundary.get("task_hashes") != list(task_hashes):
                errors.append("task_hashes does not match task_id_hashes")
            if boundary.get("split_manifest_sha256") != split_manifest_hash(task_hashes):
                errors.append("split_manifest_sha256 does not match the boundary")
    for name, required in (
        ("environment_contract", ("environment_id", "environment_revision", "native", "stateful", "reset_protocol", "tool_api")),
        ("artifact_contract", ("required", "artifact_types", "artifact_receipt_ref")),
        ("verifier_contract", ("verifier_id", "verifier_revision", "deterministic", "checks", "verifier_receipt_ref")),
    ):
        _, contract_errors = _validate_contract(boundary.get(name), name=name.removesuffix("_contract"), required=required)
        errors.extend(contract_errors)
    result = boundary.get("result_receipts")
    if not isinstance(result, Mapping):
        errors.append("result_receipts metadata is missing")
    elif result.get("recorded") is not True:
        errors.append("W&B, Tinker, and HF result receipts are unrecorded")
    else:
        _, result_errors = _result_receipt_fields(result.get("providers"))
        errors.extend(result_errors)
    expected_errors = sorted(set(errors))
    if boundary.get("errors") != expected_errors:
        errors.append("errors field does not match boundary validation errors")
    return sorted(set(errors))


def build_banker_toolbench_eval_boundary(
    revision: str,
    license_id: str,
    license_receipt_ref: str,
    tasks: Iterable[Any] | Mapping[str, Any] | str | Path | None = None,
    *,
    task_ids: Sequence[str] | None = None,
    task_id_hashes: Sequence[str] | None = None,
    environment_contract: Mapping[str, Any] | None = None,
    artifact_contract: Mapping[str, Any] | None = None,
    verifier_contract: Mapping[str, Any] | None = None,
    result_receipts: Mapping[str, Any] | None = None,
    held_out_receipt_ref: str | None = None,
    held_out_claimed: bool = False,
    source_identity: Mapping[str, Any] | None = None,
    role: str = ROLE,
) -> dict[str, Any]:
    """Build the exact E4 boundary from caller-supplied metadata only."""

    _require_exact_source(source_identity)
    if role != ROLE:
        raise BankerToolBenchBoundaryError("BankerToolBench boundary role must be primary_eval")
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        raise BankerToolBenchBoundaryError("revision must be an immutable 40-character lower-case commit SHA")
    if _looks_placeholder(license_id):
        raise BankerToolBenchBoundaryError("license_id must be recorded, not a placeholder")
    if not _immutable_reference(license_receipt_ref):
        raise BankerToolBenchBoundaryError("license_receipt_ref must be immutable and non-placeholder")
    if task_ids is not None:
        if tasks is not None or task_id_hashes is not None:
            raise BankerToolBenchBoundaryError("supply task_ids, tasks, or task_id_hashes, not multiple")
        tasks = task_ids
    if held_out_claimed and held_out_receipt_ref is None:
        # A claim without proof is a contract error, not an implicit holdout.
        held_out_error = "held-out claim requires an immutable receipt reference"
    else:
        held_out_error = None
    task_hash_values, task_errors = _extract_task_ids(tasks, task_id_hashes)
    environment, environment_errors = _validate_contract(
        environment_contract,
        name="environment",
        required=("environment_id", "environment_revision", "native", "stateful", "reset_protocol", "tool_api"),
    )
    artifact, artifact_errors = _validate_contract(
        artifact_contract,
        name="artifact",
        required=("required", "artifact_types", "artifact_receipt_ref"),
    )
    verifier, verifier_errors = _validate_contract(
        verifier_contract,
        name="verifier",
        required=("verifier_id", "verifier_revision", "deterministic", "checks", "verifier_receipt_ref"),
    )
    receipts, receipt_errors = _result_receipt_fields(result_receipts)
    errors = [*task_errors, *environment_errors, *artifact_errors, *verifier_errors, *receipt_errors]
    if held_out_error:
        errors.append(held_out_error)
    held_out_ref_valid = held_out_receipt_ref is None or _immutable_reference(held_out_receipt_ref)
    if held_out_receipt_ref is not None and not held_out_ref_valid:
        errors.append("held-out receipt reference must be immutable and non-placeholder")
    held_out_proven = held_out_receipt_ref is not None and held_out_ref_valid and not held_out_error
    boundary = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not errors else "BLOCKED",
        "boundary_status": "READY" if not (task_errors or environment_errors or artifact_errors or verifier_errors) else "BLOCKED",
        "launch_authorized": False,
        "launches_any_job": False,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "split": SPLIT_DESCRIPTION,
        "domains": list(EXPECTED_DOMAINS),
        "source_identity": {
            "kind": SOURCE_KIND,
            "dataset_id": DATASET_ID,
            "url": SOURCE_URL,
        },
        "dataset_revision": revision,
        "license": {
            "id": str(license_id),
            "receipt_ref": license_receipt_ref.strip(),
        },
        "task_count": len(task_hash_values),
        "task_id_hashes": task_hash_values,
        "task_hashes": task_hash_values,
        "task_id_aggregate_sha256": _sha256("\n".join(task_hash_values)) if task_hash_values else None,
        "task_ids_sha256": _sha256("\n".join(task_hash_values)) if task_hash_values else None,
        "split_manifest_sha256": split_manifest_hash(task_hash_values) if task_hash_values else None,
        "environment_contract": environment,
        "artifact_contract": artifact,
        "verifier_contract": verifier,
        "held_out": {
            "role": ROLE,
            "primary_eval": True,
            "receipt_proven": held_out_proven,
            "receipt_ref": held_out_receipt_ref,
        },
        "result_receipts": receipts,
        "result_claims": None,
        "errors": sorted(set(errors)),
        "evidence_scope": "evaluation boundary and receipts only; no benchmark result is asserted",
    }
    return boundary


# Common names used by local manifest callers.
build_boundary = build_banker_toolbench_eval_boundary
build_manifest = build_banker_toolbench_eval_boundary
generate_boundary = build_banker_toolbench_eval_boundary
build_banker_toolbench_manifest = build_banker_toolbench_eval_boundary


def validate_boundary(boundary: Mapping[str, Any]) -> list[str]:
    if not isinstance(boundary, Mapping):
        return ["boundary must be a JSON object"]
    try:
        _assert_metadata_only(boundary)
    except BankerToolBenchBoundaryError as exc:
        return [str(exc)]
    return _boundary_errors(boundary)


def verify_boundary(boundary: Mapping[str, Any] | str | Path) -> bool:
    if isinstance(boundary, (str, Path)):
        boundary = _read_json(boundary)
    errors = validate_boundary(boundary)
    if errors:
        raise BankerToolBenchBoundaryError("invalid BankerToolBench boundary: " + "; ".join(errors))
    if boundary.get("status") != "READY":
        raise BankerToolBenchBoundaryError("BankerToolBench boundary is blocked")
    return True


validate_manifest = validate_boundary
verify_manifest = verify_boundary
verify_banker_toolbench_eval = verify_boundary
BankerToolBenchEvalAdapterError = BankerToolBenchBoundaryError


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
    parser.add_argument("--environment-contract", type=Path)
    parser.add_argument("--artifact-contract", type=Path)
    parser.add_argument("--verifier-contract", type=Path)
    parser.add_argument("--result-receipts", type=Path)
    parser.add_argument("--held-out-receipt-ref")
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
        result_receipts = _read_json(args.result_receipts) if args.result_receipts else None
        boundary = build_banker_toolbench_eval_boundary(
            args.revision,
            args.license_id,
            args.license_receipt_ref,
            args.task_id,
            environment_contract=environment,
            artifact_contract=artifact,
            verifier_contract=verifier,
            result_receipts=result_receipts,
            held_out_receipt_ref=args.held_out_receipt_ref,
        )
        _write_json(boundary, args.out)
        return 0 if boundary["status"] == "READY" else 2
    except (BankerToolBenchBoundaryError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
