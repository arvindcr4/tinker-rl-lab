"""Offline validator for the E12/E13 Pavlov evaluation boundary adapter.

The validator is deliberately strict and pure:
- requires exactly the E12 and E13 boundaries in the expected order and naming
- binds each boundary to an authoritative source and immutable revision
- validates deterministic task IDs and split hashes for each boundary
- requires native artifact/environment/verifier receipts for each boundary
- requires explicit stateful trajectory and disallows paid launch flags
- rejects benchmark substitution toward xLAM or related sources
- requires W&B, Tinker, and Hugging Face receipt fields
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[0-9A-Za-z]{8}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

PRIMARY_EVAL = "primary_eval"
RECEIPT_PROVEN_HELDOUT = "receipt_proven_heldout"

BANNED_SOURCE_MARKERS = {
    "xlam",
    "xlama",
    "related benchmark",
    "related-benchmark",
    "related_benchmark",
}

BOUNDARY_SPECS: dict[str, dict[str, Any]] = {
    "E12": {
        "expected_name": "appbench_eval",
        "expected_role": RECEIPT_PROVEN_HELDOUT,
        "required_source_marker": "appbench",
    },
    "E13": {
        "expected_name": "openreward_games_eval",
        "expected_role": PRIMARY_EVAL,
        "required_source_marker": "openreward",
    },
}


class PavlovAppbenchOpenrewardGamesAdapterError(ValueError):
    """Raised when a Pavlov adapter contract payload is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PavlovAppbenchOpenrewardGamesAdapterError(message)


def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), f"{label} must be a mapping")
    return payload


def _require_text(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    _require(value.strip(), f"{label} cannot be empty")
    return value


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    hex_value = _require_text(value, label)
    _require(pattern.fullmatch(hex_value) is not None, f"{label} must be a hex digest of length {40 if pattern is HEX40 else 64}")
    _require(
        len(set(hex_value)) > 1,
        f"{label} must not be an all-identical placeholder digest",
    )
    return hex_value


def _require_path(value: Any, label: str) -> str:
    path = _require_text(value, label)
    _require("\n" not in path and "\r" not in path, f"{label} must be a valid path-like string")
    return path


def _validate_task_ids(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), f"{label}.task_ids must be a list")
    _require(len(value) > 0, f"{label}.task_ids cannot be empty")
    task_ids: list[str] = []
    for index, task_id in enumerate(value):
        task_ids.append(_require_hex(task_id, HEX64, f"{label}.task_ids[{index}]"))
    _require(len(task_ids) == len(set(task_ids)), f"{label}.task_ids must be unique")
    return task_ids


def _deterministic_split_hash(task_ids: list[str]) -> str:
    payload = json.dumps(sorted(task_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_split_hash(value: Any, task_ids: list[str], label: str) -> str:
    split_hash = _require_hex(value, HEX64, f"{label}.split_hash")
    expected = _deterministic_split_hash(task_ids)
    _require(
        split_hash == expected,
        f"{label}.split_hash is not the deterministic hash of task_ids",
    )
    return split_hash


def _validate_source(value: Any, label: str, required_marker: str) -> str:
    source = _require_text(value, label)
    lowered = source.lower()
    for marker in BANNED_SOURCE_MARKERS:
        _require(marker not in lowered, f"{label} references blocked source marker: {marker}")
    _require(required_marker in lowered, f"{label} must identify the authoritative source {required_marker!r}")
    return source


def _validate_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    sha256 = _require_hex(payload.get("sha256"), HEX64, f"{label}.sha256")
    path = payload.get("path")
    if path is None:
        return {"sha256": sha256}
    path_text = _require_path(path, f"{label}.path")
    return {"sha256": sha256, "path": path_text}


def _validate_wandb_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    _require_text(payload.get("project"), f"{label}.project")
    _require_text(payload.get("entity"), f"{label}.entity")
    run_id = _require_text(payload.get("run_id"), f"{label}.run_id")
    _require(RUN_ID.fullmatch(run_id) is not None, f"{label}.run_id must be 8 alphanumeric characters")
    run_url = _require_text(payload.get("run_url"), f"{label}.run_url")
    _require(run_url.startswith("https://wandb.ai/"), f"{label}.run_url must point to wandb.ai")
    return {
        "project": str(payload["project"]),
        "entity": str(payload["entity"]),
        "run_id": run_id,
        "run_url": run_url,
    }


def _validate_tinker_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    job_id = _require_text(payload.get("job_id"), f"{label}.job_id")
    _require(UUID.fullmatch(job_id) is not None, f"{label}.job_id must be a hyphenated UUID string")
    receipt_sha = _require_hex(payload.get("receipt_sha256"), HEX64, f"{label}.receipt_sha256")
    return {"job_id": job_id, "receipt_sha256": receipt_sha}


def _validate_hf_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    repo_id = _require_text(payload.get("repo_id"), f"{label}.repo_id")
    _require(REPO.fullmatch(repo_id) is not None, f"{label}.repo_id must be owner/repo")
    commit = _require_hex(payload.get("commit"), HEX40, f"{label}.commit")
    return {
        "repo_id": repo_id,
        "commit": commit,
        "receipt_sha256": _require_hex(payload.get("receipt_sha256"), HEX64, f"{label}.receipt_sha256"),
    }


def _validate_environment_contract(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    return {
        "container": _require_hex(
            payload.get("container"),
            HEX64,
            f"{label}.container",
        ),
        "decontamination": _require_hex(
            payload.get("decontamination"),
            HEX64,
            f"{label}.decontamination",
        ),
        "container_source": _require_text(payload.get("container_source"), f"{label}.container_source"),
        "decontamination_source": _require_text(
            payload.get("decontamination_source"),
            f"{label}.decontamination_source",
        ),
    }


def _validate_verifier_contract(value: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(value, label)
    verifier = {
        "artifact_verifier_receipt": _validate_receipt(
            payload.get("artifact_verifier_receipt"),
            f"{label}.artifact_verifier_receipt",
        ),
        "state_verifier_receipt": _validate_receipt(
            payload.get("state_verifier_receipt"),
            f"{label}.state_verifier_receipt",
        ),
        "wandb": _validate_wandb_receipt(payload.get("wandb"), f"{label}.wandb"),
        "tinker": _validate_tinker_receipt(payload.get("tinker"), f"{label}.tinker"),
        "hugging_face": _validate_hf_receipt(payload.get("hugging_face"), f"{label}.hugging_face"),
    }

    artifact_source = payload.get("artifact")
    if artifact_source is not None:
        verifier["artifact"] = _require_text(artifact_source, f"{label}.artifact")
    return verifier


def _validate_native_contract(value: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(value, label)
    return {
        "artifact": {
            "sha256": _require_hex(payload.get("artifact_sha256"), HEX64, f"{label}.artifact_sha256"),
            "source": _require_text(payload.get("artifact_source"), f"{label}.artifact_source"),
            "size_bytes": int(_require_non_negative_int(payload.get("artifact_size_bytes"), f"{label}.artifact_size_bytes")),
        },
        "environment": _validate_environment_contract(
            payload.get("environment"),
            f"{label}.environment",
        ),
        "verifier": _validate_verifier_contract(payload.get("verifier"), f"{label}.verifier"),
    }


def _require_non_negative_int(value: Any, label: str) -> int:
    _require(isinstance(value, int), f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _validate_boundary(action: str, payload: Mapping[str, Any], spec: Mapping[str, str]) -> dict[str, Any]:
    label = f"boundaries[{action}]"
    name = _require_text(payload.get("name"), f"{label}.name")
    _require(name == spec["expected_name"], f"{label}.name must be {spec['expected_name']!r}")
    source = _validate_source(payload.get("authoritative_source"), f"{label}.authoritative_source", spec["required_source_marker"])
    revision = _require_hex(payload.get("revision"), HEX40, f"{label}.revision")
    role = _require_text(payload.get("evaluation_role"), f"{label}.evaluation_role")
    _require(role == spec["expected_role"], f"{label}.evaluation_role must be {spec['expected_role']!r}")
    task_ids = _validate_task_ids(payload.get("task_ids"), label)
    split_hash = _validate_split_hash(payload.get("split_hash"), task_ids, label)
    license_receipt = _validate_receipt(payload.get("license"), f"{label}.license")
    native_contract = _validate_native_contract(payload.get("native_contract"), f"{label}.native_contract")
    return {
        "name": name,
        "authoritative_source": source,
        "revision": revision,
        "evaluation_role": role,
        "task_ids": task_ids,
        "split_hash": split_hash,
        "license": license_receipt,
        "native_contract": native_contract,
    }


def _normalize_boundaries(boundaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    present = set(boundaries)
    required = set(BOUNDARY_SPECS)
    _require(present == required, f"boundaries must contain exactly {sorted(required)}")

    normalized: dict[str, Any] = {}
    seen_task_ids: set[str] = set()
    for action_id in sorted(required):
        raw = _require_mapping(boundaries[action_id], f"boundaries[{action_id}]")
        normalized[action_id] = _validate_boundary(action_id, raw, BOUNDARY_SPECS[action_id])

        action_task_ids = set(normalized[action_id]["task_ids"])
        _require(
            seen_task_ids.isdisjoint(action_task_ids),
            f"boundaries[{action_id}].task_ids must not overlap with other boundaries",
        )
        seen_task_ids.update(action_task_ids)

    return normalized


def validate_appbench_openreward_games_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a Pavlov contract containing the E12 and E13 offline boundaries."""

    contract = _require_mapping(payload, "contract")
    paid_launch_allowed = contract.get("paid_launch_allowed")
    _require(isinstance(paid_launch_allowed, bool), "paid_launch_allowed must be a boolean")
    _require(not paid_launch_allowed, "paid_launch_allowed must be False")
    _require(
        contract.get("stateful_trajectory") is True,
        "stateful_trajectory must be true",
    )

    boundaries = _require_mapping(contract.get("boundaries"), "boundaries")
    normalized_boundaries = _normalize_boundaries(boundaries)

    return {
        "paid_launch_allowed": False,
        "stateful_trajectory": True,
        "boundaries": normalized_boundaries,
    }


validate_appbench_contract = validate_appbench_openreward_games_contract
validate_pavlov_appbench_openreward_games_contract = validate_appbench_openreward_games_contract
validate_pavlov_openreward_games_adapter = validate_appbench_openreward_games_contract


__all__ = [
    "PavlovAppbenchOpenrewardGamesAdapterError",
    "validate_appbench_openreward_games_contract",
    "validate_appbench_contract",
    "validate_pavlov_appbench_openreward_games_contract",
    "validate_pavlov_openreward_games_adapter",
]
