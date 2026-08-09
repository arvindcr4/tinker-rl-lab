"""Offline validator for OpenReward train/eval contracts.

The validator is intentionally strict and side-effect free:
- exact ``openreward_train`` and ``openreward_games_eval`` entries are required
- each entry must declare a pinned 40-hex revision and explicit role
- task hashes are 64-hex and non-overlapping across train/eval
- trajectory must be explicitly stateful
- artifact/state verifier receipts are required
- license/container/decontamination receipt hashes are required
- paid launches are permanently disallowed

This module has no network side effects and validates only in-memory payloads.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

TRAIN_ROLE = "train"
EVAL_ROLE = "primary_eval"


class OpenrewardSchemaError(ValueError):
    """Raised when an OpenReward contract payload is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OpenrewardSchemaError(message)


def _require_mapping(payload: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), f"{label} must be a mapping")
    return payload


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    _require(pattern.fullmatch(value) is not None, f"{label} must be hex of length {40 if pattern is HEX40 else 64}")
    return value


def _require_receipt_payload(value: Any, label: str) -> str:
    receipt = _require_mapping(value, label)
    return _require_hex(receipt.get("sha256"), HEX64, f"{label}.sha256")


def _validate_task_hashes(task_hashes: Any, label: str) -> list[str]:
    _require(isinstance(task_hashes, list), f"{label}.task_hashes must be a list")
    _require(len(task_hashes) > 0, f"{label}.task_hashes cannot be empty")
    _require(len(task_hashes) == len(set(task_hashes)), f"{label}.task_hashes must contain unique hashes")
    normalized = []
    for index, value in enumerate(task_hashes):
        _require_hex(value, HEX64, f"{label}.task_hashes[{index}]")
        normalized.append(value)
    return normalized


def _validate_split(payload: Mapping[str, Any], key: str, expected_role: str) -> dict[str, Any]:
    split = _require_mapping(payload.get(key), key)
    revision = _require_hex(split.get("revision"), HEX40, f"{key}.revision")
    role = split.get("role")
    _require(role == expected_role, f"{key}.role must be {expected_role!r}, got {role!r}")
    task_hashes = _validate_task_hashes(split.get("task_hashes"), key)
    return {
        "revision": revision,
        "role": role,
        "task_hashes": task_hashes,
    }


def validate_openreward_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an offline OpenReward contract payload and return a normalized copy."""

    contract = _require_mapping(payload, "contract")

    paid_launch_allowed = contract.get("paid_launch_allowed")
    _require(
        isinstance(paid_launch_allowed, bool),
        "paid_launch_allowed must be a boolean",
    )
    _require(not paid_launch_allowed, "paid_launch_allowed must be False")

    train = _validate_split(contract, "openreward_train", TRAIN_ROLE)
    games_eval = _validate_split(contract, "openreward_games_eval", EVAL_ROLE)

    train_hashes = set(train["task_hashes"])
    eval_hashes = set(games_eval["task_hashes"])
    _require(train_hashes.isdisjoint(eval_hashes), "task hashes between train and primary_eval must be non-overlapping")

    _require(contract.get("stateful_trajectory") is True, "stateful_trajectory must be true")

    state_receipt = _require_receipt_payload(contract.get("state_verifier_receipt"), "state_verifier_receipt")
    artifact_receipt = _require_receipt_payload(contract.get("artifact_verifier_receipt"), "artifact_verifier_receipt")

    decontam_hash = _require_receipt_payload(contract.get("decontamination"), "decontamination")
    container_hash = _require_receipt_payload(contract.get("container"), "container")
    license_hash = _require_receipt_payload(contract.get("license"), "license")

    return {
        "paid_launch_allowed": False,
        "openreward_train": train,
        "openreward_games_eval": games_eval,
        "stateful_trajectory": True,
        "artifact_verifier_receipt": {"sha256": artifact_receipt},
        "state_verifier_receipt": {"sha256": state_receipt},
        "license": {"sha256": license_hash},
        "container": {"sha256": container_hash},
        "decontamination": {"sha256": decontam_hash},
    }


validate_openreward_offline_contract = validate_openreward_contract
validate_pavlov_openreward_schema = validate_openreward_contract


__all__ = [
    "OpenrewardSchemaError",
    "validate_openreward_contract",
    "validate_openreward_offline_contract",
    "validate_pavlov_openreward_schema",
]
