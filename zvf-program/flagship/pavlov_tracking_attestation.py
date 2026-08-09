#!/usr/bin/env python3
"""Validate W&B/Tinker/Hugging Face tracking receipts offline.

This module performs deterministic, local-only checks and never:
- reads credentials
- launches jobs
- performs network calls
- uploads artifacts
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "missing",
        "none",
        "null",
        "unknown",
        "pending",
        "placeholder",
        "todo",
        "unset",
        "to_be_pinned_before_paid_runs",
        "receipt",
        "license-receipt",
    }
)

_IMMUTABLE_GIT_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_WANDW_RE = re.compile(
    r"^https://wandb\.ai/(?P<entity>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<project>[A-Za-z0-9][A-Za-z0-9._-]*)/runs/(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_REPO_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_COMMIT_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<org>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)/commit/(?P<revision>[0-9a-fA-F]{40})$"
)

WANDB_TERMINAL_STATES = frozenset({"finished", "completed"})
TINKER_TERMINAL_STATES = frozenset({"completed", "finished"})
HUGGING_FACE_VISIBILITIES = frozenset({"public", "private"})
HUGGING_FACE_CHECKPOINT_KINDS = frozenset({"initial", "periodic", "final"})
ALLOWED_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for digesting record fields."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for a plain-text payload."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _PLACEHOLDER_VALUES


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and not _is_placeholder(value) and bool(value.strip())


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and not _is_placeholder(record[name]):
            return record[name]
    return None


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(ALLOWED_SHA256_HEX.fullmatch(value.lower()))


def _hf_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_IMMUTABLE_GIT_SHA40_RE.fullmatch(value))


def _canonical_checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_url": _first_value(
            checkpoint, ("repo_url", "repo", "repository", "huggingface_repo")
        ),
        "revision": _first_value(checkpoint, ("revision", "commit", "sha")),
        "kind": str(_first_value(checkpoint, ("kind", "stage"))).strip().lower(),
        "step": checkpoint.get("step"),
        "visibility": _first_value(checkpoint, ("visibility",)),
        "safe_public_artifact": checkpoint.get("safe_public_artifact"),
        "url": _first_value(
            checkpoint, ("url", "checkpoint_url", "repo_revision_url")
        ),
    }


def compute_checkpoint_content_digest(checkpoint: Mapping[str, Any]) -> str:
    """Return the canonical digest this module binds to each checkpoint entry."""

    return sha256_json(_canonical_checkpoint_payload(checkpoint))


def _validate_wandb_receipt(receipt: Any, errors: list[str]) -> None:
    if not isinstance(receipt, Mapping):
        errors.append("wandb_run_identity must be a JSON object")
        return

    if receipt.get("online") is not True:
        errors.append("wandb_run_identity.online must be the boolean true")

    if receipt.get("acknowledged") is not True:
        errors.append("wandb_run_identity.acknowledged must be the boolean true")

    run_id = _first_value(receipt, ("run_id", "id"))
    if not _nonempty_text(run_id) or len(str(run_id).strip()) < 3:
        errors.append("wandb_run_identity.run_id must be a non-placeholder string")
    elif " " in str(run_id):
        errors.append("wandb_run_identity.run_id cannot contain whitespace")

    run_url = _first_value(receipt, ("run_url", "url", "link"))
    if not isinstance(run_url, str):
        errors.append("wandb_run_identity.run_url must be a HTTPS URL")
        return
    match = _WANDW_RE.fullmatch(run_url.strip())
    if not match:
        errors.append("wandb_run_identity.run_url has invalid wandb host/path shape")
        return
    if str(run_id) != match.group("run_id"):
        errors.append("wandb_run_identity.run_url run_id path component must match run_id")
    for key in ("entity", "project"):
        value = _first_value(receipt, (key,))
        if value is not None and str(value).strip() and str(value).strip() != match.group(key):
            errors.append(f"wandb_run_identity.{key} does not match run_url segment")

    state = _first_value(receipt, ("state", "status"))
    if not _nonempty_text(state) or str(state).strip().lower() not in WANDB_TERMINAL_STATES:
        errors.append(
            f"wandb_run_identity.state must be terminal: {sorted(WANDB_TERMINAL_STATES)}"
        )


def _validate_tinker_receipt(receipt: Any, errors: list[str]) -> None:
    if not isinstance(receipt, Mapping):
        errors.append("tinker_run_identity must be a JSON object")
        return

    run_id = _first_value(receipt, ("run_id", "id"))
    if not isinstance(run_id, str) or not _nonempty_text(run_id) or len(run_id.strip()) < 3:
        errors.append("tinker_run_identity.run_id must be a non-placeholder string")
    elif _is_placeholder(run_id):
        errors.append("tinker_run_identity.run_id must not be placeholder-like")

    state = _first_value(receipt, ("state", "status"))
    if state is not None:
        if not _nonempty_text(state) or str(state).strip().lower() not in TINKER_TERMINAL_STATES:
            errors.append(
                "tinker_run_identity.state/status, when present, must be terminal "
                f"({', '.join(sorted(TINKER_TERMINAL_STATES))})"
            )


def _validate_hf_checkpoint_list(checkpoints: Any, errors: list[str]) -> None:
    if not isinstance(checkpoints, list) or not checkpoints:
        errors.append("hf_checkpoints must be a non-empty list")
        return

    seen_pairs: set[tuple[str, int]] = set()
    kinds_seen: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        prefix = f"hf_checkpoints[{index}]"
        if not isinstance(checkpoint, Mapping):
            errors.append(f"{prefix} must be a JSON object")
            continue

        repo_url = _first_value(
            checkpoint, ("repo_url", "repo", "repository", "huggingface_repo")
        )
        if not isinstance(repo_url, str) or not _HF_REPO_URL_RE.fullmatch(repo_url.strip()):
            errors.append(f"{prefix}.repo_url must be an https Hugging Face repository URL")
            continue

        revision = _first_value(checkpoint, ("revision", "commit", "sha"))
        if not _hf_commit(revision):
            errors.append(f"{prefix}.revision must be an immutable 40-hex commit")
            continue

        checkpoint_url = _first_value(
            checkpoint, ("url", "checkpoint_url", "repo_revision_url")
        )
        commit_match = _HF_COMMIT_URL_RE.fullmatch(str(checkpoint_url).strip()) if isinstance(checkpoint_url, str) else None
        if not commit_match:
            errors.append(
                f"{prefix}.url must be an https Hugging Face commit URL for the same repo"
            )
            continue
        if commit_match.group("org") != _HF_REPO_URL_RE.fullmatch(repo_url.strip()).group("org"):
            errors.append(f"{prefix}.url repo owner does not match repo_url")
            continue
        if commit_match.group("repo") != _HF_REPO_URL_RE.fullmatch(repo_url.strip()).group("repo"):
            errors.append(f"{prefix}.url repo name does not match repo_url")
            continue
        if commit_match.group("revision") != revision:
            errors.append(f"{prefix}.url commit must match revision")
            continue

        visibility = _first_value(checkpoint, ("visibility",))
        if visibility not in HUGGING_FACE_VISIBILITIES:
            errors.append(
                f"{prefix}.visibility must be one of: {sorted(HUGGING_FACE_VISIBILITIES)}"
            )
            continue

        safe_public_artifact = checkpoint.get("safe_public_artifact")
        if not isinstance(safe_public_artifact, bool):
            errors.append(f"{prefix}.safe_public_artifact must be a boolean")
            continue
        if visibility == "public" and safe_public_artifact is not True:
            errors.append(
                f"{prefix}.safe_public_artifact must be true when visibility is public"
            )
        if visibility == "private" and safe_public_artifact is not False:
            errors.append(
                f"{prefix}.safe_public_artifact must be false when visibility is private"
            )

        kind = _first_value(checkpoint, ("kind", "stage"))
        if not isinstance(kind, str) or kind.strip().lower() not in HUGGING_FACE_CHECKPOINT_KINDS:
            errors.append(
                f"{prefix}.kind must be one of: {sorted(HUGGING_FACE_CHECKPOINT_KINDS)}"
            )
            continue
        kind = kind.strip().lower()

        step = checkpoint.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            errors.append(f"{prefix}.step must be a non-negative integer")
            continue

        pair = (kind, step)
        if pair in seen_pairs:
            errors.append(f"{prefix} has duplicate checkpoint (kind,step) pair: {pair}")
        seen_pairs.add(pair)
        kinds_seen.add(kind)

        content_digest = _first_value(checkpoint, ("content_digest", "checkpoint_digest"))
        if not _sha256(content_digest):
            errors.append(
                f"{prefix}.content_digest must be a lowercase 64-char sha256 hex string"
            )
            continue
        expected = compute_checkpoint_content_digest(checkpoint)
        if str(content_digest).lower() != expected:
            errors.append(
                f"{prefix}.content_digest must match canonical content digest binding"
            )

    if kinds_seen != HUGGING_FACE_CHECKPOINT_KINDS:
        missing = sorted(HUGGING_FACE_CHECKPOINT_KINDS - kinds_seen)
        errors.append(
            f"hf_checkpoints must cover all kinds initial/periodic/final; missing: {missing}"
        )


def validate_tracking_attestation(record: Mapping[str, Any] | None) -> list[str]:
    """Validate one tracking receipt record and return all validation errors."""

    if record is None:
        return ["record is missing"]
    if not isinstance(record, Mapping):
        return ["record must be a JSON object"]

    errors: list[str] = []
    wandb = _first_value(record, ("wandb_run_identity", "wandb", "wandb_receipt"))
    if wandb is None:
        errors.append("wandb_run_identity is required")
    else:
        _validate_wandb_receipt(wandb, errors)

    tinker = _first_value(record, ("tinker_run_identity", "tinker", "tinker_receipt"))
    if tinker is None:
        errors.append("tinker_run_identity is required")
    else:
        _validate_tinker_receipt(tinker, errors)

    checkpoints = _first_value(record, ("hf_checkpoints", "huggingface", "hf_publication"))
    if checkpoints is None:
        errors.append("hf_checkpoints is required")
    else:
        _validate_hf_checkpoint_list(checkpoints, errors)

    return errors


def validate_tracking_records(records: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> list[str]:
    """Validate one or many records and return all errors."""

    if isinstance(records, Mapping):
        return validate_tracking_attestation(records)
    if not isinstance(records, Iterable):
        return ["records must be a record or list of records"]

    all_errors: list[str] = []
    for index, record in enumerate(records):
        for error in validate_tracking_attestation(record):
            all_errors.append(f"record[{index}] {error}")
    return all_errors


def is_valid_tracking_attestation(record: Mapping[str, Any] | None) -> bool:
    """Return True when a single record contains no validation blockers."""

    return not validate_tracking_attestation(record)


__all__ = [
    "canonical_json",
    "compute_checkpoint_content_digest",
    "is_valid_tracking_attestation",
    "sha256_json",
    "sha256_text",
    "validate_tracking_attestation",
    "validate_tracking_records",
    "WANDB_TERMINAL_STATES",
    "TINKER_TERMINAL_STATES",
]
