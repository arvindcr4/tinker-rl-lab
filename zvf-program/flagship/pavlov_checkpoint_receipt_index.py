#!/usr/bin/env python3
"""Deterministic, offline indexing of checkpoint receipts for Pavlov runs.

The module validates only in-memory receipt records.  It does not call W&B,
Tinker, Hugging Face APIs, perform uploads, or launch any jobs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REQUIRED_CHECKPOINT_KINDS: tuple[str, ...] = ("initial", "periodic", "final")
_CHECKPOINT_KIND_ORDER = {
    kind: index for index, kind in enumerate(REQUIRED_CHECKPOINT_KINDS)
}
SCHEMA_VERSION = "pavlov-checkpoint-receipt-index-v1"

_PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "missing",
        "none",
        "null",
        "pending",
        "placeholder",
        "todo",
        "unset",
        "unknown",
        "to_be_pinned_before_paid_runs",
        "receipt",
        "license-receipt",
    }
)

_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WANDB_URL_RE = re.compile(
    r"^https://wandb\.ai/(?P<entity>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<project>[A-Za-z0-9][A-Za-z0-9._-]*)/runs/(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_REPO_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_COMMIT_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)/commit/(?P<revision>[0-9a-fA-F]{40})$"
)

WANDB_TERMINAL_STATES = frozenset({"finished", "completed"})
TINKER_TERMINAL_STATES = frozenset({"completed", "finished"})
HUGGING_FACE_VISIBILITIES = frozenset({"public", "private"})


def canonical_json(value: Any) -> str:
    """Return canonical JSON for deterministic hashing and indexing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a SHA-256 hex digest for plain text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 hex digest for stable JSON text."""

    return sha256_text(canonical_json(value))


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _PLACEHOLDER_VALUES


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and not _is_placeholder(value) and bool(value.strip())


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA40_RE.fullmatch(value.strip()))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip().lower()))


def compute_checkpoint_content_digest(checkpoint: Mapping[str, Any]) -> str:
    """Compute the canonical content digest for a checkpoint attestation."""

    payload = _canonical_checkpoint_payload(checkpoint)
    return sha256_json(payload)


def _canonical_checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo": _first_value(checkpoint, ("repo", "repo_url", "repository", "huggingface_repo")),
        "revision": str(_first_value(checkpoint, ("revision", "commit", "sha", "hash")).strip().lower())
        if isinstance(_first_value(checkpoint, ("revision", "commit", "sha", "hash")), str)
        else _first_value(checkpoint, ("revision", "commit", "sha", "hash")),
        "kind": str(_first_value(checkpoint, ("kind", "stage"))).strip().lower()
        if _first_value(checkpoint, ("kind", "stage")) is not None
        else None,
        "step": checkpoint.get("step"),
        "visibility": _first_value(checkpoint, ("visibility",)),
        "safe_public_artifact": checkpoint.get("safe_public_artifact"),
        "source": _first_value(checkpoint, ("source", "source_id", "suite", "suite_id")),
        "url": _first_value(checkpoint, ("url", "checkpoint_url", "repo_revision_url")),
    }


def _normalise_record(records_index: int, record: Mapping[str, Any]) -> str:
    source = _first_value(
        record,
        ("source", "source_id", "suite", "suite_id", "name", "registry_source", "run_source"),
    )
    if _nonempty_text(source):
        return str(source).strip()
    return f"record-{records_index}"


def _validate_wandb_identity(
    receipt: Any, record_prefix: str, errors: list[str], expected_run_id: str | None
) -> str | None:
    if not isinstance(receipt, Mapping):
        errors.append(f"{record_prefix}.wandb_run_identity must be an object")
        return None
    if receipt.get("online") is not True:
        errors.append(
            f"{record_prefix}.wandb_run_identity.online must be boolean true"
        )
    if receipt.get("acknowledged") is not True:
        errors.append(
            f"{record_prefix}.wandb_run_identity.acknowledged must be boolean true"
        )
    run_id = _first_value(receipt, ("run_id", "id"))
    if not _nonempty_text(run_id):
        errors.append(f"{record_prefix}.wandb_run_identity.run_id must be a non-placeholder string")
        run_id = None
    elif _is_placeholder(run_id) or " " in str(run_id):
        errors.append(
            f"{record_prefix}.wandb_run_identity.run_id must be non-empty and non-whitespace"
        )
        run_id = None

    run_url = _first_value(receipt, ("run_url", "url", "link"))
    if not isinstance(run_url, str):
        errors.append(f"{record_prefix}.wandb_run_identity.run_url must be HTTPS URL")
        run_url = ""
    match = _WANDB_URL_RE.fullmatch(str(run_url).strip()) if isinstance(run_url, str) else None
    if not match:
        errors.append(
            f"{record_prefix}.wandb_run_identity.run_url has invalid wandb host/path shape"
        )
    else:
        if run_id is not None and str(run_id) != match.group("run_id"):
            errors.append(
                f"{record_prefix}.wandb_run_identity.run_id must match run_url segment"
            )
    state = _first_value(receipt, ("state", "status"))
    if not _nonempty_text(state) or str(state).strip().lower() not in WANDB_TERMINAL_STATES:
        errors.append(
            f"{record_prefix}.wandb_run_identity.state must be one of {sorted(WANDB_TERMINAL_STATES)}"
        )

    if expected_run_id is not None and run_id is not None and run_id != expected_run_id:
        errors.append(
            f"{record_prefix}.wandb_run_identity.run_id does not match expected run_id"
        )
    return str(run_id).strip() if run_id is not None else None


def _validate_tinker_identity(
    receipt: Any, record_prefix: str, errors: list[str], expected_run_id: str | None
) -> str | None:
    if not isinstance(receipt, Mapping):
        errors.append(f"{record_prefix}.tinker_run_identity must be an object")
        return None
    run_id = _first_value(receipt, ("run_id", "id"))
    if not _nonempty_text(run_id) or _is_placeholder(run_id) or " " in str(run_id):
        errors.append(
            f"{record_prefix}.tinker_run_identity.run_id must be a non-placeholder string"
        )
        run_id = None
    state = _first_value(receipt, ("state", "status"))
    if state is not None:
        if not _nonempty_text(state) or str(state).strip().lower() not in TINKER_TERMINAL_STATES:
            errors.append(
                f"{record_prefix}.tinker_run_identity.state/status must be one of {sorted(TINKER_TERMINAL_STATES)}"
            )
    if expected_run_id is not None and run_id is not None and run_id != expected_run_id:
        errors.append(
            f"{record_prefix}.tinker_run_identity.run_id does not match expected run_id"
        )
    return str(run_id).strip() if run_id is not None else None


def _validate_checkpoint(
    checkpoint: Any,
    record_prefix: str,
    record_source: str,
    checkpoint_prefix: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{record_prefix}{checkpoint_prefix} must be an object")
        return None

    repo = _first_value(checkpoint, ("repo", "repo_url", "repository", "huggingface_repo"))
    if not _nonempty_text(repo):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.repo/repo_url must be a non-placeholder string"
        )
        return None
    if not isinstance(repo, str) or not _HF_REPO_URL_RE.fullmatch(repo.strip()):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.repo/repo_url must be a Hugging Face repository URL"
        )
        return None

    commit = _first_value(checkpoint, ("revision", "commit", "sha", "hash"))
    if not _nonempty_text(commit) or not _sha40(commit):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.revision/commit must be an immutable 40-hex commit"
        )
        return None
    commit = str(commit).strip().lower()
    repo_match = _HF_REPO_URL_RE.fullmatch(repo.strip())
    assert repo_match is not None

    checkpoint_url = _first_value(checkpoint, ("url", "checkpoint_url", "repo_revision_url"))
    if not _nonempty_text(checkpoint_url):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.url/checkpoint_url must be non-empty"
        )
        return None
    if not isinstance(checkpoint_url, str):
        errors.append(f"{record_prefix}{checkpoint_prefix}.url must be HTTPS text")
        return None
    commit_match = _HF_COMMIT_URL_RE.fullmatch(checkpoint_url.strip())
    if not commit_match:
        if "/tree/" in checkpoint_url or "/blob/" in checkpoint_url:
            errors.append(
                f"{record_prefix}{checkpoint_prefix}.url must be a commit URL, not branch-only URL"
            )
        else:
            errors.append(
                f"{record_prefix}{checkpoint_prefix}.url must be https://huggingface.co/<org>/<repo>/commit/<40-hex>"
            )
        return None
    if commit.lower() != commit_match.group("revision").lower():
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.url commit must match revision/commit"
        )
        return None
    if (
        commit_match.group("owner") != repo_match.group("owner")
        or commit_match.group("repo") != repo_match.group("repo")
    ):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.url repo must match repo/repo_url"
        )
        return None

    kind = _first_value(checkpoint, ("kind", "stage"))
    if not isinstance(kind, str) or kind.strip().lower() not in REQUIRED_CHECKPOINT_KINDS:
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.kind must be one of: {sorted(REQUIRED_CHECKPOINT_KINDS)}"
        )
        return None
    kind = kind.strip().lower()

    step = checkpoint.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        errors.append(f"{record_prefix}{checkpoint_prefix}.step must be a non-negative integer")
        return None

    visibility = _first_value(checkpoint, ("visibility",))
    if visibility not in HUGGING_FACE_VISIBILITIES:
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.visibility must be one of {sorted(HUGGING_FACE_VISIBILITIES)}"
        )
        return None

    safe_public_artifact = checkpoint.get("safe_public_artifact")
    if not isinstance(safe_public_artifact, bool):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.safe_public_artifact must be a boolean"
        )
        return None
    if visibility == "public" and safe_public_artifact is not True:
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.safe_public_artifact must be true when visibility is public"
        )
    elif visibility == "private" and safe_public_artifact is not False:
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.safe_public_artifact must be false when visibility is private"
        )

    source = _first_value(checkpoint, ("source", "source_id", "suite", "suite_id"))
    if not _nonempty_text(source):
        source = record_source
    source = str(source)

    normalized = {
        "source": source,
        "kind": kind,
        "step": step,
        "repo": repo.strip(),
        "commit": commit,
        "url": checkpoint_url.strip(),
        "visibility": visibility,
        "safe_public_artifact": safe_public_artifact,
        "content_digest": None,
    }
    expected_digest = compute_checkpoint_content_digest(
        {
            "repo": normalized["repo"],
            "revision": normalized["commit"],
            "kind": kind,
            "step": step,
            "visibility": visibility,
            "safe_public_artifact": safe_public_artifact,
            "source": source,
            "url": normalized["url"],
        }
    )
    normalized["content_digest"] = expected_digest

    provided = _first_value(checkpoint, ("content_digest", "checkpoint_digest"))
    if not _nonempty_text(provided):
        errors.append(f"{record_prefix}{checkpoint_prefix}.content_digest is required")
    elif not _sha256(provided):
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.content_digest must be 64-hex lowercase sha256"
        )
    elif provided != expected_digest:
        errors.append(
            f"{record_prefix}{checkpoint_prefix}.content_digest must match canonical content digest binding"
        )
    return normalized


def _iter_records(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    errors: list[str],
) -> list[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return [records]
    if not isinstance(records, Iterable):
        errors.append("records must be a record object or iterable of records")
        return []
    return [record for record in records]


def _validate_checkpoint_records(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> tuple[list[str], list[dict[str, Any]], dict[str, str | None]]:
    errors: list[str] = []
    ordered_records = _iter_records(records, errors)
    if not ordered_records:
        errors.append("at least one checkpoint receipt record is required")
        return errors, [], {"wandb": None, "tinker": None}
    normalized_checkpoints: list[dict[str, Any]] = []
    kinds_seen: set[str] = set()
    seen_identifiers: set[tuple[str, int, str, str, str]] = set()
    observed_run_ids: dict[str, str | None] = {"wandb": None, "tinker": None}
    for record_index, raw_record in enumerate(ordered_records):
        record_prefix = f"records[{record_index}]"
        if not isinstance(raw_record, Mapping):
            errors.append(f"{record_prefix} must be an object")
            continue
        record = dict(raw_record)

        if "run_id" in record and _nonempty_text(record["run_id"]):
            if require_run_id is not None and str(record["run_id"]).strip() != require_run_id:
                errors.append(
                    f"{record_prefix}.run_id does not match required run_id"
                )

        record_source = _normalise_record(record_index, record)
        wandb_identity = _validate_wandb_identity(
            _first_value(record, ("wandb_run_identity", "wandb")),
            record_prefix,
            errors,
            require_run_id,
        )
        tinker_identity = _validate_tinker_identity(
            _first_value(record, ("tinker_run_identity", "tinker")),
            record_prefix,
            errors,
            require_run_id,
        )
        if (
            wandb_identity is not None
            and observed_run_ids["wandb"] is not None
            and observed_run_ids["wandb"] != wandb_identity
        ):
            errors.append(
                f"{record_prefix}.wandb_run_identity.run_id drifts from previous records"
            )
        if (
            tinker_identity is not None
            and observed_run_ids["tinker"] is not None
            and observed_run_ids["tinker"] != tinker_identity
        ):
            errors.append(
                f"{record_prefix}.tinker_run_identity.run_id drifts from previous records"
            )
        if wandb_identity is not None and observed_run_ids["wandb"] is None:
            observed_run_ids["wandb"] = wandb_identity
        if tinker_identity is not None and observed_run_ids["tinker"] is None:
            observed_run_ids["tinker"] = tinker_identity

        if (
            wandb_identity is not None
            and tinker_identity is not None
            and wandb_identity != tinker_identity
        ):
            errors.append(
                f"{record_prefix} has mismatched wandb and tinker run_id values"
            )
        if wandb_identity is None and tinker_identity is None:
            # cannot infer run identity for this record.
            pass

        checkpoints = _first_value(record, ("hf_checkpoints", "checkpoints"))
        if not isinstance(checkpoints, list) or not checkpoints:
            errors.append(f"{record_prefix}.hf_checkpoints must be a non-empty list")
            continue
        for index, checkpoint in enumerate(checkpoints):
            checkpoint_prefix = f".hf_checkpoints[{index}]"
            normalized = _validate_checkpoint(
                checkpoint,
                record_prefix,
                record_source,
                checkpoint_prefix,
                errors,
            )
            if normalized is None:
                continue
            identity = (
                normalized["kind"],
                int(normalized["step"]),
                normalized["source"],
                normalized["repo"],
                normalized["commit"],
            )
            if identity in seen_identifiers:
                errors.append(
                    f"{record_prefix}.hf_checkpoints[{index}] duplicate checkpoint identity (kind, step, source, repo, commit): {identity}"
                )
            seen_identifiers.add(identity)
            kinds_seen.add(normalized["kind"])
            normalized_checkpoints.append(normalized)

    if observed_run_ids["wandb"] != observed_run_ids["tinker"] and all(
        value is not None for value in observed_run_ids.values()
    ):
        errors.append("run-id drift between W&B and Tinker receipts")
    if kinds_seen != set(REQUIRED_CHECKPOINT_KINDS):
        missing = sorted(set(REQUIRED_CHECKPOINT_KINDS) - kinds_seen)
        errors.append(
            f"checkpoint coverage is incomplete; missing kinds: {missing}"
        )
    if require_run_id is not None:
        if observed_run_ids["wandb"] is not None and observed_run_ids["wandb"] != require_run_id:
            errors.append("records contain unexpected wandb run_id")
        if observed_run_ids["tinker"] is not None and observed_run_ids["tinker"] != require_run_id:
            errors.append("records contain unexpected tinker run_id")

    return errors, normalized_checkpoints, observed_run_ids


def validate_checkpoint_receipts(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> list[str]:
    """Validate records and return all violations."""

    errors, _, _ = _validate_checkpoint_records(records, require_run_id=require_run_id)
    return errors


def validate_checkpoint_records(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> list[str]:
    """Backward-compatible alias for checkpoint-record validation."""

    return validate_checkpoint_receipts(records, require_run_id=require_run_id)


def index_checkpoint_receipts(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> dict[str, Any]:
    """Return a deterministic checkpoint index with an immutable digest."""

    errors, checkpoints, observed = _validate_checkpoint_records(
        records, require_run_id=require_run_id
    )
    if errors:
        raise ValueError("checkpoint receipts are invalid: " + "; ".join(errors))

    ordered = sorted(
        checkpoints,
        key=lambda item: (
            str(item["source"]),
            _CHECKPOINT_KIND_ORDER[str(item["kind"])],
            int(item["step"]),
            str(item["repo"]),
            str(item["commit"]),
            str(item["visibility"]),
            str(item["url"]),
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "required_kinds": list(REQUIRED_CHECKPOINT_KINDS),
        "run_ids": {"wandb": observed["wandb"], "tinker": observed["tinker"]},
        "checkpoint_count": len(ordered),
        "kind_coverage": [
            kind
            for kind in REQUIRED_CHECKPOINT_KINDS
            if any(cp["kind"] == kind for cp in ordered)
        ],
        "checkpoints": ordered,
    }
    payload["index_digest"] = sha256_json(
        {
            "schema_version": payload["schema_version"],
            "run_ids": payload["run_ids"],
            "required_kinds": payload["required_kinds"],
            "kind_coverage": payload["kind_coverage"],
            "checkpoint_count": payload["checkpoint_count"],
            "checkpoints": payload["checkpoints"],
        }
    )
    return payload


def build_checkpoint_receipt_index(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> dict[str, Any]:
    """Compatibility alias for checkpoint index construction."""

    return index_checkpoint_receipts(records, require_run_id=require_run_id)


def is_valid_checkpoint_receipts(
    records: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, require_run_id: str | None = None
) -> bool:
    """Return True when records validate with no checkpoint defects."""

    return not validate_checkpoint_receipts(records, require_run_id=require_run_id)


__all__ = [
    "REQUIRED_CHECKPOINT_KINDS",
    "SCHEMA_VERSION",
    "build_checkpoint_receipt_index",
    "compute_checkpoint_content_digest",
    "canonical_json",
    "index_checkpoint_receipts",
    "validate_checkpoint_records",
    "is_valid_checkpoint_receipts",
    "sha256_json",
    "sha256_text",
    "validate_checkpoint_receipts",
]
