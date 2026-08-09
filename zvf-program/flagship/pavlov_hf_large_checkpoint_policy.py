#!/usr/bin/env python3
"""Validate and index large-checkpoint Hugging Face export policy records offline.

The validator is strict and fail-closed. It performs only deterministic local
checks and never makes network calls, uses credentials, uploads artifacts, or
starts jobs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = "pavlov-hf-large-checkpoint-policy-v1"
REQUIRED_CHECKPOINT_KINDS = ("initial", "periodic", "final")

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

_IMMUTABLE_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_REPO_PATH_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_REPO_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)
_HF_COMMIT_URL_RE = re.compile(
    r"^https://huggingface\.co/(?P<owner>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)/commit/(?P<revision>[0-9a-fA-F]{40})$"
)
_WANDB_RUN_URL_RE = re.compile(
    r"^https://wandb\.ai/(?P<entity>[A-Za-z0-9][A-Za-z0-9._-]*)/(?P<project>[A-Za-z0-9][A-Za-z0-9._-]*)/runs/(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]*)$"
)

WANDB_TERMINAL_STATES = frozenset({"finished", "completed"})
TINKER_TERMINAL_STATES = frozenset({"finished", "completed"})
REJECTED_STATUSES = frozenset(
    {
        "rejected",
        "failed",
        "aborted",
        "partial",
        "incomplete",
        "errored",
        "disallowed",
        "timeout",
        "cancelled",
    }
)
COMPLETED_STATUSES = frozenset({"completed", "finished"})
_ALL_RECORD_STATUSES = COMPLETED_STATUSES | REJECTED_STATUSES
VALID_VISIBILITIES = frozenset({"public", "private"})


def canonical_json(value: Any) -> str:
    """Return canonical JSON for digesting values."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return SHA-256 for a plain text value."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return SHA-256 for canonical JSON."""

    return sha256_text(canonical_json(value))


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _PLACEHOLDER_VALUES


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and not _is_placeholder(value) and bool(value.strip())


def _sha40(value: Any) -> bool:
    return isinstance(value, str) and bool(_IMMUTABLE_SHA40_RE.fullmatch(value.strip()))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip().lower()))


def _first_value(record: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _normalise_repo(value: Any) -> str | None:
    if not _nonempty_text(value):
        return None
    text = str(value).strip()
    if _HF_REPO_URL_RE.fullmatch(text):
        return text
    if _HF_REPO_PATH_RE.fullmatch(text):
        return f"https://huggingface.co/{text}"
    return None


def _normalise_checkpoint_kind(value: Any) -> str | None:
    if not _nonempty_text(value):
        return None
    kind = str(value).strip().lower()
    if kind in {"initial", "initial_step_0_sampler", "initial_step_0", "initial_step"}:
        return "initial"
    if kind in {"periodic", "periodic_checkpoint", "periodic_export"}:
        return "periodic"
    if kind.startswith("periodic"):
        return "periodic"
    if kind in {"final", "final_sampler", "final_step"}:
        return "final"
    if kind.startswith("final"):
        return "final"
    return None


def compute_large_checkpoint_content_digest(checkpoint: Mapping[str, Any]) -> str:
    """Return the canonical content digest bound into checkpoint records."""

    payload = _canonical_checkpoint_payload(checkpoint)
    return sha256_json(payload)


def _canonical_checkpoint_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    revision = _first_value(checkpoint, ("revision", "commit", "sha", "hf_commit", "hash"))
    repo = _first_value(
        checkpoint,
        ("repo", "repo_url", "repository", "huggingface_repo", "hf_repo"),
    )
    return {
        "repo": _normalise_repo(repo),
        "precreated_revision": _first_value(
            checkpoint,
            (
                "precreated_revision",
                "precreated_sha",
                "base_revision",
                "resume_from_revision",
            ),
        ),
        "revision": str(revision).strip().lower() if isinstance(revision, str) else revision,
        "kind": _normalise_checkpoint_kind(_first_value(checkpoint, ("kind", "stage", "phase"))),
        "step": checkpoint.get("step"),
        "visibility": _first_value(checkpoint, ("visibility", "scope", "license_scope")),
        "safe_public_artifact": checkpoint.get("safe_public_artifact"),
        "url": _first_value(
            checkpoint,
            ("url", "checkpoint_url", "repo_revision_url", "commit_url"),
        ),
    }


def _extract_run_id_fields(record: Mapping[str, Any], errors: list[str], prefix: str) -> tuple[str | None, str | None]:
    direct = _first_value(
        record,
        (
            "run_id",
            "id",
            "tinker_run_id",
            "wandb_run_id",
        ),
    )
    if _nonempty_text(direct):
        direct = str(direct).strip()
    else:
        direct = None

    wandb = _validate_wandb_identity(_first_value(record, ("wandb_run_identity", "wandb", "wandb_receipt")),
                                    errors, f"{prefix}.wandb_run_identity", direct)
    tinker = _validate_tinker_identity(
        _first_value(record, ("tinker_run_identity", "tinker", "tinker_receipt")),
        errors,
        f"{prefix}.tinker_run_identity",
        direct,
    )

    candidates = [value for value in (direct, wandb, tinker) if _nonempty_text(value)]
    if not candidates:
        errors.append(f"{prefix}.run_id is required and cannot be placeholder-like")
        return None, None

    if len(set(candidates)) != 1:
        errors.append(f"{prefix}.run_id drift between records/identities")
        if direct is not None:
            return direct, None
        return candidates[0], None

    return candidates[0], candidates[0]


def _validate_wandb_identity(
    receipt: Any, errors: list[str], prefix: str, expected_run_id: str | None
) -> str | None:
    if receipt is None:
        errors.append(f"{prefix} is required")
        return None
    if not isinstance(receipt, Mapping):
        errors.append(f"{prefix} must be an object")
        return None

    if receipt.get("online") is not True:
        errors.append(f"{prefix}.online must be the boolean true")
    if receipt.get("acknowledged") is not True:
        errors.append(f"{prefix}.acknowledged must be the boolean true")

    run_id = _first_value(receipt, ("run_id", "id"))
    if not _nonempty_text(run_id):
        errors.append(f"{prefix}.run_id must be a non-placeholder string")
        run_id = None
    else:
        run_id = str(run_id).strip()
        if " " in run_id:
            errors.append(f"{prefix}.run_id cannot contain whitespace")

    run_url = _first_value(receipt, ("run_url", "url", "link"))
    if not isinstance(run_url, str):
        errors.append(f"{prefix}.run_url must be HTTPS")
    else:
        match = _WANDB_RUN_URL_RE.fullmatch(run_url.strip())
        if not match:
            errors.append(f"{prefix}.run_url must have exact wandb host/path shape")
        else:
            if run_id is not None and run_id != match.group("run_id"):
                errors.append(
                    f"{prefix}.run_id must match wandb run_url path segment"
                )
            entity = _first_value(receipt, ("entity",))
            project = _first_value(receipt, ("project",))
            if _nonempty_text(entity) and str(entity).strip() != match.group("entity"):
                errors.append(f"{prefix}.entity must match wandb run_url segment")
            if _nonempty_text(project) and str(project).strip() != match.group("project"):
                errors.append(f"{prefix}.project must match wandb run_url segment")

    state = _first_value(receipt, ("state", "status"))
    if not _nonempty_text(state) or str(state).strip().lower() not in WANDB_TERMINAL_STATES:
        errors.append(
            f"{prefix}.state must be terminal: {sorted(WANDB_TERMINAL_STATES)}"
        )

    return run_id


def _validate_tinker_identity(
    receipt: Any, errors: list[str], prefix: str, expected_run_id: str | None
) -> str | None:
    if receipt is None:
        errors.append(f"{prefix} is required")
        return None
    if not isinstance(receipt, Mapping):
        errors.append(f"{prefix} must be an object")
        return None

    run_id = _first_value(receipt, ("run_id", "id"))
    if not _nonempty_text(run_id):
        errors.append(f"{prefix}.run_id must be a non-placeholder string")
        run_id = None
    else:
        run_id = str(run_id).strip()
        if " " in run_id:
            errors.append(f"{prefix}.run_id cannot contain whitespace")

    if expected_run_id is not None and run_id is not None and run_id != expected_run_id:
        errors.append(f"{prefix}.run_id does not match record run_id")

    state = _first_value(receipt, ("state", "status"))
    if state is not None:
        if not _nonempty_text(state) or str(state).strip().lower() not in TINKER_TERMINAL_STATES:
            errors.append(
                f"{prefix}.state/status must be terminal: {sorted(TINKER_TERMINAL_STATES)}"
            )

    return run_id


def _extract_parent_run_id(record: Mapping[str, Any], errors: list[str], prefix: str) -> str | None:
    candidate = _first_value(
        record,
        (
            "retry_parent_run_id",
            "parent_run_id",
            "resume_parent_run_id",
            "retry_of",
        ),
    )
    if _nonempty_text(candidate):
        return str(candidate).strip()

    retry = _first_value(record, ("retry", "resume"))
    if isinstance(retry, Mapping):
        nested = _first_value(
            retry,
            ("parent_run_id", "run_id", "retry_parent_run_id", "resume_parent_run_id"),
        )
        if _nonempty_text(nested):
            return str(nested).strip()
    return None


def _extract_precreated_revision(record: Mapping[str, Any]) -> str | None:
    value = _first_value(
        record,
        (
            "precreated_revision",
            "precreated_sha",
            "base_revision",
            "resume_from_revision",
            "resume_revision",
        ),
    )
    if _nonempty_text(value):
        return str(value).strip()
    return None


def _normalize_checkpoint(checkpoint: Any, prefix: str, record_precreated: str | None, run_id: str,
                          errors: list[str], pair_seen: set[tuple[str, int]]) -> dict[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{prefix} must be a JSON object")
        return None

    repo = _normalise_repo(
        _first_value(
            checkpoint,
            (
                "repo",
                "repo_url",
                "repository",
                "huggingface_repo",
                "hf_repo",
            ),
        )
    )
    if repo is None:
        errors.append(f"{prefix}.repo must be an https Hugging Face repository URL")
        return None

    revision = _first_value(checkpoint, ("revision", "commit", "sha", "hf_commit", "hash"))
    if not _sha40(revision):
        errors.append(f"{prefix}.revision must be immutable 40-hex")
        return None
    revision = str(revision).strip()

    checkpoint_url = _first_value(
        checkpoint,
        ("url", "checkpoint_url", "repo_revision_url", "commit_url", "hf_url"),
    )
    if not _nonempty_text(checkpoint_url):
        errors.append(f"{prefix}.url/checkpoint_url must be a non-empty URL")
        return None
    if not isinstance(checkpoint_url, str):
        errors.append(f"{prefix}.url/checkpoint_url must be HTTPS text")
        return None
    checkpoint_url = checkpoint_url.strip()

    commit_match = _HF_COMMIT_URL_RE.fullmatch(checkpoint_url)
    if not commit_match:
        if "/tree/" in checkpoint_url or "/blob/" in checkpoint_url:
            errors.append(f"{prefix}.url is branch-only evidence")
        else:
            errors.append(f"{prefix}.url must be an immutable commit URL")
        return None
    if f"https://huggingface.co/{commit_match.group('owner')}/{commit_match.group('repo')}" != repo:
        errors.append(f"{prefix}.url repository must match checkpoint repo")
        return None
    if commit_match.group("revision") != revision:
        errors.append(f"{prefix}.url commit must match revision")
        return None

    kind = _normalise_checkpoint_kind(_first_value(checkpoint, ("kind", "stage", "phase")))
    if kind is None:
        errors.append(
            f"{prefix}.kind must be one of: {list(REQUIRED_CHECKPOINT_KINDS)}"
        )
        return None

    step = checkpoint.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        errors.append(f"{prefix}.step must be a non-negative integer")
        return None

    pair = (kind, int(step))
    if pair in pair_seen:
        errors.append(f"{prefix}.duplicate checkpoint kind/step pair: {pair}")
    pair_seen.add(pair)

    visibility = _first_value(checkpoint, ("visibility", "scope", "license_scope"))
    if visibility not in VALID_VISIBILITIES:
        errors.append(f"{prefix}.visibility must be public or private")
        return None

    safe_public_artifact = checkpoint.get("safe_public_artifact")
    if not isinstance(safe_public_artifact, bool):
        errors.append(f"{prefix}.safe_public_artifact must be a boolean")
        return None
    if visibility == "public" and safe_public_artifact is not True:
        errors.append(f"{prefix}.safe_public_artifact must be true for public visibility")
    if visibility == "private" and safe_public_artifact is not False:
        errors.append(f"{prefix}.safe_public_artifact must be false for private visibility")

    checkpoint_precreated = _first_value(
        checkpoint,
        (
            "precreated_revision",
            "precreated_sha",
            "base_revision",
            "resume_from_revision",
        ),
    )
    if _nonempty_text(checkpoint_precreated) and _nonempty_text(record_precreated):
        if str(checkpoint_precreated).strip() != str(record_precreated).strip():
            errors.append(f"{prefix}.precreated_revision must match record precreated_revision")
    elif _nonempty_text(record_precreated) and not _nonempty_text(checkpoint_precreated):
        errors.append(f"{prefix}.precreated_revision is required")

    content_digest = _first_value(checkpoint, ("content_digest", "checkpoint_digest"))
    if not _sha256(content_digest):
        errors.append(
            f"{prefix}.content_digest must be a lowercase 64-char SHA-256 hex string"
        )
    else:
        expected_digest = compute_large_checkpoint_content_digest(checkpoint)
        if str(content_digest).strip().lower() != expected_digest:
            errors.append(f"{prefix}.content_digest must match canonical content digest binding")

    return {
        "run_id": run_id,
        "repo": repo,
        "revision": revision,
        "kind": kind,
        "step": int(step),
        "url": checkpoint_url,
        "visibility": visibility,
        "safe_public_artifact": safe_public_artifact,
        "precreated_revision": str(record_precreated).strip() if _nonempty_text(record_precreated) else None,
    }


def _normalise_records(records: Any) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []

    if isinstance(records, Mapping):
        records_list: list[Any] = [records]
    elif isinstance(records, (str, bytes)) or not isinstance(records, Iterable):
        return ["records must be a mapping or list of mappings"], []
    else:
        try:
            records_list = list(records)
        except TypeError:
            return ["records must be iterable"], []

    if not records_list:
        return ["at least one policy record is required"], []

    repo_to_run: dict[str, str] = {}
    repo_to_kinds: dict[str, set[str]] = {}
    run_by_id: dict[str, dict[str, Any]] = {}
    normalised_records: list[dict[str, Any]] = []

    for index, record in enumerate(records_list):
        record_prefix = f"records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{record_prefix} must be a JSON object")
            continue

        status = _first_value(record, ("status", "state"))
        if not _nonempty_text(status):
            status = ""
        else:
            status = str(status).strip().lower()
        if status not in _ALL_RECORD_STATUSES:
            errors.append(
                f"{record_prefix}.status must be one of: {sorted(_ALL_RECORD_STATUSES)}"
            )

        run_id, canonical_run_id = _extract_run_id_fields(record, errors, record_prefix)
        if not _nonempty_text(run_id):
            run_id = f"{record_prefix}"
        else:
            run_id = str(run_id).strip()

        precreated_revision = _extract_precreated_revision(record)
        if not _sha40(precreated_revision):
            errors.append(f"{record_prefix}.precreated_revision must be immutable 40-hex")

        parent_run_id = _extract_parent_run_id(record, errors, record_prefix)
        if _nonempty_text(parent_run_id) and parent_run_id == run_id:
            errors.append(f"{record_prefix}.retry_parent_run_id cannot reference itself")

        checkpoints_field = _first_value(
            record,
            (
                "hf_large_checkpoints",
                "hf_checkpoints",
                "checkpoints",
                "model_checkpoints",
            ),
        )
        if not isinstance(checkpoints_field, list) or not checkpoints_field:
            errors.append(f"{record_prefix}.checkpoints must be a non-empty list")
            checkpoints_field = []

        pair_seen: set[tuple[str, int]] = set()
        kinds_for_record: set[str] = set()
        repos_for_record: set[str] = set()
        checkpoint_run_entries: list[dict[str, Any]] = []
        for checkpoint_index, checkpoint in enumerate(checkpoints_field):
            checkpoint_prefix = f"{record_prefix}.checkpoints[{checkpoint_index}]"
            normalized = _normalize_checkpoint(
                checkpoint,
                checkpoint_prefix,
                precreated_revision,
                run_id,
                errors,
                pair_seen,
            )
            if normalized is None:
                continue

            repo = normalized["repo"]
            repos_for_record.add(repo)
            kinds_for_record.add(normalized["kind"])
            repo_to_run.setdefault(repo, run_id)
            if repo in repo_to_run and repo_to_run[repo] != run_id:
                errors.append(
                    f"{checkpoint_prefix} reuses repo {repo} from another run"
                )
            existing_run = repo_to_kinds.setdefault(repo, set())
            existing_run.update(kinds_for_record)
            checkpoint_run_entries.append(normalized)

        if kinds_for_record and kinds_for_record != set(REQUIRED_CHECKPOINT_KINDS):
            missing = sorted(set(REQUIRED_CHECKPOINT_KINDS) - kinds_for_record)
            if missing:
                errors.append(
                    f"{record_prefix}.checkpoint lifecycle missing kinds: {missing}"
                )

        if status in COMPLETED_STATUSES and not kinds_for_record:
            errors.append(f"{record_prefix}.complete runs require checkpoint coverage")

        if len(repos_for_record) > 1:
            errors.append(f"{record_prefix} must use a single Hugging Face repository")

        final_revision = None
        for item in checkpoint_run_entries:
            if item["kind"] == "final":
                if final_revision is None or item["step"] >= 0:
                    final_revision = item["revision"]

        if run_id in run_by_id:
            errors.append(f"duplicate run_id {run_id!r} in record set")

        run_by_id[run_id] = {
            "record_prefix": record_prefix,
            "status": status,
            "precreated_revision": str(precreated_revision).strip() if _nonempty_text(precreated_revision) else None,
            "parent_run_id": str(parent_run_id).strip() if _nonempty_text(parent_run_id) else None,
            "checkpoints": checkpoint_run_entries,
            "final_revision": final_revision,
        }
        normalised_records.append(
            {
                "run_id": run_id,
                "status": status,
                "precreated_revision": str(precreated_revision).strip() if _nonempty_text(precreated_revision) else None,
                "parent_run_id": str(parent_run_id).strip() if _nonempty_text(parent_run_id) else None,
                "checkpoints": checkpoint_run_entries,
            }
        )

    parent_map: dict[str, str] = {}
    for run_id, record in run_by_id.items():
        parent_run_id = record["parent_run_id"]
        if not parent_run_id:
            continue
        parent_map[run_id] = parent_run_id
        parent = run_by_id.get(parent_run_id)
        if parent is None:
            errors.append(f"{record['record_prefix']}.retry_parent_run_id does not reference an existing run")
            continue
        if parent["status"] not in COMPLETED_STATUSES:
            errors.append(f"{record['record_prefix']}.retry parent run must be completed")
        if parent.get("final_revision") is None:
            errors.append(f"{record['record_prefix']}.retry parent has no final checkpoint")
        if _nonempty_text(record["precreated_revision"]) and _nonempty_text(parent.get("final_revision")):
            if str(record["precreated_revision"]) != str(parent.get("final_revision")):
                errors.append(
                    f"{record['record_prefix']}.precreated_revision must match parent final revision"
                )

    def _walk(run_id: str, visited: set[str]) -> None:
        if run_id in visited:
            errors.append("retry lineage contains a cycle")
            return
        parent = parent_map.get(run_id)
        if parent is None:
            return
        visited.add(run_id)
        _walk(parent, visited)
        visited.remove(run_id)

    for run_id in list(parent_map):
        _walk(run_id, set())

    for repo, kinds in repo_to_kinds.items():
        if kinds != set(REQUIRED_CHECKPOINT_KINDS):
            missing = sorted(set(REQUIRED_CHECKPOINT_KINDS) - kinds)
            errors.append(f"repository {repo} has incomplete checkpoint lifecycle; missing kinds: {missing}")

    return errors, normalised_records


def validate_large_checkpoint_policy(records: Any) -> list[str]:
    """Validate large-checkpoint policy records and return all fail-closed errors."""

    errors, _ = _normalise_records(records)
    return errors


def validate_large_checkpoint_policy_records(records: Any) -> list[str]:
    """Alias for policy validation with an explicit name."""

    return validate_large_checkpoint_policy(records)


def is_valid_large_checkpoint_policy(records: Any) -> bool:
    """Return True iff the provided records are valid."""

    return not validate_large_checkpoint_policy(records)


def build_large_checkpoint_policy_index(records: Any) -> dict[str, Any]:
    """Build a deterministic index over valid records with a canonical digest."""

    errors, normalised_records = _normalise_records(records)
    if errors:
        raise ValueError("large checkpoint policy validation failed: " + "; ".join(errors))

    sorted_runs = sorted(normalised_records, key=lambda item: item["run_id"])
    runs = []
    for item in sorted_runs:
        checkpoints = sorted(
            item["checkpoints"],
            key=lambda checkpoint: (
                checkpoint["repo"],
                checkpoint["kind"],
                checkpoint["step"],
                checkpoint["revision"],
            ),
        )
        runs.append(
            {
                "run_id": item["run_id"],
                "status": item["status"],
                "precreated_revision": item["precreated_revision"],
                "parent_run_id": item["parent_run_id"],
                "checkpoints": checkpoints,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_count": len(runs),
        "checkpoint_count": sum(len(item["checkpoints"]) for item in runs),
        "required_kinds": list(REQUIRED_CHECKPOINT_KINDS),
        "runs": runs,
    }
    digest = sha256_json(payload)
    return {
        **payload,
        "policy_digest": digest,
        "index_digest": digest,
    }


def index_large_checkpoint_policy(records: Any) -> dict[str, Any]:
    """Alias for building the policy index."""

    return build_large_checkpoint_policy_index(records)


def compute_large_checkpoint_policy_digest(records: Any) -> str:
    """Return deterministic digest for an entire policy record set."""

    return build_large_checkpoint_policy_index(records)["policy_digest"]


__all__ = [
    "REQUIRED_CHECKPOINT_KINDS",
    "SCHEMA_VERSION",
    "build_large_checkpoint_policy_index",
    "compute_large_checkpoint_content_digest",
    "compute_large_checkpoint_policy_digest",
    "index_large_checkpoint_policy",
    "is_valid_large_checkpoint_policy",
    "canonical_json",
    "sha256_json",
    "sha256_text",
    "validate_large_checkpoint_policy",
    "validate_large_checkpoint_policy_records",
]
