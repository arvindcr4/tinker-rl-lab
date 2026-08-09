#!/usr/bin/env python3
"""Offline result-receipt validator for exact BFCL train outcomes.

The module is intentionally fail-closed: missing evidence fields, mutable
references, synthetic/simulator markers, or non-closed scopes produce BLOCKED
status.  It emits deterministic hashes for downstream audit, never contacts
networked services, and never enables launch or paid execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pavlov-bfcl-receipt-v1"
SUITE_ID = "bfcl_train"
NATIVE_CATEGORY = "tool_use"
_NATIVE_VERDICTS = frozenset({"pass", "fail", "error"})
_ALLOWED_COST_STATUS = frozenset(
    {"authorized", "approved", "within_cap", "complete", "observed", "zero_cost"}
)
_SCOPE_KEYS = ("is_portfolio", "is_heldout")
_STAGE_REQUIREMENTS = frozenset({"initial", "periodic", "final"})
_REQUIRED_WANDB_KEYS = ("entity", "project", "group", "run_id", "run_url")
_REQUIRED_CHECKPOINT_FIELDS = ("repo_url", "revision", "url", "stage")

HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
SHA256_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
HTTPS_URL = re.compile(r"^https://[^\s]+$")
_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "pending",
    "placeholder",
    "todo",
    "tbd",
    "unset",
    "unknown",
    "to_be_pinned_before_paid_runs",
}
_BANNED_RECEIPT_MARKERS = {
    "glaiveai/glaive-function-calling-v2",
    "glaive",
    "simulated",
    "simulator",
    "tool_use_tinker.py",
    "bfclv4_tool_use.py",
    "simulatedbfclv4",
    "simulated bfclv4",
}


class BFCLReceiptBoundaryError(ValueError):
    """Raised for malformed BFCL receipt input."""


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        raise BFCLReceiptBoundaryError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BFCLReceiptBoundaryError(f"{field} must be explicit boolean false")
    return value


def _first_text(value: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return None


def _require_lower_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX40.fullmatch(text):
        raise BFCLReceiptBoundaryError(f"{field} must be immutable 40-char hex")
    return text


def _require_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX64.fullmatch(text):
        raise BFCLReceiptBoundaryError(f"{field} must be 64-char hex")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    text = text.lower()
    if not SHA256_DIGEST.fullmatch(text):
        raise BFCLReceiptBoundaryError(f"{field} must be sha256 digest")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _valid_url(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not HTTPS_URL.fullmatch(text):
        raise BFCLReceiptBoundaryError(f"{field} must be HTTPS URL")
    return text


def _contains_banned_receipt_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_RECEIPT_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _contains_banned_receipt_markers(item)
            for item in value.values()
            if item is not None
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return any(_contains_banned_receipt_markers(item) for item in value)
    return False


def _validate_scope(scope: Any) -> dict[str, bool]:
    if not isinstance(scope, Mapping):
        raise BFCLReceiptBoundaryError("scope must be an object")

    normalized: dict[str, bool] = {}
    for key in _SCOPE_KEYS:
        if key not in scope:
            raise BFCLReceiptBoundaryError(f"scope.{key} must be explicit false")
        normalized[key] = _require_bool(scope[key], f"scope.{key}")
        if normalized[key]:
            raise BFCLReceiptBoundaryError(
                f"scope.{key} must be false; this run must not claim portfolio/heldout coverage"
            )
    return normalized


def _validate_examples(records: Any) -> list[dict[str, str]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise BFCLReceiptBoundaryError("per_example must be a list of examples")

    normalized: list[tuple[str, str, str]] = []
    ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise BFCLReceiptBoundaryError(f"per_example[{index}] must be an object")
        example_id = _require_hex64(record.get("id"), f"per_example[{index}].id")
        category = _require_text(record.get("category"), f"per_example[{index}].category")
        if category != NATIVE_CATEGORY:
            raise BFCLReceiptBoundaryError(
                f"per_example[{index}].category must be {NATIVE_CATEGORY!r}"
            )

        verdict = record.get("verdict")
        if isinstance(verdict, bool):
            verdict_value = "pass" if verdict else "fail"
        elif isinstance(verdict, str):
            verdict_value = verdict.strip().lower()
            if verdict_value not in _NATIVE_VERDICTS:
                raise BFCLReceiptBoundaryError(
                    f"per_example[{index}].verdict must be pass|fail|error"
                )
        else:
            raise BFCLReceiptBoundaryError(
                f"per_example[{index}].verdict must be pass|fail|error"
            )

        normalized.append((example_id, category, verdict_value))
        ids.append(example_id)

    if not normalized:
        raise BFCLReceiptBoundaryError("per_example cannot be empty")
    if len(set(ids)) != len(ids):
        raise BFCLReceiptBoundaryError("per_example contains duplicate ids")
    if ids != sorted(ids):
        raise BFCLReceiptBoundaryError("per_example must be sorted deterministically by id")

    return [
        {"id": example_id, "category": category, "verdict": verdict_value}
        for example_id, category, verdict_value in normalized
    ]


def _validate_wandb(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise BFCLReceiptBoundaryError("wandb_run_identity must be an object")
    if _is_placeholder(raw.get("online")):
        raise BFCLReceiptBoundaryError("wandb_run_identity.online is missing")
    if raw.get("online") is not True:
        raise BFCLReceiptBoundaryError("wandb_run_identity.online must be true")

    normalized: dict[str, str] = {}
    for key in _REQUIRED_WANDB_KEYS:
        normalized[key] = _require_text(raw.get(key), f"wandb_run_identity.{key}")
    run_url = normalized["run_url"]
    if not HTTPS_URL.fullmatch(run_url):
        raise BFCLReceiptBoundaryError("wandb_run_identity.run_url must be HTTPS")
    if "wandb.ai" not in run_url:
        raise BFCLReceiptBoundaryError(
            "wandb_run_identity.run_url must be a W&B run URL"
        )
    return normalized


def _validate_tinker(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise BFCLReceiptBoundaryError("tinker_run_identity must be an object")
    run_id = _require_text(raw.get("run_id"), "tinker_run_identity.run_id")
    cost_status = _require_text(raw.get("cost_status"), "tinker_run_identity.cost_status")
    if cost_status.lower() not in _ALLOWED_COST_STATUS:
        raise BFCLReceiptBoundaryError("tinker_run_identity.cost_status is invalid")
    return {"run_id": run_id, "cost_status": cost_status.lower()}


def _validate_costs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise BFCLReceiptBoundaryError("costs must be an object")

    status = _require_text(raw.get("status"), "costs.status")
    status = status.lower()
    if status not in _ALLOWED_COST_STATUS:
        raise BFCLReceiptBoundaryError("costs.status must be authorized/observed/complete")

    total_usd = raw.get("total_usd")
    if not isinstance(total_usd, (int, float)) or isinstance(total_usd, bool):
        raise BFCLReceiptBoundaryError("costs.total_usd must be a number")
    if total_usd < 0:
        raise BFCLReceiptBoundaryError("costs.total_usd cannot be negative")

    if raw.get("paid_work") is True:
        raise BFCLReceiptBoundaryError("costs.paid_work is not allowed for offline boundary")

    return {"status": status, "total_usd": float(total_usd)}


def _validate_hf_checkpoints(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise BFCLReceiptBoundaryError("hf_checkpoints must be a list")
    if not raw:
        raise BFCLReceiptBoundaryError("hf_checkpoints cannot be empty")

    normalized: list[dict[str, Any]] = []
    seen_stage: set[str] = set()
    seen_identifiers: set[tuple[str, str]] = set()
    seen_payload: set[tuple[str, str, str]] = set()

    for index, checkpoint in enumerate(raw):
        if not isinstance(checkpoint, Mapping):
            raise BFCLReceiptBoundaryError(f"hf_checkpoints[{index}] must be an object")
        repo = _valid_url(_first_text(checkpoint, "repo_url", "repo", "repository"), f"hf_checkpoints[{index}].repo_url")
        revision = _require_lower_hex40(
            _first_text(checkpoint, "revision", "commit", "sha"),
            f"hf_checkpoints[{index}].revision",
        )
        url = _valid_url(
            _first_text(checkpoint, "url", "checkpoint_url"), f"hf_checkpoints[{index}].url"
        )
        stage = _require_text(checkpoint.get("stage"), f"hf_checkpoints[{index}].stage").lower()
        if stage not in {"initial", "periodic", "final"}:
            raise BFCLReceiptBoundaryError(
                "hf_checkpoints stage must be one of initial/periodic/final"
            )
        if checkpoint.get("safe_public_artifact") is not True:
            raise BFCLReceiptBoundaryError(
                f"hf_checkpoints[{index}].safe_public_artifact must be true"
            )
        visibility = _require_text(
            checkpoint.get("visibility"),
            f"hf_checkpoints[{index}].visibility",
        ).lower()
        if visibility not in {"public", "private"}:
            raise BFCLReceiptBoundaryError(
                f"hf_checkpoints[{index}].visibility must be public or private"
            )

        id_pair = (repo, revision)
        payload = (repo, revision, url)
        if id_pair in seen_identifiers:
            raise BFCLReceiptBoundaryError(
                f"hf_checkpoints duplicate repo/revision pair at index {index}"
            )
        if payload in seen_payload:
            raise BFCLReceiptBoundaryError(
                f"hf_checkpoints duplicate artifact payload at index {index}"
            )
        seen_identifiers.add(id_pair)
        seen_payload.add(payload)
        seen_stage.add(stage)
        normalized.append(
            {
                "repo_url": repo,
                "revision": revision,
                "url": url,
                "stage": stage,
                "safe_public_artifact": True,
                "visibility": visibility,
            }
        )

    if not _STAGE_REQUIREMENTS.issubset(seen_stage):
        missing = sorted(_STAGE_REQUIREMENTS - seen_stage)
        raise BFCLReceiptBoundaryError(
            "hf_checkpoints missing required stage(s): " + ", ".join(missing)
        )
    normalized.sort(key=lambda item: item["stage"])
    return normalized


def _validate_no_network_or_credentials(manifest: Mapping[str, Any], blockers: list[str]) -> None:
    if manifest.get("requires_network") is True:
        blockers.append("dataset download/network is disallowed for offline receipt checks")
    for field in (
        "credential_ref",
        "api_key",
        "api_token",
        "wandb_api_key",
        "hf_token",
        "oauth_token",
        "access_token",
    ):
        value = manifest.get(field)
        if not _is_placeholder(value):
            blockers.append(f"{field} is not allowed in offline boundary")


def build_receipt_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic BFCL receipt decision payload."""

    blockers: list[str] = []
    try:
        if not isinstance(manifest, Mapping):
            raise BFCLReceiptBoundaryError("manifest must be a JSON object")

        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            blockers.append(f"suite_id must be {SUITE_ID!r}, found {suite_id!r}")

        category = _require_text(manifest.get("category"), "category")
        if category != NATIVE_CATEGORY:
            blockers.append(
                f"category must be {NATIVE_CATEGORY!r}, found {category!r}"
            )

        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            raise BFCLReceiptBoundaryError("dataset must be an object")
        dataset_revision = _require_lower_hex40(
            dataset.get("revision"), "dataset.revision"
        )
        dataset_source = _require_text(dataset.get("source"), "dataset.source").lower()
        if "glaive" in dataset_source or "synthetic" in dataset_source:
            blockers.append("dataset source references synthetic/Glaive evidence")

        adapter_manifest_digest = _require_sha256(
            manifest.get("adapter_manifest_digest"), "adapter_manifest_digest"
        )

        scope = _validate_scope(manifest.get("scope"))

        per_example = _validate_examples(manifest.get("per_example"))

        wandb = _validate_wandb(
            manifest.get("wandb_run_identity")
            if isinstance(manifest.get("wandb_run_identity"), Mapping)
            else manifest.get("wandb")
        )
        tinker = _validate_tinker(
            manifest.get("tinker_run_identity")
            if isinstance(manifest.get("tinker_run_identity"), Mapping)
            else manifest.get("tinker")
        )
        hf_checkpoints = _validate_hf_checkpoints(manifest.get("hf_checkpoints"))
        costs = _validate_costs(manifest.get("costs"))
        evidence = {
            "dataset": dataset,
            "wandb": manifest.get("wandb_run_identity")
            if isinstance(manifest.get("wandb_run_identity"), Mapping)
            else manifest.get("wandb"),
            "tinker": manifest.get("tinker_run_identity")
            if isinstance(manifest.get("tinker_run_identity"), Mapping)
            else manifest.get("tinker"),
            "hf_checkpoints": manifest.get("hf_checkpoints"),
            "costs": manifest.get("costs"),
            "per_example": per_example,
        }
        if any(_contains_banned_receipt_markers(value) for value in evidence.values()):
            blockers.append("receipt contains synthetic or Glaive markers")

        _validate_no_network_or_credentials(manifest, blockers)

        blockers = sorted(set(blockers))
        per_example_count = len(per_example)
        per_example_verdicts = sorted(
            {item["verdict"] for item in per_example}
        )

        per_example_payload = {
            "suite_id": suite_id,
            "examples": per_example,
        }
        receipt_payload = {
            "dataset_revision": dataset_revision,
            "adapter_manifest_digest": adapter_manifest_digest,
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": hf_checkpoints,
            "costs": costs,
            "scope": scope,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "exact_suite": suite_id == SUITE_ID,
            "receipt_ready": not blockers,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": category,
            "dataset_revision": dataset_revision,
            "dataset_source": dataset_source,
            "adapter_manifest_digest": adapter_manifest_digest,
            "per_example_count": per_example_count,
            "per_example_verdicts": per_example_verdicts,
            "per_example_category": NATIVE_CATEGORY,
            "per_example_digest": _sha256(canonical_json(per_example_payload)),
            "receipt_identity_digest": _sha256(canonical_json(receipt_payload)),
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": hf_checkpoints,
            "costs": costs,
            "scope": scope,
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }

    except BFCLReceiptBoundaryError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.get("suite_id", "<missing>") if isinstance(manifest, Mapping) else "<invalid>",
            "exact_suite": isinstance(manifest, Mapping) and manifest.get("suite_id") == SUITE_ID,
            "receipt_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": manifest.get("category", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "dataset_revision": None,
            "dataset_source": None,
            "adapter_manifest_digest": None,
            "per_example_count": 0,
            "per_example_verdicts": [],
            "per_example_category": None,
            "per_example_digest": None,
            "receipt_identity_digest": None,
            "wandb": None,
            "tinker": None,
            "hf_checkpoints": [],
            "costs": None,
            "scope": {},
            "blockers": [str(exc)],
            "status": "BLOCKED",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("manifest must be a JSON object")
    report = build_receipt_record(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
