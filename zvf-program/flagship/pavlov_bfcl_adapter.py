#!/usr/bin/env python3
"""Offline metadata gate for the exact BFCL train suite ``bfcl_train``.

This module performs a metadata-first boundary check and never downloads
datasets, calls W&B, opens network connections, reads credentials, or launches
paid work.  The result expresses whether adapter metadata is stable enough to
enter the BFCL train boundary and separates that readiness from paid launch
approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "pavlov-bfcl-train-boundary-v1"
SUITE_ID = "bfcl_train"
NATIVE_CATEGORY = "tool_use"
NATIVE_VERIFIER_ID = "platform_tinker.tinkerrl.grpo.StrictToolCallReward"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^(?:[0-9a-f]{64}|sha256:[0-9a-f]{64}|none|null)$")
_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "todo",
    "tbd",
    "pending",
    "placeholder",
    "unset",
    "unknown",
    "to_be_pinned_before_paid_runs",
    "not_provided",
    "latest",
    "main",
    "head",
    "master",
    "tip",
}
_BANNED_EVIDENCE_MARKERS = {
    "tool_use_tinker.py",
    "simulatedbfclv4",
    "simulated bfclv4",
    "bfclv4_tool_use.py",
    "synthetic",
    "simulator",
}


class BFCLAdapterBoundaryError(ValueError):
    """Raised for malformed BFCL adapter boundary metadata."""


@dataclass(frozen=True)
class FunctionSchema:
    raw: Mapping[str, Any]

    @property
    def canonical(self) -> str:
        return canonical_json(self.raw)


def canonical_json(value: Any) -> str:
    """JSON encoding used for deterministic hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if normalized in _PLACEHOLDERS:
        return True
    return normalized.startswith(("pending:", "todo:", "to_be_pinned:", "to be pinned"))


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        raise BFCLAdapterBoundaryError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_lower_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field)
    normalized = text.lower()
    if not _REVISION_RE.fullmatch(normalized):
        raise BFCLAdapterBoundaryError(
            f"{field} must be a 40-character immutable revision hash"
        )
    return normalized


def _require_lower_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field)
    normalized = text.lower()
    if not _HASH_RE.fullmatch(normalized):
        raise BFCLAdapterBoundaryError(f"{field} must be a 64-character hash")
    return normalized


def _require_receipt_identity(value: Any, field: str) -> str:
    text = _require_text(value, field)
    normalized = text.lower()
    if normalized in {"placeholder", "todo", "none"}:
        raise BFCLAdapterBoundaryError(f"{field} is a placeholder")
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if not _HASH_RE.fullmatch(normalized):
        raise BFCLAdapterBoundaryError(f"{field} must be a sha256 digest or hex hash")
    return f"sha256:{normalized}" if isinstance(text, str) else normalized


def _contains_banned_evidence(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_EVIDENCE_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _contains_banned_evidence(item) for item in value.values() if item is not None
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return any(_contains_banned_evidence(item) for item in value)
    return False


def _validate_function_schema(schema: Mapping[str, Any]) -> FunctionSchema:
    raw = dict(schema)
    if raw.get("type") != "object":
        raise BFCLAdapterBoundaryError("function-call schema must declare type 'object'")

    properties = raw.get("properties")
    if not isinstance(properties, Mapping):
        raise BFCLAdapterBoundaryError("function-call schema must include properties")
    for key in ("tool", "arguments"):
        if key not in properties:
            raise BFCLAdapterBoundaryError(f"function-call schema missing property {key!r}")
        if not isinstance(properties[key], Mapping):
            raise BFCLAdapterBoundaryError(f"function-call schema property {key!r} must be an object")

    tool_type = properties["tool"].get("type")
    args_type = properties["arguments"].get("type")
    if tool_type != "string" or args_type != "object":
        raise BFCLAdapterBoundaryError(
            "function-call schema must define tool:string and arguments:object"
        )

    required = raw.get("required")
    if not isinstance(required, Sequence) or {"tool", "arguments"} - set(required):
        raise BFCLAdapterBoundaryError(
            "function-call schema required fields must include tool and arguments"
        )
    if raw.get("additionalProperties") is not False:
        raise BFCLAdapterBoundaryError("function-call schema must set additionalProperties false")

    return FunctionSchema(raw=raw)


def _validate_train_records(records: Any) -> list[dict[str, str]]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise BFCLAdapterBoundaryError("train_records must be a list of id/hash pairs")

    parsed: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise BFCLAdapterBoundaryError(f"train_records[{index}] must be an object")
        train_id = _require_lower_hex64(record.get("id"), f"train_records[{index}].id")
        train_hash = _require_lower_hex64(record.get("hash"), f"train_records[{index}].hash")
        parsed.append((train_id, train_hash))

    if not parsed:
        raise BFCLAdapterBoundaryError("train_records cannot be empty")

    ids = [item[0] for item in parsed]
    hashes = [item[1] for item in parsed]
    if len(set(ids)) != len(ids):
        raise BFCLAdapterBoundaryError("train_records contains duplicate ids")
    if len(set(hashes)) != len(hashes):
        raise BFCLAdapterBoundaryError("train_records contains duplicate hashes")

    # Deterministic records are canonicalized by strict ordering.
    ordered = sorted(parsed)
    if list(parsed) != ordered:
        raise BFCLAdapterBoundaryError("train_records must be sorted deterministically by id/hash")

    return [{"id": train_id, "hash": train_hash} for train_id, train_hash in ordered]


def _validate_artifact_receipts(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise BFCLAdapterBoundaryError("artifact_receipts must be a non-empty list")

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise BFCLAdapterBoundaryError(
                f"artifact_receipts[{index}] must include kind and identity"
            )
        kind = _require_text(item.get("kind"), f"artifact_receipts[{index}].kind").lower()
        identity = _require_receipt_identity(
            item.get("identity"),
            f"artifact_receipts[{index}].identity",
        )
        normalized_item = {"kind": kind, "identity": identity}
        key = (kind, identity)
        if key in seen:
            raise BFCLAdapterBoundaryError(
                f"artifact_receipts contains duplicate {kind!r}/{identity!r}"
            )
        seen.add(key)
        normalized.append(normalized_item)

    if not normalized:
        raise BFCLAdapterBoundaryError("artifact_receipts must be non-empty")
    return normalized


def build_boundary_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic BFCL boundary decision object.

    The boundary may be blocked without raising; callers can inspect the
    ``blockers`` list to understand why ``adapter_ready`` is false.
    """

    blockers: list[str] = []
    try:
        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            blockers.append(f"suite_id must be {SUITE_ID!r}, found {suite_id!r}")

        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            raise BFCLAdapterBoundaryError("dataset metadata is required")
        dataset_revision = _require_lower_hex40(dataset.get("revision"), "dataset.revision")
        dataset_license = _require_text(dataset.get("license"), "dataset.license")
        dataset_source = _require_text(dataset.get("source"), "dataset.source")
        if "glaive" in dataset_source.lower():
            blockers.append("dataset source references glaive evidence")
        if "glaiveai/glaive-function-calling-v2" in dataset_source.lower():
            blockers.append("dataset source references Glaive benchmark artifact")

        if manifest.get("requires_network") is True:
            blockers.append("dataset download/network is disallowed for offline boundary")
        if manifest.get("credential_ref") is not None:
            credential = _require_text(manifest.get("credential_ref"), "credential_ref")
            if credential:
                blockers.append("credential reference is not allowed in offline boundary")

        category = _require_text(manifest.get("category"), "category")
        if category != NATIVE_CATEGORY:
            blockers.append(f"category must be {NATIVE_CATEGORY!r}, found {category!r}")

        verifier = manifest.get("verifier")
        if not isinstance(verifier, Mapping):
            raise BFCLAdapterBoundaryError("verifier metadata is required")
        verifier_identity = _require_text(
            verifier.get("identity"),
            "verifier.identity",
        )
        if verifier_identity != NATIVE_VERIFIER_ID:
            blockers.append(
                f"verifier identity must be {NATIVE_VERIFIER_ID!r}, found {verifier_identity!r}"
            )
        verifier_category = _require_text(
            verifier.get("category"),
            "verifier.category",
        )
        if verifier_category != NATIVE_CATEGORY:
            blockers.append(
                f"verifier category must be {NATIVE_CATEGORY!r}, "
                f"found {verifier_category!r}"
            )

        schema = _validate_function_schema(
            verifier.get("function_call_schema")
            if isinstance(verifier.get("function_call_schema"), Mapping)
            else manifest.get("function_call_schema", {})
        )

        evidence_manifest = dict(manifest)
        if isinstance(evidence_manifest.get("dataset"), Mapping):
            evidence_manifest["dataset"] = dict(evidence_manifest["dataset"])
            evidence_manifest["dataset"].pop("source", None)
        if _contains_banned_evidence(evidence_manifest):
            blockers.append("evidence source references synthetic simulator or Glaive artifacts")

        train_records = _validate_train_records(manifest.get("train_records"))
        artifact_receipts = _validate_artifact_receipts(manifest.get("artifact_receipts"))

        train_records_payload = {
            "suite_id": SUITE_ID,
            "records": train_records,
        }
        function_schema_payload = schema.raw

        blockers = sorted(set(blockers))
        train_record_ids = [item["id"] for item in train_records]
        train_record_hashes = [item["hash"] for item in train_records]
        train_digest = _sha256(canonical_json([train_records_payload, function_schema_payload]))
        function_schema_digest = _sha256(schema.canonical)

        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "exact_suite": suite_id == SUITE_ID,
            "adapter_ready": not blockers,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": category,
            "verifier_identity": verifier_identity,
            "verifier_category": verifier_category,
            "dataset_revision": dataset_revision,
            "dataset_license": dataset_license,
            "dataset_source": dataset_source,
            "dataset_source_is_disallowed": "glaive" in dataset_source.lower(),
            "train_count": len(train_records),
            "train_record_ids": train_record_ids,
            "train_record_hashes": train_record_hashes,
            "train_records_digest": train_digest,
            "function_call_schema": function_schema_payload,
            "function_call_schema_sha256": function_schema_digest,
            "artifact_receipts": artifact_receipts,
            "artifact_receipt_count": len(artifact_receipts),
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except BFCLAdapterBoundaryError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.get("suite_id", "<missing>"),
            "exact_suite": manifest.get("suite_id") == SUITE_ID,
            "adapter_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": manifest.get("category", "<missing>"),
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "dataset_source_is_disallowed": False,
            "train_count": 0,
            "train_record_ids": [],
            "train_record_hashes": [],
            "train_records_digest": None,
            "function_call_schema": None,
            "function_call_schema_sha256": None,
            "artifact_receipts": [],
            "artifact_receipt_count": 0,
            "verifier_identity": None,
            "verifier_category": None,
            "status": "BLOCKED",
            "blockers": [str(exc)],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("manifest must be a JSON object")

    report = build_boundary_record(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
