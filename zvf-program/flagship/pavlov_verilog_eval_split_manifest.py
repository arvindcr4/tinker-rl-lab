#!/usr/bin/env python3
"""Offline exact T4 split/decontamination boundary for ``verilog_eval``.

This validator is intentionally metadata-first and fail-closed:

* no synthetic/Glaive/xLAM substitution
* no network, credentials, or paid work
* pinned immutable dataset/split artifacts
* deterministic task hash ordering and exact split manifest hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pavlov-verilog-eval-split-manifest-v1"
SUITE_ID = "verilog_eval"
SUITE_ROLE = "primary_eval"
_EXPECTED_CATEGORY = "code"
EXPECTED_DATASET_SOURCE = "https://github.com/NVlabs/verilog-eval"
_EXPECTED_STATEFUL = False
EXPECTED_ARTIFACT_OR_SIDE_EFFECT = True

HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
URL_RE = re.compile(r"^https://[^\s]+$")

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
    "to_be_pinned",
    "latest",
    "main",
    "head",
    "tip",
}

_BANNED_MARKERS = {
    "glaive",
    "glaiveai/glaive-function-calling-v2",
    "simulated",
    "simulator",
    "xlam",
    "salesforce/xlam-function-calling-60k",
    "tool_use_tinker.py",
    "bfclv4_tool_use.py",
}

_ALLOWED_DECONTAMINATION_STATUS = {
    "verified",
    "clean",
    "passed",
    "admissible",
    "complete",
    "completed",
}

_CREDENTIAL_KEYS = (
    "credential_ref",
    "api_key",
    "api_token",
    "hf_token",
    "hf_api_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_access_key",
    "wandb_api_key",
    "oauth_token",
    "access_token",
)


class VerilogEvalSplitManifestError(ValueError):
    """Raised for malformed verilog_eval split manifest input."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return True
    return value.strip().lower() in _PLACEHOLDERS


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        raise VerilogEvalSplitManifestError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_status(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerilogEvalSplitManifestError(f"{field} must be explicit status text")
    return value.strip()


def _require_url(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not URL_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be https URL")
    return text


def _require_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX40_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be immutable 40-char hex")
    return text


def _require_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX64_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be 64-char hex")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not SHA256_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be a sha256 digest")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _contains_banned_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _contains_banned_markers(item) for item in value.values() if item is not None
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str, bytearray)):
        return any(_contains_banned_markers(item) for item in value)
    return False


def _load_authority(contract_path: str | Path | None = None) -> Mapping[str, Any]:
    base = Path(__file__).resolve().parent
    path = Path(contract_path) if contract_path is not None else base / "pavlovs_domain_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerilogEvalSplitManifestError(f"cannot read authority contract: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerilogEvalSplitManifestError(
            f"authority contract is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise VerilogEvalSplitManifestError("authority contract must be a JSON object")

    registry = payload.get("suite_registry")
    if not isinstance(registry, Mapping):
        raise VerilogEvalSplitManifestError("authority contract missing suite_registry")
    suite = registry.get(SUITE_ID)
    if not isinstance(suite, Mapping):
        raise VerilogEvalSplitManifestError("authority contract missing verilog_eval suite")
    return suite


def _validate_source(value: Any, authority: Mapping[str, Any]) -> str:
    source = _require_url(value, "source")
    authoritative_source = _require_url(authority.get("url"), "authority suite url")
    if source.rstrip("/").lower() != authoritative_source.rstrip("/").lower():
        raise VerilogEvalSplitManifestError(
            "source must match authoritative verilog_eval source"
        )
    return source


def _validate_dataset(value: Any, authority: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("dataset must be an object")
    revision = _require_hex40(value.get("revision"), "dataset.revision")
    license_text = _require_text(value.get("license"), "dataset.license")
    source = _require_url(value.get("source"), "dataset.source")
    authoritative_source = _require_url(authority.get("url"), "contract suite url")
    if source.rstrip("/").lower() != authoritative_source.rstrip("/").lower():
        raise VerilogEvalSplitManifestError(
            "dataset.source must match authoritative verilog_eval source"
        )

    if _contains_banned_markers(value.get("name")):
        raise VerilogEvalSplitManifestError("dataset references blocked source")
    return revision, license_text, source


def _validate_category(value: Any) -> str:
    category = _require_text(value, "category")
    if category != _EXPECTED_CATEGORY:
        raise VerilogEvalSplitManifestError(
            f"category must be {_EXPECTED_CATEGORY!r}, found {category!r}"
        )
    return category


def _validate_role(value: Any) -> str:
    role = _require_text(value, "role")
    if role != SUITE_ROLE:
        raise VerilogEvalSplitManifestError(
            f"role must be {SUITE_ROLE!r}, found {role!r}"
        )
    return role


def _validate_task_hashes(value: Any, field: str = "task_id_hashes") -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerilogEvalSplitManifestError(f"{field} must be a non-empty list")
    ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_hash = _require_hex64(item, f"{field}[{index}]")
        if item_hash in seen:
            raise VerilogEvalSplitManifestError(f"{field} contains duplicate IDs")
        seen.add(item_hash)
        ids.append(item_hash)

    if not ids:
        raise VerilogEvalSplitManifestError(f"{field} cannot be empty")
    if ids != sorted(ids):
        raise VerilogEvalSplitManifestError(f"{field} must be deterministically sorted")
    return ids


def _validate_split(value: Any, task_ids: list[str]) -> tuple[list[str], str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("split must be an object")
    split_task_ids = _validate_task_hashes(
        value.get("primary_eval"), "split.primary_eval"
    )
    if split_task_ids != task_ids:
        raise VerilogEvalSplitManifestError("split.primary_eval must match task_id_hashes")

    declared = _require_sha256(
        value.get("hash") if value.get("hash") is not None else value.get("sha256"),
        "split.primary_eval.hash",
    )
    aggregate = _sha256("\n".join(split_task_ids))
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalSplitManifestError(
            "split.primary_eval hash does not match aggregate of task IDs"
        )
    return split_task_ids, aggregate, expected


def _validate_split_hashes(value: Mapping[str, Any], aggregate: str) -> dict[str, str]:
    split_hashes = value.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise VerilogEvalSplitManifestError("split_hashes must be an object")
    declared = _require_sha256(
        split_hashes.get("primary_eval"), "split_hashes.primary_eval"
    )
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalSplitManifestError(
            "split_hashes.primary_eval does not match task aggregate"
        )
    return {"primary_eval": declared}


def _validate_split_manifest_hash(
    manifest: Mapping[str, Any], aggregate: str
) -> tuple[str, str]:
    payload = canonical_json({"primary_eval": aggregate})
    observed = f"sha256:{_sha256(payload)}"
    declared = _require_sha256(manifest.get("split_manifest_hash"), "split_manifest_hash")
    if declared != observed:
        raise VerilogEvalSplitManifestError(
            "split_manifest_hash does not match observed split aggregate"
        )
    receipt_ref = _require_sha256(
        manifest.get("split_manifest_receipt_ref"), "split_manifest_receipt_ref"
    )
    return observed, receipt_ref


def _validate_decontamination(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        receipt_id = _require_hex40(value, "decontamination.receipt_id")
        return {
            "status": "verified",
            "receipt_id": receipt_id,
            "visibility": "private",
            "safe_public_artifact": True,
            "url": None,
        }

    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("decontamination must be an object")
    status = _require_status(value.get("status"), "decontamination.status").lower()
    if status not in _ALLOWED_DECONTAMINATION_STATUS:
        raise VerilogEvalSplitManifestError("decontamination.status is invalid")
    receipt_id = _require_hex40(
        value.get("receipt_id"), "decontamination.receipt_id"
    )
    visibility = _require_text(value.get("visibility", "private"), "decontamination.visibility")
    if visibility.lower() == "public" and not bool(value.get("safe_public_artifact")):
        raise VerilogEvalSplitManifestError(
            "public decontamination evidence must set safe_public_artifact=True"
        )
    return {
        "status": status,
        "receipt_id": receipt_id,
        "visibility": visibility.lower(),
        "safe_public_artifact": bool(value.get("safe_public_artifact")),
        "url": value.get("url"),
    }


def _validate_no_network_or_credentials(manifest: Mapping[str, Any], blockers: list[str]) -> None:
    if manifest.get("requires_network") is True:
        blockers.append("dataset download/network is disallowed for split validation")
    for key in _CREDENTIAL_KEYS:
        value = manifest.get(key)
        if not _is_placeholder(value):
            blockers.append(f"{key} is not allowed in offline split validation")


def build_split_manifest_record(
    manifest: Mapping[str, Any],
    *,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic verilog_eval split decision payload."""

    try:
        if not isinstance(manifest, Mapping):
            raise VerilogEvalSplitManifestError("manifest must be a JSON object")

        authority = _load_authority(contract_path)
        if str(authority.get("role", "")).strip().lower() != SUITE_ROLE:
            raise VerilogEvalSplitManifestError("contract role mismatch for verilog_eval")
        if authority.get("stateful") is not _EXPECTED_STATEFUL:
            raise VerilogEvalSplitManifestError("contract stateful value changed")
        if authority.get("artifact_or_side_effect") is not EXPECTED_ARTIFACT_OR_SIDE_EFFECT:
            raise VerilogEvalSplitManifestError(
                "contract artifact_or_side_effect value changed"
            )

        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            raise VerilogEvalSplitManifestError(f"suite_id must be {SUITE_ID!r}")

        source = _validate_source(manifest.get("source"), authority)
        category = _validate_category(manifest.get("category"))
        role = _validate_role(manifest.get("role"))
        dataset_revision, dataset_license, dataset_source = _validate_dataset(
            manifest.get("dataset"), authority
        )
        task_ids = _validate_task_hashes(manifest.get("task_id_hashes"), "task_id_hashes")
        split_ids, aggregate, aggregate_digest = _validate_split(
            manifest.get("split"), task_ids
        )
        split_hashes = _validate_split_hashes(manifest, aggregate)
        split_manifest_hash, split_manifest_receipt_ref = _validate_split_manifest_hash(
            manifest, aggregate
        )
        decontamination = _validate_decontamination(manifest.get("decontamination"))

        evidence = {
            "source": source,
            "dataset": manifest.get("dataset"),
            "split_hashes": split_hashes,
            "verifier": manifest.get("verifier"),
            "environment": manifest.get("environment"),
            "split": manifest.get("split"),
        }
        if _contains_banned_markers(evidence):
            raise VerilogEvalSplitManifestError("manifest contains blocked substitution markers")

        if _contains_banned_markers(source):
            raise VerilogEvalSplitManifestError("source includes blocked tokens")

        scope = manifest.get("scope")
        if isinstance(scope, Mapping) and scope.get("e_suite_ids"):
            raise VerilogEvalSplitManifestError("manifest scope includes E-suite entries")

        blockers: list[str] = []
        _validate_no_network_or_credentials(manifest, blockers)

        blockers = sorted(set(blockers))

        split_manifest_payload = {
            "suite_id": SUITE_ID,
            "task_ids": split_ids,
            "split_hashes": split_hashes,
            "dataset_revision": dataset_revision,
        }
        split_manifest_payload_digest = f"sha256:{_sha256(canonical_json(split_manifest_payload))}"

        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "verilog_eval_split_manifest",
            "suite_id": suite_id,
            "exact_suite": suite_id == SUITE_ID,
            "split_ready": not blockers,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": source,
            "contract_authority": {
                "source": authority.get("url"),
                "role": authority.get("role"),
                "domains": list(authority.get("domains", [])),
                "stateful": authority.get("stateful"),
                "artifact_or_side_effect": authority.get("artifact_or_side_effect"),
            },
            "category": category,
            "role": role,
            "dataset_revision": dataset_revision,
            "dataset_license": dataset_license,
            "dataset_source": dataset_source,
            "split": {
                "primary_eval": {
                    "count": len(split_ids),
                    "task_id_hashes": split_ids,
                    "aggregate_sha256": aggregate,
                    "aggregate_sha256_digest": aggregate_digest,
                }
            },
            "split_hashes": split_hashes,
            "split_manifest_hash": split_manifest_hash,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": split_manifest_receipt_ref,
            "decontamination": decontamination,
            "task_id_digest": _sha256(canonical_json(split_ids)),
            "split_manifest_payload_digest": split_manifest_payload_digest,
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except VerilogEvalSplitManifestError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "verilog_eval_split_manifest",
            "suite_id": manifest.get("suite_id", "<missing>")
            if isinstance(manifest, Mapping)
            else "<invalid>",
            "exact_suite": manifest.get("suite_id") == SUITE_ID
            if isinstance(manifest, Mapping)
            else False,
            "split_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": None,
            "contract_authority": {},
            "category": manifest.get("category", "<missing>")
            if isinstance(manifest, Mapping)
            else "<missing>",
            "role": manifest.get("role", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "split": {"primary_eval": {}},
            "split_hashes": {},
            "split_manifest_hash": manifest.get("split_manifest_hash") if isinstance(manifest, Mapping) else None,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash")
            if isinstance(manifest, Mapping)
            else None,
            "split_manifest_receipt_ref": manifest.get("split_manifest_receipt_ref")
            if isinstance(manifest, Mapping)
            else None,
            "decontamination": None,
            "task_id_digest": None,
            "split_manifest_payload_digest": None,
            "blockers": [str(exc)],
            "status": "BLOCKED",
        }


def validate_split_manifest_record(record: Mapping[str, Any]) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record root must be a JSON object"]
    ready = bool(record.get("split_ready"))
    if record.get("status") == "READY" and not ready:
        return ["record status is READY but split_ready is false"]
    return [] if record.get("status") == "READY" else list(record.get("blockers", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("manifest must be a JSON object")
    report = build_split_manifest_record(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
