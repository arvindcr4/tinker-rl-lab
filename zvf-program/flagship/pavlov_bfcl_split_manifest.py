#!/usr/bin/env python3
"""Offline validation for exact T4 BFCL split/hash/decontamination receipts.

This validator is intentionally metadata-only:

* no data download
* no network mutation
* no credentials
* no paid launch

The decision is fail-closed; any blocker makes the report blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "pavlov-bfcl-t4-split-manifest-v1"
SUITE_ID = "bfcl_train"
NATIVE_CATEGORY = "tool_use"
ROLE_TRAIN = "train"

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")

_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "todo",
    "tbd",
    "pending",
    "placeholder",
    "unrecorded",
    "not_provided",
    "to_be_pinned",
    "latest",
    "main",
    "head",
    "tip",
}

_BANNED_MARKERS = (
    "simulated",
    "synthetic",
    "simulator",
    "glaive",
    "glaiveai/glaive-function-calling-v2",
    "bfclv4_tool_use.py",
    "tool_use_tinker.py",
)

_E_SUITES = {
    "swe_bench_pro_eval",
    "frontier_swe_eval",
    "sdab_eval",
    "banker_toolbench_eval",
    "apex_agents_eval",
    "webbench_eval",
    "binaryaudit_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "agentharm_eval",
    "verilog_eval",
    "appbench_eval",
    "openreward_games_eval",
    "frontiermath_eval",
}

_ALLOWED_DECONTAMINATION_STATUS = {
    "verified",
    "clean",
    "passed",
    "admissible",
    "complete",
    "completed",
}


class BFCLSplitManifestError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    text = value.strip().lower()
    if text in _PLACEHOLDERS:
        return True
    return text.startswith(("pending:", "todo:", "to be pinned", "to_be_pinned"))


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        raise BFCLSplitManifestError(f"{field} must be a non-placeholder text value")
    return value.strip()


def _require_hash(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    text = _require_text(value, field)
    if not pattern.fullmatch(text):
        raise BFCLSplitManifestError(f"{field} must be a hexadecimal hash")
    return text


def _require_sha256_identity(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not _SHA256_RE.fullmatch(text):
        raise BFCLSplitManifestError(f"{field} must be a sha256 hash")
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    return f"sha256:{text}"


def _contains_banned_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_banned_markers(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return any(_contains_banned_markers(item) for item in value)
    return False


def _validate_task_hash_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise BFCLSplitManifestError(f"{field} must be a list")
    if not value:
        raise BFCLSplitManifestError(f"{field} cannot be empty")

    hashes: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_hash = _require_hash(item, f"{field}[{index}]", _HEX64_RE)
        if item_hash in seen:
            raise BFCLSplitManifestError(f"{field} must not contain duplicates")
        seen.add(item_hash)
        hashes.append(item_hash)

    sorted_hashes = sorted(hashes)
    if hashes != sorted_hashes:
        raise BFCLSplitManifestError(f"{field} must be sorted deterministically")
    return hashes


def _split_overview(value: Any, split_name: str, blockers: list[str]) -> dict[str, Any]:
    if isinstance(value, list):
        task_hashes = _validate_task_hash_list(value, f"split.{split_name}")
        declared = None
    elif isinstance(value, Mapping):
        raw_hashes = value.get("task_id_hashes")
        if raw_hashes is None:
            raw_hashes = value.get("task_hashes")
        task_hashes = _validate_task_hash_list(raw_hashes, f"split.{split_name}.task_id_hashes")
        declared = value.get("aggregate_sha256") or value.get("split_hash")
    else:
        blockers.append(f"split.{split_name} must be a list or an object")
        return {"task_hashes": [], "aggregate_sha256": None, "declared_aggregate_sha256": None, "count": 0}

    observed = _sha256("\n".join(task_hashes))
    declared_hash = None
    if declared is not None:
        declared = _require_text(declared, f"split.{split_name}.aggregate_sha256")
        if declared.startswith("sha256:"):
            declared = declared.removeprefix("sha256:")
        if not _HEX64_RE.fullmatch(declared):
            blockers.append(f"split.{split_name} aggregate hash is invalid")
        elif declared != observed:
            blockers.append(f"split.{split_name} aggregate hash mismatch")
        declared_hash = declared
    return {
        "task_hashes": task_hashes,
        "aggregate_sha256": observed,
        "declared_aggregate_sha256": declared_hash,
        "count": len(task_hashes),
    }


def _validate_decontamination(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        receipt_id = _require_hash(value, "decontamination", _HEX40_RE)
        return {
            "status": "verified",
            "receipt_id": receipt_id,
            "visibility": "private",
            "safe_public_artifact": True,
            "url": None,
        }

    if not isinstance(value, Mapping):
        raise BFCLSplitManifestError("decontamination must be a receipt object")

    status = _require_text(value.get("status"), "decontamination.status").lower()
    if status not in _ALLOWED_DECONTAMINATION_STATUS:
        raise BFCLSplitManifestError("decontamination.status is invalid")

    receipt = value.get("receipt_id")
    if receipt is None:
        receipt = value.get("sha256")
    receipt_id = _require_hash(receipt, "decontamination.receipt_id", _HEX40_RE)
    visibility = value.get("visibility", "private")
    if visibility == "public" and value.get("safe_public_artifact") is not True:
        raise BFCLSplitManifestError("public decontamination receipt must be safe_public_artifact=True")
    return {
        "status": status,
        "receipt_id": receipt_id,
        "visibility": visibility,
        "safe_public_artifact": bool(value.get("safe_public_artifact")),
        "url": value.get("url"),
    }


def build_split_manifest_record(manifest: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            blockers.append(f"suite_id must be {SUITE_ID!r}")
        if suite_id in _E_SUITES:
            blockers.append(f"suite_id {suite_id!r} is an E-suite and is not valid for T4 split validation")

        role = manifest.get("role", ROLE_TRAIN)
        if role != ROLE_TRAIN:
            blockers.append("suite role must be training for T4")

        category = _require_text(manifest.get("category"), "category")
        if category != NATIVE_CATEGORY:
            blockers.append(f"category must be {NATIVE_CATEGORY!r}")

        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            raise BFCLSplitManifestError("dataset is required and must be an object")

        dataset_revision = _require_hash(dataset.get("revision"), "dataset.revision", _HEX40_RE)
        dataset_license = _require_text(dataset.get("license"), "dataset.license")
        dataset_source = _require_text(dataset.get("source"), "dataset.source")
        if _contains_banned_markers(dataset_source) or _contains_banned_markers(dataset):
            blockers.append("dataset/source references synthetic simulator or Glaive evidence")

        split = manifest.get("split")
        if not isinstance(split, Mapping):
            blockers.append("split is required and must be an object")
            train_overview = {
                "task_hashes": [],
                "aggregate_sha256": None,
                "declared_aggregate_sha256": None,
                "count": 0,
            }
            primary_overview = train_overview.copy()
        else:
            train_overview = _split_overview(split.get("train"), "train", blockers)
            primary_overview = _split_overview(split.get("primary_eval"), "primary_eval", blockers)
            if not train_overview["task_hashes"] or not primary_overview["task_hashes"]:
                blockers.append("split must include non-empty train and primary_eval task hashes")
            else:
                overlap = sorted(
                    set(train_overview["task_hashes"]).intersection(primary_overview["task_hashes"])
                )
                if overlap:
                    blockers.append(f"train and primary_eval task IDs overlap: {', '.join(overlap)}")

        if manifest.get("requires_network") is True:
            blockers.append("dataset download/network is disallowed in this boundary")

        for credential_key in (
            "credential_ref",
            "api_key",
            "wandb_api_key",
            "hf_token",
            "hf_api_token",
            "openai_api_key",
            "secret_access_key",
        ):
            if manifest.get(credential_key) not in (None, False, ""):
                blockers.append(f"{credential_key} is not allowed in offline split validation")

        split_hashes = manifest.get("split_hashes")
        if not isinstance(split_hashes, Mapping):
            split_hashes = {}
            blockers.append("split_hashes must be an object")
        train_expected = train_overview["aggregate_sha256"]
        primary_expected = primary_overview["aggregate_sha256"]
        if train_expected is not None:
            declared_train = split_hashes.get("train")
            if declared_train is None:
                blockers.append("split_hashes.train is required")
            elif _require_sha256_identity(declared_train, "split_hashes.train") != f"sha256:{train_expected}":
                blockers.append("split_hashes.train does not match observed train aggregate")
        if primary_expected is not None:
            declared_primary = split_hashes.get("primary_eval")
            if declared_primary is None:
                blockers.append("split_hashes.primary_eval is required")
            elif _require_sha256_identity(declared_primary, "split_hashes.primary_eval") != f"sha256:{primary_expected}":
                blockers.append("split_hashes.primary_eval does not match observed primary_eval aggregate")

        split_manifest_hash_payload = canonical_json(
            {
                "primary_eval": primary_overview["aggregate_sha256"],
                "train": train_overview["aggregate_sha256"],
            }
        )
        split_manifest_hash = f"sha256:{_sha256(split_manifest_hash_payload)}"
        if manifest.get("split_manifest_hash") is not None:
            declared_manifest = _require_sha256_identity(manifest["split_manifest_hash"], "split_manifest_hash")
            if declared_manifest != split_manifest_hash:
                blockers.append("split_manifest_hash does not match observed split aggregates")

        split_manifest_receipt_ref = _require_sha256_identity(
            manifest.get("split_manifest_receipt_ref"), "split_manifest_receipt_ref"
        )
        decontamination = _validate_decontamination(manifest.get("decontamination"))
        decontamination_receipt_ref = manifest.get("decontamination_receipt_ref")
        if decontamination_receipt_ref is None:
            decontamination_receipt_ref = f"sha256:{decontamination['receipt_id']}"
        else:
            decontamination_receipt_ref = _require_sha256_identity(
                decontamination_receipt_ref,
                "decontamination_receipt_ref",
            )

        if isinstance(manifest.get("scope"), Mapping):
            scope_e = manifest.get("scope", {}).get("e_suite_ids")
            if scope_e is not None and isinstance(scope_e, list) and any(item in _E_SUITES for item in scope_e):
                blockers.append("manifest scope includes E-suite entries")
        if manifest.get("e_suite_ids") is not None:
            e_values = manifest.get("e_suite_ids")
            if isinstance(e_values, list) and any(item in _E_SUITES for item in e_values):
                blockers.append("manifest includes E-suite references")

        if _contains_banned_markers(decontamination):
            blockers.append("decontamination evidence references banned synthetic/Glaive sources")

        if _contains_banned_markers(manifest.get("notes")):
            blockers.append("manifest notes references synthetic/Glaive sources")

        if manifest.get("paid_launch_allowed") is True:
            blockers.append("paid_launch_allowed cannot be true in offline validator")

        blockers = sorted(set(blockers))

        split_digest = _sha256(
            canonical_json(
                {
                    "suite_id": suite_id,
                    "split": {"train": train_overview["aggregate_sha256"], "primary_eval": primary_overview["aggregate_sha256"]},
                    "decontamination_receipt_id": decontamination["receipt_id"],
                }
            )
        )
        manifest_identity_digest = split_digest

        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "role": role,
            "manifest_type": "bfcl_t4_split_manifest",
            "split_ready": not blockers,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": category,
            "dataset_revision": dataset_revision,
            "dataset_license": dataset_license,
            "dataset_source": dataset_source,
            "split": {
                "train": train_overview,
                "primary_eval": primary_overview,
            },
            "split_hashes": {
                "train": split_hashes.get("train"),
                "primary_eval": split_hashes.get("primary_eval"),
            },
            "split_manifest_hash": split_manifest_hash,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": split_manifest_receipt_ref,
            "decontamination": decontamination,
            "decontamination_receipt_ref": decontamination_receipt_ref,
            "dataset_source_is_disallowed": any(
                marker in dataset_source.lower() for marker in ("glaive", "simulated", "synthetic")
            ),
            "split_digest": f"sha256:{split_digest}",
            "manifest_identity_digest": f"sha256:{manifest_identity_digest}",
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except BFCLSplitManifestError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.get("suite_id", "<missing>"),
            "role": manifest.get("role", ROLE_TRAIN),
            "manifest_type": "bfcl_t4_split_manifest",
            "split_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "category": manifest.get("category"),
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "split": {
                "train": {},
                "primary_eval": {},
            },
            "split_hashes": {"train": None, "primary_eval": None},
            "split_manifest_hash": None,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": manifest.get("split_manifest_receipt_ref"),
            "decontamination": None,
            "decontamination_receipt_ref": manifest.get("decontamination_receipt_ref"),
            "dataset_source_is_disallowed": False,
            "split_digest": None,
            "manifest_identity_digest": None,
            "blockers": blockers or [str(exc)],
            "status": "BLOCKED",
        }


def validate_split_manifest_record(record: Mapping[str, Any]) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record root must be a JSON object"]
    return [] if record.get("status") == "READY" else list(record.get("blockers", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise SystemExit("manifest must be a JSON object")

    report = build_split_manifest_record(data)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
