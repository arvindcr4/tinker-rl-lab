#!/usr/bin/env python3
"""Offline metadata adapter for exact E10/E14 Pavlov primary-eval boundaries.

The module never performs network I/O.  It only validates that local metadata is
consistent with the published Pavlov contract and that every required receipt field
is non-placeholder and immutable before allowing a boundary to be treated as
evidence-ready.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .pavlovs_domain_contract import load_contract
except ImportError:  # pragma: no cover - direct execution fallback
    from pavlovs_domain_contract import load_contract

SCHEMA_VERSION = "pavlov-agentharm-frontiermath-adapter-v1"
ADAPTER_ID = "pavlov-e10-e14-official-boundary-adapter-v1"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

SUPPORTED_SUITE_IDS = ("agentharm_eval", "frontiermath_eval")

REQUIRED_BOUNDARY_KEYS = (
    "schema_version",
    "adapter_id",
    "suite_id",
    "suite_role",
    "primary_eval",
    "heldout",
    "model_id",
    "model_revision",
    "authoritative_source",
    "stateful",
    "artifact_or_side_effect",
    "component_only",
    "native_contract",
    "domains",
    "receipts",
)

REQUIRED_RECEIPT_KEYS = (
    "dataset_id",
    "dataset_or_source_revision",
    "license_or_approval",
    "task_id_hashes",
    "split_task_id_hash",
    "split_manifest_hash",
    "container_runtime_digest",
    "verifier_hash",
    "model_id",
    "model_revision",
    "decontamination_status",
    "budget_receipt",
    "wandb_run_identity",
    "tinker_run_identity",
    "cost_status",
    "hf_checkpoints",
    "evidence_status",
    "heldout_receipt",
)

REQUIRED_BUNDLE_KEYS = (
    "schema_version",
    "adapter_id",
    "suite_count",
    "boundaries",
    "bundle_signature",
)

_PLACEHOLDER_WORDS = {
    "",
    "none",
    "null",
    "nil",
    "na",
    "n/a",
    "undefined",
    "unknown",
    "todo",
    "tbd",
    "unset",
    "pending",
    "missing",
    "placeholder",
    "to_be_pinned_before_paid_runs",
    "to_be_pinned",
    "license-receipt",
}
_ZERO_40 = "0" * 40
_ZERO_64 = "0" * 64

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_URL_RE = re.compile(r"^https://\S+$")

_BANNED_SUBSTITUTION_MARKERS = (
    "xlam",
    "xlam-function-calling-60k",
    "salesforce/xlam-function-calling-60k",
    "gsm8k",
    "math-500",
    "math500",
    "huggingfaceh4/math-500",
    "openr1_math",
    "openr1 math",
    "openr1-math",
    "openr1_math-220k",
    "openhf/hf",
    "open-reward",
    "frontiermath_math_500",
)

_LICENSE_TOKENS = {"spdx:", "apache", "mit", "cc-", "bsd", "lgpl", "gpl"}

_VALID_DECONTAMINATION = {"verified", "complete", "completed", "passed", "clean"}
_VALID_WANDB_STATUS = {"approved", "authorized", "ready", "running", "finished"}
_VALID_TINKER_COST_STATUS = {
    "authorized",
    "approved",
    "within_cap",
    "observed",
    "complete",
    "finished",
}
_VALID_EVIDENCE_STATUS = {"prospective", "observed", "admissible", "rejected", "pending"}

_NATIVE_CONTRACT = {
    "agentharm_eval": {
        "environment": {
            "name": "policy_grader",
            "mode": "policy-grader",
            "artifact_required": True,
        },
        "verifier": {
            "name": "agentharm_policy_verifier",
            "mode": "policy-v1",
        },
        "artifact": {
            "mode": "stateful",
        },
    },
    "frontiermath_eval": {
        "environment": {
            "name": "frontiermath_native",
            "mode": "native-math",
            "artifact_required": False,
        },
        "verifier": {
            "name": "frontiermath_verifier",
            "mode": "math-native-v1",
        },
        "artifact": {
            "mode": "stateless",
        },
    },
}


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding used for hashing and signature checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash textual input with SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def aggregate_task_id_hashes(task_id_hashes: Sequence[str]) -> str:
    """Deterministically aggregate ordered task hashes into one manifest hash."""

    return sha256_text("\n".join(task_id_hashes))


def _placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_WORDS
    return False


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _as_sequence(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return None


def _is_hex40(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().removeprefix("sha256:")
    if normalized in {_ZERO_40, _ZERO_64[:40]}:
        return False
    return bool(_HEX40_RE.fullmatch(value.strip()))


def _is_hex64(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().removeprefix("sha256:")
    if normalized in {_ZERO_64, _ZERO_40}:
        return False
    return bool(_HEX64_RE.fullmatch(value.strip()))


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().removeprefix("sha256:")
    if normalized in {_ZERO_64, _ZERO_40}:
        return False
    return bool(_SHA256_RE.fullmatch(value.strip()))


def _is_immutable_revision(value: Any) -> bool:
    return _is_hex40(value) or _is_sha256(value)


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and bool(_URL_RE.fullmatch(value.strip()))


def _expected_source_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if parsed.netloc == "github.com" and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parsed.netloc and parts:
        return f"{parsed.netloc}/" + "/".join(parts)
    return parsed.netloc


def _split_is_heldout(split: str) -> bool:
    lowered = split.lower()
    return "held-out" in lowered or lowered.startswith("private")


def _load_contract_suite(contract: Mapping[str, Any], suite_id: str) -> Mapping[str, Any]:
    suites = contract.get("suite_registry", {})
    if not isinstance(suites, Mapping):
        raise ValueError("contract suite_registry must be an object")
    suite = suites.get(suite_id)
    if not isinstance(suite, Mapping):
        raise ValueError(f"contract is missing suite {suite_id}")
    return suite


def _native_contract_signature(spec: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(spec))


def _native_contract_for_suite(suite_id: str) -> Mapping[str, Any]:
    spec = _NATIVE_CONTRACT[suite_id]
    return {
        "environment": {
            "name": spec["environment"]["name"],
            "mode": spec["environment"]["mode"],
            "artifact_required": bool(spec["environment"]["artifact_required"]),
            "contract_sha256": _native_contract_signature(spec["environment"]),
        },
        "verifier": {
            "name": spec["verifier"]["name"],
            "mode": spec["verifier"]["mode"],
            "verifier_sha256": _native_contract_signature(spec["verifier"]),
        },
        "artifact": {
            "mode": spec["artifact"]["mode"],
            "artifact_sha256": _native_contract_signature(spec["artifact"]),
        },
    }


def _expected_bundle_signature(bundle: Mapping[str, Any]) -> str:
    payload = {key: bundle[key] for key in bundle if key != "bundle_signature"}
    return sha256_text(canonical_json(payload))


def update_bundle_signature(bundle: dict[str, Any]) -> str:
    """Attach a deterministic bundle signature after local mutation."""

    signature = _expected_bundle_signature(bundle)
    bundle["bundle_signature"] = signature
    return signature


def build_boundary_receipts(suite_id: str) -> dict[str, Any]:
    """Create a deterministic blank receipt layout for the suite boundary."""

    if suite_id not in SUPPORTED_SUITE_IDS:
        raise ValueError(f"unsupported suite_id {suite_id}")

    return {
        "dataset_id": None,
        "dataset_or_source_revision": None,
        "license_or_approval": None,
        "task_id_hashes": [],
        "split_task_id_hash": None,
        "split_manifest_hash": None,
        "container_runtime_digest": None,
        "verifier_hash": None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "decontamination_status": None,
        "budget_receipt": None,
        "wandb_run_identity": {
            "entity": None,
            "project": None,
            "group": None,
            "run_id": None,
            "run_url": None,
            "online": False,
        },
        "tinker_run_identity": {
            "run_id": None,
            "cost_status": "missing",
        },
        "cost_status": "missing",
        "hf_checkpoints": [],
        "evidence_status": "prospective",
        "heldout_receipt": None,
    }


def generate_boundary_bundle() -> dict[str, Any]:
    """Generate a deterministic metadata-only boundary bundle for E10/E14."""

    contract = load_contract()
    boundaries: list[dict[str, Any]] = []
    for suite_id in SUPPORTED_SUITE_IDS:
        suite = _load_contract_suite(contract, suite_id)
        split = str(suite.get("split", ""))
        source_url = str(suite.get("url", ""))
        boundary = {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
            "suite_id": suite_id,
            "suite_role": "primary_eval",
            "primary_eval": True,
            "heldout": _split_is_heldout(split),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "authoritative_source": {
                "contract_schema_version": str(contract.get("schema_version")),
                "source_url": source_url,
                "source_id": _expected_source_id(source_url),
                "split": split,
            },
            "stateful": bool(suite.get("stateful")),
            "artifact_or_side_effect": bool(suite.get("artifact_or_side_effect")),
            "component_only": False,
            "native_contract": _native_contract_for_suite(suite_id),
            "domains": sorted({str(v) for v in suite.get("domains", ())}),
            "receipts": build_boundary_receipts(suite_id),
        }
        boundaries.append(boundary)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "suite_count": len(boundaries),
        "boundaries": boundaries,
    }
    bundle["bundle_signature"] = _expected_bundle_signature(bundle)
    return bundle


def _validate_authoritative_source(
    suite_id: str,
    contract: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    suite = _load_contract_suite(contract, suite_id)
    authoritative = boundary.get("authoritative_source")
    if not isinstance(authoritative, Mapping):
        return [f"{suite_id}: authoritative_source must be an object"]

    source_url = str(suite.get("url", ""))
    if authoritative.get("source_url") != source_url:
        errors.append(f"{suite_id}: authoritative_source.source_url must equal contract url")
    if authoritative.get("split") != str(suite.get("split", "")):
        errors.append(f"{suite_id}: authoritative_source.split must equal contract split")

    source_id = str(authoritative.get("source_id", ""))
    expected_source_id = _expected_source_id(source_url)
    if _placeholder(source_id):
        errors.append(f"{suite_id}: authoritative_source.source_id is required")
    elif source_id != expected_source_id:
        errors.append(
            f"{suite_id}: authoritative_source.source_id {source_id!r} does not match expected {expected_source_id!r}"
        )

    if not _is_url(authoritative.get("source_url")):
        errors.append(f"{suite_id}: authoritative_source.source_url must be a public HTTPS URL")
    return errors


def _validate_native_contract(suite_id: str, contract: Mapping[str, Any]) -> list[str]:
    """Validate the native environment/artifact/verifier contract metadata."""

    errors: list[str] = []
    expected = _NATIVE_CONTRACT.get(suite_id)
    if expected is None:
        return [f"{suite_id}: suite has no native contract metadata definition"]
    if not isinstance(contract, Mapping):
        return [f"{suite_id}: native_contract must be an object"]

    sections = (
        ("environment", "environment", "contract_sha256"),
        ("verifier", "verifier", "verifier_sha256"),
        ("artifact", "artifact", "artifact_sha256"),
    )

    for section, key, digest_key in sections:
        observed = contract.get(section)
        if not isinstance(observed, Mapping):
            errors.append(f"{suite_id}: native_contract.{section} must be an object")
            continue

        spec = expected.get(section, {})
        for field in ("name", "mode"):
            expected_value = spec.get(field)
            if expected_value is None:
                continue
            if _placeholder(observed.get(field)):
                errors.append(f"{suite_id}: native_contract.{section}.{field} is missing")
            elif str(observed.get(field)) != str(expected_value):
                errors.append(f"{suite_id}: native_contract.{section}.{field} must be exact")

        if section == "environment":
            if not isinstance(observed.get("artifact_required"), bool):
                errors.append(
                    f"{suite_id}: native_contract.environment.artifact_required must be boolean"
                )
            elif observed.get("artifact_required") != bool(spec.get("artifact_required")):
                errors.append(
                    f"{suite_id}: native_contract.environment.artifact_required mismatch"
                )

        expected_digest = _native_contract_signature(spec)
        if _placeholder(observed.get(digest_key)):
            errors.append(f"{suite_id}: native_contract.{section}.{digest_key} is missing")
        elif str(observed.get(digest_key)) != expected_digest:
            errors.append(
                f"{suite_id}: native_contract.{section}.{digest_key} must be contract signature"
            )

    return errors


def _validate_dataset_substitution(suite_id: str, receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset_id = receipt.get("dataset_id")
    if _placeholder(dataset_id):
        errors.append(f"{suite_id}: dataset_id is missing")
        return errors
    lowered = str(dataset_id).lower()
    for marker in _BANNED_SUBSTITUTION_MARKERS:
        if marker in lowered:
            errors.append(
                f"{suite_id}: dataset_id appears to substitute xLAM or related benchmark ({marker})"
            )
            return errors
    return errors


def _validate_receipt_fields(suite_id: str, receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    for key in REQUIRED_RECEIPT_KEYS:
        if key not in receipt:
            errors.append(f"{suite_id}: receipt missing {key}")

    if errors:
        return errors

    if _placeholder(receipt.get("dataset_id")):
        errors.append(f"{suite_id}: dataset_id is missing")
    errors.extend(_validate_dataset_substitution(suite_id, receipt))

    if not _is_immutable_revision(receipt.get("dataset_or_source_revision")):
        errors.append(f"{suite_id}: dataset_or_source_revision must be immutable")

    license_value = receipt.get("license_or_approval")
    if _placeholder(license_value):
        errors.append(f"{suite_id}: license_or_approval is required")
    elif isinstance(license_value, Mapping):
        status = str(license_value.get("status", "")).strip().lower()
        approved = str(license_value.get("approved", "")).strip().lower() in {"true", "1", "yes"}
        if status not in {"approved", "verified", "signoff", "ok"} and not approved:
            errors.append(f"{suite_id}: license_or_approval status must show approval")
        proof = _first_value(
            license_value,
            ("license_id", "receipt_id", "approval_id", "approval", "sha256", "hash"),
        )
        if _placeholder(proof):
            errors.append(f"{suite_id}: license_or_approval identity is missing")
    else:
        text = str(license_value).lower()
        if any(token in text for token in _LICENSE_TOKENS) or _is_immutable_revision(license_value):
            pass
        else:
            errors.append(f"{suite_id}: license_or_approval must be concrete and approved")

    task_hashes = receipt.get("task_id_hashes")
    if not isinstance(task_hashes, list) or not task_hashes:
        errors.append(f"{suite_id}: task_id_hashes must be a non-empty list")
    else:
        normalized: list[str] = []
        for offset, raw in enumerate(task_hashes):
            value = str(raw) if isinstance(raw, str) else None
            if value is None or not _is_sha256(value):
                errors.append(f"{suite_id}: task_id_hashes[{offset}] must be a SHA-256 digest")
                continue
            normalized.append(value.lower())
        if len(normalized) != len(set(normalized)):
            errors.append(f"{suite_id}: task_id_hashes must be unique")
        split_task_hash = receipt.get("split_task_id_hash")
        if not _is_sha256(split_task_hash):
            errors.append(f"{suite_id}: split_task_id_hash must be a SHA-256 digest")
        elif normalized and aggregate_task_id_hashes(normalized) != str(split_task_hash).lower():
            errors.append(f"{suite_id}: split_task_id_hash does not match task_id_hashes")

    if not _is_sha256(receipt.get("split_manifest_hash")):
        errors.append(f"{suite_id}: split_manifest_hash must be a SHA-256 digest")

    if not _is_sha256(receipt.get("container_runtime_digest")):
        errors.append(f"{suite_id}: container_runtime_digest must be a SHA-256 digest")
    if not _is_sha256(receipt.get("verifier_hash")):
        errors.append(f"{suite_id}: verifier_hash must be a SHA-256 digest")

    if str(receipt.get("model_id")) != MODEL_ID:
        errors.append(f"{suite_id}: receipt model_id must be {MODEL_ID}")
    if receipt.get("model_revision") != MODEL_REVISION:
        errors.append(f"{suite_id}: receipt model_revision must be {MODEL_REVISION}")

    decontamination = receipt.get("decontamination_status")
    if not isinstance(decontamination, Mapping):
        errors.append(f"{suite_id}: decontamination_status must be an object")
    else:
        if str(decontamination.get("status", "")).lower() not in _VALID_DECONTAMINATION:
            errors.append(f"{suite_id}: decontamination_status.status must be verified/clean")
        if _placeholder(_first_value(decontamination, ("receipt_id", "sha256", "hash", "digest"))):
            errors.append(f"{suite_id}: decontamination_status identity is missing")

    budget = receipt.get("budget_receipt")
    if not isinstance(budget, Mapping):
        errors.append(f"{suite_id}: budget_receipt must be an object")
    else:
        status = str(budget.get("status", "")).lower()
        authorized = budget.get("authorized") is True or status in {"authorized", "approved"}
        if not authorized:
            errors.append(f"{suite_id}: budget_receipt must be authorized")
        max_usd = budget.get("maximum_usd")
        if not isinstance(max_usd, (int, float)):
            max_usd = budget.get("max_usd")
        if not isinstance(max_usd, (int, float)) or max_usd <= 0:
            errors.append(f"{suite_id}: budget_receipt.maximum_usd must be a positive number")
        if _placeholder(
            _first_value(budget, ("receipt_id", "authorization_id", "id", "sha256", "hash"))
        ):
            errors.append(f"{suite_id}: budget_receipt identity is missing")

    wandb = receipt.get("wandb_run_identity")
    if not isinstance(wandb, Mapping):
        errors.append(f"{suite_id}: wandb_run_identity must be an object")
    else:
        if wandb.get("online") is not True:
            errors.append(f"{suite_id}: wandb_run_identity.online must be true")
        for field in ("entity", "project", "group", "run_id", "run_url"):
            if _placeholder(wandb.get(field)):
                errors.append(f"{suite_id}: wandb_run_identity.{field} is missing")
        if not _is_url(wandb.get("run_url")):
            errors.append(f"{suite_id}: wandb_run_identity.run_url must be https URL")
        if str(wandb.get("mode", "online")).lower() not in {"online", "ready", "true", "1"}:
            errors.append(f"{suite_id}: wandb mode must be online/ready")

    tinker = receipt.get("tinker_run_identity")
    if not isinstance(tinker, Mapping):
        errors.append(f"{suite_id}: tinker_run_identity must be an object")
    else:
        if _placeholder(tinker.get("run_id")):
            errors.append(f"{suite_id}: tinker_run_identity.run_id is missing")
        if str(tinker.get("cost_status", "")).lower() not in _VALID_TINKER_COST_STATUS:
            errors.append(f"{suite_id}: tinker_run_identity.cost_status is invalid")
        if not _placeholder(tinker.get("run_url")) and not _is_url(tinker.get("run_url")):
            errors.append(f"{suite_id}: tinker_run_identity.run_url must be https URL")

    if str(receipt.get("cost_status", "")).lower() not in _VALID_TINKER_COST_STATUS:
        errors.append(f"{suite_id}: cost_status is invalid")

    checkpoints = receipt.get("hf_checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        errors.append(f"{suite_id}: hf_checkpoints must be a non-empty list")
    else:
        seen_stages: dict[str, int] = {}
        seen_cp_run_ids: set[str] = set()
        seen_cp_identities: set[tuple[str, str, str]] = set()
        for index, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, Mapping):
                errors.append(f"{suite_id}: hf_checkpoints[{index}] must be an object")
                continue

            stage = str(checkpoint.get("stage", "")).strip().lower()
            if stage:
                seen_stages[stage] = seen_stages.get(stage, 0) + 1
            else:
                errors.append(f"{suite_id}: hf_checkpoints[{index}].stage is missing")

            revision = checkpoint.get("revision")
            if not _is_hex40(revision):
                errors.append(
                    f"{suite_id}: hf_checkpoints[{index}].revision must be a 40-char revision"
                )

            for field in ("repo_url", "url"):
                if not _is_url(checkpoint.get(field)):
                    errors.append(f"{suite_id}: hf_checkpoints[{index}].{field} must be public HTTPS URL")

            visibility = checkpoint.get("visibility")
            if visibility not in {"public", "private"}:
                errors.append(f"{suite_id}: hf_checkpoints[{index}].visibility invalid")
            if visibility == "public" and checkpoint.get("safe_public_artifact") is not True:
                errors.append(
                    f"{suite_id}: hf_checkpoints[{index}].safe_public_artifact must be true for public visibility"
                )

            cp_run_id = checkpoint.get("run_id")
            if _placeholder(cp_run_id):
                errors.append(f"{suite_id}: hf_checkpoints[{index}].run_id is missing")
            else:
                seen_cp_run_ids.add(str(cp_run_id))

            repo_url = str(checkpoint.get("repo_url", ""))
            cp_revision = str(checkpoint.get("revision", ""))
            cp_url = str(checkpoint.get("url", ""))
            cp_identity = (repo_url, cp_revision, cp_url)
            if cp_identity in seen_cp_identities:
                errors.append(f"{suite_id}: hf_checkpoints[{index}] duplicates an existing checkpoint entry")
            else:
                seen_cp_identities.add(cp_identity)

        for required_stage in ("initial", "periodic", "final"):
            stage_count = seen_stages.get(required_stage, 0)
            if stage_count == 0:
                errors.append(f"{suite_id}: hf_checkpoints missing required stage {required_stage}")
            elif stage_count > 1:
                errors.append(
                    f"{suite_id}: hf_checkpoints duplicate entries for stage {required_stage}"
                )

        tinker_identity = receipt.get("tinker_run_identity")
        if isinstance(tinker_identity, Mapping):
            tinker_run_id = str(_first_value(tinker_identity, ("run_id", "id", "run")))
        else:
            tinker_run_id = ""
        tinker_run_id = tinker_run_id if not _placeholder(tinker_run_id) else ""
        if tinker_run_id and seen_cp_run_ids and seen_cp_run_ids != {tinker_run_id}:
            errors.append(
                f"{suite_id}: hf checkpoint run_id values must match tinker_run_identity.run_id"
            )

    if str(receipt.get("evidence_status", "")).lower() not in _VALID_EVIDENCE_STATUS:
        errors.append(f"{suite_id}: evidence_status is invalid")

    return errors


def _validate_heldout_receipt(suite_id: str, receipt: Mapping[str, Any]) -> list[str]:
    heldout_receipt = receipt.get("heldout_receipt")
    if _placeholder(heldout_receipt):
        return [f"{suite_id}: heldout suite requires heldout_receipt"]
    if not isinstance(heldout_receipt, Mapping):
        return [f"{suite_id}: heldout_receipt must be an object"]
    status = str(heldout_receipt.get("status", "")).lower()
    if status not in {"verified", "complete", "completed", "passed"}:
        return [f"{suite_id}: heldout_receipt.status is not admissible"]
    if _placeholder(_first_value(heldout_receipt, ("receipt_id", "id", "sha256", "hash"))):
        return [f"{suite_id}: heldout_receipt identity is missing"]
    return []


def validate_boundary(boundary: Mapping[str, Any]) -> list[str]:
    """Validate one suite boundary with contract-anchored checks."""

    if not isinstance(boundary, Mapping):
        return ["boundary must be an object"]

    for key in REQUIRED_BOUNDARY_KEYS:
        if key not in boundary:
            return [f"boundary missing key {key}"]

    if len(boundary) != len(REQUIRED_BOUNDARY_KEYS):
        extras = sorted(set(boundary) - set(REQUIRED_BOUNDARY_KEYS))
        if extras:
            return ["boundary has unexpected top-level fields: " + ", ".join(extras)]

    suite_id = str(boundary.get("suite_id"))
    if suite_id not in SUPPORTED_SUITE_IDS:
        return [f"unsupported suite_id {suite_id}"]

    if boundary.get("schema_version") != SCHEMA_VERSION:
        return [f"{suite_id}: schema_version mismatch"]
    if boundary.get("adapter_id") != ADAPTER_ID:
        return [f"{suite_id}: adapter_id mismatch"]
    if boundary.get("suite_role") != "primary_eval":
        return [f"{suite_id}: suite_role must be primary_eval"]
    if boundary.get("primary_eval") is not True:
        return [f"{suite_id}: primary_eval must be true"]
    if boundary.get("component_only") is not False:
        return [f"{suite_id}: component_only must be false"]

    if str(boundary.get("model_id")) != MODEL_ID:
        return [f"{suite_id}: model_id must be {MODEL_ID}"]
    if boundary.get("model_revision") != MODEL_REVISION:
        return [f"{suite_id}: model_revision must be pinned {MODEL_REVISION}"]

    contract = load_contract()
    suite = _load_contract_suite(contract, suite_id)
    if suite.get("role") != "primary_eval":
        return [f"{suite_id}: contract role must be primary_eval"]

    if tuple(boundary.get("domains", ())) != tuple(sorted({str(v) for v in suite.get("domains", ())})):
        return [f"{suite_id}: domains must match contract"]

    if bool(boundary.get("stateful")) != bool(suite.get("stateful")):
        return [f"{suite_id}: stateful flag must match contract"]

    expected_artifact = bool(suite.get("artifact_or_side_effect"))
    if bool(boundary.get("artifact_or_side_effect")) != expected_artifact:
        return [f"{suite_id}: artifact_or_side_effect flag must match contract"]

    heldout = _split_is_heldout(str(suite.get("split", "")))
    if bool(boundary.get("heldout")) != heldout:
        return [f"{suite_id}: heldout flag does not match split metadata"]

    errors = _validate_authoritative_source(suite_id, contract, boundary)
    errors.extend(_validate_native_contract(suite_id, boundary.get("native_contract", {})))
    receipt = boundary.get("receipts")
    if not isinstance(receipt, Mapping):
        errors.append(f"{suite_id}: receipts must be an object")
        return errors
    errors.extend(_validate_receipt_fields(suite_id, receipt))
    if errors:
        return errors

    if heldout:
        errors.extend(_validate_heldout_receipt(suite_id, receipt))
    return errors


def validate_adapter_bundle(bundle: Mapping[str, Any]) -> list[str]:
    """Validate the full adapter bundle and return ordered fail-closed errors."""

    if not isinstance(bundle, Mapping):
        return ["adapter bundle must be an object"]

    for key in REQUIRED_BUNDLE_KEYS:
        if key not in bundle:
            return [f"adapter bundle missing {key}"]

    if bundle.get("schema_version") != SCHEMA_VERSION:
        return ["adapter bundle schema_version mismatch"]
    if bundle.get("adapter_id") != ADAPTER_ID:
        return ["adapter bundle adapter_id mismatch"]
    boundaries = bundle.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != len(SUPPORTED_SUITE_IDS):
        return ["adapter bundle must contain exactly two suite boundaries"]

    expected_signature = _expected_bundle_signature(bundle)
    if bundle.get("bundle_signature") != expected_signature:
        return ["bundle_signature is invalid"]
    if bundle.get("suite_count") != len(SUPPORTED_SUITE_IDS):
        return ["suite_count must be 2"]

    errors: list[str] = []
    suite_ids: list[str] = []

    for boundary in boundaries:
        if not isinstance(boundary, Mapping):
            errors.append("each boundary must be an object")
            continue
        suite_id = str(boundary.get("suite_id"))
        suite_ids.append(suite_id)
        errors.extend(validate_boundary(boundary))

    if sorted(set(suite_ids)) != sorted(SUPPORTED_SUITE_IDS):
        errors.append("adapter bundle must only contain agentharm_eval and frontiermath_eval")
    if len(set(suite_ids)) != len(suite_ids):
        errors.append("adapter bundle boundaries contain duplicate suite IDs")
    return errors


def evaluate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return readiness/claim metadata for each exact suite boundary."""

    errors = validate_adapter_bundle(bundle)
    suite_readiness: dict[str, dict[str, Any]] = {}

    boundaries = bundle.get("boundaries") if isinstance(bundle, Mapping) else None
    if isinstance(boundaries, list):
        for boundary in boundaries:
            if not isinstance(boundary, Mapping):
                continue
            suite_id = str(boundary.get("suite_id"))
            receipt = boundary.get("receipts", {}) if isinstance(boundary.get("receipts"), Mapping) else {}
            heldout = bool(boundary.get("heldout"))
            heldout_proven = (not heldout) or not _validate_heldout_receipt(
                suite_id, receipt
            )
            primary_eval_claim_allowed = bool(boundary.get("primary_eval")) and bool(heldout_proven)
            suite_readiness[suite_id] = {
                "structural_heldout": heldout,
                "heldout_receipt_proven": heldout_proven,
                "primary_eval_claim_allowed": primary_eval_claim_allowed,
            }

    return {
        "status": "READY" if not errors else "BLOCKED",
        "errors": errors,
        "suite_readiness": suite_readiness,
    }


# ---------------------------------------------------------------------------
# E10 AgentHarm split binding
#
# AgentHarm ships three public splits and three held-out ("private") splits.  The
# held-out files are absent from the public dataset revision and are gated by the
# UK AI Safety Institute.  Everything below exists so that a score can only ever
# be emitted against the held-out split, with a verifier and dataset revision that
# are both pinned and hashed.  A run on the public split is harness evidence, not
# an AgentHarm score, and the emitter enforces that mechanically.
# ---------------------------------------------------------------------------

AGENTHARM_SUITE_ID = "agentharm_eval"
AGENTHARM_DATASET_ID = "ai-safety-institute/AgentHarm"
AGENTHARM_DATASET_REVISION = "e23b3fe60a0da9037314b88e5ee3a0c054970dad"
AGENTHARM_HARNESS_REPO = "UKGovernmentBEIS/inspect_evals"
AGENTHARM_HARNESS_REVISION = "b935c0e5cfa04710f016f925db75d8e81413e2cf"

AGENTHARM_HELDOUT_SPLIT = "test_private"

#: split name -> {dataset name -> path relative to the dataset root}.  Mirrors the
#: ``split_dict`` in ``inspect_evals.agentharm.utils.load_dataset`` exactly; the
#: loader flag that selects the held-out files is ``split="test_private"``.
AGENTHARM_SPLIT_FILES: Mapping[str, Mapping[str, str]] = {
    "val": {
        "harmful": "benchmark/harmful_behaviors_validation.json",
        "benign": "benchmark/benign_behaviors_validation.json",
        "chat": "benchmark/chat_validation.json",
    },
    "test_public": {
        "harmful": "benchmark/harmful_behaviors_test_public.json",
        "benign": "benchmark/benign_behaviors_test_public.json",
        "chat": "benchmark/chat_public_test.json",
    },
    "test_private": {
        "harmful": "benchmark/harmful_behaviors_test_private.json",
        "benign": "benchmark/benign_behaviors_test_private.json",
        "chat": "benchmark/chat_private_test.json",
    },
}

#: The three files that must be present before any AgentHarm score may be emitted.
AGENTHARM_HELDOUT_FILES: tuple[str, ...] = tuple(
    AGENTHARM_SPLIT_FILES[AGENTHARM_HELDOUT_SPLIT][name]
    for name in ("harmful", "benign", "chat")
)

#: Source files whose content defines the AgentHarm grader.  Hashing all of them
#: gives one verifier identity that changes if any grading path changes.
AGENTHARM_VERIFIER_SOURCES: tuple[str, ...] = (
    "agentharm.py",
    "scorer.py",
    "metric.py",
    "utils.py",
    "prompts.py",
    "agents/agent.py",
    "agents/default_agent.py",
    "agents/refusal_agent.py",
    "benchmark/grading_utils.py",
    "benchmark/harmful_grading_functions.py",
    "benchmark/benign_grading_functions.py",
)

_TASK_ID_HASH_DOMAIN = "pavlov/agentharm/task-id/v1"
_SPLIT_MANIFEST_DOMAIN = "pavlov/agentharm/split-manifest/v1"
_VERIFIER_DOMAIN = "pavlov/agentharm/verifier/v1"


class HeldoutSplitUnavailable(RuntimeError):
    """Raised when a held-out AgentHarm score is requested without the private files."""


def agentharm_task_id_hash(
    behavior_id: str,
    *,
    dataset_name: str,
    split: str,
    dataset_id: str = AGENTHARM_DATASET_ID,
    dataset_revision: str = AGENTHARM_DATASET_REVISION,
) -> str:
    """Bind one behavior ID to the immutable dataset revision and split.

    The hash is domain-separated so a task ID from one split can never collide
    with the same ID in another split or at another dataset revision.
    """

    if not isinstance(behavior_id, str) or not behavior_id.strip():
        raise ValueError("behavior_id must be a non-empty string")
    if split not in AGENTHARM_SPLIT_FILES:
        raise ValueError(f"unknown AgentHarm split {split!r}")
    if dataset_name not in AGENTHARM_SPLIT_FILES[split]:
        raise ValueError(f"unknown AgentHarm dataset {dataset_name!r}")
    if not _is_immutable_revision(dataset_revision):
        raise ValueError("dataset_revision must be an immutable 40-hex or sha256 revision")

    payload = canonical_json(
        {
            "domain": _TASK_ID_HASH_DOMAIN,
            "dataset_id": dataset_id,
            "dataset_revision": str(dataset_revision).strip().lower(),
            "split": split,
            "dataset_name": dataset_name,
            "behavior_id": behavior_id,
        }
    )
    return sha256_text(payload)


def _read_behavior_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    behaviors = data.get("behaviors") if isinstance(data, Mapping) else None
    if not isinstance(behaviors, list):
        raise ValueError(f"{path} does not contain a 'behaviors' list")
    ids: list[str] = []
    for entry in behaviors:
        if not isinstance(entry, Mapping) or "id" not in entry:
            raise ValueError(f"{path} contains a behavior without an 'id'")
        ids.append(str(entry["id"]))
    return ids


def build_agentharm_split_manifest(
    dataset_root: str | Path,
    split: str,
    *,
    dataset_id: str = AGENTHARM_DATASET_ID,
    dataset_revision: str = AGENTHARM_DATASET_REVISION,
) -> dict[str, Any]:
    """Hash every file in one AgentHarm split and bind its behavior IDs.

    Missing files are recorded, never silently skipped: ``complete`` is False and
    ``missing_files`` names them.  This is what makes the held-out gate legible.
    """

    if split not in AGENTHARM_SPLIT_FILES:
        raise ValueError(f"unknown AgentHarm split {split!r}")
    if not _is_immutable_revision(dataset_revision):
        raise ValueError("dataset_revision must be an immutable 40-hex or sha256 revision")

    root = Path(dataset_root)
    files: dict[str, Any] = {}
    missing: list[str] = []
    task_id_hashes: list[str] = []

    for dataset_name in ("harmful", "benign", "chat"):
        relative = AGENTHARM_SPLIT_FILES[split][dataset_name]
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            files[relative] = {
                "dataset_name": dataset_name,
                "present": False,
                "sha256": None,
                "behavior_count": 0,
                "behavior_ids": [],
            }
            continue

        raw = path.read_bytes()
        behavior_ids = _read_behavior_ids(path)
        files[relative] = {
            "dataset_name": dataset_name,
            "present": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "behavior_count": len(behavior_ids),
            "behavior_ids": behavior_ids,
        }
        task_id_hashes.extend(
            agentharm_task_id_hash(
                behavior_id,
                dataset_name=dataset_name,
                split=split,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
            )
            for behavior_id in behavior_ids
        )

    manifest: dict[str, Any] = {
        "domain": _SPLIT_MANIFEST_DOMAIN,
        "suite_id": AGENTHARM_SUITE_ID,
        "dataset_id": dataset_id,
        "dataset_revision": str(dataset_revision).strip().lower(),
        "split": split,
        "heldout": split == AGENTHARM_HELDOUT_SPLIT,
        "loader_split_flag": f'split="{split}"',
        "dataset_root": str(root),
        "files": files,
        "missing_files": missing,
        "complete": not missing,
        "task_count": len(task_id_hashes),
        "task_id_hashes": task_id_hashes,
        "split_task_id_hash": aggregate_task_id_hashes(task_id_hashes)
        if task_id_hashes
        else None,
    }
    # The manifest hash covers everything except the local filesystem path and the
    # hash field itself, so it is reproducible on a different machine.
    hashable = {k: v for k, v in manifest.items() if k != "dataset_root"}
    manifest["split_manifest_hash"] = sha256_text(canonical_json(hashable))
    return manifest


def agentharm_verifier_identity(
    package_root: str | Path,
    *,
    harness_repo: str = AGENTHARM_HARNESS_REPO,
    harness_revision: str = AGENTHARM_HARNESS_REVISION,
) -> dict[str, Any]:
    """Hash the AgentHarm grading sources into one verifier identity."""

    if not _is_immutable_revision(harness_revision):
        raise ValueError("harness_revision must be an immutable 40-hex or sha256 revision")

    root = Path(package_root)
    sources: dict[str, str | None] = {}
    missing: list[str] = []
    for relative in AGENTHARM_VERIFIER_SOURCES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            sources[relative] = None
            continue
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    identity: dict[str, Any] = {
        "domain": _VERIFIER_DOMAIN,
        "suite_id": AGENTHARM_SUITE_ID,
        "verifier_name": _NATIVE_CONTRACT[AGENTHARM_SUITE_ID]["verifier"]["name"],
        "verifier_mode": _NATIVE_CONTRACT[AGENTHARM_SUITE_ID]["verifier"]["mode"],
        "harness_repo": harness_repo,
        "harness_revision": str(harness_revision).strip().lower(),
        "package_root": str(root),
        "sources": sources,
        "missing_sources": missing,
        "complete": not missing,
    }
    hashable = {k: v for k, v in identity.items() if k != "package_root"}
    identity["verifier_hash"] = sha256_text(canonical_json(hashable))
    return identity


def check_heldout_split_available(dataset_root: str | Path) -> dict[str, Any]:
    """Fail-closed presence check for the three gated AgentHarm files."""

    root = Path(dataset_root)
    present: list[str] = []
    missing: list[str] = []
    for relative in AGENTHARM_HELDOUT_FILES:
        (present if (root / relative).is_file() else missing).append(relative)
    return {
        "dataset_root": str(root),
        "required_files": list(AGENTHARM_HELDOUT_FILES),
        "present_files": present,
        "missing_files": missing,
        "available": not missing,
        "loader_split_flag": f'split="{AGENTHARM_HELDOUT_SPLIT}"',
        "gate": "UK AI Safety Institute held-out split access",
    }


def _score_blockers(
    *,
    split_manifest: Mapping[str, Any],
    verifier_identity: Mapping[str, Any],
    heldout_availability: Mapping[str, Any],
    run: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []

    if run.get("is_model_score") is not True:
        blockers.append(
            "run is not a model score (is_model_score is not true); harness validation "
            "may never be promoted to an AgentHarm score"
        )

    split = str(split_manifest.get("split", ""))
    if split != AGENTHARM_HELDOUT_SPLIT:
        blockers.append(
            f"run used split {split!r}; the AgentHarm score is only defined on "
            f"split={AGENTHARM_HELDOUT_SPLIT!r} (the public split is not the held-out split)"
        )

    if heldout_availability.get("available") is not True:
        missing = ", ".join(str(m) for m in heldout_availability.get("missing_files", ()))
        blockers.append(f"held-out split files are absent: {missing or 'unknown'}")

    if not split_manifest.get("complete"):
        missing = ", ".join(str(m) for m in split_manifest.get("missing_files", ()))
        blockers.append(f"split manifest is incomplete, missing: {missing or 'unknown'}")

    if not _is_immutable_revision(split_manifest.get("dataset_revision")):
        blockers.append("split manifest dataset_revision is not an immutable revision")

    if not _is_sha256(split_manifest.get("split_manifest_hash")):
        blockers.append("split_manifest_hash is not a SHA-256 digest")

    task_hashes = split_manifest.get("task_id_hashes")
    if not isinstance(task_hashes, list) or not task_hashes:
        blockers.append("split manifest carries no task_id_hashes")
    elif len(set(task_hashes)) != len(task_hashes):
        blockers.append("split manifest task_id_hashes are not unique")
    elif aggregate_task_id_hashes(task_hashes) != str(
        split_manifest.get("split_task_id_hash", "")
    ).lower():
        blockers.append("split_task_id_hash does not match task_id_hashes")

    if not verifier_identity.get("complete"):
        missing = ", ".join(str(m) for m in verifier_identity.get("missing_sources", ()))
        blockers.append(f"verifier identity is incomplete, missing: {missing or 'unknown'}")
    if not _is_sha256(verifier_identity.get("verifier_hash")):
        blockers.append("verifier_hash is not a SHA-256 digest")
    if not _is_immutable_revision(verifier_identity.get("harness_revision")):
        blockers.append("verifier harness_revision is not an immutable revision")

    # agentharm_eval is a stateful, artifact-producing suite: the contract requires
    # an approved policy-grader artifact before any score is admissible.
    if _NATIVE_CONTRACT[AGENTHARM_SUITE_ID]["environment"]["artifact_required"]:
        artifact = run.get("policy_grader_artifact")
        if not isinstance(artifact, Mapping):
            blockers.append("approved policy-grader artifact receipt is missing")
        else:
            status = str(artifact.get("status", "")).strip().lower()
            if status not in {"approved", "verified", "signoff", "ok"}:
                blockers.append("policy-grader artifact is not approved")
            if _placeholder(
                _first_value(artifact, ("receipt_id", "approval_id", "id", "sha256", "hash"))
            ):
                blockers.append("policy-grader artifact identity is missing")

    if _placeholder(run.get("model_id")) or _placeholder(run.get("model_revision")):
        blockers.append("run model identity is missing")
    elif not _is_immutable_revision(run.get("model_revision")):
        blockers.append("run model_revision is not an immutable revision")

    return blockers


def emit_agentharm_score(
    *,
    split_manifest: Mapping[str, Any],
    verifier_identity: Mapping[str, Any],
    heldout_availability: Mapping[str, Any],
    run: Mapping[str, Any],
    raise_on_block: bool = False,
) -> dict[str, Any]:
    """Fail-closed AgentHarm score emitter.

    Returns a receipt fragment.  ``score`` is ``None`` and ``status`` is
    ``"BLOCKED"`` unless every gate passes: the run is a real model score, it used
    ``split="test_private"``, all three gated files are present, the split manifest
    and verifier identity are complete and hashed, and an approved policy-grader
    artifact receipt exists.  ``raw_score`` from a blocked run is discarded, never
    carried into ``score``.
    """

    blockers = _score_blockers(
        split_manifest=split_manifest,
        verifier_identity=verifier_identity,
        heldout_availability=heldout_availability,
        run=run,
    )

    label = str(run.get("label") or ("agentharm_heldout" if not blockers else "harness_validation"))
    if blockers and raise_on_block:
        raise HeldoutSplitUnavailable("; ".join(blockers))

    receipt: dict[str, Any] = {
        "suite_id": AGENTHARM_SUITE_ID,
        "label": label,
        "is_model_score": run.get("is_model_score") is True and not blockers,
        "score": None,
        "status": "BLOCKED" if blockers else "COMPLETE",
        "blockers": blockers,
        "split": split_manifest.get("split"),
        "heldout_split": AGENTHARM_HELDOUT_SPLIT,
        "loader_split_flag": split_manifest.get("loader_split_flag"),
        "dataset_id": split_manifest.get("dataset_id"),
        "dataset_revision": split_manifest.get("dataset_revision"),
        "split_manifest_hash": split_manifest.get("split_manifest_hash"),
        "split_task_id_hash": split_manifest.get("split_task_id_hash"),
        "task_count": split_manifest.get("task_count"),
        "verifier_hash": verifier_identity.get("verifier_hash"),
        "harness_repo": verifier_identity.get("harness_repo"),
        "harness_revision": verifier_identity.get("harness_revision"),
        "model_id": run.get("model_id"),
        "model_revision": run.get("model_revision"),
    }
    if not blockers:
        receipt["score"] = run.get("raw_score")
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "ADAPTER_ID",
    "MODEL_ID",
    "MODEL_REVISION",
    "SUPPORTED_SUITE_IDS",
    "AGENTHARM_SUITE_ID",
    "AGENTHARM_DATASET_ID",
    "AGENTHARM_DATASET_REVISION",
    "AGENTHARM_HARNESS_REPO",
    "AGENTHARM_HARNESS_REVISION",
    "AGENTHARM_HELDOUT_SPLIT",
    "AGENTHARM_SPLIT_FILES",
    "AGENTHARM_HELDOUT_FILES",
    "AGENTHARM_VERIFIER_SOURCES",
    "HeldoutSplitUnavailable",
    "canonical_json",
    "sha256_text",
    "aggregate_task_id_hashes",
    "agentharm_task_id_hash",
    "build_agentharm_split_manifest",
    "agentharm_verifier_identity",
    "check_heldout_split_available",
    "emit_agentharm_score",
    "build_boundary_receipts",
    "update_bundle_signature",
    "generate_boundary_bundle",
    "validate_boundary",
    "validate_adapter_bundle",
    "evaluate_bundle",
]
