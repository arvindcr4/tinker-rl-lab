#!/usr/bin/env python3
"""Offline metadata adapter for exact training suite T7: agentdojo_train.

The module is intentionally fail-closed.  It emits a deterministic boundary
bundle only from the local contract and validates all required receipt material
for a train run, including immutable task hashes and mandatory W&B/Tinker/HF
material.  No network, paid calls, or credentials are used.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

try:
    from .pavlovs_domain_contract import load_contract
except ImportError:  # pragma: no cover - direct execution fallback
    from pavlovs_domain_contract import load_contract

SCHEMA_VERSION = "pavlov-agentdojo-train-adapter-v1"
ADAPTER_ID = "pavlov-t7-agentdojo-train-adapter-v1"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

SUPPORTED_SUITE_IDS = ("agentdojo_train",)

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

_LICENSE_TOKENS = {"spdx:", "apache", "mit", "cc-", "bsd", "lgpl", "gpl"}
_VALID_DECONTAMINATION = {"verified", "complete", "completed", "passed", "clean"}
_VALID_WANDB_STATUS = {"approved", "authorized", "ready", "running", "finished", "online"}
_VALID_TINKER_COST_STATUS = {
    "authorized",
    "approved",
    "within_cap",
    "observed",
    "complete",
    "finished",
}
_VALID_EVIDENCE_STATUS = {"prospective", "observed", "admissible", "rejected", "pending"}

# T7-specific disjointness is enforced against known primary-suite IDs and
# prompt sources present in the contract plus legacy benchmark aliases.
_BANNED_SUBSTITUTION_MARKERS = (
    "agentharm",
    "frontiermath",
    "xlam",
    "xlam-function-calling-60k",
    "salesforce/xlam-function-calling-60k",
    "gsm8k",
    "math-500",
    "math500",
    "openhf/hf",
    "openhf",
    "frontiermath_math_500",
    "openr1_math",
    "openr1-math",
    "inspect_evals",
    "agentharm_eval",
    "frontiermath_eval",
    "swe_bench_pro_eval",
    "frontier_swe_eval",
    "sdab_eval",
    "banker_toolbench_eval",
    "apex_agents_eval",
    "webbench_eval",
    "binaryaudit_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "verilog_eval",
    "appbench_eval",
    "openreward_games_eval",
)

_NATIVE_CONTRACT = {
    "agentdojo_train": {
        "environment": {
            "name": "agentdojo_train_environment",
            "mode": "agentdojo-stateful-episodes-v1",
            "artifact_required": True,
        },
        "verifier": {
            "name": "agentdojo_train_verifier",
            "mode": "agentdojo-state-verifier-v1",
        },
        "artifact": {
            "mode": "stateful",
        },
    },
}


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding used for signature checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a deterministic text digest."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def aggregate_task_id_hashes(task_id_hashes: Sequence[str]) -> str:
    """Aggregate ordered task IDs into a deterministic split manifest hash."""

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


def _expected_source_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if parsed.netloc == "github.com" and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parsed.netloc and parts:
        return f"{parsed.netloc}/" + "/".join(parts)
    return parsed.netloc


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


def _expected_primary_eval_markers(contract: Mapping[str, Any]) -> tuple[str, ...]:
    suites = contract.get("suite_registry", {})
    markers = []
    if isinstance(suites, Mapping):
        for suite_id, suite in suites.items():
            if not isinstance(suite, Mapping):
                continue
            if suite.get("role") == "primary_eval":
                markers.append(str(suite_id).lower())
                markers.append(str(suite.get("url", "")).lower())
                url = str(suite.get("url", "")).lower()
                if url.startswith("https://github.com/"):
                    markers.extend(part for part in url.split("/")[3:] if part)
    return tuple(sorted(set(markers)))


def _load_contract_suite(contract: Mapping[str, Any], suite_id: str) -> Mapping[str, Any]:
    suites = contract.get("suite_registry", {})
    if not isinstance(suites, Mapping):
        raise ValueError("contract suite_registry must be an object")
    suite = suites.get(suite_id)
    if not isinstance(suite, Mapping):
        raise ValueError(f"contract is missing suite {suite_id}")
    return suite


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
    }


def generate_boundary_bundle() -> dict[str, Any]:
    """Generate a deterministic metadata-only boundary bundle for agentdojo_train."""

    contract = load_contract()
    primary_markers = tuple(_expected_primary_eval_markers(contract))
    boundaries: list[dict[str, Any]] = []
    for suite_id in SUPPORTED_SUITE_IDS:
        suite = _load_contract_suite(contract, suite_id)
        split = str(suite.get("split", ""))
        source_url = str(suite.get("url", ""))
        boundaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "adapter_id": ADAPTER_ID,
                "suite_id": suite_id,
                "suite_role": "train",
                "primary_eval": False,
                "heldout": False,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "authoritative_source": {
                    "contract_schema_version": str(contract.get("schema_version")),
                    "source_url": source_url,
                    "source_id": _expected_source_id(source_url),
                    "source_kind": "repo",
                    "split": split,
                    "disjoint_from_primary_markers": primary_markers,
                },
                "stateful": bool(suite.get("stateful")),
                "artifact_or_side_effect": bool(suite.get("artifact_or_side_effect")),
                "component_only": False,
                "native_contract": _native_contract_for_suite(suite_id),
                "domains": sorted({str(v) for v in suite.get("domains", ())}),
                "receipts": build_boundary_receipts(suite_id),
            }
        )
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

    split = str(suite.get("split", ""))
    if authoritative.get("split") != split:
        errors.append(f"{suite_id}: authoritative_source.split must equal contract split")

    source_id = str(authoritative.get("source_id", ""))
    expected_source_id = _expected_source_id(source_url)
    if _placeholder(source_id):
        errors.append(f"{suite_id}: authoritative_source.source_id is required")
    elif source_id != expected_source_id:
        errors.append(
            f"{suite_id}: authoritative_source.source_id {source_id!r} does not match expected {expected_source_id!r}"
        )

    source_kind = str(authoritative.get("source_kind", "")).strip().lower()
    if source_kind != "repo":
        errors.append(f"{suite_id}: authoritative_source.source_kind must be repo")

    if not isinstance(authoritative.get("disjoint_from_primary_markers"), Sequence):
        errors.append(f"{suite_id}: authoritative_source.disjoint_from_primary_markers must be a sequence")

    source_url_value = authoritative.get("source_url")
    if not isinstance(source_url_value, str) or not _URL_RE.fullmatch(source_url_value.strip()):
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


def _validate_dataset_substitution(
    suite_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    dataset_id = receipt.get("dataset_id")
    if _placeholder(dataset_id):
        errors.append(f"{suite_id}: dataset_id is required")
        return errors

    lowered = str(dataset_id).lower()
    markers = set(_BANNED_SUBSTITUTION_MARKERS) | set(_expected_primary_eval_markers(contract))
    for marker in sorted(markers):
        if marker and marker in lowered:
            errors.append(
                f"{suite_id}: dataset_id appears to substitute primary-eval prompts or components ({marker})"
            )
            return errors

    return errors


def _validate_receipt_fields(suite_id: str, receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    for key in REQUIRED_RECEIPT_KEYS:
        if key not in receipt:
            errors.append(f"{suite_id}: receipt missing {key}")
    if errors:
        return errors

    errors.extend(_validate_dataset_substitution(suite_id, receipt, contract))

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
        if any(token in text for token in _LICENSE_TOKENS) or _is_sha256(license_value):
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
        if _placeholder(receipt.get("split_task_id_hash")):
            errors.append(f"{suite_id}: split_task_id_hash is required")
        elif not _is_sha256(receipt.get("split_task_id_hash")):
            errors.append(f"{suite_id}: split_task_id_hash must be a SHA-256 digest")
        elif normalized and aggregate_task_id_hashes(normalized) != str(receipt.get("split_task_id_hash")).lower():
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
        errors.append(f"{suite_id}: receipt model_revision must be pinned {MODEL_REVISION}")

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
        if not isinstance(max_usd, (int, float)) or max_usd <= 0:
            errors.append(f"{suite_id}: budget_receipt.maximum_usd must be a positive number")
        if _placeholder(_first_value(budget, ("receipt_id", "authorization_id", "id", "sha256", "hash"))):
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
        if not isinstance(wandb.get("run_url"), str) or not _URL_RE.fullmatch(wandb.get("run_url").strip()):
            errors.append(f"{suite_id}: wandb_run_identity.run_url must be https URL")
        if str(wandb.get("mode", "online")).lower() not in _VALID_WANDB_STATUS:
            errors.append(f"{suite_id}: wandb mode must be online/ready")

    tinker = receipt.get("tinker_run_identity")
    if not isinstance(tinker, Mapping):
        errors.append(f"{suite_id}: tinker_run_identity must be an object")
    else:
        if _placeholder(tinker.get("run_id")):
            errors.append(f"{suite_id}: tinker_run_identity.run_id is missing")
        if str(tinker.get("cost_status", "")).lower() not in _VALID_TINKER_COST_STATUS:
            errors.append(f"{suite_id}: tinker_run_identity.cost_status is invalid")
        if not _placeholder(tinker.get("run_url")) and not _URL_RE.fullmatch(str(tinker.get("run_url", "")).strip()):
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
                if not isinstance(checkpoint.get(field), str) or not _URL_RE.fullmatch(
                    checkpoint.get(field).strip()
                ):
                    errors.append(
                        f"{suite_id}: hf_checkpoints[{index}].{field} must be public HTTPS URL"
                    )

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
                errors.append(f"{suite_id}: hf_checkpoints[{index}] duplicates existing checkpoint entry")
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


def validate_boundary(boundary: Mapping[str, Any]) -> list[str]:
    """Validate one training suite boundary with contract-anchored checks."""

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
    if boundary.get("suite_role") != "train":
        return [f"{suite_id}: suite_role must be train"]
    if boundary.get("primary_eval") is not False:
        return [f"{suite_id}: primary_eval must be false"]
    if boundary.get("heldout") is not False:
        return [f"{suite_id}: heldout must be false"]
    if boundary.get("component_only") is not False:
        return [f"{suite_id}: component_only must be false"]

    if str(boundary.get("model_id")) != MODEL_ID:
        return [f"{suite_id}: model_id must be {MODEL_ID}"]
    if boundary.get("model_revision") != MODEL_REVISION:
        return [f"{suite_id}: model_revision must be pinned {MODEL_REVISION}"]

    contract = load_contract()
    suite = _load_contract_suite(contract, suite_id)
    if suite.get("role") != "train":
        return [f"{suite_id}: contract role must be train"]

    if tuple(boundary.get("domains", ())) != tuple(sorted({str(v) for v in suite.get("domains", ())})):
        return [f"{suite_id}: domains must match contract"]

    if bool(boundary.get("stateful")) != bool(suite.get("stateful")):
        return [f"{suite_id}: stateful flag must match contract"]

    expected_artifact = bool(suite.get("artifact_or_side_effect"))
    if bool(boundary.get("artifact_or_side_effect")) != expected_artifact:
        return [f"{suite_id}: artifact_or_side_effect flag must match contract"]

    contract_markers = _expected_primary_eval_markers(contract)
    authoritative_markers = boundary.get("authoritative_source", {}).get("disjoint_from_primary_markers")
    if _as_sequence(authoritative_markers) is None:
        return [f"{suite_id}: authoritative_source.disjoint_from_primary_markers is required"]
    if tuple(authoritative_markers) != contract_markers:
        return [f"{suite_id}: authoritative_source.disjoint_from_primary_markers must match contract"]

    errors = _validate_authoritative_source(suite_id, contract, boundary)
    errors.extend(_validate_native_contract(suite_id, boundary.get("native_contract", {})))
    receipt = boundary.get("receipts")
    if not isinstance(receipt, Mapping):
        errors.append(f"{suite_id}: receipts must be an object")
        return errors
    errors.extend(_validate_receipt_fields(suite_id, receipt, contract))
    return errors


def validate_adapter_bundle(bundle: Mapping[str, Any]) -> list[str]:
    """Validate the full adapter bundle and return fail-closed errors."""

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
        return ["adapter bundle must contain one train suite boundary"]

    errors: list[str] = []
    if bundle.get("suite_count") != len(SUPPORTED_SUITE_IDS):
        errors.append("suite_count must be 1")
    suite_ids: list[str] = []
    for boundary in boundaries:
        if not isinstance(boundary, Mapping):
            errors.append("each boundary must be an object")
            continue
        suite_id = str(boundary.get("suite_id"))
        suite_ids.append(suite_id)
        errors.extend(validate_boundary(boundary))

    if sorted(set(suite_ids)) != sorted(SUPPORTED_SUITE_IDS):
        errors.append("adapter bundle must only contain agentdojo_train")
    if len(set(suite_ids)) != len(suite_ids):
        errors.append("adapter bundle boundaries contain duplicate suite IDs")

    expected_signature = _expected_bundle_signature(bundle)
    if bundle.get("bundle_signature") != expected_signature:
        errors.append("bundle_signature is invalid")

    return errors


def evaluate_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return readiness/claim metadata for the training bundle."""

    errors = validate_adapter_bundle(bundle)
    suite_readiness: dict[str, dict[str, Any]] = {}

    boundaries = bundle.get("boundaries") if isinstance(bundle, Mapping) else None
    if isinstance(boundaries, list):
        for boundary in boundaries:
            if not isinstance(boundary, Mapping):
                continue
            suite_id = str(boundary.get("suite_id"))
            suite_readiness[suite_id] = {
                "suite_role": boundary.get("suite_role"),
                "primary_eval_claim": bool(boundary.get("primary_eval")),
                "heldout": bool(boundary.get("heldout")),
                "train_launch_allowed": bool(len(errors) == 0),
            }

    return {
        "status": "READY" if not errors else "BLOCKED",
        "errors": errors,
        "suite_readiness": suite_readiness,
    }


__all__ = [
    "SCHEMA_VERSION",
    "ADAPTER_ID",
    "MODEL_ID",
    "MODEL_REVISION",
    "SUPPORTED_SUITE_IDS",
    "canonical_json",
    "sha256_text",
    "aggregate_task_id_hashes",
    "build_boundary_receipts",
    "update_bundle_signature",
    "generate_boundary_bundle",
    "validate_boundary",
    "validate_adapter_bundle",
    "evaluate_bundle",
]
