#!/usr/bin/env python3
"""Offline result-receipt boundary for exact ``verilog_eval`` primary runs.

The validator is fail-closed and metadata-first.  It checks:

* immutable dataset and split contracts
* explicit verifier/environment identities
* deterministic per-example IDs, categories, and verdicts
* W&B, Tinker, and Hugging Face checkpoint receipts
* explicit non-portfolio and non-hidden-then-shared heldout semantics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pavlov-verilog-eval-receipt-v1"
SUITE_ID = "verilog_eval"
SUITE_ROLE = "primary_eval"
_EXPECTED_CATEGORY = "code"
_EXPECTED_VERDICTS = {"pass", "fail", "error"}
_NATIVE_SCOPE_KEYS = ("is_portfolio", "is_heldout", "is_held_out")

HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
URL_RE = re.compile(r"^https://[^\s]+$")
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
    "to_be_pinned",
    "to_be_pinned_before_paid_runs",
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
}

_ALLOWED_SCOPE_STATUS = {"verified", "clean", "passed", "admissible", "complete"}
_ALLOWED_TINKER_STATUS = {
    "authorized",
    "approved",
    "observed",
    "within_cap",
    "complete",
    "zero_cost",
}
_ALLOWED_COST_STATUS = {
    "authorized",
    "approved",
    "within_cap",
    "complete",
    "observed",
    "zero_cost",
}
_REQUIRED_WANDB_KEYS = ("entity", "project", "group", "run_id", "run_url")
_REQUIRED_HF_STAGES = ("initial", "periodic", "final")

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


class VerilogEvalReceiptError(ValueError):
    """Raised for malformed verilog_eval receipt input."""


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
        raise VerilogEvalReceiptError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_status(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerilogEvalReceiptError(f"{field} must be explicit status text")
    return value.strip()


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise VerilogEvalReceiptError(f"{field} must be explicit boolean")
    return value


def _require_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX40_RE.fullmatch(text):
        raise VerilogEvalReceiptError(f"{field} must be 40-char hex")
    return text


def _require_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX64_RE.fullmatch(text):
        raise VerilogEvalReceiptError(f"{field} must be 64-char hex")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not SHA256_RE.fullmatch(text):
        raise VerilogEvalReceiptError(f"{field} must be a sha256 digest")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _require_url(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not URL_RE.fullmatch(text):
        raise VerilogEvalReceiptError(f"{field} must be https URL")
    return text


def _contains_banned_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_banned_markers(item) for item in value.values() if item is not None)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str, bytearray)):
        return any(_contains_banned_markers(item) for item in value)
    return False


def _load_authority(contract_path: str | Path | None = None) -> Mapping[str, Any]:
    base = Path(__file__).resolve().parent
    path = Path(contract_path) if contract_path is not None else base / "pavlovs_domain_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerilogEvalReceiptError(f"cannot read authority contract: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerilogEvalReceiptError(
            f"authority contract is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise VerilogEvalReceiptError("authority contract must be JSON object")
    registry = payload.get("suite_registry")
    if not isinstance(registry, Mapping):
        raise VerilogEvalReceiptError("authority contract missing suite_registry")
    suite = registry.get(SUITE_ID)
    if not isinstance(suite, Mapping):
        raise VerilogEvalReceiptError("authority contract missing verilog_eval suite")
    return suite


def _validate_source(value: Any, authority: Mapping[str, Any]) -> str:
    source = _require_url(value, "source")
    authoritative = _require_url(authority.get("url"), "contract suite url")
    if source.rstrip("/").lower() != authoritative.rstrip("/").lower():
        raise VerilogEvalReceiptError("source must match authoritative verilog_eval source")
    return source


def _validate_dataset(value: Any, authority: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("dataset must be an object")
    revision = _require_hex40(value.get("revision"), "dataset.revision")
    license_text = _require_text(value.get("license"), "dataset.license")
    source = _require_url(value.get("source"), "dataset.source")
    authoritative = _require_url(authority.get("url"), "contract suite url")
    if source.rstrip("/").lower() != authoritative.rstrip("/").lower():
        raise VerilogEvalReceiptError("dataset.source must match authoritative source")
    if _contains_banned_markers(source) or _contains_banned_markers(value.get("name")):
        raise VerilogEvalReceiptError("dataset references blocked source evidence")
    return revision, license_text, source


def _validate_category(value: Any) -> str:
    category = _require_text(value, "category")
    if category != _EXPECTED_CATEGORY:
        raise VerilogEvalReceiptError(
            f"category must be {_EXPECTED_CATEGORY!r}, found {category!r}"
        )
    return category


def _validate_role(value: Any) -> str:
    role = _require_text(value, "role")
    if role != SUITE_ROLE:
        raise VerilogEvalReceiptError(f"role must be {SUITE_ROLE!r}, found {role!r}")
    return role


def _validate_verifier(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("verifier must be an object")
    identity = _require_text(value.get("identity"), "verifier.identity")
    if _contains_banned_markers(identity):
        raise VerilogEvalReceiptError("verifier.identity contains blocked substitution markers")
    verifier_hash = _require_hex64(
        value.get("hash") if value.get("hash") is not None else value.get("digest"),
        "verifier.hash",
    )
    return {"identity": identity, "hash": verifier_hash}


def _validate_environment(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("environment must be an object")
    container = _require_text(value.get("container"), "environment.container")
    image = _require_text(value.get("image"), "environment.image")
    container_digest = _require_sha256(
        value.get("container_digest"), "environment.container_digest"
    )
    runtime_digest = _require_sha256(
        value.get("runtime_digest"), "environment.runtime_digest"
    )
    if _contains_banned_markers(container) or _contains_banned_markers(image):
        raise VerilogEvalReceiptError(
            "environment references blocked substitution source"
        )
    return {
        "container": container,
        "image": image,
        "container_digest": container_digest,
        "runtime_digest": runtime_digest,
    }


def _validate_task_ids(value: Any, field: str = "task_id_hashes") -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerilogEvalReceiptError(f"{field} must be a non-empty list")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_hash = _require_hex64(item, f"{field}[{index}]")
        if item_hash in seen:
            raise VerilogEvalReceiptError(f"{field} contains duplicate IDs")
        seen.add(item_hash)
        normalized.append(item_hash)

    if not normalized:
        raise VerilogEvalReceiptError(f"{field} cannot be empty")
    if normalized != sorted(normalized):
        raise VerilogEvalReceiptError(f"{field} must be deterministically sorted")
    return normalized


def _validate_split(value: Any, task_ids: list[str]) -> tuple[list[str], str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("split must be an object")
    split_ids = _validate_task_ids(
        value.get("primary_eval"), field="split.primary_eval"
    )
    if split_ids != task_ids:
        raise VerilogEvalReceiptError("split.primary_eval must match task_id_hashes")
    declared = _require_sha256(
        value.get("hash") if value.get("hash") is not None else value.get("sha256"),
        "split.primary_eval.hash",
    )
    aggregate = _sha256("\n".join(split_ids))
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalReceiptError("split.primary_eval aggregate hash does not match tasks")
    return split_ids, aggregate, expected


def _validate_split_hashes(value: Mapping[str, Any], aggregate: str) -> dict[str, str]:
    split_hashes = value.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise VerilogEvalReceiptError("split_hashes must be an object")
    declared = _require_sha256(split_hashes.get("primary_eval"), "split_hashes.primary_eval")
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalReceiptError("split_hashes.primary_eval mismatch")
    return {"primary_eval": declared}


def _validate_split_manifest_hash(
    manifest: Mapping[str, Any], aggregate: str
) -> tuple[str, str]:
    payload = canonical_json({"primary_eval": aggregate})
    observed = f"sha256:{_sha256(payload)}"
    declared = _require_sha256(manifest.get("split_manifest_hash"), "split_manifest_hash")
    if declared != observed:
        raise VerilogEvalReceiptError("split_manifest_hash does not match observed split")
    receipt_ref = manifest.get("split_manifest_receipt_ref")
    if receipt_ref is None:
        raise VerilogEvalReceiptError("split_manifest_receipt_ref is required")
    if _is_placeholder(receipt_ref):
        raise VerilogEvalReceiptError("split_manifest_receipt_ref is placeholder")
    if not SHA256_RE.fullmatch(str(receipt_ref).lower()):
        raise VerilogEvalReceiptError("split_manifest_receipt_ref must be sha256")
    if not receipt_ref.startswith("sha256:"):
        receipt_ref = "sha256:" + str(receipt_ref).lower()
    return observed, receipt_ref


def _validate_scope(value: Mapping[str, Any], *, task_aggregate: str) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("scope must be an object")
    is_portfolio = _require_bool(value.get("is_portfolio"), "scope.is_portfolio")
    if is_portfolio:
        raise VerilogEvalReceiptError("scope.is_portfolio must be false for this receipt")

    if "is_held_out" in value:
        is_heldout = _require_bool(value.get("is_held_out"), "scope.is_held_out")
    elif "is_heldout" in value:
        is_heldout = _require_bool(value.get("is_heldout"), "scope.is_heldout")
    else:
        raise VerilogEvalReceiptError("scope must include is_held_out or is_heldout")
    return {"is_portfolio": is_portfolio, "is_held_out": is_heldout}


def _validate_held_out(manifest: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    held_out = _require_bool(manifest.get("held_out"), "held_out")
    held_out_receipt_ref = manifest.get("held_out_receipt_ref")
    scope_is_held_out = bool(scope.get("is_held_out"))
    if held_out != scope_is_held_out:
        raise VerilogEvalReceiptError(
            "scope.is_held_out and held_out must agree"
        )
    if held_out and _is_placeholder(held_out_receipt_ref):
        raise VerilogEvalReceiptError("held_out_receipt_ref is required when held_out is true")
    if not held_out and held_out_receipt_ref is not None and not _is_placeholder(
        held_out_receipt_ref
    ):
        raise VerilogEvalReceiptError("held_out_receipt_ref is forbidden when held_out is false")
    if held_out:
        ref = _require_sha256(
            held_out_receipt_ref, "held_out_receipt_ref"
        )
        if not ref:
            raise VerilogEvalReceiptError("held_out_receipt_ref must be a concrete sha256")
    return held_out


def _validate_per_example(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerilogEvalReceiptError("per_example must be a list")
    if not value:
        raise VerilogEvalReceiptError("per_example cannot be empty")

    normalized: list[tuple[str, str, str]] = []
    ids: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise VerilogEvalReceiptError(f"per_example[{index}] must be an object")
        example_id = _require_hex64(item.get("id"), f"per_example[{index}].id")
        category = _require_text(item.get("category"), f"per_example[{index}].category")
        if category != _EXPECTED_CATEGORY:
            raise VerilogEvalReceiptError(
                f"per_example[{index}].category must be {_EXPECTED_CATEGORY!r}"
            )
        raw_verdict = item.get("verdict")
        if isinstance(raw_verdict, bool):
            verdict = "pass" if raw_verdict else "fail"
        elif isinstance(raw_verdict, str):
            verdict = raw_verdict.strip().lower()
        else:
            raise VerilogEvalReceiptError(
                f"per_example[{index}].verdict must be pass|fail|error"
            )
        if verdict not in _EXPECTED_VERDICTS:
            raise VerilogEvalReceiptError(
                f"per_example[{index}].verdict must be pass|fail|error"
            )
        ids.append(example_id)
        normalized.append((example_id, category, verdict))

    if len(set(ids)) != len(ids):
        raise VerilogEvalReceiptError("per_example contains duplicate ids")
    if ids != sorted(ids):
        raise VerilogEvalReceiptError("per_example must be sorted deterministically")

    return [
        {"id": example_id, "category": category, "verdict": verdict}
        for example_id, category, verdict in normalized
    ]


def _validate_wandb(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalReceiptError("wandb_run_identity must be an object")
    if raw.get("online") is not True:
        raise VerilogEvalReceiptError("wandb_run_identity.online must be true")
    values = {
        key: _require_text(raw.get(key), f"wandb_run_identity.{key}")
        for key in _REQUIRED_WANDB_KEYS
    }
    if not HTTPS_URL.fullmatch(values["run_url"]):
        raise VerilogEvalReceiptError("wandb_run_identity.run_url must be HTTPS URL")
    if "wandb.ai" not in values["run_url"]:
        raise VerilogEvalReceiptError("wandb_run_identity.run_url must be a W&B URL")
    return values


def _validate_tinker(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalReceiptError("tinker_run_identity must be an object")
    run_id = _require_text(raw.get("run_id"), "tinker_run_identity.run_id")
    cost_status = _require_status(raw.get("cost_status"), "tinker_run_identity.cost_status").lower()
    if cost_status not in _ALLOWED_TINKER_STATUS:
        raise VerilogEvalReceiptError("tinker_run_identity.cost_status is invalid")
    return {"run_id": run_id, "cost_status": cost_status}


def _validate_hf_checkpoints(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise VerilogEvalReceiptError("hf_checkpoints must be a list")
    if not raw:
        raise VerilogEvalReceiptError("hf_checkpoints cannot be empty")

    normalized: list[dict[str, Any]] = []
    seen_stage: set[str] = set()
    seen_receipts: set[tuple[str, str]] = set()
    seen_payload: set[tuple[str, str, str]] = set()

    for index, checkpoint in enumerate(raw):
        if not isinstance(checkpoint, Mapping):
            raise VerilogEvalReceiptError(f"hf_checkpoints[{index}] must be an object")
        repo = _require_url(
            checkpoint.get("repo_url"), f"hf_checkpoints[{index}].repo_url"
        )
        revision = _require_hex40(
            checkpoint.get("revision"), f"hf_checkpoints[{index}].revision"
        )
        url = _require_url(checkpoint.get("url"), f"hf_checkpoints[{index}].url")
        stage = _require_text(checkpoint.get("stage"), f"hf_checkpoints[{index}].stage").lower()
        if stage not in _REQUIRED_HF_STAGES:
            raise VerilogEvalReceiptError(
                f"hf_checkpoints[{index}].stage must be one of {', '.join(_REQUIRED_HF_STAGES)}"
            )
        if checkpoint.get("safe_public_artifact") is not True:
            raise VerilogEvalReceiptError(
                f"hf_checkpoints[{index}].safe_public_artifact must be true"
            )
        visibility = _require_text(
            checkpoint.get("visibility"), f"hf_checkpoints[{index}].visibility"
        ).lower()
        if visibility not in {"public", "private"}:
            raise VerilogEvalReceiptError(
                f"hf_checkpoints[{index}].visibility must be public or private"
            )
        if _contains_banned_markers(repo):
            raise VerilogEvalReceiptError(
                f"hf_checkpoints[{index}] contains blocked checkpoint source"
            )
        receipt_key = (repo, revision)
        payload_key = (repo, revision, url)
        if receipt_key in seen_receipts:
            raise VerilogEvalReceiptError(
                f"hf_checkpoints duplicate repo/revision at index {index}"
            )
        if payload_key in seen_payload:
            raise VerilogEvalReceiptError(
                f"hf_checkpoints duplicate checkpoint artifact at index {index}"
            )
        seen_stage.add(stage)
        seen_receipts.add(receipt_key)
        seen_payload.add(payload_key)
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

    if not set(_REQUIRED_HF_STAGES).issubset(seen_stage):
        missing = sorted(set(_REQUIRED_HF_STAGES) - seen_stage)
        raise VerilogEvalReceiptError(
            "hf_checkpoints missing required stage(s): " + ", ".join(missing)
        )
    return sorted(normalized, key=lambda row: row["stage"])


def _validate_costs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalReceiptError("costs must be an object")
    status = _require_status(raw.get("status"), "costs.status").lower()
    if status not in _ALLOWED_COST_STATUS:
        raise VerilogEvalReceiptError("costs.status is invalid")
    total_usd = raw.get("total_usd")
    if not isinstance(total_usd, (int, float)) or isinstance(total_usd, bool):
        raise VerilogEvalReceiptError("costs.total_usd must be a number")
    if total_usd < 0:
        raise VerilogEvalReceiptError("costs.total_usd cannot be negative")
    if raw.get("paid_work") is True:
        raise VerilogEvalReceiptError("costs.paid_work is not allowed")
    return {"status": status, "total_usd": float(total_usd)}


def _validate_decontamination(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        receipt_id = _require_hex40(value, "decontamination")
        return {
            "status": "verified",
            "receipt_id": receipt_id,
            "visibility": "private",
            "safe_public_artifact": True,
        }
    if not isinstance(value, Mapping):
        raise VerilogEvalReceiptError("decontamination must be an object or receipt id")
    status = _require_text(value.get("status"), "decontamination.status").lower()
    if status not in _ALLOWED_SCOPE_STATUS:
        raise VerilogEvalReceiptError("decontamination.status is invalid")
    receipt_id = _require_hex40(value.get("receipt_id"), "decontamination.receipt_id")
    visibility = _require_text(value.get("visibility"), "decontamination.visibility").lower()
    if visibility == "public" and value.get("safe_public_artifact") is not True:
        raise VerilogEvalReceiptError(
            "public decontamination evidence must set safe_public_artifact=True"
        )
    return {
        "status": status,
        "receipt_id": receipt_id,
        "visibility": visibility,
        "safe_public_artifact": bool(value.get("safe_public_artifact")),
    }


def _validate_no_network_or_credentials(manifest: Mapping[str, Any], blockers: list[str]) -> None:
    if manifest.get("requires_network") is True:
        blockers.append("dataset download/network is disallowed for receipt validation")
    if manifest.get("paid_launch_allowed") is True:
        blockers.append("paid_launch_allowed cannot be true in offline receipt")
    for key in _CREDENTIAL_KEYS:
        value = manifest.get(key)
        if not _is_placeholder(value):
            blockers.append(f"{key} is not allowed in offline receipt")


def build_receipt_record(
    manifest: Mapping[str, Any],
    *,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic verilog_eval receipt decision payload."""

    try:
        if not isinstance(manifest, Mapping):
            raise VerilogEvalReceiptError("manifest must be a JSON object")

        authority = _load_authority(contract_path)
        if str(authority.get("role", "")).strip().lower() != SUITE_ROLE:
            raise VerilogEvalReceiptError("contract role mismatch for verilog_eval")

        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            raise VerilogEvalReceiptError(f"suite_id must be {SUITE_ID!r}")
        source = _validate_source(manifest.get("source"), authority)
        category = _validate_category(manifest.get("category"))
        role = _validate_role(manifest.get("role"))
        dataset_revision, dataset_license, dataset_source = _validate_dataset(
            manifest.get("dataset"), authority
        )
        adapter_manifest_digest = _require_sha256(
            manifest.get("adapter_manifest_digest"), "adapter_manifest_digest"
        )
        verifier = _validate_verifier(manifest.get("verifier"))
        environment = _validate_environment(manifest.get("environment"))

        task_ids = _validate_task_ids(manifest.get("task_id_hashes"))
        split_ids, split_aggregate, split_aggregate_digest = _validate_split(
            manifest.get("split"), task_ids
        )
        split_hashes = _validate_split_hashes(manifest, split_aggregate)
        split_manifest_hash, split_manifest_receipt_ref = _validate_split_manifest_hash(
            manifest, split_aggregate
        )
        scope = _validate_scope(manifest.get("scope"), task_aggregate=split_aggregate)
        held_out = _validate_held_out(manifest, scope)
        held_out_receipt_proven = bool(held_out)

        per_example = _validate_per_example(manifest.get("per_example"))
        if len(per_example) != len(task_ids) or [row["id"] for row in per_example] != task_ids:
            raise VerilogEvalReceiptError(
                "per_example ids must cover exactly the task_id_hashes in order"
            )
        decontamination = _validate_decontamination(manifest.get("decontamination"))

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
            "dataset": manifest.get("dataset"),
            "verifier": verifier,
            "environment": environment,
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": hf_checkpoints,
            "scope": scope,
        }
        if _contains_banned_markers(evidence):
            raise VerilogEvalReceiptError("receipt contains blocked substitution markers")

        blockers: list[str] = []
        _validate_no_network_or_credentials(manifest, blockers)
        blockers = sorted(set(blockers))

        per_example_payload = {
            "suite_id": SUITE_ID,
            "examples": per_example,
        }
        receipt_identity_payload = {
            "dataset_revision": dataset_revision,
            "adapter_manifest_digest": adapter_manifest_digest,
            "verifier": verifier,
            "environment": environment,
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
            "authoritative_source": source,
            "contract_authority": {
                "source": authority.get("url"),
                "role": authority.get("role"),
                "domains": list(authority.get("domains", [])),
            },
            "category": category,
            "role": role,
            "dataset_revision": dataset_revision,
            "dataset_license": dataset_license,
            "dataset_source": dataset_source,
            "adapter_manifest_digest": adapter_manifest_digest,
            "verifier": verifier,
            "environment": environment,
            "split": {
                "primary_eval": {
                    "count": len(split_ids),
                    "task_id_hashes": split_ids,
                    "aggregate_sha256": split_aggregate,
                    "aggregate_sha256_digest": split_aggregate_digest,
                }
            },
            "split_hashes": split_hashes,
            "split_manifest_hash": split_manifest_hash,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": split_manifest_receipt_ref,
            "scope": scope,
            "held_out": held_out,
            "held_out_receipt_ref": manifest.get("held_out_receipt_ref"),
            "held_out_receipt_proven": held_out_receipt_proven,
            "decontamination": decontamination,
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": hf_checkpoints,
            "costs": costs,
            "per_example_count": len(per_example),
            "per_example_category": _EXPECTED_CATEGORY,
            "per_example_verdicts": sorted({item["verdict"] for item in per_example}),
            "per_example_digest": _sha256(canonical_json(per_example_payload)),
            "receipt_identity_digest": _sha256(canonical_json(receipt_identity_payload)),
            "task_id_digest": _sha256(canonical_json(task_ids)),
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except VerilogEvalReceiptError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.get("suite_id", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "exact_suite": manifest.get("suite_id") == SUITE_ID
            if isinstance(manifest, Mapping)
            else False,
            "receipt_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": None,
            "contract_authority": {},
            "category": manifest.get("category", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "role": manifest.get("role", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "adapter_manifest_digest": None,
            "verifier": None,
            "environment": None,
            "split": {"primary_eval": {}},
            "split_hashes": {},
            "split_manifest_hash": manifest.get("split_manifest_hash") if isinstance(manifest, Mapping) else None,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash")
            if isinstance(manifest, Mapping)
            else None,
            "split_manifest_receipt_ref": manifest.get("split_manifest_receipt_ref")
            if isinstance(manifest, Mapping)
            else None,
            "scope": manifest.get("scope") if isinstance(manifest, Mapping) else None,
            "held_out": bool(manifest.get("held_out")) if isinstance(manifest, Mapping) else False,
            "held_out_receipt_ref": manifest.get("held_out_receipt_ref")
            if isinstance(manifest, Mapping)
            else None,
            "held_out_receipt_proven": False,
            "decontamination": None,
            "wandb": None,
            "tinker": None,
            "hf_checkpoints": [],
            "costs": None,
            "per_example_count": 0,
            "per_example_category": None,
            "per_example_verdicts": [],
            "per_example_digest": None,
            "receipt_identity_digest": None,
            "task_id_digest": None,
            "blockers": [str(exc)],
            "status": "BLOCKED",
        }


def validate_verilog_eval_receipt_record(record: Mapping[str, Any]) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record root must be a JSON object"]
    ready = bool(record.get("receipt_ready"))
    if record.get("status") == "READY" and not ready:
        return ["record status is READY but receipt_ready is false"]
    return [] if record.get("status") == "READY" else list(record.get("blockers", []))


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
