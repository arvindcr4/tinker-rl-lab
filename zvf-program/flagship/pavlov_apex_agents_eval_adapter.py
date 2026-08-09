#!/usr/bin/env python3
"""Offline boundary adapter for the ``apex_agents_eval`` primary suite.

This module describes the exact evaluation boundary without downloading a
benchmark, launching an environment, calling W&B/Tinker/HF, or fabricating a
result.  A caller may provide local, read-only metadata; every mutable or
missing field remains ``BLOCKED`` until an immutable receipt is supplied.

The upstream facts are deliberately narrow and source-pinned: Mercor's
official dataset card, paper, blog, and Archipelago repository.  The dataset
card requires contact acceptance and forbids training/crawling, so a local
receipt must prove those access conditions before provenance can be ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .pavlovs_domain_contract import CONTRACT_PATH, load_contract
except ImportError:  # Direct execution from the flagship directory.
    from pavlovs_domain_contract import CONTRACT_PATH, load_contract


SCHEMA_VERSION = "pavlov-apex-agents-eval-boundary-v1"
SUITE_ID = "apex_agents_eval"
ROLE = "primary_eval"
STRUCTURAL_HELD_OUT = True
EXPECTED_SOURCE_URL = "https://www.mercor.com/blog/introducing-apex-agents/"
EXPECTED_DATASET_URL = "https://huggingface.co/datasets/mercor/apex-agents"
EXPECTED_ARCHIPELAGO_URL = "https://github.com/Mercor-Intelligence/archipelago"
EXPECTED_PAPER_URL = "https://arxiv.org/abs/2601.14242"
EXPECTED_PAPER_REVISION = "v3"
EXPECTED_DATASET_ID = "mercor/apex-agents"
EXPECTED_BENCHMARK_NAME = "APEX-Agents"
EXPECTED_DATASET_LICENSE = "cc-by-4.0"
EXPECTED_TASK_COUNT = 480
EXPECTED_WORLD_COUNT = 33
EXPECTED_JOB_CATEGORIES = {
    "investment_banking": {"worlds": 10, "tasks": 160},
    "management_consulting": {"worlds": 11, "tasks": 160},
    "corporate_law": {"worlds": 12, "tasks": 160},
}
EXPECTED_DOMAINS = (
    "enterprise",
    "finance",
    "long_horizon",
    "multi_domain",
    "tool_use",
)
EXACT_MAXIMUM_USD = Decimal("18.00")
EXACT_OPERATIONAL_CAP_USD = Decimal("16.50")
EXACT_SAFETY_RESERVE_USD = Decimal("1.50")
_IMMUTABLE_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MONEY_RE = re.compile(r"^[0-9]+(?:\.[0-9]{1,2})?$")
_URL_RE = re.compile(r"^https://[^\s]+$")
_SUCCESS_STATES = frozenset({"success", "succeeded", "finished", "completed", "complete"})
_DEBIT_STATES = frozenset({"settled", "charged", "recorded", "succeeded", "complete"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _immutable(value: Any) -> bool:
    return isinstance(value, str) and bool(_IMMUTABLE_RE.fullmatch(value.strip()))


def _commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_RE.fullmatch(value.strip()))


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip()))


def _money(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if not _MONEY_RE.fullmatch(text):
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    return amount if amount.is_finite() else None


def _https_url(value: Any) -> bool:
    if not isinstance(value, str) or not _URL_RE.fullmatch(value.strip()):
        return False
    parsed = urlparse(value.strip())
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _hosted_url(value: Any, host: str) -> bool:
    return _https_url(value) and urlparse(str(value).strip()).netloc.lower() == host


def _hf_checkpoint_repo(value: Any) -> bool:
    if not _hosted_url(value, "huggingface.co"):
        return False
    parts = urlparse(str(value).strip()).path.strip("/").split("/")
    return len(parts) == 2 and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts)


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _contract_suite(contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    registry = contract.get("suite_registry")
    if isinstance(registry, Mapping):
        suite = registry.get(SUITE_ID)
        return suite if isinstance(suite, Mapping) else None
    if isinstance(registry, list):
        for candidate in registry:
            if isinstance(candidate, Mapping) and candidate.get("id") == SUITE_ID:
                return candidate
    return None


def _boundary_hash_payload(boundary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in boundary.items()
        if key not in {"boundary_hash", "status", "launchable", "blockers"}
    }


def _task_id_hash(task_ids: Sequence[str]) -> str:
    return sha256_json(sorted(task_ids))


def _split_hash(split_name: str, task_ids: Sequence[str]) -> str:
    return sha256_json({"split_name": split_name, "task_ids": sorted(task_ids)})


def _task_manifest_hash(
    source_revision: str,
    split_name: str,
    task_ids: Sequence[str],
) -> str:
    return sha256_json(
        {
            "suite_id": SUITE_ID,
            "source_revision": source_revision,
            "split_name": split_name,
            "task_ids": sorted(task_ids),
        }
    )


def _split_receipt_hash(split: Mapping[str, Any], task_id_hash: str, split_manifest_hash: str) -> str:
    return sha256_json(
        {
            "name": split.get("name"),
            "disjoint_from_training": split.get("disjoint_from_training"),
            "task_id_hash": task_id_hash,
            "split_manifest_hash": split_manifest_hash,
        }
    )


def _source_identity_hash(source: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "source_id": source.get("source_id"),
            "publisher": source.get("publisher"),
            "url": source.get("url"),
            "authoritative": source.get("authoritative"),
        }
    )


def _dataset_revision_receipt_hash(dataset: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "dataset_id": dataset.get("dataset_id"),
            "revision": dataset.get("revision"),
            "source_url": dataset.get("revision_source_url"),
        }
    )


def _license_approval_hash(license_record: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "license_id": license_record.get("license_id"),
            "approved": license_record.get("approved"),
            "source_url": license_record.get("source_url"),
        }
    )


def _access_receipt_payload(access: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in access.items() if key != "access_receipt_hash"
    }


def _metadata_receipt_payload(upstream: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in upstream.items()
        if key != "metadata_receipt_hash"
    }


def _artifact_contract_hash(artifacts: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "required": artifacts.get("required"),
            "artifact_types": artifacts.get("artifact_types"),
            "source_url": artifacts.get("source_url"),
        }
    )


def _artifact_receipt_hash(artifacts: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "contract_hash": artifacts.get("contract_hash"),
            "required": artifacts.get("required"),
            "artifact_types": artifacts.get("artifact_types"),
            "source_url": artifacts.get("source_url"),
        }
    )


def _environment_receipt_hash(environment: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "native": environment.get("native"),
            "runtime": environment.get("runtime"),
            "container_digest": environment.get("container_digest"),
            "environment_digest": environment.get("environment_digest"),
            "source_url": environment.get("source_url"),
        }
    )


def _verifier_receipt_hash(verifier: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "rule": verifier.get("rule"),
            "verifier_id": verifier.get("verifier_id"),
            "revision": verifier.get("revision"),
            "verifier_hash": verifier.get("verifier_hash"),
            "source_url": verifier.get("source_url"),
        }
    )


def _wandb_receipt_hash(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: record.get(key)
            for key in (
                "online",
                "entity",
                "project",
                "group",
                "run_id",
                "run_url",
                "state",
                "success",
            )
        }
    )


def _tinker_receipt_hash(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "provider": record.get("provider"),
            "run_id": record.get("run_id"),
            "cost_status": record.get("cost_status"),
        }
    )


def _hf_receipt_hash(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: record.get(key)
            for key in (
                "stage",
                "repo_url",
                "revision",
                "url",
                "visibility",
                "safe_public_artifact",
                "data_license_safe",
                "quota_safe",
                "private_artifact_safe",
            )
        }
    )


def _placeholder_boundary(contract: Mapping[str, Any]) -> dict[str, Any]:
    suite = _contract_suite(contract) or {}
    source_url = suite.get("url", EXPECTED_SOURCE_URL)
    contract_split_description = suite.get("split")
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "role": ROLE,
        "domains": list(EXPECTED_DOMAINS),
        # The contract calls this provider split "held-out worlds", but that
        # label is not a local held-out claim until an independent receipt is
        # supplied.  Keep the working boundary a sealed primary-eval slice.
        "source_split_description": "sealed selection slice",
        "contract_split_description": contract_split_description,
        "authoritative_source": {
            "source_id": None,
            "publisher": "Mercor",
            "url": source_url,
            "authoritative": False,
            "identity_hash": None,
        },
        "dataset": {
            "dataset_id": None,
            "revision": None,
            "revision_source_url": EXPECTED_DATASET_URL,
            "revision_receipt_hash": None,
            "license": {
                "license_id": None,
                "approved": False,
                "source_url": EXPECTED_DATASET_URL,
                "approval_hash": None,
            },
        },
        "task_ids": [],
        "task_id_hash": None,
        "split_manifest_hash": None,
        "task_manifest_hash": None,
        "split": {
            "name": "sealed selection slice",
            "disjoint_from_training": True,
            "receipt_hash": None,
        },
        "native_environment": {
            "native": True,
            "runtime": None,
            "container_digest": None,
            "environment_digest": None,
            "source_url": EXPECTED_ARCHIPELAGO_URL,
            "receipt_hash": None,
        },
        "artifact_contract": {
            "required": True,
            # Artifact names are provider/runtime facts, not safe defaults.
            "artifact_types": [],
            "source_url": EXPECTED_ARCHIPELAGO_URL,
            "contract_hash": None,
            "receipt_hash": None,
        },
        "verifier_contract": {
            "rule": suite.get("verifier_rule"),
            "verifier_id": None,
            "revision": None,
            "verifier_hash": None,
            "source_url": EXPECTED_ARCHIPELAGO_URL,
            "receipt_hash": None,
        },
        "upstream_metadata": {
            "verification_status": "unverified",
            "official_sources": {
                "publisher_url": EXPECTED_SOURCE_URL,
                "dataset_url": EXPECTED_DATASET_URL,
                "paper_url": EXPECTED_PAPER_URL,
                "archipelago_url": EXPECTED_ARCHIPELAGO_URL,
                "source_receipt_hashes": {},
            },
            "benchmark_name": None,
            "dataset_id": None,
            "dataset_revision": None,
            "paper_revision": None,
            "task_count": None,
            "world_count": None,
            "job_categories": {},
            "license": None,
            "intended_use": None,
            "training_permitted": None,
            "crawling_permitted": None,
            "access_constraints": {
                "contact_acceptance_required": None,
                "contact_acceptance_confirmed": False,
                "dataset_access_confirmed": False,
                "read_only_snapshot": False,
                "web_search_enabled": None,
                "network_used": False,
                "paid_calls_made": False,
                "access_receipt_hash": None,
            },
            "metadata_receipt_hash": None,
        },
        "result_receipts": {
            "wandb": {
                "online": False,
                "entity": None,
                "project": None,
                "group": None,
                "run_id": None,
                "run_url": None,
                "state": None,
                "success": False,
                "receipt_hash": None,
            },
            "tinker": {
                "provider": "Tinker",
                "run_id": None,
                "cost_status": None,
                "receipt_hash": None,
            },
            "hf_checkpoints": [],
        },
        "primary_eval": True,
        "receipt_proven_heldout": False,
        "heldout_claim_allowed": False,
        "heldout_receipt": None,
        "related_benchmarks": [],
        "xlam_substitute": False,
        "evidence_status": "prospective",
        "scientific_evidence_status": "not_established",
        "provenance_ready": False,
        "launchable": False,
        "status": "BLOCKED",
        "blockers": [],
    }


def build_boundary(
    metadata: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic boundary, copying only caller-supplied metadata."""

    contract = load_contract() if contract is None else contract
    if not isinstance(contract, Mapping):
        raise ValueError("contract must be an object")
    boundary = _placeholder_boundary(contract)
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if key in {"schema_version", "suite_id", "role", "status", "launchable", "blockers", "boundary_hash"}:
                continue
            boundary[key] = value
    task_ids = boundary.get("task_ids")
    if (
        isinstance(task_ids, list)
        and all(isinstance(task_id, str) and task_id.strip() for task_id in task_ids)
        and len(set(task_ids)) == len(task_ids)
    ):
        # Sorting is canonicalization, not receipt generation.  Hashes and
        # immutable pins must be supplied by the read-only metadata owner and
        # are checked verbatim by validate_boundary.
        boundary["task_ids"] = sorted(task_ids)
    # Seed the hash before the first validation so the validator can audit the
    # canonical payload without a special builder-only escape hatch.
    boundary["boundary_hash"] = sha256_json(_boundary_hash_payload(boundary))
    initial_errors = validate_boundary(boundary, contract)
    boundary["provenance_ready"] = not initial_errors
    boundary["launchable"] = False
    boundary["status"] = "READY" if not initial_errors else "BLOCKED"
    boundary["blockers"] = initial_errors
    boundary["boundary_hash"] = sha256_json(_boundary_hash_payload(boundary))
    return boundary


def _validate_source(boundary: Mapping[str, Any], suite: Mapping[str, Any], errors: list[str]) -> None:
    source = _mapping(boundary.get("authoritative_source"))
    expected_url = str(suite.get("url", EXPECTED_SOURCE_URL))
    if source is None or source.get("authoritative") is not True:
        _error(errors, "authoritative_source", "typed authoritative source identity is required")
        return
    if source.get("url") != expected_url or not _https_url(source.get("url")):
        _error(errors, "authoritative_source.url", "must exactly match the official contract source URL")
    if not _nonempty(source.get("source_id")) or source.get("publisher") != "Mercor":
        _error(errors, "authoritative_source", "Mercor source_id and publisher are required")
    if not _sha256(source.get("identity_hash")) or source.get("identity_hash") != _source_identity_hash(source):
        _error(errors, "authoritative_source.identity_hash", "source identity receipt must bind the exact official identity")


def _validate_dataset(boundary: Mapping[str, Any], errors: list[str]) -> None:
    dataset = _mapping(boundary.get("dataset"))
    if (
        dataset is None
        or dataset.get("dataset_id") != EXPECTED_DATASET_ID
        or not _immutable(dataset.get("revision"))
        or dataset.get("revision_source_url") != EXPECTED_DATASET_URL
        or not _sha256(dataset.get("revision_receipt_hash"))
        or dataset.get("revision_receipt_hash") != _dataset_revision_receipt_hash(dataset)
    ):
        _error(errors, "dataset", "official dataset identity, immutable revision, source, and bound receipt are required")
        return
    license_record = _mapping(dataset.get("license"))
    if (
        license_record is None
        or license_record.get("license_id") != EXPECTED_DATASET_LICENSE
        or license_record.get("approved") is not True
        or license_record.get("source_url") != EXPECTED_DATASET_URL
        or not _sha256(license_record.get("approval_hash"))
        or license_record.get("approval_hash") != _license_approval_hash(license_record)
    ):
        _error(errors, "dataset.license", "CC-BY approval must be typed, source-pinned, and cryptographically bound")


def _validate_tasks(boundary: Mapping[str, Any], errors: list[str]) -> None:
    task_ids = boundary.get("task_ids")
    split = _mapping(boundary.get("split"))
    split_name = str((split or {}).get("name", boundary.get("source_split_description", "")))
    if not isinstance(task_ids, list) or not task_ids or any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids) or len(set(task_ids)) != len(task_ids) or task_ids != sorted(task_ids):
        _error(errors, "task_ids", "sorted deterministic unique task IDs are required")
        return
    expected_task_hash = _task_id_hash(task_ids)
    expected_split_hash = _split_hash(split_name, task_ids)
    if boundary.get("task_id_hash") != expected_task_hash or not _sha256(boundary.get("task_id_hash")):
        _error(errors, "task_id_hash", "does not match sorted task IDs")
    if boundary.get("split_manifest_hash") != expected_split_hash or not _sha256(boundary.get("split_manifest_hash")):
        _error(errors, "split_manifest_hash", "does not match split name and task IDs")
    dataset = _mapping(boundary.get("dataset")) or {}
    expected_manifest_hash = _task_manifest_hash(str(dataset.get("revision")), split_name, task_ids)
    if boundary.get("task_manifest_hash") != expected_manifest_hash or not _sha256(boundary.get("task_manifest_hash")):
        _error(errors, "task_manifest_hash", "does not bind immutable source revision and task IDs")
    if split is None or split_name != "sealed selection slice" or split.get("disjoint_from_training") is not True:
        _error(errors, "split", "primary_eval must use the sealed selection slice and typed disjointness")
    elif (
        not _sha256(split.get("receipt_hash"))
        or split.get("receipt_hash")
        != _split_receipt_hash(split, str(boundary.get("task_id_hash")), str(boundary.get("split_manifest_hash")))
    ):
        _error(errors, "split.receipt_hash", "split receipt must bind the sealed slice and task hashes")


def _validate_native_contract(boundary: Mapping[str, Any], suite: Mapping[str, Any], errors: list[str]) -> None:
    environment = _mapping(boundary.get("native_environment"))
    if (
        environment is None
        or environment.get("native") is not True
        or not _nonempty(environment.get("runtime"))
        or environment.get("source_url") != EXPECTED_ARCHIPELAGO_URL
        or not _sha256(environment.get("container_digest"))
        or not _sha256(environment.get("environment_digest"))
        or not _sha256(environment.get("receipt_hash"))
        or environment.get("receipt_hash") != _environment_receipt_hash(environment)
    ):
        _error(errors, "native_environment", "native Archipelago runtime, immutable digests, source, and bound receipt are required")
    artifacts = _mapping(boundary.get("artifact_contract"))
    artifact_types = artifacts.get("artifact_types") if artifacts is not None else None
    if (
        artifacts is None
        or artifacts.get("required") is not True
        or not isinstance(artifact_types, list)
        or not artifact_types
        or any(not _nonempty(item) for item in artifact_types)
        or artifacts.get("source_url") != EXPECTED_ARCHIPELAGO_URL
        or not _sha256(artifacts.get("contract_hash"))
        or artifacts.get("contract_hash") != _artifact_contract_hash(artifacts)
        or not _sha256(artifacts.get("receipt_hash"))
        or artifacts.get("receipt_hash") != _artifact_receipt_hash(artifacts)
    ):
        _error(errors, "artifact_contract", "native Archipelago artifact types, source, contract, and receipt are required")
    verifier = _mapping(boundary.get("verifier_contract"))
    expected_rule = suite.get("verifier_rule")
    if expected_rule is None:
        expected_rule = "primary reward must inspect environment state or native artifacts whenever task correctness depends on them"
    if (
        verifier is None
        or verifier.get("rule") != expected_rule
        or not _nonempty(verifier.get("verifier_id"))
        or not _immutable(verifier.get("revision"))
        or verifier.get("source_url") != EXPECTED_ARCHIPELAGO_URL
        or not _sha256(verifier.get("verifier_hash"))
        or not _sha256(verifier.get("receipt_hash"))
        or verifier.get("receipt_hash") != _verifier_receipt_hash(verifier)
    ):
        _error(errors, "verifier_contract", "native Archipelago verifier rule, pinned revision, source, and receipt are required")


def _validate_upstream_metadata(boundary: Mapping[str, Any], errors: list[str]) -> None:
    upstream = _mapping(boundary.get("upstream_metadata"))
    if upstream is None:
        _error(errors, "upstream_metadata", "official upstream metadata receipt is required; no local pin may be invented")
        return
    if upstream.get("verification_status") != "verified":
        _error(errors, "upstream_metadata.verification_status", "must be verified by a local read-only official-source receipt")
    sources = _mapping(upstream.get("official_sources"))
    expected_sources = {
        "publisher_url": EXPECTED_SOURCE_URL,
        "dataset_url": EXPECTED_DATASET_URL,
        "paper_url": EXPECTED_PAPER_URL,
        "archipelago_url": EXPECTED_ARCHIPELAGO_URL,
    }
    if sources is None:
        _error(errors, "upstream_metadata.official_sources", "all four official source identities are required")
    else:
        for name, expected_url in expected_sources.items():
            if sources.get(name) != expected_url:
                _error(errors, f"upstream_metadata.official_sources.{name}", "must exactly match the authoritative upstream URL")
        source_receipts = sources.get("source_receipt_hashes")
        if not isinstance(source_receipts, Mapping) or set(source_receipts) != set(expected_sources):
            _error(errors, "upstream_metadata.official_sources.source_receipt_hashes", "one immutable receipt hash is required per official source")
        elif any(not _sha256(source_receipts.get(name)) for name in expected_sources):
            _error(errors, "upstream_metadata.official_sources.source_receipt_hashes", "source receipt hashes must be canonical SHA-256 values")
    expected_facts = {
        "benchmark_name": EXPECTED_BENCHMARK_NAME,
        "dataset_id": EXPECTED_DATASET_ID,
        "paper_revision": EXPECTED_PAPER_REVISION,
        "task_count": EXPECTED_TASK_COUNT,
        "world_count": EXPECTED_WORLD_COUNT,
        "job_categories": EXPECTED_JOB_CATEGORIES,
        "license": EXPECTED_DATASET_LICENSE,
        "intended_use": "evaluation_only",
        "training_permitted": False,
        "crawling_permitted": False,
    }
    for name, expected in expected_facts.items():
        if upstream.get(name) != expected:
            _error(errors, f"upstream_metadata.{name}", "does not match the authoritative APEX-Agents metadata")
    dataset = _mapping(boundary.get("dataset")) or {}
    if upstream.get("dataset_revision") != dataset.get("revision") or not _immutable(upstream.get("dataset_revision")):
        _error(errors, "upstream_metadata.dataset_revision", "must exactly bind the selected immutable dataset revision")
    access = _mapping(upstream.get("access_constraints"))
    if access is None:
        _error(errors, "upstream_metadata.access_constraints", "dataset access conditions and a local receipt are required")
    else:
        required_booleans = {
            "contact_acceptance_required": True,
            "contact_acceptance_confirmed": True,
            "dataset_access_confirmed": True,
            "read_only_snapshot": True,
            "web_search_enabled": False,
            "network_used": False,
            "paid_calls_made": False,
        }
        for name, expected in required_booleans.items():
            if type(access.get(name)) is not bool or access.get(name) is not expected:
                _error(errors, f"upstream_metadata.access_constraints.{name}", f"must be typed {expected}")
        access_hash = access.get("access_receipt_hash")
        if not _sha256(access_hash) or access_hash != sha256_json(_access_receipt_payload(access)):
            _error(errors, "upstream_metadata.access_constraints.access_receipt_hash", "must bind the exact access conditions")
    metadata_hash = upstream.get("metadata_receipt_hash")
    if not _sha256(metadata_hash) or metadata_hash != sha256_json(_metadata_receipt_payload(upstream)):
        _error(errors, "upstream_metadata.metadata_receipt_hash", "must bind the official facts, source receipts, and access receipt")


def _validate_wandb(value: Any, errors: list[str]) -> None:
    record = _mapping(value)
    run_id = record.get("run_id") if record is not None else None
    if (
        record is None
        or record.get("online") is not True
        or not all(_nonempty(record.get(name)) for name in ("entity", "project", "group"))
        or not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", run_id) is None
        or not _hosted_url(record.get("run_url"), "wandb.ai")
        or not str(record.get("run_url")).rstrip("/").endswith(run_id)
        or str(record.get("state", "")).lower() not in _SUCCESS_STATES
        or record.get("success") is not True
        or not _sha256(record.get("receipt_hash"))
        or record.get("receipt_hash") != _wandb_receipt_hash(record)
    ):
        _error(errors, "result_receipts.wandb", "online W&B run ID/URL/group and success receipt are required")


def _validate_tinker(value: Any, errors: list[str]) -> None:
    record = _mapping(value)
    run_id = record.get("run_id") if record is not None else None
    if (
        record is None
        or str(record.get("provider", "")).lower() != "tinker"
        or not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}", run_id) is None
        or str(record.get("cost_status", "")).lower() not in {"authorized", "settled", "charged", "complete", "observed"}
        or not _sha256(record.get("receipt_hash"))
        or record.get("receipt_hash") != _tinker_receipt_hash(record)
    ):
        _error(errors, "result_receipts.tinker", "Tinker run identity, cost state, and receipt hash are required")


def _validate_hf(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        _error(errors, "result_receipts.hf_checkpoints", "initial/periodic/final HF receipts are required")
        return
    stages: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for index, checkpoint in enumerate(value):
        path = f"result_receipts.hf_checkpoints[{index}]"
        record = _mapping(checkpoint)
        if record is None:
            _error(errors, path, "checkpoint must be an object")
            continue
        stage = str(record.get("stage", "")).lower()
        repo = record.get("repo_url")
        revision = record.get("revision")
        url = record.get("url")
        visibility = record.get("visibility")
        safe = record.get("safe_public_artifact")
        ok = (
            stage in {"initial", "periodic", "final"}
            and _hf_checkpoint_repo(repo)
            and _commit(revision)
            and _hosted_url(url, "huggingface.co")
            and str(url).startswith(str(repo).rstrip("/") + "/")
            and str(revision) in str(url)
            and visibility in {"public", "private"}
            and type(safe) is bool
            and _sha256(record.get("receipt_hash"))
            and record.get("receipt_hash") == _hf_receipt_hash(record)
        )
        if visibility == "public":
            ok = ok and safe is True and record.get("data_license_safe") is True and record.get("quota_safe") is True
        elif visibility == "private":
            ok = ok and safe is False and record.get("private_artifact_safe") is True
        identity = (str(repo), str(revision))
        if identity in identities:
            _error(errors, path, "duplicate HF repository/revision")
            ok = False
        identities.add(identity)
        stages.add(stage)
        if not ok:
            _error(errors, path, "immutable HF checkpoint identity and visibility safety receipt are required")
    if stages != {"initial", "periodic", "final"}:
        _error(errors, "result_receipts.hf_checkpoints", "exact initial/periodic/final stages are required")


def _validate_results(boundary: Mapping[str, Any], errors: list[str]) -> None:
    receipts = _mapping(boundary.get("result_receipts"))
    if receipts is None:
        _error(errors, "result_receipts", "W&B/Tinker/HF result receipt mapping is required")
        return
    _validate_wandb(receipts.get("wandb"), errors)
    _validate_tinker(receipts.get("tinker"), errors)
    _validate_hf(receipts.get("hf_checkpoints"), errors)


def validate_boundary(
    boundary: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> list[str]:
    """Return exact blockers for the E5 boundary; never launches or evaluates."""

    errors: list[str] = []
    if not isinstance(boundary, Mapping):
        return ["boundary must be a JSON object"]
    contract = load_contract() if contract is None else contract
    suite = _contract_suite(contract) if isinstance(contract, Mapping) else None
    if suite is None:
        _error(errors, "suite", "apex_agents_eval is missing from the contract")
        return errors
    if boundary.get("schema_version") != SCHEMA_VERSION:
        _error(errors, "schema_version", "unsupported boundary schema")
    if boundary.get("suite_id") != SUITE_ID or boundary.get("role") != ROLE:
        _error(errors, "suite", "must be apex_agents_eval primary_eval")
    if sorted(str(value) for value in boundary.get("domains", [])) != sorted(EXPECTED_DOMAINS):
        _error(errors, "domains", "must be the exact Apex Agents contract domains")
    _validate_source(boundary, suite, errors)
    _validate_dataset(boundary, errors)
    _validate_tasks(boundary, errors)
    _validate_native_contract(boundary, suite, errors)
    _validate_upstream_metadata(boundary, errors)
    _validate_results(boundary, errors)
    if type(boundary.get("primary_eval")) is not bool or boundary.get("primary_eval") is not True:
        _error(errors, "primary_eval", "must be true")
    suite_split = str(suite.get("split", ""))
    if boundary.get("contract_split_description") != suite_split:
        _error(errors, "contract_split_description", "must preserve the provider-declared split description")
    if boundary.get("source_split_description") != "sealed selection slice":
        _error(errors, "source_split_description", "must remain a sealed selection slice until held-out proof exists")
    if type(boundary.get("receipt_proven_heldout")) is not bool:
        _error(errors, "receipt_proven_heldout", "must be a typed boolean")
    if boundary.get("receipt_proven_heldout") is not True and boundary.get("heldout_claim_allowed") is not False:
        _error(errors, "heldout_claim_allowed", "must remain false without an independent held-out receipt")
    if boundary.get("receipt_proven_heldout") is not True and boundary.get("heldout_receipt") is not None:
        _error(errors, "heldout_receipt", "must be absent until an independent held-out receipt is proven")
    if boundary.get("receipt_proven_heldout") is True:
        if type(boundary.get("heldout_claim_allowed")) is not bool or boundary.get("heldout_claim_allowed") is not True:
            _error(errors, "heldout_claim_allowed", "must be explicitly true only with an independently proven receipt")
        receipt = _mapping(boundary.get("heldout_receipt"))
        task_ids = boundary.get("task_ids")
        dataset = _mapping(boundary.get("dataset")) or {}
        if (
            receipt is None
            or receipt.get("independent") is not True
            or receipt.get("split_name") != suite_split
            or receipt.get("source_revision") != dataset.get("revision")
            or receipt.get("task_id_hash") != boundary.get("task_id_hash")
            or not isinstance(task_ids, list)
            or receipt.get("split_manifest_hash") != _split_hash(suite_split, task_ids)
            or not _sha256(receipt.get("task_id_hash"))
            or not _sha256(receipt.get("split_manifest_hash"))
            or not _sha256(receipt.get("receipt_hash"))
            or receipt.get("receipt_hash")
            != sha256_json(
                {
                    "independent": receipt.get("independent"),
                    "split_name": receipt.get("split_name"),
                    "source_revision": receipt.get("source_revision"),
                    "task_id_hash": receipt.get("task_id_hash"),
                    "split_manifest_hash": receipt.get("split_manifest_hash"),
                }
            )
        ):
            _error(errors, "heldout_receipt", "independent held-out receipt must bind the contract split, revision, and exact tasks")
    if boundary.get("related_benchmarks"):
        _error(errors, "related_benchmarks", "related benchmarks cannot substitute for apex_agents_eval")
    if boundary.get("xlam_substitute") is not False:
        _error(errors, "xlam_substitute", "xLAM cannot substitute for this primary_eval boundary")
    if boundary.get("evidence_status") not in {"prospective", "observed", "admissible", "rejected"}:
        _error(errors, "evidence_status", "recognized evidence status is required")
    if boundary.get("scientific_evidence_status") != "not_established":
        _error(errors, "scientific_evidence_status", "boundary provenance is not scientific evidence")
    if type(boundary.get("provenance_ready")) is not bool:
        _error(errors, "provenance_ready", "must be a typed boolean")
    if boundary.get("status") not in {"BLOCKED", "READY"}:
        _error(errors, "status", "must be BLOCKED or READY")
    if boundary.get("status") == "READY" and (boundary.get("provenance_ready") is not True or boundary.get("blockers") not in ([], None)):
        _error(errors, "status", "READY requires provenance_ready=true and no blockers")
    if boundary.get("status") == "BLOCKED" and boundary.get("provenance_ready") is True:
        _error(errors, "status", "BLOCKED cannot claim provenance_ready=true")
    for name, expected in (("launchable", False),):
        if type(boundary.get(name)) is not bool or boundary.get(name) is not expected:
            _error(errors, name, "must remain a typed false boolean")
    if not _sha256(boundary.get("boundary_hash")):
        _error(errors, "boundary_hash", "canonical boundary hash is required")
    elif boundary.get("boundary_hash") != sha256_json(_boundary_hash_payload(boundary)):
        _error(errors, "boundary_hash", "does not match canonical boundary contents")
    return errors


def is_valid_boundary(boundary: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> bool:
    return not validate_boundary(boundary, contract)


# Explicit aliases keep the adapter discoverable without coupling callers to
# the short internal name used by this module.
build_apex_agents_eval_boundary = build_boundary
build_apex_agents_eval_adapter = build_boundary
validate_apex_agents_eval_boundary = validate_boundary
validate_apex_agents_eval_adapter = validate_boundary
is_valid_apex_agents_eval_boundary = is_valid_boundary


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--boundary", dest="input_path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    boundary = _load_json(args.input_path)
    errors = validate_boundary(boundary, load_contract(args.contract))
    if errors:
        for error in errors:
            print(error)
        return 1
    print("VALID: Apex Agents primary_eval boundary; no result or held-out claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
