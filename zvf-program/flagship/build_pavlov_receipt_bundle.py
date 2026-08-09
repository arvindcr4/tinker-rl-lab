#!/usr/bin/env python3
"""Build and validate a deterministic, non-launching Pavlov receipt bundle.

The generator is intentionally local and zero-cost.  It describes the exact
contract suites and emits placeholders for receipts which have not been
recorded.  A bundle can only become launchable when a caller supplies complete,
immutable receipts to :func:`validate_bundle`; this module never contacts
W&B, Hugging Face, Tinker, or any other external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .pavlovs_domain_contract import CONTRACT_PATH, load_contract, validate_contract
except ImportError:  # Direct execution from the flagship directory.
    from pavlovs_domain_contract import CONTRACT_PATH, load_contract, validate_contract


SCHEMA_VERSION = "pavlov-receipt-bundle-v1"
EXPECTED_TRAINING_SUITE_COUNT = 12
EXPECTED_PRIMARY_EVAL_SUITE_COUNT = 14
EXPECTED_TOTAL_SUITE_COUNT = (
    EXPECTED_TRAINING_SUITE_COUNT + EXPECTED_PRIMARY_EVAL_SUITE_COUNT
)
LR_ARMS: tuple[tuple[str, float], ...] = (
    ("lr-1e-5", 1e-5),
    ("lr-2e-5", 2e-5),
    ("lr-4e-5", 4e-5),
)
EVIDENCE_STATUSES = frozenset({"prospective", "observed", "admissible", "rejected"})
REQUIRED_RECEIPT_FIELDS: tuple[str, ...] = (
    "dataset_or_source_revision",
    "license_or_approval",
    "split_task_id_hash",
    "container_runtime_digest",
    "verifier_hash",
    "model_revision",
    "decontamination_status",
    "budget_receipt",
    "wandb_run_identity",
    "tinker_run_identity",
    "cost_status",
    "hf_checkpoints",
)

# These are the only values treated as an absent receipt.  In particular, a
# status flag, namespace, repo name, or boolean is never enough on its own.
_PLACEHOLDER_WORDS = frozenset(
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
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_URL_RE = re.compile(r"^https://[^\s]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_IMMUTABLE_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by all bundle hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text without consulting filesystem, environment, or a service."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value using :func:`canonical_json`."""

    return sha256_text(canonical_json(value))


def _contract_suite_entries(contract: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Return suite entries while retaining duplicate IDs for validation."""

    registry = contract.get("suite_registry")
    if isinstance(registry, Mapping):
        return [(str(suite_id), suite) for suite_id, suite in registry.items() if isinstance(suite, Mapping)]
    if isinstance(registry, list):
        entries: list[tuple[str, Mapping[str, Any]]] = []
        for suite in registry:
            if isinstance(suite, Mapping):
                entries.append((str(suite.get("id", "")), suite))
        return entries
    return []


def _expected_suite_specs(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], list[str]]:
    entries = _contract_suite_entries(contract)
    ids = [suite_id for suite_id, _ in entries]
    duplicate_ids = sorted(suite_id for suite_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError("duplicate suite IDs in contract: " + ", ".join(duplicate_ids))
    training = {
        suite_id: suite
        for suite_id, suite in entries
        if suite.get("role") == "train"
    }
    primary_eval = {
        suite_id: suite
        for suite_id, suite in entries
        if suite.get("role") == "primary_eval"
    }
    domains = sorted(str(domain) for domain in contract.get("domains", []))
    if len(training) != EXPECTED_TRAINING_SUITE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TRAINING_SUITE_COUNT} training suites, found {len(training)}"
        )
    if len(primary_eval) != EXPECTED_PRIMARY_EVAL_SUITE_COUNT:
        raise ValueError(
            "expected "
            f"{EXPECTED_PRIMARY_EVAL_SUITE_COUNT} primary_eval suites, "
            f"found {len(primary_eval)}"
        )
    expected = set(training) | set(primary_eval)
    if len(expected) != EXPECTED_TOTAL_SUITE_COUNT:
        raise ValueError("training and primary_eval suite IDs are not unique")
    return training, primary_eval, domains


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_WORDS
    return False


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and not _is_placeholder(value) and bool(value.strip())


def _immutable_revision(value: Any) -> bool:
    """Accept full commit/digest-like identities, never mutable branch names."""

    if not _nonempty_text(value):
        return False
    return bool(_IMMUTABLE_RE.fullmatch(value.strip()))


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip()))


def _valid_url(value: Any) -> bool:
    return isinstance(value, str) and bool(_URL_RE.fullmatch(value.strip()))


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and not _is_placeholder(record[name]):
            return record[name]
    return None


def _normalise_identity(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _split_task_hash_valid(record: Mapping[str, Any]) -> bool:
    combined = _first_value(
        record,
        ("split_task_id_hash", "split_manifest_task_id_hash", "split_manifest_hash"),
    )
    task_hash = _first_value(record, ("task_id_hash", "task_id_hashes"))
    if isinstance(combined, Mapping):
        split_hash = _first_value(combined, ("split_manifest_hash", "split_hash"))
        task_hash = _first_value(combined, ("task_id_hash", "task_id_hashes"))
        return _sha256_digest(split_hash) and _sha256_digest(task_hash)
    if task_hash is not None:
        return _sha256_digest(combined) and _sha256_digest(task_hash)
    return _sha256_digest(combined)


def _license_receipt_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        approved = value.get("approved") is True or value.get("signoff") is True
        identity = _first_value(value, ("receipt_id", "approval_id", "license_id"))
        if approved and _nonempty_text(identity):
            return True
        if any(not _is_placeholder(value.get(key)) for key in ("receipt_id", "approval_id", "license_id", "sha256")):
            return any(
                _immutable_revision(value.get(key)) or _sha256_digest(value.get(key))
                for key in ("receipt_id", "approval_id", "license_id", "sha256")
            )
        return False
    if not _nonempty_text(value):
        return False
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDER_WORDS or "placeholder" in lowered or "to_be_" in lowered:
        return False
    # A concrete license identifier/approval label is sufficient; a generic
    # string such as "license-receipt" is intentionally rejected above.
    return bool(
        _immutable_revision(value)
        or _sha256_digest(value)
        or any(token in lowered for token in ("approved", "approval", "signoff", "cc-", "apache", "mit", "license:"))
    )


def _decontamination_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        status = value.get("status")
        identity = _first_value(value, ("receipt_id", "sha256", "hash", "digest"))
        if not (_immutable_revision(identity) or _sha256_digest(identity)):
            return False
    else:
        status = value
    if not _nonempty_text(status):
        return False
    if isinstance(value, Mapping):
        return status.strip().lower() in {
            "verified",
            "complete",
            "completed",
            "clean",
            "passed",
            "admissible",
        }
    return _sha256_digest(value) or _immutable_revision(value)


def _wandb_identity_valid(value: Any) -> bool:
    identity = _normalise_identity(value)
    if identity is None or identity.get("online") is not True:
        return False
    required = ("entity", "project", "group", "run_id", "run_url")
    if any(not _nonempty_text(identity.get(key)) for key in required):
        return False
    return _valid_url(identity.get("run_url"))


def _tinker_identity_valid(value: Any) -> bool:
    identity = _normalise_identity(value)
    if identity is None or not _nonempty_text(identity.get("run_id")):
        return False
    return _cost_status_valid(identity.get("cost_status"))


def _cost_status_valid(value: Any) -> bool:
    if not _nonempty_text(value):
        return False
    return value.strip().lower() in {
        "authorized",
        "approved",
        "within_cap",
        "zero_cost",
        "complete",
        "observed",
    }


def _budget_receipt_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        identity = _first_value(value, ("receipt_id", "authorization_id", "sha256", "hash"))
        amount = value.get("maximum_usd", value.get("max_usd"))
        authorized = value.get("authorized") is True or str(value.get("status", "")).lower() in {
            "authorized",
            "approved",
        }
        return bool(
            _nonempty_text(identity)
            and (_immutable_revision(identity) or _sha256_digest(identity) or authorized)
            and isinstance(amount, (int, float))
            and amount > 0
        )
    return _immutable_revision(value) or _sha256_digest(value)


def _hf_checkpoints_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    seen: set[tuple[str, str, str]] = set()
    seen_repo_revisions: set[tuple[str, str]] = set()
    stages: set[str] = set()
    for checkpoint in value:
        if not isinstance(checkpoint, Mapping):
            return False
        repo = _first_value(checkpoint, ("repo_url", "repo", "repository"))
        revision = _first_value(checkpoint, ("revision", "commit", "sha"))
        url = _first_value(checkpoint, ("url", "checkpoint_url", "repo_url"))
        visibility = checkpoint.get("visibility")
        safe = checkpoint.get("safe_public_artifact")
        if not _valid_url(repo) or not _immutable_revision(revision) or not _valid_url(url):
            return False
        if visibility not in {"public", "private"} or safe is not True:
            return False
        stage = str(checkpoint.get("stage", "")).strip().lower()
        if stage:
            stages.add(stage)
        identity = (str(repo), str(revision), str(url))
        repo_revision = (str(repo), str(revision))
        if identity in seen or repo_revision in seen_repo_revisions:
            return False
        seen.add(identity)
        seen_repo_revisions.add(repo_revision)
    return {"initial", "periodic", "final"}.issubset(stages)


def _receipt_completeness(record: Mapping[str, Any]) -> dict[str, bool]:
    """Return field-level readiness without treating metadata as receipts."""

    dataset_revision = _first_value(
        record,
        ("dataset_or_source_revision", "dataset_source_revision", "dataset_revision", "source_revision"),
    )
    license_value = _first_value(record, ("license_or_approval", "license", "approval"))
    container = _first_value(record, ("container_runtime_digest", "container_digest", "runtime_digest"))
    verifier = _first_value(record, ("verifier_hash", "verifier_digest", "verifier_revision"))
    model_revision = _first_value(record, ("model_revision", "model_revision_receipt"))
    decontamination = _first_value(record, ("decontamination_status", "decontamination"))
    budget = _first_value(record, ("budget_receipt", "budget_authorization"))
    wandb = _first_value(record, ("wandb_run_identity", "wandb"))
    tinker = _first_value(record, ("tinker_run_identity", "tinker"))
    cost = _first_value(record, ("cost_status",))
    hf = _first_value(record, ("hf_checkpoints", "hf_checkpoint_revisions", "checkpoints"))
    return {
        "dataset_or_source_revision": _immutable_revision(dataset_revision),
        "license_or_approval": _license_receipt_valid(license_value),
        "split_task_id_hash": _split_task_hash_valid(record),
        "container_runtime_digest": _sha256_digest(container),
        "verifier_hash": _sha256_digest(verifier),
        "model_revision": _immutable_revision(model_revision) or _sha256_digest(model_revision),
        "decontamination_status": _decontamination_valid(decontamination),
        "budget_receipt": _budget_receipt_valid(budget),
        "wandb_run_identity": _wandb_identity_valid(wandb),
        "tinker_run_identity": _tinker_identity_valid(tinker),
        "cost_status": _cost_status_valid(cost),
        "hf_checkpoints": _hf_checkpoints_valid(hf),
    }


def _record_entry_hash_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"entry_hash", "blockers", "launchable", "admissible"}
    }


def _record_blockers(record: Mapping[str, Any]) -> list[str]:
    complete = _receipt_completeness(record)
    blockers = [
        f"{record.get('suite_id', '<missing>')}: missing or invalid {field} receipt"
        for field in REQUIRED_RECEIPT_FIELDS
        if not complete[field]
    ]
    if record.get("evidence_status") not in EVIDENCE_STATUSES:
        blockers.append(f"{record.get('suite_id', '<missing>')}: invalid evidence_status")
    return blockers


def _split_classification(split_description: Any) -> str:
    text = str(split_description or "").lower()
    if any(marker in text for marker in ("held-out", "private")):
        return "held_out_or_private_described"
    return "primary_eval_not_designated_held_out"


def _new_suite_record(suite_id: str, suite: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "suite_id": suite_id,
        "role": str(suite.get("role")),
        "domains": sorted(str(domain) for domain in suite.get("domains", [])),
        "split_description": suite.get("split"),
        "split_classification": _split_classification(suite.get("split")),
        "stateful": bool(suite.get("stateful")),
        "artifact_or_side_effect": bool(suite.get("artifact_or_side_effect")),
        "dataset_or_source_revision": None,
        "dataset_revision": None,
        "source_revision": None,
        "license_or_approval": None,
        "license": None,
        "split_task_id_hash": None,
        "split_manifest_hash": None,
        "task_id_hashes": None,
        "container_runtime_digest": None,
        "container_digest": None,
        "verifier_hash": None,
        "model_revision": None,
        "model_revision_receipt": None,
        "decontamination_status": "missing",
        "budget_receipt": None,
        "wandb_run_identity": {
            "entity": None,
            "project": None,
            "group": None,
            "run_id": None,
            "run_url": None,
            "online": False,
        },
        "tinker_run_identity": {"run_id": None, "cost_status": "missing"},
        "cost_status": "missing",
        "hf_checkpoints": [],
        "evidence_status": "prospective",
        "launchable": False,
        "admissible": False,
    }
    record["blockers"] = _record_blockers(record)
    record["entry_hash"] = sha256_json(_record_entry_hash_payload(record))
    return record


def _apply_receipt_overrides(
    record: dict[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        return record
    updated = dict(record)
    for key, value in overrides.items():
        if key in {"suite_id", "role", "domains", "split_description", "split_classification"}:
            continue
        updated[key] = value
    complete = _receipt_completeness(updated)
    updated["launchable"] = all(complete.values()) and updated.get("evidence_status") in EVIDENCE_STATUSES - {"rejected"}
    updated["admissible"] = updated["launchable"] and updated.get("evidence_status") in {"observed", "admissible"}
    updated["blockers"] = _record_blockers(updated)
    updated["entry_hash"] = sha256_json(_record_entry_hash_payload(updated))
    return updated


def _company_domain_coverage(
    contract: Mapping[str, Any],
    training: Mapping[str, Mapping[str, Any]],
    primary_eval: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, list[str]]]], list[str]]:
    coverage: dict[str, dict[str, dict[str, list[str]]]] = {}
    errors: list[str] = []
    for company in contract.get("companies", []):
        if not isinstance(company, Mapping):
            continue
        name = str(company.get("name"))
        required = sorted(str(domain) for domain in company.get("domains", []))
        train_by_domain = {
            domain: sorted(
                suite_id for suite_id, suite in training.items() if domain in suite.get("domains", [])
            )
            for domain in required
        }
        eval_by_domain = {
            domain: sorted(
                suite_id for suite_id, suite in primary_eval.items() if domain in suite.get("domains", [])
            )
            for domain in required
        }
        for domain, suite_ids in train_by_domain.items():
            if not suite_ids:
                errors.append(f"{name}: required domain {domain} has no training suite")
        for domain, suite_ids in eval_by_domain.items():
            if not suite_ids:
                errors.append(f"{name}: required domain {domain} has no primary_eval suite")
        coverage[name] = {"training": train_by_domain, "primary_eval": eval_by_domain}
    return coverage, errors


def _budget_guard(contract: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    budget = contract.get("budget_gate")
    if not isinstance(budget, Mapping):
        return {"status": "BLOCKED", "paid_jobs_may_launch": False}, ["budget_gate is missing"]
    blockers: list[str] = []
    contract_status = str(contract.get("status", "")).strip().lower()
    budget_status = str(budget.get("status", "")).strip().upper()
    authorized_contract = contract_status in {"authorized", "authorized_tinker_only", "ready", "ready_to_run"}
    authorized_budget = budget_status in {"AUTHORIZED", "AUTHORIZED_TINKER_ONLY", "READY", "READY_TO_RUN"}
    paid = budget.get("paid_jobs_may_launch") is True
    maximum = budget.get("maximum_usd")
    if not isinstance(maximum, (int, float)) or maximum <= 0:
        blockers.append("budget: maximum_usd is unset or not positive")
    if paid and not (authorized_contract and authorized_budget):
        blockers.append(
            "budget: contract status and budget status conflict with paid_jobs_may_launch=true"
        )
    if not paid:
        blockers.append("budget: paid jobs are not authorized")
    return {
        "provider": budget.get("provider"),
        "paid_jobs_may_launch": bool(paid),
        "maximum_usd": maximum,
        "operational_cap_usd": budget.get("operational_cap_usd"),
        "safety_reserve_usd": budget.get("safety_reserve_usd"),
        "contract_status": contract.get("status"),
        "budget_status": budget.get("status"),
        "status_reconciled": not any("conflict" in blocker for blocker in blockers),
        "receipt_required": True,
    }, blockers


def _successive_halving() -> dict[str, Any]:
    arm_ids = [arm_id for arm_id, _ in LR_ARMS]
    return {
        "method": "successive_halving",
        "arms": arm_ids,
        "screening_arm_ids": arm_ids,
        "screening_learning_rates": [rate for _, rate in LR_ARMS],
        "short_screening_steps": 10,
        "checkpoint_steps": [5, 10],
        "selection_metric": "sealed selection slice perfect-call rate",
        "selection_rule": "maximize sealed selection slice perfect-call rate",
        "tie_breakers": ["strict mean reward", "lower estimated cost"],
        "winner_extension": "extend only the winning arm",
        "decision_gates": [
            {
                "id": "receipt_preflight",
                "requires": [
                    "all suite receipts and immutable model revisions",
                    "online W&B run identity",
                    "HF receipt for every periodic and final checkpoint",
                ],
                "on_failure": "BLOCKED; do not allocate a job",
            },
            {
                "id": "short_screening",
                "requires": ["all three arms use the same immutable train split and sealed selection slice"],
                "decision": "select by sealed selection slice metric",
            },
            {
                "id": "winner_extension",
                "requires": ["exactly one winner and valid receipts"],
                "decision": "extend only the winning arm",
            },
            {
                "id": "final_evaluation",
                "requires": ["final evaluation split is immutable and disjoint from selection"],
                "decision": "evaluate once; never promote xLAM-only evidence",
            },
        ],
        "selection_final_separation": {
            "selection_split_name": "sealed_selection_slice",
            "final_eval_split_name": "primary_eval_final",
            "selection_consulted_during_selection": True,
            "final_eval_consulted_during_selection": False,
            "must_be_disjoint": True,
            "held_out_label_requires_independent_receipt": True,
        },
    }


def build_bundle(
    contract: Mapping[str, Any] | None = None,
    receipt_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic preview bundle, optionally applying supplied receipts.

    ``receipt_overrides`` is a local test/integration seam.  It is never filled
    from environment variables or remote services; omission leaves explicit
    placeholders and therefore a blocked bundle.
    """

    if contract is None:
        contract = load_contract()
    if not isinstance(contract, Mapping):
        raise ValueError("contract must be an object")
    contract_errors = validate_contract(dict(contract))
    if contract_errors:
        raise ValueError("invalid Pavlov domain contract: " + "; ".join(contract_errors))
    training, primary_eval, domains = _expected_suite_specs(contract)
    train_domain_union = sorted({str(domain) for suite in training.values() for domain in suite.get("domains", [])})
    eval_domain_union = sorted({str(domain) for suite in primary_eval.values() for domain in suite.get("domains", [])})
    if train_domain_union != domains:
        raise ValueError("training suite domain union does not cover all declared domains")
    if eval_domain_union != domains:
        raise ValueError("primary_eval suite domain union does not cover all declared domains")
    company_coverage, coverage_errors = _company_domain_coverage(contract, training, primary_eval)
    if coverage_errors:
        raise ValueError("invalid Pavlov company coverage: " + "; ".join(coverage_errors))
    company_train_coverage = {
        company: sorted(
            {
                suite_id
                for suite_ids in details["training"].values()
                for suite_id in suite_ids
            }
        )
        for company, details in company_coverage.items()
    }
    company_eval_coverage = {
        company: sorted(
            {
                suite_id
                for suite_ids in details["primary_eval"].values()
                for suite_id in suite_ids
            }
        )
        for company, details in company_coverage.items()
    }

    overrides = receipt_overrides if isinstance(receipt_overrides, Mapping) else {}
    records: list[dict[str, Any]] = []
    for suite_id, suite in sorted({**training, **primary_eval}.items()):
        override = overrides.get(suite_id)
        records.append(_apply_receipt_overrides(_new_suite_record(suite_id, suite), override))

    budget_guard, budget_blockers = _budget_guard(contract)
    blockers = list(budget_blockers)
    for record in records:
        blockers.extend(record["blockers"])
    # The generator itself is non-launching.  Even a locally supplied complete
    # receipt set cannot cause a job to be launched by this module.
    launches_any_job = False
    all_ready = not blockers and all(record["launchable"] for record in records)
    status = "READY" if all_ready else "BLOCKED"
    primary_ids = sorted(primary_eval)
    training_ids = sorted(training)
    domain_to_training = {
        domain: sorted(
            suite_id for suite_id, suite in training.items() if domain in suite.get("domains", [])
        )
        for domain in domains
    }
    domain_to_primary_eval = {
        domain: sorted(
            suite_id
            for suite_id, suite in primary_eval.items()
            if domain in suite.get("domains", [])
        )
        for domain in domains
    }
    structural_held_out = sorted(
        suite_id
        for suite_id, suite in primary_eval.items()
        if any(marker in str(suite.get("split", "")).lower() for marker in ("held-out", "private"))
    )
    held_out_receipt_proven = sorted(
        record["suite_id"]
        for record in records
        if record["suite_id"] in structural_held_out and record["launchable"]
    )
    primary_eval_not_designated_held_out = sorted(
        set(primary_ids) - set(structural_held_out)
    )
    entry_payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_schema_version": contract.get("schema_version"),
        "training_suite_ids": training_ids,
        "primary_eval_suite_ids": primary_ids,
        "training_suite_count": len(training_ids),
        "primary_eval_suite_count": len(primary_ids),
        "suite_count": len(records),
        "training_suite_domain_map": {
            suite_id: sorted(str(domain) for domain in suite.get("domains", []))
            for suite_id, suite in sorted(training.items())
        },
        "primary_eval_suite_domain_map": {
            suite_id: sorted(str(domain) for domain in suite.get("domains", []))
            for suite_id, suite in sorted(primary_eval.items())
        },
        "domain_to_training_suite_ids": domain_to_training,
        "domain_to_primary_eval_suite_ids": domain_to_primary_eval,
        "suites": records,
        "training_suites": [record for record in records if record["role"] == "train"],
        "primary_eval_suites": [
            record for record in records if record["role"] == "primary_eval"
        ],
        "receipts": {record["suite_id"]: record for record in records},
        "suite_receipts": {record["suite_id"]: record for record in records},
        "budget_guard": budget_guard,
        "company_domain_coverage": company_coverage,
        "company_train_coverage": company_train_coverage,
        "company_eval_coverage": company_eval_coverage,
        "domains": domains,
        "structural_held_out_suite_ids": structural_held_out,
        "held_out_receipt_proven_suite_ids": held_out_receipt_proven,
        "primary_eval_not_designated_held_out_suite_ids": primary_eval_not_designated_held_out,
        "xlam_component": {
            "claim_scope": "component_only",
            "evidence_status": "observed",
            "observed_slice": "seed-809 7/100 only; not frozen portfolio evidence",
            "admissible": False,
            "launchable": False,
            "required_receipts": ["model_revision", "split_task_id_hash", "dataset_or_source_revision"],
        },
        "gsm8k": {
            "suite_id": "gsm8k_calibration",
            "role": "calibration_only",
            "evidence_status": "prospective",
            "primary_claim_allowed": False,
            "launchable": False,
        },
        "arms": [
            {"arm_id": arm_id, "learning_rate": rate, "launchable": False, "status": "BLOCKED"}
            for arm_id, rate in LR_ARMS
        ],
        "successive_halving": _successive_halving(),
        "required_receipt_fields": list(REQUIRED_RECEIPT_FIELDS),
        "evidence_statuses": sorted(EVIDENCE_STATUSES),
        "hf_policy": {
            "every_checkpoint": True,
            "periodic_and_final": True,
            "visibility": "public_or_private_per_quota_and_data_license_safety",
            "allowed_visibility": ["public", "private"],
            "unique_repo_revision_url_required": True,
            "safe_public_artifact_rule": True,
        },
        "dry_run_only": True,
        "allocation_allowed": False,
    }
    bundle_hash = sha256_json(entry_payload)
    bundle = dict(entry_payload)
    bundle.update(
        {
            "bundle_hash": bundle_hash,
            "status": status,
            "launchable": False,
            "launches_any_job": launches_any_job,
            "dry_run_only": True,
            "allocation_allowed": False,
            "blockers": blockers,
        }
    )
    return bundle


def _bundle_hash_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_hash", "status", "launchable", "launches_any_job", "blockers"}
    }


def validate_bundle(
    bundle: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> list[str]:
    """Return exact validation blockers; an empty list means receipts are ready.

    This function is deliberately side-effect free.  It does not launch jobs or
    query any service.  A generated placeholder bundle therefore returns many
    field-specific errors and remains ``BLOCKED``.
    """

    errors: list[str] = []
    if not isinstance(bundle, Mapping):
        return ["bundle must be an object"]
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is missing or unsupported")
    if bundle.get("launches_any_job") is not False:
        errors.append("launches_any_job must be false")
    if bundle.get("launchable") is not False:
        errors.append("bundle launchable must be false; receipt validation never launches")
    if bundle.get("allocation_allowed") is not False:
        errors.append("allocation_allowed must be false")

    if contract is None:
        try:
            contract = load_contract()
        except Exception as exc:  # pragma: no cover - defensive CLI path
            return [f"unable to load contract: {exc}"]
    if not isinstance(contract, Mapping):
        return errors + ["contract must be an object"]
    try:
        training, primary_eval, domains = _expected_suite_specs(contract)
    except ValueError as exc:
        errors.append(str(exc))
        training, primary_eval, domains = {}, {}, []
    expected = set(training) | set(primary_eval)
    entries = bundle.get("suites")
    if not isinstance(entries, list):
        return errors + ["suites must be a list with one entry per contract suite"]
    suite_ids = [str(entry.get("suite_id", "")) if isinstance(entry, Mapping) else "" for entry in entries]
    duplicates = sorted(suite_id for suite_id, count in Counter(suite_ids).items() if count > 1 and suite_id)
    if duplicates:
        errors.append("duplicate suite entries: " + ", ".join(duplicates))
    actual = set(suite_ids)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append("missing suite entries: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected suite entries: " + ", ".join(unexpected))
    if len(entries) != EXPECTED_TOTAL_SUITE_COUNT:
        errors.append(f"expected {EXPECTED_TOTAL_SUITE_COUNT} suite entries, found {len(entries)}")
    expected_receipts = {
        entry["suite_id"]: entry
        for entry in entries
        if isinstance(entry, Mapping) and "suite_id" in entry
    }
    if bundle.get("receipts") != expected_receipts:
        errors.append("receipts mapping must exactly mirror the suite entries")
    if bundle.get("suite_receipts") != expected_receipts:
        errors.append("suite_receipts mapping must exactly mirror the suite entries")
    expected_training_entries = [
        entry for entry in entries if isinstance(entry, Mapping) and entry.get("role") == "train"
    ]
    expected_eval_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("role") == "primary_eval"
    ]
    if bundle.get("training_suites") != expected_training_entries:
        errors.append("training_suites must exactly mirror train entries")
    if bundle.get("primary_eval_suites") != expected_eval_entries:
        errors.append("primary_eval_suites must exactly mirror primary_eval entries")

    receipt_blockers: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("suite entry must be an object")
            continue
        suite_id = str(entry.get("suite_id", "<missing>"))
        expected_suite = training.get(suite_id) or primary_eval.get(suite_id)
        if expected_suite is None:
            continue
        expected_role = str(expected_suite.get("role"))
        expected_domains = sorted(str(domain) for domain in expected_suite.get("domains", []))
        if entry.get("role") != expected_role:
            errors.append(f"{suite_id}: role does not match contract ({expected_role})")
        if sorted(str(domain) for domain in entry.get("domains", [])) != expected_domains:
            errors.append(f"{suite_id}: domains do not match contract")
        evidence = entry.get("evidence_status")
        if evidence not in EVIDENCE_STATUSES:
            errors.append(f"{suite_id}: evidence_status must be one of {sorted(EVIDENCE_STATUSES)}")
        completeness = _receipt_completeness(entry)
        missing_fields = [field for field in REQUIRED_RECEIPT_FIELDS if not completeness[field]]
        receipt_blockers.extend(_record_blockers(entry))
        if missing_fields:
            errors.extend(f"{suite_id}: missing or invalid {field} receipt" for field in missing_fields)
        computed_launchable = not missing_fields and evidence != "rejected"
        computed_admissible = computed_launchable and evidence in {"observed", "admissible"}
        if "launchable" in entry and entry.get("launchable") != computed_launchable:
            errors.append(f"{suite_id}: launchable flag does not match receipt completeness")
        if "admissible" in entry and entry.get("admissible") != computed_admissible:
            errors.append(f"{suite_id}: admissible flag does not match evidence status")
        if "entry_hash" in entry:
            expected_hash = sha256_json(_record_entry_hash_payload(entry))
            if entry.get("entry_hash") != expected_hash:
                errors.append(f"{suite_id}: entry_hash is not deterministic")

    if bundle.get("training_suite_ids") != sorted(training):
        errors.append("training_suite_ids do not exactly match the contract")
    if bundle.get("primary_eval_suite_ids") != sorted(primary_eval):
        errors.append("primary_eval_suite_ids do not exactly match the contract")
    if bundle.get("training_suite_count") != EXPECTED_TRAINING_SUITE_COUNT:
        errors.append("training_suite_count must be 12")
    if bundle.get("primary_eval_suite_count") != EXPECTED_PRIMARY_EVAL_SUITE_COUNT:
        errors.append("primary_eval_suite_count must be 14")
    if bundle.get("suite_count") != EXPECTED_TOTAL_SUITE_COUNT:
        errors.append("suite_count must be 26")
    if bundle.get("domains") != domains:
        errors.append("domains do not exactly match the contract")
    expected_structural_held_out = sorted(
        suite_id
        for suite_id, suite in primary_eval.items()
        if any(marker in str(suite.get("split", "")).lower() for marker in ("held-out", "private"))
    )
    if bundle.get("structural_held_out_suite_ids") != expected_structural_held_out:
        errors.append("structural_held_out_suite_ids must contain only explicitly described suites")
    expected_not_designated = sorted(set(primary_eval) - set(expected_structural_held_out))
    if bundle.get("primary_eval_not_designated_held_out_suite_ids") != expected_not_designated:
        errors.append("primary_eval_not_designated_held_out_suite_ids must preserve the eight non-held-out descriptions")
    proven = bundle.get("held_out_receipt_proven_suite_ids")
    if not isinstance(proven, list) or not set(proven).issubset(set(expected_structural_held_out)):
        errors.append("held_out_receipt_proven_suite_ids cannot include all primary_eval suites")
    expected_train_map = {
        suite_id: sorted(str(domain) for domain in suite.get("domains", []))
        for suite_id, suite in sorted(training.items())
    }
    expected_eval_map = {
        suite_id: sorted(str(domain) for domain in suite.get("domains", []))
        for suite_id, suite in sorted(primary_eval.items())
    }
    if bundle.get("training_suite_domain_map") != expected_train_map:
        errors.append("training_suite_domain_map does not match the contract")
    if bundle.get("primary_eval_suite_domain_map") != expected_eval_map:
        errors.append("primary_eval_suite_domain_map does not match the contract")
    expected_domain_to_training = {
        domain: sorted(
            suite_id for suite_id, suite in training.items() if domain in suite.get("domains", [])
        )
        for domain in domains
    }
    expected_domain_to_eval = {
        domain: sorted(
            suite_id
            for suite_id, suite in primary_eval.items()
            if domain in suite.get("domains", [])
        )
        for domain in domains
    }
    if bundle.get("domain_to_training_suite_ids") != expected_domain_to_training:
        errors.append("domain_to_training_suite_ids does not span all declared domains")
    if bundle.get("domain_to_primary_eval_suite_ids") != expected_domain_to_eval:
        errors.append("domain_to_primary_eval_suite_ids does not span all declared domains")
    train_union = {
        domain
        for suite in training.values()
        for domain in suite.get("domains", [])
    }
    eval_union = {
        domain
        for suite in primary_eval.values()
        for domain in suite.get("domains", [])
    }
    if train_union != set(domains):
        errors.append("training suite domains do not span all declared domains")
    if eval_union != set(domains):
        errors.append("primary_eval suite domains do not span all declared domains")
    company_coverage, company_errors = _company_domain_coverage(contract, training, primary_eval)
    if company_errors:
        errors.extend(company_errors)
    if bundle.get("company_domain_coverage") != company_coverage:
        errors.append("company_domain_coverage does not prove every required domain in train and primary_eval")
    expected_company_train = {
        company: sorted(
            {
                suite_id
                for suite_ids in details["training"].values()
                for suite_id in suite_ids
            }
        )
        for company, details in company_coverage.items()
    }
    expected_company_eval = {
        company: sorted(
            {
                suite_id
                for suite_ids in details["primary_eval"].values()
                for suite_id in suite_ids
            }
        )
        for company, details in company_coverage.items()
    }
    if bundle.get("company_train_coverage") != expected_company_train:
        errors.append("company_train_coverage does not match per-domain train coverage")
    if bundle.get("company_eval_coverage") != expected_company_eval:
        errors.append("company_eval_coverage does not match per-domain primary_eval coverage")

    gsm8k = bundle.get("gsm8k")
    if not isinstance(gsm8k, Mapping) or gsm8k.get("role") != "calibration_only" or gsm8k.get("primary_claim_allowed") is not False:
        errors.append("GSM8K must remain calibration_only and excluded from primary claims")
    xlam = bundle.get("xlam_component")
    if not isinstance(xlam, Mapping) or xlam.get("claim_scope") != "component_only" or xlam.get("launchable") is not False or xlam.get("admissible") is not False:
        errors.append("xLAM must remain component-only and non-launchable")
    hf_policy = bundle.get("hf_policy")
    if (
        not isinstance(hf_policy, Mapping)
        or hf_policy.get("every_checkpoint") is not True
        or hf_policy.get("periodic_and_final") is not True
        or hf_policy.get("visibility") == "private"
        or hf_policy.get("unique_repo_revision_url_required") is not True
        or hf_policy.get("safe_public_artifact_rule") is not True
    ):
        errors.append("HF policy must cover every checkpoint with safe per-artifact visibility and unique URLs")

    budget_guard, budget_errors = _budget_guard(contract)
    if bundle.get("budget_guard") != budget_guard:
        errors.append("budget_guard does not match contract")
    errors.extend(budget_errors)
    expected_blockers = list(budget_errors) + receipt_blockers
    if bundle.get("blockers") != expected_blockers:
        errors.append("blockers does not enumerate the exact budget and suite receipt blockers")
    if bundle.get("status") not in {"BLOCKED", "READY"}:
        errors.append("status must be BLOCKED or READY")
    # A READY label is only meaningful when no receipt/contract blocker was
    # found.  BLOCKED remains valid (and is the generator's placeholder
    # preview), while launchable stays false in either state.
    if bundle.get("status") == "READY" and errors:
        errors.append("status READY is not allowed while receipt blockers remain")

    if "bundle_hash" in bundle:
        expected_hash = sha256_json(_bundle_hash_payload(bundle))
        if bundle.get("bundle_hash") != expected_hash:
            errors.append("bundle_hash is not deterministic")
    return errors


def validate_receipt_bundle(
    bundle: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> list[str]:
    """Compatibility alias with an explicit name for callers and tests."""

    return validate_bundle(bundle, contract)


def build_receipt_bundle(
    contract: Mapping[str, Any] | None = None,
    receipt_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compatibility alias for :func:`build_bundle`."""

    return build_bundle(contract, receipt_overrides)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", "--out", dest="output", type=Path)
    parser.add_argument(
        "--validate",
        "--bundle",
        "--input",
        dest="validate_path",
        type=Path,
        help="validate an existing local bundle; exits nonzero for placeholders",
    )
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    if args.validate_path is not None:
        bundle = _load_json(args.validate_path)
        errors = validate_bundle(bundle, contract)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("VALID")
        return 0

    bundle = build_bundle(contract)
    rendered = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
