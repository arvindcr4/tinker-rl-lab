#!/usr/bin/env python3
"""Independent, offline adversarial audit for Pavlov receipt artifacts.

The campaign and receipt builders are intentionally kept separate from this
module.  This auditor re-derives the contract coverage, receipt predicates,
canonical hash rules, budget caps, and cross-artifact bindings from plain JSON.
It never imports a provider SDK, contacts a service, launches a job, or treats
``status``/``online``/a URL/ID alone as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .pavlovs_domain_contract import CONTRACT_PATH, load_contract
except ImportError:  # Direct execution from the flagship directory.
    from pavlovs_domain_contract import CONTRACT_PATH, load_contract


AUDIT_SCHEMA_VERSION = "pavlov-receipt-bundle-audit-v1"
EXPECTED_TRAINING_SUITE_COUNT = 12
EXPECTED_PRIMARY_EVAL_SUITE_COUNT = 14
EXPECTED_TOTAL_SUITE_COUNT = 26
STRUCTURAL_HELD_OUT_SUITE_IDS: tuple[str, ...] = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "frontiermath_eval",
    "openreward_games_eval",
)
PENDING_PRIMARY_EVAL_SUITE_IDS: tuple[str, ...] = (
    "binaryaudit_eval",
    "frontier_swe_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "sdab_eval",
    "swe_bench_pro_eval",
    "verilog_eval",
    "webbench_eval",
)
EXACT_MAXIMUM_USD = Decimal("18.00")
EXACT_OPERATIONAL_CAP_USD = Decimal("16.50")
EXACT_SAFETY_RESERVE_USD = Decimal("1.50")
_IMMUTABLE_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_MONEY_RE = re.compile(r"^[0-9]+(?:\.[0-9]{1,2})?$")
_SUCCESS_STATES = frozenset({"success", "succeeded", "finished", "completed", "complete"})
_DEBIT_STATES = frozenset({"settled", "charged", "recorded", "succeeded", "complete"})
_EVIDENCE_STATUSES = frozenset({"prospective", "observed", "admissible", "rejected"})
_REQUIRED_BUNDLE_RECEIPTS = (
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


def canonical_json(value: Any) -> str:
    """Return the one canonical representation used by this auditor."""

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
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _https_host(value: Any, host: str) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"https://[^\s]+", value.strip()):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme == "https" and parsed.netloc.lower() == host


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _contract_suites(contract: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], list[str], list[str]]:
    registry = contract.get("suite_registry")
    entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(registry, Mapping):
        entries = [(str(key), value) for key, value in registry.items() if isinstance(value, Mapping)]
    elif isinstance(registry, list):
        entries = [
            (str(value.get("id", "")), value)
            for value in registry
            if isinstance(value, Mapping)
        ]
    ids = [suite_id for suite_id, _ in entries]
    duplicate_ids = [suite_id for suite_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError("duplicate suite IDs in contract: " + ", ".join(sorted(duplicate_ids)))
    training = {suite_id: suite for suite_id, suite in entries if suite.get("role") == "train"}
    primary = {suite_id: suite for suite_id, suite in entries if suite.get("role") == "primary_eval"}
    domains = sorted(str(domain) for domain in contract.get("domains", []))
    structural = sorted(
        suite_id
        for suite_id, suite in primary.items()
        if any(marker in str(suite.get("split", "")).lower() for marker in ("held-out", "private"))
    )
    pending = sorted(set(primary) - set(structural))
    return training, primary, domains, structural + pending


def _bundle_hash_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_hash", "status", "launchable", "launches_any_job", "blockers"}
    }


def _entry_hash_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if key not in {"entry_hash", "blockers", "launchable", "admissible"}
    }


def _campaign_hash_payload(campaign: Mapping[str, Any]) -> dict[str, Any]:
    # Cross references are deliberately excluded to avoid a hash cycle.  The
    # cross-binding digest below binds these fields after both artifacts exist.
    return {
        key: value
        for key, value in campaign.items()
        if key not in {"manifest_hash", "campaign_hash", "bundle_hash", "receipt_bundle_hash"}
    }


def _receipt_hash_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_hash"}


def _block(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _audit_license(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    if record is None:
        _block(errors, path, "approval mapping with immutable ID and receipt hash is required")
        return False
    approved = record.get("approved") is True or record.get("signoff") is True
    identity = _first(record, ("receipt_id", "approval_id", "license_id"))
    receipt_hash = _first(record, ("approval_hash", "receipt_hash"))
    ok = approved and (_immutable(identity) or _sha256(identity)) and _sha256(receipt_hash)
    if not ok:
        _block(errors, path, "typed approval, immutable identity, and SHA-256 receipt are required")
    return ok


def _audit_split(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    if record is not None:
        split_hash = _first(record, ("split_manifest_hash", "split_hash"))
        task_hash = _first(record, ("task_id_hash", "task_id_hashes"))
        ok = _sha256(split_hash) and _sha256(task_hash)
    else:
        # A composite SHA-256 is accepted as a stable split/task manifest
        # identity, but mutable status text or an arbitrary ID is not.
        ok = _sha256(value)
    if not ok:
        _block(errors, path, "immutable split-manifest and task-ID hashes are required")
    return ok


def _audit_decontamination(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    status = record.get("status") if record is not None else None
    identity = _first(record or {}, ("receipt_id", "decontamination_id", "id"))
    receipt_hash = _first(record or {}, ("receipt_hash", "sha256", "hash", "digest"))
    ok = (
        record is not None
        and isinstance(status, str)
        and status.strip().lower() in {"verified", "complete", "completed", "clean", "passed", "admissible"}
        and (_immutable(identity) or _sha256(identity))
        and _sha256(receipt_hash)
    )
    if not ok:
        _block(errors, path, "verified status cannot replace immutable decontamination receipt")
    return ok


def _audit_budget(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    if record is None:
        _block(errors, path, "budget authorization mapping is required")
        return False
    cap = _money(record.get("maximum_usd", record.get("max_usd")))
    operational = _money(record.get("operational_cap_usd"))
    reserve = _money(record.get("safety_reserve_usd"))
    identity = _first(record, ("receipt_id", "authorization_id"))
    ok = (
        record.get("authorized") is True
        and cap == EXACT_MAXIMUM_USD
        and operational == EXACT_OPERATIONAL_CAP_USD
        and reserve == EXACT_SAFETY_RESERVE_USD
        and operational + reserve == cap
        and (_immutable(identity) or _sha256(identity))
        and _sha256(_first(record, ("receipt_hash", "authorization_hash")))
    )
    if not ok:
        _block(errors, path, "typed authorization and exact 18.00/16.50/1.50 caps are required")
    return ok


def _audit_wandb(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    if record is None:
        _block(errors, path, "online W&B identity mapping is required")
        return False
    run_id = _first(record, ("run_id", "id"))
    state = str(_first(record, ("state", "status", "run_state")) or "").strip().lower()
    ok = (
        record.get("online") is True
        and all(_nonempty(record.get(name)) for name in ("entity", "project", "group"))
        and _nonempty(run_id)
        and _https_host(record.get("run_url", record.get("url")), "wandb.ai")
        and str(record.get("run_url", record.get("url"))).rstrip("/").endswith(str(run_id))
        and state in _SUCCESS_STATES
        and (record.get("success") is True or state in {"success", "succeeded"})
        and _sha256(_first(record, ("receipt_hash", "run_identity_hash")))
    )
    if not ok:
        _block(errors, path, "typed online W&B success receipt with provider URL and hash is required")
    return ok


def _audit_tinker(value: Any, path: str, errors: list[str]) -> bool:
    record = _mapping(value)
    status = str(_first(record or {}, ("cost_status", "status")) or "").strip().lower()
    ok = (
        record is not None
        and str(record.get("provider", "")).strip().lower() == "tinker"
        and _nonempty(_first(record, ("run_id", "id")))
        and status in {"authorized", "settled", "charged", "complete", "observed"}
        and _sha256(_first(record, ("receipt_hash", "run_identity_hash")))
    )
    if not ok:
        _block(errors, path, "Tinker provider, run identity, cost state, and hash are required")
    return ok


def _audit_hf(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, list) or not value:
        _block(errors, path, "initial, periodic, and final HF checkpoints are required")
        return False
    stages: set[str] = set()
    identities: set[tuple[str, str]] = set()
    urls: set[str] = set()
    ok = True
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        record = _mapping(item)
        if record is None:
            _block(errors, item_path, "checkpoint must be an object")
            ok = False
            continue
        stage = str(record.get("stage", "")).strip().lower()
        repo = record.get("repo_url", record.get("repo"))
        revision = record.get("revision", record.get("commit", record.get("sha")))
        url = record.get("url", record.get("checkpoint_url"))
        visibility = record.get("visibility")
        safe = record.get("safe_public_artifact")
        item_ok = (
            stage in {"initial", "periodic", "final"}
            and _https_host(repo, "huggingface.co")
            and _immutable(revision)
            and _https_host(url, "huggingface.co")
            and str(url).startswith(str(repo).rstrip("/") + "/")
            and str(revision) in str(url)
            and visibility in {"public", "private"}
            and type(safe) is bool
            and _sha256(_first(record, ("receipt_hash", "publication_hash")))
        )
        if visibility == "public":
            item_ok = item_ok and safe is True and record.get("data_license_safe") is True and record.get("quota_safe") is True
        elif visibility == "private":
            item_ok = item_ok and safe is False and record.get("private_artifact_safe") is True
        identity = (str(repo), str(revision))
        if identity in identities:
            _block(errors, item_path, "duplicate repository/revision identity")
            item_ok = False
        if isinstance(url, str) and url in urls:
            _block(errors, item_path, "duplicate checkpoint URL")
            item_ok = False
        identities.add(identity)
        if isinstance(url, str):
            urls.add(url)
        stages.add(stage)
        if not item_ok:
            _block(errors, item_path, "immutable HF URL/revision and visibility safety receipt are required")
            ok = False
    if stages != {"initial", "periodic", "final"}:
        _block(errors, path, "exact initial/periodic/final stages are required")
        ok = False
    return ok


def _audit_record(entry: Mapping[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    complete = {
        "dataset_or_source_revision": _immutable(_first(entry, ("dataset_or_source_revision", "dataset_revision", "source_revision"))),
        "license_or_approval": _audit_license(_first(entry, ("license_or_approval", "license", "approval")), f"{path}.license_or_approval", errors),
        "split_task_id_hash": _audit_split(_first(entry, ("split_task_id_hash", "split_manifest_hash")), f"{path}.split_task_id_hash", errors),
        "container_runtime_digest": _sha256(_first(entry, ("container_runtime_digest", "container_digest", "runtime_digest"))),
        "verifier_hash": _sha256(_first(entry, ("verifier_hash", "verifier_digest", "verifier_revision"))),
        "model_revision": _immutable(_first(entry, ("model_revision", "model_revision_receipt"))) or _sha256(_first(entry, ("model_revision", "model_revision_receipt"))),
        "decontamination_status": _audit_decontamination(_first(entry, ("decontamination_status", "decontamination")), f"{path}.decontamination_status", errors),
        "budget_receipt": _audit_budget(_first(entry, ("budget_receipt", "budget_authorization")), f"{path}.budget_receipt", errors),
        "wandb_run_identity": _audit_wandb(_first(entry, ("wandb_run_identity", "wandb")), f"{path}.wandb_run_identity", errors),
        "tinker_run_identity": _audit_tinker(_first(entry, ("tinker_run_identity", "tinker")), f"{path}.tinker_run_identity", errors),
        "cost_status": isinstance(entry.get("cost_status"), str) and entry.get("cost_status", "").strip().lower() in {"authorized", "approved", "within_cap", "zero_cost", "complete", "observed"},
        "hf_checkpoints": _audit_hf(_first(entry, ("hf_checkpoints", "checkpoints")), f"{path}.hf_checkpoints", errors),
    }
    for field in ("container_runtime_digest", "verifier_hash"):
        if not complete[field]:
            _block(errors, f"{path}.{field}", "SHA-256 digest is required")
    if not complete["dataset_or_source_revision"]:
        _block(errors, f"{path}.dataset_or_source_revision", "immutable revision is required")
    if not complete["model_revision"]:
        _block(errors, f"{path}.model_revision", "immutable revision is required")
    if not complete["cost_status"]:
        _block(errors, f"{path}.cost_status", "recognized cost status is required")
    for name in ("launchable", "admissible", "provenance_ready"):
        if type(entry.get(name)) is not bool:
            _block(errors, f"{path}.{name}", "typed boolean is required")
    if entry.get("scientific_evidence_status") != "not_established":
        _block(errors, f"{path}.scientific_evidence_status", "provenance cannot establish scientific evidence")
    claims = _mapping(entry.get("claims"))
    if claims is None or claims.get("portfolio_evidence") is not False or claims.get("primary_eval_heldout") is not False:
        _block(errors, f"{path}.claims", "portfolio and held-out claims must be false")
    expected_ready = all(complete.values())
    if entry.get("provenance_ready") is not expected_ready:
        _block(errors, f"{path}.provenance_ready", "does not match independently checked receipts")
    evidence = entry.get("evidence_status")
    expected_launchable = expected_ready and evidence in _EVIDENCE_STATUSES - {"rejected"}
    expected_admissible = expected_launchable and evidence in {"observed", "admissible"}
    if entry.get("launchable") is not expected_launchable:
        _block(errors, f"{path}.launchable", "does not match receipts/evidence")
    if entry.get("admissible") is not expected_admissible:
        _block(errors, f"{path}.admissible", "does not match evidence status")
    return errors


def audit_bundle(bundle: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> list[str]:
    """Audit one bundle independently and return deterministic blockers."""

    errors: list[str] = []
    if not isinstance(bundle, Mapping):
        return ["bundle: JSON object is required"]
    if not _sha256(bundle.get("bundle_hash")):
        _block(errors, "bundle_hash", "canonical bundle hash is required")
    elif bundle.get("bundle_hash") != sha256_json(_bundle_hash_payload(bundle)):
        _block(errors, "bundle_hash", "does not match canonical bundle contents")
    for name in ("launches_any_job", "launchable", "allocation_allowed", "dry_run_only"):
        expected = False if name != "dry_run_only" else True
        if type(bundle.get(name)) is not bool or bundle.get(name) is not expected:
            _block(errors, name, f"must be typed {expected!r}")
    if bundle.get("scientific_evidence_status") != "not_established":
        _block(errors, "scientific_evidence_status", "must remain not_established")
    claims = _mapping(bundle.get("claims"))
    if claims is None or claims.get("portfolio_evidence") is not False or claims.get("primary_eval_heldout") is not False:
        _block(errors, "claims", "portfolio and primary_eval held-out claims must be false")
    try:
        contract = load_contract() if contract is None else contract
        training, primary, domains, classified = _contract_suites(contract)
    except Exception as exc:
        return errors + [f"contract: unable to derive suite registry ({exc})"]
    if tuple(classified[: len(STRUCTURAL_HELD_OUT_SUITE_IDS)]) != STRUCTURAL_HELD_OUT_SUITE_IDS:
        _block(errors, "contract", "structural held-out IDs do not match the frozen six")
    if tuple(classified[len(STRUCTURAL_HELD_OUT_SUITE_IDS) :]) != PENDING_PRIMARY_EVAL_SUITE_IDS:
        _block(errors, "contract", "pending primary_eval IDs do not match the frozen eight")
    expected_training = sorted(training)
    expected_primary = sorted(primary)
    if len(training) != EXPECTED_TRAINING_SUITE_COUNT or bundle.get("training_suite_ids") != expected_training:
        _block(errors, "training_suite_ids", "exact 12 training IDs are required")
    if len(primary) != EXPECTED_PRIMARY_EVAL_SUITE_COUNT or bundle.get("primary_eval_suite_ids") != expected_primary:
        _block(errors, "primary_eval_suite_ids", "exact 14 primary_eval IDs are required")
    if bundle.get("training_suite_count") != EXPECTED_TRAINING_SUITE_COUNT:
        _block(errors, "training_suite_count", "must be 12")
    if bundle.get("primary_eval_suite_count") != EXPECTED_PRIMARY_EVAL_SUITE_COUNT:
        _block(errors, "primary_eval_suite_count", "must be 14")
    if bundle.get("suite_count") != EXPECTED_TOTAL_SUITE_COUNT:
        _block(errors, "suite_count", "must be 26")
    if bundle.get("structural_held_out_suite_ids") != list(STRUCTURAL_HELD_OUT_SUITE_IDS):
        _block(errors, "structural_held_out_suite_ids", "must be the frozen six IDs")
    if bundle.get("primary_eval_not_designated_held_out_suite_ids") != list(PENDING_PRIMARY_EVAL_SUITE_IDS):
        _block(errors, "primary_eval_not_designated_held_out_suite_ids", "must be the frozen eight pending IDs")
    entries = bundle.get("suites")
    if not isinstance(entries, list):
        return errors + ["suites: list with one entry per contract suite is required"]
    ids = [entry.get("suite_id") if isinstance(entry, Mapping) else None for entry in entries]
    if len(entries) != EXPECTED_TOTAL_SUITE_COUNT:
        _block(errors, "suites", "must contain exactly 26 entries")
    if len([suite_id for suite_id, count in Counter(ids).items() if suite_id is not None and count > 1]):
        _block(errors, "suites", "duplicate suite IDs are forbidden")
    expected_ids = set(training) | set(primary)
    if set(ids) != expected_ids:
        _block(errors, "suites", "suite IDs do not exactly match contract")
    expected_receipts = {
        entry.get("suite_id"): entry for entry in entries if isinstance(entry, Mapping) and entry.get("suite_id") is not None
    }
    if bundle.get("receipts") != expected_receipts or bundle.get("suite_receipts") != expected_receipts:
        _block(errors, "receipts", "receipt maps must cryptographically mirror suite entries")
    for index, entry in enumerate(entries):
        path = f"suites[{index}]"
        if not isinstance(entry, Mapping):
            _block(errors, path, "object is required")
            continue
        suite_id = entry.get("suite_id")
        expected = training.get(suite_id) or primary.get(suite_id)
        if expected is None:
            continue
        if entry.get("role") != expected.get("role"):
            _block(errors, f"{path}.role", "does not match contract")
        if sorted(str(value) for value in entry.get("domains", [])) != sorted(str(value) for value in expected.get("domains", [])):
            _block(errors, f"{path}.domains", "do not match contract")
        if not _sha256(entry.get("entry_hash")):
            _block(errors, f"{path}.entry_hash", "canonical entry hash is required")
        elif entry.get("entry_hash") != sha256_json(_entry_hash_payload(entry)):
            _block(errors, f"{path}.entry_hash", "does not match canonical entry contents")
        errors.extend(_audit_record(entry, path))
    budget = _mapping(bundle.get("budget_guard"))
    if budget is None:
        _block(errors, "budget_guard", "budget guard mapping is required")
    else:
        if type(budget.get("paid_jobs_may_launch")) is not bool:
            _block(errors, "budget_guard.paid_jobs_may_launch", "typed boolean is required")
        for field, expected_value in (("maximum_usd", EXACT_MAXIMUM_USD), ("operational_cap_usd", EXACT_OPERATIONAL_CAP_USD), ("safety_reserve_usd", EXACT_SAFETY_RESERVE_USD)):
            if _money(budget.get(field)) != expected_value:
                _block(errors, f"budget_guard.{field}", f"must be exactly {expected_value}")
        if budget.get("paid_jobs_may_launch") is True:
            status = str(budget.get("contract_status", "")).lower()
            gate = str(budget.get("budget_status", "")).upper()
            if status not in {"authorized", "authorized_tinker_only", "ready", "ready_to_run"} or gate not in {"AUTHORIZED", "AUTHORIZED_TINKER_ONLY", "READY", "READY_TO_RUN"}:
                _block(errors, "budget_guard", "contract and budget authorization statuses conflict")
    policy = _mapping(bundle.get("hf_policy"))
    safe_rule_value = policy.get("safe_public_artifact_rule") if policy else None
    safe_rule = _mapping(safe_rule_value)
    safe_rule_complete = safe_rule_value is True or bool(
        safe_rule is not None
        and all(
            safe_rule.get(name) is True
            for name in (
                "public_requires_data_license_safe",
                "public_requires_quota_safe",
                "private_requires_private_artifact_safe",
            )
        )
    )
    if policy is None or policy.get("every_checkpoint") is not True or policy.get("periodic_and_final") is not True or policy.get("unique_repo_revision_url_required") is not True or set(policy.get("allowed_visibility", [])) != {"public", "private"} or not safe_rule_complete:
        _block(errors, "hf_policy", "public/private per-checkpoint safety policy is incomplete")
    xlam = _mapping(bundle.get("xlam_component"))
    if xlam is None or xlam.get("claim_scope") != "component_only" or xlam.get("launchable") is not False or xlam.get("admissible") is not False:
        _block(errors, "xlam_component", "must remain component-only and non-launchable")
    gsm = _mapping(bundle.get("gsm8k"))
    if gsm is None or gsm.get("role") != "calibration_only" or gsm.get("primary_claim_allowed") is not False:
        _block(errors, "gsm8k", "must remain calibration_only")
    return errors


def audit_campaign(campaign: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> list[str]:
    """Audit campaign metadata without trusting its own readiness flags."""

    errors: list[str] = []
    if not isinstance(campaign, Mapping):
        return ["campaign: JSON object is required"]
    manifest_hash = campaign.get("manifest_hash", campaign.get("campaign_hash"))
    if not _sha256(manifest_hash):
        _block(errors, "campaign.manifest_hash", "canonical manifest hash is required")
    elif manifest_hash != sha256_json(_campaign_hash_payload(campaign)):
        _block(errors, "campaign.manifest_hash", "does not match canonical manifest contents")
    for name, expected in (("launches_any_job", False), ("dry_run_only", True), ("allocation_allowed", False)):
        if type(campaign.get(name)) is not bool or campaign.get(name) is not expected:
            _block(errors, f"campaign.{name}", f"must be typed {expected!r}")
    try:
        contract = load_contract() if contract is None else contract
        training, primary, domains, classified = _contract_suites(contract)
    except Exception as exc:
        return errors + [f"contract: unable to derive suite registry ({exc})"]
    if tuple(classified[: len(STRUCTURAL_HELD_OUT_SUITE_IDS)]) != STRUCTURAL_HELD_OUT_SUITE_IDS:
        _block(errors, "campaign.contract", "structural held-out IDs do not match the frozen six")
    if tuple(classified[len(STRUCTURAL_HELD_OUT_SUITE_IDS) :]) != PENDING_PRIMARY_EVAL_SUITE_IDS:
        _block(errors, "campaign.contract", "pending primary_eval IDs do not match the frozen eight")
    if campaign.get("training_suite_ids") != sorted(training) or campaign.get("training_suite_count") != 12:
        _block(errors, "campaign.training_suite_ids", "exact 12 training IDs are required")
    primary_ids = campaign.get("primary_eval_suite_ids", campaign.get("primary_evaluation_suite_ids"))
    if primary_ids != sorted(primary) or campaign.get("primary_eval_suite_count") != 14:
        _block(errors, "campaign.primary_eval_suite_ids", "exact 14 primary_eval IDs are required")
    if campaign.get("structural_held_out_suite_ids") != list(STRUCTURAL_HELD_OUT_SUITE_IDS):
        _block(errors, "campaign.structural_held_out_suite_ids", "must be the frozen six IDs")
    pending = campaign.get("primary_eval_not_designated_held_out_suite_ids", campaign.get("pending_primary_eval_suite_ids"))
    if pending != list(PENDING_PRIMARY_EVAL_SUITE_IDS):
        _block(errors, "campaign.pending_primary_eval_suite_ids", "must be the frozen eight pending IDs")
    if set(campaign.get("held_out_suite_ids", [])) != set(STRUCTURAL_HELD_OUT_SUITE_IDS):
        _block(errors, "campaign.held_out_suite_ids", "only the six structural IDs may be labeled held-out")
    budget = _mapping(campaign.get("budget_guard"))
    if budget is None:
        _block(errors, "campaign.budget_guard", "budget guard mapping is required")
    else:
        if type(budget.get("paid_jobs_may_launch")) is not bool:
            _block(errors, "campaign.budget_guard.paid_jobs_may_launch", "typed boolean is required")
        for field, expected_value in (("maximum_usd", EXACT_MAXIMUM_USD), ("operational_cap_usd", EXACT_OPERATIONAL_CAP_USD), ("safety_reserve_usd", EXACT_SAFETY_RESERVE_USD)):
            if _money(budget.get(field)) != expected_value:
                _block(errors, f"campaign.budget_guard.{field}", f"must be exactly {expected_value}")
    policy = _mapping(campaign.get("hf_publication_policy", campaign.get("hf_checkpoint_policy")))
    safe_rule = _mapping(policy.get("safe_public_artifact_rule")) if policy else None
    legacy_safe_rule = bool(
        safe_rule is not None
        and all(
            safe_rule.get(name) is True
            for name in (
                "public_requires_data_license_safe",
                "public_requires_quota_safe",
                "private_requires_private_artifact_safe",
            )
        )
    )
    current_safe_rule = bool(
        safe_rule is not None
        and all(
            safe_rule.get(name) is True
            for name in (
                "required",
                "public_only_when_quota_and_data_license_safe",
                "private_allowed_when_publication_is_not_safe",
            )
        )
    )
    if policy is None or set(policy.get("allowed_visibility", [])) != {"public", "private"} or policy.get("unique_repo_revision_url_required") is not True or not (legacy_safe_rule or current_safe_rule):
        _block(errors, "campaign.hf_publication_policy", "public/private safety policy is incomplete")
    if campaign.get("scientific_evidence_status", "not_established") != "not_established" or campaign.get("primary_eval_evidence_ready") is not False:
        _block(errors, "campaign.evidence", "provenance readiness must not become scientific evidence")
    claim_policy = _mapping(campaign.get("claim_policy"))
    portfolio_claim_allowed = (
        claim_policy.get("portfolio_evidence_claim_allowed", claim_policy.get("xlam_observation_claim_allowed"))
        if claim_policy
        else None
    )
    primary_eval_claim_allowed = (
        claim_policy.get("primary_eval_heldout_claim_allowed", claim_policy.get("held_out_suite_claim_allowed"))
        if claim_policy
        else None
    )
    pending_separation = bool(
        claim_policy
        and (
            claim_policy.get("pending_primary_eval_are_not_held_out") is True
            or claim_policy.get("primary_eval_suite_claim_requires_independent_receipts") is True
        )
    )
    if claim_policy is None or portfolio_claim_allowed is not False or primary_eval_claim_allowed is not False or claim_policy.get("gsm8k_role") != "calibration_only" or not pending_separation:
        _block(errors, "campaign.claim_policy", "claim separation and GSM8K calibration-only policy are required")
    halving = _mapping(campaign.get("successive_halving"))
    separation = _mapping(halving.get("held_out_separation")) if halving else None
    if separation is None or separation.get("must_be_disjoint") is not True or _mapping(separation.get("selection_split"),) is None or _mapping(separation.get("final_eval_split"),) is None:
        _block(errors, "campaign.successive_halving", "sealed selection and final evaluation separation is required")
    return errors


def audit_live_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Audit a live xLAM receipt independently of its production validator."""

    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["live: JSON object is required"]
    if not _sha256(receipt.get("receipt_hash")):
        _block(errors, "live.receipt_hash", "canonical receipt hash is required")
    elif receipt.get("receipt_hash") != sha256_json(_receipt_hash_payload(receipt)):
        _block(errors, "live.receipt_hash", "does not match canonical receipt contents")
    for name, expected in (("launchable", False), ("allocation_allowed", False), ("provenance_ready", True)):
        if type(receipt.get(name)) is not bool or receipt.get(name) is not expected:
            _block(errors, f"live.{name}", f"must be typed {expected!r}")
    if receipt.get("scientific_evidence_status") != "not_established":
        _block(errors, "live.scientific_evidence_status", "must remain not_established")
    claims = _mapping(receipt.get("claims"))
    evidence = _mapping(receipt.get("evidence"))
    if claims is None or claims.get("xlam_component_only") is not True or any(claims.get(name) is not False for name in ("portfolio_evidence", "primary_eval_heldout", "held_out", "company_usefulness")):
        _block(errors, "live.claims", "xLAM component-only claims must be explicit and false for promotion")
    if evidence is None or evidence.get("scope") != "xlam_component_only" or evidence.get("status") != "observed" or any(evidence.get(name) is not False for name in ("portfolio_evidence", "primary_eval_heldout", "company_usefulness")):
        _block(errors, "live.evidence", "observed xLAM component evidence must not be portfolio evidence")
    run = _mapping(receipt.get("run"))
    if run is None or str(run.get("component", "")).lower() != "xlam" or run.get("paid") is not True or str(_first(run, ("status", "state")) or "").lower() not in _SUCCESS_STATES or not _nonempty(_first(run, ("run_id", "component_run_id"))):
        _block(errors, "live.run", "paid successful xLAM run identity is required")
    for section, revision_names in (("model", ("revision", "model_revision")), ("dataset", ("revision", "dataset_revision", "source_revision"))):
        record = _mapping(receipt.get(section))
        if record is None or not _nonempty(_first(record, ("model_id", "dataset_id", "name", "source"))) or not _immutable(_first(record, revision_names)) or not _sha256(_first(record, ("receipt_hash", "revision_receipt_hash"))):
            _block(errors, f"live.{section}", "immutable revision and revision receipt hash are required")
    _audit_wandb(receipt.get("wandb", receipt.get("wandb_run")), "live.wandb", errors)
    _audit_tinker(receipt.get("tinker"), "live.tinker", errors)
    _audit_hf(receipt.get("sampler_checkpoints", receipt.get("checkpoints")), "live.sampler_checkpoints", errors)
    budget = _mapping(receipt.get("budget"))
    if budget is None:
        _block(errors, "live.budget", "budget is required")
    else:
        if budget.get("authorized") is not True or _money(budget.get("authorized_cap_usd", budget.get("maximum_usd"))) != EXACT_MAXIMUM_USD or _money(budget.get("operational_cap_usd")) != EXACT_OPERATIONAL_CAP_USD or _money(budget.get("safety_reserve_usd")) != EXACT_SAFETY_RESERVE_USD:
            _block(errors, "live.budget", "typed authorization and exact caps are required")
        debits = budget.get("debits")
        total = Decimal("0")
        if not isinstance(debits, list) or not debits:
            _block(errors, "live.budget.debits", "debit list is required")
        else:
            for index, debit in enumerate(debits):
                record = _mapping(debit)
                amount = _money(record.get("amount_usd", record.get("amount"))) if record else None
                if record is None or not (_immutable(record.get("debit_id")) or _sha256(record.get("debit_id"))) or amount is None or amount <= 0 or str(record.get("status", record.get("state", ""))).lower() not in _DEBIT_STATES or not _sha256(_first(record, ("receipt_hash", "debit_hash"))):
                    _block(errors, f"live.budget.debits[{index}]", "immutable exact debit receipt is required")
                if amount is not None and amount > 0:
                    total += amount
            if _money(budget.get("total_debited_usd", budget.get("total_usd"))) != total or _money(budget.get("remaining_usd", budget.get("remaining"))) != EXACT_MAXIMUM_USD - total:
                _block(errors, "live.budget", "reported totals do not match exact debit arithmetic")
    evaluator = _mapping(receipt.get("evaluator_provenance", receipt.get("evaluator")))
    if evaluator is None or str(evaluator.get("status", "")).lower() not in {"verified", "complete", "admissible"} or not _immutable(evaluator.get("revision")) or not _immutable(evaluator.get("dataset_revision")) or not all(_sha256(evaluator.get(name)) for name in ("split_manifest_hash", "task_id_hash", "verifier_hash", "container_digest", "provenance_hash")) or not _immutable(evaluator.get("receipt_id")) or evaluator.get("primary_eval_heldout") is True or evaluator.get("held_out") is True:
        _block(errors, "live.evaluator_provenance", "verified evaluator provenance is incomplete or overclaims held-out evidence")
    return errors


def audit_cross_bindings(
    bundle: Mapping[str, Any],
    campaign: Mapping[str, Any],
    live_receipts: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[str]:
    """Verify the explicit cryptographic binding between all artifact classes."""

    errors: list[str] = []
    if not isinstance(bindings, Mapping):
        return ["bindings: mapping with canonical cross-binding hash is required"]
    bundle_hash = bundle.get("bundle_hash") if isinstance(bundle, Mapping) else None
    campaign_hash = (
        campaign.get("manifest_hash", campaign.get("campaign_hash"))
        if isinstance(campaign, Mapping)
        else None
    )
    live_hashes = {
        str(receipt.get("run", {}).get("run_id", index)): receipt.get("receipt_hash")
        for index, receipt in enumerate(live_receipts)
        if isinstance(receipt, Mapping)
    }
    if not _sha256(bundle_hash) or bindings.get("bundle_hash") != bundle_hash:
        _block(errors, "bindings.bundle_hash", "must equal the verified bundle hash")
    if not _sha256(campaign_hash) or bindings.get("campaign_hash") != campaign_hash:
        _block(errors, "bindings.campaign_hash", "must equal the verified campaign hash")
    if bindings.get("live_receipt_hashes") != live_hashes:
        _block(errors, "bindings.live_receipt_hashes", "must exactly map every live receipt hash")
    if isinstance(campaign, Mapping) and campaign.get("receipt_bundle_hash") != bundle_hash:
        _block(errors, "campaign.receipt_bundle_hash", "must bind campaign to bundle hash")
    if isinstance(bundle, Mapping) and bundle.get("campaign_hash") != campaign_hash:
        _block(errors, "bundle.campaign_hash", "must bind bundle to campaign hash")
    for index, receipt in enumerate(live_receipts):
        if receipt.get("bundle_hash") != bundle_hash or receipt.get("campaign_hash") != campaign_hash:
            _block(errors, f"live[{index}].binding", "live receipt must bind bundle and campaign hashes")
    payload = {
        "bundle_hash": bundle_hash,
        "campaign_hash": campaign_hash,
        "live_receipt_hashes": dict(sorted(live_hashes.items())),
    }
    if not _sha256(bindings.get("cross_binding_hash")) or bindings.get("cross_binding_hash") != sha256_json(payload):
        _block(errors, "bindings.cross_binding_hash", "does not match canonical cross-artifact binding")
    return errors


def audit_receipt_set(
    bundle: Mapping[str, Any],
    campaign: Mapping[str, Any],
    live_receipts: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit campaign, bundle, live receipts, and their cryptographic binding."""

    if isinstance(live_receipts, Mapping):
        live_receipts = [live_receipts]
    else:
        live_receipts = list(live_receipts)
    blockers = audit_bundle(bundle, contract)
    blockers.extend(audit_campaign(campaign, contract))
    for index, receipt in enumerate(live_receipts):
        blockers.extend(f"live[{index}].{error}" for error in audit_live_receipt(receipt))
    blockers.extend(audit_cross_bindings(bundle, campaign, live_receipts, bindings))
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "canonical_hashes": {
            "bundle_hash": bundle.get("bundle_hash") if isinstance(bundle, Mapping) else None,
            "campaign_hash": (
                campaign.get("manifest_hash", campaign.get("campaign_hash"))
                if isinstance(campaign, Mapping)
                else None
            ),
            "live_receipt_hashes": {
                str(index): receipt.get("receipt_hash")
                for index, receipt in enumerate(live_receipts)
                if isinstance(receipt, Mapping)
            },
        },
        "launches_any_job": False,
    }


def audit_receipts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return audit_receipt_set(*args, **kwargs)


def audit_receipt_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return audit_receipt_set(*args, **kwargs)


def validate_audit(*args: Any, **kwargs: Any) -> list[str]:
    return audit_receipt_set(*args, **kwargs)["blockers"]


audit_receipt_bundle_artifacts = audit_bundle
audit_campaign_manifest = audit_campaign
audit_live_run_receipt = audit_live_receipt
validate_bundle_audit = audit_bundle


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--audit", dest="input_path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    payload = _load_json(args.input_path)
    report = audit_receipt_set(
        payload.get("bundle", payload.get("receipt_bundle", {})),
        payload.get("campaign", {}),
        payload.get("live_receipts", payload.get("live", [])),
        payload.get("bindings", {}),
        load_contract(args.contract),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
