#!/usr/bin/env python3
"""Validate a local receipt for a paid xLAM component run.

This module only reads a JSON receipt supplied by the caller.  It deliberately
does not import W&B, Hugging Face, Tinker, or a network client: an actual run
must be proven by the immutable identities and provider receipts recorded in
the JSON.  A passing receipt is observed component evidence, never portfolio
or primary-evaluation held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .pavlovs_domain_contract import load_contract
except ImportError:  # Direct execution from the flagship directory.
    from pavlovs_domain_contract import load_contract


SCHEMA_VERSION = "pavlov-live-xlam-run-receipt-v1"
RECEIPT_TYPE = "paid_xlam_component_run"
CHECKPOINT_STAGES = frozenset({"initial", "periodic", "final"})
SUCCESS_STATES = frozenset({"success", "succeeded", "finished", "completed", "complete"})
DEBIT_STATES = frozenset({"settled", "charged", "recorded", "succeeded", "complete"})
EXACT_MAXIMUM_USD = Decimal("18.00")
EXACT_OPERATIONAL_CAP_USD = Decimal("16.50")
EXACT_SAFETY_RESERVE_USD = Decimal("1.50")
_PLACEHOLDER_WORDS = frozenset(
    {
        "",
        "none",
        "null",
        "missing",
        "pending",
        "placeholder",
        "todo",
        "unset",
        "unknown",
        "main",
        "master",
        "latest",
        "receipt",
    }
)
_REVISION_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_COMMIT_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_DECIMAL_RE = re.compile(r"^[0-9]+(?:\.[0-9]{1,2})?$")


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for optional hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_WORDS


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and not _is_placeholder(value) and bool(value.strip())


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and not _is_placeholder(mapping[name]):
            return mapping[name]
    return None


def _https_url(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"https://[^\s]+", value.strip()):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def _hosted_url(value: Any, host: str) -> bool:
    return _https_url(value) and urlparse(str(value).strip()).netloc.lower() == host


def _immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and bool(_REVISION_RE.fullmatch(value.strip()))


def _commit_40(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_40_RE.fullmatch(value.strip()))


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip()))


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and (not math.isfinite(float(value))):
        return None
    text = str(value).strip()
    if not _DECIMAL_RE.fullmatch(text):
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    return amount if amount.is_finite() else None


def _amount(mapping: Mapping[str, Any], names: Sequence[str]) -> Decimal | None:
    for name in names:
        if name in mapping:
            return _decimal(mapping[name])
    return None


def _error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _validate_scope(receipt: Mapping[str, Any], errors: list[str]) -> None:
    if receipt.get("schema_version") != SCHEMA_VERSION:
        _error(errors, "schema_version", "unsupported or missing")
    if receipt.get("receipt_type") != RECEIPT_TYPE:
        _error(errors, "receipt_type", f"must be {RECEIPT_TYPE!r}")
    if type(receipt.get("launchable")) is not bool or receipt.get("launchable") is not False:
        _error(errors, "launchable", "must be false for a receipt validator")
    if type(receipt.get("allocation_allowed")) is not bool or receipt.get("allocation_allowed") is not False:
        _error(errors, "allocation_allowed", "must be false")
    if type(receipt.get("provenance_ready")) is not bool:
        _error(errors, "provenance_ready", "must be a typed boolean")
    elif receipt.get("provenance_ready") is not True:
        _error(errors, "provenance_ready", "must be true only for a fully validated receipt")
    if receipt.get("scientific_evidence_status") != "not_established":
        _error(errors, "scientific_evidence_status", "must remain not_established")

    claims = _mapping(receipt.get("claims"))
    evidence = _mapping(receipt.get("evidence"))
    if claims is None:
        _error(errors, "claims", "explicit component-only claim policy is required")
    else:
        if claims.get("xlam_component_only") is not True:
            _error(errors, "claims.xlam_component_only", "must be true")
        for name in ("portfolio_evidence", "primary_eval_heldout", "held_out", "company_usefulness"):
            if claims.get(name) is not False:
                _error(errors, f"claims.{name}", "must be false; xLAM cannot be promoted")
    if evidence is None:
        _error(errors, "evidence", "component-only evidence policy is required")
    else:
        if evidence.get("scope") != "xlam_component_only":
            _error(errors, "evidence.scope", "must remain xlam_component_only")
        if evidence.get("status") != "observed":
            _error(errors, "evidence.status", "must be observed, not portfolio evidence")
        for name in ("portfolio_evidence", "primary_eval_heldout", "company_usefulness"):
            if evidence.get(name) is not False:
                _error(errors, f"evidence.{name}", "must be false")

    # Reject common alternate claim spellings as well.  A caller cannot evade
    # the scope gate by putting a promotion claim outside the canonical map.
    for name in (
        "portfolio_evidence",
        "portfolio_ready",
        "primary_eval_heldout",
        "heldout_claim",
        "held_out_claim",
        "company_usefulness",
    ):
        if name in receipt and receipt[name] is not False:
            _error(errors, name, "must be false or absent")


def _validate_run(receipt: Mapping[str, Any], errors: list[str]) -> None:
    run = _mapping(receipt.get("run"))
    if run is None:
        _error(errors, "run", "paid xLAM run identity is missing")
        return
    if str(run.get("component", "")).strip().lower() != "xlam":
        _error(errors, "run.component", "must identify xLAM")
    if run.get("paid") is not True:
        _error(errors, "run.paid", "must be true for an actual paid component run")
    status = str(_first(run, ("status", "state", "run_status")) or "").strip().lower()
    if status not in SUCCESS_STATES:
        _error(errors, "run.status", "must be a successful terminal state")
    run_id = _first(run, ("run_id", "component_run_id"))
    if not _nonempty_text(run_id):
        _error(errors, "run.run_id", "non-placeholder component run ID is required")


def _validate_model_dataset(receipt: Mapping[str, Any], errors: list[str]) -> None:
    model = _mapping(receipt.get("model"))
    if model is None:
        _error(errors, "model", "model identity is missing")
    else:
        if not _nonempty_text(_first(model, ("model_id", "name", "repo"))):
            _error(errors, "model.model_id", "model identity is missing")
        if not _immutable_revision(_first(model, ("revision", "model_revision"))):
            _error(errors, "model.revision", "immutable 40/64-hex revision is required")
        if not _sha256_digest(_first(model, ("receipt_hash", "revision_receipt_hash"))):
            _error(errors, "model.receipt_hash", "immutable revision receipt hash is required")

    dataset = _mapping(receipt.get("dataset"))
    if dataset is None:
        _error(errors, "dataset", "dataset identity is missing")
    else:
        if not _nonempty_text(_first(dataset, ("dataset_id", "name", "source"))):
            _error(errors, "dataset.dataset_id", "dataset identity is missing")
        if not _immutable_revision(_first(dataset, ("revision", "dataset_revision", "source_revision"))):
            _error(errors, "dataset.revision", "immutable 40/64-hex revision is required")
        if not _sha256_digest(_first(dataset, ("receipt_hash", "revision_receipt_hash"))):
            _error(errors, "dataset.receipt_hash", "immutable revision receipt hash is required")


def _validate_wandb(receipt: Mapping[str, Any], errors: list[str]) -> None:
    wandb = _mapping(receipt.get("wandb") or receipt.get("wandb_run"))
    if wandb is None:
        _error(errors, "wandb", "online W&B receipt is missing")
        return
    if wandb.get("online") is not True:
        _error(errors, "wandb.online", "must be true")
    run_id = _first(wandb, ("run_id", "id"))
    if not _nonempty_text(run_id):
        _error(errors, "wandb.run_id", "non-placeholder run ID is required")
    run_url = _first(wandb, ("run_url", "url"))
    if not _hosted_url(run_url, "wandb.ai"):
        _error(errors, "wandb.run_url", "verified HTTPS W&B URL is required")
    elif _nonempty_text(run_id) and not str(run_url).rstrip("/").endswith(str(run_id)):
        _error(errors, "wandb.run_url", "URL must identify the recorded run_id")
    state = str(_first(wandb, ("state", "status", "run_state")) or "").strip().lower()
    success = wandb.get("success") is True or state in {"success", "succeeded"}
    if state not in SUCCESS_STATES or not success:
        _error(errors, "wandb.success_state", "finished/success state is required")
    if not _sha256_digest(_first(wandb, ("receipt_hash", "run_identity_hash"))):
        _error(errors, "wandb.receipt_hash", "immutable W&B identity receipt hash is required")


def _validate_tinker(receipt: Mapping[str, Any], errors: list[str]) -> None:
    tinker = _mapping(receipt.get("tinker"))
    if tinker is None:
        _error(errors, "tinker", "Tinker receipt object is required")
        return
    if str(tinker.get("provider", "")).strip().lower() != "tinker":
        _error(errors, "tinker.provider", "must be Tinker")
    run_id = _first(tinker, ("run_id", "id"))
    if not _nonempty_text(run_id):
        _error(errors, "tinker.run_id", "Tinker run ID is required")
    cost_status = str(_first(tinker, ("cost_status", "status")) or "").strip().lower()
    if cost_status not in {"authorized", "settled", "charged", "complete", "observed"}:
        _error(errors, "tinker.cost_status", "a recorded Tinker cost status is required")
    if not _sha256_digest(_first(tinker, ("receipt_hash", "run_identity_hash"))):
        _error(errors, "tinker.receipt_hash", "immutable Tinker identity receipt hash is required")


def _validate_checkpoints(receipt: Mapping[str, Any], errors: list[str]) -> None:
    checkpoints = receipt.get("sampler_checkpoints", receipt.get("checkpoints"))
    if not isinstance(checkpoints, list) or not checkpoints:
        _error(errors, "sampler_checkpoints", "initial/periodic/final checkpoint list is required")
        return
    seen_repo_revisions: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    stages: set[str] = set()
    for index, checkpoint_value in enumerate(checkpoints):
        path = f"sampler_checkpoints[{index}]"
        checkpoint = _mapping(checkpoint_value)
        if checkpoint is None:
            _error(errors, path, "must be an object")
            continue
        raw_stage = str(checkpoint.get("stage", checkpoint.get("kind", ""))).strip().lower()
        stage = "periodic" if raw_stage.startswith("periodic") else raw_stage
        if stage not in CHECKPOINT_STAGES:
            _error(errors, f"{path}.stage", "must be initial, periodic, or final")
        else:
            stages.add(stage)
        repo_url = _first(checkpoint, ("repo_url", "repo", "repository"))
        if not _https_url(repo_url) or urlparse(str(repo_url)).netloc.lower() != "huggingface.co":
            _error(errors, f"{path}.repo_url", "HTTPS Hugging Face repository URL is required")
        revision = _first(checkpoint, ("revision", "commit", "sha"))
        if not _commit_40(revision):
            _error(errors, f"{path}.revision", "exact 40-hex commit is required")
        url = _first(checkpoint, ("url", "checkpoint_url", "verified_url"))
        if not _https_url(url):
            _error(errors, f"{path}.url", "verified HTTPS checkpoint URL is required")
        elif _https_url(repo_url) and not str(url).startswith(str(repo_url).rstrip("/") + "/"):
            _error(errors, f"{path}.url", "URL must be under repo_url")
        elif _commit_40(revision) and revision not in str(url):
            _error(errors, f"{path}.url", "URL must contain the immutable commit")
        if checkpoint.get("url_verified", checkpoint.get("verified")) is not True:
            _error(errors, f"{path}.url_verified", "must be true")
        visibility = checkpoint.get("visibility")
        safe_public_artifact = checkpoint.get("safe_public_artifact")
        if visibility not in {"public", "private"}:
            _error(errors, f"{path}.visibility", "must be public or private per checkpoint")
        if type(safe_public_artifact) is not bool:
            _error(errors, f"{path}.safe_public_artifact", "must be a typed boolean")
        elif visibility == "public":
            if safe_public_artifact is not True:
                _error(errors, f"{path}.safe_public_artifact", "public artifact must be safe")
            if checkpoint.get("data_license_safe") is not True:
                _error(errors, f"{path}.data_license_safe", "public artifact requires data-license safety")
            if checkpoint.get("quota_safe") is not True:
                _error(errors, f"{path}.quota_safe", "public artifact requires quota safety")
        elif safe_public_artifact is not False or checkpoint.get("private_artifact_safe") is not True:
            _error(errors, f"{path}.private_artifact_safe", "private artifact safety receipt is required")
        if not _sha256_digest(_first(checkpoint, ("receipt_hash", "publication_hash"))):
            _error(errors, f"{path}.receipt_hash", "immutable HF publication receipt hash is required")
        if _https_url(repo_url) and _commit_40(revision):
            identity = (str(repo_url).rstrip("/"), str(revision).lower())
            if identity in seen_repo_revisions:
                _error(errors, path, "duplicate HF repo+commit identity")
            seen_repo_revisions.add(identity)
        if _https_url(url):
            if str(url) in seen_urls:
                _error(errors, f"{path}.url", "duplicate checkpoint URL")
            seen_urls.add(str(url))
    missing_stages = sorted(CHECKPOINT_STAGES - stages)
    if missing_stages:
        _error(errors, "sampler_checkpoints", "missing stages: " + ", ".join(missing_stages))


def _validate_budget(receipt: Mapping[str, Any], errors: list[str], contract: Mapping[str, Any] | None) -> None:
    budget = _mapping(receipt.get("budget"))
    if budget is None:
        _error(errors, "budget", "exact budget debit receipt is missing")
        return
    if str(budget.get("currency", "")).upper() != "USD":
        _error(errors, "budget.currency", "must be USD")
    cap = _amount(budget, ("authorized_cap_usd", "maximum_usd", "cap_usd"))
    operational_cap = _amount(budget, ("operational_cap_usd",))
    safety_reserve = _amount(budget, ("safety_reserve_usd",))
    if cap != EXACT_MAXIMUM_USD:
        _error(errors, "budget.authorized_cap_usd", "must be exactly 18.00 USD")
    if operational_cap != EXACT_OPERATIONAL_CAP_USD:
        _error(errors, "budget.operational_cap_usd", "must be exactly 16.50 USD")
    if safety_reserve != EXACT_SAFETY_RESERVE_USD:
        _error(errors, "budget.safety_reserve_usd", "must be exactly 1.50 USD")
    if (
        cap is not None
        and operational_cap is not None
        and safety_reserve is not None
        and operational_cap + safety_reserve != cap
    ):
        _error(errors, "budget", "operational cap plus safety reserve must equal 18.00")
    if budget.get("authorized") is not True:
        _error(errors, "budget.authorized", "must be a typed true authorization")
    if not _immutable_revision(_first(budget, ("authorization_id", "receipt_id"))):
        _error(errors, "budget.authorization_id", "immutable authorization identity is required")
    if not _sha256_digest(_first(budget, ("authorization_hash", "receipt_hash"))):
        _error(errors, "budget.authorization_hash", "immutable authorization receipt hash is required")
    if contract is not None:
        contract_budget = _mapping(contract.get("budget_gate"))
        contract_max = _amount(contract_budget or {}, ("maximum_usd",))
        if cap is not None and contract_max is not None and cap > contract_max:
            _error(errors, "budget.authorized_cap_usd", "exceeds contract maximum_usd")

    debits = budget.get("debits")
    if not isinstance(debits, list) or not debits:
        _error(errors, "budget.debits", "at least one exact debit receipt is required")
        return
    total = Decimal("0")
    debit_ids: set[str] = set()
    for index, debit_value in enumerate(debits):
        path = f"budget.debits[{index}]"
        debit = _mapping(debit_value)
        if debit is None:
            _error(errors, path, "must include debit_id, amount_usd, and settled status")
            continue
        debit_id = _first(debit, ("debit_id", "id", "receipt_id"))
        if not (_immutable_revision(debit_id) or _sha256_digest(debit_id)):
            _error(errors, f"{path}.debit_id", "immutable debit identity is required")
        elif str(debit_id) in debit_ids:
            _error(errors, f"{path}.debit_id", "duplicate debit identity")
        else:
            debit_ids.add(str(debit_id))
        amount = _amount(debit, ("amount_usd", "amount"))
        if amount is None or amount <= 0:
            _error(errors, f"{path}.amount_usd", "positive exact decimal amount is required")
        else:
            total += amount
        state = str(debit.get("status", debit.get("state", ""))).strip().lower()
        if state not in DEBIT_STATES:
            _error(errors, f"{path}.status", "must be settled/charged/recorded")
        if not _sha256_digest(_first(debit, ("receipt_hash", "debit_hash"))):
            _error(errors, f"{path}.receipt_hash", "immutable debit receipt hash is required")

    reported_total = _amount(budget, ("total_debited_usd", "total_usd", "debited_usd"))
    if reported_total is None or reported_total != total:
        _error(errors, "budget.total_debited_usd", "must exactly equal the sum of debit amounts")
    remaining = _amount(budget, ("remaining_usd", "remaining"))
    if cap is None or remaining is None or remaining != cap - total:
        _error(errors, "budget.remaining_usd", "must exactly equal cap minus debits")
    if cap is not None and total > cap:
        _error(errors, "budget.debits", "total debits exceed authorized cap")


def _validate_evaluator(receipt: Mapping[str, Any], errors: list[str]) -> None:
    evaluator = _mapping(receipt.get("evaluator_provenance", receipt.get("evaluator")))
    if evaluator is None:
        _error(errors, "evaluator_provenance", "verified evaluator provenance is required")
        return
    status = str(evaluator.get("status", "")).strip().lower()
    if status not in {"verified", "complete", "admissible"}:
        _error(errors, "evaluator_provenance.status", "must be verified")
    if not _nonempty_text(_first(evaluator, ("evaluator_id", "name", "id"))):
        _error(errors, "evaluator_provenance.evaluator_id", "evaluator identity is required")
    if not _immutable_revision(_first(evaluator, ("revision", "evaluator_revision"))):
        _error(errors, "evaluator_provenance.revision", "immutable evaluator revision is required")
    if not _immutable_revision(_first(evaluator, ("dataset_revision", "source_revision"))):
        _error(errors, "evaluator_provenance.dataset_revision", "immutable evaluator dataset revision is required")
    for name in ("split_manifest_hash", "task_id_hash", "task_id_hashes", "verifier_hash"):
        if name in {"task_id_hashes"} and name not in evaluator:
            continue
        if not _sha256_digest(evaluator.get(name)):
            _error(errors, f"evaluator_provenance.{name}", "64-hex SHA-256 receipt is required")
    if not _sha256_digest(evaluator.get("container_digest")):
        _error(errors, "evaluator_provenance.container_digest", "64-hex container digest is required")
    if not _nonempty_text(_first(evaluator, ("receipt_id", "provenance_id"))):
        _error(errors, "evaluator_provenance.receipt_id", "provenance receipt identity is required")
    if not _immutable_revision(_first(evaluator, ("receipt_id", "provenance_id"))):
        _error(errors, "evaluator_provenance.receipt_id", "immutable provenance receipt identity is required")
    if not _sha256_digest(evaluator.get("provenance_hash")):
        _error(errors, "evaluator_provenance.provenance_hash", "immutable provenance hash is required")
    if evaluator.get("primary_eval_heldout") is True or evaluator.get("held_out") is True:
        _error(errors, "evaluator_provenance", "held-out claims require a separate proven primary_eval receipt")


def _receipt_hash_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_hash"}


def validate_live_run_receipt(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return exact blockers; an empty list means observed component receipt is valid."""

    if not isinstance(receipt, Mapping):
        return ["receipt must be a JSON object"]
    errors: list[str] = []
    _validate_scope(receipt, errors)
    _validate_run(receipt, errors)
    _validate_model_dataset(receipt, errors)
    _validate_wandb(receipt, errors)
    _validate_tinker(receipt, errors)
    _validate_checkpoints(receipt, errors)
    _validate_budget(receipt, errors, contract)
    _validate_evaluator(receipt, errors)
    if not _sha256_digest(receipt.get("receipt_hash")):
        _error(errors, "receipt_hash", "required canonical 64-hex SHA-256 bundle hash")
    elif receipt.get("receipt_hash") != sha256_json(_receipt_hash_payload(receipt)):
        _error(errors, "receipt_hash", "does not match canonical receipt contents")
    if errors and receipt.get("provenance_ready") is True:
        _error(errors, "provenance_ready", "cannot be true while receipt blockers remain")
    return errors


def validate_receipt(
    receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> list[str]:
    """Compatibility alias for callers using the shorter validator name."""

    return validate_live_run_receipt(receipt, contract)


def is_valid_live_run_receipt(
    receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> bool:
    return not validate_live_run_receipt(receipt, contract)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", "--input", "--validate", dest="receipt_path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = _load_json(args.receipt_path)
    contract = load_contract(args.contract) if args.contract is not None else None
    errors = validate_live_run_receipt(receipt, contract)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("VALID: observed xLAM component-only receipt; no portfolio or held-out claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
