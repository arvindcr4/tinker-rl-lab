#!/usr/bin/env python3
"""Build a deterministic, non-launching Pavlov domain campaign manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .pavlovs_domain_contract import CONTRACT_PATH, load_contract, validate_contract
except ImportError:  # Direct execution from the flagship directory.
    from pavlovs_domain_contract import CONTRACT_PATH, load_contract, validate_contract


EXPECTED_TRAINING_SUITE_COUNT = 12
EXPECTED_PRIMARY_EVAL_SUITE_COUNT = 14
EXPECTED_STRUCTURAL_HELD_OUT_SUITE_COUNT = 6
# Backwards-compatible name for callers that imported the old count constant;
# held-out is now reserved for the six suites whose contract split text
# independently says held-out/private.
EXPECTED_HELD_OUT_SUITE_COUNT = EXPECTED_STRUCTURAL_HELD_OUT_SUITE_COUNT
LR_ARMS: tuple[tuple[str, float], ...] = (
    ("lr-1e-5", 1e-5),
    ("lr-2e-5", 2e-5),
    ("lr-4e-5", 4e-5),
)

# These are deliberately receipt names rather than configuration fields.  A
# contract can describe a budget or a W&B namespace without proving that the
# corresponding immutable receipt exists.
REQUIRED_RECEIPTS: tuple[tuple[str, str], ...] = (
    ("immutable_split", "immutable train/eval split"),
    ("dataset_revision", "dataset revision"),
    ("license", "dataset and benchmark license"),
    ("split_manifest_hash", "split manifest and disjoint task-ID hash"),
    ("container_digest", "container/environment digest"),
    ("decontamination", "decontamination"),
    ("budget", "budget authorization"),
    ("wandb_online", "online W&B initialization"),
    ("hf_publication", "Hugging Face publication for every checkpoint"),
    ("verifier", "verifier revision/digest"),
    ("model_revision", "model revision"),
)

RECEIPT_ALIASES: dict[str, tuple[str, ...]] = {
    "immutable_split": (
        "immutable_split",
        "immutable_train_eval_split",
        "train_eval_split",
    ),
    "dataset_revision": (
        "dataset_revision",
        "dataset_revisions",
        "dataset_revision_receipt",
    ),
    "license": (
        "license",
        "licenses",
        "license_receipt",
        "license_signoff",
    ),
    "split_manifest_hash": (
        "split_manifest_hash",
        "split_manifest",
        "task_id_hashes",
        "task_id_hash",
        "split",
    ),
    "container_digest": (
        "container_digest",
        "environment_digest",
        "container",
    ),
    "decontamination": (
        "decontamination",
        "decontamination_receipt",
        "contamination",
    ),
    "budget": (
        "budget",
        "budget_receipt",
        "budget_authorization",
    ),
    "wandb_online": (
        "wandb_online",
        "online_wandb",
        "wandb",
        "wandb_receipt",
    ),
    "hf_publication": (
        "hf_publication",
        "huggingface_publication",
        "hugging_face_publication",
        "hf",
        "huggingface",
        "hugging_face",
    ),
    "verifier": (
        "verifier",
        "verifier_receipt",
        "verifier_revision",
        "verifier_digest",
    ),
    "model_revision": (
        "model_revision",
        "model_revisions",
        "model_revision_receipts",
    ),
}

_NEGATIVE_RECEIPT_STATUSES = {
    "blocked",
    "failed",
    "invalid",
    "missing",
    "pending",
    "rejected",
    "unset",
}

_PLACEHOLDER_RECEIPT_VALUES = {
    "",
    "none",
    "null",
    "missing",
    "pending",
    "placeholder",
    "receipt",
    "license-receipt",
    "decontamination-receipt",
    "budget-receipt",
    "to_be_pinned_before_paid_runs",
}
_IMMUTABLE_REVISION_RE = re.compile(
    r"^(?:sha256:)?[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$"
)
_SHA256_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_HTTPS_URL_RE = re.compile(r"^https://[^\s]+$")


def _non_placeholder_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip().lower() not in _PLACEHOLDER_RECEIPT_VALUES
    )


def _immutable_revision(value: Any) -> bool:
    return bool(
        _non_placeholder_text(value)
        and _IMMUTABLE_REVISION_RE.fullmatch(value.strip())
    )


def _sha256_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and _SHA256_DIGEST_RE.fullmatch(value.strip())
    )


def _https_url(value: Any) -> bool:
    return bool(isinstance(value, str) and _HTTPS_URL_RE.fullmatch(value.strip()))


def _receipt_field(value: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        candidate = value.get(name)
        if candidate is not None and not (
            isinstance(candidate, str)
            and candidate.strip().lower() in _PLACEHOLDER_RECEIPT_VALUES
        ):
            return candidate
    return None


def _license_receipt_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        receipt_id = _receipt_field(value, "receipt_id", "approval_id", "license_id")
        digest = _receipt_field(value, "sha256", "hash", "digest")
        approved = value.get("approved") is True or value.get("signoff") is True
        return bool(
            _non_placeholder_text(receipt_id)
            and (approved or _sha256_digest(digest) or _immutable_revision(receipt_id))
        )
    if not _non_placeholder_text(value):
        return False
    text = value.strip().lower()
    return bool(
        _immutable_revision(value)
        or _sha256_digest(value)
        or any(token in text for token in ("approved", "approval", "signoff", "cc-", "apache", "mit", "license:"))
    )


def _split_manifest_receipt_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        combined = _receipt_field(value, "split_task_id_hash", "split_manifest_task_id_hash")
        split_hash = _receipt_field(value, "split_manifest_hash", "split_hash")
        task_hash = _receipt_field(value, "task_id_hashes", "task_id_hash")
        if combined is not None:
            return _sha256_digest(combined)
        return _sha256_digest(split_hash) and _sha256_digest(task_hash)
    # A single SHA-256 value is accepted only as a documented composite of the
    # immutable split manifest and task-ID set; status/booleans are rejected.
    return _sha256_digest(value)


def _decontamination_receipt_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        status = value.get("status")
        if not isinstance(status, str) or status.strip().lower() not in {
            "verified",
            "complete",
            "completed",
            "clean",
            "passed",
        }:
            return False
        identity = _receipt_field(value, "receipt_id", "sha256", "hash", "digest")
        return _non_placeholder_text(identity) and (
            _immutable_revision(identity) or _sha256_digest(identity)
        )
    return _sha256_digest(value) or _immutable_revision(value)


def _budget_receipt_present(value: Any) -> bool:
    if isinstance(value, Mapping):
        identity = _receipt_field(value, "receipt_id", "authorization_id", "sha256", "hash")
        maximum = value.get("maximum_usd", value.get("max_usd"))
        authorized = value.get("authorized") is True or str(value.get("status", "")).strip().lower() in {
            "authorized",
            "approved",
        }
        return bool(
            _non_placeholder_text(identity)
            and (_immutable_revision(identity) or _sha256_digest(identity) or authorized)
            and isinstance(maximum, (int, float))
            and maximum > 0
        )
    return _sha256_digest(value) or _immutable_revision(value)


def _wandb_receipt_present(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("online") is not True:
        return False
    run_id = _receipt_field(value, "run_id")
    run_url = _receipt_field(value, "run_url")
    return bool(_non_placeholder_text(run_id) and _https_url(run_url))


def _hf_receipt_present(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return False
    seen: set[tuple[str, str, str]] = set()
    seen_repo_revisions: set[tuple[str, str]] = set()
    stages: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            return False
        repo_url = _receipt_field(checkpoint, "repo_url", "repo", "repository")
        revision = _receipt_field(checkpoint, "revision", "commit", "sha")
        checkpoint_url = _receipt_field(
            checkpoint, "url", "checkpoint_url", "repo_revision_url"
        )
        visibility = checkpoint.get("visibility")
        if not (
            _https_url(repo_url)
            and _immutable_revision(revision)
            and _https_url(checkpoint_url)
            and visibility in {"public", "private"}
            and checkpoint.get("safe_public_artifact") is True
        ):
            return False
        stage = str(checkpoint.get("stage", "")).strip().lower()
        if stage:
            stages.add(stage)
        identity = (str(repo_url), str(revision), str(checkpoint_url))
        repo_revision = (str(repo_url), str(revision))
        if identity in seen or repo_revision in seen_repo_revisions:
            return False
        seen.add(identity)
        seen_repo_revisions.add(repo_revision)
    # Initial, periodic, and final artifacts must each be independently
    # addressable.  Their visibility is chosen per artifact, never hardcoded.
    return {"initial", "periodic", "final"}.issubset(stages)


def _receipt_sources(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return non-secret receipt registries without inspecting environment state."""

    sources: list[Mapping[str, Any]] = []
    for key in (
        "receipts",
        "launch_receipts",
        "provenance_receipts",
        "receipt_registry",
        "tracking_receipts",
        "wandb",
        "hf_publication",
        "huggingface",
        "verifier",
        "model_revisions",
    ):
        value = contract.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    for key in ("tracking", "provenance"):
        value = contract.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    return sources


def _lookup_receipt(
    sources: Sequence[Mapping[str, Any]], receipt_name: str
) -> tuple[Any, str | None]:
    for source in sources:
        for alias in RECEIPT_ALIASES[receipt_name]:
            if alias in source:
                return source[alias], alias
    return None, None


def _receipt_present(value: Any, receipt_name: str) -> bool:
    """Recognise immutable receipts, never a status flag or config namespace."""

    if receipt_name == "wandb_online":
        return _wandb_receipt_present(value)
    if receipt_name == "hf_publication":
        return _hf_receipt_present(value)
    if receipt_name == "license":
        return _license_receipt_present(value)
    if receipt_name == "split_manifest_hash":
        return _split_manifest_receipt_present(value)
    if receipt_name == "decontamination":
        return _decontamination_receipt_present(value)
    if receipt_name == "budget":
        return _budget_receipt_present(value)
    if receipt_name in {"dataset_revision", "model_revision"}:
        if isinstance(value, Mapping):
            candidate = _receipt_field(
                value, "revision", "model_revision", "dataset_revision", "sha256", "hash", "digest"
            )
            return _immutable_revision(candidate) or _sha256_digest(candidate)
        return _immutable_revision(value) or _sha256_digest(value)
    if receipt_name in {"container_digest", "verifier"}:
        if isinstance(value, Mapping):
            candidate = _receipt_field(value, "digest", "sha256", "hash", "revision")
            return _sha256_digest(candidate)
        return _sha256_digest(value)
    if receipt_name == "immutable_split":
        if isinstance(value, Mapping):
            candidate = _receipt_field(value, "receipt_id", "sha256", "hash", "digest")
            return _immutable_revision(candidate) or _sha256_digest(candidate)
        return _immutable_revision(value) or _sha256_digest(value)
    return False


def _model_revision_status(
    contract: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, bool], list[str]]:
    """Require a receipt for every model candidate, keyed by its stable role."""

    raw, _ = _lookup_receipt(sources, "model_revision")
    models = contract.get("model_candidates", [])
    if not isinstance(models, list):
        return False, {}, ["model_revision"]

    per_model: dict[str, bool] = {}
    missing: list[str] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        role = str(model.get("role", model.get("model_id", "<missing>")))
        value: Any = raw
        if isinstance(raw, Mapping):
            value = raw.get(role, raw.get(str(model.get("model_id"))))
            # A mapping with an explicit global receipt can apply to all models.
            if value is None and any(
                key in raw for key in ("all", "receipt", "sha256", "status", "verified")
            ):
                value = raw
        present = _receipt_present(value, "model_revision")
        pinned = _immutable_revision(model.get("revision")) or _sha256_digest(
            model.get("revision")
        )
        per_model[role] = bool(present and pinned)
        if not per_model[role]:
            missing.append(f"model_revision:{role}")
    return not missing, per_model, missing


def _receipt_status(contract: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, str | None]]:
    sources = _receipt_sources(contract)
    status: dict[str, bool] = {}
    source_names: dict[str, str | None] = {}
    for receipt_name, _ in REQUIRED_RECEIPTS:
        value, source_name = _lookup_receipt(sources, receipt_name)
        status[receipt_name] = _receipt_present(value, receipt_name)
        source_names[receipt_name] = source_name

    # An explicit immutable split receipt wins.  Otherwise the receipt is only
    # complete when all of its immutable provenance components are recorded.
    if not status["immutable_split"]:
        status["immutable_split"] = all(
            status[name]
            for name in (
                "dataset_revision",
                "license",
                "split_manifest_hash",
                "container_digest",
                "decontamination",
            )
        )

    model_ok, _, _ = _model_revision_status(contract, sources)
    status["model_revision"] = model_ok
    return status, source_names


SUITE_PROVENANCE_RECEIPTS = (
    "dataset_revision",
    "license",
    "split_manifest_hash",
    "container_digest",
    "decontamination",
)


def _suite_receipt_status(
    contract: Mapping[str, Any], suite_ids: Sequence[str]
) -> tuple[dict[str, dict[str, bool]], list[str]]:
    """Check per-suite provenance records without treating role text as evidence."""

    sources = _receipt_sources(contract)
    suite_registries: list[Mapping[str, Any]] = []
    for source in sources:
        for key in (
            "suite_receipts",
            "primary_eval_suites",
            "held_out_suites",
            "training_suites",
            "suites",
        ):
            value = source.get(key)
            if isinstance(value, Mapping):
                suite_registries.append(value)

    statuses: dict[str, dict[str, bool]] = {}
    pending: list[str] = []
    for suite_id in suite_ids:
        local: dict[str, Any] = {}
        for registry in suite_registries:
            value = registry.get(suite_id)
            if isinstance(value, Mapping):
                local.update(value)
        local_sources: Sequence[Mapping[str, Any]] = [local]
        suite_status = {}
        for receipt_name in SUITE_PROVENANCE_RECEIPTS:
            value, _ = _lookup_receipt(local_sources, receipt_name)
            suite_status[receipt_name] = _receipt_present(value, receipt_name)
        suite_status["immutable_split"] = all(
            suite_status[receipt_name] for receipt_name in SUITE_PROVENANCE_RECEIPTS
        )
        statuses[suite_id] = suite_status
        if not suite_status["immutable_split"]:
            pending.append(suite_id)
    return statuses, pending


def _suite_registry(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate train/primary-eval coverage without overclaiming held-out status."""

    suites = contract.get("suite_registry")
    if not isinstance(suites, Mapping):
        raise ValueError("invalid Pavlov suite registry: suite_registry must be an object")

    training = {
        str(suite_id): suite
        for suite_id, suite in suites.items()
        if isinstance(suite, Mapping) and suite.get("role") == "train"
    }
    primary_eval = {
        str(suite_id): suite
        for suite_id, suite in suites.items()
        if isinstance(suite, Mapping) and suite.get("role") == "primary_eval"
    }
    structural_held_out = {
        suite_id: suite
        for suite_id, suite in primary_eval.items()
        if any(
            marker in str(suite.get("split", "")).lower()
            for marker in ("held-out", "private")
        )
    }
    pending_held_out = {
        suite_id: suite
        for suite_id, suite in primary_eval.items()
        if suite_id not in structural_held_out
    }
    errors: list[str] = []
    if len(training) != EXPECTED_TRAINING_SUITE_COUNT:
        errors.append(
            f"expected {EXPECTED_TRAINING_SUITE_COUNT} training suites, found {len(training)}"
        )
    if len(primary_eval) != EXPECTED_PRIMARY_EVAL_SUITE_COUNT:
        errors.append(
            "expected "
            f"{EXPECTED_PRIMARY_EVAL_SUITE_COUNT} primary_eval suites, found {len(primary_eval)}"
        )
    if len(structural_held_out) != EXPECTED_STRUCTURAL_HELD_OUT_SUITE_COUNT:
        errors.append(
            "expected "
            f"{EXPECTED_STRUCTURAL_HELD_OUT_SUITE_COUNT} suites with "
            "held-out/private split descriptions, "
            f"found {len(structural_held_out)}"
        )

    domains = {str(domain) for domain in contract.get("domains", [])}
    training_domains = {
        suite_id: sorted({str(domain) for domain in suite.get("domains", [])})
        for suite_id, suite in training.items()
    }
    primary_eval_domains = {
        suite_id: sorted({str(domain) for domain in suite.get("domains", [])})
        for suite_id, suite in primary_eval.items()
    }
    held_out_domains = {
        suite_id: sorted({str(domain) for domain in suite.get("domains", [])})
        for suite_id, suite in structural_held_out.items()
    }
    domain_training = {
        domain: sorted(
            suite_id
            for suite_id, suite_domains in training_domains.items()
            if domain in suite_domains
        )
        for domain in sorted(domains)
    }
    domain_primary_eval = {
        domain: sorted(
            suite_id
            for suite_id, suite_domains in primary_eval_domains.items()
            if domain in suite_domains
        )
        for domain in sorted(domains)
    }
    domain_held_out = {
        domain: sorted(
            suite_id
            for suite_id, suite_domains in held_out_domains.items()
            if domain in suite_domains
        )
        for domain in sorted(domains)
    }
    errors.extend(
        f"{domain}: no training suite in manifest"
        for domain, suite_ids in domain_training.items()
        if not suite_ids
    )
    errors.extend(
        f"{domain}: no primary_eval suite in manifest"
        for domain, suite_ids in domain_primary_eval.items()
        if not suite_ids
    )
    if errors:
        raise ValueError("invalid Pavlov suite registry: " + "; ".join(errors))

    def records(registry: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": suite_id,
                "role": suite.get("role"),
                "domains": sorted({str(domain) for domain in suite.get("domains", [])}),
                "split": suite.get("split"),
                "stateful": bool(suite.get("stateful")),
                "artifact_or_side_effect": bool(suite.get("artifact_or_side_effect")),
            }
            for suite_id, suite in sorted(registry.items())
        ]

    return {
        "training": training,
        "primary_eval": primary_eval,
        "structural_held_out": structural_held_out,
        "pending_held_out": pending_held_out,
        "training_ids": sorted(training),
        "primary_eval_ids": sorted(primary_eval),
        "structural_held_out_ids": sorted(structural_held_out),
        "pending_held_out_ids": sorted(pending_held_out),
        "training_domains": training_domains,
        "primary_eval_domains": primary_eval_domains,
        "held_out_domains": held_out_domains,
        "domain_training": domain_training,
        "domain_primary_eval": domain_primary_eval,
        "domain_held_out": domain_held_out,
        "training_records": records(training),
        "primary_eval_records": records(primary_eval),
        "held_out_records": records(structural_held_out),
    }


def _xlam_only_scope(contract: Mapping[str, Any]) -> bool:
    explicit = contract.get("xlam_only")
    if explicit is True:
        return True
    for key in ("experiment_scope", "claim_scope", "scope", "status"):
        value = contract.get(key)
        if isinstance(value, str) and "xlam" in value.lower() and "only" in value.lower():
            return True
    for key in ("claims", "non_claims", "objective"):
        value = contract.get(key)
        text = json.dumps(value, sort_keys=True).lower()
        if "xlam-only" in text or "xlam only" in text:
            return True
    return False


def _successive_halving_contract() -> dict[str, Any]:
    arms = [arm_id for arm_id, _ in LR_ARMS]
    gates = [
        {
            "id": "receipt_preflight",
            "requires": [
                "every arm has all required immutable provenance receipts",
                "online W&B initialization is verified before paid work",
                "every periodic and final checkpoint has an HF publication receipt",
            ],
            "on_failure": "BLOCKED; do not allocate a job",
        },
        {
            "id": "short_screening",
            "requires": [
                "all three LR arms use the identical immutable train split, seed, sampler, "
                "reward, and sealed selection slice",
                "each arm completes the short tracked screening horizon",
            ],
            "decision": "select by sealed selection slice perfect-call rate",
        },
        {
            "id": "tie_break",
            "requires": ["a tie at the primary metric"],
            "decision": "break ties by strict mean reward, then lower estimated cost",
        },
        {
            "id": "winner_extension",
            "requires": ["exactly one winner is selected and its receipts remain valid"],
            "decision": "extend only the winning arm with fixed-interval checkpoints",
        },
        {
            "id": "final_evaluation",
            "requires": [
                "the final evaluation split is immutable and disjoint from the sealed selection slice",
                "the final evaluation split is not consulted during arm selection",
            ],
            "decision": (
                "evaluate the selected arm once; never promote xLAM-only evidence "
                "to company usefulness"
            ),
        },
    ]
    return {
        "method": "successive_halving",
        "arms": arms,
        "screening_arm_ids": arms,
        "screening_learning_rates": [rate for _, rate in LR_ARMS],
        "short_screening_steps": 10,
        "checkpoint_steps": [5, 10],
        "selection_metric": "sealed selection slice perfect-call rate",
        "selection_rule": "maximize sealed selection slice perfect-call rate",
        "tie_breakers": ["strict mean reward", "lower estimated cost"],
        "winner_extension": "extend only the winning arm",
        "decision_gates": gates,
        "held_out_separation": {
            "selection_split": {
                "name": "sealed_selection_slice",
                "purpose": "select among the three screening arms",
                "consulted_during_selection": True,
                "receipt_required": True,
            },
            "final_eval_split": {
                "name": "primary_eval_final",
                "purpose": "final unbiased evaluation of the selected arm",
                "consulted_during_selection": False,
                "receipt_required": True,
            },
            "must_be_disjoint": True,
            "held_out_label_requires_independent_receipt": True,
        },
    }


def build_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    errors = validate_contract(contract)
    if errors:
        raise ValueError("invalid Pavlov domain contract: " + "; ".join(errors))

    registry = _suite_registry(contract)
    suites = contract["suite_registry"]
    training = registry["training"]
    primary_eval = registry["primary_eval"]
    company_eval_coverage: dict[str, list[str]] = {}
    company_train_coverage: dict[str, list[str]] = {}
    company_domain_coverage: dict[str, dict[str, dict[str, list[str]]]] = {}
    coverage_errors: list[str] = []
    for company in contract["companies"]:
        company_name = company["name"]
        required_domains = {str(domain) for domain in company["domains"]}
        inherited_eval = sorted(
            suite_id
            for suite_id, suite in primary_eval.items()
            if required_domains.intersection(suite["domains"])
        )
        inherited_train = sorted(
            suite_id
            for suite_id, suite in training.items()
            if required_domains.intersection(suite["domains"])
        )
        eval_by_domain = {
            domain: sorted(
                suite_id
                for suite_id, suite in primary_eval.items()
                if domain in set(suite["domains"])
            )
            for domain in sorted(required_domains)
        }
        train_by_domain = {
            domain: sorted(
                suite_id
                for suite_id, suite in training.items()
                if domain in set(suite["domains"])
            )
            for domain in sorted(required_domains)
        }
        missing_train = sorted(domain for domain, ids in train_by_domain.items() if not ids)
        missing_eval = sorted(domain for domain, ids in eval_by_domain.items() if not ids)
        if missing_train:
            coverage_errors.append(
                f"{company_name}: required domains without training coverage {missing_train}"
            )
        if missing_eval:
            coverage_errors.append(
                f"{company_name}: required domains without primary_eval coverage {missing_eval}"
            )
        company_train_coverage[company_name] = inherited_train
        company_eval_coverage[company_name] = inherited_eval
        company_domain_coverage[company_name] = {
            "training": train_by_domain,
            "primary_eval": eval_by_domain,
        }
    if coverage_errors:
        raise ValueError("invalid Pavlov company coverage: " + "; ".join(coverage_errors))

    blockers: list[str] = []
    budget = contract["budget_gate"]
    if not budget["paid_jobs_may_launch"]:
        blockers.append("paid jobs disabled pending explicit user budget cap")
    if not isinstance(budget.get("maximum_usd"), (int, float)) or budget["maximum_usd"] <= 0:
        blockers.append("maximum_usd is unset or not positive")
    if any(
        model["revision"] == "TO_BE_PINNED_BEFORE_PAID_RUNS"
        for model in contract["model_candidates"]
    ):
        blockers.append("model revisions are not pinned")

    receipt_status, receipt_sources = _receipt_status(contract)
    all_receipt_suite_ids = sorted(
        set(registry["training_ids"]) | set(registry["primary_eval_ids"])
    )
    suite_receipt_status, pending_suite_receipts = _suite_receipt_status(
        contract, all_receipt_suite_ids
    )
    pending_primary_eval_receipts = [
        suite_id
        for suite_id in registry["primary_eval_ids"]
        if suite_id in pending_suite_receipts
    ]
    pending_training_receipts = [
        suite_id
        for suite_id in registry["training_ids"]
        if suite_id in pending_suite_receipts
    ]
    missing_receipts = [
        receipt_name
        for receipt_name, _ in REQUIRED_RECEIPTS
        if not receipt_status[receipt_name]
    ]
    for receipt_name, description in REQUIRED_RECEIPTS:
        if not receipt_status[receipt_name]:
            blockers.append(f"missing {description} receipt")
    if not receipt_status["license"]:
        blockers.append("dataset and benchmark licenses require recorded sign-off")
    if not (
        receipt_status["dataset_revision"]
        and receipt_status["split_manifest_hash"]
    ):
        blockers.append("dataset revisions and disjoint task-ID hashes require freezing")
    if pending_suite_receipts:
        blockers.append(
            "training and primary_eval suites lack per-suite immutable provenance receipts: "
            + ", ".join(pending_suite_receipts)
        )

    contract_status = str(contract.get("status", "")).strip().lower()
    budget_status_is_authorized = contract_status in {
        "authorized",
        "authorized_tinker_only",
        "ready",
        "ready_to_run",
    }
    budget_gate_status = str(budget.get("status", "")).strip().upper()
    budget_gate_is_authorized = budget_gate_status in {
        "AUTHORIZED",
        "AUTHORIZED_TINKER_ONLY",
        "READY",
        "READY_TO_RUN",
    }
    budget_status_conflict = bool(budget.get("paid_jobs_may_launch")) and not (
        budget_status_is_authorized and budget_gate_is_authorized
    )
    if budget_status_conflict:
        blockers.append(
            "contract status "
            f"{contract.get('status')!r} and budget status {budget.get('status')!r} "
            "conflicts with paid_jobs_may_launch=true"
        )

    xlam_only = _xlam_only_scope(contract)
    if xlam_only:
        blockers.append("xLAM-only runs cannot launch or claim Pavlov company usefulness")
    primary_eval_evidence_ready = (
        not pending_suite_receipts
        and not missing_receipts
        and not budget_status_conflict
        and not xlam_only
    )
    structural_held_out_receipt_ready = (
        not any(
            suite_id in pending_suite_receipts
            for suite_id in registry["structural_held_out_ids"]
        )
        and not missing_receipts
        and not budget_status_conflict
        and not xlam_only
    )

    wandb_tracking = contract.get("wandb", {})
    if not isinstance(wandb_tracking, Mapping):
        wandb_tracking = {}
    wandb_config = {
        "entity": wandb_tracking.get("entity") or "arvindcr4-pes-university",
        "project": wandb_tracking.get("project") or "tinker-rl-lab-pavlov",
        "group": wandb_tracking.get("group") or "pavlov-successive-halving",
        "mode": "online",
        "receipt_required_before_launch": True,
    }

    hf_policy = {
        "every_checkpoint": True,
        "periodic_and_final": True,
        "visibility_policy": "public_or_private_per_quota_and_data_license_safety",
        "allowed_visibility": ["public", "private"],
        "visibility_options": ["public", "private"],
        "visibility": "decide_per_checkpoint",
        "checkpoint_identity_fields": ["stage", "repo_url", "revision", "url"],
        "unique_repo_revision_url_required": True,
        "safe_public_artifact_rule": {
            "required": True,
            "public_only_when_quota_and_data_license_safe": True,
            "private_allowed_when_publication_is_not_safe": True,
        },
        "receipt_required_before_launch": True,
    }
    model_revision_ok, model_revision_by_role, model_revision_missing = _model_revision_status(
        contract, _receipt_sources(contract)
    )

    arms: list[dict[str, Any]] = []
    primary_model = next(
        (model for model in contract["model_candidates"] if model.get("role") == "primary"),
        {},
    )
    for arm_id, learning_rate in LR_ARMS:
        arm_blockers = list(blockers)
        if model_revision_missing and "model_revision" not in missing_receipts:
            arm_blockers.append("missing model revision receipt for the primary model")
        arms.append(
            {
                "id": arm_id,
                "arm_id": arm_id,
                "learning_rate": learning_rate,
                "lr": learning_rate,
                "learning_rate_label": f"{learning_rate:.0e}".replace("e-0", "e-"),
                "lr_label": f"{learning_rate:.0e}".replace("e-0", "e-"),
                "stage": "short_screening",
                "steps": 10,
                "model_role": "primary",
                "model_id": primary_model.get("model_id"),
                "training_suite_ids": list(registry["training_ids"]),
                "primary_eval_suite_ids": list(registry["primary_eval_ids"]),
                "held_out_suite_ids": list(registry["structural_held_out_ids"]),
                "required_receipts": [name for name, _ in REQUIRED_RECEIPTS],
                "wandb": dict(wandb_config),
                "hf_publication_policy": dict(hf_policy),
                "launchable": not arm_blockers,
                "status": "READY" if not arm_blockers else "BLOCKED",
                "receipt_status": dict(receipt_status),
                "missing_receipts": list(missing_receipts),
                "primary_eval_suite_receipt_status": {
                    suite_id: dict(status)
                    for suite_id, status in suite_receipt_status.items()
                    if suite_id in registry["primary_eval_ids"]
                },
                "suite_receipt_status": {
                    suite_id: dict(status)
                    for suite_id, status in suite_receipt_status.items()
                },
                "pending_suite_receipts": list(pending_suite_receipts),
                "pending_training_receipts": list(pending_training_receipts),
                "pending_primary_eval_receipts": list(pending_primary_eval_receipts),
                "blockers": arm_blockers,
            }
        )

    successive_halving = _successive_halving_contract()
    domain_ids = sorted({str(domain) for domain in contract["domains"]})

    return {
        "schema_version": "pavlovs-campaign-manifest-v1",
        "contract_schema_version": contract["schema_version"],
        "status": "BLOCKED" if blockers else "READY",
        "launches_any_job": False,
        "dry_run_only": True,
        "allocation_allowed": False,
        "launchable": not blockers,
        "models": contract["model_candidates"],
        "training_suite_ids": list(registry["training_ids"]),
        "primary_evaluation_suite_ids": list(registry["primary_eval_ids"]),
        "primary_eval_suite_ids": list(registry["primary_eval_ids"]),
        "held_out_suite_ids": list(registry["structural_held_out_ids"]),
        "heldout_suite_ids": list(registry["structural_held_out_ids"]),
        "structural_held_out_suite_ids": list(registry["structural_held_out_ids"]),
        "pending_held_out_suite_ids": list(registry["pending_held_out_ids"]),
        "pending_primary_eval_suite_ids": list(registry["pending_held_out_ids"]),
        "training_suite_count": len(registry["training_ids"]),
        "primary_eval_suite_count": len(registry["primary_eval_ids"]),
        "held_out_suite_count": len(registry["structural_held_out_ids"]),
        "training_suites": registry["training_records"],
        "primary_eval_suites": registry["primary_eval_records"],
        "held_out_suites": registry["held_out_records"],
        "training_suite_domains": registry["training_domains"],
        "primary_eval_suite_domains": registry["primary_eval_domains"],
        "held_out_suite_domains": registry["held_out_domains"],
        "domain_training_suite_ids": registry["domain_training"],
        "domain_primary_eval_suite_ids": registry["domain_primary_eval"],
        "domain_held_out_suite_ids": registry["domain_held_out"],
        "training_suite_domain_map": registry["training_domains"],
        "primary_eval_suite_domain_map": registry["primary_eval_domains"],
        "held_out_suite_domain_map": registry["held_out_domains"],
        "domain_to_training_suite_ids": registry["domain_training"],
        "domain_to_primary_eval_suite_ids": registry["domain_primary_eval"],
        "domain_to_held_out_suite_ids": registry["domain_held_out"],
        "company_train_coverage": company_train_coverage,
        "company_domain_coverage": company_domain_coverage,
        "domains": domain_ids,
        "gsm8k_role": suites["gsm8k_calibration"]["role"],
        "company_eval_coverage": company_eval_coverage,
        "sampling_contract": contract["sampling_contract"],
        "reward_contract": contract["reward_contract"],
        "evaluation_contract": contract["evaluation_contract"],
        "contamination_contract": contract["contamination_contract"],
        "budget_guard": {
            "provider": budget.get("provider"),
            "paid_jobs_may_launch": bool(budget.get("paid_jobs_may_launch")),
            "maximum_usd": budget.get("maximum_usd"),
            "operational_cap_usd": budget.get("operational_cap_usd"),
            "safety_reserve_usd": budget.get("safety_reserve_usd"),
            "contract_status": contract.get("status"),
            "budget_status": budget.get("status"),
            "status_reconciled": not budget_status_conflict,
            "receipt_required": True,
        },
        "wandb": wandb_config,
        "wandb_entity": wandb_config["entity"],
        "wandb_project": wandb_config["project"],
        "wandb_group": wandb_config["group"],
        "hf_publication_policy": hf_policy,
        "hf_checkpoint_policy": hf_policy,
        "split_policy": {
            "train_eval_task_ids_must_be_disjoint": True,
            "selection_split_and_final_eval_split_must_be_disjoint": True,
            "immutable_receipt_required": True,
        },
        "verifier_policy": {
            "rule": contract["reward_contract"]["verifier_rule"],
            "revision_or_digest_receipt_required": True,
        },
        "receipt_policy": {
            "required": [name for name, _ in REQUIRED_RECEIPTS],
            "descriptions": {name: description for name, description in REQUIRED_RECEIPTS},
            "checkpoint_rule": (
                "every periodic and final sampler checkpoint must have an HF receipt"
            ),
            "fail_closed": True,
        },
        "required_receipts": [name for name, _ in REQUIRED_RECEIPTS],
        "receipt_status": receipt_status,
        "receipt_sources": receipt_sources,
        "missing_receipts": missing_receipts,
        "primary_eval_suite_receipt_status": suite_receipt_status,
        "suite_receipt_status": suite_receipt_status,
        "pending_suite_receipts": pending_suite_receipts,
        "pending_training_receipts": pending_training_receipts,
        "pending_primary_eval_receipts": pending_primary_eval_receipts,
        "primary_eval_evidence_ready": primary_eval_evidence_ready,
        "model_revision_receipts": {
            "all_present_and_pinned": model_revision_ok,
            "by_role": model_revision_by_role,
            "missing": model_revision_missing,
        },
        "arms": arms,
        "successive_halving": successive_halving,
        "claim_policy": {
            "xlam_only_launch_allowed": False,
            "xlam_only_claim_allowed": False,
            "xlam_only_scope_detected": xlam_only,
            "xlam_observation_status": (
                "observed seed-809 slice only; not frozen portfolio evidence"
            ),
            "xlam_observation_claim_allowed": False,
            "xlam_requires_immutable_revisions_and_split_hashes": True,
            "gsm8k_role": "calibration_only",
            "held_out_suite_claim_allowed": structural_held_out_receipt_ready,
            "held_out_suite_claim_ids": (
                list(registry["structural_held_out_ids"])
                if structural_held_out_receipt_ready
                else []
            ),
            "held_out_suite_claim_requires_receipts": True,
            "primary_eval_suite_claim_requires_independent_receipts": True,
            "company_usefulness_requires_all_named_domains": True,
            "domain_coverage_is_not_production_readiness": True,
        },
        "non_claims": contract["non_claims"],
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(load_contract(args.contract))
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
