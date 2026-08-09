"""Fail-closed E8 Life-Sci-Bench primary-evaluation boundary.

The repository has no local Life-Sci-Bench data, environment, or verifier.
This module therefore defines and validates the exact receipt boundary without
downloading anything or fabricating a result.  A protocol boundary may be
schema-valid while still lacking an immutable dataset/license pin; an observed
receipt is accepted only when its native state/artifact verifier and all W&B,
Tinker, and HF evidence fields are present.

Life-Sci-Bench is the contract's ``primary_eval`` suite E8.  BrowserGym,
WebBench, ScienceWorld, xLAM, and other related benchmarks are explicitly
non-substitutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

try:  # Package import used by the flagship test suite.
    from . import pavlov_browsergym_adapter as t3_adapter
except ImportError:  # Direct execution from this directory remains offline-safe.
    import pavlov_browsergym_adapter as t3_adapter


SCHEMA_VERSION = "pavlov-lifescibench-e8-boundary-v1"
RECEIPT_SCHEMA_VERSION = "pavlov-lifescibench-e8-result-receipt-v1"
SUITE_ID = "lifescibench_eval"
ROLE = "primary_eval"
SPLIT = "evaluation"
SOURCE_ID = "openai-life-sci-bench-official"
SOURCE_NAME = "Life-Sci-Bench"
SOURCE_URL = "https://openai.com/index/introducing-life-sci-bench/"
DATASET_ID = "lifescibench"
E6_SUITE_ID = "webbench_eval"
XLAM_ID = "xlam"
ALLOWED_DOMAINS = ("science", "long_horizon", "tool_use")
REJECTED_SUBSTITUTES = (
    "xlam",
    "browsergym_train",
    "webbench_eval",
    "scienceworld_train",
    "scibench",
)
NATIVE_ENVIRONMENT_NAME = "lifescibench.native_environment"
NATIVE_VERIFIER_NAME = "lifescibench.native_verifier"
NATIVE_OBSERVATION_SCHEMA = "lifescibench.observation-action-artifact-v1"
RECEIPT_STATUS_PROTOCOL_ONLY = "PROTOCOL_ONLY_NO_RESULT"
RECEIPT_STATUS_OBSERVED = "OBSERVED_PRIMARY_EVAL_RECEIPT"
CLAIM_BOUNDARY = "E8_PRIMARY_EVAL_RECEIPT_ONLY"
UNPINNED = "UNPINNED_REQUIRED"

# ---------------------------------------------------------------------------
# Published LifeSciBench facts (primary source, retrieved 2026-08-09).
#
# Announcement: https://openai.com/index/introducing-life-sci-bench/ (2026-06-17)
# Preprint:     https://cdn.openai.com/pdf/b4299379-0a97-4ffa-8b9b-c3fbb299caa9/
#               lifescibench_preprint.pdf
#
# These constants record what the provider has *documented*.  They are not a
# local copy of the task package: no task text, artifact, or rubric is present
# in this repository, and preprint appendix A.5 ("Data Availability and Safety
# Disclosure") states that public release of tasks, rubrics, artifacts, or
# evaluation materials may be limited by licensing, privacy, proprietary
# information, or biological safety considerations.
#
# NOTE ON ``domain`` vs ``bio_domain``: ``ALLOWED_DOMAINS`` above is the
# *campaign* capability tag set declared in ``pavlovs_domain_contract.json``
# for suite ``lifescibench_eval``.  It is deliberately NOT the benchmark's own
# biological taxonomy.  ``PUBLISHED_BIO_DOMAINS`` below is the provider's
# stratification taxonomy, carried on task rows as ``bio_domain``.
# ---------------------------------------------------------------------------
PUBLISHED_SOURCE_PAPER_URL = (
    "https://cdn.openai.com/pdf/b4299379-0a97-4ffa-8b9b-c3fbb299caa9/"
    "lifescibench_preprint.pdf"
)
PUBLISHED_ANNOUNCEMENT_DATE = "2026-06-17"
PUBLISHED_TASK_COUNT = 750
PUBLISHED_RUBRIC_CRITERIA_COUNT = 19020
PUBLISHED_ARTIFACT_COUNT = 1062
PUBLISHED_PASS_THRESHOLD = "0.70"
PUBLISHED_WORKFLOWS = (
    "evidence_handling",
    "analysis",
    "design_and_optimization",
    "scientific_reasoning",
    "validation_and_operations",
    "translation",
    "scientific_communication",
)
PUBLISHED_BIO_DOMAINS = (
    "genomics",
    "chemistry_medchem",
    "protein_structural_biology",
    "molecular_cell_biology",
    "assays_screening",
    "bioinformatics_comp_bio",
    "clinical_translational_science",
)
# The provider reports two metrics; neither is produced anywhere in this module.
PUBLISHED_METRICS = ("normalized_rubric_score", "task_pass_rate")

# ---------------------------------------------------------------------------
# Synthetic fixture marker.
#
# Every value emitted by ``build_synthetic_fixture`` carries this marker, and
# both validators reject any payload containing it.  The fixture exists solely
# to exercise local plumbing (hashing, manifest construction, disjointness
# proof, receipt emission).  It can never be mistaken for Life-Sci-Bench data
# and can never be promoted into a result.
# ---------------------------------------------------------------------------
SYNTHETIC_MARKER = "SYNTHETIC-NOT-LIFESCIBENCH"
SYNTHETIC_TASK_ID_PREFIX = f"{SYNTHETIC_MARKER}-"
SYNTHETIC_REJECTION_ERROR = (
    f"payload carries the {SYNTHETIC_MARKER} marker: a synthetic fixture is "
    "local plumbing scaffolding and can never be Life-Sci-Bench data, a "
    "Life-Sci-Bench result, or a score"
)

_IMMUTABLE_REVISION_RE = re.compile(r"^(?:sha256:[0-9a-f]{64}|[0-9a-f]{40})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_REFERENCE_RE = re.compile(
    r"(?:[?&](?:token|api[_-]?key|access[_-]?token|password|secret)=|"
    r"(?:token|api[_-]?key|access[_-]?token|password|secret)\s*[:=])",
    re.IGNORECASE,
)

Wandb_REQUIRED_CONFIG_FIELDS = (
    "campaign",
    "suite_id",
    "suite_role",
    "model_id",
    "model_revision",
    "adapter_revision",
    "dataset_revision",
    "split_manifest_hash",
    "container_digest",
    "seed",
    "verifier_type",
    "stateful",
    "artifact_or_side_effect",
)
WANDB_REQUIRED_METRICS = (
    "eval/lifescibench_success_rate",
    "eval/lifescibench_reward_mean",
    "eval/lifescibench_action_count_mean",
)
TINKER_REQUIRED_FIELDS = (
    "run_id",
    "initial_sampler",
    "periodic_samplers",
    "final_sampler",
    "checkpoint_receipt",
)
HF_REQUIRED_FIELDS = (
    "repository",
    "revision",
    "checkpoint_manifest",
    "c0_receipt",
    "exported",
)
REQUIRED_BOUNDARY_FIELDS = (
    "schema_version",
    "source",
    "dataset",
    "role",
    "split",
    "native_environment",
    "native_verifier",
    "task_manifest",
    "eval_split_manifest_hash",
    "train_split_manifest_hash",
    "heldout_policy",
    "receipt_contract",
    "claims",
)
REQUIRED_RECEIPT_FIELDS = (
    "schema_version",
    "receipt_status",
    "source",
    "dataset",
    "role",
    "split",
    "dataset_revision",
    "license_id",
    "task_manifest",
    "eval_split_manifest_hash",
    "heldout_proof",
    "native_verifier",
    "wandb",
    "tinker",
    "hf",
    "cost",
    "substitute_suite_id",
    "e6_substitute",
    "xlam_substitute",
    "portfolio_evidence",
    "claim_boundary",
    "receipt_hash",
)


class LifeSciBenchSchemaError(ValueError):
    """Raised by strict E8 metadata helpers."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LifeSciBenchSchemaError(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not _is_mapping(value):
        raise LifeSciBenchSchemaError(f"{label} must be an object")
    copied = dict(value)
    try:
        t3_adapter.assert_secret_free(copied)
    except t3_adapter.SecretMaterialError as exc:
        raise LifeSciBenchSchemaError(str(exc)) from exc
    return copied


def _nonempty(value: Any, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return ""
    return value


def _digest(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{label} must be a lowercase SHA-256 digest")
        return False
    # A receipt must not use the conventional all-zero sentinel as evidence.
    # It has the right shape but carries no observation, action, state, or
    # artifact identity and would otherwise make a tampered row look valid.
    if value == "0" * 64:
        errors.append(f"{label} must not be an all-zero sentinel digest")
        return False
    return True


def _immutable_revision(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, str) or _IMMUTABLE_REVISION_RE.fullmatch(value) is None:
        errors.append(f"{label} must be an immutable 40-hex or sha256: revision")
        return False
    return True


def _safe_reference(value: Any, label: str, errors: list[str], *, observed: bool = True) -> bool:
    reference = _nonempty(value, label, errors)
    if not reference or not observed:
        return bool(reference)
    if reference.startswith("offline://") or "@" in reference or _SECRET_REFERENCE_RE.search(reference):
        errors.append(f"{label} contains a placeholder or credential-like reference")
        return False
    return True


def _task_hash(task_id: str, dataset_revision: str) -> str:
    return sha256_json({"task_id": task_id, "dataset_revision": dataset_revision})


def task_manifest_hash(task_manifest: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered evaluation task rows, preserving split identity."""

    return sha256_json([dict(item) for item in task_manifest])


def task_id_hash(task_id: str, dataset_revision: str) -> str:
    """Public, deterministic task-ID hash bound to the dataset revision.

    The hash covers both the task identifier and the dataset revision, so the
    same task identifier under a different package revision produces a
    different digest.  This is what makes an immutable task manifest verifiable
    against the package it came from.
    """

    if not isinstance(task_id, str) or not task_id:
        raise LifeSciBenchSchemaError("task_id must be a non-empty string")
    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise LifeSciBenchSchemaError("dataset_revision must be a non-empty string")
    return _task_hash(task_id, dataset_revision)


def contains_synthetic_marker(value: Any) -> bool:
    """True when any part of ``value`` carries the synthetic-fixture marker."""

    try:
        return SYNTHETIC_MARKER in _canonical_json(value)
    except LifeSciBenchSchemaError:
        return SYNTHETIC_MARKER in repr(value)


def build_task_row(
    task_id: str,
    *,
    dataset_revision: str,
    family: str,
    domain: str,
    workflow: str | None = None,
    bio_domain: str | None = None,
    split: str = SPLIT,
    artifact_expected: bool = True,
) -> dict[str, Any]:
    """Build one immutable task-manifest row with a deterministic task hash.

    ``domain`` is the campaign capability tag (``ALLOWED_DOMAINS``).
    ``workflow`` / ``bio_domain`` are the provider's published taxonomies and
    are optional so that rows predating the published taxonomy still validate.
    """

    if not isinstance(family, str) or not family:
        raise LifeSciBenchSchemaError("family must be a non-empty string")
    if domain not in ALLOWED_DOMAINS:
        raise LifeSciBenchSchemaError(
            f"domain must be one of {ALLOWED_DOMAINS!r}, got {domain!r}"
        )
    if workflow is not None and workflow not in PUBLISHED_WORKFLOWS:
        raise LifeSciBenchSchemaError(
            f"workflow must be a published LifeSciBench workflow, got {workflow!r}"
        )
    if bio_domain is not None and bio_domain not in PUBLISHED_BIO_DOMAINS:
        raise LifeSciBenchSchemaError(
            f"bio_domain must be a published LifeSciBench domain, got {bio_domain!r}"
        )
    if artifact_expected is not True:
        raise LifeSciBenchSchemaError("artifact_expected must be true for E8 rows")
    row: dict[str, Any] = {
        "task_id": task_id,
        "task_id_hash": task_id_hash(task_id, dataset_revision),
        "family": family,
        "domain": domain,
        "split": split,
        "artifact_expected": True,
    }
    if workflow is not None:
        row["workflow"] = workflow
    if bio_domain is not None:
        row["bio_domain"] = bio_domain
    return row


def build_split_manifest(
    task_specs: Sequence[Mapping[str, Any]],
    *,
    dataset_revision: str,
    split: str,
) -> dict[str, Any]:
    """Build a split manifest (ordered rows plus its content hash).

    ``task_specs`` rows need ``task_id``, ``family`` and ``domain``; they may
    also carry ``workflow`` and ``bio_domain``.  Duplicate task identifiers are
    rejected, because a duplicated identifier would let one episode stand in
    for two manifest entries.
    """

    if not isinstance(task_specs, Sequence) or isinstance(task_specs, (str, bytes)):
        raise LifeSciBenchSchemaError("task_specs must be a sequence of mappings")
    if not task_specs:
        raise LifeSciBenchSchemaError("task_specs must not be empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, spec in enumerate(task_specs):
        if not _is_mapping(spec):
            raise LifeSciBenchSchemaError(f"task_specs[{index}] must be an object")
        data = dict(spec)
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise LifeSciBenchSchemaError(f"task_specs[{index}].task_id is required")
        if task_id in seen:
            raise LifeSciBenchSchemaError(f"task_specs[{index}] duplicates task_id {task_id!r}")
        seen.add(task_id)
        rows.append(
            build_task_row(
                task_id,
                dataset_revision=dataset_revision,
                family=data.get("family"),
                domain=data.get("domain"),
                workflow=data.get("workflow"),
                bio_domain=data.get("bio_domain"),
                split=split,
                artifact_expected=data.get("artifact_expected", True),
            )
        )
    return {
        "split": split,
        "dataset_revision": dataset_revision,
        "task_count": len(rows),
        "rows": rows,
        "manifest_hash": task_manifest_hash(rows),
    }


def build_heldout_proof(
    *,
    train_manifest: Mapping[str, Any],
    eval_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the evaluation split is held out from the training split.

    Fails closed on any overlap.  Task-ID disjointness alone is not enough: a
    reworded near-duplicate of a trained task carries a different identifier but
    the same family, so family disjointness is proven separately.
    """

    train = _mapping(train_manifest, "train_manifest")
    evaluation = _mapping(eval_manifest, "eval_manifest")
    train_hash = train.get("manifest_hash")
    eval_hash = evaluation.get("manifest_hash")
    if not isinstance(train_hash, str) or _SHA256_RE.fullmatch(train_hash) is None:
        raise LifeSciBenchSchemaError("train_manifest.manifest_hash must be a SHA-256 digest")
    if not isinstance(eval_hash, str) or _SHA256_RE.fullmatch(eval_hash) is None:
        raise LifeSciBenchSchemaError("eval_manifest.manifest_hash must be a SHA-256 digest")
    if train_hash == eval_hash:
        raise LifeSciBenchSchemaError("train and evaluation split hashes are identical")

    def _ids_and_families(manifest: Mapping[str, Any], label: str) -> tuple[set[str], set[str]]:
        rows = manifest.get("rows")
        if not isinstance(rows, list) or not rows:
            raise LifeSciBenchSchemaError(f"{label}.rows must be a non-empty list")
        ids: set[str] = set()
        families: set[str] = set()
        for index, row in enumerate(rows):
            if not _is_mapping(row):
                raise LifeSciBenchSchemaError(f"{label}.rows[{index}] must be an object")
            data = dict(row)
            task_id = data.get("task_id")
            family = data.get("family")
            if not isinstance(task_id, str) or not task_id:
                raise LifeSciBenchSchemaError(f"{label}.rows[{index}].task_id is required")
            if not isinstance(family, str) or not family:
                raise LifeSciBenchSchemaError(f"{label}.rows[{index}].family is required")
            ids.add(task_id)
            families.add(family)
        return ids, families

    train_ids, train_families = _ids_and_families(train, "train_manifest")
    eval_ids, eval_families = _ids_and_families(evaluation, "eval_manifest")
    shared_ids = sorted(train_ids & eval_ids)
    if shared_ids:
        raise LifeSciBenchSchemaError(
            f"train/evaluation task_ids overlap: {shared_ids[:5]}"
        )
    shared_families = sorted(train_families & eval_families)
    if shared_families:
        raise LifeSciBenchSchemaError(
            f"train/evaluation families overlap: {shared_families[:5]}"
        )
    proof: dict[str, Any] = {
        "train_split_manifest_hash": train_hash,
        "eval_split_manifest_hash": eval_hash,
        "disjoint_task_ids": True,
        "disjoint_family_ids": True,
        "unseen_families": sorted(eval_families),
    }
    proof["proof_hash"] = sha256_json(proof)
    return proof


def build_pinned_boundary(
    *,
    dataset_revision: str,
    license_id: str,
    environment_revision: str,
    verifier_revision: str,
    eval_manifest: Mapping[str, Any],
    train_split_manifest_hash: str,
) -> dict[str, Any]:
    """Pin the offline boundary against acquired package/environment identities.

    This is the function the campaign calls once the six access-gated artifacts
    arrive.  Until then it has nothing real to pin.
    """

    evaluation = _mapping(eval_manifest, "eval_manifest")
    boundary = build_offline_e8_boundary()
    boundary["dataset"].update(
        {
            "revision": dataset_revision,
            "license_id": license_id,
            "license_status": "approved",
        }
    )
    boundary["native_environment"]["revision"] = environment_revision
    boundary["native_verifier"]["revision"] = verifier_revision
    boundary["task_manifest"] = [dict(row) for row in evaluation.get("rows", [])]
    boundary["eval_split_manifest_hash"] = evaluation.get("manifest_hash")
    boundary["train_split_manifest_hash"] = train_split_manifest_hash
    boundary["metadata_status"] = "PINNED_PENDING_OBSERVED_RECEIPT"
    return boundary


def build_e8_result_receipt(
    *,
    boundary: Mapping[str, Any],
    heldout_proof: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    wandb_evidence: Mapping[str, Any],
    tinker_evidence: Mapping[str, Any],
    hf_evidence: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble a canonical E8 result receipt and seal it with ``receipt_hash``.

    Emission is deliberately separate from validation: this builds the exact
    payload shape, and ``validate_e8_receipt`` decides whether it is acceptable.
    """

    data = _mapping(boundary, "boundary")
    dataset = dict(data.get("dataset") or {})
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_status": RECEIPT_STATUS_OBSERVED,
        "source": dict(data.get("source") or {}),
        "dataset": dataset,
        "role": ROLE,
        "split": SPLIT,
        "dataset_revision": dataset.get("revision"),
        "license_id": dataset.get("license_id"),
        "task_manifest": [dict(row) for row in data.get("task_manifest") or []],
        "eval_split_manifest_hash": data.get("eval_split_manifest_hash"),
        "heldout_proof": dict(heldout_proof),
        "native_verifier": {
            "name": NATIVE_VERIFIER_NAME,
            "environment_name": NATIVE_ENVIRONMENT_NAME,
            "environment_revision": dict(data.get("native_environment") or {}).get("revision"),
            "observation_schema": NATIVE_OBSERVATION_SCHEMA,
            "verifier_revision": dict(data.get("native_verifier") or {}).get("revision"),
            "checked": True,
            "stateful": True,
            "artifact_or_side_effect": True,
            "artifact_required": True,
            "episode_rows": [dict(row) for row in episode_rows],
        },
        "wandb": dict(wandb_evidence),
        "tinker": dict(tinker_evidence),
        "hf": dict(hf_evidence),
        "cost": dict(cost),
        "substitute_suite_id": None,
        "e6_substitute": False,
        "xlam_substitute": False,
        "portfolio_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _synthetic_digest(label: str) -> str:
    return sha256_json({"marker": SYNTHETIC_MARKER, "label": label})


def build_synthetic_fixture() -> dict[str, Any]:
    """Build an end-to-end SYNTHETIC fixture that exercises local plumbing only.

    Every identifier carries ``SYNTHETIC_MARKER``, so both validators reject the
    whole payload.  No score, pass rate, or rubric value is produced anywhere:
    ``task_success`` is ``False`` on every row and every metric is ``0.0``,
    present only because the receipt schema requires the keys to exist.
    """

    dataset_revision = _synthetic_digest("dataset-revision")[:40]
    environment_revision = _synthetic_digest("environment-revision")[:40]
    verifier_revision = _synthetic_digest("verifier-revision")[:40]
    license_id = f"{SYNTHETIC_MARKER}-license"

    eval_specs = [
        {
            "task_id": f"{SYNTHETIC_TASK_ID_PREFIX}eval-0001",
            "family": f"{SYNTHETIC_MARKER}-family-eval-alpha",
            "domain": "science",
            "workflow": "evidence_handling",
            "bio_domain": "genomics",
        },
        {
            "task_id": f"{SYNTHETIC_TASK_ID_PREFIX}eval-0002",
            "family": f"{SYNTHETIC_MARKER}-family-eval-beta",
            "domain": "long_horizon",
            "workflow": "analysis",
            "bio_domain": "bioinformatics_comp_bio",
        },
        {
            "task_id": f"{SYNTHETIC_TASK_ID_PREFIX}eval-0003",
            "family": f"{SYNTHETIC_MARKER}-family-eval-gamma",
            "domain": "tool_use",
            "workflow": "translation",
            "bio_domain": "clinical_translational_science",
        },
    ]
    train_specs = [
        {
            "task_id": f"{SYNTHETIC_TASK_ID_PREFIX}train-0001",
            "family": f"{SYNTHETIC_MARKER}-family-train-delta",
            "domain": "science",
            "workflow": "scientific_reasoning",
            "bio_domain": "molecular_cell_biology",
        },
        {
            "task_id": f"{SYNTHETIC_TASK_ID_PREFIX}train-0002",
            "family": f"{SYNTHETIC_MARKER}-family-train-epsilon",
            "domain": "tool_use",
            "workflow": "design_and_optimization",
            "bio_domain": "chemistry_medchem",
        },
    ]
    eval_manifest = build_split_manifest(
        eval_specs, dataset_revision=dataset_revision, split=SPLIT
    )
    train_manifest = build_split_manifest(
        train_specs, dataset_revision=dataset_revision, split="train"
    )
    heldout_proof = build_heldout_proof(
        train_manifest=train_manifest, eval_manifest=eval_manifest
    )
    boundary = build_pinned_boundary(
        dataset_revision=dataset_revision,
        license_id=license_id,
        environment_revision=environment_revision,
        verifier_revision=verifier_revision,
        eval_manifest=eval_manifest,
        train_split_manifest_hash=train_manifest["manifest_hash"],
    )
    episode_rows = [
        {
            "task_id": row["task_id"],
            "task_id_hash": row["task_id_hash"],
            "family": row["family"],
            "domain": row["domain"],
            "observation_hash": _synthetic_digest(f"observation::{row['task_id']}"),
            "action_hash": _synthetic_digest(f"action::{row['task_id']}"),
            "state_hash": _synthetic_digest(f"state::{row['task_id']}"),
            "artifact_digest": _synthetic_digest(f"artifact::{row['task_id']}"),
            # Always False: the fixture must never look like a measurement.
            "task_success": False,
        }
        for row in eval_manifest["rows"]
    ]
    receipt = build_e8_result_receipt(
        boundary=boundary,
        heldout_proof=heldout_proof,
        episode_rows=episode_rows,
        wandb_evidence={
            "observed": True,
            "run_id": f"{SYNTHETIC_MARKER}-wandb-run",
            "url": f"https://wandb.ai/{SYNTHETIC_MARKER}/e8-fixture/runs/synthetic",
            "project": f"{SYNTHETIC_MARKER}-project",
            "config_hash": _synthetic_digest("wandb-config"),
            "sample_manifest_hash": _synthetic_digest("wandb-sample-manifest"),
            # Zeroed placeholders; the keys exist only to satisfy the schema.
            "metrics": {name: 0.0 for name in WANDB_REQUIRED_METRICS},
        },
        tinker_evidence={
            "observed": True,
            "run_id": f"{SYNTHETIC_MARKER}-tinker-run",
            "initial_sampler": f"tinker://{SYNTHETIC_MARKER}/initial",
            "periodic_samplers": [f"tinker://{SYNTHETIC_MARKER}/step-0001"],
            "final_sampler": f"tinker://{SYNTHETIC_MARKER}/final",
            "checkpoint_receipt": f"{SYNTHETIC_MARKER}/checkpoint.json",
        },
        hf_evidence={
            "observed": True,
            "repository": f"{SYNTHETIC_MARKER}/e8-fixture",
            "revision": _synthetic_digest("hf-revision")[:40],
            "checkpoint_manifest": f"{SYNTHETIC_MARKER}/checkpoint-manifest.json",
            "c0_receipt": f"{SYNTHETIC_MARKER}/c0-receipt.json",
            "exported": True,
        },
        cost={"currency": "USD", "charged_usd": 0.0, "cap_usd": 1.0, "within_cap": True},
    )
    return {
        "marker": SYNTHETIC_MARKER,
        "notice": (
            "Synthetic local-plumbing fixture. Contains no Life-Sci-Bench task, "
            "artifact, rubric, or measurement. Produces no score. Rejected by "
            "validate_e8_boundary and validate_e8_receipt by design."
        ),
        "score": None,
        "dataset_revision": dataset_revision,
        "license_id": license_id,
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "heldout_proof": heldout_proof,
        "boundary": boundary,
        "receipt": receipt,
    }


def build_offline_e8_boundary() -> dict[str, Any]:
    """Return metadata-only E8 protocol scaffolding; it intentionally stays blocked."""

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "source_id": SOURCE_ID,
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "source_kind": "official_source_identity",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": UNPINNED,
            "license_id": UNPINNED,
            "license_status": "pending",
            "license_source": SOURCE_URL,
        },
        "role": ROLE,
        "split": SPLIT,
        "native_environment": {
            "name": NATIVE_ENVIRONMENT_NAME,
            "revision": UNPINNED,
            "reset_per_task": True,
            "observation_schema": NATIVE_OBSERVATION_SCHEMA,
            "stateful": True,
            "artifact_or_side_effect": True,
        },
        "native_verifier": {
            "name": NATIVE_VERIFIER_NAME,
            "revision": UNPINNED,
            "native_success_field": "task_success",
            "observation_hash_field": "observation_hash",
            "action_hash_field": "action_hash",
            "state_hash_field": "state_hash",
            "artifact_digest_field": "artifact_digest",
            "artifact_required": True,
        },
        "task_manifest": [],
        "eval_split_manifest_hash": task_manifest_hash([]),
        "train_split_manifest_hash": UNPINNED,
        "heldout_policy": {
            "primary_eval": True,
            "evaluation_only": True,
            "disjoint_task_ids_required": True,
            "disjoint_family_ids_required": True,
            "receipt_proven_heldout": False,
        },
        "receipt_contract": {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "observed_result_required": True,
            "native_verifier_required": True,
            "wandb_required": True,
            "tinker_required": True,
            "hf_required": True,
            "cost_required": True,
            "wandb_config_fields": list(Wandb_REQUIRED_CONFIG_FIELDS),
            "wandb_metric_keys": list(WANDB_REQUIRED_METRICS),
            "tinker_fields": list(TINKER_REQUIRED_FIELDS),
            "hf_fields": list(HF_REQUIRED_FIELDS),
        },
        "claims": {
            "primary_eval": True,
            "receipt_proven_heldout": False,
            "portfolio_evidence": False,
            "e6_substitute": False,
            "xlam_substitute": False,
            "rejected_substitutes": list(REJECTED_SUBSTITUTES),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "metadata_status": "BLOCKED_METADATA_ONLY",
    }


@dataclass(frozen=True)
class BoundaryValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "primary_eval": True,
            "receipt_proven_heldout": bool(self.metrics.get("receipt_proven_heldout", False)),
            "portfolio_evidence": False,
            "paid_launch_authorized": False,
        }


def _validate_task_manifest(
    value: Any,
    *,
    dataset_revision: Any,
    errors: list[str],
) -> tuple[bool, list[Mapping[str, Any]]]:
    if not isinstance(value, list) or not value:
        errors.append("task_manifest must contain immutable evaluation task rows")
        return False, []
    rows: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    valid = True
    for index, item in enumerate(value):
        if not _is_mapping(item):
            errors.append(f"task_manifest[{index}] must be an object")
            valid = False
            continue
        row = dict(item)
        rows.append(row)
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"task_manifest[{index}].task_id is required")
            valid = False
        elif task_id in ids:
            errors.append(f"task_manifest[{index}] duplicates task_id")
            valid = False
        else:
            ids.add(task_id)
        expected_hash = _task_hash(task_id, dataset_revision) if isinstance(task_id, str) else ""
        if row.get("task_id_hash") != expected_hash or not _digest(row.get("task_id_hash"), f"task_manifest[{index}].task_id_hash", errors):
            errors.append(f"task_manifest[{index}] task hash is not deterministic")
            valid = False
        elif row["task_id_hash"] in hashes:
            errors.append(f"task_manifest[{index}] duplicates task_id_hash")
            valid = False
        else:
            hashes.add(row["task_id_hash"])
        if row.get("split") != SPLIT:
            errors.append(f"task_manifest[{index}].split must be evaluation")
            valid = False
        if row.get("artifact_expected") is not True:
            errors.append(f"task_manifest[{index}] must require an artifact")
            valid = False
        if row.get("domain") not in ALLOWED_DOMAINS:
            errors.append(f"task_manifest[{index}] has an unbound domain")
            valid = False
        if not isinstance(row.get("family"), str) or not row["family"]:
            errors.append(f"task_manifest[{index}].family is required for held-out proof")
            valid = False
        workflow = row.get("workflow")
        if workflow is not None and workflow not in PUBLISHED_WORKFLOWS:
            errors.append(f"task_manifest[{index}].workflow is not a published LifeSciBench workflow")
            valid = False
        bio_domain = row.get("bio_domain")
        if bio_domain is not None and bio_domain not in PUBLISHED_BIO_DOMAINS:
            errors.append(f"task_manifest[{index}].bio_domain is not a published LifeSciBench domain")
            valid = False
    return valid, rows


def validate_e8_boundary(boundary: Mapping[str, Any]) -> BoundaryValidationResult:
    """Validate protocol metadata; no result or paid-launch authorization is inferred."""

    errors: list[str] = []
    if not _is_mapping(boundary):
        return BoundaryValidationResult(False, ("boundary must be an object",), {})
    try:
        t3_adapter.assert_secret_free(boundary)
    except t3_adapter.SecretMaterialError as exc:
        errors.append(str(exc))
    if contains_synthetic_marker(boundary):
        errors.append(SYNTHETIC_REJECTION_ERROR)
    data = dict(boundary)
    for field_name in REQUIRED_BOUNDARY_FIELDS:
        if field_name not in data:
            errors.append(f"missing boundary field: {field_name}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the E8 boundary schema")
    source = data.get("source")
    if not _is_mapping(source):
        errors.append("source must be an object")
    else:
        source_data = dict(source)
        for field_name, expected in (
            ("source_id", SOURCE_ID),
            ("name", SOURCE_NAME),
            ("url", SOURCE_URL),
            ("source_kind", "official_source_identity"),
        ):
            if source_data.get(field_name) != expected:
                errors.append(f"source.{field_name} is not the authoritative E8 identity")
    if data.get("role") != ROLE:
        errors.append("E8 role must be primary_eval")
    if data.get("split") != SPLIT:
        errors.append("E8 split must be evaluation")

    dataset = data.get("dataset")
    dataset_revision: Any = None
    if not _is_mapping(dataset):
        errors.append("dataset must be an object")
    else:
        dataset_data = dict(dataset)
        dataset_revision = dataset_data.get("revision")
        if dataset_data.get("dataset_id") != DATASET_ID:
            errors.append("dataset.dataset_id must be lifescibench")
        _immutable_revision(dataset_revision, "dataset.revision", errors)
        if dataset_data.get("license_status") != "approved":
            errors.append("dataset license must be explicitly approved")
        license_id = _nonempty(dataset_data.get("license_id"), "dataset.license_id", errors)
        if license_id == UNPINNED:
            errors.append("dataset.license_id must be pinned, not UNPINNED_REQUIRED")
        license_source = _nonempty(
            dataset_data.get("license_source"), "dataset.license_source", errors
        )
        if license_source and license_source != SOURCE_URL:
            errors.append("dataset.license_source must identify the official E8 source")
        _safe_reference(license_source, "dataset.license_source", errors)
    if dataset_revision is None:
        dataset_revision = UNPINNED

    environment = data.get("native_environment")
    environment_valid = True
    if not _is_mapping(environment):
        errors.append("native_environment must be an object")
        environment_valid = False
    else:
        environment_data = dict(environment)
        if environment_data.get("name") != NATIVE_ENVIRONMENT_NAME:
            errors.append("native_environment.name is not pinned")
        environment_valid = _immutable_revision(environment_data.get("revision"), "native_environment.revision", errors)
        for field_name in ("reset_per_task", "stateful", "artifact_or_side_effect"):
            if environment_data.get(field_name) is not True:
                errors.append(f"native_environment.{field_name} must be true")
        if environment_data.get("observation_schema") != NATIVE_OBSERVATION_SCHEMA:
            errors.append("native_environment.observation_schema is not the E8 schema")

    verifier = data.get("native_verifier")
    verifier_valid = True
    if not _is_mapping(verifier):
        errors.append("native_verifier must be an object")
        verifier_valid = False
    else:
        verifier_data = dict(verifier)
        if verifier_data.get("name") != NATIVE_VERIFIER_NAME:
            errors.append("native_verifier.name is not the E8 native verifier")
        verifier_valid = _immutable_revision(verifier_data.get("revision"), "native_verifier.revision", errors)
        if verifier_data.get("artifact_required") is not True:
            errors.append("native_verifier must require artifacts")
        for field_name in (
            "native_success_field",
            "observation_hash_field",
            "action_hash_field",
            "state_hash_field",
            "artifact_digest_field",
        ):
            _nonempty(verifier_data.get(field_name), f"native_verifier.{field_name}", errors)

    manifest_valid, task_rows = _validate_task_manifest(
        data.get("task_manifest"), dataset_revision=dataset_revision, errors=errors
    )
    eval_hash = data.get("eval_split_manifest_hash")
    if not _digest(eval_hash, "eval_split_manifest_hash", errors):
        eval_hash_valid = False
    else:
        try:
            expected_eval_hash = task_manifest_hash(task_rows)
        except LifeSciBenchSchemaError as exc:
            errors.append(f"task_manifest cannot be hashed: {exc}")
            eval_hash_valid = False
        else:
            eval_hash_valid = eval_hash == expected_eval_hash
            if not eval_hash_valid:
                errors.append("eval_split_manifest_hash does not match task_manifest")
    train_hash = data.get("train_split_manifest_hash")
    # Split manifests are content hashes, not source revisions.  Keep their
    # representation as a bare SHA-256 digest so held-out proof can compare
    # the exact train/evaluation manifests without conflating the two kinds of
    # immutable identity.
    train_hash_valid = _digest(train_hash, "train_split_manifest_hash", errors)
    if eval_hash == train_hash:
        errors.append("train and evaluation split hashes must be disjoint")
        train_hash_valid = False

    heldout_policy = data.get("heldout_policy")
    heldout_policy_valid = True
    if not _is_mapping(heldout_policy):
        errors.append("heldout_policy must be an object")
        heldout_policy_valid = False
    else:
        policy = dict(heldout_policy)
        for field_name in (
            "primary_eval",
            "evaluation_only",
            "disjoint_task_ids_required",
            "disjoint_family_ids_required",
        ):
            if policy.get(field_name) is not True:
                errors.append(f"heldout_policy.{field_name} must be true")
                heldout_policy_valid = False
        if policy.get("receipt_proven_heldout") is not False:
            errors.append("protocol boundary cannot claim receipt-proven held-out evidence")
            heldout_policy_valid = False

    contract = data.get("receipt_contract")
    contract_valid = True
    if not _is_mapping(contract):
        errors.append("receipt_contract must be an object")
        contract_valid = False
    else:
        contract_data = dict(contract)
        if contract_data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            errors.append("receipt_contract schema_version is not E8")
            contract_valid = False
        for field_name in (
            "observed_result_required",
            "native_verifier_required",
            "wandb_required",
            "tinker_required",
            "hf_required",
            "cost_required",
        ):
            if contract_data.get(field_name) is not True:
                errors.append(f"receipt_contract.{field_name} must be true")
                contract_valid = False
        for field_name, expected in (
            ("wandb_config_fields", Wandb_REQUIRED_CONFIG_FIELDS),
            ("wandb_metric_keys", WANDB_REQUIRED_METRICS),
            ("tinker_fields", TINKER_REQUIRED_FIELDS),
            ("hf_fields", HF_REQUIRED_FIELDS),
        ):
            values = contract_data.get(field_name)
            if not isinstance(values, list) or not set(expected).issubset(values):
                errors.append(f"receipt_contract.{field_name} is incomplete")
                contract_valid = False

    claims = data.get("claims")
    claims_valid = True
    if not _is_mapping(claims):
        errors.append("claims must be an object")
        claims_valid = False
    else:
        claims_data = dict(claims)
        if claims_data.get("primary_eval") is not True:
            errors.append("claims.primary_eval must be true")
            claims_valid = False
        for field_name in ("receipt_proven_heldout", "portfolio_evidence", "e6_substitute", "xlam_substitute"):
            if claims_data.get(field_name) is not False:
                errors.append(f"claims.{field_name} must remain false at protocol stage")
                claims_valid = False
        if claims_data.get("claim_boundary") != CLAIM_BOUNDARY:
            errors.append("claims.claim_boundary is not E8-only")
            claims_valid = False
        rejected = claims_data.get("rejected_substitutes")
        if not isinstance(rejected, list) or not set(REJECTED_SUBSTITUTES).issubset(rejected):
            errors.append("claims.rejected_substitutes must include xLAM and related suites")
            claims_valid = False

    metrics = {
        "schema_valid": not errors,
        "source_identity_valid": _is_mapping(source) and not any(error.startswith("source.") for error in errors),
        "immutable_revision_valid": bool(dataset_revision and _IMMUTABLE_REVISION_RE.fullmatch(str(dataset_revision)))
        and environment_valid
        and verifier_valid
        and train_hash_valid,
        "task_manifest_valid": manifest_valid and eval_hash_valid,
        "native_contract_valid": environment_valid and verifier_valid,
        "receipt_contract_valid": contract_valid,
        "heldout_policy_valid": heldout_policy_valid,
        "claims_valid": claims_valid,
        "primary_eval": True,
        "receipt_proven_heldout": False,
        "portfolio_evidence": False,
        "paid_launch_authorized": False,
    }
    return BoundaryValidationResult(not errors, tuple(errors), metrics)


def _validate_evidence_block(
    value: Any,
    *,
    name: str,
    required_fields: Sequence[str],
    errors: list[str],
) -> bool:
    if not _is_mapping(value):
        errors.append(f"{name} evidence must be an object")
        return False
    data = dict(value)
    valid = data.get("observed") is True
    if data.get("observed") is not True:
        errors.append(f"{name} evidence must be observed for a result receipt")
    for field_name in required_fields:
        if field_name not in data:
            errors.append(f"{name} evidence missing {field_name}")
    for field_name in required_fields:
        if field_name in data and field_name not in (
            "observed",
            "metrics",
            "exported",
            "periodic_samplers",
        ):
            _safe_reference(data[field_name], f"{name}.{field_name}", errors)
    if "periodic_samplers" in required_fields:
        samplers = data.get("periodic_samplers")
        if not isinstance(samplers, list) or not samplers:
            errors.append(f"{name}.periodic_samplers must be a non-empty list")
        else:
            for index, sampler in enumerate(samplers):
                _safe_reference(sampler, f"{name}.periodic_samplers[{index}]", errors)
    return valid and not any(error.startswith(name) for error in errors)


@dataclass(frozen=True)
class ReceiptValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "receipt_hash": self.receipt_hash,
            "primary_eval": True,
            "receipt_proven_heldout": bool(self.metrics.get("receipt_proven_heldout", False)),
            "portfolio_evidence": False,
            "paid_launch_authorized": False,
        }


def validate_e8_receipt(
    result_receipt: Mapping[str, Any],
    *,
    boundary: Mapping[str, Any] | None = None,
) -> ReceiptValidationResult:
    """Validate observed E8 evidence; never promote it to portfolio evidence."""

    errors: list[str] = []
    receipt_hash = ""
    if not _is_mapping(result_receipt):
        return ReceiptValidationResult(False, ("result receipt must be an object",), {})
    try:
        t3_adapter.assert_secret_free(result_receipt)
    except t3_adapter.SecretMaterialError as exc:
        errors.append(str(exc))
    if contains_synthetic_marker(result_receipt):
        errors.append(SYNTHETIC_REJECTION_ERROR)
    data = dict(result_receipt)
    for field_name in REQUIRED_RECEIPT_FIELDS:
        if field_name not in data:
            errors.append(f"missing result receipt field: {field_name}")
    supplied_hash = data.get("receipt_hash")
    if isinstance(supplied_hash, str) and _SHA256_RE.fullmatch(supplied_hash):
        receipt_hash = supplied_hash
        try:
            expected_hash = sha256_json(
                {key: value for key, value in data.items() if key != "receipt_hash"}
            )
        except LifeSciBenchSchemaError as exc:
            errors.append(f"receipt payload cannot be hashed: {exc}")
        else:
            if supplied_hash != expected_hash:
                errors.append("receipt_hash does not match canonical payload")
    else:
        errors.append("receipt_hash must be a SHA-256 digest")

    if data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("result receipt schema_version is not E8")
    if data.get("receipt_status") != RECEIPT_STATUS_OBSERVED:
        errors.append("result receipt must be an observed primary-eval receipt")
    source = data.get("source")
    if not _is_mapping(source):
        errors.append("result source must be an object")
    else:
        source_data = dict(source)
        for field_name, expected in (
            ("source_id", SOURCE_ID),
            ("name", SOURCE_NAME),
            ("url", SOURCE_URL),
            ("source_kind", "official_source_identity"),
        ):
            if source_data.get(field_name) != expected:
                errors.append(f"result source.{field_name} is not official E8")
    if data.get("role") != ROLE:
        errors.append("result role must be primary_eval")
    if data.get("split") != SPLIT:
        errors.append("result split must be evaluation")
    dataset = data.get("dataset")
    if not _is_mapping(dataset):
        errors.append("result dataset must be an object")
        dataset_data: dict[str, Any] = {}
    else:
        dataset_data = dict(dataset)
    if dataset_data.get("dataset_id") != DATASET_ID:
        errors.append("result dataset_id must be lifescibench")
    dataset_revision = data.get("dataset_revision")
    _immutable_revision(dataset_revision, "result dataset_revision", errors)
    license_id = _nonempty(data.get("license_id"), "result license_id", errors)
    if license_id == UNPINNED:
        errors.append("result license_id must be pinned, not UNPINNED_REQUIRED")
    if dataset_data.get("revision") != dataset_revision:
        errors.append("result dataset.revision does not match dataset_revision")
    if dataset_data.get("license_id") != license_id:
        errors.append("result dataset.license_id does not match license_id")
    if dataset_data.get("license_status") != "approved":
        errors.append("result dataset license must be explicitly approved")
    if dataset_data.get("license_source") != SOURCE_URL:
        errors.append("result dataset.license_source must identify the official E8 source")

    manifest_valid, task_rows = _validate_task_manifest(
        data.get("task_manifest"), dataset_revision=dataset_revision, errors=errors
    )
    eval_hash = data.get("eval_split_manifest_hash")
    eval_hash_valid = _digest(eval_hash, "result eval_split_manifest_hash", errors)
    if eval_hash_valid:
        try:
            expected_eval_hash = task_manifest_hash(task_rows)
        except LifeSciBenchSchemaError as exc:
            errors.append(f"result task manifest cannot be hashed: {exc}")
            eval_hash_valid = False
        else:
            if eval_hash != expected_eval_hash:
                errors.append("result eval_split_manifest_hash does not match task rows")
                eval_hash_valid = False

    if boundary is not None:
        boundary_result = validate_e8_boundary(boundary)
        if boundary_result.metrics.get("schema_valid") is not True:
            errors.append("provided E8 boundary is not protocol-valid")
        boundary_data = dict(boundary)
        boundary_dataset = boundary_data.get("dataset")
        if _is_mapping(boundary_dataset) and boundary_dataset.get("revision") != dataset_revision:
            errors.append("result dataset revision differs from boundary")
        if _is_mapping(boundary_dataset) and boundary_dataset.get("license_id") != license_id:
            errors.append("result license differs from boundary")
        if boundary_data.get("eval_split_manifest_hash") != eval_hash:
            errors.append("result eval split hash differs from boundary")

    heldout = data.get("heldout_proof")
    heldout_valid = True
    if not _is_mapping(heldout):
        errors.append("heldout_proof must be an object")
        heldout_valid = False
    else:
        proof = dict(heldout)
        for field_name in (
            "train_split_manifest_hash",
            "eval_split_manifest_hash",
            "proof_hash",
        ):
            if not _digest(proof.get(field_name), f"heldout_proof.{field_name}", errors):
                heldout_valid = False
        if proof.get("eval_split_manifest_hash") != eval_hash:
            errors.append("heldout proof eval hash does not match receipt")
            heldout_valid = False
        if boundary is not None:
            boundary_train_hash = dict(boundary).get("train_split_manifest_hash")
            if proof.get("train_split_manifest_hash") != boundary_train_hash:
                errors.append("heldout proof train hash differs from boundary")
                heldout_valid = False
        if proof.get("train_split_manifest_hash") == eval_hash:
            errors.append("heldout train/evaluation hashes are not disjoint")
            heldout_valid = False
        for field_name in ("disjoint_task_ids", "disjoint_family_ids"):
            if proof.get(field_name) is not True:
                errors.append(f"heldout_proof.{field_name} must be true")
                heldout_valid = False
        if not isinstance(proof.get("unseen_families"), list) or not proof["unseen_families"]:
            errors.append("heldout_proof.unseen_families is required")
            heldout_valid = False
        try:
            expected_proof_hash = sha256_json(
                {key: value for key, value in proof.items() if key != "proof_hash"}
            )
        except LifeSciBenchSchemaError as exc:
            errors.append(f"heldout_proof cannot be hashed: {exc}")
            heldout_valid = False
        else:
            if proof.get("proof_hash") != expected_proof_hash:
                errors.append("heldout_proof.proof_hash does not match canonical proof")
                heldout_valid = False

    native = data.get("native_verifier")
    native_valid = True
    if not _is_mapping(native):
        errors.append("native_verifier receipt block must be an object")
        native_valid = False
    else:
        native_data = dict(native)
        for field_name, expected in (
            ("name", NATIVE_VERIFIER_NAME),
            ("environment_name", NATIVE_ENVIRONMENT_NAME),
            ("observation_schema", NATIVE_OBSERVATION_SCHEMA),
        ):
            if native_data.get(field_name) != expected:
                errors.append(f"native_verifier.{field_name} is not pinned")
                native_valid = False
        _immutable_revision(native_data.get("environment_revision"), "native_verifier.environment_revision", errors)
        _immutable_revision(native_data.get("verifier_revision"), "native_verifier.verifier_revision", errors)
        if boundary is not None:
            boundary_data = dict(boundary)
            boundary_environment = boundary_data.get("native_environment")
            boundary_verifier = boundary_data.get("native_verifier")
            if _is_mapping(boundary_environment) and native_data.get("environment_revision") != boundary_environment.get("revision"):
                errors.append("result environment revision differs from boundary")
                native_valid = False
            if _is_mapping(boundary_verifier) and native_data.get("verifier_revision") != boundary_verifier.get("revision"):
                errors.append("result verifier revision differs from boundary")
                native_valid = False
        if native_data.get("checked") is not True:
            errors.append("native_verifier.checked must be true")
            native_valid = False
        for field_name in ("stateful", "artifact_or_side_effect", "artifact_required"):
            if native_data.get(field_name) is not True:
                errors.append(f"native_verifier.{field_name} must be true")
                native_valid = False
        rows = native_data.get("episode_rows")
        manifest_by_id = {
            row.get("task_id"): row
            for row in task_rows
            if isinstance(row.get("task_id"), str)
        }
        if not isinstance(rows, list) or not rows:
            errors.append("native_verifier.episode_rows are required")
            native_valid = False
        else:
            seen_ids: set[str] = set()
            for index, row in enumerate(rows):
                if not _is_mapping(row):
                    errors.append(f"native_verifier.episode_rows[{index}] must be an object")
                    native_valid = False
                    continue
                row_data = dict(row)
                task_id = row_data.get("task_id")
                if not isinstance(task_id, str) or task_id in seen_ids:
                    errors.append(f"native_verifier.episode_rows[{index}] task_id is missing or duplicated")
                    native_valid = False
                else:
                    seen_ids.add(task_id)
                manifest_row = manifest_by_id.get(task_id)
                if manifest_row is None:
                    errors.append(f"episode_rows[{index}].task_id is not in task_manifest")
                    native_valid = False
                expected_hash = _task_hash(task_id, dataset_revision) if isinstance(task_id, str) else ""
                if row_data.get("task_id_hash") != expected_hash:
                    errors.append(f"native_verifier.episode_rows[{index}] task hash mismatch")
                    native_valid = False
                for field_name in ("observation_hash", "action_hash", "state_hash", "artifact_digest"):
                    if not _digest(row_data.get(field_name), f"episode_rows[{index}].{field_name}", errors):
                        native_valid = False
                if row_data.get("task_success") not in (True, False):
                    errors.append(f"episode_rows[{index}].task_success must be boolean")
                    native_valid = False
                if row_data.get("domain") not in ALLOWED_DOMAINS:
                    errors.append(f"episode_rows[{index}].domain is not E8-bound")
                    native_valid = False
                if not isinstance(row_data.get("family"), str) or not row_data["family"]:
                    errors.append(f"episode_rows[{index}].family is required")
                    native_valid = False
                if manifest_row is not None:
                    if row_data.get("family") != manifest_row.get("family"):
                        errors.append(f"episode_rows[{index}].family does not match task_manifest")
                        native_valid = False
                    if row_data.get("domain") != manifest_row.get("domain"):
                        errors.append(f"episode_rows[{index}].domain does not match task_manifest")
                        native_valid = False
            if set(manifest_by_id) != seen_ids:
                errors.append("native_verifier.episode_rows must cover every evaluation task")
                native_valid = False

    wandb_valid = _validate_evidence_block(
        data.get("wandb"),
        name="wandb",
        required_fields=("observed", "run_id", "url", "project", "config_hash", "metrics", "sample_manifest_hash"),
        errors=errors,
    )
    wandb = data.get("wandb")
    if _is_mapping(wandb):
        wandb_data = dict(wandb)
        _digest(wandb_data.get("config_hash"), "wandb.config_hash", errors)
        _digest(wandb_data.get("sample_manifest_hash"), "wandb.sample_manifest_hash", errors)
        metrics = wandb_data.get("metrics")
        if not _is_mapping(metrics):
            errors.append("wandb.metrics must be an object")
        else:
            for metric_name in WANDB_REQUIRED_METRICS:
                if metric_name not in metrics:
                    errors.append(f"wandb.metrics missing {metric_name}")
                else:
                    try:
                        metric_value = Decimal(str(metrics[metric_name]))
                        if not metric_value.is_finite():
                            raise InvalidOperation
                    except (InvalidOperation, ValueError):
                        errors.append(f"wandb.metrics.{metric_name} must be finite")

    tinker_valid = _validate_evidence_block(
        data.get("tinker"), name="tinker", required_fields=TINKER_REQUIRED_FIELDS, errors=errors
    )
    hf_valid = _validate_evidence_block(
        data.get("hf"), name="hf", required_fields=HF_REQUIRED_FIELDS, errors=errors
    )
    hf = data.get("hf")
    if _is_mapping(hf):
        hf_data = dict(hf)
        _immutable_revision(hf_data.get("revision"), "hf.revision", errors)
        if hf_data.get("exported") is not True:
            errors.append("hf.exported must be true")

    cost = data.get("cost")
    cost_valid = True
    if not _is_mapping(cost):
        errors.append("cost must be an object")
        cost_valid = False
    else:
        cost_data = dict(cost)
        if cost_data.get("currency") != "USD":
            errors.append("cost.currency must be USD")
            cost_valid = False
        charged = cost_data.get("charged_usd")
        cap = cost_data.get("cap_usd")
        try:
            charged_d = Decimal(str(charged))
            cap_d = Decimal(str(cap))
            if not charged_d.is_finite() or charged_d < 0 or not cap_d.is_finite() or cap_d <= 0 or charged_d > cap_d:
                raise InvalidOperation
        except (InvalidOperation, ValueError):
            errors.append("cost charged_usd/cap_usd must be finite and within cap")
            cost_valid = False
        if cost_data.get("within_cap") is not True:
            errors.append("cost.within_cap must be true")
            cost_valid = False

    for field_name, expected in (
        ("substitute_suite_id", None),
        ("e6_substitute", False),
        ("xlam_substitute", False),
        ("portfolio_evidence", False),
        ("claim_boundary", CLAIM_BOUNDARY),
    ):
        if expected is None:
            value = data.get(field_name)
            if value is not None:
                if value in REJECTED_SUBSTITUTES:
                    errors.append("related benchmark or xLAM substitution is rejected")
                else:
                    errors.append("any non-null substitute_suite_id is rejected")
        elif data.get(field_name) != expected:
            errors.append(f"{field_name} must be {expected!r}")

    metrics = {
        "schema_valid": not errors,
        "primary_eval": True,
        "receipt_proven_heldout": bool(heldout_valid and manifest_valid and eval_hash_valid),
        "native_verifier_valid": native_valid,
        "wandb_valid": wandb_valid,
        "tinker_valid": tinker_valid,
        "hf_valid": hf_valid,
        "cost_valid": cost_valid,
        "e6_substitute": False,
        "xlam_substitute": False,
        "portfolio_evidence": False,
        "paid_launch_authorized": False,
    }
    return ReceiptValidationResult(not errors, tuple(errors), metrics, receipt_hash)


def _cli_payload() -> dict[str, Any]:
    boundary = build_offline_e8_boundary()
    return {"boundary": boundary, "validation": validate_e8_boundary(boundary).to_dict()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the blocked offline E8 boundary")
    args = parser.parse_args(argv)
    payload = _cli_payload()
    if args.json:
        print(_canonical_json(payload))
    else:
        print(
            "E8 Life-Sci-Bench boundary: "
            f"{payload['validation']['ok']} "
            "(metadata-only; no held-out receipt or paid call)"
        )
    return 0 if payload["validation"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
