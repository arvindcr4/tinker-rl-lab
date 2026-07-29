from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping, Sequence

from .protocol import PilotProtocol, canonical_fingerprint


CHECKPOINT_STEPS = (20, 40, 60, 80, 100)
EVALUATION_STEPS = (0, 20, 40, 60, 80, 100)
CORPUS_CHECKPOINT_GROUPS = (20, 40, 60, 80)
REQUIRED_CHECKPOINT_FILES = (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
    "training_state.json",
    "token_flop_ledger.json",
    "source_manifest.json",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ArtifactValidationError(RuntimeError):
    """A pilot artifact cannot be accepted or resumed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactValidationError(message)


def _validated_fingerprint(record: Mapping[str, Any], *, label: str) -> str:
    fingerprint = record.get("fingerprint")
    _require(
        isinstance(fingerprint, str) and bool(SHA256.fullmatch(fingerprint)),
        f"{label} has invalid fingerprint",
    )
    payload = copy.deepcopy(dict(record))
    del payload["fingerprint"]
    actual = canonical_fingerprint(payload)
    _require(actual == fingerprint, f"{label} fingerprint mismatch")
    return fingerprint


def with_fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(record))
    payload.pop("fingerprint", None)
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_gradient_comparison(
    receipt: Mapping[str, Any],
    *,
    expected_step: int,
    relation_field: str,
    cosine_field: str,
    relative_l2_field: str,
    left_norm: float,
    right_norm: float,
    left_zero_relation: str,
    right_zero_relation: str,
) -> None:
    if left_norm == 0.0 and right_norm == 0.0:
        expected_relation = "joint_zero"
    elif left_norm == 0.0:
        expected_relation = left_zero_relation
    elif right_norm == 0.0:
        expected_relation = right_zero_relation
    else:
        expected_relation = "nonzero"
    _require(
        receipt.get(relation_field) == expected_relation,
        f"gradient receipt {expected_step} {relation_field} is inconsistent with its norms",
    )
    cosine = receipt.get(cosine_field)
    relative_l2 = receipt.get(relative_l2_field)
    if expected_relation != "nonzero":
        _require(
            cosine is None and relative_l2 is None,
            f"gradient receipt {expected_step} zero-vector comparison must use null diagnostics",
        )
        return
    _require(
        _finite_number(cosine) and -1.0 <= cosine <= 1.0,
        f"gradient receipt {expected_step} cosine is outside [-1, 1]: {cosine_field}={cosine}",
    )
    _require(
        _finite_number(relative_l2) and relative_l2 >= 0.0,
        f"gradient receipt {expected_step} relative L2 is negative or non-finite: "
        f"{relative_l2_field}={relative_l2}",
    )


def _validate_gradient_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_step: int,
    expected_condition: str,
    expected_group_fingerprint: str,
) -> None:
    _require(receipt.get("step") == expected_step, "gradient receipt step sequence mismatch")
    _require(
        receipt.get("condition") == expected_condition,
        f"gradient receipt {expected_step} condition mismatch",
    )
    _require(
        receipt.get("group_fingerprint") == expected_group_fingerprint,
        f"gradient receipt {expected_step} replay-group mismatch",
    )
    finite_fields = (
        "selected_loss",
        "intended_loss",
        "native_loss",
        "optimizer_learning_rate",
    )
    for field in finite_fields:
        value = receipt.get(field)
        _require(
            _finite_number(value),
            f"gradient receipt {expected_step} field is non-finite: {field}={value}",
        )
    norms: dict[str, float] = {}
    for field in (
        "intended_gradient_norm",
        "native_gradient_norm",
        "selected_gradient_norm",
    ):
        value = receipt.get(field)
        _require(
            _finite_number(value) and value >= 0.0,
            f"gradient receipt {expected_step} gradient norm is negative or non-finite: {field}",
        )
        norms[field] = float(value)
    _validate_gradient_comparison(
        receipt,
        expected_step=expected_step,
        relation_field="gradient_relation",
        cosine_field="gradient_cosine",
        relative_l2_field="gradient_relative_l2",
        left_norm=norms["intended_gradient_norm"],
        right_norm=norms["native_gradient_norm"],
        left_zero_relation="intended_zero",
        right_zero_relation="native_zero",
    )
    _validate_gradient_comparison(
        receipt,
        expected_step=expected_step,
        relation_field="selected_vs_intended_relation",
        cosine_field="selected_vs_intended_cosine",
        relative_l2_field="selected_vs_intended_relative_l2",
        left_norm=norms["selected_gradient_norm"],
        right_norm=norms["intended_gradient_norm"],
        left_zero_relation="selected_zero",
        right_zero_relation="intended_zero",
    )
    expected_update = "no_op_zero_gradient" if norms["selected_gradient_norm"] == 0.0 else "applied"
    _require(
        receipt.get("optimizer_update") == expected_update,
        f"gradient receipt {expected_step} optimizer update is inconsistent with selected norm",
    )
    for field in ("active_rows", "active_tokens"):
        value = receipt.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"gradient receipt {expected_step} count is non-positive: {field}",
        )


def validate_corpus_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
) -> dict[str, Any]:
    corpus_binding = protocol.corpus_binding(regime, seed)
    _require(manifest.get("schema_version") == "flagship-pilot-corpus-v2", "wrong corpus schema")
    _require(manifest.get("status") == "complete", "corpus is not complete")
    _require(
        manifest.get("protocol_sha256") == corpus_binding["protocol_sha256"],
        "corpus protocol hash mismatch",
    )
    _require(manifest.get("regime") == regime, "corpus regime mismatch")
    _require(manifest.get("seed") == seed, "corpus seed mismatch")

    runtime = protocol.payload["runtime"]
    contract = runtime["execution_contract"]
    regime_contract = protocol.payload["regimes"][regime]
    _require(manifest.get("model") == runtime["model"], "corpus model identity mismatch")
    _require(manifest.get("dataset") == regime_contract["dataset"], "corpus dataset mismatch")
    _require(
        manifest.get("dataset_revision") == regime_contract["dataset_revision"],
        "corpus dataset revision mismatch",
    )
    expected_order = contract["train_order_hash"][regime][str(seed)]
    _require(manifest.get("train_order_hash") == expected_order, "corpus train-order hash mismatch")

    groups = manifest.get("groups")
    expected_groups = int(contract["accepted_groups_per_corpus"])
    _require(
        isinstance(groups, list) and len(groups) == expected_groups,
        "corpus must contain 100 groups",
    )
    group_fingerprints: list[str] = []
    group_tokens = 0
    expected_active = 8 if regime == "balanced_equal_length" else 6
    for index, group in enumerate(groups):
        _require(group.get("index") == index, f"corpus group {index} index mismatch")
        fingerprint = group.get("fingerprint")
        _require(
            isinstance(fingerprint, str) and bool(SHA256.fullmatch(fingerprint)),
            f"corpus group {index} has invalid fingerprint",
        )
        group_fingerprints.append(fingerprint)
        _require(
            group.get("active_rows") == expected_active, f"corpus group {index} active-row mismatch"
        )
        if regime == "balanced_equal_length":
            _require(
                group.get("selected_length_cv") == 0.0, f"balanced group {index} CV is not zero"
            )
        else:
            _require(
                float(group.get("selected_length_cv", -1.0)) >= 0.35,
                f"filtered group {index} CV is below 0.35",
            )
        charged = group.get("charged_generated_tokens")
        _require(
            isinstance(charged, int) and charged > 0, f"corpus group {index} token charge invalid"
        )
        group_tokens += charged
    _require(
        len(set(group_fingerprints)) == expected_groups, "corpus group fingerprints are not unique"
    )

    rejected = manifest.get("rejected_generated_tokens")
    _require(isinstance(rejected, int) and rejected >= 0, "corpus rejected-token charge invalid")
    total = manifest.get("charged_generated_tokens")
    _require(total == group_tokens + rejected, "corpus charged-token ledger does not sum")
    _require(
        total <= int(contract["charged_generated_token_ceiling"]), "corpus exceeds token ceiling"
    )

    files = manifest.get("artifact_files")
    _require(isinstance(files, dict) and files, "corpus artifact file hashes are missing")
    for path, digest in files.items():
        _require(isinstance(path, str) and path, "corpus artifact path is invalid")
        _require(
            isinstance(digest, str) and bool(SHA256.fullmatch(digest)),
            f"invalid corpus file hash: {path}",
        )
    sources = manifest.get("source_manifest")
    _require(
        sources == corpus_binding["source_manifest"],
        "corpus source manifest does not match frozen reuse binding",
    )

    wandb = manifest.get("wandb")
    tracking = contract["tracking"]
    _require(isinstance(wandb, dict), "corpus W&B identity is missing")
    _require(wandb.get("project") == tracking["wandb_project"], "corpus W&B project mismatch")
    _require(isinstance(wandb.get("run_id"), str) and wandb["run_id"], "corpus W&B run ID missing")
    _require(isinstance(wandb.get("run_url"), str) and wandb["run_url"], "corpus W&B URL missing")

    resume = manifest.get("corpus_resume")
    _require(isinstance(resume, dict), "corpus resume provenance is missing")
    _require(
        resume.get("schema_version") == "flagship-pilot-corpus-resume-v1",
        "wrong corpus resume schema",
    )
    _require(resume.get("enabled") is True, "corpus resume contract is not enabled")
    _require(
        tuple(resume.get("checkpoint_groups", ())) == CORPUS_CHECKPOINT_GROUPS,
        "corpus checkpoint cadence mismatch",
    )
    resume_count = resume.get("resume_count")
    _require(isinstance(resume_count, int) and resume_count >= 0, "corpus resume count is invalid")
    attempts = resume.get("attempts")
    _require(
        isinstance(attempts, list) and len(attempts) == resume_count + 1,
        "corpus attempt ledger does not match resume count",
    )
    for attempt_index, attempt in enumerate(attempts):
        _require(isinstance(attempt, dict), f"corpus attempt {attempt_index} is invalid")
        _require(
            isinstance(attempt.get("run_id"), str) and attempt["run_id"],
            f"corpus attempt {attempt_index} run ID is missing",
        )
        _require(
            isinstance(attempt.get("run_url"), str) and attempt["run_url"],
            f"corpus attempt {attempt_index} run URL is missing",
        )
        start_group = attempt.get("start_group")
        completed_through = attempt.get("completed_through")
        _require(
            isinstance(start_group, int) and start_group in (0, *CORPUS_CHECKPOINT_GROUPS),
            f"corpus attempt {attempt_index} start group is invalid",
        )
        _require(
            isinstance(completed_through, int)
            and start_group <= completed_through <= expected_groups,
            f"corpus attempt {attempt_index} completion boundary is invalid",
        )
    _require(
        attempts[-1]["run_id"] == wandb["run_id"],
        "corpus final W&B run is not the last resume attempt",
    )
    latest = resume.get("latest_checkpoint")
    _require(isinstance(latest, dict), "corpus latest checkpoint provenance is missing")
    _require(
        latest.get("completed_groups") == CORPUS_CHECKPOINT_GROUPS[-1],
        "corpus latest checkpoint does not cover group 80",
    )
    _require(
        isinstance(latest.get("fingerprint"), str)
        and bool(SHA256.fullmatch(latest["fingerprint"])),
        "corpus latest checkpoint fingerprint is invalid",
    )
    _require(
        isinstance(latest.get("hf_commit"), str) and bool(GIT_SHA.fullmatch(latest["hf_commit"])),
        "corpus latest checkpoint commit is invalid",
    )

    normalized = copy.deepcopy(dict(manifest))
    normalized["fingerprint"] = _validated_fingerprint(manifest, label="corpus")
    return normalized


def validate_corpus_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
) -> dict[str, Any]:
    """Validate an immutable, resumable corpus prefix before it is trusted."""

    corpus_binding = protocol.corpus_binding(regime, seed)

    _require(
        manifest.get("schema_version") == "flagship-pilot-corpus-checkpoint-v1",
        "wrong corpus checkpoint schema",
    )
    _require(manifest.get("status") == "partial", "corpus checkpoint is not partial")
    _require(
        manifest.get("protocol_sha256") == corpus_binding["protocol_sha256"],
        "corpus checkpoint protocol hash mismatch",
    )
    _require(manifest.get("regime") == regime, "corpus checkpoint regime mismatch")
    _require(manifest.get("seed") == seed, "corpus checkpoint seed mismatch")

    runtime = protocol.payload["runtime"]
    contract = runtime["execution_contract"]
    regime_contract = protocol.payload["regimes"][regime]
    _require(manifest.get("model") == runtime["model"], "corpus checkpoint model identity mismatch")
    _require(
        manifest.get("dataset") == regime_contract["dataset"], "corpus checkpoint dataset mismatch"
    )
    _require(
        manifest.get("dataset_revision") == regime_contract["dataset_revision"],
        "corpus checkpoint dataset revision mismatch",
    )
    expected_order = contract["train_order_hash"][regime][str(seed)]
    _require(
        manifest.get("train_order_hash") == expected_order,
        "corpus checkpoint train-order hash mismatch",
    )

    completed = manifest.get("completed_groups")
    _require(
        isinstance(completed, int) and completed in CORPUS_CHECKPOINT_GROUPS,
        "corpus checkpoint boundary is not in the amended cadence",
    )
    groups = manifest.get("groups")
    _require(
        isinstance(groups, list) and len(groups) == completed,
        "corpus checkpoint group prefix is incomplete",
    )
    expected_active = 8 if regime == "balanced_equal_length" else 6
    charged_total = 0
    fingerprints: list[str] = []
    for index, group in enumerate(groups):
        _require(group.get("index") == index, f"corpus checkpoint group {index} index mismatch")
        _require(
            isinstance(group.get("source_row_index"), int),
            f"corpus checkpoint group {index} source row is invalid",
        )
        fingerprint = group.get("fingerprint")
        _require(
            isinstance(fingerprint, str) and bool(SHA256.fullmatch(fingerprint)),
            f"corpus checkpoint group {index} fingerprint is invalid",
        )
        fingerprints.append(fingerprint)
        _require(
            group.get("active_rows") == expected_active,
            f"corpus checkpoint group {index} active-row mismatch",
        )
        if regime == "balanced_equal_length":
            _require(
                group.get("selected_length_cv") == 0.0,
                f"corpus checkpoint balanced group {index} CV is not zero",
            )
        else:
            _require(
                float(group.get("selected_length_cv", -1.0)) >= 0.35,
                f"corpus checkpoint filtered group {index} CV is below 0.35",
            )
        artifact_path = group.get("artifact_path")
        _require(
            artifact_path == f"groups/group-{index:03d}.pt",
            f"corpus checkpoint group {index} artifact path mismatch",
        )
        charged = group.get("charged_generated_tokens")
        _require(
            isinstance(charged, int) and charged > 0,
            f"corpus checkpoint group {index} token charge is invalid",
        )
        charged_total += charged
    _require(len(set(fingerprints)) == completed, "corpus checkpoint fingerprints are not unique")
    _require(
        manifest.get("charged_generated_tokens") == charged_total,
        "corpus checkpoint charged-token ledger does not sum",
    )
    _require(
        charged_total <= int(contract["charged_generated_token_ceiling"]),
        "corpus checkpoint exceeds token ceiling",
    )

    flop_ledger = manifest.get("flop_ledger")
    _require(isinstance(flop_ledger, dict), "corpus checkpoint FLOP ledger is missing")
    expected_profiled = tuple(
        int(step) for step in contract["flop_counter"]["profiled_steps"] if int(step) <= completed
    )
    _require(
        tuple(flop_ledger.get("profiled_steps", ())) == expected_profiled,
        "corpus checkpoint profiler coverage mismatch",
    )
    for field in ("profiled_generated_tokens", "profiled_generation_flops"):
        value = flop_ledger.get(field)
        _require(
            isinstance(value, (int, float)) and value > 0,
            f"corpus checkpoint {field} is missing",
        )

    files = manifest.get("artifact_files")
    expected_files = {"source_manifest.json"} | {
        f"groups/group-{index:03d}.pt" for index in range(completed)
    }
    _require(isinstance(files, dict), "corpus checkpoint artifact hashes are missing")
    _require(set(files) == expected_files, "corpus checkpoint artifact file set mismatch")
    for path, digest in files.items():
        _require(
            isinstance(digest, str) and bool(SHA256.fullmatch(digest)),
            f"invalid corpus checkpoint file hash: {path}",
        )

    sources = manifest.get("source_manifest")
    _require(isinstance(sources, dict) and sources, "corpus checkpoint source manifest is missing")
    _require(
        sources == corpus_binding["source_manifest"],
        "corpus checkpoint source manifest does not match frozen reuse binding",
    )
    for path, digest in sources.items():
        _require(isinstance(path, str) and path, "corpus checkpoint source path is invalid")
        _require(
            isinstance(digest, str) and bool(SHA256.fullmatch(digest)),
            f"invalid corpus checkpoint source hash: {path}",
        )

    resume_count = manifest.get("resume_count")
    _require(
        isinstance(resume_count, int) and resume_count >= 0,
        "corpus checkpoint resume count is invalid",
    )
    attempts = manifest.get("attempts")
    _require(
        isinstance(attempts, list) and len(attempts) == resume_count + 1,
        "corpus checkpoint attempt ledger does not match resume count",
    )
    for attempt_index, attempt in enumerate(attempts):
        _require(isinstance(attempt, dict), f"corpus checkpoint attempt {attempt_index} is invalid")
        _require(
            isinstance(attempt.get("run_id"), str) and attempt["run_id"],
            f"corpus checkpoint attempt {attempt_index} run ID is missing",
        )
        _require(
            isinstance(attempt.get("run_url"), str) and attempt["run_url"],
            f"corpus checkpoint attempt {attempt_index} run URL is missing",
        )
        start_group = attempt.get("start_group")
        completed_through = attempt.get("completed_through")
        _require(
            isinstance(start_group, int) and start_group in (0, *CORPUS_CHECKPOINT_GROUPS),
            f"corpus checkpoint attempt {attempt_index} start group is invalid",
        )
        _require(
            isinstance(completed_through, int) and start_group <= completed_through <= completed,
            f"corpus checkpoint attempt {attempt_index} completion boundary is invalid",
        )
    wall_clock = manifest.get("wall_clock_seconds")
    _require(
        isinstance(wall_clock, (int, float)) and wall_clock > 0,
        "corpus checkpoint wall-clock ledger is invalid",
    )

    normalized = copy.deepcopy(dict(manifest))
    normalized["fingerprint"] = _validated_fingerprint(
        manifest, label=f"corpus-checkpoint-{completed}"
    )
    return normalized


def validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        manifest.get("schema_version") == "flagship-pilot-checkpoint-v1", "wrong checkpoint schema"
    )
    step = manifest.get("step")
    _require(step in CHECKPOINT_STEPS, "checkpoint step is not in the frozen cadence")
    unit = plan["unit"]
    _require(
        manifest.get("unit_fingerprint") == plan["fingerprint"], "checkpoint unit hash mismatch"
    )
    _require(
        manifest.get("protocol_sha256") == plan["protocol"]["sha256"],
        "checkpoint protocol hash mismatch",
    )
    _require(
        manifest.get("corpus_fingerprint") == corpus["fingerprint"],
        "checkpoint corpus hash mismatch",
    )
    for field in ("condition", "regime", "seed"):
        _require(manifest.get(field) == unit[field], f"checkpoint {field} mismatch")
    _require(manifest.get("replay_cursor") == step, "checkpoint replay cursor mismatch")
    _require(
        manifest.get("gradient_receipt_count") == step, "checkpoint gradient receipt count mismatch"
    )

    ledger = manifest.get("token_flop_ledger")
    _require(isinstance(ledger, dict), "checkpoint token/FLOP ledger is missing")
    _require(
        ledger.get("charged_generated_tokens") == corpus["charged_generated_tokens"],
        "checkpoint corpus token charge mismatch",
    )
    for phase in ("policy_forward_flops", "diagnostic_backward_flops", "optimizer_backward_flops"):
        value = ledger.get(phase)
        _require(isinstance(value, (int, float)) and value > 0, f"checkpoint {phase} is missing")

    files = manifest.get("files")
    _require(isinstance(files, dict), "checkpoint file manifest is missing")
    _require(
        set(REQUIRED_CHECKPOINT_FILES).issubset(files),
        "checkpoint file set is incomplete",
    )
    evaluation_steps = tuple(manifest.get("evaluation_steps", ()))
    expected_evidence = {f"evaluations/step-{value:03d}.jsonl" for value in evaluation_steps}
    observed_evidence = {path for path in files if path.startswith("evaluations/")}
    _require(observed_evidence == expected_evidence, "checkpoint evaluation evidence set mismatch")
    unexpected = set(files) - set(REQUIRED_CHECKPOINT_FILES) - expected_evidence
    _require(not unexpected, f"checkpoint contains unexpected files: {sorted(unexpected)}")
    for path, digest in files.items():
        _require(
            isinstance(digest, str) and bool(SHA256.fullmatch(digest)),
            f"invalid checkpoint hash: {path}",
        )

    normalized = copy.deepcopy(dict(manifest))
    normalized["fingerprint"] = _validated_fingerprint(manifest, label=f"checkpoint-{step}")
    return normalized


def greatest_compatible_checkpoint(
    manifests: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    for manifest in manifests:
        step = manifest.get("step", "unknown")
        try:
            valid.append(validate_checkpoint_manifest(manifest, plan=plan, corpus=corpus))
        except ArtifactValidationError as exc:
            errors.append(f"checkpoint-{step}: {exc}")
    if not valid:
        return None, tuple(errors)
    return max(valid, key=lambda record: int(record["step"])), tuple(errors)


def validate_full_record(
    record: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    _require(record.get("schema_version") == "flagship-pilot-unit-v1", "wrong unit record schema")
    _require(record.get("status") == "completed", "unit record is not complete")
    unit = plan["unit"]
    _require(record.get("unit_fingerprint") == plan["fingerprint"], "unit fingerprint mismatch")
    _require(record.get("corpus_fingerprint") == corpus["fingerprint"], "unit corpus mismatch")
    for field in ("condition", "regime", "seed"):
        _require(record.get(field) == unit[field], f"unit {field} mismatch")
    _require(record.get("training_steps") == 100, "unit does not contain 100 training steps")
    _require(
        record.get("gradient_receipt_count") == 100, "unit does not contain 100 gradient receipts"
    )
    _require(
        tuple(record.get("checkpoint_steps", ())) == CHECKPOINT_STEPS,
        "unit checkpoint set is incomplete",
    )

    manifest = record.get("manifest")
    _require(isinstance(manifest, dict), "unit run manifest is missing")
    _require(
        manifest.get("schema_version") == "flagship-pilot-run-manifest-v1",
        "wrong unit run-manifest schema",
    )
    _require(
        manifest.get("corpus_fingerprint") == corpus["fingerprint"],
        "unit run-manifest corpus mismatch",
    )
    receipts = manifest.get("gradient_receipts")
    _require(
        isinstance(receipts, list) and len(receipts) == 100,
        "unit run manifest does not contain 100 gradient receipts",
    )
    for step, receipt in enumerate(receipts, start=1):
        _require(isinstance(receipt, dict), f"gradient receipt {step} is malformed")
        _validate_gradient_receipt(
            receipt,
            expected_step=step,
            expected_condition=unit["condition"],
            expected_group_fingerprint=corpus["groups"][step - 1]["fingerprint"],
        )

    evaluations = record.get("evaluations")
    _require(
        isinstance(evaluations, list) and len(evaluations) == len(EVALUATION_STEPS),
        "unit evaluation set is incomplete",
    )
    _require(
        tuple(item.get("step") for item in evaluations) == EVALUATION_STEPS,
        "unit evaluation cadence mismatch",
    )
    for item in evaluations:
        _require(
            item.get("heldout_n") == 128,
            f"evaluation step {item.get('step')} held-out size mismatch",
        )
        _require(
            isinstance(item.get("evidence_sha256"), str)
            and bool(SHA256.fullmatch(item["evidence_sha256"])),
            f"evaluation step {item.get('step')} evidence hash invalid",
        )

    ledger = record.get("token_flop_ledger")
    _require(isinstance(ledger, dict), "unit token/FLOP ledger is missing")
    _require(
        ledger.get("charged_generated_tokens") == corpus["charged_generated_tokens"],
        "unit corpus token charge mismatch",
    )
    for phase in (
        "replay_generation_flops",
        "policy_forward_flops",
        "diagnostic_backward_flops",
        "optimizer_backward_flops",
    ):
        value = ledger.get(phase)
        _require(isinstance(value, (int, float)) and value > 0, f"unit {phase} is missing")

    wandb = record.get("wandb")
    _require(
        isinstance(wandb, dict) and wandb.get("state") == "finished", "W&B run is not finished"
    )
    _require(isinstance(wandb.get("run_id"), str) and wandb["run_id"], "W&B run ID is missing")
    tracking = plan["runtime"].get("tracking")
    if tracking is not None:
        _require(wandb.get("project") == tracking["wandb_project"], "W&B project mismatch")
    hub = record.get("hugging_face")
    _require(
        isinstance(hub, dict) and hub.get("private") is True,
        "Hugging Face repository is not private",
    )
    _require(hub.get("repo") == plan["identity"]["hf_repo"], "Hugging Face repository mismatch")
    _require(
        isinstance(hub.get("commit"), str) and bool(GIT_SHA.fullmatch(hub["commit"])),
        "Hugging Face commit invalid",
    )
    _require(
        tuple(hub.get("checkpoint_steps", ())) == CHECKPOINT_STEPS,
        "remote checkpoint set is incomplete",
    )
    for field in ("final_adapter_sha256", "manifest_sha256"):
        digest = hub.get(field)
        _require(
            isinstance(digest, str) and bool(SHA256.fullmatch(digest)), f"remote {field} invalid"
        )

    normalized = copy.deepcopy(dict(record))
    normalized["fingerprint"] = _validated_fingerprint(record, label="full unit record")
    return normalized
