from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

from .protocol import PilotProtocol, canonical_fingerprint


CHECKPOINT_STEPS = (20, 40, 60, 80, 100)
EVALUATION_STEPS = (0, 20, 40, 60, 80, 100)
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
    _require(isinstance(fingerprint, str) and bool(SHA256.fullmatch(fingerprint)), f"{label} has invalid fingerprint")
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


def validate_corpus_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
) -> dict[str, Any]:
    _require(manifest.get("schema_version") == "flagship-pilot-corpus-v1", "wrong corpus schema")
    _require(manifest.get("status") == "complete", "corpus is not complete")
    _require(manifest.get("protocol_sha256") == protocol.sha256, "corpus protocol hash mismatch")
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
    _require(isinstance(groups, list) and len(groups) == expected_groups, "corpus must contain 100 groups")
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
        _require(group.get("active_rows") == expected_active, f"corpus group {index} active-row mismatch")
        if regime == "balanced_equal_length":
            _require(group.get("selected_length_cv") == 0.0, f"balanced group {index} CV is not zero")
        else:
            _require(
                float(group.get("selected_length_cv", -1.0)) >= 0.35,
                f"filtered group {index} CV is below 0.35",
            )
        charged = group.get("charged_generated_tokens")
        _require(isinstance(charged, int) and charged > 0, f"corpus group {index} token charge invalid")
        group_tokens += charged
    _require(len(set(group_fingerprints)) == expected_groups, "corpus group fingerprints are not unique")

    rejected = manifest.get("rejected_generated_tokens")
    _require(isinstance(rejected, int) and rejected >= 0, "corpus rejected-token charge invalid")
    total = manifest.get("charged_generated_tokens")
    _require(total == group_tokens + rejected, "corpus charged-token ledger does not sum")
    _require(total <= int(contract["charged_generated_token_ceiling"]), "corpus exceeds token ceiling")

    files = manifest.get("artifact_files")
    _require(isinstance(files, dict) and files, "corpus artifact file hashes are missing")
    for path, digest in files.items():
        _require(isinstance(path, str) and path, "corpus artifact path is invalid")
        _require(isinstance(digest, str) and bool(SHA256.fullmatch(digest)), f"invalid corpus file hash: {path}")

    wandb = manifest.get("wandb")
    tracking = contract["tracking"]
    _require(isinstance(wandb, dict), "corpus W&B identity is missing")
    _require(wandb.get("project") == tracking["wandb_project"], "corpus W&B project mismatch")
    _require(isinstance(wandb.get("run_id"), str) and wandb["run_id"], "corpus W&B run ID missing")
    _require(isinstance(wandb.get("run_url"), str) and wandb["run_url"], "corpus W&B URL missing")

    normalized = copy.deepcopy(dict(manifest))
    normalized["fingerprint"] = _validated_fingerprint(manifest, label="corpus")
    return normalized


def validate_checkpoint_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    _require(manifest.get("schema_version") == "flagship-pilot-checkpoint-v1", "wrong checkpoint schema")
    step = manifest.get("step")
    _require(step in CHECKPOINT_STEPS, "checkpoint step is not in the frozen cadence")
    unit = plan["unit"]
    _require(manifest.get("unit_fingerprint") == plan["fingerprint"], "checkpoint unit hash mismatch")
    _require(manifest.get("protocol_sha256") == plan["protocol"]["sha256"], "checkpoint protocol hash mismatch")
    _require(manifest.get("corpus_fingerprint") == corpus["fingerprint"], "checkpoint corpus hash mismatch")
    for field in ("condition", "regime", "seed"):
        _require(manifest.get(field) == unit[field], f"checkpoint {field} mismatch")
    _require(manifest.get("replay_cursor") == step, "checkpoint replay cursor mismatch")
    _require(manifest.get("gradient_receipt_count") == step, "checkpoint gradient receipt count mismatch")

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
        _require(isinstance(digest, str) and bool(SHA256.fullmatch(digest)), f"invalid checkpoint hash: {path}")

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
    _require(record.get("gradient_receipt_count") == 100, "unit does not contain 100 gradient receipts")
    _require(tuple(record.get("checkpoint_steps", ())) == CHECKPOINT_STEPS, "unit checkpoint set is incomplete")

    evaluations = record.get("evaluations")
    _require(isinstance(evaluations, list) and len(evaluations) == len(EVALUATION_STEPS), "unit evaluation set is incomplete")
    _require(tuple(item.get("step") for item in evaluations) == EVALUATION_STEPS, "unit evaluation cadence mismatch")
    for item in evaluations:
        _require(item.get("heldout_n") == 128, f"evaluation step {item.get('step')} held-out size mismatch")
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
    _require(isinstance(wandb, dict) and wandb.get("state") == "finished", "W&B run is not finished")
    _require(isinstance(wandb.get("run_id"), str) and wandb["run_id"], "W&B run ID is missing")
    tracking = plan["runtime"].get("tracking")
    if tracking is not None:
        _require(wandb.get("project") == tracking["wandb_project"], "W&B project mismatch")
    hub = record.get("hugging_face")
    _require(isinstance(hub, dict) and hub.get("private") is True, "Hugging Face repository is not private")
    _require(hub.get("repo") == plan["identity"]["hf_repo"], "Hugging Face repository mismatch")
    _require(isinstance(hub.get("commit"), str) and bool(GIT_SHA.fullmatch(hub["commit"])), "Hugging Face commit invalid")
    _require(tuple(hub.get("checkpoint_steps", ())) == CHECKPOINT_STEPS, "remote checkpoint set is incomplete")
    for field in ("final_adapter_sha256", "manifest_sha256"):
        digest = hub.get(field)
        _require(isinstance(digest, str) and bool(SHA256.fullmatch(digest)), f"remote {field} invalid")

    normalized = copy.deepcopy(dict(record))
    normalized["fingerprint"] = _validated_fingerprint(record, label="full unit record")
    return normalized
