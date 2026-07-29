from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    CHECKPOINT_STEPS,
    validate_corpus_checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_corpus_manifest,
    validate_full_record,
)
from .evaluation import validate_evidence
from .flops import REQUIRED_TRAINING_PHASES
from .protocol import (
    REPO_ROOT,
    PilotProtocol,
    PilotUnit,
    build_screening_plan,
    sha256_file,
)
from .remote_core import expected_runtime_versions
from .remote_training import _heldout_source


class VerificationError(RuntimeError):
    """Remote pilot evidence does not meet the frozen acceptance contract."""


SMOKE_PREFIX = "FPILOT_SMOKE_RESULT "


def verify_preflight_log(
    *, protocol: PilotProtocol, log_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    if not log_path.is_file():
        raise VerificationError(f"preflight log is missing: {log_path}")
    payloads = [
        line[len(SMOKE_PREFIX) :]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(SMOKE_PREFIX)
    ]
    if len(payloads) != 1:
        raise VerificationError(
            f"preflight log must contain exactly one smoke result; found {len(payloads)}"
        )
    try:
        smoke = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise VerificationError("preflight smoke result is not valid JSON") from exc
    if smoke.get("status") != "smoke_pass":
        raise VerificationError(f"preflight smoke status is not passing: {smoke.get('status')}")
    versions = smoke.get("runtime_versions")
    expected_versions = expected_runtime_versions(protocol)
    if (
        not isinstance(versions, dict)
        or {key: versions.get(key) for key in expected_versions} != expected_versions
    ):
        raise VerificationError("preflight runtime package versions do not match the protocol")
    python_version = str(versions.get("python", ""))
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?(?:.*)?", python_version)
    if match is None or not ((3, 11) <= tuple(map(int, match.groups())) < (3, 13)):
        raise VerificationError(
            f"preflight Python version is outside >=3.11,<3.13: {python_version}"
        )
    accelerator = smoke.get("accelerator")
    if not isinstance(accelerator, str) or "A100" not in accelerator.upper():
        raise VerificationError(f"preflight did not run on an A100: {accelerator}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(smoke.get("group_fingerprint", ""))):
        raise VerificationError("preflight group fingerprint is malformed")
    if (
        not isinstance(smoke.get("charged_generated_tokens"), int)
        or smoke["charged_generated_tokens"] <= 0
    ):
        raise VerificationError("preflight generated-token charge is non-positive")
    phase_flops = smoke.get("phase_flops")
    if not isinstance(phase_flops, dict):
        raise VerificationError("preflight phase FLOP receipt is missing")
    for phase in REQUIRED_TRAINING_PHASES:
        value = phase_flops.get(phase)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise VerificationError(f"preflight phase FLOPs are invalid: {phase}={value}")
    receipt = smoke.get("receipt")
    if not isinstance(receipt, dict):
        raise VerificationError("preflight training receipt is missing")
    if receipt.get("condition") != "intended_full" or receipt.get("step") != 1:
        raise VerificationError("preflight training receipt selected the wrong condition or step")
    if (
        receipt.get("gradient_relation") != "nonzero"
        or receipt.get("selected_vs_intended_relation") != "nonzero"
        or receipt.get("optimizer_update") != "applied"
    ):
        raise VerificationError("preflight training receipt is degenerate or did not update")
    positive_fields = (
        "intended_gradient_norm",
        "native_gradient_norm",
        "selected_gradient_norm",
    )
    finite_fields = (
        "selected_loss",
        "intended_loss",
        "native_loss",
        "gradient_cosine",
        "gradient_relative_l2",
        "selected_vs_intended_cosine",
        "selected_vs_intended_relative_l2",
        "optimizer_learning_rate",
    )
    for field in finite_fields + positive_fields:
        value = receipt.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise VerificationError(f"preflight receipt field is non-finite: {field}={value}")
    for field in positive_fields:
        if receipt[field] <= 0:
            raise VerificationError(f"preflight gradient norm is non-positive: {field}")
    for field in ("gradient_cosine", "selected_vs_intended_cosine"):
        if not -1.0 <= receipt[field] <= 1.0:
            raise VerificationError(
                f"preflight receipt cosine is outside [-1, 1]: {field}={receipt[field]}"
            )
    for field in ("gradient_relative_l2", "selected_vs_intended_relative_l2"):
        if receipt[field] < 0.0:
            raise VerificationError(
                f"preflight receipt relative L2 is negative: {field}={receipt[field]}"
            )
    for field in ("active_rows", "active_tokens"):
        if not isinstance(receipt.get(field), int) or receipt[field] <= 0:
            raise VerificationError(f"preflight receipt count is non-positive: {field}")
    acceptance = {
        "schema_version": "flagship-pilot-preflight-acceptance-v1",
        "status": "accepted",
        "protocol_sha256": protocol.sha256,
        "log_path": str(log_path),
        "smoke": smoke,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    return acceptance


def _verify_files(root: Path, files: Mapping[str, str], *, label: str) -> None:
    for relative, digest in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"{label} file hash mismatch: {relative}")


def _verify_source_manifest(
    stored: Mapping[str, str], *, expected: Mapping[str, str] | None = None
) -> None:
    if not stored:
        raise VerificationError("source manifest is empty")
    if expected is not None:
        if dict(stored) != dict(expected):
            raise VerificationError("source manifest does not match frozen corpus binding")
        return
    for relative, digest in stored.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"source manifest mismatch: {relative}")


def _verify_wandb_run(wandb_api: Any, identity: Mapping[str, Any], *, label: str) -> Any:
    entity = identity.get("entity")
    project = identity.get("project")
    run_id = identity.get("run_id")
    if not all(isinstance(value, str) and value for value in (entity, project, run_id)):
        raise VerificationError(f"{label} W&B identity is incomplete")
    run = wandb_api.run(f"{entity}/{project}/{run_id}")
    if run.state != "finished":
        raise VerificationError(f"{label} W&B run is not finished: {run.state}")
    return run


def verify_corpus_remote(
    *,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
    hf_api: Any,
    wandb_api: Any,
) -> tuple[dict[str, Any], Path, str]:
    from huggingface_hub import snapshot_download

    plan = build_screening_plan(
        protocol,
        PilotUnit(condition="intended_full", regime=regime, seed=seed),
    )
    repo = plan["identity"]["corpus_hf_repo"]
    info = hf_api.repo_info(repo_id=repo, repo_type="dataset")
    if info.private is not True:
        raise VerificationError(f"corpus repository is not private: {repo}")
    corpus_binding = plan["corpus_binding"]
    revision = (
        str(corpus_binding["hf_commit"])
        if corpus_binding["status"] == "accepted_complete"
        else str(info.sha)
    )
    root = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            token=hf_api.token,
        )
    )
    manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    manifest = validate_corpus_manifest(
        manifest,
        protocol=protocol,
        regime=regime,
        seed=seed,
    )
    _verify_files(root, manifest["artifact_files"], label="corpus")
    _verify_source_manifest(manifest["source_manifest"], expected=corpus_binding["source_manifest"])
    if corpus_binding["status"] == "accepted_complete":
        if manifest["fingerprint"] != corpus_binding["corpus_fingerprint"]:
            raise VerificationError("accepted corpus fingerprint changed from frozen binding")
    latest = manifest["corpus_resume"]["latest_checkpoint"]
    checkpoint = _verify_corpus_checkpoint_remote(
        repo=repo,
        revision=latest["hf_commit"],
        protocol=protocol,
        regime=regime,
        seed=seed,
        token=hf_api.token,
    )
    if checkpoint["fingerprint"] != latest["fingerprint"]:
        raise VerificationError("corpus latest checkpoint fingerprint mismatch")
    if checkpoint["completed_groups"] != latest["completed_groups"]:
        raise VerificationError("corpus latest checkpoint boundary mismatch")
    run = _verify_wandb_run(wandb_api, manifest["wandb"], label="corpus")
    if run.config.get("protocol_sha256") != corpus_binding["protocol_sha256"]:
        raise VerificationError("corpus W&B protocol hash mismatch")
    return manifest, root, revision


def _verify_corpus_checkpoint_remote(
    *,
    repo: str,
    revision: str,
    protocol: PilotProtocol,
    regime: str,
    seed: int,
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    prefix = "resume"
    manifest_path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=f"{prefix}/corpus_checkpoint_manifest.json",
        token=token,
    )
    manifest = validate_corpus_checkpoint_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8")),
        protocol=protocol,
        regime=regime,
        seed=seed,
    )
    for relative, digest in manifest["artifact_files"].items():
        local = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                filename=f"{prefix}/{relative}",
                token=token,
            )
        )
        if sha256_file(local) != digest:
            raise VerificationError(f"corpus checkpoint remote hash mismatch: {relative}")
    _verify_source_manifest(
        manifest["source_manifest"],
        expected=protocol.corpus_binding(regime, seed)["source_manifest"],
    )
    return manifest


def _verify_checkpoint_remote(
    *,
    repo: str,
    revision: str,
    step: int,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    prefix = f"checkpoints/step-{step}"
    manifest_path = hf_hub_download(
        repo_id=repo,
        repo_type="model",
        revision=revision,
        filename=f"{prefix}/checkpoint_manifest.json",
        token=token,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = validate_checkpoint_manifest(manifest, plan=plan, corpus=corpus)
    for relative, digest in manifest["files"].items():
        remote_path = f"{prefix}/{relative}"
        local = Path(
            hf_hub_download(
                repo_id=repo,
                repo_type="model",
                revision=revision,
                filename=remote_path,
                token=token,
            )
        )
        if sha256_file(local) != digest:
            raise VerificationError(f"checkpoint-{step} remote hash mismatch: {relative}")
    return manifest


def verify_unit_remote(
    *,
    protocol: PilotProtocol,
    condition: str,
    regime: str,
    seed: int,
    hf_api: Any,
    wandb_api: Any,
    output_path: Path | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, snapshot_download

    plan = build_screening_plan(
        protocol,
        PilotUnit(condition=condition, regime=regime, seed=seed),
    )
    corpus, _, corpus_commit = verify_corpus_remote(
        protocol=protocol,
        regime=regime,
        seed=seed,
        hf_api=hf_api,
        wandb_api=wandb_api,
    )
    repo = plan["identity"]["hf_repo"]
    info = hf_api.repo_info(repo_id=repo, repo_type="model")
    if info.private is not True:
        raise VerificationError(f"unit repository is not private: {repo}")
    latest_revision = str(info.sha)
    full_record_path = hf_hub_download(
        repo_id=repo,
        repo_type="model",
        revision=latest_revision,
        filename="full_record.json",
        token=hf_api.token,
    )
    full_record = json.loads(Path(full_record_path).read_text(encoding="utf-8"))
    full_record = validate_full_record(full_record, plan=plan, corpus=corpus)
    if full_record["manifest"]["plan"]["fingerprint"] != plan["fingerprint"]:
        raise VerificationError("remote run manifest contains a different unit plan")
    if full_record["manifest"]["corpus_commit"] != corpus_commit:
        raise VerificationError("unit run manifest corpus commit mismatch")
    _verify_source_manifest(full_record["manifest"]["source_manifest"])

    artifact_revision = full_record["hugging_face"]["commit"]
    artifact_root = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            revision=artifact_revision,
            allow_patterns=["final/**", "evaluations/**", "run_manifest.json"],
            token=hf_api.token,
        )
    )
    adapter = artifact_root / "final/adapter/adapter_model.safetensors"
    run_manifest = artifact_root / "run_manifest.json"
    if sha256_file(adapter) != full_record["hugging_face"]["final_adapter_sha256"]:
        raise VerificationError("final adapter hash mismatch")
    if sha256_file(run_manifest) != full_record["hugging_face"]["manifest_sha256"]:
        raise VerificationError("run manifest hash mismatch")

    questions, answers, source_indices = _heldout_source(protocol, regime)
    for evaluation in full_record["evaluations"]:
        evidence = artifact_root / evaluation["evidence_path"]
        summary = validate_evidence(
            evidence,
            regime=regime,
            questions=questions,
            answers=answers,
            source_indices=source_indices,
        )
        if summary["evidence_sha256"] != evaluation["evidence_sha256"]:
            raise VerificationError(f"held-out evidence hash mismatch at step {evaluation['step']}")
        if summary["accuracy"] != evaluation["accuracy"]:
            raise VerificationError(f"held-out accuracy mismatch at step {evaluation['step']}")

    checkpoint_manifests = [
        _verify_checkpoint_remote(
            repo=repo,
            revision=latest_revision,
            step=step,
            plan=plan,
            corpus=corpus,
            token=hf_api.token,
        )
        for step in CHECKPOINT_STEPS
    ]
    run = _verify_wandb_run(wandb_api, full_record["wandb"], label="unit")
    remote_plan = run.config.get("plan", {})
    if remote_plan.get("fingerprint") != plan["fingerprint"]:
        raise VerificationError("unit W&B plan fingerprint mismatch")
    if run.config.get("corpus_fingerprint") != corpus["fingerprint"]:
        raise VerificationError("unit W&B corpus fingerprint mismatch")

    receipt = {
        "schema_version": "flagship-pilot-acceptance-v1",
        "status": "accepted",
        "unit": plan["unit"],
        "unit_fingerprint": plan["fingerprint"],
        "corpus_fingerprint": corpus["fingerprint"],
        "corpus_commit": corpus_commit,
        "hf_repo": repo,
        "hf_latest_commit": latest_revision,
        "hf_artifact_commit": artifact_revision,
        "checkpoint_manifest_fingerprints": {
            str(manifest["step"]): manifest["fingerprint"] for manifest in checkpoint_manifests
        },
        "wandb_run_id": run.id,
        "final_accuracy": full_record["evaluations"][-1]["accuracy"],
        "full_record": full_record,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
