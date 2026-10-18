"""Synthetic v2 receipts; retained historical artifacts are never rewritten."""

from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from tests._shared_fakes import load_module
from tests.test_next_submission_preflight import valid_payload

ROOT = Path(__file__).resolve().parents[1]
GATE = load_module(
    "next_submission_preflight_gate",
    ROOT / "zvf-program/next-submission/verify_preflight_matrix.py",
)
PROTOCOL = GATE.load_json(GATE.DEFAULT_PROTOCOL)


def synthetic_matrix(tmp_path):
    protocol = copy.deepcopy(PROTOCOL)
    paths = []
    for task, arm in sorted(GATE.expected_cells(protocol)):
        manifest, payload, request = copy.deepcopy(valid_payload())
        request.update(
            schema_version="aiml-next-preflight-request-v1",
            task=task,
            arm=arm,
            protocol_sha256=GATE.sha256(GATE.DEFAULT_PROTOCOL),
            seed=1009,
        )
        request = GATE.common.seal_request(
            request,
            decoder=request["decoder"],
            tracking=request["tracking"],
            source_files=manifest["source_files"],
        )
        for filename, keys in GATE.SOURCE_BINDINGS.items():
            protocol["bindings"][keys[0]] = manifest["source_files"][filename]
        audit = payload["audit_record"]
        audit.update(
            task_id=task,
            arm_id=arm,
            seed=1009,
            generated_rollouts=16 if arm == "grpo_g8" else 10,
            telemetry_version="generated-and-optimizer-observed-v2",
            update_eligible_groups=1,
            updated_groups=1,
            truncated_generated_rollouts=0,
            unknown_finish_rollouts=0,
            update_observations=[
                {
                    "optimizer_step": 1,
                    "active_loss_tokens": 32,
                    "finite_gradient_norm": 0.25,
                    "parameter_changed": True,
                    "finite_parameters": True,
                    "optimizer_step_observed": True,
                    "update_applied": True,
                    "update_eligible_groups": 1,
                }
            ],
        )
        config = payload["run_config"]
        config.update(
            task=task,
            arm=arm,
            seed=1009,
            unit_fingerprint=request["fingerprint"],
            stack_fingerprint=request["stack_fingerprint"],
            protocol_sha256=request["protocol_sha256"],
        )
        row = next(t for t in protocol["tasks"] if t["task_id"] == task)
        config["dataset_bindings"] = {
            "training_dataset": row["training_dataset"],
            "training_revision": row["training_dataset_revision"],
            "evaluation_dataset": row.get("evaluation_dataset", row["training_dataset"]),
            "evaluation_revision": row.get(
                "evaluation_dataset_revision", row["training_dataset_revision"]
            ),
        }
        manifest.update(run_config=copy.deepcopy(config), audit_record=copy.deepcopy(audit))
        payload["remote"] = {
            "hf_repo": "owner/private",
            "hf_commit": "e" * 40,
            "wandb_run_id": "unit",
            "wandb_run_url": "https://wandb.ai/unit",
        }
        receipt = GATE.common.completed_receipt(
            request,
            result=payload,
            manifest=manifest,
            verification={
                "hf_private": True,
                "hf_repo": "owner/private",
                "hf_commit": "e" * 40,
                "hf_files": ["run_manifest.json", "final/adapter_model.safetensors"],
                "wandb": {
                    "state": "finished",
                    "run_id": "unit",
                    "run_url": "https://wandb.ai/unit",
                },
            },
            cleanup={
                "session_absent_verified": False,
                "compute_terminal_verified": True,
                "provider_observation": "COMPLETED",
                "resource_id": "job-unit",
            },
            return_code=0,
        )
        path = tmp_path / f"{task}-{arm}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)
    return protocol, paths


def mutate(path, fn):
    receipt = json.loads(path.read_text())
    fn(receipt)
    path.write_text(json.dumps(receipt))


def test_gate_accepts_consistent_synthetic_update_declarations(tmp_path):
    protocol, paths = synthetic_matrix(tmp_path)
    assert GATE.evaluate_matrix(protocol, paths)["confirmatory_execution_gate"] == "pass"


def test_gate_rejects_incomplete_matrix(tmp_path):
    protocol, paths = synthetic_matrix(tmp_path)
    with pytest.raises(GATE.PreflightGateError, match="matrix is incomplete"):
        GATE.evaluate_matrix(protocol, paths[:-1])


def test_gate_rejects_unreconciled_remote_receipt(tmp_path):
    protocol, paths = synthetic_matrix(tmp_path)
    mutate(paths[0], lambda r: r["remote_verification"].update(hf_private=False))
    with pytest.raises(GATE.PreflightGateError, match="HF repository is not private"):
        GATE.evaluate_matrix(protocol, paths)


@pytest.mark.parametrize("field", ["source_commit", "protocol_sha256"])
def test_gate_rejects_missing_revision_fields(tmp_path, field):
    protocol, paths = synthetic_matrix(tmp_path)
    mutate(paths[0], lambda r: r.pop(field))
    with pytest.raises(GATE.PreflightGateError, match=field):
        GATE.evaluate_matrix(protocol, paths)


@pytest.mark.parametrize(("field", "value"), [("generated_rollouts", 0), ("updated_groups", 99)])
def test_gate_rejects_impossible_counts(tmp_path, field, value):
    protocol, paths = synthetic_matrix(tmp_path)

    def invalid(r):
        r["payload"]["audit_record"][field] = value
        r["manifest"]["audit_record"][field] = value

    mutate(paths[0], invalid)
    with pytest.raises(GATE.PreflightGateError):
        GATE.evaluate_matrix(protocol, paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "unrelated/model"), ("optimizer_steps", 0), ("objective_fingerprint", "f" * 64)],
)
def test_gate_binds_runtime_to_registered_experiment(tmp_path, field, value):
    protocol, paths = synthetic_matrix(tmp_path)

    def invalid(r):
        r["payload"]["run_config"][field] = value
        r["manifest"]["run_config"][field] = value

    mutate(paths[0], invalid)
    with pytest.raises(GATE.PreflightGateError):
        GATE.evaluate_matrix(protocol, paths)


def test_repeated_fingerprint_strings_are_not_attestation(tmp_path):
    protocol, paths = synthetic_matrix(tmp_path)

    def invalid(r):
        r["stack_fingerprint"] = "f" * 64
        for key in ("payload", "manifest"):
            r[key]["run_config"]["stack_fingerprint"] = "f" * 64

    mutate(paths[0], invalid)
    with pytest.raises(GATE.PreflightGateError, match="manifest"):
        GATE.evaluate_matrix(protocol, paths)


def test_a004_eligibility_is_not_an_observed_update(tmp_path):
    protocol, paths = synthetic_matrix(tmp_path)
    for path in paths:

        def legacy(r):
            for key in ("payload", "manifest"):
                r[key]["audit_record"].pop("telemetry_version")

        mutate(path, legacy)
    report = GATE.evaluate_matrix(protocol, paths)
    assert report["confirmatory_execution_gate"] == "blocked"
    assert report["scientific_seams_verified"] is False
