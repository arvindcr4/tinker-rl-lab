from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_preflight", PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def valid_payload():
    request = {
        "task": "gsm8k",
        "arm": "contrast_early_stop_g2_to_g8",
        "seed": 211,
        "fingerprint": "a" * 64,
        "stack_fingerprint": "b" * 64,
        "source_commit": "c" * 40,
        "protocol_sha256": "d" * 64,
        "gpu": "L4",
        "provider": "huggingface_jobs",
        "hardware_flavor": "l4x1",
        "tracking_packages": {"trackio": "0.34.0"},
    }
    audit = {
        "task_id": "gsm8k",
        "arm_id": "contrast_early_stop_g2_to_g8",
        "seed": 211,
        "status": "complete",
        "evidence_tier": "preflight_not_scientific_evidence",
        "heldout_correct": 4,
        "heldout_n": 8,
        "heldout_accuracy": 0.5,
        "charged_generated_tokens": 100,
        "generated_rollouts": 10,
        "rollout_groups": 2,
        "updated_groups": 1,
        "all_wrong_fraction": 0.5,
        "all_correct_fraction": 0.0,
        "mixed_fraction": 0.5,
    }
    run_config = {
        "task": request["task"],
        "arm": request["arm"],
        "seed": request["seed"],
        "unit_fingerprint": request["fingerprint"],
        "stack_fingerprint": request["stack_fingerprint"],
        "source_commit": request["source_commit"],
        "protocol_sha256": request["protocol_sha256"],
        "max_steps": 1,
        "heldout_n": 8,
        "provider": request["provider"],
        "hardware_flavor": request["hardware_flavor"],
        "tracking_backends": ["wandb", "trackio"],
    }
    trace = [
        {"index": index, "correct": index < 4, "completion_sha256": f"{index:064x}"}
        for index in range(8)
    ]
    source_files = {
        path.name: PREFLIGHT.sha256_file(path)
        for path in (PREFLIGHT.REMOTE_SCRIPT, PREFLIGHT.SAMPLER, PREFLIGHT.TRL_ADAPTER)
    }
    manifest = {
        "schema_version": "aiml-next-preflight-run-v1",
        "status": "complete",
        "evidence_class": "preflight-not-evidence",
        "audit_record": audit,
        "run_config": copy.deepcopy(run_config),
        "heldout_trace": trace,
        "source_files": source_files,
        "runtime_versions": {
            "trl": "1.8.0",
            "transformers": "5.13.1",
            "datasets": "4.8.5",
            "peft": "0.19.1",
            "torchao": "0.17.0",
            "wandb": "0.28.0",
            "trackio": "0.34.0",
            "gpu": "NVIDIA L4",
        },
    }
    result = {
        "schema_version": "aiml-next-preflight-result-v1",
        "status": "completed",
        "evidence_class": "preflight-not-evidence",
        "audit_record": audit,
        "run_config": run_config,
        "runtime_versions": manifest["runtime_versions"],
    }
    return manifest, result, request


def test_valid_intervention_preflight_contract_passes():
    PREFLIGHT.validate_manifest(*valid_payload())


def test_preflight_cannot_be_promoted_to_scientific_evidence():
    manifest, result, request = copy.deepcopy(valid_payload())
    result["evidence_class"] = "confirmatory"
    with pytest.raises(RuntimeError, match="promoted"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_intervention_generation_formula_is_enforced():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["audit_record"]["generated_rollouts"] = 16
    result["audit_record"]["generated_rollouts"] = 16
    with pytest.raises(RuntimeError, match="G2-to-G8"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_source_hash_drift_is_rejected():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["source_files"]["contrast_sampler.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hashes"):
        PREFLIGHT.validate_manifest(manifest, result, request)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda manifest, result, request: manifest["audit_record"].update(status="running"),
            "status",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(task_id="math500"),
            "identity",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(updated_groups=99),
            "updated-group",
        ),
        (
            lambda manifest, result, request: manifest["audit_record"].update(
                charged_generated_tokens=1
            ),
            "charged generated tokens",
        ),
    ],
)
def test_semantically_invalid_audit_receipts_are_rejected(mutation, match):
    manifest, result, request = copy.deepcopy(valid_payload())
    mutation(manifest, result, request)
    result["audit_record"] = copy.deepcopy(manifest["audit_record"])
    with pytest.raises(RuntimeError, match=match):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_result_run_config_must_match_manifest():
    manifest, result, request = copy.deepcopy(valid_payload())
    result["run_config"]["task"] = "math500"
    with pytest.raises(RuntimeError, match="run configs differ"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_observed_gpu_and_versions_must_match_request():
    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["runtime_versions"]["gpu"] = "NVIDIA T4"
    result["runtime_versions"] = copy.deepcopy(manifest["runtime_versions"])
    with pytest.raises(RuntimeError, match="does not match requested"):
        PREFLIGHT.validate_manifest(manifest, result, request)

    manifest, result, request = copy.deepcopy(valid_payload())
    manifest["runtime_versions"]["trl"] = "1.9.0"
    result["runtime_versions"] = copy.deepcopy(manifest["runtime_versions"])
    with pytest.raises(RuntimeError, match="package mismatch"):
        PREFLIGHT.validate_manifest(manifest, result, request)


def test_result_log_parser_uses_last_marker():
    payload = {"schema_version": "aiml-next-preflight-result-v1"}
    lines = ["noise", "NEXT_PREFLIGHT_RESULT " + __import__("json").dumps(payload)]
    assert PREFLIGHT.result_from_log(lines) == payload


def test_failure_summary_is_bounded_and_strips_ansi():
    lines = [f"line {index}" for index in range(20)] + ["\x1b[31mquota rejected\x1b[0m"]
    summary = PREFLIGHT.failure_summary(lines, limit=3)
    assert summary == "line 18 | line 19 | quota rejected"


def test_hardware_retry_archives_incompatible_receipt(tmp_path):
    existing = {"fingerprint": "a" * 64, "status": "failed", "gpu": "A100"}

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    path = PREFLIGHT.archive_incompatible_result(
        output_dir=tmp_path,
        unit="gsm8k__adaptive__s211",
        existing=existing,
        new_fingerprint="b" * 64,
        atomic_json=write,
    )

    assert path == tmp_path / "results/history/gsm8k__adaptive__s211__aaaaaaaaaaaa.json"
    assert __import__("json").loads(path.read_text(encoding="utf-8")) == existing
