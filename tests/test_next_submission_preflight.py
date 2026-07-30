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
        "run_config": run_config,
        "heldout_trace": trace,
        "source_files": source_files,
    }
    result = {
        "schema_version": "aiml-next-preflight-result-v1",
        "evidence_class": "preflight-not-evidence",
        "audit_record": audit,
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


def test_result_log_parser_uses_last_marker():
    payload = {"schema_version": "aiml-next-preflight-result-v1"}
    lines = ["noise", "NEXT_PREFLIGHT_RESULT " + __import__("json").dumps(payload)]
    assert PREFLIGHT.result_from_log(lines) == payload
