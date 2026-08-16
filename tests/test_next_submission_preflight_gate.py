from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/verify_preflight_matrix.py"
SPEC = importlib.util.spec_from_file_location("next_submission_preflight_gate", PATH)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

PROTOCOL = GATE.load_json(ROOT / "zvf-program/next-submission/preregistration.json")
RESULTS = ROOT / "zvf-program/next-submission/results/preflight/results"


KNOWN_SEAMS = {
    "gsm8k/contrast_early_stop_g2_to_g8:mixed_reward_optimizer_update",
    "gsm8k/contrast_early_stop_g2_to_g8:homogeneous_early_stop",
    "math500/contrast_early_stop_g2_to_g8:mixed_reward_optimizer_update",
    "math500/contrast_early_stop_g2_to_g8:homogeneous_early_stop",
    "math500/grpo_g8:mixed_reward_optimizer_update",
    "gsm8k/grpo_g8:mixed_reward_optimizer_update",
}


def current_receipts() -> list[Path]:
    return sorted(RESULTS.glob("*.json"))


def copy_receipts(tmp_path: Path) -> list[Path]:
    paths = []
    for source in current_receipts():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        paths.append(target)
    return paths


def mutate_receipt(path: Path, mutation) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutation(receipt)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_current_matrix_verifies_infrastructure_and_reports_a_consistent_gate():
    report = GATE.evaluate_matrix(PROTOCOL, current_receipts())
    assert report["infrastructure_matrix_verified"] is True
    assert report["receipt_count"] == len(current_receipts())
    missing = report["missing_scientific_seams"]
    assert set(missing) <= KNOWN_SEAMS
    if missing:
        assert report["scientific_seams_verified"] is False
        assert report["confirmatory_execution_gate"] == "blocked"
    else:
        assert report["scientific_seams_verified"] is True
        assert report["confirmatory_execution_gate"] == "pass"


def test_gate_rejects_incomplete_matrix():
    receipts = [p for p in current_receipts() if "math500__grpo_g8" not in p.name]
    with pytest.raises(GATE.PreflightGateError, match="matrix is incomplete"):
        GATE.evaluate_matrix(PROTOCOL, receipts)


def test_gate_rejects_unreconciled_remote_receipt(tmp_path):
    paths = copy_receipts(tmp_path)
    mutate_receipt(
        paths[0], lambda receipt: receipt["remote_verification"].update(hf_private=False)
    )
    with pytest.raises(GATE.PreflightGateError, match="HF repository is not private"):
        GATE.evaluate_matrix(PROTOCOL, paths)


def test_gate_rejects_scientific_stack_drift(tmp_path):
    paths = copy_receipts(tmp_path)

    def drift(receipt):
        receipt["stack_fingerprint"] = "0" * 64
        receipt["payload"]["run_config"]["stack_fingerprint"] = "0" * 64
        receipt["manifest"]["run_config"]["stack_fingerprint"] = "0" * 64

    mutate_receipt(paths[0], drift)
    with pytest.raises(GATE.PreflightGateError, match="one frozen scientific stack"):
        GATE.evaluate_matrix(PROTOCOL, paths)


def test_gate_passes_only_after_every_cell_exercises_a_real_update(tmp_path):
    paths = copy_receipts(tmp_path)

    for path in paths:

        def add_update(receipt):
            audit = copy.deepcopy(receipt["payload"]["audit_record"])
            groups = audit["rollout_groups"]
            audit["updated_groups"] = 1
            audit["mixed_fraction"] = 1 / groups
            remaining = 1 - audit["mixed_fraction"]
            audit["all_correct_fraction"] = remaining / 2
            audit["all_wrong_fraction"] = remaining / 2
            receipt["payload"]["audit_record"] = audit
            receipt["manifest"]["audit_record"] = copy.deepcopy(audit)

        mutate_receipt(path, add_update)

    report = GATE.evaluate_matrix(PROTOCOL, paths)
    assert report["scientific_seams_verified"] is True
    assert report["confirmatory_execution_gate"] == "pass"
    assert report["missing_scientific_seams"] == []
