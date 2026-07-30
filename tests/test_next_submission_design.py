from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "zvf-program/next-submission"
SPEC = importlib.util.spec_from_file_location("next_submission_verifier", HERE / "verify_design.py")
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture()
def contract():
    protocol = VERIFIER.load_json(HERE / "preregistration.json")
    claims = VERIFIER.load_json(HERE / "claim_ledger.json")
    results = VERIFIER.load_json(HERE / "results_contract.json")
    blueprint = (HERE / "MANUSCRIPT_BLUEPRINT.md").read_text(encoding="utf-8")
    return protocol, claims, results, blueprint


def verify(candidate, *, bindings: bool = False):
    protocol, claims, results, blueprint = candidate
    return VERIFIER.verify_contract(
        protocol,
        claims,
        results,
        blueprint,
        repo_root=ROOT,
        check_bindings=bindings,
    )


def test_frozen_design_passes_with_hash_bindings(contract):
    report = verify(contract, bindings=True)
    assert report["status"] == "NEXT_SUBMISSION_DESIGN_CONTRACT_PASS"
    assert report["initial_training_units"] == 64
    assert report["gpu_authorized"] is True
    assert report["result_claims_authorized"] is False


def test_gpu_execution_requires_the_bound_authorization_receipt(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["authorization"]["receipt_sha256"] = "0" * 64
    with pytest.raises(VERIFIER.DesignContractError, match="authorization receipt hash"):
        verify(candidate)


def test_execution_authorization_cannot_expand_to_publication(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["authorization"]["publish"] = True
    with pytest.raises(VERIFIER.DesignContractError, match="publication"):
        verify(candidate)


def test_missing_task_arm_cell_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["matrix"].pop()
    with pytest.raises(VERIFIER.DesignContractError, match="matrix"):
        verify(candidate)


def test_math_training_corpus_drift_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    math_task = next(row for row in candidate[0]["tasks"] if row["task_id"] == "math500")
    math_task["training_dataset"] = "HuggingFaceH4/MATH-500"
    with pytest.raises(VERIFIER.DesignContractError, match="MATH training"):
        verify(candidate)


def test_single_seed_design_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["paired_seed_plan"]["planning_seed_count"] = 1
    candidate[0]["paired_seed_plan"]["seeds"] = [211]
    candidate[0]["power_plan"]["planning_seed_count_after_inflation"] = 1
    with pytest.raises(VERIFIER.DesignContractError, match="single-seed|power"):
        verify(candidate)


def test_underpowered_seed_count_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["paired_seed_plan"]["planning_seed_count"] = 8
    candidate[0]["paired_seed_plan"]["seeds"] = candidate[0]["paired_seed_plan"]["seeds"][:8]
    candidate[0]["power_plan"]["planning_seed_count_after_inflation"] = 8
    with pytest.raises(VERIFIER.DesignContractError, match="power"):
        verify(candidate)


def test_missing_cost_power_reassessment_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["power_plan"]["cost_variance_source"] = "not planned"
    with pytest.raises(VERIFIER.DesignContractError, match="cost variance"):
        verify(candidate)


def test_relaxed_underpowered_cap_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["power_plan"]["cap_exceeded_verdict"] = "PROCEED_ANYWAY"
    with pytest.raises(VERIFIER.DesignContractError, match="fail-closed"):
        verify(candidate)


def test_use_inspired_without_external_receipt_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["contribution_policy"]["use_inspired_authorized"] = True
    with pytest.raises(VERIFIER.DesignContractError, match="use-inspired"):
        verify(candidate)


def test_achieved_claim_in_design_only_packet_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[1]["claims"][0]["status"] = "achieved"
    with pytest.raises(VERIFIER.DesignContractError, match="achieved"):
        verify(candidate)


def test_filename_in_main_result_table_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[2]["main_table_columns"].append("filename")
    with pytest.raises(VERIFIER.DesignContractError, match="filename"):
        verify(candidate)


def test_non_numeric_main_table_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[2]["numeric_main_table_columns"] = ["completed_seeds"]
    with pytest.raises(VERIFIER.DesignContractError, match="numeric"):
        verify(candidate)


def test_results_before_method_are_rejected(contract):
    candidate = copy.deepcopy(contract)
    order = candidate[0]["manuscript_contract"]["section_order"]
    method = order.index("Method")
    results = order.index("Results")
    order[method], order[results] = order[results], order[method]
    with pytest.raises(VERIFIER.DesignContractError, match="method"):
        verify(candidate)


def test_stale_e1_verdict_cannot_become_primary_evidence(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["prior_evidence_boundary"]["allowed_use"] = "DAPO DISAPPEARS is a primary result."
    with pytest.raises(VERIFIER.DesignContractError, match="prior evidence"):
        verify(candidate)


def test_source_commit_drift_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[0]["sources"]["rlhf_book"]["commit"] = "0" * 40
    with pytest.raises(VERIFIER.DesignContractError, match="RLHF Book commit"):
        verify(candidate)


def test_incomplete_seed_provenance_is_rejected(contract):
    candidate = copy.deepcopy(contract)
    candidate[2]["required_seed_row_fields"].remove("objective_fingerprint")
    with pytest.raises(VERIFIER.DesignContractError, match="provenance"):
        verify(candidate)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(VERIFIER.DesignContractError, match="duplicate JSON key"):
        VERIFIER.load_json(path)
