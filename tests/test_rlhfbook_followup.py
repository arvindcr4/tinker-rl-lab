from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "zvf-program/experiments-next/verify_rlhfbook_followup.py"
PROTOCOL_PATH = REPO_ROOT / "zvf-program/experiments-next/rlhfbook_followup_preregistration.json"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_rlhfbook_followup", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_followup_contract_is_inert_and_evidence_bounded() -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    result = verifier.verify_contract(payload, REPO_ROOT)

    assert result["status"] == "POSTTRAINING_FOUNDATIONS_FOLLOWUP_CONTRACT_PASS"
    assert result["gpu_authorized"] is False
    assert isinstance(result["live_checkout_matches_accepted_source"], bool)
    assert result["live_checkout_matches_accepted_source"] is (
        result["live_objective_sha256"] == result["accepted_objective_sha256"]
    )


def test_proxy_signal_cannot_complete_the_learning_claim() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    boundary = payload["decision_rules"]["claim_boundary"]

    assert "cannot support a learning" in boundary
    assert payload["evaluation_contract"]["proxy_metrics_are_not_quality_metrics"] is True
    assert payload["authorization"]["amendment_required_before_execution"] is True


def test_course_foundations_are_bound_to_assumptions_and_diagnostics() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    course = payload["course_binding"]
    hypothesis_ids = {item["id"] for item in payload["hypotheses"]}
    telemetry = set(payload["required_telemetry"])

    assert course["url"] == "https://harvard-cs2824-s26.github.io/"
    assert course["source_commit"] == "5dcc34e3b861da632371645fb05aebb12a40d23c"
    assert "do not transfer" in course["use_boundary"]
    assert {
        "H5_coverage",
        "H6_distribution_shift",
        "H7_error_attribution",
    } <= hypothesis_ids
    assert {
        "correct_completion_coverage",
        "importance_weight_effective_sample_size",
        "fisher_quadratic_step",
        "approximation_error_proxy",
        "verifier_error_rate",
    } <= telemetry
    assert payload["decision_rules"]["stage_order"][1] == "S1_foundations_mapping"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("course_binding"),
        lambda payload: payload["decision_rules"].pop("theory_boundary"),
    ],
)
def test_missing_foundations_sections_fail_closed(mutation) -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(payload)
    mutation(candidate)

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_contract(candidate, REPO_ROOT)
