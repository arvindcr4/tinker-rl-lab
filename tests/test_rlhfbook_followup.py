from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPO_ROOT / "zvf-program/experiments-next/verify_rlhfbook_followup.py"
PROTOCOL_PATH = REPO_ROOT / "zvf-program/experiments-next/rlhfbook_followup_preregistration.json"
THEORY_LEDGER_PATH = REPO_ROOT / "zvf-program/experiments-next/theory_transfer_ledger.json"


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

    assert result["status"] == "POSTTRAINING_FOUNDATIONS_CONTRACT_LINT_PASS"
    assert result["gpu_authorized"] is False
    assert result["promotion_authorized"] is False
    assert result["verified_payload_sha256"] == verifier.canonical_json_sha256(payload)
    assert result["theory_claim_count"] == 7
    assert result["theory_source_file_count"] == 12
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
        lambda payload: payload["scope"]["must_not_modify_or_relabel"].remove(
            "zvf-program/flagship/paper/review_bundle.zip"
        ),
        lambda payload: next(
            stage for stage in payload["stages"] if stage["id"] == "S1_foundations_mapping"
        ).__setitem__("evidence", ""),
    ],
)
def test_missing_foundations_sections_fail_closed(mutation) -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(payload)
    mutation(candidate)

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_contract(candidate, REPO_ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("book_binding"),
        lambda payload: payload["book_binding"].__setitem__("chapters", None),
        lambda payload: payload["hypotheses"].__setitem__(0, "not-an-object"),
        lambda payload: payload["hypotheses"][0].pop("claim"),
        lambda payload: payload["evaluation_contract"].__setitem__(
            "answer_checks", ["same_verifier", "same_verifier"]
        ),
        lambda payload: payload["scope"]["live_checkout_observation"].__setitem__(
            "observed_sha256", "not-a-hash"
        ),
        lambda payload: payload["scope"].__setitem__("frozen_campaign", "not-r4-2"),
        lambda payload: payload["decision_rules"].__setitem__(
            "claim_boundary", "Proxy metrics may establish learning."
        ),
    ],
)
def test_malformed_or_semantically_weakened_contract_fails_closed(mutation) -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(payload)
    mutation(candidate)

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_contract(candidate, REPO_ROOT)


def test_theory_ledger_blocks_unsupported_promotion() -> None:
    verifier = load_verifier()
    ledger = json.loads(THEORY_LEDGER_PATH.read_text(encoding="utf-8"))
    assert verifier.verify_theory_ledger(ledger) == {
        "claim_count": 7,
        "source_file_count": 12,
    }

    candidate = copy.deepcopy(ledger)
    candidate["claims"][0]["transfer_status"] = "verified_theorem"
    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_theory_ledger(candidate)


def test_theory_ledger_source_manifest_is_exact() -> None:
    verifier = load_verifier()
    ledger = json.loads(THEORY_LEDGER_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(ledger)
    candidate["source_manifests"][0]["files"][0]["sha256"] = "0" * 64

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_theory_ledger(candidate)
