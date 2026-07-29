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
OFFLINE_PACKET_PATH = REPO_ROOT / "zvf-program/experiments-next/offline_falsification_packet.json"


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
    assert result["offline_packet_status"] == "not_run"
    assert result["offline_stage_count"] == 3
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


def test_offline_packet_is_decision_complete_but_inert() -> None:
    verifier = load_verifier()
    packet = json.loads(OFFLINE_PACKET_PATH.read_text(encoding="utf-8"))

    assert verifier.verify_offline_packet(packet) == {"status": "not_run", "stage_count": 3}
    assert (
        packet["label_contract"]["registered_reward_name"]
        != packet["label_contract"]["independent_target_name"]
    )
    assert packet["training_amendment_boundary"]["status"] == "required_before_s3_s5"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda packet: packet.__setitem__("status", "passed"),
        lambda packet: packet["label_contract"].__setitem__(
            "one_class_rule", "Pool one-class strata into the aggregate."
        ),
        lambda packet: packet["primary_analysis"].__setitem__(
            "minimum_meaningful_log_loss_reduction_nats_per_completion", 0.0
        ),
        lambda packet: packet["stage_receipts"][2].__setitem__("status", "passed"),
        lambda packet: packet["training_amendment_boundary"].__setitem__(
            "matched_input_rule", "Require byte-identical realized completions."
        ),
    ],
)
def test_offline_packet_rejects_undefined_or_fabricated_decisions(mutation) -> None:
    verifier = load_verifier()
    packet = json.loads(OFFLINE_PACKET_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(packet)
    mutation(candidate)

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_offline_packet(candidate)


def test_payload_is_bound_to_the_protocol_file(tmp_path: Path) -> None:
    verifier = load_verifier()
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"fabricated": true}', encoding="utf-8")

    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_contract(payload, REPO_ROOT, unrelated)


@pytest.mark.parametrize("payload", [[], None, "not-an-object"])
def test_public_contract_verifier_rejects_non_objects(payload) -> None:
    verifier = load_verifier()
    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_contract(payload, REPO_ROOT)


def test_semantic_rewrites_of_pinned_subcontracts_are_rejected() -> None:
    verifier = load_verifier()
    ledger = json.loads(THEORY_LEDGER_PATH.read_text(encoding="utf-8"))
    ledger["claims"][-1]["transfer_status"] = "diagnostic_analogy_only"
    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_theory_ledger(ledger)

    packet = json.loads(OFFLINE_PACKET_PATH.read_text(encoding="utf-8"))
    packet["label_contract"]["one_class_rule"] = (
        "NOT_IDENTIFIABLE is never required; every one-class stratum passes."
    )
    with pytest.raises(verifier.FollowupContractError):
        verifier.verify_offline_packet(packet)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"passed","status":"not_run"}', encoding="utf-8")

    with pytest.raises(verifier.FollowupContractError):
        verifier.load_json(path)


def passing_s2_receipt() -> dict:
    digest = "a" * 64
    return {
        "schema_version": "s2-offline-alignment-receipt-v1",
        "stage_id": "S2_offline_alignment",
        "input_hashes": {
            "prompt_manifest_sha256": digest,
            "completion_manifest_sha256": digest,
            "registered_reward_sha256": digest,
            "independent_checker_sha256": digest,
            "independent_label_manifest_sha256": digest,
            "split_manifest_sha256": digest,
            "categorical_level_manifest_sha256": digest,
            "auxiliary_score_implementation_sha256": digest,
        },
        "class_counts_by_stratum": {
            "all_correct_R": {
                "group_count": 2000,
                "Y_ind_0_count": 100,
                "Y_ind_1_count": 1900,
            },
            "all_wrong_R": {
                "group_count": 2000,
                "Y_ind_0_count": 1900,
                "Y_ind_1_count": 100,
            },
        },
        "contrast_results": [
            {
                "contrast_id": contrast_id,
                "point_estimate_nats_per_completion": 0.02,
                "lower_98_75_bound": 0.005,
            }
            for contrast_id in (
                "spectral_vs_control",
                "entropic_vs_control",
                "spectral_vs_placebo",
                "entropic_vs_placebo",
            )
        ],
        "checker_independence_receipt": {
            "checker_id": "independent-checker-v1",
            "implementation_sha256": digest,
            "used_for_policy_optimization": False,
            "used_for_auxiliary_score_construction": False,
        },
        "checker_blinding_receipt": {
            "arm_visible": False,
            "auxiliary_score_visible": False,
            "terminal_reward_visible": False,
            "seed_visible": False,
            "checkpoint_selection_visible": False,
        },
        "adjudication_receipt": {
            "primary_label": "raw_Y_ind",
            "sensitivity_label": "adjudicated_Y_ind",
            "sample_n_per_task_reward_disagreement_stratum": 100,
            "seed": 20260731,
            "third_procedure_blinded": True,
        },
        "power_receipt": {
            "achieved_power": 0.85,
            "required_total_groups_per_stratum": 7000,
        },
        "overall_decision": "PASS",
    }


def test_s2_receipt_adjudicator_computes_fail_closed_decisions() -> None:
    verifier = load_verifier()
    packet = json.loads(OFFLINE_PACKET_PATH.read_text(encoding="utf-8"))
    receipt = passing_s2_receipt()
    assert verifier.adjudicate_s2_receipt(packet, receipt) == "PASS"

    not_identifiable = copy.deepcopy(receipt)
    not_identifiable["class_counts_by_stratum"]["all_wrong_R"]["Y_ind_1_count"] = 0
    not_identifiable["overall_decision"] = "NOT_IDENTIFIABLE"
    assert verifier.adjudicate_s2_receipt(packet, not_identifiable) == "NOT_IDENTIFIABLE"

    not_feasible = copy.deepcopy(receipt)
    not_feasible["power_receipt"]["achieved_power"] = 0.7
    not_feasible["overall_decision"] = "NOT_FEASIBLE"
    assert verifier.adjudicate_s2_receipt(packet, not_feasible) == "NOT_FEASIBLE"

    failed = copy.deepcopy(receipt)
    failed["contrast_results"][0]["point_estimate_nats_per_completion"] = 0.005
    failed["overall_decision"] = "FAIL"
    assert verifier.adjudicate_s2_receipt(packet, failed) == "FAIL"

    fabricated = copy.deepcopy(receipt)
    fabricated["contrast_results"][0]["lower_98_75_bound"] = -0.001
    with pytest.raises(verifier.FollowupContractError):
        verifier.adjudicate_s2_receipt(packet, fabricated)
