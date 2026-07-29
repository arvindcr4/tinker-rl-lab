#!/usr/bin/env python3
"""Validate the prospective RLHF Book and CS2824 follow-up without mutating r4-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "rlhfbook_followup_preregistration.json"
REPO_ROOT = HERE.parents[1]
EXPECTED_BOOK_COMMIT = "3624df9ef62177c2c3d6d824f5c2bb740f31041f"
EXPECTED_COURSE_COMMIT = "5dcc34e3b861da632371645fb05aebb12a40d23c"
EXPECTED_BOOK_REPOSITORY = "https://github.com/natolambert/rlhf-book"
EXPECTED_COURSE_REPOSITORY = "https://github.com/harvard-cs2824-s26/harvard-cs2824-s26.github.io"
EXPECTED_FROZEN_CAMPAIGN = "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2"
EXPECTED_ANSWER_CHECKS = {
    "registered_strict_parser",
    "independent_symbolic_or_numeric_verifier",
}
EXPECTED_FORMAT_PERTURBATIONS = {
    "registered_template",
    "semantically equivalent alternate answer template",
}
EXPECTED_THEORY_LEDGER_PATH = "zvf-program/experiments-next/theory_transfer_ledger.json"
EXPECTED_THEORY_LEDGER_SHA256 = "e53e8d5973670f6b13a0509415617a1d61367ca679786a450cddc66555a9034f"
EXPECTED_OFFLINE_PACKET_PATH = "zvf-program/experiments-next/offline_falsification_packet.json"
EXPECTED_OFFLINE_PACKET_SHA256 = "6a1933205f1218e3aca3081eef97e3fee02b828baafd8537239a4f907409c7fa"
EXPECTED_THEORY_LEDGER_CANONICAL_SHA256 = (
    "e69a285fb8adb86e9b455ccdc4bc79ad5b63cef81d209283e32ee6f73038e7e3"
)
EXPECTED_OFFLINE_PACKET_CANONICAL_SHA256 = (
    "dc2e67e4354db74f8942c3df89e50265354700dbf1ff9c0ec8fccc56776b3311"
)
EXPECTED_PROTOCOL_PAYLOAD_SHA256 = (
    "250521c15cc6c5ff6c4f0703537ae8a3fb68acbefc6d574037330c28454d81ae"
)
EXPECTED_AUDIT_BINDINGS = {
    "zvf-program/experiments-next/RLHFBOOK_IMPROVEMENT_AUDIT.md": (
        "b501ba7be0dcc47c07b78a3b610a86150216adbb6918f7ba31a68bfba491e907"
    ),
    "zvf-program/experiments-next/HARVARD_CS2824_IMPROVEMENT_AUDIT.md": (
        "78a58a159fc5b5a154928def4785341fec63385fb54dac7acefcac586025034a"
    ),
}
EXPECTED_THEORY_CLAIM_IDS = {
    "C1_group_contrast_and_loss_weighting",
    "C2_sparse_reward_stationarity_gap",
    "C3_distribution_mismatch_stationarity",
    "C4_kl_fisher_policy_geometry",
    "C5_approximation_estimation_decomposition",
    "C6_proxy_and_evaluation_confounds",
    "C7_offpolicy_reward_regression",
}
EXPECTED_TRANSFER_STATUS_BY_CLAIM = {
    "C1_group_contrast_and_loss_weighting": "empirical_hypothesis_only",
    "C2_sparse_reward_stationarity_gap": "diagnostic_analogy_only",
    "C3_distribution_mismatch_stationarity": "diagnostic_analogy_only",
    "C4_kl_fisher_policy_geometry": "diagnostic_analogy_only",
    "C5_approximation_estimation_decomposition": "diagnostic_analogy_only",
    "C6_proxy_and_evaluation_confounds": "empirical_hypothesis_only",
    "C7_offpolicy_reward_regression": "hypothesis_source_only",
}
EXPECTED_SOURCE_FILE_HASHES = {
    "book/chapters/06-policy-gradients.md": "6671f67b6d635ab8cc1f1859345edeead9a3f7489e5410a5c4653cb1b6c4e444",
    "book/chapters/14-over-optimization.md": "81736529e9237c079fa39d27a96635b45805af827cb287f5e2ee20959d4dfd81",
    "book/chapters/15-regularization.md": "3b93ac53551725c90a29780e731094e406bb61325804627ef3a2d1dd2858127e",
    "book/chapters/16-evaluation.md": "d9c1764b4cf85e869eac82a6a30f73a5578ddab4faf69c410c8e7d66ce5a4525",
    "book/chapters/appendix-c-practical.md": "c08f557a7f3e2ae80339f6bac9a8028ee22b39158f743008f86a4e7273879e1a",
    "slides/PG_global_conv1.pdf": "26309b138a546eff684ed586809919de9a0360f2d0aeb8fd3ad64ccc546cd86f",
    "slides/npg.pdf": "e88bf7b22fd56577179accaf828e232a6cc7af41aad3e7310cc5b3c254f01c9f",
    "slides/PG_global_conv2.pdf": "ee21ee7f5cc52e878007a643e23cbca1dac5ca8ea402a4fb1d8c94b858f64f5d",
    "slides/NPG_ppo.pdf": "309137fbfec2dc7c01cc1489aaa867f768b5e4a253bdea3359f633eb4529a8db",
    "slides/RLHF.pdf": "f12bd048818e817069cb0ef0f46ea90f22219ca3e3a9b016e748db1c3727194a",
    "slides/regressing_rewards.pdf": "381168dca1129f785618844c420018d7ec3c707c2b910d6a7fd5b04f24f1677a",
    "CS2824projects.html": "30ef1ea58da08abe7873564858feb95e8caae3d7f8061247b1092cf59edfcf41",
}


class FollowupContractError(RuntimeError):
    """Raised when the prospective protocol loses a required guardrail."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FollowupContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    require(isinstance(payload, dict), f"{path} must contain one JSON object")
    return payload


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for candidate, value in payload.items():
            if candidate == key:
                values.append(value)
            values.extend(_find_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_find_values(value, key))
    return values


def verify_theory_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(ledger, Mapping), "theory ledger must be an object")
    require(
        canonical_json_sha256(ledger) == EXPECTED_THEORY_LEDGER_CANONICAL_SHA256,
        "theory ledger semantic digest drift",
    )
    require(ledger.get("schema_version") == "theory-transfer-ledger-v1", "ledger schema drift")
    require(
        ledger.get("status") == "assumption_audit_only_not_evidence",
        "ledger evidence status drift",
    )
    require(ledger.get("promotion_authorized") is False, "ledger authorizes promotion")

    manifests = ledger["source_manifests"]
    require(isinstance(manifests, list) and len(manifests) == 2, "source manifests drift")
    manifest_ids = {item["manifest_id"] for item in manifests}
    require(
        manifest_ids == {"rlhf_book_3624df9", "harvard_cs2824_5dcc34e"},
        "source manifest identities drift",
    )
    flattened_hashes = {
        item["path"]: item["sha256"] for manifest in manifests for item in manifest["files"]
    }
    require(flattened_hashes == EXPECTED_SOURCE_FILE_HASHES, "source file manifest drift")

    claims = ledger["claims"]
    require(isinstance(claims, list), "theory claims must be a list")
    require(len(claims) == len(EXPECTED_THEORY_CLAIM_IDS), "duplicate theory claim")
    require({item["claim_id"] for item in claims} == EXPECTED_THEORY_CLAIM_IDS, "claim set drift")
    allowed_assumption_statuses = {"verified", "empirical_proxy", "unverified", "violated"}
    allowed_transfer_statuses = {
        "diagnostic_analogy_only",
        "empirical_hypothesis_only",
        "hypothesis_source_only",
    }
    required_claim_fields = {
        "claim_id",
        "claim_type",
        "statement",
        "source_refs",
        "primary_reference",
        "formal_domain",
        "assumptions",
        "llm_mapping",
        "transfer_status",
        "observable_proxy",
        "falsifier",
        "permitted_language",
    }
    required_mapping_fields = {
        "state",
        "action",
        "trajectory",
        "reward",
        "policy",
        "comparator",
        "reference_policy",
        "prompt_distribution",
        "rollout_distribution",
        "evaluation_distribution",
    }
    for claim in claims:
        require(isinstance(claim, Mapping), "theory claim is malformed")
        require(required_claim_fields <= set(claim), "theory claim field missing")
        require(
            all(
                isinstance(claim[field], str) and claim[field].strip()
                for field in required_claim_fields
                if field not in {"source_refs", "assumptions", "llm_mapping"}
            ),
            f"theory claim text missing for {claim.get('claim_id', '<unknown>')}",
        )
        require(
            claim["transfer_status"] in allowed_transfer_statuses
            and claim["transfer_status"] == EXPECTED_TRANSFER_STATUS_BY_CLAIM[claim["claim_id"]],
            "unsupported theorem transfer status",
        )
        require(claim["source_refs"], "theory claim source missing")
        require(
            all(
                ref["manifest_id"] in manifest_ids
                and ref["path"] in EXPECTED_SOURCE_FILE_HASHES
                and isinstance(ref["locator"], str)
                and ref["locator"].strip()
                for ref in claim["source_refs"]
            ),
            "theory claim source reference drift",
        )
        require(claim["assumptions"], "theory claim assumptions missing")
        require(
            all(
                isinstance(item["name"], str)
                and item["name"].strip()
                and item["status"] in allowed_assumption_statuses
                and isinstance(item["evidence_or_proxy"], str)
                and item["evidence_or_proxy"].strip()
                for item in claim["assumptions"]
            ),
            "theory assumption row malformed",
        )
        require(
            required_mapping_fields == set(claim["llm_mapping"]),
            "LLM mapping fields drift",
        )
        require(
            all(
                isinstance(value, str) and value.strip() for value in claim["llm_mapping"].values()
            ),
            "LLM mapping value missing",
        )

    return {"claim_count": len(claims), "source_file_count": len(flattened_hashes)}


def verify_offline_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(packet, Mapping), "offline packet must be an object")
    require(
        canonical_json_sha256(packet) == EXPECTED_OFFLINE_PACKET_CANONICAL_SHA256,
        "offline packet semantic digest drift",
    )
    require(
        packet.get("schema_version") == "offline-falsification-packet-v1",
        "offline packet schema drift",
    )
    require(packet.get("status") == "not_run", "offline packet is not prospective")
    require(packet.get("scope") == "offline_s0_s2_only", "offline packet scope drift")
    require(packet.get("gpu_authorized") is False, "offline packet authorizes GPU execution")
    require(
        packet.get("external_run_authorized") is False,
        "offline packet authorizes external execution",
    )
    require(packet.get("promotion_authorized") is False, "offline packet authorizes promotion")
    require(
        set(packet["theory_claim_ids"]) <= EXPECTED_THEORY_CLAIM_IDS,
        "offline packet references an unknown theory claim",
    )

    labels = packet["label_contract"]
    require(labels["unit"] == "completion", "independent label unit drift")
    require(
        labels["registered_reward_name"] == "R" and labels["independent_target_name"] == "Y_ind",
        "registered reward and independent target drift",
    )
    require(
        "not used to optimize" in labels["independence_requirement"], "label independence drift"
    )
    require("not arm" in labels["blinding"], "label blinding drift")
    require(
        "NOT_IDENTIFIABLE" in labels["one_class_rule"] and "never" in labels["one_class_rule"],
        "one-class non-identifiability rule drift",
    )

    data = packet["data_contract"]
    require(data["cluster_unit"] == "prompt_id", "cluster unit drift")
    require(
        data["development_fraction"] + data["untouched_test_fraction"] == 1.0,
        "offline split fractions drift",
    )
    require(
        type(data["cross_fit_folds_within_development"]) is int
        and data["cross_fit_folds_within_development"] == 5,
        "cross-fit fold count drift",
    )
    require(
        type(data["minimum_total_groups_per_terminal_reward_stratum"]) is int
        and data["minimum_total_groups_per_terminal_reward_stratum"] == 6667,
        "offline total group count drift",
    )
    require(
        type(data["minimum_development_groups_per_terminal_reward_stratum"]) is int
        and data["minimum_development_groups_per_terminal_reward_stratum"] == 4667,
        "offline development group count drift",
    )
    require(
        type(data["minimum_untouched_test_groups_per_terminal_reward_stratum"]) is int
        and data["minimum_untouched_test_groups_per_terminal_reward_stratum"] == 2000,
        "offline untouched-test group count drift",
    )
    require(
        set(data["required_terminal_reward_strata"]) == {"all_correct_R", "all_wrong_R"},
        "terminal-reward strata drift",
    )
    required_hashes = {
        "prompt_manifest_sha256",
        "completion_manifest_sha256",
        "registered_reward_sha256",
        "independent_checker_sha256",
        "independent_label_manifest_sha256",
        "split_manifest_sha256",
        "auxiliary_score_implementation_sha256",
    }
    require(
        set(data["required_hashes_before_analysis"]) == required_hashes,
        "offline input hash contract drift",
    )

    analysis = packet["primary_analysis"]
    require(
        analysis["minimum_meaningful_log_loss_reduction_nats_per_completion"] == 0.01,
        "minimum meaningful effect drift",
    )
    bootstrap = analysis["bootstrap"]
    require(bootstrap["unit"] == "prompt_id", "bootstrap unit drift")
    require(
        type(bootstrap["replicates"]) is int and bootstrap["replicates"] == 10000,
        "bootstrap replicate count drift",
    )
    require(
        type(bootstrap["seed"]) is int and bootstrap["seed"] == 20260729, "bootstrap seed drift"
    )
    require(bootstrap["familywise_confidence_level"] == 0.95, "familywise level drift")
    require(
        bootstrap["contrastwise_confidence_level"] == 0.9875,
        "contrastwise level drift",
    )
    require(
        "all four confirmatory contrasts" in bootstrap["multiplicity_rule"],
        "multiplicity rule drift",
    )
    require(
        "NOT_IDENTIFIABLE" not in analysis["arm_pass_rule"]
        and "Both-class support" in analysis["arm_pass_rule"],
        "arm pass rule permits non-identifiable strata",
    )
    estimator = analysis["estimator_specification"]
    require(estimator["model"] == "L2-penalized logistic regression", "estimator drift")
    require(estimator["penalty_C"] == 1.0, "regularization drift")
    require(estimator["hyperparameter_selection"].startswith("none"), "model selection drift")
    require(
        "five fold-model probabilities" in estimator["cross_fit_predictions"],
        "prediction rule drift",
    )
    require(len(analysis["confirmatory_contrasts"]) == 4, "confirmatory contrast family drift")

    power = packet["power_gate"]
    require(power["target_power"] == 0.8, "power target drift")
    require(power["familywise_alpha"] == 0.05, "power alpha drift")
    require(power["confirmatory_contrast_count"] == 4, "power contrast count drift")
    require(power["null_effect_nats_per_completion"] == 0.0, "power null drift")
    require(power["alternative_effect_nats_per_completion"] == 0.01, "power alternative drift")
    require(
        power["maximum_total_groups_per_terminal_reward_stratum"] == 10000,
        "power budget drift",
    )
    require("NOT_FEASIBLE" in power["decision_rule"], "power infeasibility rule drift")

    receipts = packet["stage_receipts"]
    require(
        [item["stage_id"] for item in receipts]
        == [
            "S0_isolation",
            "S1_foundations_mapping",
            "S2_offline_alignment",
        ],
        "offline receipt stage order drift",
    )
    require(all(item["status"] == "not_run" for item in receipts), "stage result fabricated")
    require(
        all(item["required_fields"] and item["path"].endswith(".json") for item in receipts),
        "stage receipt schema missing",
    )
    s2_fields = set(receipts[2]["required_fields"])
    require(
        {
            "checker_independence_receipt",
            "checker_blinding_receipt",
            "adjudication_receipt",
        }
        <= s2_fields,
        "checker governance receipts missing",
    )

    amendment = packet["training_amendment_boundary"]
    require(
        amendment["status"] == "required_before_s3_s5",
        "training amendment boundary drift",
    )
    require(len(amendment["must_specify"]) == 8, "training amendment threshold set drift")
    require(
        "byte-identical prompt schedules" in amendment["matched_input_rule"]
        and "arm-specific completion-manifest hashes" in amendment["matched_input_rule"],
        "on-policy matched-input rule drift",
    )
    return {"status": packet["status"], "stage_count": len(receipts)}


def _verify_contract(
    payload: Mapping[str, Any],
    repo_root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    require(isinstance(payload, Mapping), "protocol must be an object")
    require(
        canonical_json_sha256(payload) == EXPECTED_PROTOCOL_PAYLOAD_SHA256,
        "protocol semantic digest drift",
    )
    require(protocol_path.is_file(), "protocol path is missing")
    protocol_file_payload = load_json(protocol_path)
    require(
        canonical_json_sha256(protocol_file_payload) == canonical_json_sha256(payload),
        "verified payload does not match protocol file",
    )
    require(
        payload.get("schema_version") == "posttraining-foundations-followup-v3",
        "schema drift",
    )
    require(payload.get("status") == "proposed_not_authorized", "follow-up is not inert")

    book = payload["book_binding"]
    require(isinstance(book, Mapping), "book binding is missing or malformed")
    require(book["url"] == "https://rlhfbook.com/", "book URL drift")
    require(book["source_repository"] == EXPECTED_BOOK_REPOSITORY, "book repository drift")
    require(book["source_commit"] == EXPECTED_BOOK_COMMIT, "book commit drift")
    required_chapters = {
        "06-policy-gradients",
        "14-over-optimization",
        "15-regularization",
        "16-evaluation",
        "appendix-c-practical",
    }
    require(required_chapters <= set(book["chapters"]), "required book chapters are missing")

    course = payload.get("course_binding")
    require(isinstance(course, Mapping), "course binding is missing or malformed")
    require(course["url"] == "https://harvard-cs2824-s26.github.io/", "course URL drift")
    require(
        course["source_repository"] == EXPECTED_COURSE_REPOSITORY,
        "course repository drift",
    )
    require(course["source_commit"] == EXPECTED_COURSE_COMMIT, "course commit drift")
    required_materials = {
        "slides/lecture_15.pdf",
        "slides/npg.pdf",
        "slides/PG_global_conv1.pdf",
        "slides/PG_global_conv2.pdf",
        "slides/NPG_ppo.pdf",
        "slides/RLHF.pdf",
        "slides/regressing_rewards.pdf",
        "CS2824projects.html",
    }
    require(
        required_materials <= set(course["materials"]),
        "required course materials are missing",
    )
    require(
        "do not transfer" in course["use_boundary"],
        "course theorem transfer boundary is missing",
    )

    ledger_ref = payload["theory_transfer_ledger"]
    require(isinstance(ledger_ref, Mapping), "theory ledger reference is malformed")
    require(ledger_ref["path"] == EXPECTED_THEORY_LEDGER_PATH, "theory ledger path drift")
    require(
        ledger_ref["sha256"] == EXPECTED_THEORY_LEDGER_SHA256,
        "theory ledger registered digest drift",
    )
    ledger_path = repo_root / ledger_ref["path"]
    require(ledger_path.is_file(), "theory transfer ledger is missing")
    require(sha256(ledger_path) == ledger_ref["sha256"], "theory transfer ledger digest mismatch")
    ledger_summary = verify_theory_ledger(load_json(ledger_path))

    packet_ref = payload["offline_falsification_packet"]
    require(isinstance(packet_ref, Mapping), "offline packet reference is malformed")
    require(packet_ref["path"] == EXPECTED_OFFLINE_PACKET_PATH, "offline packet path drift")
    require(
        packet_ref["sha256"] == EXPECTED_OFFLINE_PACKET_SHA256,
        "offline packet registered digest drift",
    )
    packet_path = repo_root / packet_ref["path"]
    require(packet_path.is_file(), "offline falsification packet is missing")
    require(sha256(packet_path) == packet_ref["sha256"], "offline packet digest mismatch")
    packet_summary = verify_offline_packet(load_json(packet_path))

    audit_bindings = payload["audit_bindings"]
    require(audit_bindings == EXPECTED_AUDIT_BINDINGS, "audit binding drift")
    require(
        all(sha256(repo_root / path) == digest for path, digest in audit_bindings.items()),
        "audit document digest mismatch",
    )

    scope = payload["scope"]
    require(isinstance(scope, Mapping), "scope is missing or malformed")
    require(
        isinstance(scope["objective"], str) and scope["objective"].strip(),
        "scope objective is missing",
    )
    require(scope["frozen_campaign"] == EXPECTED_FROZEN_CAMPAIGN, "frozen campaign drift")
    immutable = set(scope["must_not_modify_or_relabel"])
    require(
        "zvf-program/flagship/pilot_preregistration.json" in immutable, "frozen protocol missing"
    )
    require(
        "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2" in immutable,
        "r4-2 is not protected",
    )
    require(
        "zvf-program/flagship/paper/review_bundle.zip" in immutable,
        "frozen review bundle is not protected",
    )
    require("off-policy replay" in scope["frozen_classification"], "r4-2 classification drift")

    observation = scope["live_checkout_observation"]
    require(isinstance(observation, Mapping), "live checkout observation is malformed")
    objective_path = repo_root / observation["path"]
    require(objective_path.is_file(), "live objective source is missing")
    live_hash = sha256(objective_path)
    require(observation["observed_sha256"] == live_hash, "recorded live objective hash drift")
    accepted_hash = observation["accepted_unit_source_sha256"]
    require(len(accepted_hash) == 64, "accepted objective hash is malformed")

    acceptance_path = (
        repo_root / "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/acceptance/"
        "fpilot__intended_full__balanced_equal_length__s23.json"
    )
    acceptance = load_json(acceptance_path)
    recorded_hashes = set(_find_values(acceptance, "zvf-program/flagship/pilot/objective.py"))
    require(accepted_hash in recorded_hashes, "registered accepted objective hash is not evidenced")

    hypotheses = payload["hypotheses"]
    require(
        isinstance(hypotheses, list)
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("claim"), str)
            and item["claim"].strip()
            and isinstance(item.get("falsified_if"), str)
            for item in hypotheses
        ),
        "hypothesis is missing an id, claim, or falsifier",
    )
    require(
        {item["id"] for item in hypotheses}
        == {
            "H1_alignment",
            "H2_learning",
            "H3_robustness",
            "H4_no_proxy_exploitation",
            "H5_coverage",
            "H6_distribution_shift",
            "H7_error_attribution",
        },
        "hypothesis set drift",
    )
    require(all(item["falsified_if"].strip() for item in hypotheses), "hypothesis lacks falsifier")

    stages = payload["stages"]
    require(
        all(
            isinstance(item, Mapping)
            and set(item) == {"id", "gate", "evidence"}
            and all(
                isinstance(item.get(field), str) and item[field].strip()
                for field in ("id", "gate", "evidence")
            )
            for item in stages
        ),
        "stage is missing an id, gate, or evidence requirement",
    )
    stage_ids = [item["id"] for item in stages]
    require(stage_ids == payload["decision_rules"]["stage_order"], "stage order drift")
    require(
        stage_ids
        == [
            "S0_isolation",
            "S1_foundations_mapping",
            "S2_offline_alignment",
            "S3_positive_control_feasibility",
            "S4_matched_training",
            "S5_robust_evaluation",
        ],
        "required staged gates are missing",
    )

    telemetry = set(payload["required_telemetry"])
    required_telemetry = {
        "all_wrong_fraction",
        "all_correct_fraction",
        "mixed_fraction",
        "gradient_norm",
        "correct_completion_coverage",
        "mixed_group_yield_per_charged_token",
        "importance_weight_max",
        "importance_weight_effective_sample_size",
        "sampler_policy_version",
        "data_policy_lag_steps",
        "positive_advantage_clip_fraction",
        "negative_advantage_clip_fraction",
        "approx_kl_to_old_policy",
        "kl_to_initial_reference",
        "fisher_quadratic_step",
        "completion_cap_hit_rate",
        "parser_disagreement_rate",
        "charged_generated_tokens",
        "measured_flops",
        "heldout_pass_at_1",
        "heldout_pass_at_8",
        "optimization_error_proxy",
        "estimation_uncertainty",
        "approximation_error_proxy",
        "verifier_error_rate",
    }
    require(required_telemetry <= telemetry, "optimization or evaluation telemetry is incomplete")

    conditions = payload["conditions"]
    require(isinstance(conditions, Mapping), "conditions are missing or malformed")
    require(
        conditions
        == {
            "control": "registered_group_relative_control",
            "baseline": "centered_reward_without_std_normalization",
            "experimental": ["spectral_legendre", "entropic_givens"],
            "placebo": "variance_matched_random_auxiliary_score",
        },
        "comparison conditions drift",
    )

    evaluation = payload["evaluation_contract"]
    require(isinstance(evaluation, Mapping), "evaluation contract is missing or malformed")
    require(evaluation["development_and_test_are_disjoint"] is True, "test leakage allowed")
    require(
        evaluation["proxy_metrics_are_not_quality_metrics"] is True, "proxy promoted to quality"
    )
    require(
        len(evaluation["answer_checks"]) == 2
        and set(evaluation["answer_checks"]) == EXPECTED_ANSWER_CHECKS,
        "independent answer checks drift",
    )
    require(
        len(evaluation["format_perturbations"]) == 2
        and set(evaluation["format_perturbations"]) == EXPECTED_FORMAT_PERTURBATIONS,
        "format robustness checks drift",
    )

    rules = payload.get("decision_rules")
    require(isinstance(rules, Mapping), "decision rules are missing or malformed")
    theory_boundary = rules.get("theory_boundary")
    require(isinstance(theory_boundary, str), "theory boundary is missing or malformed")
    require(
        "formal domain and assumptions" in theory_boundary,
        "theory assumption boundary is missing",
    )
    require(
        "does not establish global optimality" in theory_boundary,
        "small-gradient guardrail is missing",
    )
    require(
        "cannot support a learning" in rules["claim_boundary"],
        "proxy-to-learning claim boundary drift",
    )
    require("Preserve negative results" in rules["stop"], "negative-result stop rule drift")
    require(
        rules["promotion"].startswith("A contract-lint pass is not a stage pass"),
        "promotion boundary drift",
    )

    authorization = payload["authorization"]
    require(
        authorization
        == {
            "gpu": False,
            "external_runs": False,
            "amendment_required_before_execution": True,
        },
        "prospective file accidentally authorizes execution",
    )

    review_zip = repo_root / "zvf-program/flagship/paper/review_bundle.zip"
    review_digest_file = repo_root / "zvf-program/flagship/paper/REVIEW_BUNDLE.sha256"
    require(review_zip.is_file() and review_digest_file.is_file(), "frozen review bundle missing")
    expected_outer = review_digest_file.read_text(encoding="utf-8").split()[0]
    require(sha256(review_zip) == expected_outer, "frozen review bundle outer digest mismatch")

    return {
        "status": "POSTTRAINING_FOUNDATIONS_CONTRACT_LINT_PASS",
        "verified_payload_sha256": canonical_json_sha256(payload),
        "protocol_file_sha256": sha256(protocol_path),
        "live_objective_sha256": live_hash,
        "accepted_objective_sha256": accepted_hash,
        "live_checkout_matches_accepted_source": live_hash == accepted_hash,
        "frozen_review_bundle_sha256": expected_outer,
        "gpu_authorized": False,
        "promotion_authorized": False,
        "theory_ledger_sha256": ledger_ref["sha256"],
        "theory_claim_count": ledger_summary["claim_count"],
        "theory_source_file_count": ledger_summary["source_file_count"],
        "offline_packet_sha256": packet_ref["sha256"],
        "offline_packet_status": packet_summary["status"],
        "offline_stage_count": packet_summary["stage_count"],
    }


def verify_contract(
    payload: Mapping[str, Any],
    repo_root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    try:
        return _verify_contract(payload, repo_root, protocol_path)
    except FollowupContractError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise FollowupContractError(
            f"protocol structure is missing or malformed: {type(exc).__name__}: {exc}"
        ) from exc


def deep_verify_review_bundle(repo_root: Path) -> None:
    review_zip = repo_root / "zvf-program/flagship/paper/review_bundle.zip"
    with tempfile.TemporaryDirectory(prefix="rlhfbook-review-bundle-") as temporary:
        target = Path(temporary)
        with zipfile.ZipFile(review_zip) as archive:
            archive.extractall(target)
        subprocess.run(
            [
                sys.executable,
                str(target / "verify_claims.py"),
                "--repo-root",
                str(target / "repository"),
            ],
            cwd=target,
            check=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--deep-review-bundle",
        action="store_true",
        help="extract the frozen bundle to a temporary directory and rerun its verifier",
    )
    args = parser.parse_args(argv)

    payload = load_json(args.protocol.resolve())
    result = verify_contract(payload, args.repo_root.resolve(), args.protocol.resolve())
    if args.deep_review_bundle:
        deep_verify_review_bundle(args.repo_root.resolve())
        result["deep_review_bundle"] = "pass"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
