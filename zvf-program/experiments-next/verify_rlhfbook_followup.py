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


def verify_contract(
    payload: Mapping[str, Any],
    repo_root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    require(
        payload.get("schema_version") == "posttraining-foundations-followup-v2",
        "schema drift",
    )
    require(payload.get("status") == "proposed_not_authorized", "follow-up is not inert")

    book = payload["book_binding"]
    require(book["url"] == "https://rlhfbook.com/", "book URL drift")
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

    scope = payload["scope"]
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
    objective_path = repo_root / observation["path"]
    require(objective_path.is_file(), "live objective source is missing")
    live_hash = sha256(objective_path)
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

    evaluation = payload["evaluation_contract"]
    require(evaluation["development_and_test_are_disjoint"] is True, "test leakage allowed")
    require(
        evaluation["proxy_metrics_are_not_quality_metrics"] is True, "proxy promoted to quality"
    )
    require(len(evaluation["answer_checks"]) >= 2, "independent answer check missing")
    require(len(evaluation["format_perturbations"]) >= 2, "format robustness check missing")

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
        "status": "POSTTRAINING_FOUNDATIONS_FOLLOWUP_CONTRACT_PASS",
        "protocol_sha256": sha256(protocol_path),
        "live_objective_sha256": live_hash,
        "accepted_objective_sha256": accepted_hash,
        "live_checkout_matches_accepted_source": live_hash == accepted_hash,
        "frozen_review_bundle_sha256": expected_outer,
        "gpu_authorized": False,
    }


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
