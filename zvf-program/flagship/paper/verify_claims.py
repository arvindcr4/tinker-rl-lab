#!/usr/bin/env python3
"""Audit the manuscript against checked-in, content-addressed evidence.

This verifier performs no network access and never regenerates training data.
It checks receipt integrity and internal invariants; it does not claim to
recompute gradients from model checkpoints.  Use --repo-root when the script is
copied outside its repository, including an extracted review bundle.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"required evidence is missing: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived_sha256(path: Path) -> dict[str, str]:
    """Hash regular files in a frozen tar archive by repository-relative name."""
    result: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            name = member.name.split("/", 1)[-1]
            result[name] = hashlib.sha256(handle.read()).hexdigest()
    return result


def python_constant(path: Path, name: str) -> Any:
    """Read one literal module constant without importing runtime dependencies."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"missing Python constant {name}: {path}")


def locate_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.resolve()]
    else:
        here = Path(__file__).resolve().parent
        candidates = [Path.cwd().resolve(), here, *here.parents]
    marker = Path("zvf-program/flagship/pilot_preregistration.json")
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "could not locate repository evidence; pass --repo-root PATH. "
        f"Searched: {searched}"
    )


def homogeneous_probability(p: float, group_size: int) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if p in (0.0, 1.0):
        return 1.0
    return math.exp(group_size * math.log(p)) + math.exp(
        group_size * math.log1p(-p)
    )


def informative_probability(p: float, group_size: int) -> float:
    """Stable 1-p^G-(1-p)^G, evaluated at the nearer boundary."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    edge = min(p, 1.0 - p)
    if edge == 0.0:
        return 0.0
    return -math.expm1(group_size * math.log1p(-edge)) - edge**group_size


def groups_for_first_contrast(p: float, group_size: int, delta: float) -> int | float:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    q = informative_probability(p, group_size)
    if q == 0.0:
        return math.inf
    return math.ceil(math.log(delta) / math.log1p(-q))


def decision_reversal(q: float, retries: int, cost: float, benefit: float) -> dict[str, float]:
    """Two-action batch-retry witness; all m retry costs are committed."""
    if not 0.0 < q < 1.0 or retries < 1 or cost <= 0.0 or benefit <= 0.0:
        raise ValueError("invalid decision-reversal parameters")
    retry_cost = cost * retries
    recoverable = benefit * (1.0 - (1.0 - q) ** retries) - retry_cost
    if recoverable <= 0.0:
        raise ValueError("parameters do not induce an action reversal")
    return {
        "retry_value_irrecoverable": -retry_cost,
        "retry_value_recoverable": recoverable,
        "minimax_regret_lower_bound": retry_cost * recoverable / (retry_cost + recoverable),
    }


def sanity_check_math() -> dict[str, Any]:
    # These are regression/sanity checks for the displayed formulae, not a
    # substitute for the analytic proofs in the manuscript.
    for group_size in (2, 4, 8, 16):
        for left_i in range(1, 99):
            left = left_i / 100.0
            right = min(0.99, left + 0.01)
            midpoint = (left + right) / 2.0
            assert homogeneous_probability(midpoint, group_size) <= (
                homogeneous_probability(left, group_size)
                + homogeneous_probability(right, group_size)
            ) / 2.0 + 1e-15

    for p in (1e-12, 0.001, 0.01, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-12):
        for group_size in (2, 8, 32):
            q = informative_probability(p, group_size)
            f = 1.0 - q
            n = groups_for_first_contrast(p, group_size, 0.05)
            assert isinstance(n, int)
            assert math.exp(n * math.log1p(-q)) <= 0.05 * (1.0 + 1e-12)
            if n > 1:
                assert math.exp((n - 1) * math.log1p(-q)) > 0.05 * (1.0 - 1e-12)
            assert 0.0 < f < 1.0

    # Fixed-G boundary asymptotic and the exact all-G lower bound G/q >= 1/e.
    epsilon = 1e-8
    boundary_errors: dict[str, float] = {}
    for group_size in (2, 4, 8, 16, 32):
        q = informative_probability(epsilon, group_size)
        scaled_cost = epsilon * group_size / q
        assert abs(q / (group_size * epsilon) - 1.0) < 1e-6
        assert scaled_cost >= 1.0 - 1e-12
        assert abs(scaled_cost - 1.0) < 1e-6
        boundary_errors[str(group_size)] = abs(scaled_cost - 1.0)

    # Appendix integer bracket, including the G>=2 constraint.
    appendix_cases = 0
    for p in (0.01, 0.1, 0.3, 0.5, 0.8, 0.99):
        heavy = max(p, 1.0 - p)
        for tau in (0.9, 0.5, 0.1, 0.01):
            exact = next(
                group_size
                for group_size in range(2, 10000)
                if homogeneous_probability(p, group_size) <= tau
            )
            lower = max(2, math.ceil(math.log(tau) / math.log(heavy)))
            upper = max(2, math.ceil(math.log(tau / 2.0) / math.log(heavy)))
            assert lower <= exact <= upper
            appendix_cases += 1

    # Finite-action value-of-information check.
    posterior_utilities = ((1.0, 0.0), (0.0, 1.0))
    with_information = sum(max(row) for row in posterior_utilities) / 2.0
    without_information = max(
        sum(row[action] for row in posterior_utilities) / 2.0 for action in (0, 1)
    )
    assert with_information >= without_information

    reversal = decision_reversal(q=0.1, retries=8, cost=0.02, benefit=1.0)
    assert reversal["retry_value_irrecoverable"] < 0.0
    assert reversal["retry_value_recoverable"] > 0.0

    return {
        "status": "PASS",
        "scope": "executable sanity checks; analytic proofs remain in the manuscript",
        "boundary_epsilon": epsilon,
        "boundary_rollout_cost_relative_errors": boundary_errors,
        "appendix_integer_cases": appendix_cases,
        "decision_reversal_example": reversal,
        "value_of_information_witness": {
            "with_information": with_information,
            "without_information": without_information,
        },
    }


def verify_s1(repo_root: Path) -> dict[str, Any]:
    s1 = repo_root / "zvf-program/flagship/s1"
    results = s1 / "results"
    freeze_path = results / "implementation_freeze.json"
    freeze = load_json(freeze_path)
    assert freeze["status"] == "S1_PASS"
    assert freeze["intended_case_count"] == {"trl": 14, "verl": 14}
    assert freeze["controller_case_count"] == 36
    assert freeze["errors"] == []
    assert freeze["tolerances"] == {"atol": 1e-8, "dtype": "float64", "rtol": 1e-6}
    assert python_constant(s1 / "reference.py", "ATOL") == 1e-8

    receipt_summaries: dict[str, Any] = {}
    for stack in ("trl", "verl"):
        pointer = freeze[f"{stack}_receipt"]
        path = repo_root / pointer["path"]
        assert sha256(path) == pointer["sha256"]
        receipt = load_json(path)
        assert receipt["status"] == "PASS"
        assert receipt["fixture_digest"] == freeze["fixture_digest"]
        assert len(receipt["intended_cases"]) == 14
        assert all(case["verdict"] == "PASS" and case["conforms"] for case in receipt["intended_cases"])
        assert len(receipt["controller_matrix"]) == 36
        assert {case["policy"] for case in receipt["controller_matrix"]} == {
            "static_g8", "static_g16", "symmetric_zvf", "failure_only",
            "boundary_aware", "full_triage",
        }
        assert {case["case"] for case in receipt["controller_matrix"]} == {
            "all_wrong", "all_correct", "mixed", "noisy", "missing", "delayed"
        }
        verdict_counts = Counter(case["verdict"] for case in receipt["native_cases"])
        expected = (
            {"MATERIAL_DIFFERENCE": 4, "NOT_TESTED": 1}
            if stack == "trl"
            else {"MATERIAL_DIFFERENCE": 1, "NOT_TESTED": 4}
        )
        assert dict(verdict_counts) == expected
        assert [case["verdict"] for case in receipt["native_cases"]] == freeze["native_verdicts"][stack]

        for name, expected_hash in freeze["source_hashes"][stack].items():
            assert sha256(s1 / name) == expected_hash

        receipt_summaries[stack] = {
            "receipt_sha256": pointer["sha256"],
            "intended_cases": len(receipt["intended_cases"]),
            "native_verdict_counts": expected,
            "controller_cases": len(receipt["controller_matrix"]),
        }

    return {
        "status": "PASS",
        "freeze_sha256": sha256(freeze_path),
        "fixture_digest": freeze["fixture_digest"],
        "s1_zero_threshold": 1e-8,
        "receipts": receipt_summaries,
    }


def verify_campaign(repo_root: Path) -> dict[str, Any]:
    campaign = repo_root / "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2"
    prereg = load_json(repo_root / "zvf-program/flagship/pilot_preregistration.json")
    launch_manifest_path = campaign / "launch_manifest.json"
    launch_manifest = load_json(launch_manifest_path)
    state = load_json(campaign / "supervisor_state.json")
    jobs = state["jobs"]
    manifest_ids = [item["id"] for item in launch_manifest["jobs"]]
    assert len(manifest_ids) == len(set(manifest_ids)) == 31
    assert set(manifest_ids) == set(jobs)
    assert launch_manifest["job_count"] == 31
    assert launch_manifest["preflight_job_count"] == 1
    assert launch_manifest["corpus_job_count"] == 6
    assert launch_manifest["unit_job_count"] == 24
    assert state["manifest_fingerprint"] == launch_manifest["fingerprint"]
    statuses = Counter(job["status"] for job in jobs.values())
    expected_statuses = {
        "accepted": 10,
        "descoped_contract_infeasible": 14,
        "failed_infrastructure": 2,
        "failed_validation": 1,
        "pending_quota_reset": 4,
    }
    assert dict(statuses) == expected_statuses
    assert len(jobs) == 31

    conditions = ("epsilon_only", "intended_full", "native_trl", "reduction_only")
    expected_scientific_ids = {
        "fpilot__epsilon_only__balanced_equal_length__s11",
        "fpilot__epsilon_only__balanced_equal_length__s23",
        "fpilot__intended_full__balanced_equal_length__s23",
        "fpilot__native_trl__balanced_equal_length__s23",
        "fpilot__reduction_only__balanced_equal_length__s11",
        "fpilot__reduction_only__balanced_equal_length__s23",
    }
    expected_accepted = expected_scientific_ids | {
        "corpus__balanced_equal_length__s11",
        "corpus__balanced_equal_length__s23",
        "corpus__balanced_equal_length__s37",
        "preflight__a100_stack_smoke",
    }
    expected_dispositions = {
        "accepted": expected_accepted,
        "failed_infrastructure": {
            "fpilot__intended_full__balanced_equal_length__s11",
            "fpilot__native_trl__balanced_equal_length__s11",
        },
        "failed_validation": {"corpus__filtered_variable_length__s11"},
        "pending_quota_reset": {
            f"fpilot__{condition}__balanced_equal_length__s37"
            for condition in conditions
        },
        "descoped_contract_infeasible": {
            "corpus__filtered_variable_length__s23",
            "corpus__filtered_variable_length__s37",
            *{
                f"fpilot__{condition}__filtered_variable_length__s{seed}"
                for condition in conditions
                for seed in (11, 23, 37)
            },
        },
    }
    actual_dispositions = {
        status: {key for key, job in jobs.items() if job["status"] == status}
        for status in expected_dispositions
    }
    assert actual_dispositions == expected_dispositions

    filtered_failure = jobs["corpus__filtered_variable_length__s11"]["last_error"]
    assert "CV 0.000000 is below 0.350000" in filtered_failure
    assert prereg["runtime"]["model"]["id"] == "Qwen/Qwen3-1.7B"
    assert prereg["runtime"]["execution_contract"]["max_completion_length"] == 512
    assert "require CV>=0.35" in prereg["runtime"]["execution_contract"]["selection_mask_algorithm"]
    assert "Qwen/Qwen3-1.7B" in state["operational_notes"]["metadata_correction"]
    assert all(
        "Qwen2.5-0.5B" not in job["last_error"]
        for job in jobs.values()
        if job["status"] == "descoped_contract_infeasible"
    )

    execution = prereg["runtime"]["execution_contract"]
    assert execution["heldout_split"]["balanced_equal_length"].endswith(
        "test indices 0..127 in source order"
    )
    assert execution["heldout_n"] == 128
    assert execution["decoding"]["heldout"]["do_sample"] is False
    assert execution["decoding"]["heldout"]["temperature"] == 0.0
    assert execution["reward_parsers"]["balanced_equal_length"] == (
        "strict final GSM8K #### integer match"
    )
    pilot_objective = repo_root / "zvf-program/flagship/pilot/objective.py"
    assert python_constant(pilot_objective, "NUMERICAL_FLOOR") == 1e-12
    assert python_constant(pilot_objective, "TRL_REWARD_EPSILON") == 1e-4

    expected_versions = {
        item.split("==", 1)[0]: item.split("==", 1)[1]
        for item in prereg["runtime"]["package_pins"]
    }
    control_archive = repo_root / "zvf-program/flagship/pilot/provenance/r3-control-source.tar.gz"
    frozen_sources = archived_sha256(control_archive)
    scientific: list[dict[str, Any]] = []
    corpus_fingerprints: dict[int, set[str]] = defaultdict(set)
    step_zero_hashes: set[str] = set()
    verified_source_paths: set[str] = set()
    for path in sorted((campaign / "acceptance").glob("fpilot__*.json")):
        record = load_json(path)
        unit = record["unit"]
        assert unit["id"] == path.stem
        assert unit["id"] in expected_scientific_ids
        assert unit["regime"] == "balanced_equal_length"
        assert record["status"] == "accepted"
        versions = record["full_record"]["runtime_versions"]
        assert {name: versions[name] for name in expected_versions} == expected_versions
        source_manifest = record["full_record"]["manifest"]["source_manifest"]
        assert source_manifest
        assert all(len(value) == 64 for value in source_manifest.values())
        for source_path, expected_hash in source_manifest.items():
            local_source = repo_root / source_path
            if local_source.is_file():
                assert sha256(repo_root / source_path) == expected_hash
            elif source_path in frozen_sources:
                assert frozen_sources[source_path] == expected_hash
            else:
                raise AssertionError(f"source is absent from review evidence: {source_path}")
            verified_source_paths.add(source_path)

        receipts = record["full_record"]["manifest"]["gradient_receipts"]
        assert len(receipts) == 100
        relations = Counter(item["gradient_relation"] for item in receipts)
        assert set(relations) <= {"joint_zero", "nonzero"}
        nonzero = []
        for item in receipts:
            if item["gradient_relation"] == "joint_zero":
                assert item["intended_gradient_norm"] == 0.0
                assert item["native_gradient_norm"] == 0.0
                assert item["gradient_cosine"] is None
                assert item["gradient_relative_l2"] is None
            else:
                assert item["intended_gradient_norm"] > 0.0
                assert item["native_gradient_norm"] > 0.0
                assert -1.0 <= item["gradient_cosine"] <= 1.0
                assert item["gradient_relative_l2"] >= 0.0
                nonzero.append(item)

        evaluations = record["full_record"]["evaluations"]
        assert [item["step"] for item in evaluations] == [0, 20, 40, 60, 80, 100]
        assert all(item["heldout_n"] == 128 for item in evaluations)
        assert all(item["unique_row_hashes"] == 128 for item in evaluations)
        assert all(item["accuracy"] == item["correct"] / 128 for item in evaluations)
        assert record["final_accuracy"] == evaluations[-1]["accuracy"]
        step_zero_hashes.add(evaluations[0]["evidence_sha256"])

        ledger = record["full_record"]["token_flop_ledger"]
        assert ledger["charged_generated_tokens"] > 0
        assert ledger["total_active_tokens"] > 0
        assert ledger["total_padded_tokens"] >= ledger["total_active_tokens"]
        assert all(ledger[name] > 0 for name in (
            "policy_forward_flops", "diagnostic_backward_flops",
            "optimizer_backward_flops", "replay_generation_flops",
        ))
        corpus_fingerprints[unit["seed"]].add(record["corpus_fingerprint"])

        scientific.append({
            "id": unit["id"],
            "condition": unit["condition"],
            "seed": unit["seed"],
            "start_accuracy": evaluations[0]["accuracy"],
            "final_accuracy": record["final_accuracy"],
            "heldout_n": 128,
            "charged_generated_tokens": ledger["charged_generated_tokens"],
            "joint_zero": relations["joint_zero"],
            "nonzero": relations["nonzero"],
            "one_sided_zero": 0,
            "minimum_nonzero_cosine": min(item["gradient_cosine"] for item in nonzero),
            "maximum_nonzero_relative_l2": max(item["gradient_relative_l2"] for item in nonzero),
            "wandb_run_id": record["wandb_run_id"],
            "hf_artifact_commit": record["hf_artifact_commit"],
            "acceptance_sha256": sha256(path),
        })

    assert {item["id"] for item in scientific} == expected_scientific_ids
    assert all(len(values) == 1 for values in corpus_fingerprints.values())
    assert len(step_zero_hashes) == 1
    assert {item["joint_zero"] for item in scientific} == {62, 65}
    assert all(item["one_sided_zero"] == 0 for item in scientific)

    expected_endpoints = {
        "fpilot__epsilon_only__balanced_equal_length__s11": (20, 17, 62, 0.999858, 0.016844, 396672),
        "fpilot__reduction_only__balanced_equal_length__s11": (20, 19, 62, 0.999844, 0.017681, 396672),
        "fpilot__epsilon_only__balanced_equal_length__s23": (20, 17, 65, 0.999894, 0.014576, 400448),
        "fpilot__intended_full__balanced_equal_length__s23": (20, 20, 65, 0.999881, 0.015440, 400448),
        "fpilot__native_trl__balanced_equal_length__s23": (20, 17, 65, 0.999894, 0.014576, 400448),
        "fpilot__reduction_only__balanced_equal_length__s23": (20, 20, 65, 0.999881, 0.015440, 400448),
    }
    for item in scientific:
        start, final, zeros, cosine, relative_l2, tokens = expected_endpoints[item["id"]]
        assert item["start_accuracy"] == start / 128
        assert item["final_accuracy"] == final / 128
        assert item["joint_zero"] == zeros
        assert math.isclose(item["minimum_nonzero_cosine"], cosine, abs_tol=5e-7)
        assert math.isclose(item["maximum_nonzero_relative_l2"], relative_l2, abs_tol=5e-7)
        assert item["charged_generated_tokens"] == tokens

    return {
        "status": "PASS",
        "campaign_state_sha256": sha256(campaign / "supervisor_state.json"),
        "launch_manifest_sha256": sha256(launch_manifest_path),
        "preregistration_sha256": sha256(repo_root / "zvf-program/flagship/pilot_preregistration.json"),
        "campaign_status_counts": expected_statuses,
        "accepted_decomposition": {"corpora": 3, "preflight": 1, "scientific_units": 6},
        "verified_frozen_source_files": len(verified_source_paths),
        "campaign_thresholds": {
            "intended_zero_floor": 1e-12,
            "native_trl_reward_epsilon": 1e-4,
        },
        "filtered_failure": filtered_failure,
        "accepted_scientific_units": scientific,
        "receipt_scope": (
            "receipt integrity and internal invariants; gradients and predictions "
            "are not recomputed from model checkpoints"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="repository root, or the repository/ directory in an extracted review bundle",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    root = locate_repo_root(args.repo_root)
    report = {
        "schema_version": 2,
        "repository_root": str(root),
        "claims": {
            "MATH_SANITY": sanity_check_math(),
            "S1_RECEIPTS": verify_s1(root),
            "R4_2_CAMPAIGN_RECEIPTS": verify_campaign(root),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
