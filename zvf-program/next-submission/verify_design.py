#!/usr/bin/env python3
"""Fail-closed verifier for the next AI/ML submission design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_PROTOCOL = HERE / "preregistration.json"
EXPECTED_BOOK_COMMIT = "1bd092d80d9a247fa51003c32cedd8fc4679ce47"
EXPECTED_BOOK_FILES = {
    "book/chapters/06-policy-gradients.md": "6671f67b6d635ab8cc1f1859345edeead9a3f7489e5410a5c4653cb1b6c4e444",
    "book/chapters/14-over-optimization.md": "81736529e9237c079fa39d27a96635b45805af827cb287f5e2ee20959d4dfd81",
    "book/chapters/15-regularization.md": "389dc58d110d1d5bcf30f9c2a6b5539f7db29aeffc3aec310bac5a27cef17fed",
    "book/chapters/16-evaluation.md": "d9c1764b4cf85e869eac82a6a30f73a5578ddab4faf69c410c8e7d66ce5a4525",
    "book/chapters/appendix-c-practical.md": "c08f557a7f3e2ae80339f6bac9a8028ee22b39158f743008f86a4e7273879e1a",
}
EXPECTED_COURSE_COMMIT = "5dcc34e3b861da632371645fb05aebb12a40d23c"
EXPECTED_COURSE_FILES = {
    "slides/PG_global_conv1.pdf": "26309b138a546eff684ed586809919de9a0360f2d0aeb8fd3ad64ccc546cd86f",
    "slides/PG_global_conv2.pdf": "ee21ee7f5cc52e878007a643e23cbca1dac5ca8ea402a4fb1d8c94b858f64f5d",
    "slides/NPG_ppo.pdf": "309137fbfec2dc7c01cc1489aaa867f768b5e4a253bdea3359f633eb4529a8db",
    "slides/RLHF.pdf": "f12bd048818e817069cb0ef0f46ea90f22219ca3e3a9b016e748db1c3727194a",
    "CS2824projects.html": "30ef1ea58da08abe7873564858feb95e8caae3d7f8061247b1092cf59edfcf41",
}


class DesignContractError(ValueError):
    """Raised when the design would permit a reviewer-facing evidence failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DesignContractError(message)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DesignContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    require(isinstance(payload, dict), f"{path.name} must contain a JSON object")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_planning_seed_count(power: dict[str, Any]) -> tuple[int, int]:
    alpha = float(power["per_task_alpha"])
    target_power = float(power["target_power"])
    sd = float(power["planning_sd"])
    margin = float(power["accuracy_margin"])
    inflation = float(power["variance_transfer_inflation"])
    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_power = NormalDist().inv_cdf(target_power)
    raw = math.ceil(((z_alpha + z_power) * sd / margin) ** 2)
    inflated = math.ceil(raw * inflation)
    return raw, inflated


def verify_contract(
    protocol: dict[str, Any],
    claims: dict[str, Any],
    results: dict[str, Any],
    blueprint: str,
    *,
    repo_root: Path = REPO_ROOT,
    check_bindings: bool = True,
) -> dict[str, Any]:
    require(
        protocol.get("schema_version") == "aiml-next-preregistration-v1", "protocol schema drift"
    )
    require(
        protocol.get("status") == "DESIGN_FROZEN_EXECUTION_AUTHORIZED",
        "design status drift",
    )

    auth = protocol.get("authorization")
    require(isinstance(auth, dict), "authorization block missing")
    require(auth.get("gpu") is True, "GPU execution is not authorized")
    require(
        all(
            auth.get(key) is False
            for key in ("external_user_recruitment", "publish", "submit", "promotion")
        ),
        "execution authorization expanded into external recruitment or publication",
    )
    authorization_path = repo_root / str(auth.get("receipt_path"))
    require(authorization_path.is_file(), "execution authorization receipt missing")
    require(
        auth.get("receipt_sha256") == sha256(authorization_path),
        "execution authorization receipt hash drift",
    )
    authorization = load_json(authorization_path)
    require(
        authorization.get("schema_version") == "aiml-next-execution-authorization-v1",
        "execution authorization schema drift",
    )
    authorization_scope = authorization.get("scope")
    require(isinstance(authorization_scope, dict), "execution authorization scope missing")
    require(
        all(
            authorization_scope.get(key) is True
            for key in (
                "local_implementation",
                "local_verification",
                "remote_gpu_preflight",
                "remote_confirmatory_matrix",
            )
        ),
        "execution authorization is incomplete",
    )
    require(
        all(
            authorization_scope.get(key) is False
            for key in (
                "external_user_recruitment",
                "publish",
                "submit",
                "push",
                "promote_result_claims_before_complete_receipts",
            )
        ),
        "execution authorization silently permits an external or evidence-promotion action",
    )

    contribution = protocol.get("contribution_policy")
    require(isinstance(contribution, dict), "contribution policy missing")
    require(
        contribution.get("use_inspired_authorized") is False,
        "use-inspired claim lacks external evidence",
    )
    receipt_path = repo_root / str(contribution.get("use_inspired_receipt_path"))
    require(
        not receipt_path.exists(),
        "an unverified external receipt cannot silently unlock use-inspired status",
    )

    sources = protocol.get("sources")
    require(isinstance(sources, dict), "source bindings missing")
    book = sources.get("rlhf_book")
    course = sources.get("harvard_cs2824")
    require(
        isinstance(book, dict) and book.get("commit") == EXPECTED_BOOK_COMMIT,
        "RLHF Book commit drift",
    )
    require(book.get("files") == EXPECTED_BOOK_FILES, "RLHF Book file binding drift")
    require(
        isinstance(course, dict) and course.get("commit") == EXPECTED_COURSE_COMMIT,
        "CS2824 commit drift",
    )
    require(course.get("files") == EXPECTED_COURSE_FILES, "CS2824 file binding drift")

    scope = protocol.get("claim_scope")
    require(isinstance(scope, dict), "claim scope missing")
    tasks = scope.get("included_tasks")
    require(tasks == ["gsm8k", "math500"], "included task scope drift")
    excluded = scope.get("excluded")
    require(isinstance(excluded, dict), "excluded scope must be explicit")
    require(
        {
            "humaneval",
            "mbpp",
            "synthetic_tool_use",
            "ppo",
            "other_models",
            "subjective_rewards",
        }.issubset(excluded),
        "review-relevant exclusions missing",
    )

    treatment = protocol.get("treatment")
    require(isinstance(treatment, dict), "treatment missing")
    arms = treatment.get("arms")
    require(isinstance(arms, list) and len(arms) == 2, "design must have exactly two arms")
    arm_ids = [arm.get("arm_id") for arm in arms]
    require(arm_ids == ["grpo_g8", "contrast_early_stop_g2_to_g8"], "arm definition drift")
    require(len(treatment.get("fixed_components", [])) >= 10, "matched treatment fields incomplete")

    seed_plan = protocol.get("paired_seed_plan")
    require(isinstance(seed_plan, dict), "paired seed plan missing")
    seeds = seed_plan.get("seeds")
    planned_n = seed_plan.get("planning_seed_count")
    require(
        isinstance(seeds, list) and len(seeds) == planned_n, "seed count does not match seed list"
    )
    require(len(set(seeds)) == len(seeds), "training seeds must be unique")
    require(planned_n > 1, "single-seed design rejected")
    require(
        seed_plan.get("independent_unit") == "training seed within a frozen task-model-stack cell",
        "independent unit drift",
    )
    require(set(seeds).isdisjoint({11, 23, 37, 53, 71, 89, 107, 131}), "new seeds overlap E1")

    power = protocol.get("power_plan")
    require(isinstance(power, dict), "power plan missing")
    raw_n, inflated_n = expected_planning_seed_count(power)
    require(
        power.get("normal_approximation_seed_count_before_inflation") == raw_n,
        "raw power calculation drift",
    )
    require(
        power.get("planning_seed_count_after_inflation") == inflated_n == planned_n,
        "inflated power calculation drift",
    )
    require(power.get("cost_effect_scale") == "paired_log_token_ratio", "cost effect scale drift")
    require(
        math.isclose(
            float(power.get("cost_success_boundary_log_ratio")),
            math.log(0.8),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "cost boundary drift",
    )
    require(
        "blinded first-eight" in str(power.get("cost_variance_source")),
        "cost variance plan missing",
    )
    require(
        power.get("blinded_reassessment_after_completed_pairs_per_cell") < planned_n,
        "variance reassessment is not interim",
    )
    require(power.get("maximum_seed_count_per_cell") >= planned_n, "maximum seed count below plan")
    require(
        power.get("final_seed_count_rule")
        == "maximum of capability and cost requirements across both tasks, capped at 24",
        "joint seed-count rule drift",
    )
    require(
        power.get("cap_exceeded_verdict") == "STOP_UNDERPOWERED",
        "underpowered cap is not fail-closed",
    )
    require(power.get("reassessment_may_reduce_n") is False, "reassessment may not reduce n")
    require(
        power.get("reassessment_may_inspect_arm_effect") is False,
        "reassessment may not inspect arm effects",
    )
    require(
        power.get("power_receipt_required_before_final_analysis") is True,
        "power receipt gate missing",
    )

    estimands = protocol.get("primary_estimands")
    require(isinstance(estimands, dict), "primary estimands missing")
    require(
        "log(intervention/baseline)" in estimands.get("cost", ""),
        "cost estimand is not paired log ratio",
    )
    require(
        math.isclose(
            float(estimands.get("cost_success_boundary_log_ratio")),
            math.log(0.8),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "estimand cost boundary drift",
    )

    task_rows = protocol.get("tasks")
    require(isinstance(task_rows, list) and len(task_rows) == 2, "task definitions incomplete")
    heldout_by_task = {row.get("task_id"): row.get("heldout_n") for row in task_rows}
    require(heldout_by_task == {"gsm8k": 1000, "math500": 500}, "held-out sizes drift")
    task_by_id = {row["task_id"]: row for row in task_rows}
    require(
        {
            "training_dataset": "openai/gsm8k",
            "training_dataset_revision": "740312add88f781978c0658806c59bc2815b9866",
            "training_split": "train",
        }.items()
        <= task_by_id["gsm8k"].items(),
        "GSM8K training corpus drift",
    )
    require(
        {
            "training_dataset": "DigitalLearningGmbH/MATH-lighteval",
            "training_dataset_revision": "0530c78699ea5e8eb5530600900e1f328b48acad",
            "training_split": "train",
            "evaluation_dataset": "HuggingFaceH4/MATH-500",
            "evaluation_dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        }.items()
        <= task_by_id["math500"].items(),
        "MATH training or evaluation corpus drift",
    )

    matrix = protocol.get("matrix")
    require(isinstance(matrix, list), "matrix missing")
    expected_cells = {(task, arm) for task in tasks for arm in arm_ids}
    actual_cells = {(row.get("task_id"), row.get("arm_id")) for row in matrix}
    require(
        len(matrix) == 4 and actual_cells == expected_cells,
        "task-arm matrix is incomplete or duplicated",
    )
    for row in matrix:
        require(row.get("status") == "planned", "design-only matrix contains a result status")
        require(row.get("seed_set") == "paired_seed_plan", "matrix cell uses an unpaired seed set")
        require(row.get("heldout_n") == heldout_by_task[row["task_id"]], "cell held-out size drift")

    prior = protocol.get("prior_evidence_boundary")
    require(isinstance(prior, dict), "prior evidence boundary missing")
    require(
        prior.get("allowed_use")
        == "Worst observed paired held-out standard deviation only, as a conservative planning prior.",
        "prior evidence was promoted",
    )
    forbidden = " ".join(prior.get("forbidden_uses", []))
    require(
        "stale DAPO DISAPPEARS verdict is quarantined" in forbidden,
        "stale E1 verdict is not quarantined",
    )

    require(
        claims.get("schema_version") == "aiml-next-claim-ledger-v1", "claim ledger schema drift"
    )
    claim_rows = claims.get("claims")
    require(
        isinstance(claim_rows, list)
        and {row.get("claim_id") for row in claim_rows}
        == {
            "C1_cost_reduction",
            "C2_capability_noninferiority",
            "C3_joint_operator_rule",
            "C4_mechanism",
            "C5_use_inspired",
        },
        "claim ledger incomplete",
    )
    require(
        not any(row.get("status") in {"achieved", "passed", "supported"} for row in claim_rows),
        "planned design contains achieved claims",
    )
    use_claim = next(row for row in claim_rows if row["claim_id"] == "C5_use_inspired")
    require(
        use_claim.get("status") == "blocked_external_receipt", "use-inspired claim is not blocked"
    )
    require(
        all(
            row.get("primary_table_reference") and row.get("required_columns") for row in claim_rows
        ),
        "claim lacks table-bound result fields",
    )

    require(
        results.get("schema_version") == "aiml-next-results-contract-v1",
        "results contract schema drift",
    )
    require(
        results.get("status") == "template_no_results", "results template was relabeled as evidence"
    )
    main_columns = results.get("main_table_columns")
    numeric_columns = results.get("numeric_main_table_columns")
    prohibited = set(results.get("prohibited_main_table_columns", []))
    require(
        isinstance(main_columns, list) and isinstance(numeric_columns, list),
        "main table contract missing",
    )
    require(
        set(numeric_columns).issubset(main_columns) and len(numeric_columns) >= 10,
        "main table lacks numeric results",
    )
    require(
        set(main_columns).isdisjoint(prohibited), "filename or artifact field appears in main table"
    )
    require(
        {
            "task",
            "arm",
            "completed_seeds",
            "paired_cost_effect",
            "paired_accuracy_effect",
            "verdict",
        }.issubset(main_columns),
        "main table omits a primary result",
    )
    seed_fields = set(results.get("required_seed_row_fields", []))
    required_provenance = {
        "claim_ids",
        "run_id",
        "source_commit",
        "protocol_sha256",
        "model_fingerprint",
        "tokenizer_fingerprint",
        "stack_fingerprint",
        "objective_fingerprint",
        "task_split",
        "reward_parser_sha256",
        "training_steps",
        "checkpoint_rule",
        "evaluation_rule",
        "evidence_tier",
        "receipt_sha256",
    }
    require(required_provenance.issubset(seed_fields), "seed-row provenance contract incomplete")

    manuscript = protocol.get("manuscript_contract")
    require(isinstance(manuscript, dict), "manuscript contract missing")
    require(manuscript.get("method_before_results") is True, "method must precede results")
    require(
        manuscript.get("one_primary_numeric_table") is True,
        "one primary numerical table not enforced",
    )
    require(manuscript.get("filenames_in_main_text") is False, "filenames allowed in main text")
    require(manuscript.get("all_primary_cells_visible") is True, "primary cells may be hidden")
    order = manuscript.get("section_order")
    require(order.index("Method") < order.index("Results"), "results precede method")

    require(
        "<!-- MAIN_TEXT_START -->" in blueprint and "<!-- MAIN_TEXT_END -->" in blueprint,
        "blueprint main-text markers missing",
    )
    main_text = blueprint.split("<!-- MAIN_TEXT_START -->", 1)[1].split(
        "<!-- MAIN_TEXT_END -->", 1
    )[0]
    require(
        not re.search(
            r"(?:[A-Za-z0-9_-]+\.(?:json|py|csv|tsv)|(?:zvf-program|platform_hybrid|autoresearch)/)",
            main_text,
        ),
        "internal filename appears in manuscript main text",
    )
    require(
        main_text.index("## 3. Method") < main_text.index("## 5. Results"),
        "blueprint places results before method",
    )
    require(
        "one generated numerical table" in main_text,
        "blueprint permits filenames instead of results",
    )

    if check_bindings:
        bindings = protocol.get("bindings")
        require(isinstance(bindings, dict), "artifact bindings missing")
        for name in (
            "claim_ledger",
            "results_contract",
            "manuscript_blueprint",
            "execution_authorization",
            "contrast_sampler",
            "trl_sampler_adapter",
            "remote_preflight",
            "preflight_launcher",
            "preflight_secure_exec",
            "preflight_environment_check",
        ):
            path = repo_root / str(bindings.get(f"{name}_path"))
            require(path.is_file(), f"bound {name} missing")
            require(bindings.get(f"{name}_sha256") == sha256(path), f"bound {name} hash drift")

    return {
        "status": "NEXT_SUBMISSION_DESIGN_CONTRACT_PASS",
        "tasks": len(tasks),
        "arms": len(arms),
        "primary_cells": len(matrix),
        "planned_seeds_per_cell": planned_n,
        "initial_training_units": len(matrix) * planned_n,
        "use_inspired_authorized": False,
        "gpu_authorized": True,
        "result_claims_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    bindings = protocol["bindings"]
    claims = load_json(REPO_ROOT / bindings["claim_ledger_path"])
    results = load_json(REPO_ROOT / bindings["results_contract_path"])
    blueprint = (REPO_ROOT / bindings["manuscript_blueprint_path"]).read_text(encoding="utf-8")
    report = verify_contract(protocol, claims, results, blueprint)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
