#!/usr/bin/env python3
"""Fail-closed verifier for the next AI/ML submission design."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
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


EXPECTED_SEEDS = [
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
    293, 307, 311, 313, 317, 331, 337, 347,
]
EXPECTED_SEED_24_RESERVE = 349
EXPECTED_FIXED_COMPONENTS = [
    "initial checkpoint",
    "tokenizer",
    "prompt order",
    "reward parser",
    "optimizer",
    "learning rate",
    "batch size",
    "training steps",
    "completion cap",
    "checkpoint rule",
    "held-out prompts",
    "decoder",
    "evaluator",
    "objective hyperparameters",
    "adapter configuration",
]
EXPECTED_TELEMETRY = [
    "charged_generated_tokens",
    "rollout_groups",
    "all_wrong_fraction",
    "all_correct_fraction",
    "mixed_fraction",
    "two_sample_false_homogeneity",
    "policy_ratio_q05_q50_q95",
    "clip_fraction_by_advantage_sign",
    "kl_to_data_policy",
    "kl_to_initial_reference",
    "parser_disagreement",
    "completion_clipped_fraction",
    "wall_clock_seconds",
]
EXPECTED_OBJECTIVE = {
    "epsilon": 0.2,
    "epsilon_high": 0.2,
    "importance_sampling_level": "token",
    "scale_rewards": "group",
    "loss_type": "grpo",
    "beta": 0.0,
}
EXPECTED_OPTIMIZER = {
    "learning_rate": 1e-06,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "optim": "adamw_torch_fused",
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "num_iterations": 1,
    "mask_truncated_completions": False,
}
EXPECTED_PRECISION = {
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_use_reentrant": False,
}
EXPECTED_ADAPTER = {
    "method": "lora",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": "all-linear",
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
# A004: seam-verification rollout-group caps, both strictly below the 60 rollout
# groups of one confirmatory unit.
EXPECTED_SEAM_ROLLOUT_GROUP_CAP = {
    "contrast_early_stop_g2_to_g8": 48,
    "grpo_g8": 24,
}
EXPECTED_ANALYSIS = {
    "confidence_intervals": "seed-paired bootstrap with deterministic streams and paired-t sensitivity",
    "multiplicity": "intersection-union across cost and capability within task; Holm across two tasks",
    "blinded_variance_reassessment": "pooled, arm-label-hidden variance only; may increase n to 24 and cannot change outcomes, margins, tasks, or estimands",
    "missingness": "intention-to-treat; exact resume for infrastructure interruption; no failed run is silently dropped or relabeled as scientific failure",
    "negative_result_rule": "A boundary-crossing interval, failed power receipt, or incomplete cell is INCONCLUSIVE, not equivalence or failure.",
    "checkpoint_rule": "final registered checkpoint; never select by online reward",
}
EXPECTED_JOINT_SUCCESS = (
    "Both boundaries pass on both tasks; any missing or invalid cell yields INCONCLUSIVE."
)
EXPECTED_CANONICALIZATION = (
    "JSON document parsed, with the values of bindings.execution_authorization_sha256 "
    "and authorization.receipt_sha256 set to the empty string, re-serialized with "
    "two-space indentation and a trailing newline"
)


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


def _load_power_implementation(repo_root: Path, bindings: dict[str, Any]) -> Any:
    path = repo_root / str(bindings.get("power_implementation_path"))
    require(path.is_file(), "power implementation missing")
    require(
        bindings.get("power_implementation_sha256") == sha256(path),
        "power implementation hash drift",
    )
    spec = importlib.util.spec_from_file_location("next_power_implementation", path)
    require(spec is not None and spec.loader is not None, "power implementation unreadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("minimum_paired_t_sample_size", "paired_t_power_one_sided"):
        require(hasattr(module, name), f"power implementation lacks {name}")
    return module


def canonical_protocol_sha256(protocol: dict[str, Any]) -> str:
    """Hash the protocol with the two authorization-hash fields blanked.

    This matches the canonicalization string recorded in
    execution_authorization.json and lets the authorization receipt anchor the
    exact protocol revision it authorizes without a circular hash dependency.
    """
    clone = json.loads(json.dumps(protocol))
    clone.get("bindings", {})["execution_authorization_sha256"] = ""
    clone.get("authorization", {})["receipt_sha256"] = ""
    text = json.dumps(clone, indent=2) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_planning_seed_count(
    power: dict[str, Any], power_impl: Any
) -> tuple[int, int]:
    alpha = float(power["planning_alpha_worst_case"])
    target_power = float(power["target_power"])
    sd = float(power["planning_sd"])
    margin = float(power["accuracy_margin"])
    inflation = float(power["variance_transfer_inflation"])
    raw = power_impl.minimum_paired_t_sample_size(margin / sd, alpha, target_power)
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
    amendments = protocol.get("amendments")
    require(
        isinstance(amendments, list) and len(amendments) == 4,
        "prospective protocol amendment ledger missing",
    )
    (
        amendment_record,
        decoder_amendment_record,
        hardening_amendment_record,
        seam_amendment_record,
    ) = amendments
    require(
        isinstance(amendment_record, dict)
        and amendment_record.get("amendment_id") == "A001_math_unbraced_boxed_targets"
        and amendment_record.get("status") == "prospective_before_confirmatory_execution",
        "MATH parser amendment identity or timing drift",
    )
    amendment_path = repo_root / str(amendment_record.get("path"))
    require(amendment_path.is_file(), "MATH parser amendment receipt missing")
    require(
        amendment_record.get("sha256") == sha256(amendment_path),
        "MATH parser amendment hash drift",
    )
    amendment = load_json(amendment_path)
    require(
        amendment.get("timing", {}).get("confirmatory_rows_completed_before_amendment") == 0
        and amendment.get("timing", {}).get("confirmatory_outcomes_inspected") is False,
        "MATH parser amendment is not prospective",
    )
    change = amendment.get("change")
    require(
        isinstance(change, dict)
        and change.get("training_rows_before_filtering") == 7500
        and change.get("rows_retained_after_amendment") == 7500
        and change.get("rows_dropped") == 0,
        "MATH parser amendment corpus accounting drift",
    )
    require(
        isinstance(decoder_amendment_record, dict)
        and decoder_amendment_record.get("amendment_id") == "A002_qwen3_non_thinking_decoder"
        and decoder_amendment_record.get("status") == "prospective_before_confirmatory_execution",
        "Qwen decoder amendment identity or timing drift",
    )
    decoder_amendment_path = repo_root / str(decoder_amendment_record.get("path"))
    require(decoder_amendment_path.is_file(), "Qwen decoder amendment receipt missing")
    require(
        decoder_amendment_record.get("sha256") == sha256(decoder_amendment_path),
        "Qwen decoder amendment hash drift",
    )
    decoder_amendment = load_json(decoder_amendment_path)
    require(
        decoder_amendment.get("timing", {}).get("confirmatory_rows_completed_before_amendment") == 0
        and decoder_amendment.get("timing", {}).get("confirmatory_outcomes_inspected") is False,
        "Qwen decoder amendment is not prospective",
    )
    require(
        isinstance(hardening_amendment_record, dict)
        and hardening_amendment_record.get("amendment_id") == "A003_confirmatory_hardening"
        and hardening_amendment_record.get("status") == "prospective_before_confirmatory_execution",
        "confirmatory-hardening amendment identity or timing drift",
    )
    hardening_amendment_path = repo_root / str(hardening_amendment_record.get("path"))
    require(hardening_amendment_path.is_file(), "confirmatory-hardening amendment receipt missing")
    require(
        hardening_amendment_record.get("sha256") == sha256(hardening_amendment_path),
        "confirmatory-hardening amendment hash drift",
    )
    hardening_amendment = load_json(hardening_amendment_path)
    require(
        hardening_amendment.get("timing", {}).get("confirmatory_rows_completed_before_amendment") == 0
        and hardening_amendment.get("timing", {}).get("confirmatory_outcomes_inspected") is False,
        "confirmatory-hardening amendment is not prospective",
    )
    hardening_change = hardening_amendment.get("change", {})
    require(
        hardening_change.get("objective") == EXPECTED_OBJECTIVE
        and hardening_change.get("optimizer") == EXPECTED_OPTIMIZER
        and hardening_change.get("adapter") == EXPECTED_ADAPTER,
        "confirmatory-hardening amendment treatment block drift",
    )
    require(
        hardening_change.get("seed_plan", {}).get("planning_seed_count") == len(EXPECTED_SEEDS)
        and hardening_change.get("power_plan", {}).get("planning_alpha_worst_case") == 0.0125,
        "confirmatory-hardening amendment seed or power block drift",
    )
    require(
        isinstance(seam_amendment_record, dict)
        and seam_amendment_record.get("amendment_id") == "A004_seam_verification_window"
        and seam_amendment_record.get("status") == "prospective_before_confirmatory_execution",
        "seam-verification amendment identity or timing drift",
    )
    seam_amendment_path = repo_root / str(seam_amendment_record.get("path"))
    require(seam_amendment_path.is_file(), "seam-verification amendment receipt missing")
    require(
        seam_amendment_record.get("sha256") == sha256(seam_amendment_path),
        "seam-verification amendment hash drift",
    )
    seam_amendment = load_json(seam_amendment_path)
    require(
        seam_amendment.get("timing", {}).get("confirmatory_rows_completed_before_amendment") == 0
        and seam_amendment.get("timing", {}).get("confirmatory_outcomes_inspected") is False,
        "seam-verification amendment is not prospective",
    )
    seam_window = seam_amendment.get("change", {}).get("seam_verification_window", {})
    require(
        seam_window.get("rollout_groups_cap") == EXPECTED_SEAM_ROLLOUT_GROUP_CAP
        and seam_window.get("optimizer_step_budget") == 24
        and seam_window.get("unchanged", {}).get("heldout_n") == 8
        and seam_window.get("unchanged", {}).get("num_generations") == 8,
        "seam-verification amendment window block drift",
    )
    require(
        seam_amendment.get("change", {}).get("gate_unchanged", {}).get(
            "require_live_mixed_reward_optimizer_update_per_cell"
        )
        is True,
        "seam-verification amendment relaxes the confirmatory execution gate",
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
    require(
        authorization.get("protocol_path") == "zvf-program/next-submission/preregistration.json",
        "execution authorization protocol path drift",
    )
    require(
        authorization.get("protocol_canonicalization") == EXPECTED_CANONICALIZATION,
        "execution authorization canonicalization rule drift",
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
    require(
        treatment.get("decoder")
        == {
            "enable_thinking": False,
            "training": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "evaluation": {"do_sample": False},
        },
        "Qwen decoder contract drift",
    )
    arms = treatment.get("arms")
    require(isinstance(arms, list) and len(arms) == 2, "design must have exactly two arms")
    arm_ids = [arm.get("arm_id") for arm in arms]
    require(arm_ids == ["grpo_g8", "contrast_early_stop_g2_to_g8"], "arm definition drift")
    require(treatment.get("objective") == EXPECTED_OBJECTIVE, "objective hyperparameters drift")
    require(treatment.get("optimizer") == EXPECTED_OPTIMIZER, "optimizer hyperparameters drift")
    require(
        treatment.get("precision_and_memory") == EXPECTED_PRECISION,
        "precision and memory settings drift",
    )
    require(treatment.get("adapter") == EXPECTED_ADAPTER, "adapter configuration drift")
    require(
        treatment.get("fixed_components") == EXPECTED_FIXED_COMPONENTS,
        "matched treatment fields drift",
    )

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
    require(seeds == EXPECTED_SEEDS, "preregistered seed values drift")
    require(
        seed_plan.get("seed_24_reserve") == EXPECTED_SEED_24_RESERVE,
        "seed reserve for the reassessment cap drift",
    )
    require(
        "ascending primes strictly greater than 293" in str(seed_plan.get("seed_derivation_rule")),
        "seed derivation rule missing",
    )

    power = protocol.get("power_plan")
    require(isinstance(power, dict), "power plan missing")
    require(
        "exact one-sided noncentral paired-t" in str(power.get("power_method")),
        "power method is not the exact noncentral-t calculation",
    )
    require(
        math.isclose(
            float(power.get("planning_alpha_worst_case")),
            float(power.get("per_task_alpha")) / 2.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "Holm worst-case planning alpha drift",
    )
    power_impl = _load_power_implementation(repo_root, protocol.get("bindings", {}))
    raw_n, inflated_n = expected_planning_seed_count(power, power_impl)
    require(
        power.get("exact_t_seed_count_before_inflation") == raw_n,
        "raw exact-t power calculation drift",
    )
    require(
        power.get("planning_seed_count_after_inflation") == inflated_n == planned_n,
        "inflated power calculation drift",
    )
    require(
        "joint" in str(power.get("joint_power_disclosure")),
        "joint power disclosure missing",
    )
    require(
        "log(0.8)" in str(power.get("blinded_reassessment_method"))
        and "noncentral paired-t" in str(power.get("blinded_reassessment_method")),
        "blinded reassessment method unspecified",
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
    require(
        math.isclose(
            float(estimands.get("cost_success_boundary")),
            -0.2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "cost success boundary drift",
    )
    require(
        math.isclose(
            float(estimands.get("capability_noninferiority_boundary")),
            -0.01,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "capability non-inferiority boundary drift",
    )
    require(
        estimands.get("joint_success") == EXPECTED_JOINT_SUCCESS,
        "joint success rule drift",
    )

    analysis = protocol.get("analysis")
    require(isinstance(analysis, dict), "analysis rules missing")
    for analysis_key, analysis_value in EXPECTED_ANALYSIS.items():
        require(
            analysis.get(analysis_key) == analysis_value,
            f"analysis rule drift: {analysis_key}",
        )

    require(
        protocol.get("telemetry") == EXPECTED_TELEMETRY,
        "preregistered telemetry list drift",
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

    preflight_gate = protocol.get("preflight_execution_gate")
    require(isinstance(preflight_gate, dict), "preflight execution gate missing")
    require(
        preflight_gate.get("evidence_class") == "preflight-gate-not-scientific-evidence",
        "preflight gate was promoted to scientific evidence",
    )
    require(
        preflight_gate.get("required_task_arm_cells") == len(expected_cells)
        and all(
            preflight_gate.get(key) is True
            for key in (
                "require_private_huggingface_artifact",
                "require_finished_wandb_run",
                "require_provider_session_absence",
                "require_shared_scientific_stack_fingerprint",
                "require_non_clipped_completion_path",
                "require_live_mixed_reward_optimizer_update_per_cell",
                "require_live_homogeneous_early_stop_per_intervention_task",
            )
        )
        and preflight_gate.get("failure_action") == "block_confirmatory_execution",
        "preflight execution gate is permissive or incomplete",
    )

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
    require(
        set(EXPECTED_TELEMETRY).issubset(seed_fields),
        "seed-row telemetry contract incomplete",
    )

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
            "protocol_amendment_001",
            "protocol_amendment_002",
            "protocol_amendment_003",
            "power_implementation",
            "preflight_launcher",
            "preflight_matrix_verifier",
            "hf_jobs_preflight_launcher",
            "kaggle_preflight_launcher",
            "gcp_preflight_launcher",
            "gcp_receipt_bucket",
            "preflight_secure_exec",
            "preflight_environment_check",
            "preflight_colab_setup",
        ):
            path = repo_root / str(bindings.get(f"{name}_path"))
            require(path.is_file(), f"bound {name} missing")
            require(bindings.get(f"{name}_sha256") == sha256(path), f"bound {name} hash drift")

    require(
        authorization.get("protocol_canonical_sha256") == canonical_protocol_sha256(protocol),
        "execution authorization does not anchor the current protocol revision",
    )

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
