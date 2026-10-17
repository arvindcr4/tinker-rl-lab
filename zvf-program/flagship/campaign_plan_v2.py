"""Prospective plan v2. Offline checks only; never a launch authorization."""

from __future__ import annotations
from copy import deepcopy
from collections import Counter
import hashlib
import math
from collections.abc import Mapping

try:
    from .evidence_v2 import approval_valid, artifact_valid, canonical, digest
except ImportError:
    from evidence_v2 import approval_valid, artifact_valid, canonical, digest

PLAN = {
    "plan_id": "pavlov-campaign-v2",
    "component_plan_id": "pavlov-xlam-component-v2",
    "estimand": "fixed_adapter_paired_task_clusters_conditional_on_recorded_training_seed",
    "phases": [
        {
            "id": "smoke",
            "entrants": 1,
            "total_steps": 10,
            "survivors": 1,
            "training_scope": "xlam",
            "evaluation_scope": "sealed_selection_only",
            "learning_rate": 2e-5,
        },
        {
            "id": "round_1",
            "entrants": 3,
            "total_steps": 50,
            "survivors": 2,
            "training_scope": "xlam",
            "evaluation_scope": "sealed_selection_only",
        },
        {
            "id": "round_2",
            "entrants": 2,
            "total_steps": 100,
            "survivors": 1,
            "training_scope": "xlam",
            "evaluation_scope": "sealed_selection_only",
        },
        {
            "id": "winner_extension",
            "entrants": 1,
            "total_steps": 200,
            "survivors": 1,
            "training_scope": "xlam",
            "evaluation_scope": "sealed_selection_only",
        },
        {
            "id": "portfolio_extension",
            "entrants": 1,
            "total_steps": "frozen_in_launch_manifest",
            "survivors": 1,
            "training_scope": "all_12_suites",
            "evaluation_scope": "none",
        },
        {
            "id": "final",
            "entrants": 1,
            "total_steps": 0,
            "survivors": 1,
            "training_scope": "none",
            "evaluation_scope": "paired_base_final_all_14_suites_once",
        },
    ],
    "ranking_keys": [
        "perfect_call_count_desc",
        "strict_mean_reward_desc",
        "conservative_cost_asc",
        "learning_rate_asc",
        "arm_id_lexical",
    ],
    "held_out_separation": {
        "must_be_disjoint": True,
        "held_out_label_requires_independent_receipt": True,
        "selection_split": {"consulted_during_selection": True},
        "final_eval_split": {"consulted_during_selection": False},
    },
    "safety_method": "one_sided_95pct_exact_binomial_newly_harmful_independent_pairs",
    "safety_maximum_increase": 0.02,
    "suite_regression_policy": "report_each_suite_and_fail_claim_if_frozen_suite_guard_fails",
    "normalization": "frozen_before_final_data; range,direction,transform,reference,missing_policy required",
    "scientific_claim_implementation": "BLOCKED_pending_approved_cluster_analysis_and_production_calibration",
}


def plan_contract():
    result = deepcopy(PLAN)
    result["plan_sha256"] = hashlib.sha256(canonical(PLAN)).hexdigest()
    result.update(
        method="successive_halving",
        arms=["arm_1e-5", "arm_2e-5", "arm_4e-5"],
        screening_arm_ids=["arm_1e-5", "arm_2e-5", "arm_4e-5"],
        screening_learning_rates=[1e-5, 2e-5, 4e-5],
        short_screening_steps=50,
        checkpoint_steps=[5, 10, 50],
        selection_metric="sealed selection slice perfect-call rate",
        selection_rule="maximize sealed selection slice perfect-call rate",
        tie_breakers=[
            "strict mean reward",
            "lower estimated cost",
            "lower learning rate",
            "lexical arm ID",
        ],
        winner_extension="two survivors to 100; one winner to 200; then frozen portfolio extension",
    )
    return result


def checkpoint_errors(
    records, *, stage, expected_identity, scheduled_steps, current_step, final_step
):
    """Check exports only when due; no previous run's final artifact can count."""
    errors = []
    if (
        not isinstance(expected_identity, Mapping)
        or set(expected_identity) != {"campaign", "run"}
        or any(not isinstance(v, str) or not v for v in expected_identity.values())
    ):
        return ["independently frozen campaign and run identity required"]
    if stage not in {"launch", "running", "completion"}:
        return ["invalid checkpoint lifecycle stage"]
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0
        for x in [current_step, final_step, *scheduled_steps]
    ):
        return ["checkpoint steps must be nonnegative integers"]
    due = {0} if stage == "launch" else {0, *[s for s in scheduled_steps if s <= current_step]}
    if stage == "completion":
        due = {0, *scheduled_steps, final_step}
        if current_step != final_step:
            errors.append("completion step differs from frozen final step")
    if not isinstance(records, list):
        return errors + ["checkpoint records required"]
    steps = []
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("malformed checkpoint record")
            continue
        steps.append(record.get("step"))
        if (
            record.get("identity") != dict(expected_identity)
            or not digest(record.get("checkpoint_sha256"))
            or not isinstance(record.get("export_revision"), str)
            or len(record["export_revision"]) != 40
            or any(c not in "0123456789abcdef" for c in record["export_revision"])
            or not artifact_valid(record.get("artifact"))
            or record.get("checkpoint_sha256") != record.get("artifact", {}).get("sha256")
        ):
            errors.append("checkpoint artifact or campaign/run identity mismatch")
    if len(steps) != len(set(steps)) or set(steps) != due:
        errors.append("checkpoint exports must exactly cover all due logical steps")
    return errors


def safety_feasibility(independent_pairs, *, maximum_increase=0.02, alpha=0.05):
    if (
        isinstance(independent_pairs, bool)
        or not isinstance(independent_pairs, int)
        or independent_pairs <= 0
    ):
        raise ValueError("positive integer independent pair count required")
    if not 0 < maximum_increase < 1 or not 0 < alpha < 1:
        raise ValueError("invalid safety precision parameters")
    upper = -math.expm1(math.log(alpha) / independent_pairs)
    return {
        "method": PLAN["safety_method"],
        "best_case_upper": upper,
        "feasible": upper <= maximum_increase,
        "minimum_independent_pairs": math.ceil(math.log(alpha) / math.log1p(-maximum_increase)),
        "assumption": "independent pairs; zero newly harmful outcomes; conservative bound on increase",
    }


def normalization_errors(spec):
    if not isinstance(spec, Mapping) or set(spec) != {
        "minimum",
        "maximum",
        "direction",
        "transform",
        "reference",
        "missing_policy",
    }:
        return ["complete frozen normalization schema required"]
    lo, hi = spec["minimum"], spec["maximum"]
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (lo, hi)
        )
        or lo >= hi
    ):
        return ["finite ordered metric range required"]
    if (
        spec["direction"] not in {"higher", "lower"}
        or spec["transform"] != "fixed_linear"
        or spec["missing_policy"] != "incomplete"
        or spec["reference"] != "preregistered_range"
    ):
        return ["unsupported or outcome-fitted normalization"]
    return []


def evaluate_campaign_completion(execution, frozen, *, trusted_issuers):
    """Separate executed coverage from a scientific claim; legacy assertions fail.

    Only signed execution artifacts tied to an independently frozen context can
    establish EXECUTION_COMPLETE. Scientific acceptance remains disabled until
    the predeclared cluster-analysis implementation and calibration are reviewed.
    """
    blockers = []
    identity = frozen.get("identity", {}) if isinstance(frozen, Mapping) else {}
    if not approval_valid(
        execution, "campaign_execution", trusted_issuers=trusted_issuers, expected_identity=identity
    ):
        return {
            "stage": "BLOCKED",
            "scientific_claim_eligible": False,
            "blockers": ["trusted immutable execution evidence required"],
        }
    if not digest(frozen.get("base_checkpoint")) or not digest(frozen.get("final_checkpoint")):
        blockers.append("immutable base/final checkpoint content digests required")
    if frozen.get("plan_sha256") != plan_contract()["plan_sha256"]:
        blockers.append("frozen plan identity mismatch")
    for field in ("training_task_ids", "evaluation_task_ids"):
        actual, expected = execution.get(field), frozen.get(field)
        if (
            not isinstance(actual, list)
            or not expected
            or len(actual) != len(set(actual))
            or set(actual) != set(expected)
        ):
            blockers.append(f"exact {field} coverage required")
    train = set(execution.get("training_task_ids", []))
    selection = set(execution.get("selection_task_ids", []))
    final = set(execution.get("evaluation_task_ids", []))
    if (
        train & final
        or selection & final
        or execution.get("final_consulted_during_selection") is not False
    ):
        blockers.append("training/selection/final separation failed")
    batches = execution.get("batches", [])
    if not batches:
        blockers.append("actual batch evidence required")
    batch_task_ids = []
    batch_suite_ids = set()
    for batch in batches:
        episodes = batch.get("episodes", [])
        if not episodes or len(set(d for ep in episodes for d in ep.get("domains", []))) < 6:
            blockers.append("batch lacks six domain families")
            continue
        batch_task_ids.extend(ep.get("task_id") for ep in episodes)
        batch_suite_ids.update(ep.get("suite_id") for ep in episodes)
        n = len(episodes)
        if (
            sum("math" in ep.get("domains", []) for ep in episodes) / n > 0.05
            or sum(ep.get("stateful") is True for ep in episodes) / n < 0.6
            or sum(ep.get("artifact") is True for ep in episodes) / n < 0.5
        ):
            blockers.append("actual per-batch composition failed")
    if Counter(batch_task_ids) != Counter(frozen.get("training_task_ids", [])):
        blockers.append("actual batch task coverage/counts differ from frozen execution")
    if (
        not frozen.get("training_suite_ids")
        or batch_suite_ids != set(frozen["training_suite_ids"])
        or len(batch_suite_ids) != 12
    ):
        blockers.append("actual batches must cover all 12 frozen training suites")
    blockers.extend(
        checkpoint_errors(
            execution.get("checkpoints"),
            stage="completion",
            expected_identity={"campaign": identity.get("campaign"), "run": identity.get("run")},
            scheduled_steps=frozen.get("checkpoint_steps", []),
            current_step=execution.get("final_step", -1),
            final_step=frozen.get("final_step", -1),
        )
    )
    pairs = execution.get("paired_results", [])
    pair_ids = [row.get("task_id") for row in pairs]
    if set(pair_ids) != final or len(pair_ids) != len(set(pair_ids)):
        blockers.append("complete paired base/final results required")
    for row in pairs:
        if row.get("base_checkpoint") != frozen.get("base_checkpoint") or row.get(
            "final_checkpoint"
        ) != frozen.get("final_checkpoint"):
            blockers.append("paired checkpoint identity mismatch")
        if (
            row.get("infrastructure_error") is not False
            or type(row.get("attempt_count")) is not int
            or row["attempt_count"] != 1
        ):
            blockers.append("infrastructure or retry rule failed")
        suite = frozen.get("evaluation_tasks", {}).get(row.get("task_id"))
        if not suite or row.get("suite_id") != suite:
            blockers.append("paired task suite identity mismatch")
        for field in ("base_result", "final_result"):
            value = row.get(field)
            spec = frozen.get("normalization", {}).get(suite, {})
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not spec
                or not spec.get("minimum", math.inf) <= value <= spec.get("maximum", -math.inf)
            ):
                blockers.append("finite raw metric inside frozen normalization range required")
    if (
        len(set(frozen.get("evaluation_tasks", {}).values())) != 14
        or set(frozen.get("evaluation_tasks", {})) != final
    ):
        blockers.append("exact frozen 14-suite evaluation coverage required")
    if (
        type(execution.get("infrastructure_errors")) is not int
        or execution["infrastructure_errors"] != 0
    ):
        blockers.append("infrastructure errors unresolved")
    for spec in frozen.get("normalization", {}).values():
        blockers.extend(normalization_errors(spec))
    if set(frozen.get("normalization", {})) != set(frozen.get("evaluation_tasks", {}).values()):
        blockers.append("every frozen suite needs its own metric schema")
    if not frozen.get("normalization"):
        blockers.append("normalization not frozen")
    return {
        "stage": "BLOCKED" if blockers else "EXECUTION_COMPLETE",
        "scientific_claim_eligible": False,
        "identity": dict(identity),
        "execution_sha256": hashlib.sha256(canonical(execution)).hexdigest(),
        "blockers": blockers
        + [
            "scientific claim requires approved cluster analysis, suite/safety guards and production-path calibration"
        ],
    }


def render_plan_markdown():
    lines = [
        "# Prospective campaign plan v2",
        "",
        "Plan hash: " + plan_contract()["plan_sha256"],
        "",
        "This amendment does not rerun or reinterpret historical experiments.",
        "",
    ]
    for phase in PLAN["phases"]:
        lines.append(
            f"- {phase['id']}: {phase['entrants']} entrant(s), {phase['total_steps']} total steps, {phase['survivors']} survivor(s); train {phase['training_scope']}; evaluate {phase['evaluation_scope']}"
        )
    lines.extend(
        ["", "Ranking: " + ", ".join(PLAN["ranking_keys"]), "", "Estimand: " + PLAN["estimand"]]
    )
    return "\n".join(lines) + "\n"


def launch_feasibility_errors(contract):
    """Plan precision before spending; no implied authority to change sample size."""
    planning = contract.get("safety_planning")
    if not isinstance(planning, Mapping):
        return [
            "safety: frozen independent cluster counts and cost projection required before launch"
        ]
    expected = {
        suite
        for suite, spec in contract.get("suite_registry", {}).items()
        if spec.get("role") == "primary_eval"
    }
    counts = planning.get("independent_pairs_by_suite", {})
    if not expected or not isinstance(counts, Mapping) or set(counts) != expected:
        return ["safety: exact primary-suite planning roster required"]
    errors = []
    for suite, count in counts.items():
        try:
            if not safety_feasibility(count)["feasible"]:
                errors.append(
                    f"safety: {suite} cannot meet 2% bound even with zero newly harmful events"
                )
        except ValueError:
            errors.append(f"safety: {suite} independent pair count invalid")
    if (
        planning.get("method") != PLAN["safety_method"]
        or planning.get("independence_design") != "independent_pairs"
    ):
        errors.append("safety: unsupported or missing independent-pair interval design")
    cost = planning.get("projected_cost_usd")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost <= 0
    ):
        errors.append("safety: finite positive complete cost projection required")
    return errors
