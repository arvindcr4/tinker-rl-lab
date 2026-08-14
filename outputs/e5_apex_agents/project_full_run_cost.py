#!/usr/bin/env python3
"""Cost projection for a full 480-task judged APEX-Agents evaluation.

Every count here is MEASURED from the pinned dataset revision and the pinned
Archipelago source. The only estimated quantity is the size of a model's answer,
which cannot be known before a run -- so it is swept, not guessed, and the
judge rate card is a parameter rather than a hardcoded price.

Nothing in this script calls a model, a provider, Tinker, or W&B.

Writes: outputs/e5_apex_agents/full_run_cost_projection.json
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "hf_dataset" / "tasks_and_rubrics.json"
OUT = HERE / "full_run_cost_projection.json"

# --- Tinker rate card: from zvf-program/flagship/eval_apex_agents.py ---------
PREFILL_USD_PER_M = Decimal("0.54")
SAMPLE_USD_PER_M = Decimal("1.335")
MAX_PROMPT_TOKENS = 8192
MAX_RESPONSE_TOKENS = 512

# --- Measured from the pinned Archipelago grading source --------------------
# runner/evals/output_llm/utils/prompts.py  select_grading_system_prompt(None, False)
GRADING_SYSTEM_TOKENS = 4110 // 4
# runner/evals/output_llm/utils/services/artifact_evaluate.py
ARTIFACT_SELECT_SYSTEM_TOKENS = 2024 // 4
# runner/evals/output_llm/negative_criteria.py
NEGATIVE_CRITERIA_ENABLED = False

JUDGE_OUTPUT_TOKENS = 300
ARTIFACT_SELECT_OUTPUT_TOKENS = 150
ARTIFACT_LISTING_TOKENS = 500
CHARS_PER_TOKEN = 4

# Agent answer length is the one unknown -- swept, not assumed.
AGENT_OUTPUT_TOKEN_SWEEP = [500, 2000, 8000]

# Judge rate card is a PARAMETER. These are placeholder tiers so the owner can
# substitute the real per-token rates (the receipt carries them as
# `llm_judge_model_rates`). They are NOT quoted vendor prices.
JUDGE_RATE_TIERS = {
    "placeholder_cheap": {"input_usd_per_m": "0.10", "output_usd_per_m": "0.40"},
    "placeholder_mid": {"input_usd_per_m": "1.00", "output_usd_per_m": "5.00"},
    "placeholder_frontier": {"input_usd_per_m": "3.00", "output_usd_per_m": "15.00"},
}

REMAINING_BUDGET_USD = Decimal("15.00")


def usd(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'))}"


def main() -> int:
    tasks = json.loads(DATASET.read_text(encoding="utf-8"))
    n_tasks = len(tasks)
    criteria = [len(t["rubric"]) for t in tasks]
    n_criteria = sum(criteria)
    file_tasks = [t for t in tasks if t["gold_response_type"] == "file"]
    file_criteria = sum(len(t["rubric"]) for t in file_tasks)
    mean_prompt_tokens = (
        sum(len(t["prompt"]) for t in tasks) / n_tasks / CHARS_PER_TOKEN
    )
    mean_criterion_tokens = (
        sum(len(c["criteria"]) for t in tasks for c in t["rubric"])
        / n_criteria
        / CHARS_PER_TOKEN
    )

    # ---- A. Tinker agent cost ---------------------------------------------
    def task_cost(max_steps: int) -> Decimal:
        prefill = Decimal(max_steps * MAX_PROMPT_TOKENS) * PREFILL_USD_PER_M / 1_000_000
        sample = Decimal(max_steps * MAX_RESPONSE_TOKENS) * SAMPLE_USD_PER_M / 1_000_000
        return prefill + sample

    tinker = {}
    for max_steps in (50, 100):
        per_task = task_cost(max_steps)
        rows = {}
        for rollouts, label in ((1, "pass@1"), (8, "pass@8_leaderboard_parity")):
            total = per_task * n_tasks * rollouts
            rows[label] = {
                "rollouts_per_task": rollouts,
                "total_usd": usd(total),
                "over_remaining_budget_by": f"{total / REMAINING_BUDGET_USD:.1f}x",
            }
        tinker[f"max_steps_{max_steps}"] = {
            "per_task_ceiling_usd": usd(per_task),
            "tasks_affordable_at_remaining_budget": int(
                REMAINING_BUDGET_USD / per_task
            ),
            "share_of_suite_affordable": (
                f"{float(REMAINING_BUDGET_USD / per_task) / n_tasks * 100:.1f}%"
            ),
            **rows,
        }

    # ---- B. Judge cost -----------------------------------------------------
    grading_calls = n_criteria
    select_calls_floor = file_criteria          # only tasks that produce files
    select_calls_ceiling = n_criteria           # every task leaves some artifact
    judge_calls = {
        "grading_calls": grading_calls,
        "artifact_selection_calls_floor": select_calls_floor,
        "artifact_selection_calls_ceiling": select_calls_ceiling,
        "negative_criteria_calls": 0,
        "negative_criteria_note": (
            "NEGATIVE_CRITERIA_ENABLED is False in the pinned source, and the "
            "APEX rubric carries no negative_criteria field"
        ),
        "total_floor": grading_calls + select_calls_floor,
        "total_ceiling": grading_calls + select_calls_ceiling,
    }

    judge = {}
    for agent_tokens in AGENT_OUTPUT_TOKEN_SWEEP:
        grading_in = (
            GRADING_SYSTEM_TOKENS
            + mean_prompt_tokens
            + mean_criterion_tokens
            + agent_tokens
        )
        select_in = (
            ARTIFACT_SELECT_SYSTEM_TOKENS
            + mean_prompt_tokens
            + mean_criterion_tokens
            + ARTIFACT_LISTING_TOKENS
        )
        tiers = {}
        for tier, rates in JUDGE_RATE_TIERS.items():
            rin = Decimal(rates["input_usd_per_m"])
            rout = Decimal(rates["output_usd_per_m"])
            def total(select_calls: int) -> Decimal:
                tin = Decimal(grading_calls) * Decimal(str(round(grading_in))) + \
                      Decimal(select_calls) * Decimal(str(round(select_in)))
                tout = Decimal(grading_calls * JUDGE_OUTPUT_TOKENS) + Decimal(
                    select_calls * ARTIFACT_SELECT_OUTPUT_TOKENS
                )
                return tin * rin / 1_000_000 + tout * rout / 1_000_000
            tiers[tier] = {
                "pass@1_floor_usd": usd(total(select_calls_floor)),
                "pass@1_ceiling_usd": usd(total(select_calls_ceiling)),
                "pass@8_floor_usd": usd(total(select_calls_floor) * 8),
            }
        judge[f"agent_output_{agent_tokens}_tokens"] = {
            "grading_call_input_tokens": round(grading_in),
            "artifact_selection_call_input_tokens": round(select_in),
            "by_rate_tier": tiers,
        }

    projection = {
        "schema_version": "pavlov-e5-apex-agents-cost-projection-v1",
        "suite_id": "apex_agents_eval",
        "dataset_revision": "92c86856cf1b11f9833a8a076b3a45a63afa3929",
        "no_model_or_provider_was_called": True,
        "measured_from_dataset": {
            "tasks": n_tasks,
            "rubric_criteria_total": n_criteria,
            "rubric_criteria_min": min(criteria),
            "rubric_criteria_max": max(criteria),
            "rubric_criteria_mean": round(n_criteria / n_tasks, 3),
            "file_output_tasks": len(file_tasks),
            "file_output_criteria": file_criteria,
            "mean_task_prompt_tokens": round(mean_prompt_tokens),
            "mean_criterion_tokens": round(mean_criterion_tokens),
        },
        "measured_from_archipelago": {
            "grading_system_prompt_tokens": GRADING_SYSTEM_TOKENS,
            "artifact_selection_system_prompt_tokens": ARTIFACT_SELECT_SYSTEM_TOKENS,
            "negative_criteria_enabled": NEGATIVE_CRITERIA_ENABLED,
            "judge_calls_per_criterion": "1 grading + 1 artifact-selection when the snapshot diff is non-empty",
        },
        "assumptions": [
            f"{CHARS_PER_TOKEN} characters per token",
            "agent answer length is swept over 500 / 2000 / 8000 tokens rather than assumed",
            "judge rate tiers are PLACEHOLDERS, not vendor quotes -- substitute the real rate card",
            "Tinker cost uses the ceiling (every step saturates max_prompt_tokens); real usage is lower",
            "artifact excerpt tokens inside the grading prompt are not modelled and push the judge cost up for the 58 file-output tasks",
        ],
        "remaining_budget_usd": str(REMAINING_BUDGET_USD),
        "tinker_agent_cost": tinker,
        "judge_call_counts": judge_calls,
        "judge_cost": judge,
        "wall_clock": {
            "per_task_serial_minutes_estimate": "5-15 (docker env boot, world populate, agent loop, snapshot, grading)",
            "full_suite_serial_hours_estimate": "40-120",
            "note": "wall clock, not money, may be the binding constraint on a full run",
        },
        "verdict": {
            "full_480_task_pass@1_affordable": False,
            "headline": (
                "A full 480-task pass@1 run costs about "
                f"${usd(task_cost(50) * n_tasks)} in Tinker alone at max_steps=50 -- "
                f"roughly {float(task_cost(50) * n_tasks / REMAINING_BUDGET_USD):.0f}x the "
                f"${REMAINING_BUDGET_USD} remaining, before a single judge call."
            ),
            "affordable_scope": (
                f"About {int(REMAINING_BUDGET_USD / task_cost(50))} tasks "
                f"({float(REMAINING_BUDGET_USD / task_cost(50)) / n_tasks * 100:.0f}% of the suite) "
                "at the Tinker ceiling, and that spends the entire remaining budget "
                "with nothing left for E11 or E13."
            ),
            "recommendation": (
                "Do not attempt the full suite. If an APEX number is wanted, run a "
                "pre-registered stratified subset with the task IDs sealed in the "
                "split manifest before the run, and report it explicitly as a subset "
                "estimate with confidence intervals -- never as a suite score "
                "comparable to the published leaderboard, which is pass@8 over all 480."
            ),
        },
    }
    OUT.write_text(json.dumps(projection, indent=2) + "\n", encoding="utf-8")

    print(projection["verdict"]["headline"])
    print()
    print("Tinker (agent) ceiling, max_steps=50:")
    t = tinker["max_steps_50"]
    print(f"  per task            ${t['per_task_ceiling_usd']}")
    print(f"  480 tasks pass@1    ${t['pass@1']['total_usd']}  ({t['pass@1']['over_remaining_budget_by']} over budget)")
    print(f"  480 tasks pass@8    ${t['pass@8_leaderboard_parity']['total_usd']}")
    print(f"  affordable at $15   {t['tasks_affordable_at_remaining_budget']} tasks ({t['share_of_suite_affordable']} of suite)")
    print()
    print(f"Judge calls: {judge_calls['total_floor']}-{judge_calls['total_ceiling']} "
          f"({grading_calls} grading + {select_calls_floor}-{select_calls_ceiling} artifact-selection)")
    for tier in JUDGE_RATE_TIERS:
        row = judge["agent_output_2000_tokens"]["by_rate_tier"][tier]
        print(f"  {tier:22s} pass@1 ${row['pass@1_floor_usd']}-${row['pass@1_ceiling_usd']}")
    print()
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
