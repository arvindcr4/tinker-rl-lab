#!/usr/bin/env python3
"""Feed REAL APEX-Agents gold responses through the REAL Archipelago verifier.

This is `harness_validation`, NOT a benchmark run.

  is_model_score: false
  suite score:    null

No model produced any of this text. The "agent answer" fed to the verifier is
the dataset's own expert-authored `gold_response`. That makes this a plumbing
proof, not a capability measurement: it shows the pinned Archipelago grading
runner ingests real APEX task/verifier/world identifiers and real gold text,
runs its `final_answer` helper over them, and returns per-criterion verdicts.

It says NOTHING about how hard the tasks are or how any model would score.

Controls (this is why the result means something):

  positive  -- pattern drawn from THIS task's own gold response  -> expect 1.0
  swap      -- pattern drawn from a DIFFERENT task's gold        -> expect 0.0
  sentinel  -- a token that cannot occur in any gold response    -> expect 0.0

Without the swap control a passing run would be a tautology: a verifier that
returned 1.0 unconditionally would look identical. The swap control forces the
verifier to demonstrate it is reading the answer under test.

Only `pattern_match_check` is used -- registered
`eval_types=[EvalType.PROGRAMMATIC]`. No LLM judge, no paid API call, no
Tinker, no W&B.

Snapshot caveat, stated plainly: the two snapshot zips are the inert synthetic
pair from the sibling harness validation. The real `world_files_zipped/` bulk
(20 GB) was deliberately NOT downloaded. Snapshots are scaffolding here; the
data under test is the gold response text, which is real.

Usage:
    python3 run_gold_response_validation.py [--tasks-per-domain 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE.parent
ARCHIPELAGO = E5 / "sprint.8cEFaN" / "archipelago"
GRADING = ARCHIPELAGO / "grading"
VENV_PYTHON = E5 / "venv-grading" / "bin" / "python"
DATASET = E5 / "hf_dataset"
FIXTURES = HERE / "fixtures"
WORK = HERE / "gold_out"

ARCHIPELAGO_REVISION = "1c3dcd4694b313020cd626699c9c7cc1c0a2fc58"
DATASET_REVISION = "92c86856cf1b11f9833a8a076b3a45a63afa3929"

# A token that cannot appear in expert-written professional-services prose.
SENTINEL = "ZZQX_NEVER_OCCURS_9F3A1D"
# Length of the literal slice lifted out of a gold response as the pattern.
SLICE_LEN = 60
# Only tasks with a gold response comfortably longer than the slice qualify.
MIN_GOLD_LEN = 200


def strip_keys(env: dict[str, str]) -> dict[str, str]:
    blocked = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "VERTEX_PROJECT",
        "REDUCTO_API_KEY",
    }
    return {k: v for k, v in env.items() if k not in blocked}


def distinctive_slice(text: str) -> str:
    """A contiguous literal lifted from the middle of the gold response.

    Taken from the middle rather than the start so it cannot accidentally be a
    generic opener ("Yes, the supplier can...") shared across tasks.
    """
    collapsed = re.sub(r"\s+", " ", text).strip()
    start = max(0, len(collapsed) // 2 - SLICE_LEN // 2)
    return collapsed[start : start + SLICE_LEN]


def as_pattern(literal: str) -> str:
    """Regex that tolerates whitespace/newline differences in the answer."""
    return r"\s+".join(re.escape(tok) for tok in literal.split())


def select_tasks(tasks: list[dict], per_domain: int) -> list[dict]:
    """Deterministic selection: sorted by task_id, first N per domain."""
    eligible = [
        t
        for t in tasks
        if t.get("gold_response_type") == "text"
        and isinstance(t.get("gold_response"), str)
        and len(t["gold_response"]) >= MIN_GOLD_LEN
        and t.get("rubric")
    ]
    by_domain: dict[str, list[dict]] = {}
    for task in sorted(eligible, key=lambda t: t["task_id"]):
        by_domain.setdefault(task["domain"], []).append(task)
    chosen: list[dict] = []
    for domain in sorted(by_domain):
        chosen.extend(by_domain[domain][:per_domain])
    return chosen


def build_case(task: dict, donor: dict) -> dict | None:
    """One task's three verifiers. Returns None if the swap control is unsound."""
    own = distinctive_slice(task["gold_response"])
    foreign = distinctive_slice(donor["gold_response"])
    # The swap control is only a control if the donor phrase genuinely does not
    # occur in this task's answer.
    if re.search(as_pattern(foreign), task["gold_response"], re.IGNORECASE):
        return None
    if not re.search(as_pattern(own), task["gold_response"], re.IGNORECASE):
        return None

    # Real verifier_ids from the dataset rubric, so real identifiers flow
    # through the runner. Suffixes keep them unique within the run.
    base_vid = task["rubric"][0]["verifier_id"]
    return {
        "task_id": task["task_id"],
        "world_id": task["world_id"],
        "domain": task["domain"],
        "task_name": task["task_name"],
        "rubric_criteria_count": len(task["rubric"]),
        "gold_response_len": len(task["gold_response"]),
        "gold_response_sha256": hashlib.sha256(
            task["gold_response"].encode("utf-8")
        ).hexdigest(),
        "donor_task_id": donor["task_id"],
        "verifiers": [
            {
                "verifier_id": f"{base_vid}__positive",
                "control": "positive",
                "expected_score": 1.0,
                "pattern": as_pattern(own),
            },
            {
                "verifier_id": f"{base_vid}__swap",
                "control": "swap",
                "expected_score": 0.0,
                "pattern": as_pattern(foreign),
            },
            {
                "verifier_id": f"{base_vid}__sentinel",
                "control": "sentinel",
                "expected_score": 0.0,
                "pattern": re.escape(SENTINEL),
            },
        ],
    }


def grade_case(case: dict, task: dict, index: int) -> dict:
    case_dir = WORK / f"{index:02d}_{case['task_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # The graded "agent answer" IS the dataset's gold response.
    (case_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": task["gold_response"]},
                ],
                "output": {"source": "apex_gold_response_harness_validation"},
                "status": "completed",
                "time_elapsed": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "grading_settings.json").write_text(
        json.dumps({"llm_judge_model": "unused/no-llm-judge-in-this-run"}),
        encoding="utf-8",
    )
    (case_dir / "eval_configs.json").write_text(
        json.dumps(
            [
                {
                    "eval_config_id": "ec_pattern_match",
                    "eval_config_name": "Pattern Match Check (programmatic)",
                    "eval_defn_id": "pattern_match_check",
                    "eval_config_values": {},
                }
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "verifiers.json").write_text(
        json.dumps(
            [
                {
                    "verifier_id": v["verifier_id"],
                    "verifier_version": 1,
                    "world_id": case["world_id"],
                    "task_id": case["task_id"],
                    "eval_config_id": "ec_pattern_match",
                    "verifier_values": {
                        "is_primary_objective": v["control"] == "positive",
                        "pattern": v["pattern"],
                        "search_target": "final_answer",
                        "case_sensitive": False,
                    },
                    "verifier_index": i,
                    "verifier_dependencies": None,
                }
                for i, v in enumerate(case["verifiers"])
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "scoring_config.json").write_text(
        json.dumps(
            {
                "scoring_config_id": "sc_gold_validation",
                "scoring_config_name": "Template",
                "scoring_defn_id": "template",
                "scoring_config_values": {},
            }
        ),
        encoding="utf-8",
    )

    env = strip_keys(dict(os.environ))
    env["PYTHONPATH"] = str(GRADING)
    completed = subprocess.run(
        [
            str(VENV_PYTHON), "-m", "runner.main",
            "--grading-run-id", f"gr_gold_{index:02d}",
            "--trajectory-id", case["task_id"],
            "--initial-snapshot", str(FIXTURES / "initial_snapshot.zip"),
            "--final-snapshot", str(FIXTURES / "initial_snapshot.zip"),
            "--trajectory", str(case_dir / "trajectory.json"),
            "--grading-settings", str(case_dir / "grading_settings.json"),
            "--verifiers", str(case_dir / "verifiers.json"),
            "--eval-configs", str(case_dir / "eval_configs.json"),
            "--scoring-config", str(case_dir / "scoring_config.json"),
            "--output", str(case_dir / "grades.json"),
        ],
        cwd=GRADING,
        env=env,
        capture_output=True,
        text=True,
    )
    (case_dir / "grading_run.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        return {**case, "ok": False, "error": f"runner exited {completed.returncode}"}

    grades = json.loads((case_dir / "grades.json").read_text(encoding="utf-8"))
    observed = {v["verifier_id"]: v["score"] for v in grades["verifier_results"]}
    results = []
    ok = grades.get("grading_run_status") == "completed"
    for v in case["verifiers"]:
        got = observed.get(v["verifier_id"])
        match = got == v["expected_score"]
        ok = ok and match
        results.append(
            {
                "control": v["control"],
                "verifier_id": v["verifier_id"],
                "expected": v["expected_score"],
                "observed": got,
                "match": match,
            }
        )
    return {
        **{k: v for k, v in case.items() if k != "verifiers"},
        "grading_run_status": grades.get("grading_run_status"),
        "controls": results,
        "ok": ok,
        "evidence_dir": str(case_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-per-domain", type=int, default=4)
    args = parser.parse_args()

    for required in (VENV_PYTHON, DATASET / "tasks_and_rubrics.json"):
        if not required.exists():
            print(f"FAIL: missing {required}", file=sys.stderr)
            return 1

    revision = subprocess.run(
        ["git", "-C", str(ARCHIPELAGO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != ARCHIPELAGO_REVISION:
        print(f"FAIL: archipelago at {revision}", file=sys.stderr)
        return 1

    tasks_bytes = (DATASET / "tasks_and_rubrics.json").read_bytes()
    tasks = json.loads(tasks_bytes)
    selected = select_tasks(tasks, args.tasks_per_domain)
    print(f"selected {len(selected)} real APEX tasks across "
          f"{len({t['domain'] for t in selected})} domains")

    WORK.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, task in enumerate(selected):
        donor = selected[(i + 1) % len(selected)]
        case = build_case(task, donor)
        if case is None:
            print(f"  skipped {task['task_id']}: swap control not sound")
            continue
        cases.append((case, task))

    results = []
    for index, (case, task) in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] grading {case['task_id']} "
              f"({case['domain']}) ...", flush=True)
        results.append(grade_case(case, task, index))

    passed = sum(1 for r in results if r["ok"])
    control_totals: dict[str, dict[str, int]] = {}
    for r in results:
        for c in r.get("controls", []):
            bucket = control_totals.setdefault(c["control"], {"match": 0, "total": 0})
            bucket["total"] += 1
            bucket["match"] += int(c["match"])

    receipt = {
        "schema_version": "pavlov-e5-apex-agents-gold-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": "apex_agents_eval",
        "kind": "harness_validation",
        "is_model_score": False,
        "suite_score": None,
        "status": "PASS" if passed == len(results) and results else "FAIL",
        "claim": (
            "The pinned Archipelago grading runner ingests real APEX-Agents "
            "task/world/verifier identifiers and real expert-authored gold "
            "responses, runs its final_answer helper over them, and returns "
            "per-criterion verdicts that discriminate between the answer under "
            "test and a foreign answer."
        ),
        "not_claimed": [
            "any APEX-Agents suite score",
            "any model capability measurement",
            "that gold responses were graded against the real APEX rubric "
            "(the real rubric is LLM-judged; no judge was invoked)",
        ],
        "dataset": {
            "dataset_id": "mercor/apex-agents",
            "dataset_revision": DATASET_REVISION,
            "tasks_and_rubrics_sha256": hashlib.sha256(tasks_bytes).hexdigest(),
            "gold_responses_are_real": True,
            "snapshots_are_synthetic_scaffolding": True,
            "world_files_downloaded": False,
        },
        "native_verifier": {
            "repository": "https://github.com/Mercor-Intelligence/archipelago",
            "revision": revision,
        },
        "eval_defn_id": "pattern_match_check",
        "eval_types": ["PROGRAMMATIC"],
        "llm_judge_used": False,
        "paid_api_calls": 0,
        "tinker_calls": 0,
        "tasks_graded": len(results),
        "tasks_fully_matching_expectation": passed,
        "control_summary": control_totals,
        "cases": results,
    }
    receipt_path = WORK / "gold_response_validation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"tasks graded: {len(results)} | fully matching expectation: {passed}")
    for control, bucket in sorted(control_totals.items()):
        print(f"  {control:9s} {bucket['match']}/{bucket['total']} as expected")
    print(f"receipt: {receipt_path}")
    if passed != len(results) or not results:
        print("FAIL: at least one case did not match expectation", file=sys.stderr)
        return 1
    print("PASS: harness_validation (is_model_score=false, suite score=null)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
