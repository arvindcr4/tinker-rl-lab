#!/usr/bin/env python3
"""Score whether Reviewer 9kjk's follow-up is answered with reviewer-visible evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
RESPONSE = REPO_ROOT / "zvf-program/flagship/paper/NEURIPS_2026_REVIEWER_9KJK_FOLLOWUP.md"
LEDGER = REPO_ROOT / "zvf-program/flagship/paper/REVIEWER_9KJK_SCOPE_LEDGER.json"
PLAN = REPO_ROOT / "zvf-program/flagship/paper/MAY_REVIEW_RESPONSE_AND_ACL_PLAN.md"


def _postable(text: str) -> str:
    start = "<!-- POSTABLE_REPLY_START -->"
    end = "<!-- POSTABLE_REPLY_END -->"
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def evaluate() -> dict[str, object]:
    response = RESPONSE.read_text(encoding="utf-8") if RESPONSE.exists() else ""
    plan = PLAN.read_text(encoding="utf-8") if PLAN.exists() else ""
    postable = _postable(response)
    try:
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {}

    cells = ledger.get("coverage_cells", []) if isinstance(ledger, dict) else []
    source_rules = ledger.get("source_derived_design_rules", []) if isinstance(ledger, dict) else []
    statuses = {cell.get("status") for cell in cells if isinstance(cell, dict)}

    checks = [
        (
            "structural_concession",
            "cannot be evaluated" in postable.lower()
            and "filenames" in postable.lower()
            and "do not supply missing numerical results" in postable.lower(),
        ),
        (
            "full_factorial_why",
            "retrospectively" in postable.lower()
            and "predeclared complete matrix" in postable.lower()
            and "missing evidence" in postable.lower(),
        ),
        (
            "single_seed_why",
            "single-seed" in postable.lower()
            and "exploratory" in postable.lower()
            and "cannot justify comparative inference" in postable.lower(),
        ),
        (
            "use_inspired_withdrawn",
            "withdraw the use-inspired contribution type" in postable.lower(),
        ),
        (
            "actual_results_visible",
            all(
                token in postable
                for token in (
                    "164/200",
                    "166, 165, 161, 168, and 173",
                    "2/22",
                    "zero-variance fraction (ZVF) was 1",
                    "gradient utilization (GU), was 0",
                )
            ),
        ),
        (
            "no_filenames_in_postable_reply",
            bool(postable)
            and not re.search(
                r"(?:[A-Za-z0-9_-]+\.(?:json|py|csv|tsv|md|tex|pdf)|"
                r"(?:experiments|platform_hybrid|zvf-program|autoresearch|submission)/[A-Za-z0-9_./-]+)",
                postable,
            ),
        ),
        (
            "complete_scope_ledger",
            len(cells) == 20
            and {(cell.get("analysis_id"), cell.get("corpus")) for cell in cells}
            == {
                (analysis, corpus)
                for analysis in ("stack_sensitivity", "heldout_capability", "ppo_grpo", "zvf_gu", "early_triage")
                for corpus in ("gsm8k", "humaneval", "math", "synthetic_tool_use")
            },
        ),
        (
            "ledger_cells_are_adjudicated",
            bool(cells)
            and all(
                isinstance(cell, dict)
                and cell.get("status") in {"evaluated", "not_evaluated", "quarantined"}
                and cell.get("seed_basis")
                and cell.get("reviewed_result")
                and cell.get("permitted_inference")
                for cell in cells
            )
            and {"evaluated", "not_evaluated", "quarantined"}.issubset(statuses),
        ),
        (
            "source_rules_applied",
            len(source_rules) >= 4
            and any("rlhfbook.com/c/06-policy-gradients" in str(rule.get("source")) for rule in source_rules)
            and any("rlhfbook.com/c/16-evaluation" in str(rule.get("source")) for rule in source_rules)
            and any("CS2824" in str(rule.get("source")) for rule in source_rules),
        ),
        (
            "post_submission_boundary",
            ledger.get("post_submission_evidence", {}).get("review_status") == "not_reviewed_not_used_to_answer_score",
        ),
        (
            "future_replication_and_matrix_gate",
            ledger.get("future_submission_gates", {}).get("replication_and_evaluation_sample_sizes")
            == "prospectively_justified_by_power_or_precision_targets"
            and ledger.get("future_submission_gates", {}).get("all_in_scope_cells_accounted_for") is True,
        ),
        (
            "decision_record_updated",
            "Reviewer 9kjk follow-up decision (2026-07-30)" in plan
            and "methodology/reproducibility" in plan
            and "do not ask for score reconsideration" in plan.lower(),
        ),
    ]

    passed = sum(bool(ok) for _, ok in checks)
    return {
        "score": passed,
        "maximum": len(checks),
        "checks": [{"name": name, "passed": bool(ok)} for name, ok in checks],
        "postable_characters": len(postable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.details:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
