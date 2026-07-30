#!/usr/bin/env python3
"""Independent holdout checks for the Reviewer 9kjk response candidate."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[1]
RESPONSE = ROOT / "zvf-program/flagship/paper/NEURIPS_2026_REVIEWER_9KJK_FOLLOWUP.md"
LEDGER = ROOT / "zvf-program/flagship/paper/REVIEWER_9KJK_SCOPE_LEDGER.json"
EARLY = ROOT / "platform_hybrid/experiments/zvf_predictive_validation_results.json"
REVIEWED_PDF = ROOT / "autoresearch/reason-260727-2155/openreview_submission_CXbcYe69BQ.pdf"
EXPECTED_PDF_SHA256 = "b15ac7e5f673473cf8edc07634f6acbd9fcd54b9f0d5d1f75b106565a174a62d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"HOLDOUT_FAIL: {message}")


def main() -> int:
    text = RESPONSE.read_text(encoding="utf-8")
    postable = text.split("<!-- POSTABLE_REPLY_START -->", 1)[1].split(
        "<!-- POSTABLE_REPLY_END -->", 1
    )[0]
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    early = json.loads(EARLY.read_text(encoding="utf-8"))["metrics"]
    pdf_sha = hashlib.sha256(REVIEWED_PDF.read_bytes()).hexdigest()

    require(pdf_sha == EXPECTED_PDF_SHA256, "reviewed PDF identity drift")
    require(early["protocol"]["deduplicated_runs"] == 22, "early-run population drift")
    require(early["n_collapse"] == 2, "collapse count drift")
    require(early["task_breakdown"]["gsm8k"]["n"] == 8, "GSM8K count drift")
    require(early["task_breakdown"]["tool_use"]["collapse"] == 2, "tool collapse drift")
    require(early["task_breakdown"]["unknown"]["n"] == 12, "unknown-task count drift")
    require(early["collapse_auc"]["early_reward_low"]["auc"] == 1.0, "reward baseline drift")

    cells = ledger["coverage_cells"]
    require(len(cells) == 20, "coverage ledger is not 5 analyses by 4 corpora")
    require(
        all(cell["status"] in {"evaluated", "not_evaluated", "quarantined"} for cell in cells),
        "unadjudicated coverage cell",
    )
    require(
        sum(cell["status"] == "not_evaluated" for cell in cells) >= 12,
        "missing-cell extent is understated",
    )

    require(len(postable.strip()) <= 5000, "reply exceeds conservative 5,000-character bound")
    require("reconsider" not in postable.lower(), "score-reconsideration request survived")
    require("withdraw the use-inspired contribution type" in postable.lower(), "use-inspired not withdrawn")
    require("HumanEval and MATH have no numerical main-result evaluation" in postable, "missing corpora hidden")
    require("cannot justify comparative inference" in postable, "single-seed explanation is excusatory")
    require("one-sample seed-level test" in postable, "seed-level test identity is ambiguous")
    require("two-case descriptive concordance" in postable, "early triage still sounds validated")
    require("prospectively justified by power or precision targets" in postable, "future replication rule is arbitrary")
    require("at least five" not in postable.lower(), "unsupported universal five-seed rule survived")
    require("post-submission" not in postable.lower() and "40-unit" not in postable.lower(), "new evidence leaked into reviewed response")
    require("DAPO" not in postable and "E1" not in postable, "future study leaked into reviewed response")
    require(
        not re.search(
            r"(?:[A-Za-z0-9_-]+\.(?:json|py|csv|tsv|md|tex|pdf)|"
            r"(?:experiments|platform_hybrid|zvf-program|autoresearch|submission)/[A-Za-z0-9_./-]+)",
            postable,
        ),
        "artifact filename or repository path leaked into postable reply",
    )

    print("HOLDOUT_REVIEWER_9KJK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
