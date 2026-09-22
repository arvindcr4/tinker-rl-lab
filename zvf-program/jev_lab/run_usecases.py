#!/usr/bin/env python3
"""Run the jev use-case layer against live campaign artifacts (2026-09-19).

Executes every pattern offline-capable now; jev-dependent judgments run only
if the service is healthy, else fall back and say so. Writes one summary
receipt. Usage:

    python3 zvf-program/jev_lab/run_usecases.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jev_judge import healthcheck  # noqa: E402
from usecases import (  # noqa: E402
    eval_judge, mine_ledgers, rerank_fuse, safety_gate,
)

REPO = HERE.parents[1]
OUT = REPO / "outputs/jev_receipts/USECASE_RUN_2026-09-19.json"


def main() -> None:
    health = healthcheck()
    summary: dict = {
        "schema_version": "jev-usecase-run-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "service_healthy": health["healthy"],
        "health": health,
    }

    # U3 — rerank & fuse: find the receipt that answers a live research question
    query = "which lane is blocked on AWS vCPU quota before any launch"
    candidates = []
    finish = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish"
    for name in (
        "e6_continuation/REPORT_v1.md", "e9_completion/HANDOFF.md",
        "e2_completion/decision_v19/lifecycle_decision.md",
        "e13_continuation/decision_v11/DECISION.md",
        "STATE_RECONCILIATION_2026-09-19.md", "DECISION_PACKAGE_2026-09-19.md",
    ):
        p = finish / name
        if p.is_file():
            candidates.append({"id": name, "text": p.read_text()[:2000]})
    summary["u3_rerank_fuse"] = rerank_fuse(query, candidates, label="u3_quota_blocker_lookup")

    # U4 — tiered eval judge: claims from the final results report
    report = (REPO / "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md").read_text()
    claims = [
        "Three replacement scopes are strictly complete: E8 public, E10 benign, E11 native.",
        "E14 Omni-MATH official accuracy 51.31 percent is reproduced with 4426 of 4428 accepted dispositions.",
        "The campaign's deterministic code checks passed 11 of 11 on 2026-09-19.",
        "Every remaining lane can be completed for about 108 dollars total new spend.",
    ]
    summary["u4_eval_judge"] = eval_judge(report, claims, label="u4_final_report")

    # U5 — safety gate reflexes
    summary["u5_safety_gate"] = {
        "destructive_example": safety_gate("git reset --hard origin/main"),
        "launch_example": safety_gate("uv run --with modal python launch_wave10.py"),
        "read_only_example": safety_gate("python3 verify_ledger_2026-09-19.py"),
    }

    # U7 — bulk ledger mining over the finish corpus (offline fallback works)
    summary["u7_mine_ledgers"] = mine_ledgers(finish, limit=40)

    OUT.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "u7_mine_ledgers"}, indent=1)[:2500])
    mined = summary["u7_mine_ledgers"]
    print(f"\nu7 mined {mined['mined']} ledgers; jev_used on "
          f"{sum(1 for l in mined['ledgers'] if l['jev_used'])}")
    print("receipt:", OUT)


if __name__ == "__main__":
    main()
