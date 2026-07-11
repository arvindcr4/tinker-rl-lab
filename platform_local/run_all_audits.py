#!/usr/bin/env python3
"""Run submission audits through their structured result interface."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from platform_local import (
    anonymization_repro_audit,
    claim_strength_audit,
    export_guard_audit,
    paper_sync_audit,
    submission_claim_audit,
    submission_package_audit,
    submission_workflow_audit,
)
from utils.audit_utils import (
    AuditContext,
    AuditIssue,
    AuditSuiteResult,
    evaluate_audit,
    render_suite,
)


AuditFunction = Callable[[AuditContext], Iterable[str | AuditIssue]]
AUDITS: tuple[tuple[str, AuditFunction], ...] = (
    ("claim_issues", submission_claim_audit.get_issues),
    ("sync_issues", paper_sync_audit.get_issues),
    ("anon_issues", anonymization_repro_audit.get_issues),
    ("strength_issues", claim_strength_audit.get_issues),
    ("package_issues", submission_package_audit.get_issues),
    ("workflow_issues", submission_workflow_audit.get_issues),
    ("export_guard_issues", export_guard_audit.get_issues),
)


def run_suite(
    audits: tuple[tuple[str, AuditFunction], ...] = AUDITS,
    context: AuditContext | None = None,
) -> AuditSuiteResult:
    shared_context = context or AuditContext()
    return AuditSuiteResult(
        audits=tuple(evaluate_audit(name, audit, shared_context) for name, audit in audits)
    )


def main() -> int:
    result = run_suite()
    print(render_suite(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
