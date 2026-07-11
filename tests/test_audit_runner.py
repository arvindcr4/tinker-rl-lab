from __future__ import annotations

from platform_local.run_all_audits import run_suite
from utils.audit_utils import (
    AuditContext,
    AuditIssue,
    AuditResult,
    evaluate_audit,
    render_audit,
    render_suite,
)


def test_audit_result_is_the_cli_and_test_surface():
    result = evaluate_audit(
        "demo_issues",
        lambda _context: ["first", AuditIssue("second")],
        AuditContext(),
    )

    assert result == AuditResult(
        name="demo_issues",
        issues=(AuditIssue("first"), AuditIssue("second")),
    )
    assert not result.passed
    assert render_audit(result) == "METRIC demo_issues=2\nfirst\nsecond"


def test_audit_runner_collects_results_without_subprocess_or_regex():
    context = AuditContext()
    suite = run_suite(
        audits=(
            ("passing", lambda _context: []),
            ("failing", lambda _context: ["problem"]),
        ),
        context=context,
    )

    assert [result.name for result in suite.audits] == ["passing", "failing"]
    assert [result.name for result in suite.failures] == ["failing"]
    rendered = render_suite(suite)
    assert "METRIC audits_total=2" in rendered
    assert "METRIC suite_issues=1" in rendered


def test_repository_audit_suite_currently_passes():
    assert run_suite().passed
