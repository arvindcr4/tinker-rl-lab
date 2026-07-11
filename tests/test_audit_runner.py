from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from platform_local import reviewer_caveat_audit, scientific_audit
from platform_local.run_all_audits import AUDITS
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


def test_repository_suite_registers_every_audit_module():
    audit_dir = Path(__file__).parents[1] / "platform_local"
    expected_modules = {
        f"platform_local.{path.stem}"
        for path in audit_dir.glob("*_audit.py")
        if path.name not in {"run_all_audits.py"}
    }
    registered_modules = {audit.__module__ for _, audit in AUDITS}

    assert registered_modules == expected_modules


def test_reviewer_caveat_audit_reads_from_the_shared_context():
    result = evaluate_audit(
        "caveat_issues",
        reviewer_caveat_audit.get_issues,
        AuditContext(),
    )

    assert isinstance(result, AuditResult)


def test_scientific_audit_executes_its_grouped_checks(monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = ""

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return Completed()

    monkeypatch.setattr(scientific_audit.subprocess, "run", fake_run)
    result = evaluate_audit(
        "scientific_issues",
        scientific_audit.get_issues,
        AuditContext(),
    )

    assert isinstance(result, AuditResult)
    assert calls, "the grouped scientific checks must not silently return without running"


def test_every_audit_compatibility_entrypoint_runs_without_traceback():
    repo_root = Path(__file__).parents[1]
    audit_scripts = sorted((repo_root / "platform_local").glob("*_audit.py"))

    for script in audit_scripts:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode in {0, 1}, output
        assert "Traceback" not in output, f"{script.name}: {output}"
