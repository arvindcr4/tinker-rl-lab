from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("toolchain", ["pdflatex", "tectonic", "pdflatex_only"])
@pytest.mark.parametrize("outcome", ["success", "build_failure", "missing_tool", "empty_journal"])
def test_scientific_builds_preserve_source_artifacts(tmp_path, monkeypatch, toolchain, outcome):
    compiler = "pdflatex" if toolchain == "pdflatex" else "tectonic"
    available_tools = {
        "pdflatex": {"pdflatex", "bibtex"},
        "tectonic": {"tectonic"},
        "pdflatex_only": {"pdflatex", "tectonic"},
    }[toolchain]
    context = AuditContext()
    context.ROOT = tmp_path
    context.FINAL_DIR = tmp_path / "paper sources"
    context.FINAL_DIR.mkdir()
    for name in (
        "grpo_agentic_llm_paper.tex",
        "grpo_agentic_llm_paper_anonymous.tex",
        "grpo_agentic_llm_paper.md",
        "capstone_final_report.md",
        "SUBMISSION_CHECKLIST.md",
        "supplementary_appendix.tex",
        "evaluate_gsm8k_test.py",
    ):
        (context.FINAL_DIR / name).write_text("")
    for stem in (
        "grpo_agentic_llm_paper",
        "grpo_agentic_llm_paper_anonymous",
        "supplementary_appendix",
    ):
        for suffix in (".pdf", ".aux", ".bbl", ".blg", ".log", ".out"):
            (context.FINAL_DIR / f"{stem}{suffix}").write_bytes(b"author artifact")
    before = {path.name: path.read_bytes() for path in context.FINAL_DIR.iterdir()}
    calls = []
    output_dirs = []
    monkeypatch.setenv("BIBINPUTS", f"existing-bib-search{os.pathsep}")
    monkeypatch.setattr(
        scientific_audit.shutil,
        "which",
        lambda name: name if name in available_tools else None,
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        output_dir = Path(kwargs["cwd"])
        if command[0] == "pdflatex":
            for arg in command:
                if arg.startswith("-output-directory="):
                    output_dir = Path(arg.partition("=")[2])
            assert Path(kwargs["cwd"]) == context.FINAL_DIR
        elif command[0] == "tectonic":
            output_dir = Path(command[command.index("--outdir") + 1])
            assert "--only-cached" in command
            assert "--print" in command
            assert Path(kwargs["cwd"]) == context.FINAL_DIR
        else:
            assert kwargs["env"]["BIBINPUTS"] == (
                f"{context.FINAL_DIR}{os.pathsep}existing-bib-search{os.pathsep}"
            )
        output_dirs.append(output_dir)
        (output_dir / f"{Path(command[-1]).stem}.pdf").write_bytes(b"audit output")
        if outcome == "missing_tool":
            raise FileNotFoundError(2, "No such file or directory", command[0])
        stdout = (
            "Warning--empty journal in example"
            if outcome == "empty_journal" and command[0] in {"bibtex", "tectonic"}
            else ""
        )
        return SimpleNamespace(returncode=int(outcome == "build_failure"), stdout=stdout)

    monkeypatch.setattr(scientific_audit.subprocess, "run", fake_run)
    result = evaluate_audit("scientific_issues", scientific_audit.get_issues, context)

    assert {path.name: path.read_bytes() for path in context.FINAL_DIR.iterdir()} == before
    assert output_dirs and all(path != context.FINAL_DIR for path in output_dirs)
    assert all(not path.exists() for path in output_dirs)
    latex_codes = [issue.code for issue in result.issues if issue.code.startswith("latex.")]
    if outcome in {"success", "empty_journal"}:
        assert len(calls) == (7 if compiler == "pdflatex" else 3)
        warning_count = (1 if compiler == "pdflatex" else 3) if outcome == "empty_journal" else 0
        assert latex_codes == ["latex.bibtex.empty_journal"] * warning_count
    elif outcome == "build_failure":
        assert len(calls) == 1
        assert latex_codes == [
            "latex.main.pass1" if compiler == "pdflatex" else "latex.main.tectonic"
        ]
    else:
        assert len(calls) == 1
        assert latex_codes == [f"latex.tool_missing:{compiler}"]


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
