from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """One stable, machine-readable audit finding."""

    code: str


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Structured result returned by every repository audit."""

    name: str
    issues: tuple[AuditIssue, ...] = ()

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def passed(self) -> bool:
        return not self.issues

    def metric_line(self) -> str:
        return f"METRIC {self.name}={self.issue_count}"


@dataclass(frozen=True, slots=True)
class AuditSuiteResult:
    """Collection of audit results with one suite-level exit decision."""

    audits: tuple[AuditResult, ...]

    @property
    def failures(self) -> tuple[AuditResult, ...]:
        return tuple(result for result in self.audits if not result.passed)

    @property
    def passed(self) -> bool:
        return not self.failures


# Provide standard file context lazily to avoid redundant I/O on import
class AuditContext:
    def __init__(self):
        self.ROOT = Path(__file__).resolve().parent.parent
        self.FINAL_DIR = self.ROOT / "platform_tinker" / "reports" / "final"

        self._cache = {}

    def read_text(self, filepath):
        if filepath not in self._cache:
            try:
                self._cache[filepath] = Path(filepath).read_text()
            except FileNotFoundError:
                self._cache[filepath] = ""
        return self._cache[filepath]

    @property
    def anon(self):
        return self.read_text(self.FINAL_DIR / "grpo_agentic_llm_paper_anonymous.tex").lower()

    @property
    def anon_tex(self):
        return self.anon

    @property
    def capstone(self):
        return self.read_text(self.FINAL_DIR / "capstone_final_report.md").lower()

    @property
    def checklist(self):
        return self.read_text(self.FINAL_DIR / "SUBMISSION_CHECKLIST.md").lower()

    @property
    def main_tex(self):
        return self.read_text(self.FINAL_DIR / "grpo_agentic_llm_paper.tex").lower()

    @property
    def paper(self):
        return self.main_tex

    @property
    def readme(self):
        return self.read_text(self.FINAL_DIR / "README.md").lower()

    @property
    def export_script(self):
        return self.read_text(self.FINAL_DIR / "prepare_blind_review_package.py").lower()

    @property
    def script(self):
        return self.read_text(self.FINAL_DIR / "evaluate_gsm8k_test.py").lower()

    @property
    def submission(self):
        return self.read_text(self.FINAL_DIR / "SUBMISSION_README.md").lower()

    @property
    def supp(self):
        return self.read_text(self.FINAL_DIR / "supplementary_appendix.tex").lower()

    @property
    def text(self):
        return self.read_text(self.FINAL_DIR / "capstone_final_report.tex").lower()

    @property
    def files(self):
        return {"main_tex": self.main_tex, "anon_tex": self.anon_tex}


AuditFunction = Callable[[AuditContext], Iterable[str | AuditIssue]]


def evaluate_audit(
    metric_name: str,
    get_issues: AuditFunction,
    context: AuditContext | None = None,
) -> AuditResult:
    """Run one audit through the structured result interface."""
    raw_issues = get_issues(context or AuditContext())
    issues = tuple(
        issue if isinstance(issue, AuditIssue) else AuditIssue(str(issue)) for issue in raw_issues
    )
    return AuditResult(name=metric_name, issues=issues)


def render_audit(result: AuditResult) -> str:
    lines = [result.metric_line(), *(issue.code for issue in result.issues)]
    return "\n".join(lines)


def render_suite(result: AuditSuiteResult) -> str:
    lines: list[str] = []
    for audit in result.audits:
        lines.extend((f"=== {audit.name} ===", render_audit(audit), ""))
    lines.extend(
        (
            f"METRIC suite_issues={len(result.failures)}",
            f"METRIC audits_total={len(result.audits)}",
            f"METRIC audits_passing={len(result.audits) - len(result.failures)}",
        )
    )
    if result.failures:
        lines.append("Failing audits:")
        lines.extend(f"  - {audit.name}: issues={audit.issue_count}" for audit in result.failures)
    else:
        lines.append("All audits passing.")
    return "\n".join(lines)


def run_audit(metric_name: str, get_issues: AuditFunction) -> int:
    """Compatibility CLI for invoking one audit module directly."""
    result = evaluate_audit(metric_name, get_issues)
    print(render_audit(result))
    return 0 if result.passed else 1
