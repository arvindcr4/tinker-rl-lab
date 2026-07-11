#!/usr/bin/env python3
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audit_utils import run_audit


def get_issues(ctx):
    issues = []

    # Build artifacts should not sit in the review package directory.
    for name in [
        "grpo_agentic_llm_paper_anonymous.aux",
        "grpo_agentic_llm_paper_anonymous.log",
        "grpo_agentic_llm_paper_anonymous.out",
        "grpo_agentic_llm_paper_anonymous.pdf",
    ]:
        if (ctx.FINAL_DIR / name).exists():
            issues.append(f"build_artifact_present:{name}")

    submission_text = (ctx.FINAL_DIR / "SUBMISSION_README.md").read_text().lower()
    if "fresh clone" not in submission_text:
        issues.append("submission_readme_missing_clean_export_guidance")
    if "do not include generated build artifacts" not in submission_text:
        issues.append("submission_readme_missing_build_artifact_exclusion_note")

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("package_issues", get_issues))
