#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []

    # Submission README must explicitly tell submitters to use the anonymized
    # paper source/package for blind review, and to exclude the non-anonymous
    # version from blind-review bundles.
    submission_lc = ctx.submission
    if "for blind review submissions" not in submission_lc or "anonymized" not in submission_lc or "source/package" not in submission_lc:
        issues.append("submission_readme_missing_anonymized_package_guidance")
    if "exclude the non-anonymous" not in submission_lc or "blind-review bundles" not in submission_lc:
        issues.append("submission_readme_missing_nonanonymous_exclusion_note")

    # Checklist should reference an anonymized package note when it discusses
    # anonymous submission.
    checklist_lc = ctx.checklist
    if "anonymous submission" in checklist_lc and "anonymized paper source/package" not in checklist_lc:
        issues.append("checklist_missing_anonymized_package_note")

    return issues

if __name__ == '__main__':
    run_audit('blind_package_issues', get_issues)