#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if "python platform_local/run_all_audits.py" not in ctx.submission:
        issues.append("submission_readme_missing_audit_suite_step")
    if "do not include generated build artifacts" not in ctx.submission:
        issues.append("submission_readme_missing_build_artifact_exclusion_note")
    if (
        "remove this section or replace it with the venue's anonymized contact mechanism"
        not in ctx.submission
    ):
        issues.append("submission_readme_missing_blind_review_contact_note")
    
    return issues

if __name__ == '__main__':
    run_audit('workflow_issues', get_issues)