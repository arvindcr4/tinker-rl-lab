#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if "for blind review submissions, submit the anonymized ctx.paper source/package" not in ctx.submission:
        issues.append("submission_readme_missing_anonymized_package_guidance")
    if "exclude the non-anonymous ctx.paper source from blind-review bundles" not in ctx.submission:
        issues.append("submission_readme_missing_nonanonymous_exclusion_note")
    if "anonymous ctx.submission" in ctx.checklist and "anonymized ctx.paper source/package" not in ctx.checklist:
        issues.append("checklist_missing_anonymized_package_note")
    
    return issues

if __name__ == '__main__':
    run_audit('blind_package_issues', get_issues)