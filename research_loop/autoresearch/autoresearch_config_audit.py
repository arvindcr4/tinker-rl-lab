#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if "suite_issues" not in ctx.md:
        issues.append("autoresearch_md_missing_suite_metric")
    if "run_all_audits.py" not in ctx.md:
        issues.append("autoresearch_md_missing_unified_suite_reference")
    if "reviewer_issues" not in ctx.md:
        issues.append("autoresearch_md_missing_primary_reviewer_metric_context")
    
    return issues

if __name__ == '__main__':
    run_audit('config_issues', get_issues)