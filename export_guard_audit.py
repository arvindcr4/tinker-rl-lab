#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if "run_all_audits.py" not in ctx.script:
        issues.append("export_script_missing_audit_guard")
    if "--skip-audits" not in ctx.script:
        issues.append("export_script_missing_skip_audits_override")
    if "subprocess.run" not in ctx.script:
        issues.append("export_script_not_invoking_audit_process")
    
    return issues

if __name__ == '__main__':
    run_audit('export_guard_issues', get_issues)