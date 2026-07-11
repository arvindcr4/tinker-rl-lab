#!/usr/bin/env python3
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audit_utils import run_audit


def get_issues(ctx):
    issues = []

    # The export/bundling script must invoke the audit suite before packaging
    # and must support a documented skip override.
    if "platform_local/run_all_audits.py" not in ctx.export_script:
        issues.append("export_script_missing_audit_guard")
    if "--skip-audits" not in ctx.export_script:
        issues.append("export_script_missing_skip_audits_override")
    if "subprocess.run" not in ctx.export_script:
        issues.append("export_script_not_invoking_audit_process")

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("export_guard_issues", get_issues))
