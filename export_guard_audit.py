#!/usr/bin/env python3
# TODO: Address limitations from adversarial review:
# - ZVF Metric limitations (tautological, fragile across domains, symptom not root cause)
# - Early-training snapshot problem (30-50 steps)
# - Closed-source confound (Tinker API black box)
# - Failure to prove generalization (insignificant gains)
# - Single-seed extrapolations (N=1 runs)
# Note: No direct fix is applicable in this script.
from pathlib import Path

script = Path("reports/final/prepare_blind_review_package.py").read_text().lower()
issues = []

if "run_all_audits.py" not in script:
    issues.append("export_script_missing_audit_guard")
if "--skip-audits" not in script:
    issues.append("export_script_missing_skip_audits_override")
if "subprocess.run" not in script:
    issues.append("export_script_not_invoking_audit_process")

print(f"METRIC export_guard_issues={len(issues)}")
print("Export guard checks passed." if not issues else "\n".join(issues))
