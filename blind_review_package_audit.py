#!/usr/bin/env python3
from pathlib import Path

# TODO: Address limitations identified in adversarial review:
# - ZVF is borderline tautological and fragile across domains (symptom, not root cause).
# - The "Early-Training Snapshot" Problem (30-50 step runs).
# - The Closed-Source Confound (Tinker API is a black box).
# - Failure to Prove Generalization (results are not statistically significant).
# - Single-Seed Extrapolations (N=1 runs).
submission = Path("reports/final/SUBMISSION_README.md").read_text().lower()
checklist = Path("reports/final/SUBMISSION_CHECKLIST.md").read_text().lower()
issues = []

if "for blind review submissions, submit the anonymized paper source/package" not in submission:
    issues.append("submission_readme_missing_anonymized_package_guidance")
if "exclude the non-anonymous paper source from blind-review bundles" not in submission:
    issues.append("submission_readme_missing_nonanonymous_exclusion_note")
if "anonymous submission" in checklist and "anonymized paper source/package" not in checklist:
    issues.append("checklist_missing_anonymized_package_note")

print(f"METRIC blind_package_issues={len(issues)}")
print("Blind-review package checks passed." if not issues else "\n".join(issues))
