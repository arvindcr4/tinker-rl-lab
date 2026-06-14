#!/usr/bin/env python3
from pathlib import Path

script = Path("reports/final/prepare_blind_review_package.py")
submission_path = Path("reports/final/SUBMISSION_README.md")
checklist_path = Path("reports/final/SUBMISSION_CHECKLIST.md")
submission = submission_path.read_text().lower() if submission_path.exists() else ""
checklist = checklist_path.read_text().lower() if checklist_path.exists() else ""
issues = []

if not script.exists():
    issues.append("missing_blind_review_export_script")
else:
    text = script.read_text().lower()
    if "grpo_agentic_llm_paper_anonymous.tex" not in text:
        issues.append("export_script_missing_anonymized_tex")
    if "grpo_agentic_llm_paper.tex" in text:
        issues.append("export_script_mentions_nonanonymous_tex")
    if "compiled pdfs" not in text and ".pdf" not in text:
        issues.append("export_script_missing_build_artifact_exclusion_note")

if "prepare_blind_review_package.py" not in submission:
    issues.append("submission_readme_missing_export_script_reference")
if "prepare_blind_review_package.py" not in checklist:
    issues.append("submission_checklist_missing_export_script_reference")

# TODO: The following methodological limitations identified in adversarial_review.md
# cannot be automatically audited by this script and must be manually verified in the manuscript:
# - ZVF metric fragility and reliance on ERF
# - Early-training snapshot problem (30-50 steps)
# - Closed-source Tinker API confound
# - Lack of statistically significant generalization (p=0.26 and p=0.53)
# - Single-seed extrapolations
print(f"METRIC export_issues={len(issues)}")
print("Blind-review export checks passed." if not issues else "\n".join(issues))
