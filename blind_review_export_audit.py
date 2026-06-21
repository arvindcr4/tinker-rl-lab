#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if not (ctx.FINAL_DIR / 'prepare_blind_review_package.py').exists():
        issues.append("missing_blind_review_export_script")
    else:
        text = (ctx.FINAL_DIR / 'prepare_blind_review_package.py').read_text().lower()
        if "grpo_agentic_llm_paper_anonymous.tex" not in text:
            issues.append("export_script_missing_anonymized_tex")
        if "grpo_agentic_llm_paper.tex" in text:
            issues.append("export_script_mentions_nonanonymous_tex")
        if "compiled pdfs" not in text and ".pdf" not in text:
            issues.append("export_script_missing_build_artifact_exclusion_note")
    
    if "prepare_blind_review_package.py" not in ctx.submission:
        issues.append("submission_readme_missing_export_script_reference")
    if "prepare_blind_review_package.py" not in ctx.checklist:
        issues.append("submission_checklist_missing_export_script_reference")
    
    return issues

if __name__ == '__main__':
    run_audit('export_issues', get_issues)