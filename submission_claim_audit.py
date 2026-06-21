#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    # README should clearly label non-standardized headline metrics.
    if "## Key Results (Training-Set)" not in ctx.readme:
        issues.append("readme_missing_key_results_scope_header")
    if "tool results are internal/custom" not in ctx.readme.lower():
        issues.append("readme_missing_tool_custom_caveat")
    if "50-problem" not in ctx.readme.lower() or "subset" not in ctx.readme.lower():
        issues.append("readme_missing_humaneval_subset_caveat")
    if "held-out" not in ctx.readme.lower():
        issues.append("readme_missing_heldout_language")
    
    # Checklist should not present headline results without caveats.
    if "Key Results to Highlight" in ctx.checklist and "preliminary / custom" not in ctx.checklist.lower():
        issues.append("checklist_missing_preliminary_key_results_label")
    if "50-problem subset" not in ctx.checklist.lower():
        issues.append("checklist_missing_humaneval_subset_note")
    if "custom internal" not in ctx.checklist.lower() and "custom judge-derived" not in ctx.checklist.lower():
        issues.append("checklist_missing_tool_custom_note")
    if "training-set reward" not in ctx.checklist.lower():
        issues.append("checklist_missing_training_set_math_note")
    if "checkpoints available" in ctx.checklist.lower() or "checkpoints available" in ctx.readme.lower():
        issues.append("misleading_checkpoint_availability_claim")
    
    # Submission README should also preserve caveats and avoid completion claims.
    if "Key Results Summary" in ctx.submission and "preliminary / custom" not in ctx.submission.lower():
        issues.append("submission_missing_preliminary_label")
    if "50-problem subset" not in ctx.submission.lower():
        issues.append("submission_missing_humaneval_subset_note")
    if "custom internal" not in ctx.submission.lower() and "custom judge-derived" not in ctx.submission.lower():
        issues.append("submission_missing_tool_custom_note")
    if "training-set reward" not in ctx.submission.lower():
        issues.append("submission_missing_training_set_math_note")
    if "model checkpoint urls" in ctx.submission.lower():
        issues.append("submission_overstates_checkpoint_release")
    if re.search(r"\[x\].*9-page limit satisfied", ctx.submission):
        issues.append("submission_has_unverified_page_count_checkbox")
    
    # Ideas backlog should stay focused on unrun, concrete experiment paths.
    if "ctx.paper-improvement roadmap" in ctx.ideas.lower() or "plan audit" in ctx.ideas.lower():
        issues.append("ideas_contains_already_done_meta_work")
    
    return issues

if __name__ == '__main__':
    run_audit('claim_issues', get_issues)