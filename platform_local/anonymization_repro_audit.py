#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    # Blind-review hygiene.
    if "pes university" in ctx.submission or "mtech dsai" in ctx.submission:
        issues.append("submission_readme_contains_institution_contact")
    if "huggingface.co/madhu2133" in ctx.supp:
        issues.append("supplementary_contains_identifying_hf_username")
    
    # Reproducibility language should avoid brittle/local-only paths.
    if "/tmp/gsm8k_" in ctx.supp or "/tmp/grpo_" in ctx.supp:
        issues.append("supplementary_contains_local_tmp_log_paths")
    if "model checkpoints are hosted on tinker and huggingface hub" in ctx.supp:
        issues.append("supplementary_overstates_checkpoint_hosting_availability")
    
    # Anonymous ctx.paper should stay anonymous.
    if "anonymous institution" not in ctx.anon:
        issues.append("anonymous_paper_missing_anonymous_institution_marker")
    if "acknowledgments" in ctx.anon and "pes university" in ctx.anon:
        issues.append("anonymous_paper_contains_institution_in_acknowledgments")
    
    return issues

if __name__ == '__main__':
    run_audit('anon_issues', get_issues)