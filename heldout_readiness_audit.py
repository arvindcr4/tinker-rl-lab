#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    # Evaluation ctx.script should compute CI / preserve deterministic protocol metadata.
    if "bootstrap" not in ctx.script and "confidence_interval" not in ctx.script and "ci_" not in ctx.script:
        issues.append("eval_script_missing_confidence_interval")
    if "dataset_split" not in ctx.script or "temperature" not in ctx.script or "seed" not in ctx.script:
        issues.append("eval_script_missing_protocol_metadata")
    if 'choices=["test"]' not in ctx.script and "choices=['test']" not in ctx.script:
        issues.append("eval_script_not_locked_to_test_split")
    
    # Docs should not imply universal checkpoint availability.
    if "all training runs produce tinker-hosted model checkpoints" in ctx.capstone:
        issues.append("capstone_overstates_checkpoint_availability")
    if "all runs produce tinker-hosted checkpoints" in ctx.paper:
        issues.append("paper_overstates_checkpoint_availability")
    if "all tinker training runs, logs, and model checkpoints are available" in ctx.capstone:
        issues.append("capstone_claims_all_checkpoints_available")
    
    # README should frame checkpoint access conditionally.
    if "if checkpoint still available" not in ctx.readme:
        issues.append("readme_missing_conditional_checkpoint_access_language")
    
    return issues

if __name__ == '__main__':
    run_audit('readiness_issues', get_issues)