#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    if "50-problem subset" not in ctx.text:
        issues.append("missing_humaneval_subset_caveat")
    if "custom" not in ctx.text:
        issues.append("missing_tool_custom_caveat")
    if "training-set" not in ctx.text and "training reward" not in ctx.text:
        issues.append("missing_training_set_math_caveat")
    if "held-out" not in ctx.text:
        issues.append("missing_heldout_language")
    if "rloo" not in ctx.text and "reinforce++" not in ctx.text:
        issues.append("missing_baseline_positioning")
    if "reliable tool caller" in ctx.text or "reliable tool callers" in ctx.text:
        issues.append("has_reliable_tool_caller_overclaim")
    if "grpo enables significant capability gains" in ctx.text and "custom" not in ctx.text[:4000]:
        issues.append("abstract_missing_custom_eval_context")
    
    return issues

if __name__ == '__main__':
    run_audit('capstone_issues', get_issues)