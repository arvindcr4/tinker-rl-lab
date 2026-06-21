#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re

def get_issues(ctx):
    issues = []
    
    for name, text in ctx.files.items():
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", ctx.text, re.S)
        abstract = m.group(1) if m else ctx.text[:2000]
    
        if "custom" not in abstract:
            issues.append(f"{name}_abstract_missing_custom_eval_caveat")
        if "50-problem subset" not in abstract:
            issues.append(f"{name}_abstract_missing_humaneval_subset_caveat")
        if "training-set reward" not in abstract:
            issues.append(f"{name}_abstract_missing_training_reward_caveat")
        if "held-out" not in abstract:
            issues.append(f"{name}_abstract_missing_heldout_caveat")
    
    return issues

if __name__ == '__main__':
    run_audit('abstract_issues', get_issues)