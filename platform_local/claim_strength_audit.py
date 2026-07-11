#!/usr/bin/env python3
from utils.audit_utils import run_audit


def get_issues(ctx):
    issues = []

    for name, text in ctx.files.items():
        if "can reliably improve small language models" in text:
            issues.append(f"{name}_uses_reliably_improve_opening_claim")
        if "can reliably optimize reward on verifiable tasks" in text:
            issues.append(f"{name}_uses_reliably_optimize_opening_claim")
        if "practical, compute-efficient method" in text:
            issues.append(f"{name}_uses_broad_compute_efficient_claim")
        if (
            "strong gains on task-specific metrics" in text
            and "custom internal evaluation protocol" in text
        ):
            # okay; keep pass
            pass
        elif "strong gains on task-specific metrics" in text:
            issues.append(f"{name}_strong_gains_without_nearby_scope_qualifier")

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("strength_issues", get_issues))
