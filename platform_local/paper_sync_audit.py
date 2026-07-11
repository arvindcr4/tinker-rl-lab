#!/usr/bin/env python3
from utils.audit_utils import run_audit


def get_issues(ctx):
    issues = []
    checks = {
        "50-problem subset": "50-problem subset",
        "custom tool evaluation": "custom",
        "training-set reward": "training-set reward",
        "held-out": "held-out",
        "RLOO": "rloo",
        "REINFORCE++": "reinforce++",
        "Step-DPO": "step-dpo",
    }

    for label, needle in checks.items():
        if needle in ctx.main_tex and needle not in ctx.anon_tex:
            issues.append(f"anonymous_missing:{label}")

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("sync_issues", get_issues))
