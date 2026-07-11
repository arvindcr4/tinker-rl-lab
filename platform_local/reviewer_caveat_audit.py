#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audit_utils import AuditIssue, run_audit


def get_issues(ctx):
    issues: list[AuditIssue] = []
    paper = ctx.paper
    supplementary = ctx.supp
    report = ctx.capstone

    def need(cond, code, msg):
        if not cond:
            issues.append(AuditIssue(code=code, message=msg))

    need(
        "held-out gsm8k evaluation is still pending" in paper
        or "held-out test evaluation" in paper,
        "heldout_scope",
        "Main ctx.paper should explicitly state that held-out GSM8K evaluation is still pending and math claims are not generalization claims.",
    )
    need(
        ("kl" in paper and "entropy" in paper and "limitation" in paper)
        or ("kl regularization" in supplementary and "entropy" in supplementary),
        "kl_entropy_limit",
        "Paper/supplement should discuss lack of KL anchoring or entropy diagnostics as a limitation and future mitigation.",
    )
    need(
        ("custom reward-derived scenario scores" in paper)
        or ("inter-rater reliability was not measured" in paper)
        or ("custom metrics" in supplementary and "inter-rater reliability" in supplementary),
        "tool_eval_protocol",
        "Tool-calling section should clarify that the multi-turn scores are custom scenario/reward scores and that inter-rater reliability was not measured.",
    )
    need(
        ("50-problem subset" in paper or "50-item subset" in paper)
        and (
            "full humaneval" in paper
            or "standard harness" in paper
            or "non-standard subset" in paper
        ),
        "codegen_subset",
        "Paper should clearly disclose that code generation used a 50-problem subset rather than the full standard HumanEval harness and avoid significance claims.",
    )
    need(
        ("exploration" in paper and "group size" in paper and "temperature" in paper)
        or ("capacity threshold" in supplementary and "exploration" in supplementary),
        "capacity_confound",
        "Capacity-threshold discussion should explicitly acknowledge exploration/reward-sparsity confounds and name the missing ablations (group size, temperature, curriculum).",
    )
    need(
        ("routing entropy" in paper)
        or ("expert load imbalance" in supplementary)
        or ("routing diagnostics" in paper)
        or ("routing entropy" in supplementary),
        "moe_diagnostics",
        "MoE discussion should explicitly say that routing entropy / expert-load diagnostics are missing and are future work.",
    )
    need(
        ("compute budget" in paper or "gpu-hours" in paper or "tokens processed" in paper)
        and ("data split" in paper or "splits" in supplementary),
        "budget_and_splits",
        "Paper/supplement should explicitly summarize compute budgets and data split limitations.",
    )
    need(
        (
            "toolrm" in paper
            or "fc-rewardbench" in paper
            or "rloo" in paper
            or "reinforce++" in paper
            or "s-grpo" in paper
            or "proxy state" in paper
            or "qlora" in paper
        ),
        "related_work_positioning",
        "Paper should explicitly position itself against missing evaluation/baseline families (e.g. ToolRM, FC-RewardBench, proxy-state evaluation, RLOO/REINFORCE++, S-GRPO, QLoRA context).",
    )
    need(
        "release code" in paper
        or "release code" in report
        or "evaluation scripts" in supplementary,
        "replication_release",
        "Report should explicitly state what code/prompts/evaluation assets are or are not released for replication.",
    )
    need(
        "near-perfect verifier score" not in paper,
        "ctx.paper.near_perfect_overclaim",
        "Main ctx.paper should not describe GSM8K training-set reward as near-perfect verifier score.",
    )
    need(
        "confirming the threshold" not in report and "confirms the threshold" not in report,
        "report.threshold_overclaim",
        "Capstone report should avoid saying a single-seed 4B result confirms the capacity threshold.",
    )
    need(
        "8b: 100% peak on gsm8k" not in report,
        "report.peak_table_overclaim",
        "Summary tables should not present peak GSM8K training-step numbers in a way that reads like benchmark performance.",
    )

    return issues


if __name__ == "__main__":
    raise SystemExit(run_audit("caveat_issues", get_issues))
