#!/usr/bin/env python3
from utils.audit_utils import run_audit
import re


def get_issues(ctx):
    issues = []

    def need(cond, code, msg):
        if not cond:
            issues.append((code, msg))

    need(
        "held-out gsm8k evaluation is still pending" in PAPER
        or "held-out test evaluation" in PAPER,
        "heldout_scope",
        "Main ctx.paper should explicitly state that held-out GSM8K evaluation is still pending and math claims are not generalization claims.",
    )
    need(
        ("kl" in PAPER and "entropy" in PAPER and "limitation" in PAPER)
        or ("kl regularization" in SUPP and "entropy" in SUPP),
        "kl_entropy_limit",
        "Paper/supplement should discuss lack of KL anchoring or entropy diagnostics as a limitation and future mitigation.",
    )
    need(
        ("custom reward-derived scenario scores" in PAPER)
        or ("inter-rater reliability was not measured" in PAPER)
        or ("custom metrics" in SUPP and "inter-rater reliability" in SUPP),
        "tool_eval_protocol",
        "Tool-calling section should clarify that the multi-turn scores are custom scenario/reward scores and that inter-rater reliability was not measured.",
    )
    need(
        ("50-problem subset" in PAPER or "50-item subset" in PAPER)
        and (
            "full humaneval" in PAPER
            or "standard harness" in PAPER
            or "non-standard subset" in PAPER
        ),
        "codegen_subset",
        "Paper should clearly disclose that code generation used a 50-problem subset rather than the full standard HumanEval harness and avoid significance claims.",
    )
    need(
        ("exploration" in PAPER and "group size" in PAPER and "temperature" in PAPER)
        or ("capacity threshold" in SUPP and "exploration" in SUPP),
        "capacity_confound",
        "Capacity-threshold discussion should explicitly acknowledge exploration/reward-sparsity confounds and name the missing ablations (group size, temperature, curriculum).",
    )
    need(
        ("routing entropy" in PAPER)
        or ("expert load imbalance" in SUPP)
        or ("routing diagnostics" in PAPER)
        or ("routing entropy" in SUPP),
        "moe_diagnostics",
        "MoE discussion should explicitly say that routing entropy / expert-load diagnostics are missing and are future work.",
    )
    need(
        ("compute budget" in PAPER or "gpu-hours" in PAPER or "tokens processed" in PAPER)
        and ("data split" in PAPER or "splits" in SUPP),
        "budget_and_splits",
        "Paper/supplement should explicitly summarize compute budgets and data split limitations.",
    )
    need(
        (
            "toolrm" in PAPER
            or "fc-rewardbench" in PAPER
            or "rloo" in PAPER
            or "reinforce++" in PAPER
            or "s-grpo" in PAPER
            or "proxy state" in PAPER
            or "qlora" in PAPER
        ),
        "related_work_positioning",
        "Paper should explicitly position itself against missing evaluation/baseline families (e.g. ToolRM, FC-RewardBench, proxy-state evaluation, RLOO/REINFORCE++, S-GRPO, QLoRA context).",
    )
    need(
        "release code" in PAPER or "release code" in REPORT or "evaluation scripts" in SUPP,
        "replication_release",
        "Report should explicitly state what code/prompts/evaluation assets are or are not released for replication.",
    )
    need(
        "near-perfect verifier score" not in PAPER,
        "ctx.paper.near_perfect_overclaim",
        "Main ctx.paper should not describe GSM8K training-set reward as near-perfect verifier score.",
    )
    need(
        "confirming the threshold" not in REPORT and "confirms the threshold" not in REPORT,
        "report.threshold_overclaim",
        "Capstone report should avoid saying a single-seed 4B result confirms the capacity threshold.",
    )
    need(
        "8b: 100% peak on gsm8k" not in REPORT,
        "report.peak_table_overclaim",
        "Summary tables should not present peak GSM8K training-step numbers in a way that reads like benchmark performance.",
    )

    return issues


if __name__ == "__main__":
    run_audit("caveat_issues", get_issues)
