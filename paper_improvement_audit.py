#!/usr/bin/env python3
"""
Paper Improvement Audit
=======================
Counts unresolved reviewer issues in the GRPO paper.
Each check maps to a specific discovery report concern.
Lower is better (0 = all issues addressed).
"""

import re
import sys

def read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""

def audit_paper() -> dict:
    paper = read_file("reports/final/grpo_agentic_llm_paper.tex")
    appendix = read_file("reports/final/supplementary_appendix.tex")
    bib = read_file("reports/final/references.bib")
    full = paper + "\n" + appendix

    issues = {}

    # === Discovery 1: Evaluation Validity Gaps ===

    # 1. Training-set results not clearly labeled
    # Check if tables/captions contain "training-set" or "training reward" qualifiers
    train_reward_labels = len(re.findall(r'training[- ](?:set|reward|prompt)', full, re.I))
    if train_reward_labels < 4:  # should appear near every results table
        issues["D1_train_label"] = "Training-set results not consistently labeled (found %d, need >=4)" % train_reward_labels

    # 2. Missing comprehensive methods table (compute budgets)
    if not re.search(r'GPU[- ]hours|tokens processed|compute budget|wall[- ]clock', full, re.I):
        issues["D1_compute_budget"] = "No compute budget table (GPU-hours, tokens, wall-clock)"

    # 3. Missing data sizes/splits table
    if not re.search(r'dataset.*size|train.*split.*test|data.*composition', full, re.I):
        issues["D1_data_splits"] = "No explicit dataset sizes/splits documentation"

    # 4. Missing decoding settings documentation
    if not re.search(r'(?:decoding|greedy|temperature.*sampling|Decoding.*Protocol|evaluation.*protocol)', full, re.I):
        issues["D1_decoding"] = "No decoding settings documented for evaluation"

    # === Discovery 2: Capacity/Exploration/Reward Sparsity ===

    # 5. Group composition analysis missing
    if not re.search(r'frac.*(?:all[_-]?bad|all[_-]?good|mixed)|group.*composition|zero[- ](?:reward|advantage).*(?:fraction|rate|percent)', full, re.I):
        issues["D2_group_composition"] = "No group-composition analysis (frac_all_bad/good/mixed)"

    # 6. Exploration confounds not discussed quantitatively
    if not re.search(r'group.*size.*(?:32|64)|temperature.*(?:sweep|ablat)|curriculum.*(?:learning|strategy)', full, re.I):
        issues["D2_exploration_ablations"] = "No discussion of group-size/temperature/curriculum rescue ablations"

    # 7. Missing comparison to RLOO/REINFORCE++
    if "RLOO" not in full and "REINFORCE++" not in full:
        issues["D2_rloo_reinforce"] = "No mention of RLOO/REINFORCE++ baselines"

    # === Discovery 3: Training Stability & MoE Diagnostics ===

    # 8. KL/entropy telemetry not reported for main experiments
    if not re.search(r'entropy.*(?:collapse|trajectory|decay)|KL.*(?:trajectory|divergence.*SFT|anchor)', full, re.I):
        issues["D3_kl_entropy"] = "No KL/entropy telemetry reported for main experiments"

    # 9. MoE routing diagnostics missing
    if not re.search(r'router.*(?:entropy|shift|metric)|expert.*(?:load|balance|utiliz)', full, re.I):
        issues["D3_moe_routing"] = "No MoE routing diagnostics (router entropy, expert load)"

    # 10. Zero-loss step analysis not quantified
    if not re.search(r'zero[- ]loss.*(?:\d+%|fraction|rate)|skipped.*update', full, re.I):
        issues["D3_zero_loss"] = "Zero-loss step frequency not quantified across models"

    # === Discovery 4: PEFT Baseline Positioning ===

    # 11. Missing ToolRM/FC-RewardBench references
    if "ToolRM" not in full and "FC-RewardBench" not in full:
        issues["D4_tool_benchmarks"] = "No reference to ToolRM/FC-RewardBench"

    # 12. Missing S-GRPO/StepGRPO references
    if "S-GRPO" not in full and "StepGRPO" not in full and "step-wise GRPO" not in full.lower():
        issues["D4_step_grpo"] = "No reference to step-wise GRPO variants"

    # 13. Missing DPO/Step-DPO as baselines
    if not re.search(r'DPO.*baseline|compare.*DPO|DPO.*comparison', full, re.I):
        issues["D4_dpo_baseline"] = "DPO not positioned as comparison baseline"

    # 14. QR-Adaptor/LoTA-QAF not discussed
    if "QR-Adaptor" not in full and "LoTA" not in full and "QR-LoRA" not in full:
        issues["D4_peft_context"] = "No discussion of advanced PEFT methods (QR-Adaptor/LoTA-QAF)"

    # 15. Missing future experiments section with standardized eval plan
    if not re.search(r'(?:future|planned).*(?:experiment|evaluation|ablation).*(?:plan|roadmap|protocol)', full, re.I):
        issues["D_future_plan"] = "No structured future experiments/evaluation plan"

    # === Cross-cutting ===

    # 16. Related work too thin
    related_work_match = re.search(r'\\section\{Related Work\}(.*?)\\section', full, re.S)
    if related_work_match:
        rw_text = related_work_match.group(1)
        rw_cites = len(re.findall(r'\\cite[tp]?\{', rw_text))
        if rw_cites < 8:
            issues["X_related_work_thin"] = "Related work has only %d citations (need >=8)" % rw_cites

    # 17. References too few
    bib_entries = len(re.findall(r'@\w+\{', bib))
    if bib_entries < 15:
        issues["X_bib_entries"] = "Bibliography has only %d entries (need >=15)" % bib_entries

    return issues


def main():
    issues = audit_paper()
    n = len(issues)

    print(f"METRIC reviewer_issues={n}")
    print(f"METRIC total_checks=17")
    print(f"METRIC resolved={17 - n}")

    if issues:
        print(f"\n--- {n} UNRESOLVED ISSUES ---")
        for k, v in sorted(issues.items()):
            print(f"  [{k}] {v}")
    else:
        print("\nAll 17 reviewer issues resolved!")

    return n


if __name__ == "__main__":
    sys.exit(main())
