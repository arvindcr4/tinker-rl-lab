#!/usr/bin/env python3
import re
from pathlib import Path

readme = Path("reports/final/README.md").read_text()
checklist = Path("reports/final/SUBMISSION_CHECKLIST.md").read_text()
submission = Path("reports/final/SUBMISSION_README.md").read_text()

issues = []

# README should clearly label non-standardized headline metrics.
if "## Key Results (Training-Set)" not in readme:
    issues.append("readme_missing_key_results_scope_header")
if "tool results are internal/custom" not in readme.lower():
    issues.append("readme_missing_tool_custom_caveat")
if "50-problem" not in readme.lower() or "subset" not in readme.lower():
    issues.append("readme_missing_humaneval_subset_caveat")
if "held-out" not in readme.lower():
    issues.append("readme_missing_heldout_language")

# Checklist should not present headline results without caveats.
if "Key Results to Highlight" in checklist and "preliminary / custom" not in checklist.lower():
    issues.append("checklist_missing_preliminary_key_results_label")
if "50-problem subset" not in checklist.lower():
    issues.append("checklist_missing_humaneval_subset_note")
if "custom internal" not in checklist.lower() and "custom judge-derived" not in checklist.lower():
    issues.append("checklist_missing_tool_custom_note")
if "training-set reward" not in checklist.lower():
    issues.append("checklist_missing_training_set_math_note")
if "checkpoints available" in checklist.lower() or "checkpoints available" in readme.lower():
    issues.append("misleading_checkpoint_availability_claim")

# Submission README should also preserve caveats and avoid completion claims.
if "Key Results Summary" in submission and "preliminary / custom" not in submission.lower():
    issues.append("submission_missing_preliminary_label")
if "50-problem subset" not in submission.lower():
    issues.append("submission_missing_humaneval_subset_note")
if "custom internal" not in submission.lower() and "custom judge-derived" not in submission.lower():
    issues.append("submission_missing_tool_custom_note")
if "training-set reward" not in submission.lower():
    issues.append("submission_missing_training_set_math_note")
if "model checkpoint urls" in submission.lower():
    issues.append("submission_overstates_checkpoint_release")
if re.search(r"\[x\].*9-page limit satisfied", submission):
    issues.append("submission_has_unverified_page_count_checkbox")

# TODO: Add checks for the limitations identified in the adversarial review (adversarial_review.md):
# 1. ZVF metric fragility: Not a programmatic framework; breaks down outside math tasks (saturates at 1.0 for format-gated tasks, requiring ERF).
# 2. Early-training snapshot problem: 30-50 step training runs due to API budget constraints limit conclusions about asymptotic convergence.
# 3. Closed-source confound: 73% performance gap with Tinker API may be due to undisclosed managed defaults, not algorithmic superiority.
# 4. Lack of statistically significant generalization: Held-out GSM8K (+1.3%, p=0.26) and HumanEval (p=0.53) gains are not statistically significant.
# 5. Single-seed extrapolations: Frontier model analyses (e.g., MoE routing volatility, Nemotron-120B collapse) rely on highly variant N=1 runs.

print(f"METRIC claim_issues={len(issues)}")
print("All submission claim checks passed." if not issues else "\n".join(issues))
