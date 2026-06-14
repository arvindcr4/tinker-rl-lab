#!/usr/bin/env python3
from pathlib import Path

root = Path("reports/final")
issues = []

# TODO: The following methodological limitations were raised in the adversarial review 
# and cannot be fixed by this packaging audit script. They require changes to the paper/experiments:
# - ZVF Metric flaws: Borderline tautological, fragile across domains (e.g., tool-use), symptom not root cause.
# - Early-Training Snapshot: 30-50 step training runs are insufficient to observe asymptotic RL dynamics.
# - Closed-Source Confound: Tinker is a closed-source black box, confounding algorithmic comparisons.
# - Failure to Prove Generalization: Lack of statistically significant generalization (+1.3% on GSM8K, p=0.26; HumanEval p=0.53).
# - Single-Seed Extrapolations: Frontier model analyses rely on N=1 runs, a major statistical vulnerability.

# Build artifacts should not sit in the review package directory.
for name in [
    "grpo_agentic_llm_paper_anonymous.aux",
    "grpo_agentic_llm_paper_anonymous.log",
    "grpo_agentic_llm_paper_anonymous.out",
    "grpo_agentic_llm_paper_anonymous.pdf",
]:
    if (root / name).exists():
        issues.append(f"build_artifact_present:{name}")

submission = (root / "SUBMISSION_README.md").read_text().lower()
if "fresh clone" not in submission:
    issues.append("submission_readme_missing_clean_export_guidance")
if "do not include generated build artifacts" not in submission:
    issues.append("submission_readme_missing_build_artifact_exclusion_note")

print(f"METRIC package_issues={len(issues)}")
print("Submission packaging checks passed." if not issues else "\n".join(issues))
