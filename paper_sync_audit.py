#!/usr/bin/env python3
from pathlib import Path

main_tex = Path("reports/final/grpo_agentic_llm_paper.tex").read_text().lower()
anon_tex = Path("reports/final/grpo_agentic_llm_paper_anonymous.tex").read_text().lower()
# markdown file is superseded header-only; skip sync checks

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
    if needle in main_tex and needle not in anon_tex:
        issues.append(f"anonymous_missing:{label}")

print(f"METRIC sync_issues={len(issues)}")
print("All paper mirror sync checks passed." if not issues else "\n".join(issues))
