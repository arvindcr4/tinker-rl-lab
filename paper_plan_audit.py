#!/usr/bin/env python3
from pathlib import Path

plan = Path('reports/final/PAPER_IMPROVEMENT_PLAN.md').read_text()
readme = Path('reports/final/README.md').read_text()
checklist = Path('reports/final/SUBMISSION_CHECKLIST.md').read_text()

issues = []
required_plan_terms = [
    'held-out gsm8k',
    'humaneval',
    'tool-calling',
    'rloo',
    'reinforce++',
    'dpo',
    'moe',
    'kl',
    'entropy',
    'compute budget',
    'confidence intervals',
]
for term in required_plan_terms:
    if term not in plan.lower():
        issues.append(f'missing_plan_term:{term}')
if 'PAPER_IMPROVEMENT_PLAN.md' not in readme:
    issues.append('readme_missing_plan_link')
if 'Held-out GSM8K evaluation' not in checklist:
    issues.append('checklist_missing_heldout_eval')

print(f'METRIC plan_issues={len(issues)}')
print(f'METRIC required_terms={len(required_plan_terms)}')
print(f'METRIC covered_terms={len(required_plan_terms) - sum(1 for t in required_plan_terms if t not in plan.lower())}')

if issues:
    print('\n'.join(issues))
else:
    print('Plan audit passed.')
