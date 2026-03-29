# Autoresearch Dashboard: Paper Improvement (Discovery Report)

**Segment 8 | Runs:** 34-43 | **Kept:** 10 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** reviewer_issues: 7 (#34)
**Best:** reviewer_issues: 0 (#35+, all audits passing)

| # | commit | reviewer_issues | status | description |
|---|--------|----------------|--------|-------------|
| 34 | bc76690 | 7 | keep | baseline: 7 unresolved issues (10 already fixed) |
| 35 | ac62de1 | 0 (-100%) | keep | all 17 issues: related work, datasets, decoding, future plan |
| 36 | 3e596f0 | 0 | keep | KL/entropy/group-composition diagnostics table |
| 37 | e0f343b | 0 | keep | reward limitations and safety discussion |
| 38 | 84d91fd | 0 | keep | expanded to 30 checks + claim/sync audits |
| 39 | ee9b3e1 | 0 | keep | scope-qualified conclusion rewrite |
| 40 | 7368f31 | 0 | keep | 4B model data: capacity threshold narrowed to 3B-4B |
| 41 | 52f409f | 0 | keep | 5-seed multi-seed table, abstract caveats synced |
| 42 | bf00a22 | 0 | keep | answered 4 reviewer questions (SFT, subset, penalty, code) |
| 43 | a145bd0 | 0 | keep | per-run compute budget table |

## Paper Changes Summary

### Main Paper (grpo_agentic_llm_paper.tex)
- **Abstract:** Custom eval + HumanEval subset caveats, synced across 3 versions
- **Section 3:** Datasets/Splits table, Decoding Protocol subsection
- **Section 4:** SFT 0% explanation, penalty mechanism details, subset rationale
- **Section 5:** Group-composition analysis, 4B model data (3B→4B threshold)
- **Section 5:** Reward Limitations + Safety Considerations subsections
- **Section 6:** Expanded Related Work (5 paragraphs, 15 new citations)
- **Section 7:** NEW — Planned Experiments Roadmap (4 phases)
- **Section 8:** Scope-qualified conclusion with quantitative details
- **Acknowledgments:** Reproducibility paragraph
- **Multi-seed:** 3→5 seeds (30.5% SD=3.3%)
- **Bibliography:** 9→24 entries

### Appendix (supplementary_appendix.tex)
- KL/entropy/group-composition diagnostics table (arithmetic logs)
- Comprehensive zero-loss dynamics table (all GSM8K runs: 3B, 4B, 8B)
- Per-run compute budget table (tokens, wall-clock, GPU-hours)

### All Audits Passing
- paper_plan_audit.py: 30/30 checks
- submission_claim_audit.py: 0 issues
- Abstract consistency: 3/3 versions synced
