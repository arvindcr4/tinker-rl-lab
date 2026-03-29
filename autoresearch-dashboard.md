# Autoresearch Dashboard: Paper Improvement

**Runs:** 41 | **Kept:** 41 | **Discarded:** 0 | **Crashed:** 0
**Segment 8 Baseline:** reviewer_issues: 7 (#34)
**Segment 8 Best:** reviewer_issues: 0 (#35-41)

| # | commit | reviewer_issues | status | description |
|---|--------|----------------|--------|-------------|
| 34 | bc76690 | 7 | keep | baseline: 7 unresolved issues (10 already fixed) |
| 35 | ac62de1 | 0 (-100%) | keep | all 17 issues resolved: expanded related work, datasets/decoding, future plan |
| 36 | 3e596f0 | 0 (-100%) | keep | added KL/entropy/group-composition diagnostics table |
| 37 | e0f343b | 0 (-100%) | keep | added reward limitations and safety discussion, expanded to 25 checks |
| 38 | 84d91fd | 0 (-100%) | keep | all 30 checks pass + 0 claim issues |
| 39 | ee9b3e1 | 0 (-100%) | keep | rewrote conclusion with scope-qualified findings |
| 40 | 7368f31 | 0 (-100%) | keep | added 4B model data, narrowed capacity threshold to 3B-4B |
| 41 | 52f409f | 0 (-100%) | keep | expanded multi-seed to 5 seeds, synced abstracts with caveats |

## Changes Summary

### Paper (grpo_agentic_llm_paper.tex)
- **Related Work:** 5 new paragraphs, 15 new citations (RLOO, REINFORCE++, S-GRPO, DPO, ToolRM, FC-RewardBench, curriculum, QR-Adaptor, LoTA-QAF)
- **Datasets & Splits:** New table with per-domain sizes and eval protocols
- **Decoding Protocol:** New subsection documenting evaluation settings
- **Capacity Analysis:** Group-composition analysis, 4B model data, 3B-4B threshold
- **Multi-Seed:** 3→5 seeds (42, 137, 256, 512, 999), updated stats
- **Reward Limitations:** New subsection on coarseness and hacking risk
- **Safety:** New subsection on tool execution risks
- **Future Plan:** Section 7 with 4-phase experimental roadmap
- **Conclusion:** Scope-qualified findings with quantitative details
- **Abstract:** Custom eval and HumanEval subset caveats, synced across 3 versions

### Appendix (supplementary_appendix.tex)
- **Diagnostics Table:** KL/entropy/group-composition from arithmetic logs
- **Zero-Loss Table:** Comprehensive analysis across all GSM8K runs (3B, 4B, 8B)

### Bibliography (references.bib)
- 9→24 entries
