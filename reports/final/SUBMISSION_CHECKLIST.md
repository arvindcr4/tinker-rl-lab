# Capstone Report & Conference Paper Submission Checklist

## Final Report (capstone_final_report.md)
- [x] Abstract with key results
- [x] Introduction with contributions
- [x] Related Work section
- [x] Methodology (GRPO algorithm, infrastructure, models)
- [x] Experiments (tool calling, code generation, math reasoning)
- [x] Results and Analysis
- [x] Summary of 7 Findings
- [x] Limitations section
- [x] Experimental Details (run registry, budget)
- [x] Conclusion
- [x] References

## Conference Paper (NeurIPS/ICML Format)
- [x] LaTeX source (grpo_agentic_llm_paper.tex)
- [x] Bibliography (references.bib)
- [x] Style file (nips_style.sty)
- [x] Markdown version (grpo_agentic_llm_paper.md)

### NeurIPS/ICML Requirements Checklist
- [ ] **Page limit:** 9 pages content + references (currently ~6 pages)
- [ ] **Anonymous submission:** Remove author names for blind review
- [ ] **Supplementary materials:** Optional appendix with additional experiments
- [ ] **Code/匿名化:** Ensure code is anonymized if submitting supplementary

## Document Files
```
reports/final/
├── capstone_final_report.md      # Full capstone report (28KB)
├── capstone_final_report.docx     # Word version
├── grpo_agentic_llm_paper.tex    # NeurIPS LaTeX paper
├── grpo_agentic_llm_paper.md     # Markdown version
├── references.bib                # Bibliography
├── nips_style.sty                # NeurIPS style file
└── SUBMISSION_CHECKLIST.md       # This file
```

## Remaining Tasks for Final Submission

### High Priority
1. [ ] **Anonymous submission:** Create anonymized version of LaTeX paper
2. [ ] **Review page count:** Ensure within 9-page limit
3. [ ] **Figure captions:** Add clear, informative captions
4. [ ] **Supplementary appendix:** Add any additional experiments

### Medium Priority
1. [ ] **Response to reviewers:** Prepare point-by-point responses
2. [ ] **Supplementary video:** Optional 5-minute video
3. [ ] **Code release:** Clean repository for anonymized code submission

### Submission Venues
1. **NeurIPS 2026** - Deadline typically in May
2. **ICML 2026** - Deadline typically in January
3. **EMNLP 2026** - Deadline typically in July
4. **COLM 2026** - Deadline typically in February

## Key Results to Highlight

| Finding | Result | Evidence |
|---------|--------|----------|
| Tool calling | 0%→92% JSON validity | Qwen2.5-1.5B |
| Multi-turn | 0.72→0.91 quality | Qwen2.5-3B |
| Code gen | 32%→40% HumanEval | Qwen3-8B |
| GSM8K train reward | 30.0% ± 2.5% (3 seeds) | Qwen3-8B |
| Capacity threshold | 3B fails, 8B succeeds | Multi-model |
| Synthetic gap | 3-8x easier than real | xlam-60k |

## Notes for Authors

- Paper should focus on empirical findings, not theoretical contributions
- Emphasize reproducibility (Tinker SDK, checkpoints available)
- Highlight novel findings (capacity threshold, MoE volatility, two-phase learning)
- Acknowledge limitations honestly
