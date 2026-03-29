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
1. [ ] **Held-out GSM8K evaluation:** Run full test-set evaluation and update Section 4.3.3
2. [ ] **Standardized claim audit:** Remove or narrow any claim not backed by held-out or standard evaluation
3. [ ] **Tool-calling protocol disclosure:** Document schemas, prompts, rubric, and judge details; add standardized benchmark if possible
4. [ ] **Code-generation standardization:** Re-run full HumanEval/MBPP harness with pass@k and confidence intervals
5. [ ] **Methods/evaluation tables:** Add splits, budgets, decoding settings, seeds, and verifier/judge details
6. [ ] **Anonymous submission:** Create anonymized version of LaTeX paper, use the anonymized paper source/package for blind-review bundles, and export with `python reports/final/prepare_blind_review_package.py --force` so audits run before packaging
7. [ ] **Review page count:** Ensure within 9-page limit
8. [ ] **Figure captions:** Add clear, informative captions
9. [ ] **Supplementary appendix:** Add any additional experiments

### Medium Priority
1. [ ] **Response to reviewers:** Prepare point-by-point responses
2. [ ] **Supplementary video:** Optional 5-minute video
3. [ ] **Code release:** Clean repository for anonymized code submission
4. [ ] **Run full audit suite:** `python run_all_audits.py` before creating the final submission package (now includes workflow + packaging checks)

### Submission Venues
1. **NeurIPS 2026** - Deadline typically in May
2. **ICML 2026** - Deadline typically in January
3. **EMNLP 2026** - Deadline typically in July
4. **COLM 2026** - Deadline typically in February

## Key Results to Highlight (Preliminary / Custom results must be labeled)

| Finding | Result | Evidence | Caveat |
|---------|--------|----------|--------|
| Tool calling | 0%→92% JSON validity | Qwen2.5-1.5B | custom internal tool-calling setup |
| Multi-turn | 0.72→0.91 quality | Qwen2.5-3B | custom internal / judge-derived scenario score |
| Code gen | 32%→40% HumanEval | Qwen3-8B | preliminary 50-problem subset, not full canonical harness |
| GSM8K training-set reward | 30.0% ± 2.5% (3 seeds) | Qwen3-8B | not held-out test accuracy |
| Capacity threshold | 3B fails, 8B succeeds | Multi-model | currently setup-specific; exploration confounds remain |
| Synthetic gap | 3-8x easier than real | xlam-60k | depends on custom schema distribution |

## Notes for Authors

- Paper should focus on empirical findings, not theoretical contributions
- Emphasize reproducibility carefully (Tinker SDK, released scripts/configs, and explicit checkpoint availability status)
- Highlight novel findings (capacity threshold, MoE volatility, two-phase learning)
- Acknowledge limitations honestly, especially custom internal tool scores, the 50-problem HumanEval subset, and training-set math metrics vs held-out evaluation
