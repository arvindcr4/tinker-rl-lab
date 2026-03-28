# GRPO for Agentic LLM Fine-Tuning — Submission Instructions

## Paper Information
- **Title:** GRPO for Agentic LLM Fine-Tuning: Empirical Studies on Tool Use, Code Generation, and Math Reasoning
- **Format:** NeurIPS/ICML LaTeX (9-page limit + references)
- **Files:** Main paper + optional Supplementary Appendix

---

## Submission Files

### Required Files
| File | Description |
|------|-------------|
| `grpo_agentic_llm_paper.tex` | Main LaTeX paper (NeurIPS format) |
| `references.bib` | Bibliography |
| `nips_style.sty` | NeurIPS style file |
| `supplementary_appendix.tex` | Supplementary materials (optional) |

### Optional Anonymized Version
| File | Description |
|------|-------------|
| `grpo_agentic_llm_paper_anonymous.tex` | Anonymized for blind review |

---

## Compilation Instructions

### Compile Main Paper Only
```bash
pdflatex grpo_agentic_llm_paper.tex
bibtex grpo_agentic_llm_paper
pdflatex grpo_agentic_llm_paper.tex
pdflatex grpo_agentic_llm_paper.tex
```

### Compile with Supplementary
```bash
pdflatex grpo_agentic_llm_paper.tex
bibtex grpo_agentic_llm_paper
pdflatex grpo_agentic_llm_paper.tex
pdflatex grpo_agentic_llm_paper.tex
```

Note: The supplementary_appendix.tex is input automatically by the main paper before `\end{document}`.

### Dependencies
- Standard LaTeX distribution (TeX Live 2020+)
- Packages: times, amsmath, amssymb, booktabs, graphicx, hyperref, caption, subcaption, algorithm, algpseudocode, microtype, siunitx

---

## Page Count Estimate

| Section | Est. Pages |
|---------|-----------|
| Abstract | 0.25 |
| Introduction | 1.0 |
| Background | 0.75 |
| Experimental Setup | 1.0 |
| Experiments | 1.5 |
| Analysis | 1.0 |
| Related Work | 0.5 |
| Conclusion | 0.5 |
| Acknowledgments | 0.25 |
| **Main Paper Total** | **~6.75 pages** |
| Supplementary Appendix | ~5 pages |
| References | ~0.5 pages |

**Status:** Main paper is well within the 9-page limit. Supplementary adds ~5 pages of additional experimental details.

---

## Submission Checklist

### Content Requirements
- [x] 9-page limit satisfied (main paper ~6.75 pages)
- [x] Abstract with key results
- [x] Introduction with contributions clearly stated
- [x] Background on GRPO algorithm
- [x] Experimental setup with hyperparameters
- [x] Results on three task domains
- [x] Analysis of findings
- [x] Related work discussion
- [x] Conclusion with limitations
- [x] References in correct format

### Supplementary Materials (if included)
- [x] Complete run registry (17 Tinker runs)
- [x] Training hyperparameters tables
- [x] Failure case analysis
- [x] Additional experimental details

### Formatting
- [x] NeurIPS/ICML LaTeX style
- [x] Times New Roman font
- [x] Proper section numbering
- [x] Table captions above tables
- [x] Figure captions below figures

### Blind Review (if anonymized)
- [ ] Author names removed from `grpo_agentic_llm_paper_anonymous.tex`
- [ ] No identifying information in acknowledgments
- [ ] Git history does not reveal authors (create fresh clone)

---

## Key Results Summary

| Finding | Result | Model |
|---------|--------|-------|
| Tool calling (JSON validity) | 0\% → 92\% | Qwen2.5-1.5B |
| Multi-turn tool quality | 0.72 → 0.91 | Qwen2.5-3B |
| HumanEval pass@1 | 32\% → 40\% | Qwen3-8B |
| GSM8K (multi-seed) | 30.0\% ± 2.5\% | Qwen3-8B |
| Capacity threshold | 3B fails, 8B succeeds | Multiple |
| Synthetic vs real gap | 3--8x difficulty difference | xlam-60k |

---

## Supplementary Appendix Contents

1. **Complete Training Run Registry** (Tables S1--S3)
   - 7 tool-use Tinker runs with Run IDs
   - 7 GSM8K Tinker runs with seeds/ranks
   - 5 Colab QLoRA experiments

2. **Training Hyperparameters** (Tables S4--S5)
   - Tinker SDK configuration
   - LoRA adapter targets and ranks

3. **Failure Case Analysis** (Section S3)
   - 3B parameter failure mode (56\% zero-loss)
   - MoE volatility analysis (2.43× higher variance)
   - MBPP evaluation failure (parsing bug)

4. **Additional Experimental Details** (Section S4)
   - Reward function weighting rationale
   - Evaluation protocol limitations
   - Multi-turn reward breakdown

5. **Qualitative Examples** (Section S5)
   - Successful/failed tool calls
   - Two-phase learning trajectory

6. **Platform and Compute Costs** (Table S6)
   - Tinker + Colab usage
   - Estimated total cost ~\$50/person

7. **Code and Reproducibility** (Section S7)
   - Script locations
   - Model checkpoint URLs

---

## Contact
For questions about this submission, contact the authors through the PES University MTech CSE program.
