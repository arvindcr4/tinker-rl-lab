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

For blind review submissions, submit the anonymized paper source/package instead of the author-identified paper source.

---

## Compilation Instructions

### Compile Main Paper Only
```bash
pdflatex grpo_agentic_llm_paper.tex
bibtex grpo_agentic_llm_paper
pdflatex grpo_agentic_llm_paper.tex
pdflatex grpo_agentic_llm_paper.tex
```

### Compile Supplementary Appendix Separately
```bash
pdflatex supplementary_appendix.tex
pdflatex supplementary_appendix.tex
```

Note: `supplementary_appendix.tex` is a standalone document and should be compiled separately from the main paper.

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

**Status:** Estimated main-paper length is ~6.75 pages before final recompilation, but page count should be rechecked from the generated PDF before submission. Supplementary adds ~5 pages of additional experimental details.

---

## Submission Checklist

### Content Requirements
- [ ] 9-page limit re-verified from final compiled PDF (current estimate: ~6.75 pages)
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
- [ ] Do not include generated build artifacts (`.aux`, `.log`, `.out`, compiled PDFs) in the submission package
- [ ] Exclude the non-anonymous paper source from blind-review bundles
- [ ] Run `python run_all_audits.py` before exporting the final blind-review package
- [ ] Generate the export bundle with `python reports/final/prepare_blind_review_package.py --force` (this runs `python run_all_audits.py` first unless `--skip-audits` is explicitly used)

---

## Key Results Summary (Preliminary / Custom results must be labeled)

| Finding | Result | Model | Caveat |
|---------|--------|-------|--------|
| Tool calling (JSON validity) | 0\% → 92\% | Qwen2.5-1.5B | custom internal tool-calling setup |
| Multi-turn tool quality | 0.72 → 0.91 | Qwen2.5-3B | custom judge-derived internal scenario score |
| HumanEval pass@1 | 32\% → 40\% | Qwen3-8B | preliminary 50-problem subset, not full canonical harness |
| GSM8K training-set reward | 30.0\% ± 2.5\% | Qwen3-8B | training-set reward, not held-out test accuracy |
| Capacity threshold | 3B fails, 8B succeeds | Multiple | setup-specific until exploration confounds are ablated |
| Synthetic vs real gap | 3--8x difficulty difference | xlam-60k | depends on custom schema distribution |

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
   - Explicit checkpoint availability status and recovery instructions

---

### Evaluation Caveat
- Tool results are currently based on custom internal evaluations and should not be presented as standardized benchmark outcomes.
- HumanEval currently reflects a 50-problem subset rather than the full canonical harness.
- GSM8K headline GRPO numbers are training-set reward metrics; full held-out evaluation remains the highest-priority missing experiment.

## Contact
For blind-review submissions, remove this section or replace it with the venue's anonymized contact mechanism.
