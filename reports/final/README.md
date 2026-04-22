# Reports - Final Submission

This directory contains the final capstone report and conference paper material for the GRPO reward-diversity study.

## Current Status

The final capstone report is framed as an exploratory systems study, not as a broad proof that GRPO improves reasoning. The strongest positive evidence is structured tool-call emission after SFT warm-up. The strongest negative check is the held-out GSM8K result: Qwen3-8B base scores 82.0% on the matched 200-example slice, while GRPO checkpoints average 83.3% (SD 2.2, p=0.26), so the lift is small and not statistically significant.

See `result_ledger.md` for the source-of-truth result table and `ARTIFACT_SANITIZATION.md` for credential-exclusion rules.

## Files

### Improvement Plan
- `PAPER_IMPROVEMENT_PLAN.md` - prioritized plan to address reviewer concerns, standardize evaluation, and strengthen the paper

### Capstone Report
- `capstone_final_report.md` - Full capstone report (honest about limitations)
- `capstone_final_report.docx` - Word version

### Conference Paper (NeurIPS/ICML Format)
- `grpo_agentic_llm_paper.tex` - LaTeX source
- `grpo_agentic_llm_paper.md` - Markdown version
- `grpo_agentic_llm_paper_anonymous.tex` - Anonymized for blind review
- `references.bib` - Bibliography
- `nips_style.sty` - NeurIPS/ICML style

### Evaluation
- `evaluate_gsm8k_test.py` - Held-out GSM8K evaluation script (Tinker API or local HF)
- `run_heldout_parallel.sh` - Run 5-seed parallel evaluation
- `supplementary_appendix.tex` - Additional experimental details

### Reproducibility Notebooks
- `../../submission_colab.ipynb` - Standard GRPO training + evaluation Colab
- `../../advanced_rl_colab.ipynb` - Dr. GRPO, DAPO, DPO Colab (advanced algorithms)

## Key Results

These numbers should not all be read as standardized benchmark claims:
- **Tool results are internal/custom** and still need standardized evaluator disclosure or replacement.
- **HumanEval is currently a 50-problem subset result**, not yet the canonical full-harness benchmark.
- **Math has a completed held-out check**, but it is negative for a strong GRPO-improvement claim.

| Task | Before | After | Scope note |
|------|--------|-------|------------|
| JSON Tool Calls | 0% | 92% | custom internal tool-calling setup |
| Multi-turn Quality | 0.72 | 0.91 | custom judge-derived internal scenario score |
| HumanEval Pass@1 | 32% | 40% | preliminary 50-problem subset |
| GSM8K Train Reward | - | 30.0% ± 2.5% | training-set reward, not held-out test accuracy |
| GSM8K Held-Out Accuracy | 82.0% base | 83.3% GRPO | +1.3pp, p=0.26; not significant |

## Paper Status

✅ **Completed**: Honest limitation disclosure  
✅ **Completed**: W&B logging (17 Tinker runs uploaded to tinker-rl-scaling project)  
✅ **Completed**: Advanced RL notebook (Dr. GRPO, DAPO, DPO) -- `advanced_rl_colab.ipynb`  
✅ **Completed**: Submission Colab -- `submission_colab.ipynb`  
✅ **Completed**: All 13 audits passing (0 issues)  
✅ **Completed**: Held-out GSM8K evaluation (5 seeds x 200 examples; non-significant lift)  
⚠️ **Pending**: Standardized tool-calling evaluation / judge protocol disclosure  
⚠️ **Pending**: Canonical full HumanEval/MBPP evaluation  
⚠️ **Pending**: Fully local reproduction path without hosted-service permissions

## Highest-Leverage Next Step

Add a minimal local reproduction path that does not require private Tinker, W&B, or checkpoint permissions. The best next experiment is a matched baseline table with Best-of-N, rejection SFT, DAPO/Dr.GRPO-style variants, and RLOO/REINFORCE++ on the same small task slices.

## Audit Suite

Before submission or export, run:

```bash
python run_all_audits.py
```

This verifies the paper, capstone, submission docs, anonymization hygiene, held-out-evaluation readiness, claim strength, packaging checks, and submission-workflow checks in one pass.

For blind-review export, you can generate a clean bundle with:

```bash
python reports/final/prepare_blind_review_package.py --force
```

By default, the export script runs `python run_all_audits.py` first and refuses to package files if the audit suite is failing.

## Authors

Arvind C R, Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L, Dhruva N Murthy, Mohammad Rafi  
Group 6, MTech DSAI, PES University
