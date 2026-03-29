# Reports - Final Submission

This directory contains the final capstone report and conference paper for the GRPO Agentic LLM Fine-Tuning project.

## ⚠️ CRITICAL: Standardized Evaluation Still Required

**The paper now acknowledges its evaluation-scope limitations, but key reviewer-facing gaps still need to be closed.** The strongest current evidence is in training-set reward optimization and training dynamics, not yet in fully standardized held-out generalization across math, tool calling, and code generation.

See `PAPER_IMPROVEMENT_PLAN.md` for the concrete remediation roadmap.

### To Complete the Paper (A-grade path):

Run the held-out GSM8K test evaluation:

```bash
# With Tinker (if checkpoint still available)
TINKER_API_KEY=your_key python evaluate_gsm8k_test.py \
    --use_tinker \
    --run_id 5db4e965 \
    --output gsm8k_test_results.json

# With local model (requires GPU)
python evaluate_gsm8k_test.py \
    --model_name Qwen/Qwen3-8B \
    --output gsm8k_test_results.json
```

If results show >40% accuracy on held-out test, update Section 4.3.3 with actual numbers. This will transform the paper from a training dynamics study to a true generalization claim.

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
- `evaluate_gsm8k_test.py` - **CRITICAL**: Script to run held-out GSM8K evaluation
- `supplementary_appendix.tex` - Additional experimental details

## Key Results (Training-Set)

These numbers should not all be read as standardized benchmark claims:
- **Tool results are internal/custom** and still need standardized evaluator disclosure or replacement.
- **HumanEval is currently a 50-problem subset result**, not yet the canonical full-harness benchmark.
- **Math is the strongest current evidence**, but the main reported GRPO math numbers are still training-set reward metrics until full held-out evaluation is completed.

| Task | Before | After | Scope note |
|------|--------|-------|------------|
| JSON Tool Calls | 0% | 92% | custom internal tool-calling setup |
| Multi-turn Quality | 0.72 | 0.91 | custom judge-derived internal scenario score |
| HumanEval Pass@1 | 32% | 40% | preliminary 50-problem subset |
| GSM8K Train Reward | - | 30.0% ± 2.5% | training-set reward, not held-out test accuracy |

## Paper Status

✅ **Completed**: Honest limitation disclosure  
⚠️ **Pending**: Full held-out GSM8K evaluation  
⚠️ **Pending**: Standardized tool-calling evaluation / judge protocol disclosure  
⚠️ **Pending**: Canonical full HumanEval/MBPP evaluation  
⚠️ **Pending**: Reproducibility packaging for prompts, schemas, and checkpoints

## Highest-Leverage Next Step

Run `evaluate_gsm8k_test.py` on the trained checkpoint and update Section 4.3.3 with full held-out results. That is the single highest-leverage improvement, but not the only one: the paper also needs standardized tool/code evaluation or narrower claim boundaries. See `PAPER_IMPROVEMENT_PLAN.md` for the full sequencing.

## Authors

Arvind C R, Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L, Dhruva N Murthy, Mohammad Rafi  
Group 6, MTech DSAI, PES University
