# Reports - Final Submission

This directory contains the final capstone report and conference paper for the GRPO Agentic LLM Fine-Tuning project.

## ⚠️ CRITICAL: Held-Out Evaluation Required

**The paper has been updated to honestly acknowledge its evaluation scope limitation.** The primary metrics measure training-set reward optimization, NOT held-out test-set generalization. This is documented prominently in the paper.

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

| Task | Before | After | 
|------|--------|-------|
| JSON Tool Calls | 0% | 92% |
| Multi-turn Quality | 0.72 | 0.91 |
| HumanEval Pass@1 | 32% | 40% |
| GSM8K Train Reward | - | 30.0% ± 2.5% |

## Paper Status

✅ **Completed**: Tool calling results (validated)  
✅ **Completed**: Code generation results (HumanEval verified)  
⚠️ **Pending**: GSM8K held-out test evaluation  
✅ **Completed**: Honest limitation disclosure

## The One Change to Get an A

Run `evaluate_gsm8k_test.py` on your trained checkpoint and update Section 4.3.3 with the results. If accuracy >50%, you have a strong generalization claim. If <40%, the paper correctly positions itself as a training dynamics study.

## Authors

Arvind C R, Sandhya Jeyaraj, Arumugam Chetty K, Madhu Kumara L, Dhruva N Murthy, Mohammad Rafi  
Group 6, MTech DSAI, PES University
