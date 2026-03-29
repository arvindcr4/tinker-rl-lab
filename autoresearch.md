# Autoresearch: Paper Improvement — Address Discovery Report Issues

## Objective
Systematically address all reviewer issues from the Discovery Report on the GRPO paper. Improve the paper's scientific rigor, presentation clarity, related work positioning, and evaluation transparency without inventing new experimental results.

## Metrics
- **Primary**: reviewer_issues (count, lower is better) — from paper_improvement_audit.py (30 checks)
- **Secondary**: claim_issues (from submission_claim_audit.py), total_checks, resolved

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `reports/final/grpo_agentic_llm_paper.tex` — Main conference paper
- `reports/final/supplementary_appendix.tex` — Supplementary material
- `reports/final/references.bib` — Bibliography
- `reports/final/README.md` — Submission README with caveats
- `reports/final/SUBMISSION_CHECKLIST.md` — Submission checklist
- `paper_improvement_audit.py` — 30-check reviewer issue audit
- `paper_plan_audit.py` — Same audit (alternative name)
- `submission_claim_audit.py` — Claim consistency audit for ancillary docs

## Off Limits
- `experiments/` — existing experiment data (read-only for analysis)
- `atropos/` �� core project code
- No inventing new experimental results

## Constraints
- All improvements must be text/analysis changes — no fabricated results
- Existing telemetry (arithmetic_metrics.jsonl) can be extracted and tabulated
- GSM8K logs can be parsed for zero-loss/reward statistics

## What's Been Tried
### Run 35 (KEEP, 7→0 issues): Major paper overhaul
- Expanded Related Work with 5 paragraphs: RLOO/REINFORCE++, S-GRPO/StepGRPO, DPO/Step-DPO, ToolRM/FC-RewardBench, curriculum learning, QR-Adaptor/LoTA-QAF
- Added Datasets and Splits table with per-domain sizes and eval protocols
- Added Decoding and Evaluation Protocol subsection
- Added group-composition analysis to capacity-threshold section
- Added Section 7: Planned Experiments and Standardized Evaluation Roadmap (4 phases)
- Added 15 new bibliography entries
- Created automated 30-check reviewer audit

### Run 36 (KEEP): Appendix diagnostics
- Added quantitative KL/entropy/group-composition diagnostics table from arithmetic_metrics.jsonl
- Entropy collapse 0.773→0.0002, KL near zero, group saturation documented

### Run 37 (KEEP): Reward limitations + safety
- Added Reward Function Limitations subsection (coarseness, hacking risk)
- Added Safety Considerations subsection (incorrect execution, loops, hallucination)

### Run 38 (KEEP): Comprehensive zero-loss table
- Added comprehensive zero-loss dynamics table across all GSM8K runs to appendix
- Key finding: 4B zero-loss is productive (all-correct saturation), 3B zero-loss is stalled (all-incorrect)
- LoRA rank doesn't affect zero-loss rates, supporting paper's rank-independence finding

### Key Insights
- The paper already had many caveats from previous sessions; this session deepened them
- The arithmetic task telemetry is the richest data source — GSM8K logs lack entropy/KL
- The 4B model's 68% zero-loss from ALL-CORRECT groups is an interesting counter-finding
- All improvements that can be made without new GPU runs are now done
- Remaining improvements require new experiments (held-out eval, ablations, baselines)
