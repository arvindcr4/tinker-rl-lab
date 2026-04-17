# Autoresearch: Session 7 — 3B GRPO Rescue via Parallel-Agent Loop

> Inspired by Ryan Li's Paradigm Optimization Arena winner (1,039 strategies,
> bitter-lesson). Loop = propose-hypothesis → run → score → keep/discard →
> feed learnings → loop. Started 2026-04-11.

## Objective
Resolve the single largest open question in the GRPO paper: **is the 3B model's
GSM8K failure a hard capacity wall, or can it be rescued by a combination of
group_size / temperature / reward-shape / curriculum / advantage-normalization
changes?** Phase 1 explores on the fast Qwen3-0.6B / 1.7B proxy. Phase 2
verifies winners on real 3B models (Qwen3-3B, Llama-3B) across 5 seeds. Phase 3
triggers a from-scratch reset agent when the loop plateaus for 3 waves.

If rescue succeeds → the paper's capacity-threshold claim softens, a new
recipe is added to Section 7. If rescue fails → the claim is strengthened
by 1000+ experiments of evidence.

## Metrics
- **Primary**: `last10_avg_accuracy` — mean binary reward on GSM8K over last 10
  training steps (emitted by `research_loop/train.py` as `METRIC`).
- **Secondary**: `peak_accuracy`, `first5_avg_accuracy`, `zero_reward_pct`,
  `zero_loss_pct`, `wall_clock_seconds`.

Phase 2 switches primary to real held-out GSM8K accuracy via
`reports/final/evaluate_gsm8k_test.py`, 5-seed median.

## How to Run
Autoresearch loop has moved to the new `research_loop/` framework:

- `python research_loop/coordinator.py status` — dashboard
- `python research_loop/coordinator.py wave new --size 8 --phase 1` — create wave
- `python research_loop/run_one.py --config <variant.yaml>` — single experiment
- `python research_loop/coordinator.py wave ingest wave_NNN` — close wave

Each wave = N parallel Claude Code agents, each with one hypothesis. Results
append to `research_loop/results.jsonl` AND to this repo's `autoresearch.jsonl`
(history). See `research_loop/README.md` for the full plan, budget, and knobs.

## Prior Session — Paper Improvement (Runs 35-44, superseded)
Earlier session's objective was reviewer-issue minimization on the GRPO paper
via `paper_improvement_audit.py`. Reached 0 issues by Run 44; remaining
improvements required compute. This session is that compute.

## Off Limits (unchanged)

## Files in Scope
- `reports/final/grpo_agentic_llm_paper.tex` — Main conference paper
- `reports/final/supplementary_appendix.tex` — Supplementary material
- `reports/final/references.bib` — Bibliography
- `reports/final/README.md` — Submission README with caveats
- `reports/final/SUBMISSION_CHECKLIST.md` — Submission checklist
- `paper_improvement_audit.py` — 30-check reviewer issue audit
- `paper_plan_audit.py` — Same audit (alternative name)
- `submission_claim_audit.py` — Claim consistency audit for ancillary docs
- `run_all_audits.py` — Unified suite for all paper/submission hygiene audits

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
