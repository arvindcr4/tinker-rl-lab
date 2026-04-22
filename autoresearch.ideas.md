# Autoresearch Ideas - TinkerRL Submission

## Deferred Optimizations (high-value, lower priority)

### High-Value Experiments (for future work, NOT this submission)
- [ ] Recover or regenerate the trained math adapters, then run full held-out GSM8K evaluation with fixed decoding and bootstrap confidence intervals; this is the single highest-value experiment for turning the math story from training dynamics into a generalization claim.
- [ ] Add standardized tool-calling evaluation via FC-RewardBench / ToolRM-style single-turn judging and proxy-state/final-state multi-turn evaluation with judge reliability checks.
- [ ] Add full HumanEval/MBPP canonical harness runs with pass@1/pass@k and bootstrap confidence intervals, keeping the current 50-problem subset only as a pilot appendix result.
- [ ] Add 3B rescue experiments that combine larger GRPO group sizes (32-128), higher sampling temperature, and easy-to-hard curriculum to test whether the current failure is exploration/reward-sparsity rather than pure capacity.
- [ ] Add lightweight KL-to-SFT and entropy-regularized GRPO ablations with telemetry dashboards for KL, entropy, zero-advantage rate, and group composition (all-bad/mixed/all-good).
- [ ] Add MoE-specific routing diagnostics (router entropy, expert load variance, router shift ratio, clip fraction) to turn the current volatility observation into a quantitative stability finding.
- [ ] Add matched baseline comparisons against SFT-only, DPO/Step-DPO, RLOO/REINFORCE++, and step-wise GRPO variants to support stronger efficiency and algorithm-selection claims.

## Completed Improvements (this session)

### LaTeX Quality
- [x] Fixed citation `zvpEGAS2025` → `rlzvp2025` in main.tex and main_anon.tex
- [x] Reduced overfull warnings from 4 to 1 (last one is in author section, hard to fix)
- [x] Fixed table widths in tab:ppo_grpo, tab:task_grpo, tab:dense_moe

### Submission Package
- [x] Updated anonymous paper (main_anon.pdf)
- [x] Updated checksums in submission/contents/
- [x] Added VERSION.json with bundle metadata
- [x] All verification infrastructure in place (12 checks passing)

### Claims Documentation
- [x] Added "Claims We Do Not Make" section to paper and reports
- [x] Created REVIEWER_VERIFICATION.md (claim-centric verification)
- [x] Created EVAL_PROTOCOL.md (dataset splits, reward parsers, claim status)
- [x] Created SOURCE_PRECEDENCE.md (Qwen PPO discrepancy explanation)
- [x] Created FIGURE_PROVENANCE.md (figure generation scripts)
- [x] Created scripts/validate_master_results_schema.py

## Ideas Pruned (not feasible for this submission)
- Scaling law claims: explicitly disclaimed per reviewer feedback
- Algorithm leaderboard claims: explicitly disclaimed per no-go list
- Faithful GRPO implementation: explicitly disclaimed per no-go list

## Current Score: 94/100
- LaTeX: 18/20 (1 minor overfull warning in author section)
- Pages: 15/15 (60 pages)
- Figures/Tables: 15/15 (19 figs, 21 tables)
- Bibliography: 6/10 (26 citations, 188 entries)
- Experiments: 10/15 (95 results)
- Code quality: 7/10 (75 py files, 107 docstrings)
- Figure files: 10/10 (9/9 present)
- Verification: 10/10 (12/12 checks)
- Claims docs: 5/5