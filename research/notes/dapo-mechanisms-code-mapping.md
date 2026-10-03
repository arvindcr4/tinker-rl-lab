---
title: 'DAPO mechanisms mapped to grpo.py shaping flags'
id: dapo-mechanisms-code-mapping
tags:
- rl-verification
- extract
created: '2026-10-03T15:05:00Z'
status: review
tier: institutional
type: note
content_type: review
parent: dapo-an-open-source-llm-reinforcement-learning
summary: 'Extract: DAPO §3.1/§3.2/§3.4 mechanisms and exactly where each lands in grpo.py (clip plumbing, degenerate-group filter, truncation mask).'
---

# DAPO mechanisms → code mapping

Parent: [[dapo-an-open-source-llm-reinforcement-learning]]

DAPO (Yu et al., arXiv 2503.14476v2) contributes three mechanisms this repo
implements, each verified against the paper full text on 2026-10-03:

1. **Clip-Higher (§3.1, values §4.1).** `clip(r, 1−ε_low, 1+ε_high)` with
   ε_low=0.2, ε_high=0.28. Lands as `epsilon_low`/`epsilon_high` in
   `CanonicalSpec`, `TRLAlgorithmConfig` (legacy `epsilon` kwarg maps to
   both), and the generated TRL `GRPOConfig` text. The Tinker on-policy
   loop has no clip ratio, so nothing lands there.
2. **Dynamic sampling (§3.2, Eq. 11).** Constraint
   `0 < |{correct}| < G` filters prompts with accuracy 0 or 1.
   Lands as `is_degenerate_group()` plus the bounded refill loop in the
   step loop (`dynamic_sampling`, cap `dynamic_sampling_max_resamples`),
   with skips logged to W&B.
3. **Overlong shaping (§3.4).** Overlong Filtering masks the loss of
   truncated samples instead of scoring them wrong. Lands as
   `apply_truncation_mask()` (`mask_truncated_responses`); the soft
   length penalty (Eq. 13) is deliberately not implemented.

Complementary de-bias (unbiased `R − mean(R)`) comes from
[[published-as-a-conference-paper-at-colm-2025]]; batch-global
normalization from [[reinforce-stabilizing-critic-free]]; the NLL
auxiliary from [[vapo-efficient-and-reliable-reinforcement-learning-for]].
