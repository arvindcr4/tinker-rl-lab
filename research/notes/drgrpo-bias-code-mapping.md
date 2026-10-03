---
title: Dr. GRPO bias analysis mapped to debias flag
id: drgrpo-bias-code-mapping
tags:
- rl-verification
- extract
created: '2026-10-03T15:05:00Z'
updated: '2026-10-03T15:13:27.329144Z'
status: evergreen
type: note
tier: institutional
content_type: review
parent: published-as-a-conference-paper-at-colm-2025
deprecated: false
summary: 'Extract: Dr. GRPO §3.1 length/difficulty biases and the unbiased advantage,
  mapped to normalize_rewards(unbiased=True) and contrasted with global normalization.'
---

# Dr. GRPO bias analysis → code mapping

Parent: [[published-as-a-conference-paper-at-colm-2025]]

Dr. GRPO (Liu et al., COLM 2025, arXiv 2503.20783) diagnoses two biases in
GRPO (§3.1, verified against full text 2026-10-03): response-level length
bias from dividing by `|o_i|`, and question-level difficulty bias from
dividing centered rewards by `std(R)`. The fix (§3.2, Fig. 1) removes both
normalizers: `Â = R − mean(R)`.

In this repo that lands as `normalize_rewards(..., unbiased=True)` wired
through `GRPOConfig.debias_advantages` (default off). Note the repo's loss
uses per-response token-sum-then-mean rather than GRPO's `1/|o_i|`
averaging, so only the `std(R)` half of the paper's fix applies here —
the flag is documented as the std-removal, not a full Dr. GRPO port.

Contrast with [[reinforce-stabilizing-critic-free]]: REINFORCE++ keeps
std normalization but pools μ/σ batch-globally
(`normalize_advantages_global`), addressing cross-group scale instead of
within-group bias. The two flags compose (`unbiased` applies at global
scope). Truncation handling that motivated part of the length-bias
discussion lives in [[dapo-mechanisms-code-mapping]].
