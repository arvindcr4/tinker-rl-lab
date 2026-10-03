---
title: RL citation verification report
id: final_report_rl-verification
tags:
- rl-verification
created: '2026-10-03T15:06:00Z'
updated: '2026-10-03T15:13:27.469541Z'
status: evergreen
type: note
tier: institutional
content_type: review
deprecated: false
summary: 'Final report: full-text verification verdicts for the DAPO, Dr. GRPO, REINFORCE++,
  and VAPO claims cited by grpo.py comments and flags.'
---

# RL citation verification report (2026-10-03)

Every paper cited by `grpo.py` comments/flags was fetched in full text and
checked claim by claim. Sources: [[dapo-an-open-source-llm-reinforcement-learning]],
[[published-as-a-conference-paper-at-colm-2025]],
[[reinforce-stabilizing-critic-free]],
[[vapo-efficient-and-reliable-reinforcement-learning-for]].
Analysis extracts: [[dapo-mechanisms-code-mapping]], [[drgrpo-bias-code-mapping]].

| Claim | Verdict | Location |
|---|---|---|
| DAPO decoupled clip 0.2/0.28 | CONFIRMED | §3.1 mechanism, §4.1 values |
| DAPO dynamic sampling 0<#correct<G | CONFIRMED | §3.2, Eq. 11 |
| DAPO overlong filtering (mask, not wrong) | CONFIRMED | §3.4 |
| Dr. GRPO unbiased advantage R−mean(R) | CONFIRMED | §3.2, Fig. 1 |
| Dr. GRPO length + difficulty biases | CONFIRMED | §3.1 |
| Dr. GRPO venue COLM 2025 | CONFIRMED | paper header |
| R++ global advantage normalization | CONFIRMED | Abstract, §3.1 Eq. 5 |
| R++ author Hu, year 2025 | CONFIRMED | arXiv 2501.03262 |
| VAPO NLL aux on correct (L=L_PPO+μ·L_NLL) | CONFIRMED | §4.3, Eq. 9–10 |
| VAPO μ=0.01 | DENIED — paper uses μ=0.1 | §5.1 item 6 |
| VAPO Yue et al., 2025 | CONFIRMED | arXiv 2504.05118 |

Resulting code change: `nll_aux_coef` comment cites VAPO §4.3 and the
paper's μ=0.1 (repo default stays 0.0/opt-in). No other comment needed
correction. Claim 10's denial is recorded here so the 0.01 figure is not
re-propagated.
