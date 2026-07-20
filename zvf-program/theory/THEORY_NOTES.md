# ZVF Program — Pillar 2 proof audit

**Status:** T1--T3 are proved in `zvf_theory.tex` under their explicit
assumptions. The paper remains a conditional theory note: it does not claim
that i.i.d. binary-reward assumptions hold universally, that one informative
group guarantees policy improvement, or that the T3 proxy is the uniquely
correct learning objective.

## Closed proof obligations

| Result | Closed obligation | Resolution |
|---|---|---|
| T1 | Choose the canonical estimator | ZVF is treated as a sample mean of Bernoulli group indicators, not advertised as a non-trivial U-statistic. |
| T1 | Finite-sample mean and asymptotic variance | Exact unbiasedness and the binomial-proportion CLT follow from the i.i.d. indicator model. |
| T1/T2 | Boundary-aware observed-rate guarantee | The controller corollary uses a one-sided Clopper--Pearson lower bound, giving at least nominal coverage under the stated model. |
| T2 | Mixture versus fixed-difficulty geometric law | A fresh prompt difficulty is drawn for each independent group, so marginal informative indicators are i.i.d. Bernoulli with rate `E[q(p)]`. |
| T2 | Improvement overclaim | The theorem is restricted to rollouts-to-informative-group/nonzero-reward-gradient; it makes no monotone reward-improvement claim. |
| T3 | Global `G>=4` inequality | A binomial pair-counting argument proves `P(mixed) <= G p(1-p)`, with equality only for `G in {2,3}` on interior `p`. |
| T3 | Differentiation under the integral | `p^G |log p| <= 1/(eG)` supplies domination locally in continuous `G`. |
| T3 | Existence, uniqueness, and rounding | The discrete optimum is proved directly; no continuous uniqueness or rounding claim is needed. |
| T3 | Beta-prior bookkeeping | The three Beta-function terms are expanded explicitly in the corollary. |
| Controller | Obsolete `G*(phi_hat)` feedback loop | Removed: the proved proxy optimum is prior-independent. The remaining hysteretic policy is explicitly empirical. |

## Assumptions and scope

| Assumption | Statement | Consequence if violated |
|---|---|---|
| A1 | Binary rewards from a deterministic verifier | Dense/noisy rewards require a different zero-signal definition. |
| A2 | Rollouts are i.i.d. within a prompt group | Correlated decoding changes `p^G + (1-p)^G`. |
| A3 | Prompt groups are i.i.d. draws from a fixed population | Curriculum/replay makes the estimand local and invalidates the geometric product law unless modeled. |
| A4 | The difficulty distribution has interior mass | At a fully trivial or saturated boundary the CLT variance is zero and T3's strict inequality has no interior mass to integrate. |

The checked-in E-T1 analysis already tests A3 stress cases: curriculum ordering
retains local-stage coverage but loses global-population coverage, while
stratified batches recover the global estimand. That is empirical support for a
reporting/design rule, not a proof for arbitrary dependent sampling.

## Research extensions, not missing proof steps

1. Derive and compare richer objectives based on expected squared GRPO update,
   Fisher information, wall-clock cost, and held-out gain. T3 proves only that
   the current proxy cannot justify adaptive group-size control.
2. Establish dependence-robust confidence sequences or block-bootstrap
   intervals for curriculum, replay, and asynchronous sampling.
3. Analyze the finite-action Pillar-7 controller under learning-induced
   distribution shift. Its benefit remains a prospective empirical claim.
4. Generalize ZVF to dense, multi-objective, or noisy-verifier rewards.

## Reference status

The manuscript uses the canonical shared bibliography. DeepSeekMath is the GRPO
source; Zhou et al. is cited only for neighboring policy-gradient U-statistic
analysis. The mixed-group probability `1-p^G-(1-p)^G` is an elementary
Bernoulli complement and needs no attribution. No equivalence between Zhou et
al.'s estimand and the contrast-availability statistic is claimed.
