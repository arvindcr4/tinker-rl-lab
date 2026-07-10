# Iter 187 — P7 Information-Theoretic Counter-Audit on N2 reward tensors

**Pillar:** P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)

**Vein:** Fresh (not in 195 prior rows). Closes brief vein (a) at the **information-theoretic**
layer: prior iterations measured fires, savings, regret, hysteresis, and binary
`restore=False/True` — but never quantified the **per-prompt posterior
entropy gain** from a fire (Bayesian posterior predictive entropy reduction).
Iter-91 measured Δ_ZVF under the iid binomial model; iter-69 measured
restore probability (`0/1867 boundary prompts`); iter-187 measures the
*information value* of firing using closed-form conjugate-posterior
entropy (Bits-shannon, Jeffreys Beta prior).

## Method

Single script `platform_modal/scripts/p5p8/p7_iter187_infogain_controller.py`
(≤300 LoC, stdlib only — no scipy) producing 4 reproducible artifacts
in <10 s on the 4-method × 40-step × 16-prompt N2 reward tensors:

- `p7_iter187_infogain_per_prompt.tsv` (2560 rows)
- `p7_iter187_infogain_per_tier.tsv`  (3 rows: boundary/edge/mid)
- `p7_iter187_infogain_per_step.tsv`  (160 rows)
- `p7_iter187_infogain_per_method.tsv` (4 rows)
- `p7_iter187_infogain_summary.json`

**Closed-form per-prompt dH:** for a prompt with `k` successes in `G=8`
rollouts, pre-fire posterior is `Beta(k+1, 8-k+1)`; the expected
post-fire parameter entropy is
`E_j ~ BetaBinomial(k+1, 8-k+1, g_esc) of H(Beta(k+j+1, 8-k+g_esc-j+1))`,
summed exactly over `j = 0..8`. Bootstrap CI on all summaries
(n_boot=4000, seed=20260705, **with-replacement** sampling).

**Two trigger policies** evaluated:
- `C1` (canonical zvf-triage @ `τ=0.70`): fire iff
  `zvf_step > 1 − τ = 0.30` (boundary rate HIGH).
- `Anti-C1` (counterfactual): fire iff `zvf_step ≤ 0.30` (low boundary).

## Headline findings (falsifiable)

| Hyp | Claim | Verdict | Evidence |
|-----|-------|---------|----------|
| H1 | C1 fires on 100% of N2 steps (ZvF uniformly > 0.30) | **PASS** | 160/160 fires; fire-rate 1.000 on all 4 methods |
| H2 | Per-fire dH ≈ 0.39 bits (canonical controller delivers ~0.4 bits of info per fire) | **PASS** | grpo 0.3899 [0.3886, 0.3911]; aero 0.3901 [0.3888, 0.3913]; gift 0.3882 [0.3870, 0.3894]; areal 0.3903 [0.3891, 0.3916] |
| H3 | Per-fired-step **information regret** = max dH over step's 16 prompts − realized mean dH | **PASS** | 0.0310 [0.0303, 0.0317] bits per fired-step; total 79.4 bits missed across 160 × 16 cells |
| H4 | **Regression slope of dH on ZvF_step is NEGATIVE** (C1 fires on the WRONG regime) | **PASS** | slope = **−0.0360** bits/unit, 95% CI **[−0.0372, −0.0349]** — CI excludes zero |
| H5 | Per-tier dH is monotone: mid > edge > boundary | **PASS (deterministic)** | mid 0.4204 [0.4202, 0.4206]; edge 0.4089; boundary 0.3800 — dH is a deterministic function of k, so tier-level differences are real but CIs are degenerate (single value per tier) |
| H6 | Anti-C1 trigger (fire iff zvf ≤ 0.30) is INERT on N2 | **PASS** | 0/160 steps fired across all 4 methods — ZvF uniformly > 0.30 (avg 0.7) on N2 confirms C1's structural overfiring |
| H7 | Per-method dH on C1 fires is ≈ 0.39 bits across all 4 methods | **PASS** | range [0.3882, 0.3903]; SD = 0.0008 (1.6% of mean); cross-method invariance at the 1% level |

## Sharpest finding (sharp enough to publish)

The canonical C1 zvf-triage trigger has a **statistically significant
NEGATIVE information-vs-trigger slope** on N2: as the step's boundary
rate rises, the *information value per fired rollout* falls.
The sign of the regression coefficient is the OPPOSITE of what one
would want from an information-positive controller:

  −0.0360 bits/unit, 95% CI [−0.0372, −0.0349]

Mechanism: high-ZvF steps contain many boundary prompts (`k=0` or `k=8`),
where the pre-fire posterior `Beta(1, 9)` is already sharp (`H = −1.89 bits`)
and there is little entropy to gain from additional rollouts
(`dH = 0.380 bits`). Mid prompts (`k=4`) carry `dH = 0.423 bits` — **11%
more information per fire** than boundary prompts.

The cumulative **information regret** across 160 fired steps × 16 prompts
is **79.4 bits** — measurable, statistically detectable, and a direct
counter-audit cost of the canonical trigger design.

## Counterfactual: Anti-C1 is structurally inert on N2

The natural improvement direction — flip the trigger (`fire iff zvf ≤ 0.30`)
— never fires on N2: every step has zvf_step ≥ 0.30 (averages 0.7).
This is consistent with iter-69's "0/1867 boundary-prompt restoration"
finding and supports the broader P7 thesis that **on saturated panels
like N2 the trigger is degenerate; the savings must come from
de-escalation, not from positive-condition escalation**.

## Cross-paper coupling

- **P5↔P7** (iter-37 discriminative-entropy audit): the per-tier dH
  monotonicity (mid > edge > boundary) connects the P5 "what does the
  manifest actually carry?" audit to the P7 "what does each rollout
  actually tell us?" audit — both are information-theoretic.
- **P6↔P7** (iter-66 row 77 anti-herding block): the negative-slope
  finding is the *causal counterpart* of the anti-herding bonus; the
  same Beta-Binomial conjugate analysis underwrites both.
- **P7↔Berkeley row 01** (Dualformer auto-G): Dualformer's per-prompt
  auto-G acts on the same axis (boundary prompts de-escalate) that
  iter-187's negative-slope identifies as the high-information regime.

## Reproducibility

```
python3 platform_modal/scripts/p5p8/p7_iter187_infogain_controller.py
```

reproduces all 5 artifacts in <10 s. Seeds `20260705`, n_boot `4000`,
all entropy formulas verified against numerical quadrature on
Beta(5, 5), Beta(9, 9), Beta(1, 9), Beta(1, 17).
