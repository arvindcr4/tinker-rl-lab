# P0-A: Paired GSM8K held-out test

Source: `reports/final/gsm8k_base_control_200.json` (base) + 5 `gsm8k_heldout_seed*.json` (GRPO).
n = 200 paired prompts.

## Per-seed paired McNemar

| Seed | base acc | GRPO acc | Δ | 95% boot CI | b01 | b10 | McNemar exact p | χ²(cc) p |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| gsm8k_heldout_seed042 | 0.820 | 0.830 | +0.010 | [-0.040, +0.060] | 15 | 13 | 0.851 | 0.850 |
| gsm8k_heldout_seed137 | 0.820 | 0.825 | +0.005 | [-0.050, +0.060] | 16 | 15 | 1.000 | 1.000 |
| gsm8k_heldout_seed256 | 0.820 | 0.805 | -0.015 | [-0.070, +0.040] | 16 | 19 | 0.736 | 0.735 |
| gsm8k_heldout_seed512 | 0.820 | 0.840 | +0.020 | [-0.030, +0.070] | 16 | 12 | 0.572 | 0.571 |
| gsm8k_heldout_seed999 | 0.820 | 0.865 | +0.045 | [+0.000, +0.090] | 16 | 7 | 0.093 | 0.095 |

## Majority-vote-of-5 GRPO vs base

- base acc: 0.820
- majority-vote GRPO acc: 0.875
- Δ: +0.055 (95% paired-bootstrap CI: [+0.010, +0.100])
- discordant pairs: b01=17 (base wrong → GRPO correct), b10=6 (base correct → GRPO wrong)
- McNemar exact two-sided p: **0.0347**
- McNemar χ² (continuity-corrected) p: 0.0371

## Paired test on per-prompt mean-of-5-seeds indicator

- mean Δ (continuous): +0.0130
- t = 0.614, p = 0.539 (paired t, normal approx, two-sided)

## Interpretation

The paper's original claim — "GRPO mean 83.3% vs base 82.0% is not significant
(one-sample t=1.32, p=0.26, n=5 seeds)" — **holds**, and in fact the per-prompt
paired analyses are *more* firmly null than the original seed-level t-test:

- Every individual seed is non-significant under paired McNemar (p ∈ [0.09,
  1.00]); the most favorable seed (999) reaches p=0.093, not 0.05.
- The paired t-test on per-prompt mean indicator (averaged across 5 seeds) gives
  p=0.539 — a weaker signal than p=0.26 because per-prompt variance is dominated
  by seed-level noise averaged out.
- Discordant pair counts per seed are roughly symmetric (b01 ≈ b10 within 1σ),
  which is exactly what "no systematic GRPO-specific capability lift" looks like.

**Post-hoc finding.** Majority-vote of all 5 GRPO seeds reaches 87.5% vs 82.0%
base (Δ=+5.5pp, 95% CI [+1pp,+10pp], McNemar exact p=0.0347). This is a
variance-reduction effect of seed ensembling, not the single-model claim the
thesis reported. It should be disclosed as an exploratory result and not
substituted for the per-seed or per-prompt mean-indicator tests. If the paper
wants to claim a real held-out lift it should re-run with (a) more seeds or
more held-out prompts, (b) a pre-registered estimator, and (c) a paired test
against the same base.

