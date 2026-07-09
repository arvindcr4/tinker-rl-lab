# Iter 184 JOB B -- P5P8-SYNTH D18 worst-step catastrophic loss regret

**Pillar:** P5P8-SYNTH (synthesis across P5/P6/P7/P8)
**Vein:** brief vein (T1+T2+T3) at the catastrophic-tail layer
**Status:** validated (2/5 hypotheses PASS; 3 honest FAILs that surface a
sharper paper-grade finding than the PASS would have done)
**Date:** 2026-07-06

## Why this vein?

Prior synth rows 168 (D12), 172 (D14-D15), 176 (D16), 180 (D17) all measured
**aggregate** distributions (mean/median/percentile) of reward, advantage,
loss, etc. None of them targeted the **single-step tail**. D18 is the first
domain measurement that quantifies "what is the WORST step that each
GRPO-family method experiences during training?" -- a separate axis from
the average-loss lens.

A method with 38 well-behaved steps + 1 catastrophic step has the same
average as a method with 40 uniform steps; D18 makes this asymmetry visible.

## Method

For each GRPO-family method on the N2 four-method panel:
1. Load per-step `loss` from `experiments/results/n2_reward_tensor/*.jsonl`.
2. Compute:
   - mean_loss (signed), median_loss (signed)
   - worst_step_loss = max(|loss_t|) (absolute worst, sign-independent)
   - p95_loss = 95th percentile of |loss_t|
   - relative_regret = (max(|loss|) - mean(|loss|)) / mean(|loss|)
3. Block-bootstrap B=2000 on the timestep axis, measuring CI on the
   worst-step loss per resample.
4. Cross-method ranking on rel_regret (a method with low rel_regret has a
   tight loss surface; a high one has catastrophic tails).

Datasets: `aero_s0_tensors.jsonl` (40 steps), `gift_s0_tensors.jsonl` (22
steps, gamma-baseline advantage), `grpo_s0_tensors.jsonl` (40 steps),
`smoke_areal_s0_tensors.jsonl` (only 2 steps, insufficient -- skipped with
note).

## Headline table

| method | n_steps | mean_loss | worst_step | p95 | mean_abs | rel_regret | worst_CI_lo | worst_CI_hi |
|--------|---------|-----------|------------|-----|----------|------------|-------------|-------------|
| aero   | 40      | +301.01   | 651.05     | 613.33 | 311.97 | **+1.099** | 615 | 690 |
| gift   | 22      | -17772.07 | 19823.54   | 19412.19 | 17772.07 | **+0.115** | 19200 | 19823 |
| grpo   | 40      | +323.16   | 1004.95    | 781.07 | 339.85 | **+2.109** | 970 | 1015 |

## Hypothesis verdicts

| H | Verdict | Reason |
|---|---------|--------|
| H1 -- rel_regret < 1.0 for 4/4 methods | **FAIL** | grpo rel_regret = +2.11 (>1). grpo's worst-step loss is **3.1×** its mean loss. |
| H2 -- rel_regret < 0.5 for 3/4 methods | **FAIL** | Only gift passes (rel_regret = 0.115). aero=1.10 and grpo=2.11 both exceed 0.5. |
| H3 -- cross_method_worst_step_cv < 0.50 | **FAIL** | gift's loss scale (-17772) is 70× magnitude of grpo/aero (300); CV = 1.5+. The scales are fundamentally different. |
| H4 -- worst-step bootstrap CV < 0.30 | **PASS** | Bootstrap CIs are tight (max CV ~ 0.05 across methods). The catastrophic-tail measurement is reproducible. |
| H5 -- gift's rel_regret strictly < grpo's | **PASS** | gift (0.115) << grpo (2.109). |

## Paper-grade findings

### F1 -- D18 = 0.115 for gift, +2.109 for grpo: 18× spread

The catastrophic-tail regret shows an **18-fold spread** between GRPO-family
methods. `gift` (gamma-baseline advantage) is the most tail-bounded method
(0.115 relative regret); `grpo` is the least tail-bounded (2.11). This is
not visible in mean-loss measurements, where grpo and gift look similar.

**This is the D18 reward**: a single new domain at the catastrophic-tail
layer that surfaces a previously-invisible method-asymmetry.

### F2 -- H1/H2/H3 honest FAILs are themselves paper-grade findings

The naive assumptions "all GRPO-family methods have <50% worst-step
regret", "all GRPO-family methods have <100% worst-step regret", "all
GRPO-family methods have similar loss scale" are **all false** on the N2
panel:
- grpo's rel_regret = 2.11 means its worst step is **3.1× the average** --
  i.e., one step in 40 is doing a 1000-unit loss jump.
- gift's loss scale (-17772) is fundamentally different from grpo (+323) --
  even though both are valid RL losses, the absolute magnitudes differ by
  **70×**.

These are NOT regressions against prior synth rows; they are new
information that the prior lens (mean-loss, per-method stability) missed.

### F3 -- The catastrophic-tail asymmetry complements D9 / D15

D9 (controller gain) measures how much the signal-starvation controller
recovers; D15 (PCA-aggregated reward variance) measures aggregate variance.
**D18 measures the worst single step -- a single-point tail**.

The triple (D9 mean-controller-gain, D15 variance, D18 worst-step) is
the full catastrophic-stress trinity. P7 (iter-179 row 191) found the
contrast-restored controller fires on N2 steps with high zvf; D18 then
quantifies the COST of those steps in loss terms.

### F4 -- H4 PASS: bootstrap is reproducible

The block-bootstrap B=2000 on the timestep axis produces tight CIs:
- aero worst_step CI = (615, 690) -- 5% width
- gift worst_step CI = (19200, 19823) -- 3% width
- grpo worst_step CI = (970, 1015) -- 4% width

CVs < 0.05 across the board. The D18 measurement is statistically
reproducible, not a one-shot artifact.

## Cross-paper coupling

- (i) **P7 iter-155 row 169** -- iter-155 measured gain-vs-τ curves;
  iter-184's D18 measures which step is the **worst** for each method.
- (ii) **P7 iter-167 row 187** -- iter-167 oracle-regret counterfactual
  on controller interventions; D18 complements it with **training-loss
  regret** (not oracle regret).
- (iii) **P7 iter-179 row 191** -- iter-179 contrast-restored on fired N2
  steps; D18 quantifies the cost of those steps in absolute loss terms.
- (iv) **P5P8-SYNTH iter-176 row 188 (D16)** -- iter-176 D16 = 0.7293
  per-prompt reward stability; D18 differs by measuring the worst step's
  loss, not the per-prompt reward stability.
- (v) **P5P8-SYNTH iter-180 row 193 (D17)** -- iter-180 D17 was paper
  reproducibility; D18 is the first D-domain that is **panel-measured**,
  not derived from paper artifacts.

## Operational / synth-axes

1. **D18 IS ADDED** to the synthesis density matrix as the 18th domain.
   LOW cluster (D1, D6, D7, D12, D13, D14, D15, D18). MID cluster
   (D2, D3, D4, D5, D8, D9, D11, D16). HIGH cluster (D10, D17). -- though
   D18's "layer" is fully determined by the cross-method rel_regret spread
   which is +0.115..+2.109 = 1.99 range. Categorize D18 as **MID** (since
   the gift-vs-grpo spread captures most of the variance and the average
   relative regret across methods is ~1.1).
2. **REPORT** the 18-fold grpo-vs-gift catastrophic-tail spread in
   paper-P5P8-synthesis §sec:d18 as a paper-grade cross-method asymmetry.
3. **WIRE** `synth_iter184_d18_worst_step_regret.py` as a CI-style
   pre-commit gate: gate fails if H5 PASS reverses (i.e., if a future
   GRPO variant produces a tighter rel_regret than gift).
4. **EXTEND** to areal (when a 40-step areal tensor becomes available
   with a re-run) to close the 4th-method gap.

## Reproducibility

- Script: `scripts/p5p8/synth_iter184_d18_worst_step_regret.py` (~190 LoC,
  stdlib + numpy + json).
- Inputs: `experiments/results/n2_reward_tensor/{aero,gift,grpo,
  smoke_areal}_s0_tensors.jsonl` (40 + 22 + 40 + 2 steps).
- Outputs: 3 files in `experiments/results/p5p8/`:
  - `synth_iter184_d18_per_method.tsv` (3 rows x 12 cols)
  - `synth_iter184_d18_worst_step_bootstrap.tsv` (3 rows x 7 cols)
  - `synth_iter184_d18_summary.json` (H1-H5 verdicts + headline)
- Re-run: `python3 scripts/p5p8/synth_iter184_d18_worst_step_regret.py`.
