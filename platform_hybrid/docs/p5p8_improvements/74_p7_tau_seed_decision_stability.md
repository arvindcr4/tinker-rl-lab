# 74 — P7 τ seed-robustness: per-step DECISION STABILITY on N10 (iter 63)

**Pillar:** P7 (Pillar 3 — adaptive-G controller)
**Vein:** brief vein (c) — seed-robustness of the trigger threshold on the
growing N10 panel, *sharpened* beyond the existing
`sec:p7-controller-seedrobust` headline. The existing analysis reports
**per-τ firing-rate mean ± CI** across the 5 seeds; this iteration
adds the **per-STEP cross-seed decision agreement** and the
**two-threshold (τ_esc, τ_des) sweep** that the iter-51 controller's
asymmetric structure motivates but the prior art never measured.

## The gap this iter fills

Iter 51 unified the controller as a **two-threshold policy** (escalate
when `zvf ≥ τ_esc`, de-escalate when `zvf ≥ τ_des`); iter 59's
counterfactual replay on the 4-method N2 tensors validated the
**de-escalation** branch (savings +9.39% CI [+6.82%, +12.03%]) and
showed the **escalation** branch is operationally inert on boundary
prompts (0/1867 restores). The existing seed-robustness sweep in
`sec:p7-controller-seedrobust` reports only the **firing-rate**
summary across the N10 5 seeds; it does NOT answer:

1. **Per-step decision agreement**: at the iter-51 default `τ=0.70`,
   do all 5 seeds fire on the SAME steps, or on different steps?
2. **Two-threshold operating window**: with `τ_des=0.95` and
   `τ_esc ∈ {0.5, 0.6, 0.7, 0.8, 0.85}` × `τ_des ∈ {0.80, 0.85, 0.90}`,
   where is the (savings > 0) × (cross-seed agreement high) sweet
   spot on N10?
3. **Per-seed ZVF trajectory correlation**: do the 5 seeds even
   follow the same zvf trajectory through training? If not, per-step
   triggers are inherently noisy and only aggregate statistics can be
   expected to be seed-stable.

This iter measures all three.

## Data and methods

- **Panel**: N10 — 5 seeds × 15 steps each of GRPO on Qwen3.5-4B at
  G=8, identical stack (`experiments/results/n10_seed_expansion/n10_grpo_s{42,179,316,453,590}.json`).
- **Controller**: two-threshold, parameterised
  `(τ_esc, τ_des)`, `g_esc=16, g_des=4, g_base=8`.
  The default iter-51 setting `(τ_esc=0.70, τ_des=0.95)` is included.
- **Single-τ sweep** (12 values `0.30…0.95`): for each seed, record
  per-step G choice, fire set, savings, wrong-fires (zvf>0.99 → g=16),
  saturated-fires (zvf==1.0 → g=16). Bootstrap CIs `B=10000`.
- **Two-threshold sweep**: 5×3 grid, measures total escalation fires,
  total de-escalation fires, and savings %.
- **Cross-seed agreement metrics**: (i) Jaccard of fire sets over
  all C(5,2)=10 pairs; (ii) per-step full agreement (do all 5 seeds
  pick the same G at step i?); (iii) Kendall-τ of per-step zvf
  trajectory pairs (a measure of step-order correlation, not just
  rank correlation of summary statistics).
- **Heldout-ZVF correlation**: Pearson r with bootstrap CI on
  `(heldout_acc, mean_zvf)` and `(heldout_acc, last10_zvf)`.

## Headline results

### Finding 1 — N10 is an ESCALATION-ONLY panel

The global zvf max across all 5 seeds × 15 steps is **0.875** (no
step ever saturates; the iter-51 `τ_des=0.95` de-escalation branch
**NEVER fires** on N10). Hence on N10 the controller is
**escalation-only** and ALL savings are non-positive
(`p7_tau_two_threshold_sweep.tsv`). The de-escalation branch — the
one iter-59 showed is responsible for +9.39% of savings on N2 — has
**zero validation power on N10**. The N10 panel is the **wrong panel
to validate the iter-51 controller's headline savings claim**; that
claim rests entirely on the iter-59 N2 counterfactual.

The panel that DOES validate the de-escalation branch would need
zvf reaching ≥ 0.95 — a "saturated" run regime, e.g. a long-horizon
run or a panel of a method that collapses to all-1/all-0 groups.
The N10 5-seed panel is the WRONG population for that test.

### Finding 2 — Per-step decisions are NOT seed-stable at iter-51 default

At `τ_esc=0.70`, the iter-51 default, the cross-seed **fire-set
Jaccard is 0.133** and the **per-step full agreement is 0.133**
(meaning 13.3% of steps see all 5 seeds pick the same G;
86.7% of steps see at least one seed disagree with the others)
(`p7_tau_seed_stability.tsv`).

In plain terms: when seed 42 fires at step 7, seed 179 fires at
step 1, seed 316 at step 11, etc. The seeds do **not agree on which
steps are escalation-worthy**, even though they **do agree on the
expected number of escalation events** (4.20 ± 1.33 per seed,
CI [3.00, 5.40], which is the existing `sec:p7-controller-seedrobust`
headline).

This is a **per-step vs aggregate decoupling** of seed-robustness:
the **firing-rate is a stable summary**, the **firing LOCATION is
not**. Implication: a controller tuned on aggregate statistics
(e.g. "fire on 28% of steps") is seed-robust; a controller that
makes a *different decision per step based on per-step state*
(e.g. "fire on the specific step where zvf crosses τ") is not
seed-robust in N10.

### Finding 3 — Cross-seed ZVF trajectories are essentially UNCORRELATED

Mean cross-seed Kendall-τ of per-step zvf trajectories:
**−0.060** (min −0.336, max +0.327; 10 pairs) — i.e. the seeds
follow **uncorrelated difficulty trajectories** through training.
Two seeds with Kendall-τ = −0.336 means: when seed A has a high-zvf
step, seed B is more likely to have a *low*-zvf step at the same
training step. Training-step index is a meaningless axis for the
zvf signal across seeds.

This explains Finding 2 mechanically: if the per-step zvf
trajectories are noise-like across seeds, then per-step threshold
crossings are also noise-like across seeds. The **aggregate** rate
of threshold crossings is the only stable thing, and that is
exactly what the existing `sec:p7-controller-seedrobust` headline
already measured.

### Finding 4 — Heldout–ZVF correlation reproduces across seeds

Pearson r (heldout_acc, last10_zvf) = **0.607** with bootstrap CI
**[−0.048, 1.000]** on 5 seeds (matching the existing paper claim
of r=0.607 with CI [0.408, 0.779] from iter 51; the wider CI here
reflects B=10000 percentile resamples on n=5, where the CI is
fundamentally wide). The 0.458 mean-zvf correlation matches the
existing r=0.458 [-0.314, 0.578] from iter 51.

The takeaway: **aggregate ZVF statistics are the right summary
level**; per-step ZVF is too noisy to be a seed-stable trigger
input on this panel.

## Two-threshold operating window (N10)

| (τ_esc, τ_des) | esc fires | des fires | savings % | 95% CI |  |
|---|---:|---:|---:|---|---|
| (0.50, 0.80) | 60 | 3 | −78.00% | narrow | (de-esc catches 3 zvf=0.875 steps) |
| (0.60, 0.80) | 42 | 3 | −54.00% | narrow |  |
| (0.70, 0.80) | 18 | 3 | −22.00% | narrow |  |
| (0.80, 0.80) | 0 | 3 | **+2.00%** | narrow | (no escalation, just de-esc) |
| (0.80, 0.85) | 0 | 3 | **+2.00%** | narrow |  |
| (0.80, 0.90) | 3 | 0 | −4.00% |  |  |
| (0.85, 0.80) | 0 | 3 | **+2.00%** | narrow |  |
| (0.85, 0.85) | 0 | 3 | **+2.00%** | narrow |  |
| (0.85, 0.90) | 3 | 0 | −4.00% |  |  |

`p7_tau_two_threshold_sweep.tsv` — full 5×3 grid.

The **only positive-savings cell** is at `τ_esc ≥ 0.80` AND
`τ_des ≤ 0.85` (so de-esc fires on the 3 zvf=0.875 steps; esc
never fires). This is a **trivial operating point** (the
controller does almost nothing), but it is the **unique τ-cell
where savings > 0 on N10**. This is the **headroom ceiling of the
iter-51 controller on the N10 panel**: it can save at most
**+2.00% rollouts** by de-escalating at the topmost zvf steps.

## Three falsifiable claims

1. **N10 cannot validate the iter-51 controller's de-escalation
   branch** (no zvf ≥ 0.95 in panel). The de-escalation branch's
   +9.39% N2 savings number cannot be reproduced on N10; the
   +2.00% is the N10-specific upper bound.
2. **Per-step decision-stability is decoupled from
   summary-stability** on N10: Jaccard 0.133 at τ=0.70 vs
   4.20±1.33 fires/seed. The seed-robustness literature for RL
   controllers should distinguish these two layers of stability.
3. **Cross-seed ZVF trajectory Kendall-τ ≈ 0** on N10
   (mean −0.060): training-step index is not a meaningful
   alignment for the ZVF signal across seeds, so per-step triggers
   are noise.

## Cross-paper coupling

- **Berkeley row-01 Dualformer auto-G** (su2024dualformer): the
  Dualformer rule *also* conditions on a per-prompt / per-step
  signal, and would suffer the same per-step non-stability on N10.
  This iter's Finding 2-3 generalises: any per-step RL controller
  on this panel has the same property.
- **Berkeley row-19 AlphaProof γ*=0 smoothing**:
  the iter-51 controller's escalation branch fires on **boundary**
  steps (zvf ≥ 0.70) where AlphaProof would apply its
  `γ*=0` smoother (i.e. neutralise the gradient). The two
  mechanisms coincide on the "no contrast, no update" regime;
  AlphaProof does it implicitly via γ*=0, the iter-51 controller
  does it by skipping the rollouts entirely (savings 0). On N10
  both mechanisms are mostly inert (the iter-51 escalation
  branch has 0/75 saturating fires across 5 seeds × 15 steps).
- **Iter 51 existing sweep**: this iter reproduces the 4.20±1.33
  headline and extends it with the per-step decoupling.

## Deliverables

- `scripts/p5p8/p7_tau_seed_decision_stability.py` (≤300 lines)
- `experiments/results/p5p8/p7_tau_seed_stability.tsv` (single-τ sweep)
- `experiments/results/p5p8/p7_tau_two_threshold_sweep.tsv` (5×3 grid)
- `experiments/results/p5p8/p7_tau_seed_stability_summary.json`
