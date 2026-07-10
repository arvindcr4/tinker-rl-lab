# Iter 183 — P7 Trigger-Threshold Seed-Robustness on N10 5-seed Panel

## Pillar
P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)

## Brief vein picked
**Vein (c) — seed-robustness of the trigger threshold on the growing n10_seed_expansion panel**

Prior P7 iters have calibrated the trigger threshold against the N2 reward
tensor corpus (4 methods × 40 steps × 16 prompts × G=8, exact per-prompt ZVF).
The brief lists "seed-robustness on growing n10_seed_expansion panel" as a
distinct sub-vein. The panel grew live after iter-155 (`p7_iter155_tau_stability_5seed.py`),
and iter-183 is the first re-audit of **the trigger-bit firing rate** at
this panel specifically, with bootstrap CIs and cross-seed TOST.

## Hypothesis matrix (6 settled; 4 PASS + 2 honest FAIL)

| H | claim | bar | result |
|---|---|---|---|
| H1 | mean_rate(τ=0.55) > 0.50 with CI lo > 0.40 | rate_high | **PASS** (0.6000, CI lo 0.4933) |
| H2 | mean_rate(τ=0.80) < 0.10 with CI hi < 0.20 | rate_low | **PASS** (0.0400, CI hi 0.0933) |
| H3 | firing rate monotone non-increasing in τ on 5/5 seeds | 5/5 | **PASS** (5/5) |
| H4 | cross-seed SD of rate at τ=0.70 < 0.20 | <0.20 | **PASS** (0.0989) |
| H5 | TOST-equivalence across all 10 seed pairs at τ=0.65 (margin 0.15) | 10/10 | **FAIL** (0/10 — honest negative) |
| H6 | mean Spearman ρ of fire-bit vectors > 0.50 at τ=0.65 | >0.50 | **FAIL** (-0.0611 — honest negative) |

**Verdict: 4/6 PASS.** The two FAILs are not bugs — they are sharpest findings.

## Sharpest paper-grade findings

### F1 — Aggregate firing rate IS seed-robust (H1, H2, H3, H4 all PASS)
At the aggregate firing-rate layer (one number per (seed, τ)), the trigger
behaviour is well-defined and monotone in τ:
- 0.50 → 0.840 mean_rate (cross_sd 0.112) — signal-rich regime
- 0.55 → 0.600 mean_rate (cross_sd 0.116) — practical trigger zone
- 0.65 → 0.280 mean_rate (cross_sd 0.099) — natural break-point
- 0.80 → 0.040 mean_rate (cross_sd 0.060) — sparse regime

The **natural break-point at τ ∈ {0.65, 0.70, 0.75}** — three τ values with
identical mean_rate 0.28 and cross_sd 0.099 — is the empirical "trigger
should not be tuned here" zone; below it the trigger fires on more than half
of steps, above it it fires on < 5%. The cross_sd 0.099 at the natural
break-point is **less than half** of the 0.20 bar (H4 PASS decisive).

### F2 — Per-step fire pattern is NOT seed-portable (H5, H6 both FAIL)
At the per-step level, different seeds fire on different steps:
- TOST at margin 0.15 fails on 10/10 seed pairs at τ=0.65 (paired-step
  bootstrap CIs of difference are wider than 0.15)
- Spearman ρ = -0.06 — essentially zero rank correlation between
  fire-bit vectors across seed pairs

This is a **strong, sharp negative finding** that the trigger fires on
different steps for different seeds, even though the AGGREGATE firing rate
is seed-robust. The seed-portability is at the population level (≈ 28% of
steps fire at τ=0.65, regardless of seed), NOT at the step level (which
specific steps fire is seed-dependent).

### F3 — Two-tier seed-portability of the adaptive-G controller
Combined with iter-167 (oracle regret, controller-level counterfactual)
and iter-155 (5-seed τ-stability on the gain-vs-τ curve), the P7 controller
has **two-tier seed-portability**:

| Tier | Level | Seed-robust? | Iter evidence |
|---|---|---|---|
| 1 — Population | mean firing rate, mean gain, mean contrast | **YES** (cross_sd < 0.20 at all τ) | iter-183 H1-H4 |
| 2 — Instance | which step fires, which prompt restored | **NO** (Spearman ≈ 0, TOST fails) | iter-183 H5-H6 |

For a production deployment, this is sufficient — the controller needs to
be calibrated to a desired **expected firing rate**, not a per-step
fire schedule. For a paper-grade analysis, the H5-H6 failures should be
reported honestly, because reviewer #2 will run the same TOST and find
the per-step non-portability on their own if we don't surface it.

### F4 — Empirical anchoring of FRONTIER Round 2 (ZVF = signal availability)
H1 + H2 jointly show that ZVF is a strong, monotone, seed-robust
**trigger signal**: above τ=0.80 the signal vanishes (mean_rate < 0.05
on all 5 seeds); below τ=0.50 the signal saturates (mean_rate > 0.66 on
all 5 seeds); the natural operating zone is τ ∈ [0.55, 0.75] where
mean_rate transitions from 0.60 → 0.28. The trigger is well-calibrated
at the population level, validating FRONTIER Round 2's claim that ZVF is
"what the sampler + group size + difficulty distribution expose to GRPO
as zero/nonzero advantage" — a structural, not noisy, signal.

## Cross-paper coupling

| Coupled pillar | Coupling vein | iter |
|---|---|---|
| P7 iter-79 multitrigger | baseline N2 tau sweep | 79 |
| P7 iter-87/88 hysteresis | single-seed τ-stability on N10 | 87, 88 |
| P7 iter-99 seed-threshold | first seed-robustness audit (single τ) | 99 |
| P7 iter-135 tau-stability N10 | first N10 1-seed τ-stability | 135 |
| P7 iter-155 τ-stability 5-seed | pre-iter-183 5-seed baseline | 155 |
| P7 iter-167 oracle regret | controller-level counterfactual | 167 |
| P7 iter-171 headline CIs | headline-layer bootstrap CIs | 171 |
| P7 iter-175 calibrated-hybrid | dualformer + alphaproof fusion | 175 |
| P7 iter-179 contrast-restored | per-prompt restored contrast on fired N2 steps | 179 |
| **P7 iter-183 (this)** | **trigger-bit seed-robustness on growing N10 panel** | **183** |
| FRONTIER Round 2 (ZVF = signal) | H1+H2 jointly anchor FRONTIER claim | frontier |

iter-183 closes the orthogonal **trigger-bit** layer of seed-robustness:
prior iters covered τ-stability of the gain curve (iter-155), population
firing-rate (iter-99 single-τ), and aggregate headline CIs (iter-171);
iter-183 adds **bootstrap CI on per-seed firing rate + cross-seed TOST +
Spearman ρ on fire-bit vectors**, which is the standard reviewer-2
expectation for "seed-robustness" claims.

## Operational recommendations

1. **REPORT** the two-tier seed-portability explicitly in paper_P7 §
   `sec:p7-design-rules` — population-robust, instance-not-portable.
2. **ADD** `tab:p7-iter183-cross-seed-ci` to `paper_P7_zvf_controller.tex`
   showing the 8-row cross-seed table with bootstrap CIs.
3. **CITE** iter-183 alongside iter-167 (oracle regret) when claiming
   "the controller is seed-portable" — the population-level claim survives,
   the step-level claim does not.
4. **WIRE** `python3 platform_modal/scripts/p5p8/p7_iter183_trigger_seed_robust_n10.py`
   as a CI-style pre-commit gate for any future P7 controller variant —
   gate fails if cross_sd > 0.20 at τ=0.70 OR if monotone seeds < 5/5.
5. **EXTEND** in next-iter: include the iter-131 per-prompt adaptive
   G-star signal in the fire-bit vector (so the test covers G* choice
   seed-robustness, not just fire-bit seed-robustness).

## Artifacts

| Path | rows × cols | bytes |
|---|---|---|
| `platform_modal/scripts/p5p8/p7_iter183_trigger_seed_robust_n10.py` | 285 LoC | stdlib only |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_per_obs.tsv` | 75 × 12 | seed,step,zvf,reward + 8 fire bits |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_per_seed_rate.tsv` | 40 × 7 | per-(seed,τ) rate + 95% bootstrap CI |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_cross_seed_ci.tsv` | 8 × 7 | per-τ mean + cross_sd + cross-bootstrap CI |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_tost.tsv` | 80 × 8 | per-(pair,τ) paired-block-bootstrap + TOST |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_spearman.tsv` | 80 × 3 | per-(τ,pair) Spearman ρ |
| `platform_hybrid/experiments/results/p5p8/p7_iter183_summary.json` | H1-H6 verdicts | structured |

## What this is NOT

- It is not a recalibration of τ — the existing τ ∈ [0.55, 0.75] is the
  correct operating zone and iter-183 confirmsthis.
- It is not a Tinker rerun — the N10 5-seed panel already exists in
  `platform_hybrid/experiments/results/n10_seed_expansion/`.
- It is not a paper rebuild — the iteration is a delta-layer finding,
  not a section add. The P7 design-rules section should grow by ~3 lines
  in next-iter synthesis.
- It is not a new controller — it is an **audit** of the existing
  C4 controller's seed-robustness at the trigger-bit layer.

## Why this iteration

The brief lists 4 sub-veins for P7. iter-179 (last iter) covered vein (a)
in its strongest form (per-prompt contrast-restored). iter-175 covered
vein (b) (Calibrated-Hybrid C6 = dualformer + alphaproof). iter-171
covered vein (d) (headline bootstrap CIs). **Vein (c) was the remaining
gap** — and the brief calls it out by name. iter-183 closes it.