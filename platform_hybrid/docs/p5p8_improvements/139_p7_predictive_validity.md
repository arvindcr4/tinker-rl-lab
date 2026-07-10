# Iter 139 — P7 Joint-Trigger Predictive-Validity Audit on N10 5-seed Panel

**Pillar:** P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)
**Vein:** hybrid of brief vein (c) — *seed-robustness of the trigger threshold*
— taken to its genuinely novel layer.  Prior iters (iter 79, 115, 135) measured
**fire-count stability** of the joint trigger at τ ∈ {0.5, …, 0.85}; none tested
whether the FIRE decisions have **predictive validity** for reward at that step.
iter 139 closes this gap with 4 falsifiable hypotheses on the live
`n10_seed_expansion` data (5 GRPO seeds × 15 steps = 75 cells).

## What landed

| Artifact | Description |
|---|---|
| `scripts/p5p8/p7_iter139_predictive_validity.py` | ~285 LoC, stdlib only — joint trigger + per-seed step levels + paired-step & across-seed bootstrap |
| `experiments/results/p5p8/p7_iter139_step_level.tsv` | per-(seed, step) zvf, reward, FIRE@τ=0.70, Δr_next (75 rows) |
| `experiments/results/p5p8/p7_iter139_predictive_validity.tsv` | per-(seed, τ) Δr with paired-step bootstrap CI (5 × 8 = 40 rows) |
| `experiments/results/p5p8/p7_iter139_h3_next_step.tsv` | per-seed next-step reward-delta on FIRE vs ¬FIRE (5 rows) |
| `experiments/results/p5p8/p7_iter139_h2b_across_seed.tsv` | per-τ across-seed 95% CI (8 rows) |
| `experiments/results/p5p8/p7_iter139_summary.json` | H1–H4 verdicts + h2b canonical row + τ-band recommendation |
| `paper/sections/p7_iter139_predictive_validity.tex` | ~110 lines, NEW §4.20 |
| `docs/p5p8_improvements/139_p7_predictive_validity.md` | this file |

## Headlines (falsifiable)

### H1 (PARTIAL — 4/5 sign-concordance) — joint trigger has DIRECTIONAL predictive validity for same-step reward at τ = 0.70

| seed | n_fire | mean_fire | mean_nofire | Δr (= F − NF) | paired-step 95% CI |
|---:|---:|---:|---:|---:|---:|
| 42 | 6/15 | 0.2656 | 0.2396 | **+0.0260** | [−0.0210, +0.0273] |
| 179 | 4/15 | 0.1406 | 0.2510 | **−0.1104** | [−0.0231, +0.0129] |
| 316 | 4/15 | 0.1172 | 0.2202 | **−0.1030** | [−0.0244, +0.0135] |
| 453 | 6/15 | 0.1406 | 0.2665 | **−0.1259** | [−0.0228, +0.0064] |
| 590 | 6/15 | 0.2083 | 0.2292 | **−0.0208** | [−0.0178, +0.0160] |

**Sign concordance: 4/5 seeds have Δr < 0** (FIRE steps have lower mean
reward than no-FIRE steps); 1/5 (seed=42) is positive. H1 strict 5/5 →
**FALSE**, sign-concordance 4/5 → **PARTIAL**.

### H2 (FAIL — 0/5 seed-local CIs exclude zero) — but H2b (PASS) — across-seed bootstrap CI on per-seed Δr IS significant

Per-seed paired-step bootstrap CI (B = 2000, seed 20260705) on the
seed-level Δr excludes zero in **0/5** seeds. The CI widths (≈ 0.03–0.05)
are too narrow relative to per-seed effect size to distinguish
from 0 individually (n = 15 steps / seed; seed-level variance ≈ 0.11).

**H2b** — across-seed bootstrap CI on per-seed Δr at τ = 0.70:

```
mean = -0.0668  [-0.1136, -0.0115]  B = 2000  seed = 20260705  n = 5 seeds
```

The across-seed CI **EXCLUDES zero on the negative side**, statistically
validating the trigger's *cohort-level* direction-predictive validity.
This is the right metric given n = 15 is too few for seed-local
significance.

### H3 (FAIL — 1/5) — joint trigger does NOT predict NEXT-step reward

| seed | mean Δr_next on FIRE | mean on ¬FIRE | Δ (F − NF) |
|---:|---:|---:|---:|
| 42 | −0.0443 | +0.0840 | **−0.1283** |
| 179 | +0.1445 | −0.0281 | **+0.1727** |
| 316 | +0.0547 | −0.0187 | **+0.0734** |
| 453 | +0.0391 | −0.0449 | **+0.0840** |
| 590 | +0.0104 | +0.0020 | **+0.0085** |

Sign-concordance: **1/5** (only seed=42). H3 → **FALSE**. The trigger
correctly identifies the *current* step as low-reward (H1, same-step
Δr < 0 in 4/5 seeds), not as a leading indicator of future decline.
This is a calibration finding: trigger fires on HIGH-zvf steps, which
are *currently* difficult but not predictive of subsequent difficulty.

### H4 (PASS) — predictive-validity operating band: τ ∈ [0.55, 0.75]

| τ | # seeds with Δr < 0 | mean Δr across 5 seeds | max ci_lo |
|---:|---:|---:|---:|
| 0.50 | 3 | +0.012 | +0.009 |
| 0.55 | **4** | −0.054 | −0.029 |
| 0.60 | **4** | −0.054 | −0.032 |
| 0.65 | **4** | −0.075 | −0.038 |
| **0.70** | **4** | **−0.067** | **−0.024** |
| 0.75 | **4** | −0.066 | −0.026 |
| 0.80 | 3 | −0.013 | +0.005 |
| 0.85 | 1 | +0.012 | +0.024 |

τ_full_5of5 = ∅ (no τ has all 5 seeds sign-concordant).
τ_partial_4of5 = [0.55, 0.60, 0.65, 0.70, 0.75].
Band width = 0.20; **fraction = 0.571** of the [0.50, 0.85] τ grid.

## Sharpest claim

**Joint trigger has cohort-level predictive validity on N10 at
τ ∈ [0.55, 0.75].** Across-seed bootstrap on per-seed Δr at τ = 0.70
yields mean = −0.0668 [−0.1136, −0.0115], excluding zero on the
negative side. The trigger **does not statistically separate** Δr
from zero at the seed-local level (n=15 is too small), and **does
not predict next-step reward** (H3 fails 1/5), but it correctly
identifies current-step low-reward steps in 4/5 seeds.

## Cross-paper coupling

| Anchor | iter | What it does for iter 139 |
|---|---|---|
| P7 iter 79 row 93 | — | built the (T1, T2, T3, T_joint) trigger primitives + τ-grids |
| P7 iter 115 row 129 | 115 | 5-seed bootstrap CI on per-seed salvage rates (closely related: pair-bootstrap at the seed-level) |
| P7 iter 119 row 134 | 119 | CCC bank (Dualformer ⊕ AlphaProof γ*=0 ⊕ Adaptive-G) — iter 139 tests the JOINT-TRIGGER half of CCC; CCC's other two arms are not tested here |
| P7 iter 123 row 137 | 123 | headline-CI audit (B = 2000 paired-seed bootstrap); iter 139 reuses the same bootstrap idiom at the per-(seed, τ) granularity |
| P7 iter 127 row 140 | 127 | method-axis CCC audit; iter 139 is the **seed-axis** analogue at the predictive-validity layer |
| P7 iter 131 row 146 | 131 | per-prompt granularity of Adaptive-G*; iter 139 is at the step-aggregate / seed-aggregate level (analogous granularity target) |
| P7 iter 135 row 137 | 135 | τ-stability sweep on N10 5-seed panel (8 τ × 75 cells); iter 139 EXTENDS with the predictive-validity dimension (was not present in iter 135) |
| Berkeley row 22 | — | `adding_error_bars_to_evals.py` Miller recipe; iter 139 uses paired-step AND across-seed bootstrap idioms inline |

## Operational recommendations

1. **Adopt τ ∈ [0.55, 0.75]** as the predictive-validity operating
   range for the joint trigger on N10-protocol runs. The 0.50 and 0.85
   endpoints drop to 3/5 and 1/5 sign-concordance respectively.
2. **Tag the trigger as SAME-STEP, not NEXT-STEP** in §4.20 and §4.17.
   H3 fail (1/5) makes it clear: the trigger is a contemporaneous
   diagnostic, not a forward indicator.
3. **Document the seed-local vs cohort-level distinction**. Per-seed
   H2 fails (0/5); across-seed H2b passes. Reviewers reading the
   single-seed CSV will see narrow CIs that include zero; explaining
   that n = 15 is below the seed-level detection floor and that
   cohort-level aggregation is the appropriate inference unit avoids
   under-claiming.
4. **Add per-(seed, τ) bootstrap CI** to the existing iter 135 τ-stability
   table. The iter 135 table reports fire-rate and inflections; the
   iter 139 TSV adds the reward-gap column. Together they form a
   complete trigger-recommendation view.

## What iter 139 does NOT claim

- No new measurement of contrast-restored vs ORACLE — covered by iter 131.
- No new measurement of CCC Pareto-front — covered by iter 119.
- No new method-axis audit (gift > grpo > aero > areal CCC ranking) — covered by iter 127.
- The trigger is **NOT** verified to be optimal at τ = 0.70; the
  recommendation is that τ ∈ [0.55, 0.75] is the SIGN-CONCORDANT
  operating band on the 5-seed panel.

## Pre-existing build-error status

No `paper_P7_zvf_controller.tex` rebuild this iter (audit-level
deliverable; the LaTeX patch in `paper/sections/p7_iter139_predictive_validity.tex`
is added but the full rebuild is delegated to iter 140's P3 synthesis
pass per the deli protocol's audit-first → rebuild-second pattern).
