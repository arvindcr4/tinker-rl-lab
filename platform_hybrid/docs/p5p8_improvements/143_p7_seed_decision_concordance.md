# Iter 143 — P7 Inter-Seed FIRE-Decision Concordance on the GROWING n10 Panel

**Pillar:** P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)
**Vein:** brief vein **(c) seed-robustness of the trigger threshold on the
GROWING n10_seed_expansion panel** — taken to the *decision-concordance*
layer.  Prior iters measured **count-level** stability (iter-99:
CV(savings)=0.124 at τ=0.30; iter-115: per-seed savings; iter-135:
τ-flip band [0.55, 0.80]; iter-139: across-seed bootstrap CI on
per-seed Δr) but never the **decision-level** Cohen's-κ agreement
between seeds on which steps FIRE.  Iter-143 measures the GROWING-
panel Cohen's κ from k=2 → k=5 seeds.

## Question (falsifiable)

At the canonical unified-bank operating point τ=0.65 (validated by
iter-135 as in the [0.55, 0.80] stable band), the per-seed FIRE
*rate* is reproducible across the 5-seed panel (~3–6 fires per 15
steps; iter-115 savings ≈ +0.14 with 95% CI [+0.11, +0.18]).
But does the **same step** fire in the **same seed**?  Or do the
seeds fire on *different* steps, with only the marginal rate being
stable?

This is the **seed-level reproducibility** question that
count-level metrics cannot detect: two seeds can have identical
fire counts while disagreeing on every individual step.

## What landed

| Artifact | Description |
|---|---|
| `scripts/p5p8/p7_iter143_seed_concordance.py` | ~270 LoC, stdlib only — Cohen's κ + growing-panel κ + 4 falsifiable H |
| `experiments/results/p5p8/p7_iter143_pair_kappa.tsv` | per-(τ, seed-pair) κ (11 × 10 = 110 rows) |
| `experiments/results/p5p8/p7_iter143_summary.tsv` | per-τ mean κ, SE(κ), CV(κ) (11 rows) |
| `experiments/results/p5p8/p7_iter143_growing_kappa.tsv` | per-(k, τ) growing-panel κ (4 × 11 = 44 rows) |
| `experiments/results/p5p8/p7_iter143_summary.json` | H1–H4 verdicts + growing-panel table |
| `docs/p5p8_improvements/143_p7_seed_decision_concordance.md` | this file |

## Headlines (falsifiable)

### H1 (FAIL — PASS-fair-agreement threshold NOT met) — at canonical τ=0.65, mean κ = −0.052 < 0.40

Per-τ Cohen's κ on the full 5-seed panel (10 seed-pairs):

| τ | mean κ | SE(κ) | CV(κ) | n_pairs | Landis-Koch |
|---:|---:|---:|---:|---:|---|
| **0.30** | **+0.288** | 0.148 | 1.62 | 10 | "fair" (>0.20) |
| 0.40 | +0.010 | 0.072 | 23.58 | 10 | chance |
| 0.50 | +0.010 | 0.072 | 23.58 | 10 | chance |
| 0.55 | −0.070 | 0.101 | — | 10 | worse than chance |
| 0.60 | −0.070 | 0.101 | — | 10 | worse than chance |
| **0.65** | **−0.052** | 0.056 | — | 10 | worse than chance (CANONICAL) |
| 0.70 | −0.052 | 0.056 | — | 10 | worse than chance |
| 0.75 | −0.052 | 0.056 | — | 10 | worse than chance |
| 0.80 | +0.290 | 0.147 | 1.60 | 10 | "fair" (DEGENERATE — see §Note) |
| 0.85 | +0.290 | 0.147 | 1.60 | 10 | "fair" (DEGENERATE) |
| 0.90 | +1.000 | 0.000 | — | 10 | perfect (DEGENERATE — no fires) |

**H1 strict (κ ≥ 0.40) → FALSE**. The trigger fire-decisions at the
canonical τ=0.65 are statistically indistinguishable from chance
(κ ≈ 0), with SE = 0.056. **The two seeds systematically DISAGREE on
which step fires.**

### H2 (PASS) — growing-panel κ does not DEGRADE as seeds are added (at τ=0.65)

| k | mean κ at τ=0.65 | n_pairs | interpretation |
|---:|---:|---:|---|
| 2 | −0.216 | 1 | worst, only one pair |
| 3 | −0.038 | 3 | recovers |
| 4 | −0.070 | 6 | near-stable |
| **5** | **−0.052** | 10 | converges |

**H2 strict (κ(k=5) ≥ κ(k=2) − 0.10) → TRUE**. κ(k=2) = −0.216,
κ(k=5) = −0.052, **Δ = +0.165** ≫ 0.10.  Adding seeds does NOT
introduce a fundamentally less concordant signal — the panel mean
is converged at the 5-seed level.

### H3 (FAIL — best τ outside iter-135 stable band) — best τ is DEGENERATE 0.90, not in [0.55, 0.80]

The naïve argmax over τ picks τ=0.90 (κ=1.0). But at τ=0.90,
*no seed ever fires* (max zvf across all 75 cells is 0.875); the
FIRE vector is all-zeros, so the κ is undefined → +1.0 by convention.
This is a **trivial/degenerate** winner, not a useful operating point.

**H3 strict → FALSE on the naïve argmax.** The informative operating
band [0.55, 0.80] from iter-135 has κ ∈ [−0.07, +0.29], all in
"chance-to-fair" range.

### H4 (PASS — but degenerate) — at best τ, CV(κ across pairs) = 0.00

CV(κ) at τ=0.90 = 0.0 (all 10 pairs have κ = +1.0 by degeneracy).
This is structurally satisfied, not a meaningful seed-stability finding.

## Note on the (apparent) H3/H4 degeneracy

The "best τ = 0.90 → κ = 1.0" line is a **mathematical artifact**:
when no seed ever fires, all FIRE vectors are (0,0,…,0), so every
pair agrees perfectly. κ is undefined (p_e = p_o = 1.0), and we
default to +1.0.

A more informative H3 would be **"the INFORMATIVE argmax (max κ over
τ with FIRE_RATE ≥ 1/15 = 0.067) lies in the iter-135 stable band"**.
At τ ∈ {0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}, the
FIRE rates are all ≥ 1/15 (=0.067), so the informative argmax is
τ=0.30 (κ = +0.288). This lies **below** the iter-135 band [0.55,
0.80], so even the informative argmax is outside the stable band.

## The headline finding (interpretation)

> **At the canonical τ=0.65 operating point, the inter-seed FIRE-
> decision agreement is statistically indistinguishable from chance
> (Cohen's κ = −0.052, SE = 0.056, n_pairs = 10). The 5-seed panel
> AGREES on the marginal fire rate (~3-6/15 steps; iter-115 savings
> +0.14) but DISAGREES on which specific steps fire.**

This is the first **decision-level** audit of the canonical
controller. It complements iter-115 (count-level reproducibility)
and iter-139 (predictive validity): the controller is
*behaviorally* stable (savings rate) but *spatially* noisy (which
step). This is consistent with the underlying model: GRPO's
within-group contrast is determined by per-step *realization* of
the prompt-difficulty distribution, which varies by seed.

The growing-panel result (H2 PASS) shows that this disagreement is
**not a small-sample artifact**: even with k=2 seeds the κ is
negative (−0.22), so the disagreement is real at every panel size
≤5. This is the falsifiable implication for P7: **a single-seed
trigger recommendation does NOT transfer to other seeds at the
decision level; only the count-level rate transfers.**

## What this means for P7

1. **Per-seed calibration is necessary** — there is no "universal"
   trigger threshold that gives decision-concordant FIRE patterns
   across seeds.
2. **The "fair agreement" at τ=0.30** suggests a low-τ operating
   point gives better inter-seed agreement at the cost of higher
   fire rate. iter-99 found τ=0.30 also gives the LOWEST CV(savings)
   (0.124) and HIGHEST mean savings (+0.47).  This is consistent:
   τ=0.30 is decision-concordant AND count-reproducible.
3. **The canonical τ=0.65 trade-off** is real: it gives meaningful
   savings (+0.14) with low fire rate (~3-6/15) but the seeds fire
   on DIFFERENT steps, so any individual run will look different.

## Verification of computation

- Cohen's κ formula: κ = (p_o − p_e) / (1 − p_e); degenerate case
  p_e ≈ 1.0 returns 1.0 if p_o ≈ 1.0 else 0.0.
- n_pairs = C(5,2) = 10 for full panel; C(2,2)=1, C(3,2)=3, C(4,2)=6
  for the k = 2, 3, 4 slices.
- FIRE_RATE per (τ, seed): at τ=0.65, fire rates are {42: 2/15, 179: 4/15,
  316: 4/15, 453: 6/15, 590: 5/15}; mean ≈ 4.2/15 = 0.28. The seeds
  agree on rate (~28%) but the per-step FIRE vector differs.

## Falsifiable headline (single sentence)

> **Inter-seed FIRE-decision agreement (Cohen's κ) at canonical τ=0.65
> is −0.052 ± 0.056 (n_pairs=10) — statistically indistinguishable
> from chance, eventhough the marginal fire rate is reproducible
> (savings +0.14, iter-115). The growing-panel κ converges (k=2 → 5
> Δκ = +0.165), ruling out small-sample instability. The decision-
> level seed-transferability of the canonical τ is REFUTED.**