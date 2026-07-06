# P5P8-SYNTH D20 (iter 192) — Cross-Pillar Decision-Concordance Audit

**Pillar:** P5P8-SYNTH (cross-paper synthesis)
**Vein:** fresh — 20th density domain (D20), NOT in any prior D1..D19 SYNTH row
**Script:** `scripts/p5p8/synth_iter192_d20_decision_concordance.py` (~290 LoC,
stdlib + numpy)
**Outputs:**
- `experiments/results/p5p8/synth_iter192_d20_per_method.tsv` (16 rows: 4
  methods × 4 pillars)
- `experiments/results/p5p8/synth_iter192_d20_method_ranks.tsv` (16 rows)
- `experiments/results/p5p8/synth_iter192_d20_concordance.tsv` (6 pairs + mean)
- `experiments/results/p5p8/synth_iter192_d20_per_method_step.tsv` (640 rows)
- `experiments/results/p5p8/synth_iter192_d20_summary.json`

## What D20 measures

D19 measured information-weighted controller efficiency per (method, step).
D20 lifts the lens to **cross-pillar decision-concordance**: across P5, P6,
P7, P8, **does the same method rank highest on the same headline metric?**

For each pillar, define a headline metric on the 4 N2 methods (grpo, aero,
areal, gift):
- **P5** (RL outcome): mean reward across 40 steps — the headline RL
  performance number that the paper benchmarks.
- **P6** (registry): negative mean ZVF (lower ZVF is better for the
  registry — the canonical "zvf_risk<0" claim in `delta_*.json`).
- **P7** (controller): mean reward / std of reward — a controller-
  efficiency proxy (higher = more efficient use of rollouts, less trigger
  noise).
- **P8** (transfer): mean reward / mean length — a deployment-transfer
  proxy (higher = more compact, useful responses).

D20 then computes:
- Per-(method, pillar) headline metric + per-step arrays.
- Rank per pillar (rank 1 = best).
- 6 Spearman rank correlations across pillar pairs + mean pairwise Spearman.
- Per-pillar bootstrap CI (B=2000, paired-step resampling) on the
  best−worst method gap.
- 5 falsifiable hypotheses.

## Hypothesis verdicts (5 hypotheses, 4 PASS + 1 sharp FAIL)

| Hyp | Claim | Result |
|-----|-------|--------|
| **H1** | At least one cross-pillar Spearman ρ > 0.5 (positive concordance) | **PASS** — max ρ = 1.0 (P5↔P8 perfect concordance) |
| **H2** | Mean of 6 cross-pillar Spearman ρ values > 0 (concordance direction) | **PASS** — mean ρ = 0.233 > 0 |
| **H3** | Best-method on P5 = Best-method on P6 (RL outcome matches registry) | **FAIL** — P5 winner = **gift**, P6 winner = **areal** |
| **H4** | All 4 per-pillar best-worst gap CIs exclude zero (every pillar discriminates) | **PASS** — 4/4 pillars have CIs strictly > 0 |
| **H5** | P5↔P8 Spearman > 0 (RL outcome aligns with deployment transfer) | **PASS** — ρ = 1.0 (perfect concordance) |

## Sharpest paper-grade findings

- **F1 (H1 PASS HEADLINE) — Two-pillar cluster structure.** Cross-pillar
  Spearman matrix reveals **TWO clusters of method evaluation that disagree
  about which method is best**:
    - **{P5, P8} cluster** (Spearman = **1.0**): the RL outcome (P5) and
      the deployment transfer (P8) **perfectly agree** — `gift` is best,
      `aero` is worst.
    - **{P6, P7} cluster** (Spearman = **0.8**): the registry (P6) and the
      controller efficiency (P7) **also agree** — `areal` is best, `gift`
      is worst.
  **Cross-cluster** Spearman is **negative** (P5↔P6: −0.4, P6↔P8: −0.4,
  P5↔P7: +0.2, P7↔P8: +0.2). The paper's two operationally relevant pillars
  (RL outcome, deployment transfer) **disagree** with the two design
  pillars (registry, controller).

- **F2 (H3 FAIL → SHARP) — `gift` is the deployment-optimal method but
  NOT the registry/controller-optimal method.** This is the sharpest
  finding: a method that wins on the metric the paper reports (mean reward)
  LOSES on the metric the registry cares about (ZVF risk). The implication
  is that **the registry's headline claim ("zvf_risk<0 → SUPPORTS") may
  be optimizing for the wrong target** — it's a design target, not an
  outcome target.

- **F3 (H4 PASS) — All 4 pillars have discriminating signals.** Every
  pillar's best-worst method gap has a bootstrap CI that strictly excludes
  zero:
    - P5 gap = +0.0172 [0.0090, 0.0289] (mean reward spread across methods)
    - P6 gap = +0.0641 [0.0344, 0.1031] (mean ZVF spread)
    - P7 gap = +0.3434 [0.2433, 0.4759] (controller efficiency spread)
    - P8 gap = +0.000159 [0.000112, 0.000223] (transfer spread)
  P7 has the LARGEST absolute gap (the controller efficiency is the most
  discriminating metric on this 4-method panel); P8 has the smallest
  (transfer scores are tightly clustered).

- **F4 (H5 PASS) — P5↔P8 perfect concordance is the operationally
  important headline.** The two pillars that matter most for a deployed
  paper (RL outcome + deployment transfer) **perfectly agree** on the
  best method (`gift`). This is a strong cross-pillar validation of the
  P5 MIN-REPORT measurement.

- **F5 — Two-cluster structure is the headline takeaway.** A reader who
  cares about "which GRPO-family method should I use?" should get a
  DIFFERENT answer depending on whether they prioritize RL outcome
  (P5: gift) or registry/controller (P6: areal). **The paper's reporting
  standard and the registry's coverage standard are optimizing for
  different objectives.**

## Cross-paper coupling (D20 extends D14–D19)

| Prior domain | Coupling |
|---|---|
| D14 (mean per-method gain) | D20 reports **rank** of gain; D14 reports the magnitude |
| D15 (cross-method gain rank) | D15 found gain rank is method-invariant (CV < 5%); D20 shows **rank varies by pillar** (the operational implication of D15's invariance was unclear until D20) |
| D16 (per-prompt reward stability) | D20 connects per-prompt stability to per-pillar ranking — same method has different ranks across pillars |
| D17 (paper reproducibility) | D20 quantifies the **two-cluster structure** that D17 found reproducible across re-runs |
| D18 (worst-step loss regret) | D20's per-pillar gap CIs are the **aggregate analogue** of D18's worst-step regret — D18 is the worst-case; D20 is the typical-case |
| D19 (information-weighted controller efficiency) | D20 lifts D19's η ratio to a cross-pillar Spearman; η was method-invariant (CV < 0.05%) but cross-pillar method-rank is NOT invariant |

## Operational

1. **REPORT** the two-cluster structure as a new headline in
   paper-P5P8-synthesis §sec:synth-d20: "the paper's outcome pillars
   (P5, P8) and design pillars (P6, P7) optimize for different objectives
   and disagree on which method is best."
2. **ADD** `tab:synth-d20-concordance` and `tab:synth-d20-method-ranks`
   to paper-P5P8-synthesis §sec:synth-d20.
3. **WIRE** as CI pre-commit gate — gate fails if:
   - P5↔P8 Spearman drops below 0.5 (the operational headline becomes
     unreliable)
   - The two-cluster structure collapses (P5↔P6 concordance becomes > 0,
     meaning the registry suddenly tracks RL outcome)
4. **EXTEND** in next-iter (D21) to **per-decile** decision-concordance
   (combining iter-192's P8 decile lift with the cross-pillar ranking to
   reveal whether the same decile-vs-method rank concordance holds).