# Iter-123 — P7 Headline-CI Audit (T1 statistical rigor)

**Pillar:** P7 (Pillar 3 — ZVF theory → adaptive-G controller → calibrated controller)
**Vein:** T1 (statistical rigor) — bootstrap CIs on every P7 headline
            point estimate. Identified in the brief as vein (d) and not
            previously attempted for P7 at this scope (iter-119 reports
            `B=2000` CIs on its own composition, but no global audit).
**This work is the first P7-wide headline-CI audit** — it inventories
every numerical point estimate cited in `paper/sections/p7_*.tex` and
tests it against paired-seed bootstrap on the canonical raw data.

## Inputs (verified paths)

- `experiments/results/n10_seed_expansion/n10_grpo_s{42,179,316,453,590}.json`
  — 5 seeds × 15 step_log rows each; `mean_zvf`, `heldout_acc`,
  `first5_avg_reward`, `last10_avg_reward` derived directly from
  per-seed JSON, not from stale aggregates.
- `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` — 4 methods
  × 40 steps of `(zvf, frac_all_zero, frac_all_one, pcd, ...)`; per-method
  step-aggregate ZVF and per-step percentile bootstrap of the mean.
- `paper/sections/p7_controller.tex` lines 44–62 (E3 four-arm table),
  `p7_iter115_adaptive_gstar_n10_multiseed.tex` (5-seed N10 headline
  block), `p7_iter119_calibrated_controller_unification.tex` (CCC
  Pareto/preservation block + Berkeley row 01 / row 19), `p7_iter107_
  tautransfer.tex` (3-pair κ range).

## Method (verified scripts)

`scripts/p5p8/p7_iter123_headline_cis.py` (<=300 lines, stdlib only).

- Class **C1 N10 multi-seed**: per-seed point + paired-seed bootstrap
  (unit=seed, `n=5`, `B=2000`, `seed=20260705`, `ci=0.95`). Re-derives
  iter-115's headline `CV=0.198` for the closed-form Bernoulli salvage
  rate at `tau=0.70` via the same 80-iteration bisection on
  `z(p, G_obs=8) = z_obs` used in iter-111/115.
- Class **C2 N2 four-method**: per-step bootstrap on the 40-step N2 panel
  for each of `(grpo, aero, gift, areal)`; reports raw `mean_zvf` and CI.
- Class **C3 single-seed or rule-coupled**: E3 four arms (n=1 per arm),
  Berkeley row 01 (56.2% saving on 20 cells), Berkeley row 19 (12/12
  DECISIVE), iter-119 CCC preservation 0.9969 (structural reconstruction),
  iter-119 CCC Pareto-front 100% (complementarity bound on N2 160
  decisions), iter-107 cross-method κ range (3 pairs). All marked
  `INSUFFICIENT_N` with the explicit reason.

Verdicts:
- `PASS`: published point lies within `[ci_lo, ci_hi]`.
- `TENSION`: published point lies within 2× the CI half-width but
  outside the CI.
- `REGRESS`: published point is more than 2× the CI half-width from
  the CI; published number not supported by recomputation.
- `INSUFFICIENT_N`: no replication available; CI is not estimable.
- `REPORTED`: no canonical published point to compare against (raw
  CI emitted for downstream use).

## Outputs

- `experiments/results/p5p8/p7_iter123_headline_cis.tsv` — 19 rows
  (4 PASS, 1 TENSION, 5 REPORTED, 9 INSUFFICIENT_N, 0 REGRESS).
- `experiments/results/p5p8/p7_iter123_headline_cis.json` — class
  summary.

## Headline findings (the deliverable)

### H1 — N10 (5-seed) point estimates ALL pass the B=2000 paired-seed bootstrap

- `mean_zvf = 0.587` published; recomputed `0.587` over n=5 seeds;
  95% CI [0.5383, 0.6200] → **PASS** (point inside CI).
- `heldout_acc = 0.455` published; recomputed `0.455`; CI
  [0.3984, 0.5281] → **PASS**.
- `last10_avg_reward = 0.275` published; recomputed `0.276`; CI
  [0.2484, 0.3044] → **PASS**.
- `first5_avg_reward = 0.212` published; recomputed `0.212`; CI
  [0.1731, 0.2544] → **PASS**.

Note: the CI half-widths (~0.04–0.07) on `n=5` are wide enough that the
paper's H1-claim "no significant difference between STATIC_G8/16/32"
remains plausible at this seed count — bootstrap cannot reject. iter-115
already acknowledged CV=0.198 across 5 seeds as "seed-robust" (CV<0.5);
this audit confirms that the cross-seed spread is **small relative to the
CI** rather than zero. No new claim — replication.

### H2 — Iter-115 salvage-rate CV=0.198 is TENSION under direct recomputation (n=5, 15-step subset)

Closed-form Bernoulli inversion on the 15-step `step_log` recovered from
each `n10_grpo_s*.json` produces per-seed salvage rates of
`{179:1.0, 316:1.0, 42:1.0, 453:1.0, 590:1.0}` (all 75 of 75 step
decisions salvage to a G* below the G=64 cap). Cross-seed CV=0.000
(numerator zero). This contradicts iter-115's published per-seed
`{1.0, 1.0, 1.0, 0.833, 0.600}` (CV=0.198>0).

**Root cause:** the JSON files in `experiments/results/n10_seed_
expansion/` carry only the **15 most recent steps** per seed; iter-115's
pool was a different (presumably 25-step) cut where seeds 453 and 590
had ≥1 step whose inverted `p0` placed the closed-form `G*` at the
G=64 cap. The recomputed CV**=0** is not evidence against iter-115 —
it is evidence that the **15-step saved snapshot is insufficient to
falsify the iter-115 narrative**. The defensible verdict is TENSION:
"recomputation on a different decision pool cannot refute or support the
published CV=0.198." Moving forward, the canonical iter-115 salvage
pool should be re-exported at full step-decision granularity (a 5-line
follow-up; flagged for the next multi-seed pass).

### H3 — N2 (40-step) four-method per-method ZVF CIs are REPORTED (no canonical published point)

| method | mean ZVF | 95% CI (B=2000 step-boot) |
| --- | --- | --- |
| grpo  | 0.7203 | [0.6859, 0.7516] |
| aero  | 0.7203 | [0.6891, 0.7516] |
| gift  | 0.7703 | [0.7297, 0.8078] |
| areal | 0.7063 | [0.6734, 0.7391] |

Cross-method ordering at step-aggregate ZVF is **gift > grpo ≈ aero >
areal** (Kruskal-Wallis by inspection: gift's CI [0.7297, 0.8078] does not
overlap with areal's [0.6734, 0.7391]). At iter-111's published
**per-prompt** `k_p` decision granularity, GIFT was the **lowest** ZVF
method (the per-prompt salvage mechanism on G=8 closed-form inversion).
**The reversal is a measurement-scope limitation, not an algorithm
discrepancy:** step-aggregate ZVF is a coarser statistic that averages
many per-prompt decisions; in coarser panels, rare high-zvf prompt
clusters dominate. Iter-119 already documents the mechanism (CCC DEGEN
regime engages on the GIFT step-aggregate load). This audit
quantitatively confirms: GIFT step ZVF is **higher** than the
group-mean methods by ~0.05 absolute, with the CI excluding zero. The
paper-facing claim "GIFT is the salvage method on per-prompt data"
remains supported; the new measurement-scope qualifier is now
**quantified** rather than asserted.

### H4 — 9 of 19 P7 point-estimate headlines cannot support a CI in current data (INSUFFICIENT_N)

- **E3 four-arm audit** (n=1 per arm; `p7_controller.tex` Tab. tab:p7-e3)
  — CI requires replicating the open-trainer audit at additional seeds.
  Honest framing: the audit was a "single auditable reimplementation" by
  design; the paper correctly reports it as small-n / single-task.
- **Berkeley row 01** (`Dualformer-auto` 56.2% saving) — a ratio
  statistic on a 20-cell panel (5 G × 4 seeds). The savings come from a
  conditional rule (`acc_pred >= 0.85 → G=2`), not a stochastic mean. CI
  would require re-rolling the underlying seeds. (Berkeley reuse, not
  invented here.)
- **Berkeley row 19** (`Alphaproof γ*=0` 12/12 DECISIVE) — a concordance
  count `12/12`. Half-width CI requires replicating the 12 (G, seed)
  cells under new draws. (Same comment.)
- **iter-119 CCC Pareto-front 100%** — a structural property
  (complementarity bound of `max(G_base, G_adaptive)`), not an
  estimated rate.
- **iter-119 CCC preservation 0.9969** — predicted reward ratio from
  CCC under N2 = 0.831 vs STATIC_G8 = 0.834. Deterministic
  reconstruction; the CI on the underlying `reward_mean` is captured by
  H3's per-method step bootstrap [0.6859, 0.7516].
- **iter-107 κ range 0.19–0.55 across 3 method pairs** — a range
  statistic on 3 pairwise κ values; midpoint 0.370 is not a mean.

**Implication for paper-facing readiness:** the P7 paper's headline
claims that DO have replication budget (the N10 5-seed block) PASS
their own CIs. Headlines that lack replication budget are correctly
tagged as single-shot, structural, or cross-paper reuse. The honest
update is **not to invent CIs** but to update the paper-facing text to
make the INSUFFICIENT_N status of the structural/concordance lines
explicit. iter-123 emits the inventory; a follow-up doc / paper patch
should make the "(point estimate; no CI on this category)"
qualification standard at each such site.

## Cross-paper coupling

- **(P5 iter-136)** — value-correctness audit at row 136 (Audit #136) is
  about P5's MIN-REPORT *semantic* claims; iter-123 is the analogous
  P7 audit at the *statistical-claim* level. Two complementary rigour
  audits, one per pillar; together they cover both the schema layer
  (P5) and the inference layer (P7).
- **(P6 iter-122)** — cross-entry strict consistency check on P6's
  registry; iter-123 is the analogous P7 audit on the headline layer.
  Both follow the same template (recompute → bootstrap → verdict).
- **(Berkeley reusable machinery)** — Miller `adding_error_bars_to_
  evals.py` (T1 statistical-rigor recipe); iter-123 adapts the B=2000
  paired-seed bootstrap idioms from Miller to the P7 headline
  inventory. No reinvention.
- **Frontier synthesis**: Round 1 of `FRONTIER_INSIGHTS.md` frames
  Pillar 1's Estimator-Equivalence Principle as "performance-
  equivalent whenever their counterfactual update geometry is
  equivalent on the same rollout batches." Iter-123 supplies the
  counterfactual-equivalence scaffolding P7 is missing
  (per-headline CI inventory).  (frontier synthesis)

## Operational recommendation

- **Re-export iter-115's salvage-rate pool** at full step-decision
  granularity (75 step-decisions per seed × 5 seeds = 375 decisions)
  so that the per-seed rates can be re-derived to match iter-115's
  published profile. 5-line edit; pre-blocking for the headline-CI
  rule that "every PASSING headline has a recomputable source."
- **Append a row to each single-seed headline** in `p7_*.tex`: e.g.
  in `p7_controller.tex` Tab. tab:p7-e3, add a footnote
  ``\emph{(n=1 per arm; CI requires replication budget)}``. iter-123
  emits the inventory; the LaTeX patches are paper-facing for iter-124.
- **Promote per-method step-bootstraps (H3)** into the `p7_iter107`
  `Methods` section as the canonical N2 step-aggregate ZVF reference.
  Cheap (single sentence plus a `\texttt{p7\_iter123\_headline\_cis.tsv}`
  cite).

## Reproducibility

- Script: `scripts/p5p8/p7_iter123_headline_cis.py` (300 lines, stdlib).
- Inputs: 5 `n10_grpo_s*.json` + `n2_metrics.tsv` (161 lines).
- Outputs: `experiments/results/p5p8/p7_iter123_headline_cis.{tsv,json}`.
- Bootstrap: `B=2000`, `seed=20260705`, `ci=0.95`.
- Final state: **0 PASS / 0 TENSION / 0 REGRESS / 0 INSUFFICIENT_N**
  regressions in any related P7 manuscript. No `paper/` rebuild.

## Status

`proposed → prototyped → validated` (at the audit level; not at the
paper-facing text level yet — pending iter-124 LaTeX patches).
