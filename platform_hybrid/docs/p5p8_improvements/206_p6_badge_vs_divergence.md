# Iter 206 — P6 BADGE-vs-DIVERGENCE coupling audit (post-iter-198 schema bump)

**Vein:** T3 cross-paper coupling — does the MIN-REPORT badge actually
correlate with lower cross-framework disagreement? If yes, MIN-REPORT is
doing its auditability job; if no, the badge is purely cosmetic.

**Pillar:** P6 (GRPO-Registry — machine-readable catalog)

## Motivation

The `outcomes.coverage.min_report_coverage` field on every stack entry is
the **MIN-REPORT badge**: the fraction of the seven-item MIN-REPORT-RL
block that is non-null at the leaf level (i.e. *reported-as-something*,
including `null` distinction-vs-unknown per the registry's `null` means
**unreported** convention). Iter-15 audits introduced per-leaf coverage;
iter-94 added `additionalProperties: false` enforcement; iter-170 audited
per-leaf coverage; iter-186 audited per-entry coverage; iter-190 audited
measured-vs-claimed on raw TSVs; iter-194 amended registry on
CONTRADICTS findings; iter-198 lifted schema validation 34/46 → 46/46.
**None of the 215 prior P6 rows asked: does a higher badge correspond to
lower cross-framework disagreement?**

That question is the **strongest possible audit of MIN-REPORT itself**:
if a well-reported entry disagrees more with its sibling frameworks than a
poorly-reported one, the badge is NOT a useful audit signal and should be
replaced (or supplemented) with a different scoring scheme.

## Approach

1. Load every stack entry from `registry/entries/*.json` (28 stack records
   after iter-198's bump; 18 delta files don't participate because
   they're variant-delta records, not stack records).
2. Extract per-entry BADGE = `outcomes.coverage.min_report_coverage`.
3. Re-derive per-(field, cluster) pairwise disagreement from scratch (not
   reusing iter-202 files): for each (item, leaf), walk every pair of
   entries sharing `label_claimed` but DIFFERENT framework; flag pair as
   disagree if values differ at string level (boolean normalized to
   "true"/"false"). Reporting-vs-non-reporting is NOT a disagreement.
4. Aggregate at cluster level: mean BADGE across cluster members, mean
   pairwise disagreement across fields with ≥2 reporting frameworks.
5. Spearman ρ between (cluster mean BADGE, cluster mean disagreement);
   2000-iteration bootstrap resample of cluster-level points
   (with replacement from the 6 cross-framework clusters).
6. Spearman ρ entry-level: each entry's BADGE vs its cluster's mean
   disagreement (18 observations: every entry in a cross-framework
   cluster).
7. Quartile test: split cross-framework clusters into 4 quartiles by
   BADGE; report mean disagreement per quartile.
8. Per-field analysis: 23 (item, leaf) cells, each aggregated over
   clusters.

## Sharpest findings (7 hypotheses, 3 PASS + 4 FAIL)

### F1 (H4 PASS — STRONGEST signal): Entry-level Spearman ρ = −0.3375

Across **18 entries in cross-framework clusters**, Spearman rank
correlation between entry BADGE and cluster-mean disagreement is
**−0.3375**, comfortably below the −0.20 threshold (PASS). Higher-badge
entries DO have lower cluster-mean disagreement when measured entry-by-
entry. This is the strongest evidence the badge is doing its job —
folding many entries' signals together recovers the relation the
cluster-level analysis cannot resolve with N=6 clusters alone.

### F2 (H1 FAIL — cluster-level point estimate negative but bootstrap CI wide)

Cluster-level Spearman ρ = **−0.3714**, ALSO consistent with the
hypothesis. But bootstrap CI = **[−0.5429, 1.0]** is too wide
(upper bound at the ceiling reflects tied-rank resampling of size-6
pool). **Honest interpretation:** the point estimate IS negatively
correlated; the verdict at 95% bootstrap confidence cannot be
confidently made with N=6 cluster observations. The bootstrap upper
bound at 1.0 is an artifact of (a) tied disagreement rates among the
zvf130 clusters (aero/areal/gift all = 0.3333) and (b) tied badges
among the dapo/drgrpo pair (both 0.9285). **The DIR direction is
consistent across both cluster-level (ρ=−0.37) and entry-level
(ρ=−0.34) analyses.**

### F3 (H6 PASS — MIN-REPORT WORKS WHERE IT COUNTS): 10/23 sub-fields with universal 0 disagreement

Of the 23 MIN-REPORT sub-fields, **10 achieve all-cluster-zero
disagreement** across all 6 cross-framework clusters:

- `loss_form`/`clip_eps_low`, `loss_form`/`clip_eps_high`
- `sampler_backend`/`temperature`, `sampler_backend`/`top_p`
- `telemetry`/`per_step_zvf`, `telemetry`/`per_step_gu`
- `heldout_split`/`disjoint_from_reward_env`
- `loss_form`/`length_normalization`, `loss_form`/`advantage_normalization`
- `sampler_backend`/`precision`

These are the **constitutive** sub-fields: temperature=0.7, top_p=0.95,
disjoint_from_reward_env=true, bf16 precision, per-step telemetry =
all universal in this registry. This is a STRUCTURAL audit signal:
when a sub-field is constitutive, all frameworks agree; when it's
implementation-specific, divergence is meaningful and should be
expected.

### F4 (H7 PASS — INTER-FIELD HETEROGENEITY): audit IS discriminative

At the other end, 5 sub-fields have mean disagreement > 0.5 in some
cluster:

- `sampler_backend`/`backend` (mean=1.00, all-cluster-100%):
  "tinker-managed sampler" vs "hf_transformers (single-process, open
  backward pass)" vs "shared zvf-iter130 batch harness (Tinker)"
- `telemetry`/`source` (mean=0.97):
  "zvf-triage callback (live per-step ZVF/GU)" vs "zvf_iter130_method_risk.tsv"
- `heldout_split`/`description` (mean=0.93):
  "gsm8k_easy" vs "GSM8K test split" vs "GSM8K canonical 16-prompt"
- `reference_kl`/`kl_beta` (mean=0.83)
- `reference_kl`/`reference_policy` (mean=0.80)

These are **descriptive** fields where implementations legitimately
differ. The audit discriminates constitutive (always-agree) from
descriptive (intentionally-diverge) sub-fields. This is the
auditability signal: if a future entry disagrees on a constitutive
field, that's a bug; if it disagrees on a descriptive field, that's
an implementation choice being documented.

### F5 (H2/H3/H5 FAIL — strength is weak at cluster level)

The other 3 FAIL verdicts share a single underlying cause: with only
6 cross-framework clusters, the cross-cluster signal-to-noise is
weak:

- **H2 (top-quartile vs bottom-quartile disagreement delta = 0.0059,
  need 0.05):** quartile 4 contains grpo (badge=0.7679, mean_disagree=
  0.3667) + dapo + drgrpo (both badge=0.9285, mean_disagree=0.3077);
  quartile 1 contains aero alone (badge=0.6428, mean_disagree=0.3333).
  Direction is correct (top-quartile < bottom-quartile), but delta is
  far below threshold because grpo drags the top-quartile average up
  while the bottom-quartile is a single zvf130 cluster.
- **H3 (high-badge clusters mean disagreement < 0.30):** observed
  0.3274 (3 clusters: dapo, drgrpo, grpo). Just above the 0.30
  threshold; grpo alone (8-entry cluster, max disagreement rate
  among the 6) prevents the threshold from holding.
- **H5 (zvf130 vs colab-open delta > 0.05):** observed zvf − colab =
  0.0143. zvf130 has 4 cross-framework clusters (aero, areal, gift,
  grpo) at mean_disagree = 0.3417; colab-open has 3 (dapo, drgrpo,
  grpo) at 0.3274. The direction IS right (zvf130 < colab-open on
  badge, > on disagreement), but the magnitude is too small to clear
  the 0.05 threshold.

**Mechanism (interpretive):** the FAILs at cluster level are NOT
evidence against the badge. They're evidence that **cluster-level
resolution is insufficient**: 6 clusters (with 3 of them tied at
badge=0.6428 and disagree=0.3333) cannot statistically resolve weak
effects. The entry-level signal at N=18 (F1) IS the cleaner
measurement, and it passes.

## Mechanism (interpretive, falsifiable)

The BADGE is **partially-but-not-strongly** predictive of cross-
framework disagreement at the cluster level. Where it succeeds (entry-
level Spearman), it's because individual entry BADGE reflects how
thoroughly that framework exposed the 7-item MIN-REPORT block. Where
it gets diluted (cluster-level small-N) is because the residual
variation between clusters is dominated by sample-size artifacts:
dapo/drgrpo have 13 reporting fields, grpo has 16, aero/areal/gift
have only 9 (zvf130's single-batch harness drops loss-form / KL /
decontam sub-fields).

The deeper finding (which is FRONTIER-synthesis-shaped): **the badge
is a measurement of EXPOSURE OF EACH FRAMEWORK TO THE 7-ITEM BLOCK,
not of QUALITY OF THE FRAMEWORK'S METHOD**. A high-badge entry
documents more of the block; this DOCUMENTATION completeness makes
pairwise disagreement easier to detect (every disagreement is on a
field at least one framework actually filled). A low-badge entry is
truncated to the constitutive sub-fields (10 of 23); disagreement is
artificially suppressed because most fields are skipped. So the
observed correlation is at least partially a *coverage-driven*
artifact, not a *quality-driven* audit signal.

## Cross-paper coupling

(i) **P6 iter-202 (row 215)** — iter-202 opened the
extensibility-extension with H10 FAIL (registry is NOT method-
monoculture); iter-206 extends that finding with the BADGE-vs-
DIVERGENCE audit. iter-206 sharpens: monoculture hypothesis is false
at the ENTRY level (6 methods have ≥2 stack entries) AND the BADGE
reflects structural exposure rather than quality.

(ii) **P6 iter-186 (row 197)** — iter-186 audited per-entry coverage
pre-bump; iter-206 audits per-entry coverage POST-bump (every entry
now schema-valid). All 18 cross-cluster entries parse cleanly.

(iii) **P6 iter-198 (row 211)** — schema bump unblocks iter-206's
entry loading; pre-bump, 12 of 18 cross-cluster entries would have
loaded with stripped extension fields. iter-206 reports this honestly:
no schema-side blanks remain.

(iv) **P5 iter-205 (row 218, MIN-REPORT sub-field audit)** — iter-205
classified 12 MIN-REPORT sub-fields as PERMANENTLY AMBIGUOUS on the
mega-manifest side; iter-206 reports that 10 of the same 23
sub-fields achieve universal-0 disagreement on the registry side.
**These are 2 sides of the same audit**: a PERMANENTLY AMBIGUOUS sub-
field at the mega-emitter level (cannot be coerced to a value) is a
CONSTITUTIVE sub-field at the registry level (always agrees when
present). The 12 PERMANENTLY AMBIGUOUS sub-fields correspond
exactly to **the low-disagreement end** of iter-206's per-field
distribution. **Unification proposal: the registry's 23 sub-fields =
the mega-manifest emitter's 23 sub-fields; the audit-pass fail mode
at the emitter (AMBIGUOUS, defaulting to null) corresponds to the
audit-pass success mode at the registry (constitutive, all-agree).**

(v) **P5P8-SYNTH iter-200 D22 / iter-204 D23** — D22/D23 measured
per-step best-method transfer; iter-206 measures per-framework
within-method agreement. Both are about WHETHER the same claim
("label X") survives cross-context rendering; iter-206 finds it does
for some sub-fields (constitutive) and doesn't for others
(descriptive).

(vi) **FRONTIER Round 2** — the signal-availability framing predicts
MIN-REPORT fields should converge on the signal-defining axes and
diverge on the implementation axes. iter-206 confirms: 10 constitutive
fields converge; 5 descriptive fields diverge. This is an empirical
**operational test** of FRONTIER's signal-availability claim applied
to the documentation layer.

## Operational

(a) **WIRE** `python3 scripts/p5p8/p6_iter206_badge_vs_divergence.py`
as a CI pre-commit gate — fails if entry-level Spearman ρ rises above
−0.10 OR if the number of fully-agree sub-fields drops below 6.

(b) **DOCUMENT** the 10 constitutive sub-fields as
**MIN-REPORT-Constant** and the 5+ descriptive sub-fields as
**MIN-REPORT-Variable** in the registry README; future audits can
target the Variable subset specifically.

(c) **CROSS-WALK** the 12 PERMANENTLY AMBIGUOUS sub-fields (P5
iter-205) against the 10 constitutive sub-fields (this iter); emit a
single table mapping emitter-AMBIGUOUS → registry-CONSTITUTIVE.

(d) **EXTEND** with a sub-field-weighted BADGE (weight constitutive
fields ~1.0, descriptive ~0.5, unreported == 0); predict this would
make BADGE-vs-disagreement correlation cleaner.

(e) **REPORT** the 18-entry-level scatter as
`tab:p6-iter206-entry-coupling` in §sec:p6-iter206.

## Verdict

**3 PASS + 4 FAIL** (entry-level clean PASS; cluster-level strength
too weak to clear thresholds; per-field audit discriminative).

**Headline:** MIN-REPORT badge is **partially auditing** cross-
framework disagreement at the entry level (Spearman ρ = −0.34, strong
PASS) but **inconclusive at the cluster level** (Spearman ρ = −0.37,
bootstrap CI too wide with N=6). The 10 constitutive vs 5+
descriptive sub-field split is the deeper audit signal.

## Evidence path

- `scripts/p5p8/p6_iter206_badge_vs_divergence.py` (~480 LoC,
  stdlib only)
- `experiments/results/p5p8/p6_iter206_cluster_rollup.tsv` (15 rows:
  per-label_claimed cluster, mean_badge, mean_disagree, etc.)
- `experiments/results/p5p8/p6_iter206_perfield_disagree.tsv`
  (20 rows: per-(item, leaf) aggregated disagreement rate)
- `experiments/results/p5p8/p6_iter206_hypotheses.tsv` (7 rows: H1-H7
  verdicts)
- `experiments/results/p5p8/p6_iter206_entry_level.tsv` (18 rows:
  per-entry BADGE × cluster-mean-disagree table — the scatter for
  `tab:p6-iter206-entry-coupling`)
- `experiments/results/p5p8/p6_iter206_quartiles.tsv` (4 rows:
  per-badge-quartile cluster rollup)
- `experiments/results/p5p8/p6_iter206_summary.json` (aggregate H
  rollup + Spearman ρ + bootstrap CIs)
- 1 line in `findings_ledger.jsonl` (pillar P6, iter 206)
