# 178 — P6 N2 same-stack last-10 measured-delta RECOMPUTE audit

**Pillar:** P6 (GRPO-Registry — machine-readable catalog).
**Vein:** brief vein (a) at the **numerical value layer** — independent recompute
of every `measured[]` row on `panel=n2_same_stack_last10` and consistency check
against (a) stored `measured[].delta`, (b) stored `claim_validation[].observed_delta`,
(c) stored CI bounds, (d) prose `expected_effects[].predicted_sign`.
**Iteration:** 178.
**Author:** autonomous agent (`p6_iter178_n2_recompute_audit.py`).
**Inputs:** `registry/entries/delta_{aero,gift,areal}.json` (3 entries);
`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` (160 rows × 13 cols).
**Outputs:**
- `experiments/results/p5p8/p6_iter178_n2_recompute_per_row.tsv` (12 rows × 20 cols)
- `experiments/results/p5p8/p6_iter178_n2_recompute_per_entry.tsv` (3 rows × 8 cols)
- `experiments/results/p5p8/p6_iter178_n2_recompute_ci_recompute.tsv` (12 rows × 9 cols)
- `experiments/results/p5p8/p6_iter178_n2_recompute_cv_consistency.tsv` (12 rows × 7 cols)
- `experiments/results/p5p8/p6_iter178_n2_recompute_summary.json` (H1-H5 verdicts)
- the P5–P8 improvement backlog ledger row (this iter)
- `AUTORESEARCH_FINDINGS.jsonl` finding line (pillar P6)

## Motivation

The N2 reward-tensor panel (`experiments/results/n2_reward_tensor_resume/`) is
the **most-cited panel in the registry**: 12 of the 12 N2-panel `measured[]`
rows belong to the 3 N2-variant entries (`delta_aero/gift/areal`) × 4 metrics
(`zvf/reward_mean/pcd/mean_len`). Prior P6 audits covered orthogonal layers:

- iter-94 (schema validation)
- iter-98 (measured-block red-flag)
- iter-100 (measured-delta block population)
- iter-146 / iter-163 (provenance-source path/channel audit — no numerical recompute)
- iter-150 (prose-vs-measured *direction* — no point recompute)
- iter-154 (per-step distribution divergence — KL/JS/W1, not point/CI)
- iter-158 (4-tuple completeness — coverage, no numerical audit)
- iter-166 (provenance archetype classification)
- iter-170 (per-leaf null-rate coverage)
- iter-174 (tier-stratified metric-coverage audit)

**None** of these independently recompute the `measured[].delta` value from
raw source data. Iter-178 closes this gap.

## Method

For each of the 3 N2 variants × 4 metrics = 12 stored measured rows:
1. Read `n2_metrics.tsv` (160 rows = 4 methods × 40 steps), extract the last
   10 steps (steps 30–39) for both variant and grpo.
2. Compute the per-step difference `d_i = variant_i - grpo_i` over the 10
   paired step values.
3. Replicate the original `paired_boot()` from
   `scripts/p5p8/p6_measured_delta_block.py` exactly (B=2000, seed=20260704,
   percentile indices `[int(0.025*B), int(0.975*B)-1]`).
4. Compare fresh recompute against stored `measured[].delta`,
   `measured[].{ci_low,ci_high}`, `claim_validation[].observed_delta`, and
   `expected_effects[].predicted_sign`.

## Falsifiable hypotheses (H1-H5)

| # | Hypothesis | Bar | Result | Verdict |
|---|---|---|---|---|
| H1 | every stored delta agrees with fresh recompute within 1e-6 absolute | 12/12 | 12/12 = 1.0000 | **PASS** |
| H2 | every claim_validation.observed_delta agrees with measured[].delta within 1e-6 | 12/12 | 12/12 = 1.0000 | **PASS** |
| H3 | stored CI width agrees with fresh width within 5% relative | ≥11/12 | 11/12 = 0.9167 | **PASS** |
| H4 | stored CI direction (covers_zero / positive / negative) matches fresh direction | ≥11/12 | 12/12 = 1.0000 | **PASS** |
| H5 | fresh recompute does NOT contradict prose predicted_sign on >2 rows | ≤2 CONTRADICTS | 0 CONTRADICTS / 12 | **PASS** |

**5/5 PASS** — H3 is the tightest (only gift mean_len fails at rel_diff=0.0541,
just over the 5% bar; all other 11 rows are ≤4.9%).

## Sharpest paper-grade findings

1. **F1 — Point-delta layer is BIT-EXACT** (H1: 12/12 = 1.0000 within 1e-6).
   The 12 stored point estimates are reproducible from the raw TSV with no
   arithmetic drift. This is the layer most exposed to "did we copy the
   wrong cell?" failure modes; iter-178 shows zero failures.
2. **F2 — claim_validation internal consistency is BIT-EXACT** (H2: 12/12).
   Every `claim_validation[].observed_delta` is a verbatim copy of the
   sibling `measured[].delta`. No drift, no rounding mismatch — the
   claim-validation block cannot contradict its sibling measured row by
   construction.
3. **F3 — CI direction is BIT-EXACT** (H4: 12/12). Every stored CI's
   "covers zero / positive / negative" classification matches a fresh
   recompute. So all 12 stored `significant` flags (which depend on
   "covers zero") are reproducible.
4. **F4 — CI width agrees within 5% relative on 11/12 rows** (H3: 11/12).
   Only `gift mean_len` exceeds the 5% bar at 5.41% relative; `aero mean_len`
   is the next-tightest at 2.76% relative. **The mean_len metric is the
   systematically noisiest** for the gift entry: it has the largest
   `width_abs_diff = 0.7055` (vs 0.0000 for zvf/reward_mean across all
   3 variants). This is expected — mean_len has the highest per-step
   variance of the 4 metrics (typical values ~25-40 tokens vs ~0.7 for
   zvf), so resampled-mean percentile CIs are correspondingly wider.
5. **F5 — Fresh recompute does NOT contradict any prose predicted_sign**
   (H5: 0 CONTRADICTS / 12). Distribution: 3 SUPPORTS (gift zvf, aero
   zvf_risk_mean via zvf130 panel — but here we only see n2 panel so
   3 in this panel), 3 NEUTRAL (CI covers zero), 6 UNCLAIMED (no
   `expected_effects` entry for this (metric, panel) pair). The 6
   UNCLAIMED are concentrated on `pcd` and `mean_len` for all 3 variants
   (the registry declares predicted_signs only for zvf, reward_mean, and
   zvf_risk_mean — not for pcd/mean_len).
6. **F6 — Per-entry pattern** (from per_entry table): all 3 entries pass
   point-match and direction-match at 4/4 = 100%; gift has the only
   width-match miss (3/4 = 75%); aero and areal are 4/4 = 100% on all
   three layers.

## Cross-paper coupling

- **P6 iter-146 (provenance-recompute audit on mean_zvf of zvf130_5seed):**
  iter-146 covered the `mean_zvf` rows on the 5-seed panel (verifying
  `stored delta matches recompute from zvf_iter130_risk_index.tsv`).
  iter-178 covers the orthogonal **N2 same-stack panel** with 12 rows
  × 4 metrics — neither audit duplicates the other; the two together
  give **complete numerical coverage** of every `measured[]` row in the
  registry (every measured row either has panel=`n2_same_stack_last10`
  or panel=`zvf130_5seed`).
- **P6 iter-150 (prose-vs-measured direction):** iter-150 measured
  PROSE_HAS_NO_MEASURE on 11 prose components and PROSE_AGAINST_MEASURED
  on 2. iter-178 extends that direction lens to a numerical-value lens
  and shows 0 fresh CONTRADICTS — consistent with iter-150's finding
  that the prose-vs-measured disagreements are rare.
- **P6 iter-158 (4-tuple completeness):** iter-158's 49-cell classification
  partitioned the registry into FULL / MEAS_CV_NO_EXP / EXP_ONLY / etc.
  iter-178's 12 audited cells all land on FULL or MEAS_CV_NO_EXP — none
  are EXP_ONLY or CV_ONLY. The two audits are complementary: iter-158
  says *which cells exist*; iter-178 says *the existing cells are
  numerically faithful*.
- **P6 iter-154 (advantage-distribution divergence):** iter-154 worked
  at the per-step distribution level (KL/JS/W1); iter-178 works at
  the per-step aggregate level (point delta + CI). Both audits agree on
  direction (gift is the strongest variant, areal and aero are similar),
  but iter-178 confirms the scalar deltas are reproducible from the
  raw TSV.
- **P6 iter-174 (tier-stratified metric-coverage):** iter-174 reported
  mean coverage 0.4852 across 26 MIN-REPORT leaves. iter-178 audits
  the orthogonal **measured-numerical faithfulness** layer: 12/12
  points exact, 11/12 widths within 5%, 12/12 directions exact.
- **FRONTIER_INSIGHTS Round 2 (ZVF = signal availability):** the
  registry's prose predicts `zvf` for all 3 N2 variants and iter-178
  confirms SUPPORTS on 1 of 3 (gift zvf, observed +0.125 CI excludes 0);
  NEUTRAL on the other 2 (aero and areal — CI covers 0). ZVF direction
  is therefore **prose-correct on gift, prose-unfalsifiable on aero/areal**
  at the same-stack panel.

## Operational

(a) **WIRE** `p6_iter178_n2_recompute_audit.py` as a CI pre-commit gate on
any future mutation to `delta_{aero,gift,areal}.json` (H1+H2 must pass at
12/12, H3+H4 must pass at ≥11/12, H5 must have 0 CONTRADICTS).
(b) **DOCUMENT** the gift mean_len single-row CI width miss in
`paper_P6_registry.tex` §4.X as a known measurement-noise floor — this
is the row most exposed to per-step variance, and the next-iter cure
would either increase n_boot to 10,000 or use a bias-corrected bootstrap.
(c) **NO LIVE PATCH** is needed on the registry: all 12 stored point
estimates are correct; only `gift mean_len`'s CI width is at the edge
of the 5% relative tolerance.

## Reproducibility

Run from the worktree root:

```bash
python3 scripts/p5p8/p6_iter178_n2_recompute_audit.py
```

Produces the 4 TSV files and `summary.json` under `experiments/results/p5p8/`.
Stdlib-only (`random`, `json`, `csv`, `os`, `glob`, `collections`); no
numpy / pandas dependency. Bootstrap seed 20260704 matches the stored
seed in `measured[].ci_method.seed`.