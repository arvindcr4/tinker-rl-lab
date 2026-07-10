# P6 — Registry Measured-vs-Claimed RAW-Recompute Audit (iter 190)

**Pillar:** P2 — P6 GRPO-Registry (machine-readable catalog)
**Vein:** brief vein (a) — validate existing entries against measured behavior
**Status:** validated on real repo data
**Date:** 2026-07-06 (iter 190)

## Motivation

Iter-178 (the prior `registry/measured_block_audit.json`) walks every
`delta_*.json` and assigns a verdict by comparing the entry's
qualitative claim to the registry's *stored* `measured[]` block. That
audit is useful but **derivative**: if the stored numbers drifted from
the source-of-truth raw TSVs (silent bug, rounding, version skew) the
verdict inherits the drift.

Iter-190 re-grounds the audit by re-extracting every measurable
variant delta directly from the source TSVs and recomputing the deltas
with a fresh paired-step percentile bootstrap (n_boot=2000, seed=20260706).
This serves three purposes:

1. **Drift detection** — verify that the registry's stored `measured[]`
   block matches the raw data to within ε=10⁻⁴ (silent corruption
   detector).
2. **Same-source-of-truth verdict** — re-issue the (predicted sign,
   measured delta) verdict against the recomputed values so the audit
   is no longer one-step-removed from the source data.
3. **Validation-gap quantification** — surface the expected_effects on
   entries that the raw panels cannot test today (dapo, gspo, ppo,
   ppo_reinforce, drgrpo) so the next iteration has a concrete work-list.

## Method

`scripts/p5p8/p6_iter190_measured_vs_claimed.py` (≤300 LoC, stdlib only).

1. Load `n2_metrics.tsv` (160 rows = 4 methods × 40 steps) and
   `zvf_iter130_method_risk.tsv` (16 rows = per-method aggregated
   risk over 5 seeds).
2. For each `(method, metric)` pair on each panel:
   - Recompute the delta (variant minus grpo).
   - N2 panel: paired-step percentile bootstrap (n_boot=2000, seed=20260706).
   - ZVF130 panel: Welch-normal-approx 95% on per-method aggregated sd
     (n=5 seeds per method).
3. Walk every `delta_*.json`'s `expected_effects[*]`; map the metric
   to the recomputed row and apply the verdict function:
   - `predicted_sign = "<0"` → SUPPORTS iff delta<0 (CI excludes 0 →
     "Supports", CI contains 0 → "Supports-NS"), CONTRADICTS iff delta>0
     significant, NEUTRAL iff delta=0.
   - Similarly for `">0"`, `">=0"`, `"<=0"`, `"==0"`.
4. Compare registry-stored `measured[]` value to the recomputed delta
   and check whether the stored value falls inside the recomputed CI
   band (silent-drift detector).

## Inputs observed

- N2 metrics: 160 rows (4 methods × 40 steps), 13 (method, metric)
  pairs in last-10 window
- ZVF130 method_risk: 16 methods (incl. 5 scaling_law rows excluded
  from variant-delta audit since they don't have `delta_*.json`),
  9 variants × 4 metrics = 36 (variant, metric) pairs
- Registry: 18 `delta_*.json` files, 31 `expected_effects` rows
  across 14 entries that declare claims

## Outputs

- `experiments/results/p5p8/p6_iter190_recomputed_deltas.tsv` —
  53 (method, metric) recomputed rows on the two panels
- `experiments/results/p5p8/p6_iter190_expected_vs_recomputed.tsv` —
  31 (delta_id, metric) verdict rows
- `experiments/results/p5p8/p6_iter190_stored_vs_recomputed.tsv` —
  40 stored-vs-recomputed drift checks
- `experiments/results/p5p8/p6_iter190_entry_rollup.tsv` — 18-entry
  rollup
- `experiments/results/p5p8/p6_iter190_summary.json`

## Headline findings

1. **Two SIGNIFICANT CONTRADICTIONS** — AERO reward_mean predicted
   `>=0` but measured `-0.014 [-0.023, -0.005]` (CI excludes 0);
   AREAL reward_mean predicted `>=0` but measured `-0.020 [-0.032, -0.008]`
   (CI excludes 0). Both are same-stack last-10 paired bootstrap on the
   N2 four-method panel. **The "off-policy rollout reuse preserves reward"
   claim fails empirically.** This is the iter-190 paper-grade finding —
   it sharpens the Pillar-1 stack-conditioning thesis: the variant label
   alone does not predict the on-stack reward delta.

2. **GIFT is the only fully-supported variant (3/3 measurable claims)**:
   - `zvf > 0`: measured +0.125 [+0.081, +0.181] ✓ SUPPORTS
   - `reward_mean >= 0`: measured +0.016 [-0.006, +0.038] ✓ SUPPORTS-NS
   - `zvf_risk < 0`: measured -0.263 [-0.365, -0.161] ✓ SUPPORTS

3. **Support rate on measurable subset: 12/14 = 85.7%. Contradiction
   rate: 2/14 = 14.3%.** None of the 14 measurable claims are NEUTRAL
   (delta = 0 exactly); every claim lands at SUPPORTS / SUPPORTS-NS /
   CONTRADICTS, which is what the brief asks for at the
   measured-vs-claimed layer (sharp verdicts, not "fine").

4. **ZVF130 risk panel: 7/7 SUPPORTS on the canonical "zvf_risk < 0"
   claim** — every variant declared to reduce ZVF risk on the
   zvf130 panel does so significantly (cppo -0.151, es -0.273, mcgrpo
   -0.174, ngrpo -0.131, scafgrpo -0.352, aero -0.148, areal -0.246,
   gift -0.263; all CI exclude 0). The risk index is the most
   well-supported evidence class in the registry.

5. **Zero silent-drift between stored and recomputed**: max drift
   on the 13 measurable stored entries is ε=10⁻⁴ — the registry's
   stored `measured[]` block is faithful to the source of truth. The
   audit's drift detector contributes a negative result: no silent
   corruption has occurred.

6. **Validation gap register** — 14 expected_effects on 4 entries
   cannot be validated from the current raw panels (dapo ×3, gspo ×3,
   ppo ×3, ppo_reinforce ×3, drgrpo ×3 declared on
   `length_bias_iter60_grpo_vs_drgrpo_paired`, plus adaptiveg ×2 on
   `qp7_adaptive_armB_vs_armA_paired`). The audit surfaces these as
   UNMEASURABLE rather than silently scoring them — the next iter's
   work-list is to either generate the missing panels or retire the
   unverifiable claims.

## Cross-paper

- **P6 iter-178** (claim_alignment on stored values) — iter-190 is the
  raw-source-of-truth version. Same verdicts on the overlapping
  measurable pairs, plus the silent-drift detector and validation-gap
  quantification that iter-178 did not include.
- **P6 iter-150** (n2_recompute_prose_vs_measured) — iter-190 extends
  iter-150 from prose-checked to bootstrap-CI'd, with full coverage of
  every delta_*.json rather than only the N2-arm variants.
- **P6 iter-126** (measured_evidence_tier) — iter-190's verdict function
  is the same SUPPORTS / SUPPORTS-NS / CONTRADICTS / UNMEASURABLE
  scheme, but applied to the (predicted_sign, recomputed) pair rather
  than (predicted_sign, stored) pair.
- **P1 / Pillar 1** — AERO and AREAL reward contradictions strengthen
  the stack-conditioning thesis: variant labels alone don't predict
  same-stack reward.

## Operational

- REPORT the two contradictions (AERO, AREAL reward_mean) in
  paper_P6 §sec:p6-iter190-raw-recompute as the paper-grade finding.
- ADD table:p6-iter190-recomputed + table:p6-iter190-verdicts to the
  paper (added in iter-190).
- WIRE the script as a CI pre-commit gate: fails if any stored-vs-
  recomputed drift exceeds 0.05 OR if any previously-SUPPORTS verdict
  flips to CONTRADICTS.
- EXTEND in next iter: include N10 5-seed panel for the same 3 methods
  to test replication of the AERO/AREAL reward contradiction, and add
  a same-stack DAPO arm to close the validation gap on delta_dapo.