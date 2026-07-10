# 56 — P6 / Pillar 2 Measured-vs-Claimed Validation of the GRPO-Registry

**Vein (from iter-46 brief)**: (a) — validate existing registry entries
against measured behavior. The N2 same-stack tensors give measured
GRPO/AERO/GIFT/AREAL deltas (iter-34), and the zvf_iter130 5-seed risk
index covers 9 methods (iter-34 also). This iter links those two
artifacts to a new registry block: `expected_effects` (human-supplied
forward reference of the change text's predicted sign) and
`claim_validation` (machine-audited per-(metric, panel) verdict
table).

## Method

Two additive schema blocks land in this iter:

  1. **`expected_effects[]`** — a list of `{metric, panel,
     predicted_sign, rationale}` tuples. The `predicted_sign`
     operators are constrained to `>0 | <0 | >=0 | <=0 | =0`. The
     rationale is a one-sentence clause that ties the prediction to
     the change text. Entries that don't have this block are still
     audited; rows for which no prediction is declared are
     `UNCLAIMED`.

  2. **`claim_validation[]`** — machine-generated, one row per
     `measured[]` row: `{metric, panel, predicted_sign,
     observed_delta, ci_low, ci_high, significant, verdict,
     rationale}`. The `verdict` enum is fixed:
     `SUPPORTS | NEUTRAL | CONTRADICTS | UNCLAIMED`.

The verifier:

  - SUPPORTS    : measured CI excludes 0 AND observed sign matches
                  predicted sign.
  - NEUTRAL     : measured CI includes 0 (cannot falsify yet), OR
                  the observed sign matches the prediction but the
                  CI still includes 0.
  - CONTRADICTS : measured CI excludes 0 AND observed sign is
                  OPPOSITE to the predicted sign.
  - UNCLAIMED   : no `expected_effect` declared for this (metric,
                  panel) pair.

The verifier is exactly deterministic — every `delta_*.json` is
written, re-validated against the schema (must still pass), and
the full registry is then re-validated end-to-end.

## Headline (falsifiable)

```
22 measured rows across 8 variant_delta records (aero, gift, areal,
cppo, ngrpo, mcgrpo, es, scafgrpo); 11 variant-delta records total
(the 3 records not audited are delta_dapo / delta_drgrpo /
delta_gspo, which have no measured block yet).

  SUPPORTS     9  (40.9%)   -- 1 in 8 variant rows aligns with the
                              change text's predicted sign
  NEUTRAL      3  (13.6%)   -- CI includes 0
  CONTRADICTS  2  ( 9.1%)   -- 2 of 8 contradict the change text
  UNCLAIMED    8  (36.4%)   -- mean_zvf (point estimate only, no
                              per-seed SD) on the zvf130 panel

significant-share (SUPPORTS+CONTRADICTS) / total = 0.500

Registry re-validation: 31/31 entries pass, 0 fail.
```

The interesting finding: **2 of the 8 same-stack method rows
CONTRADICT the change text's directional claim**. Both AERO and
AREAL — on the N2 same-stack last-10 panel — produce a
significantly LOWER mean reward than vanilla GRPO (delta =
-0.0141 and -0.0195 respectively, CIs both strictly negative).
The change text for AERO claims "no reason to expect a reward
loss" (predicted_sign `>=0`); for AREAL it claims "at least
reward-neutral". The measurement refutes both.

This is itself the audit the benchmark needs: **AERO and AREAL's
off-policy/autoscaler tricks that lower ZVF risk also measurably
cost reward on the same stack** — a paper-reviewer-fatal drift
that the registry now surfaces explicitly via the
`claim_validation` block.

## Per-delta verdict table

| delta | SUPPORTS | NEUTRAL | CONTRADICTS | UNCLAIMED |
|---|---|---|---|---|
| delta_aero    | 1 | 1 | 1 | 1 |
| delta_areal   | 1 | 1 | 1 | 1 |
| delta_gift    | 2 | 1 | 0 | 1 |
| delta_cppo    | 1 | 0 | 0 | 1 |
| delta_ngrpo   | 1 | 0 | 0 | 1 |
| delta_mcgrpo  | 1 | 0 | 0 | 1 |
| delta_es      | 1 | 0 | 0 | 1 |
| delta_scafgrpo| 1 | 0 | 0 | 1 |

8 deltas SUPPORTS `zvf_risk_mean < 0` on the 5-seed zvf130 panel
(all 8 reduce risk vs GRPO, CI strictly negative); this is the
strongest convergent finding — **all 8 GRPO-family variants in
the worktree reduce the ZVF-risk metric by a significant margin**.
The 3 N2 same-stack methods show: AERO/AREAL trade reward for
ZVF, GIFT trades ZVF for reward, NONE trades both.

## Why this matters for P6

The iter-34 measured block added WHAT was measured; this iter
adds whether the measurement SUPPORTS the change text's
directional claim. Without `expected_effects`, the measured
block is descriptive ("here is the delta"); with it, the block
becomes a controlled falsification framework. A reviewer can
now ask: "Does the registry entry's claim survive a same-stack
test?" — and the answer is machine-readable per (delta,
component, metric, panel).

The CONTRADICTS verdicts for AERO and AREAL are the strongest
P6 deliverable: the registry now contains, for the first time,
**measured effect that is the opposite sign of the
human-readable claim**, with full CI provenance. This is exactly
the kind of self-auditing surface a benchmark paper should ship
rather than wait for a reviewer to discover.

## Artifacts

- `platform_modal/scripts/p5p8/p6_measured_vs_claim.py` (~280 LoC, stdlib + jsonschema)
- `platform_hybrid/experiments/results/p5p8/p6_measured_vs_claim.tsv` (22 rows)
- `platform_hybrid/experiments/results/p5p8/p6_measured_vs_claim_summary.json`
- 8 `delta_*.json` entries extended with `expected_effects` +
  `claim_validation` blocks
- `platform_hybrid/registry/schema.json` updated with two new `$defs`:
  `expected_effect` and `claim_validation_row`
- `platform_hybrid/registry/query.py` extended with `claim-validation` subcommand
- `platform_hybrid/registry/README.md` updated with the new quick-start snippet

## Cross-paper coupling

- **Same evidence base as P5** (the N2 same-stack last-10 panel
  feeds both the iter-45 stack-conditioning eta^2 and this iter's
  measured-vs-claim audit). P5 says "stack, not label"; P6 says
  "and the measured deltas on the same stack refuted 2 of 8
  direction-claims". The two are complementary, not redundant.
- **Same evidence base as P7** (the zvf130 5-seed risk index is
  the input to P7's controller signal calibration on iter-31
  and iter-43). P7's controller is justified by the fact that
  **all 8 GRPO-family variants reduce zvf_risk_mean on the
  5-seed panel** — this iter now exposes that as a per-delta
  claim_validation row.
- **Berkeley unpacking recipe**: this iter uses the same
  pair-of-blocks pattern as `platform_modal/scripts/berkeley/eval_protocol_hardening.py`:
  declare an expected effect, then machine-validate the measurement
  against it. The 2 CONTRADICTS verdicts are exactly the
  "theory-predicts-but-data-refutes" rows the iter-26
  BERKELEY_IMPROVEMENTS audit said we should be catching.

## Limitations and next iter (parked)

- **DAPO/Dr.GRPO/GSPO have no measured block yet**: 3 of the 11
  variant-delta records (delta_dapo, delta_drgrpo, delta_gspo)
  are present in the registry but carry no `measured[]` block —
  they have either no same-stack panel or no zvf130 row. Adding
  a measured block for these is the natural precondition for a
  full 11/11 audit.
- **`mean_zvf` is UNCLAIMED on all 8 rows** because the iter-34
  measured block uses a point estimate (no per-seed SD). A
  per-seed recomputation that stores `mag_mean_sd` would let us
  fill these 8 rows; this is exactly the kind of "compute
  disproportionate to evidence" gating the iter-32 #43 reject
  applies, so we record it parked rather than drive it.
- **The expected_effect seeds are text-derived, not citation-
  extracted**: AERO's `>=0` for `reward_mean` is a conservative
  directional reading of "no reason to expect a reward loss".
  A future iter could replace the manual seed with a
  per-(method, paper-clause) extraction from the source PDFs.
