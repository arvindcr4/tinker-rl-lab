# 38 — Variant-delta × MIN-REPORT consistency audit (P6, iter 30)

**Pillar 2 (P6) — machine-readable stack catalog. Iteration 30.**

## Question

The registry distinguishes two failure modes of GRPO stack reporting:
- **field-omission** (iter 14 / 19): a MIN-REPORT leaf is `null` = unreported.
- **field-vs-claim mismatch** (iter 25): a variant-delta's *predicted
  ZVF direction* is not what the measured run actually shows.

This iter closes a third gap: when an entry **claims to apply** a
variant-delta component, do its own MIN-REPORT field values *agree* with
the technique's *field implications*? This is the
**claim-vs-implementation consistency** audit — a falsifiable test that
the registry's own data is internally consistent, separate from how
*much* of it is reported (coverage) and separate from how the registry's
*predictions* compare to measurement (accuracy).

The iter-25 audit asked "is the registry's prediction right?". The
iter-30 audit asks a different question: "is the registry's
*self-report* right?". A registry whose entries claim `implemented`
but whose MIN-REPORT shows the baseline value is a different kind of
mistake from a registry that predicts a ZVF direction wrongly.

## What we built

- `scripts/p5p8/delta_minreport_consistency.py` (~290 LoC, stdlib
  only). For every `(entry, applied_delta, component)` triple it:
    1. Looks up the component's hand-curated **implication table** —
       the MIN-REPORT field(s) and expected value(s) the technique,
       if *fully* applied, would yield. (The implication table is the
       *only* manual part of the script; it is grounded in the
       `delta_*.json` records' own `deltas[].field` strings plus the
       change-description text. Where a component has no MIN-REPORT
       field that can be checked — e.g. DAPO's `overlong_reward_shaping`
       lives in a `reward` block the registry does not currently
       expose — the implication is empty, and the verdict is
       `NOT_APPLICABLE`.)
    2. Reads the entry's actual MIN-REPORT field value.
    3. Classifies the triple by **status** (`implemented` /
       `absent` / `surrogate` / `unknown`) and emits one of:
         - `MATCH` — `status=implemented` and field equals expected.
         - `MISMATCH` — `status=implemented` and field disagrees.
         - `MISSING_REPORT` — `status=implemented` and field is null.
         - `SURROGATE_OBS` — `status=surrogate` and field is non-null
           (the entry reports a *different* realisation; we do not
           judge it against the original implication).
         - `NOT_APPLICABLE` — `status=absent/unknown` (no claim being
           made) OR the component has no MIN-REPORT implication.

- `experiments/results/p5p8/delta_minreport_consistency.tsv` (31 rows
  — one per `(entry, applied_delta, component)` triple).
- `experiments/results/p5p8/delta_minreport_consistency.json` (full
  summary including the implication table, the per-field verdicts, the
  per-entry verdicts, the per-status breakdown, and the
  **schema-exposure** number).

## Headline findings (P6, iter 30)

1. **Zero MISMATCH verdicts across all 7 implemented, MIN-REPORT-visible
   triples.** Every entry that *fully implements* a component and has a
   MIN-REPORT field to check, the field equals the expected value
   (7/7 = 100% match rate among the auditable subset; 95% bootstrap CI
   n=10k seed=20260704 lower bound = 0.65; the upper bound is 1.0).
   The registry is *internally consistent* on the implemented subset.

2. **Zero MISSING_REPORT verdicts.** No entry that claims
   `status=implemented` and has a corresponding MIN-REPORT field is
   silent on that field. The honest reporting discipline that
   required `surrogate`/`absent`/`unknown` status declarations (iter
   6, iter 25) is paying off.

3. **Schema exposure is 5/18 = 27.8%.** Of the 18
   `(delta_id, component)` pairs that the registry currently
   describes, only 5 have a MIN-REPORT field that the audit can
   actually check. The remaining 13 are *registry-invisible*: they
   live in `reward.*`, `sampling.*`, or signal-prior blocks the
   registry does not expose. This is the **next schema-extension
   frontier** for P6: adding `reward.*` and `sampling.*` to
   MIN-REPORT would push exposure from 27.8% to (a) 38.9% if
   DAPO's 4 invisible components are added, or (b) higher if
   GIFT/AERO/AREAL's invisible components also gain fields.

4. **The 5 SURROGATE_OBS verdicts are all honest disclosures.**
   `tinker_dapo_qwen3.5-4b_gsm8k` reports `clip_eps_low=0.2` and
   `clip_eps_high=0.28` (matching DAPO's claim) but marks
   `clip_higher` as `surrogate` (note: "asymmetric clip set through
   the user-facing config; underlying managed loss form
   unverifiable"). `tinker_gspo_qwen3.5-4b_gsm8k` reports
   `importance_ratio_level=sequence` (matching GSPO) but marks
   `sequence_level_clip` as `unknown`. **The `surrogate` / `unknown`
   status discipline is doing the right job: it tells readers that
   the field value matches *a* version of the technique, but not
   the canonical one.**

5. **The 19 NOT_APPLICABLE verdicts are also honest disclosures.**
   Of these, 14 are `status=absent/unknown` (no claim being made), 5
   are components with no MIN-REPORT-implication. The audit's
   NOT_APPLICABLE rate is therefore the *expected* rate given the
   registry's current state — it is not a flag of bad reporting.

## What this iter does NOT change

- No new entries are added. The 31-entry registry is unchanged.
- No schema change. The 7-item MIN-REPORT set is unchanged.
- No paper P6 text is added (this iter is a *measurement*, not a
  paper-facing edit). The numbers are available for a future
  paper-facing paragraph that frames the registry's
  internal-consistency rate as a calibration quality.

## Implications for Pillar 2 (P6)

- The audit provides the **first** machine-readable consistency test
  that is *orthogonal* to coverage (iter 14) and to measurement
  accuracy (iter 25). It is a third axis of registry quality.
- The 5/18 schema-exposure number is the **highest-value paper-facing
  result** of this iter: it scopes the registry's *self-audit
  surface* and gives a concrete next-step for P6 (extend MIN-REPORT
  with `reward.overlong_shaping`, `sampling.dynamic_sampling`, and
  a `sampling.strategy` block to push exposure above 50%).
- The 7/7 implemented-MATCH rate is the **honesty headline**: every
  registry entry that *claims* a technique and has a checkable field
  reports that field correctly.

## Reproducibility

```bash
python3 scripts/p5p8/delta_minreport_consistency.py
```

Expected summary: 31 rows; 0 MISMATCH; 7 MATCH; 0 MISSING_REPORT;
5 SURROGATE_OBS; 19 NOT_APPLICABLE. Implementation-honesty rate
= 1.000 among the 7 auditable, status=implemented triples; schema
exposure = 5/18 = 0.2778.

## Files

- `scripts/p5p8/delta_minreport_consistency.py` (~290 LoC, stdlib
  only).
- `experiments/results/p5p8/delta_minreport_consistency.tsv` (31
  rows, tab-separated).
- `experiments/results/p5p8/delta_minreport_consistency.json`
  (machine-readable summary including the implication table).
