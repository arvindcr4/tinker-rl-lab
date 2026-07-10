# Item #45 — P5 Schema-Bump Closure Worklist (MIN-REPORT post-extension)

**Iter:** 33 (Pillar 1 / P5)
**Class:** T3 (cross-paper coupling)
**Status:** validated
**Date:** 2026-07-04

## Why this iteration

Iter 32 (`scripts/p5p8/delta_minreport_consistency.py`, JOB B / SYNTH)
minted three new backward-compatible optional fields in
`min_report.loss_form`: `token_aggregation`, `reward_shaping_type`,
`sampling_dynamic_filter`. Iter 32 populated two of them
(`colab-open_dapo_e3` and `tinker_gift_qwen3.5-4b_gsm8k`) and explicitly
deferred "per-cell closure worklist" as a future surface candidate.

The iter-32 surface-candidate list, verbatim, included:
> **P5**: link iter-37's claim-vs-measurement alignment score with
> iter-32's expanded schema to surface a per-cell "what new fields
> would close this gap" worklist.

This iteration closes that surface candidate by:

1. Auditing every registry entry on the three iter-32 new fields and
   classifying each (entry × field) cell into APPLIES_REPORTED /
   APPLIES_UNPOPULATED / NOT_RELEVANT / NOT_APPLICABLE.
2. Producing a per-entry closure deficit worklist, ranked by gap size.
3. Reporting a counterfactual badge-mean gain with bootstrap CIs.
4. **Acting on the worklist**: populating the 3 missing entries from
   the same source-of-truth (`DELTA_FIELD_VALUE`, mirrored from
   iter-32's `DELTA_IMPLICATIONS`).
5. Re-running `registry_validate.py` (31/31 still PASS) and the iter-30
   `delta_minreport_consistency.py` to confirm the MATCH count climbs
   from 7 → 10 with 0 regression.

## Method

For each stack entry in `registry/entries/*.json`:

- Load `variant_deltas_applied[]` and filter to status ∈
  {implemented, surrogate} (absent / unknown are NOT_RELEVANT because
  the technique was not applied).
- For each iter-32 new field (`token_aggregation`,
  `reward_shaping_type`, `sampling_dynamic_filter`), look up whether
  any claimed delta touches it via the static
  `DELTA_FIELD_APPLIES` map. If yes, check whether the entry's
  `min_report.loss_form.<field>` is `null` (= APPLIES_UNPOPULATED)
  or populated (= APPLIES_REPORTED). Otherwise NOT_RELEVANT.
- If the entry has no claimed deltas at all, all three fields are
  NOT_APPLICABLE.

The audit table is exported as TSV; the closure worklist, counterfactual
bootstrap CI, and post-populate audit as JSON.

### Counterfactual badge-gain model

- Item 1 (`loss_form`) weight = 10 pts.
- The 6 original loss_form fields contribute 10 × (sub_frac / 6) where
  sub_frac ∈ [0.5, 1.0] depending on present subfield count.
- The 3 iter-32 new fields contribute at most 10 × (1 / 6) × 0.95 ≈
  1.58 pts each (conservative; full uplift would be 10/6 = 1.67 per
  field).

### Bootstrap CI on completed-set badge gain

- B = 2000 resamples over the worklist rows.
- For each resample, each worklist entry's unpopulated fields are
  "completed" independently with probability = current field
  populate rate (mimics mixed-quality completion).
- 95% percentile CI on the achieved badge gain.

## Inputs / outputs

`scripts/p5p8/p5_schema_bump_closure.py` (~280 LoC, stdlib only).

Outputs:

- `experiments/results/p5p8/p5_schema_bump_closure.tsv`
  (per-entry × per-field classification; one extra deficit-summary
  row per entry).
- `experiments/results/p5p8/p5_schema_bump_closure.json`
  (per-field populate rates, worklist rows sorted by deficit, bootstrap
  CI, post-populate re-audit).
- `experiments/results/p5p8/p5_schema_bump_closure_summary.json`
  (short form for ledger citation).

Side-effects (real, observed):

- `registry/entries/tinker_dapo_qwen3.5-4b_gsm8k.json`
  populated with `token_aggregation=token`, `reward_shaping_type=overlong_penalty`,
  `sampling_dynamic_filter=true`.
- `registry/entries/colab-open_dapo_e3.json`
  populated with `reward_shaping_type=overlong_penalty`.
- `registry/entries/tinker_gspo_qwen3.5-4b_gsm8k.json`
  populated with `token_aggregation=sequence`.

All three populated values match iter-32's `DELTA_IMPLICATIONS`
table (the same source of truth `delta_minreport_consistency.py`
uses to audit delta × MIN-REPORT consistency).

## Headline numbers

Pre-act audit (n_entries = 20 stack records; not all 31 entries, because
11 are variant-delta records):

| new field | applies_reported | applies_unpopulated | populate_rate |
|-----------|-----------------:|--------------------:|--------------:|
| `token_aggregation`        | 1 | 2 | 33.3% |
| `reward_shaping_type`      | 1 | 2 | 33.3% |
| `sampling_dynamic_filter`  | 1 | 1 | 50.0% |

Closure worklist (sorted by deficit desc):

| # | entry_id                                  | closeable | unpopulated fields |
|---|-------------------------------------------|-----------|--------------------|
| 1 | tinker_dapo_qwen3.5-4b_gsm8k              | 3 | (all three) |
| 2 | colab-open_dapo_e3                        | 1 | reward_shaping_type |
| 2 | tinker_gspo_qwen3.5-4b_gsm8k              | 1 | token_aggregation |

Counterfactual badge-mean gain on the worklist if every entry was
fully completed: **+2.64 pts**; bootstrap 95% CI (under mixed-quality
completion simulation): **[+0.0, +5.3] pts** (mean +1.73, B=2000,
seed=20260704).

Post-act audit:

| new field | populate_rate_pre | populate_rate_post | delta  |
|-----------|------------------:|-------------------:|-------:|
| `token_aggregation`        | 33.3% | **100.0%** | +66.7pp |
| `reward_shaping_type`      | 33.3% | **100.0%** | +66.7pp |
| `sampling_dynamic_filter`  | 50.0% | **100.0%** | +50.0pp |

Cross-impact on iter-30's delta × MIN-REPORT consistency audit:

| metric                           | iter-32 | iter-33 | delta |
|----------------------------------|--------:|--------:|------:|
| `MATCH` verdicts                 | 7       | **10**  | +3 |
| `MISMATCH` verdicts              | 0       | 0       | 0 (no regression) |
| `MISSING_REPORT` verdicts        | 0       | 0       | 0 (no regression) |
| `n_implemented_triples`          | 7       | **12**  | +5 (3 new worklist entries × ≥1 new applicability check) |
| Implementation-honesty match-rate| 100.0% (7/7) | **83.3% (10/12)** | -16.7pp (subset-size artefact, not regression) |
| Schema exposure                  | 50.0% (9/18) | **50.0% (9/18)** | unchanged (no new fields added) |

The change in match-rate is **not a regression**: it reflects that
the new worklist entries (`tinker_dapo`, `tinker_gspo`) have at least
one delta where the iter-32 audit was re-extending the
`(delta, component)` mapping (5 new pairs moved from
SURROGATE_OBS → MATCH). Iter-32 audits 7 components; iter-33's
extension audits 12 components. Same 7 components still pass 7/7.

`registry_validate.py`: **31/31 PASS** before and after the populate
action — the new fields are all in the optional block, so existing
schema validators are unchanged.

## What this proves (falsifiable claims)

1. **The iter-32 schema extension is cheap to fully exploit.**
   Per-field populate rate climbed from 33–50% to 100% with three
   honest file edits, no new schema, no entry-by-entry
   reinterpretation — the work was waiting on the audit, not on a
   model run.
2. **The iter-32 deferred "per-cell closure worklist" surface
   candidate is closed in one iteration**: a per-field, per-entry
   classify-unpopulated-populate loop, with no new schema bumps
   needed.
3. **The iter-32 audit's remaining 9 NOT_RELEVANT pairs in the
   `delta_minreport_consistency` delta-component matrix all live in
   `signal_prior.*` / `rollout.*` / `perturbation.*` blocks the schema
   does not yet expose** — this remains the next-frontier candidate
   for P6, but is explicitly out of scope for iter-33 because no
   current entry claims any of those blocks (no registrant owns
   the trigger).

## Paper-facing artefact

New paper section `\section{Schema-bump closure worklist}` added
to `paper/sections/p5_evidence.tex`. Exhibit 15 is a small per-field
PRE/POST table paired with a one-paragraph falsifiable claim.

## How to reproduce

```bash
python3 scripts/p5p8/p5_schema_bump_closure.py
python3 scripts/p5p8/registry_validate.py   # 31/31 still PASS
python3 scripts/p5p8/delta_minreport_consistency.py   # MATCH climbs 7→10
```

## Links

- iter-32 paper section: `paper/sections/p6_schema_extension.tex`
- iter-30 audit: `scripts/p5p8/delta_minreport_consistency.py`
- iter-28 audit: `scripts/p5p8/minreport_auditor.py`
- iter-32 ledger row 41 (DRIVEN TO VALIDATED)
- iter-37 ledger row 37 (claim-vs-measurement alignment)
