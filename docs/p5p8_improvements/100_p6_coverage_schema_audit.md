# P6 iter-186 — Registry coverage audit + schema validation

**Iter:** 186 — **Pillar:** P6 (GRPO-Registry)
**Veins:** (b) coverage audit + (c) schema validation
**Scripts:** `scripts/p5p8/p6_iter186_coverage_audit.py`
**Outputs:**
- `experiments/results/p5p8/p6_iter186_coverage.tsv` (one row / variant_delta)
- `experiments/results/p5p8/p6_iter186_stack_minreport.tsv` (one row / stack)
- `experiments/results/p5p8/p6_iter186_schema_valid.tsv` (one row / entry)
- `experiments/results/p5p8/p6_iter186_coverage.json` (aggregate)

## Why this iteration

The P6 registry has grown organically over ~150 iters. Two unverified
assumptions had crept in: (a) every variant_delta has at least one
machine-verifiable measured row grounded in a real provenance path, and
(b) every entry still parses the schema. Iter 186 audits both — across
all 46 entries — and produces a single TSV/JSON pair so the next P6 iter
can close the highest-leverage gap without re-discovering it.

## Method

The audit has three parts, all stdlib + the existing `jsonschema` 4.19.2
already on the system path:

1. **variant_coverage**: for every `delta_*.json`, classify the presence
   of `measured[]`, `expected_effects[]`, `claim_validation[]`. Cross-
   reference against (i) the `zvf130_<label>.json` 5-seed risk panel and
   (ii) any `tinker_*` / `wandb_*` trace whose `label_claimed` matches
   the variant's `name`. Verdict: `FULL` (≥3 measured & ≥3 claim_validation)
   | `PARTIAL` (≥1 measured) | `EXPECTED-ONLY` (declared, never measured)
   | `BLANK` (no measured, no expected).
2. **stack_coverage**: for every `stack` record (tinker_, wandb_, colab-_,
   trl_, verl_, openrlhf_, zvf130_), count min_report field-fill at the
   7-item layer (loss_form, reference_kl, …) and at the leaf level.
3. **schema_validate**: dispatch each entry to its `record_type`-branch
   (`stack_record` vs `variant_delta_record`) and re-validate via
   `Draft202012Validator`. The naive oneOf validation produces a single
   composite error whose `.message` is the data instance (useless for
   triage); branch dispatch surfaces real `additionalProperties`,
   `required`, and `enum` failures.

## Headline findings

### Coverage (variant_delta, 18 entries)

| Verdict         | Count | Entries                                                                                       |
|-----------------|-------|-----------------------------------------------------------------------------------------------|
| FULL            | 9     | aero, areal, cppo, drgrpo, es, gift, mcgrpo, ngrpo, scafgrpo                                  |
| PARTIAL         | 3     | adaptiveg, tool_use_llama-8b-inst, tool_use_qwen3-32b                                         |
| EXPECTED-ONLY   | 4     | **dapo, gspo, ppo, ppo_reinforce** — declared predictions, zero measured rows                 |
| BLANK           | 2     | **liteppo, reinforce** — empty delta records                                                  |

**6 of 18 variant_delta records (33%) have ZERO measured rows.**
Of these 6: 2 have a tinker trace available that could feed a measured
row (dapo via `tinker_dapo_qwen3.5-4b_gsm8k`, gspo via
`tinker_gspo_qwen3.5-4b_gsm8k`); 4 (liteppo, ppo, ppo_reinforce, reinforce)
have neither risk-panel data nor raw traces and would require either
new runs or a stub `measured=null` declaration.

Cross-reference: 10/18 entries have a `zvf130_<id>.json` that could feed
`measured[panel=zvf130_5seed]`; 5/18 have a `tinker_*` / `wandb_*` trace
that could feed a custom stack panel.

### Schema validation (all 46 entries)

| Valid | Count | Notes                                                                              |
|-------|-------|------------------------------------------------------------------------------------|
| yes   | 34    | Includes every tinker_* and wandb_* stack record, plus liteppo/reinforce (BLANK)   |
| no    | 12    | 4 distinct drift classes — see below                                               |

**12 of 46 entries (26%) fail schema validation.** This is the registry
drift that the brief flagged: the schema has not been bumped in lockstep
with the iter-history that added fields to entries.

| Drift class                                                     | # entries | Where                                    |
|-----------------------------------------------------------------|-----------|------------------------------------------|
| `iter128_recompute_note` at root not in `variant_delta_record`   | 8         | aero, areal, cppo, es, gift, mcgrpo, ngrpo, scafgrpo |
| `iter_recomputed` inside `measured[]` not in `measured_delta`    | 8         | same 8 entries                           |
| `citation.bibkey / arxiv = None` not accepted                   | 2         | tool_use_llama-8b-inst, tool_use_qwen3-32b |
| `variant_deltas_applied[].status` enum missing some values      | 2         | zvf130_tool_use_llama-8b-inst, zvf130_tool_use_qwen3-32b |

### Stack records (28 entries)

The MIN-REPORT leaf-fill distribution (per `p6_iter186_stack_minreport.tsv`)
shows the tinker_qwen3.5-4b variants and the wandb_* traces fill the
full 7-item MIN-REPORT (~85–100% leaf-fill), while the open-source
framework records (trl_, verl_, openrlhf_) and the zvf130_ traces fill
<60%. The 7-item layer mostly tracks stack maturity (open managed traces
fill more than open-source-framework shells), not registry negligence.

## Hypothesis audit

**H1 (coverage)**: ≥ 75% of variant_delta entries should have ≥ 1
measured row at iter 186. → **FAIL** at 12/18 = 66.7%. Below target.

**H2 (schema-drift)**: All entries should pass branch-dispatch schema
validation at iter 186. → **FAIL** at 12/46 = 73.9% valid. The drift is
concentrated in 3 patterns from iters 128 and 178.

**H3 (cross-reference leverage)**: The 6 BLANK/EXPECTED-ONLY entries
should have either a `zvf130_*.json` risk trace or a tinker/wandb
trace available to feed a measured row. → **PARTIAL**: 2/6 (dapo, gspo)
have tinker traces; 4/6 (liteppo, ppo, ppo_reinforce, reinforce) have
neither and need new runs OR a documented `measured=null` decision.

**H4 (drift is bounded)**: Schema-drift should be concentrated in
additive fields, not retro-renamed or removed ones. → **PASS**: all 4
drift classes are additive (new fields added by later iters without
schema bump); no destructive change has been made.

## What this changes for P6

1. The audit is reproducible: `python3 scripts/p5p8/p6_iter186_coverage_audit.py`
   regenerates all four artifacts in <2 s.
2. The 6 BLANK/EXPECTED-ONLY variant_deltas are the next-iter priority
   for measured-row backfill (vein d, iter-182 already added
   `delta_ppo_reinforce` as the prototype). Order of attack:
   `delta_dapo` (has tinker trace) → `delta_gspo` (has tinker trace) →
   `delta_ppo`, `delta_ppo_reinforce` (no data; need new trace OR
   measured=null declaration with provenance explanation) →
   `delta_liteppo`, `delta_reinforce` (similarly unmaterialised).
3. The schema-drift is fixable in 4 small PRs — bumping `variant_delta_record`
   to allow `iter128_recompute_note`, bumping `measured_delta` to allow
   `iter_recomputed`, loosening `citation` to allow nullable
   `bibkey/arxiv`, and extending `status` enum. **Deferred** to a
   dedicated iter-187 schema-bump proposal (lower priority than measured
   backfill — the audit itself surfaces the drift for review).

## Build status

- Schema JSON untouched in iter 186 (drift surfaced, not patched — see
  above rationale).
- `paper/paper_P6_registry.tex` not rebuilt in iter 186 — the audit
  adds new TSVs/JSONs that are referenced by the paper only after the
  schema bump + measured-row backfill land.
- No dependencies on Tinker runs.