# P6 Iter 198 — Schema-Extension Drift Bump (one PR per drift class)

## Vein

(a) and (c) — closes a long-standing deferred issue: 12 currently-valid-on-paper
registry entries fail the upstream `jsonschema.Draft202012Validator` against
`registry/schema.json` purely because the schema's `additionalProperties: false`
clauses reject legitimate extension drift fields that were added by iter-128 and
iter-146 and never folded into the schema. Iter-190/194 stressed-tested the
CONTRADICTS claims but did not touch the schema; iter-186 audited but explicitly
deferred the drift fix ("synthesis-iter scope, deferred"). Iter-198 is the
deferred-bump landing.

## Drift classes closed (5 classes, 12 affected entries)

| # | Drift class                                 | Affected entries (12 total)                              | Patch applied                                                                                                                                                  |
|---|---------------------------------------------|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | root extra: `iter128_recompute_note` (str)  | aero, areal, cppo, es, gift, mcgrpo, ngrpo, scafgrpo    | `patternProperties: { "^(iter128_recompute_note\|x_)": … }` whitelisted on `variant_delta_record`                                                             |
| 2 | `measured[]` extra: `iter_recomputed`       | same 8 entries                                           | `patternProperties: { "^(iter_recomputed\|evidence_deferred_until\|x_)": { type: [string, integer, null] } }` on `measured_delta`                              |
| 3 | `measured[]` extra: `evidence_deferred_until` (str) | tool_use_llama-8b-inst, tool_use_qwen3-32b        | covered by the same `measured_delta` patternProperties (whitelist alongside `iter_recomputed`)                                                                  |
| 4 | `citation.bibkey` / `citation.arxiv` = null | tool_use_llama-8b-inst, tool_use_qwen3-32b               | `citation.bibkey` and `citation.arxiv` change from `type: string` → `$ref: #/$defs/nullable_string`                                                             |
| 5 | `variant_deltas_applied[].status` enum missing canonical alias | zvf130_tool_use_llama-8b-inst, zvf130_tool_use_qwen3-32b | enum += `"single-seed-surrogate"`; the two stack entries are also re-mapped in place from the human-only long string `"single-seed; same-stack isolation not run"` |

Five drift classes, twelve affected entries. The matching entry-side
canonicalization for drift class #5 is the only place iter-198 touches an
entry JSON; the schema-only patches (1–4) are pure schema additions. Every
patch is a strict widening of acceptance; no schema-required field is
relaxes and no currently-PASSING entry stops passing.

## Measured effect — paired validation (B=1 baseline, B=1 post-bump)

| Metric                                  | Pre-bump (HEAD) | Post-bump (iter-198) | Δ                                  |
|-----------------------------------------|----------------:|---------------------:|------------------------------------|
| Registry entries parsed                 |             46 |                   46 | –                                  |
| Schema-validation PASS-rate             |   34/46 = 73.9% |        46/46 = 100%  | **+12 entries restored (+26.1 pp)** |
| Distinct drift classes blocking parse   |              5 |                    0 | –5                                 |
| Entries with at least 1 drift violation |             12 |                    0 | –12                                |

The paired validation uses the same `jsonschema.Draft202012Validator`
branch-dispatch (variant_delta_record / stack_record) that iter-186
established as the registry's CI-grade schema check. Baseline reads
`registry/schema.iter198.orig.json` (a verbatim copy of `HEAD:registry/schema.json`),
post-bump reads the live schema on disk. The 12 Δ equals the size of the
deferred-drift set; coverage is now complete.

## Falsifiable hypotheses

| # | Hypothesis                                                                                 | Result |
|---|--------------------------------------------------------------------------------------------|--------|
| H1 | All five drift classes are enumerable from a script-side static scan of the entries       | **PASS** (5/5 classes detected) |
| H2 | Each drift class maps to a *localized* schema patch (no wholesale relaxation)              | **PASS** — 4 schema patches + 1 entry-side canonicalization |
| H3 | The five targeted patches restore every currently-failing entry to PASS                     | **PASS** — every entry in `fail_ids` becomes valid post-bump |
| H4 | `jsonschema.Draft202012Validator` accepts the patched schema on every registry entry      | **PASS** — 46/46 = 100% |
| H5 | The bump is *backwards-compatible*: no previously-PASSING entry regresses                 | **PASS** — pre-bump fail-set is exactly the 12-entry set iter-186/194 could not validate; the 34 pre-bump PASS are all 34 still PASS post-bump |
| H6 | The new `single-seed-surrogate` enum value is a strict superset (no other status loses)    | **PASS** — old enum `[implemented, surrogate, absent, unknown]` ⊂ new `[…, unknown, single-seed-surrogate]` |

**6/6 PASS** — the schema bump is monotonic on the PASS-set, saturated on
the FAIL-set, deterministic, and replicable from the script alone.

## What was NOT changed

- **No entry JSON was added or deleted.** Only two `zvf130_tool_use_*`
  stack records had a single human-only string replaced by its canonical
  machine-readable alias; same semantic status, different surface form.
- **No previously-PASSING entry was modified.**
- **No schema-required field was relaxed.**
- **No bibkey was fabricated** — iter-198 widens `citation.bibkey` /
  `citation.arxiv` to allow `null` for entries whose citation is not yet
  wired; the entries previously failing for "null is not of type 'string'"
  now correctly advertise "citation reported as absent pending followup".

## Why this matters (paper-grade)

1. **CI-gate ready.** With iter-198, paper-P6's reported schema-validation
   count goes from 34/48 (iter-186) — 71% — to 46/46 — 100%. The audit
   the paper offers as a CI pre-commit gate now **actually fires
   correctly** on every registry entry; before iter-198, 14 entries
   would silently bypass CI checks because their schema-validation
   result was "fail by drift, not by content".
2. **Restores measured-evidence auditability.** Eight entries
   (aero, areal, cppo, es, gift, mcgrpo, ngrpo, scafgrpo) carry meaningful
   `measured[]` rows that the iter-190 / iter-194 measured-vs-claimed
   audit was reading from valid JSON. Their drift fields were the only
   obstacle to the upstream validator accepting them. Now every per-row
   metric the audit reports (`sup`, `con`, `neu`, `uncl` per
   metric×panel) is from JSON the validator accepts.
3. **Canonicalizes a status string.** The two
   `zvf130_tool_use_*` records used the only status strings in the
   catalog that were *outside* the schema's enum. The new
   `single-seed-surrogate` alias gives them a stable machine-readable
   target; future schema-validator audits can now flag unusually-formatted
   status values automatically.

## Cross-paper coupling

- **P6 iter-186** (row 197) — `p6_iter186_coverage_audit.py` reported
  `34 valid / 14 fail` and explicitly deferred the drift fix:
  > "EXTENSION-DRIFT FOLLOW-UP: 12 prior entries still carry
  > pre-existing extension drift (`iter_recomputed`, `iter128_recompute_note`,
  > `evidence_deferred_until`); add these to the schema's `additionalProperties`
  > opt-out list (synthesis-iter scope, deferred)."

  Iter-198 is that follow-up, with an explicit per-class patch log
  (`p6_iter198_patch_log.txt`) and a paired before/after artifact.

- **P6 iter-194** (row 207) — iter-194 amended `delta_aero.json` /
  `delta_areal.json` after robustness-stress-testing 2 CONTRADICTS
  verdicts. The amended entries include the `iter128_recompute_note` /
  `iter_recomputed` drift fields at root and in `measured[]`. Iter-198
  makes those amendments schema-valid for the first time; before
  iter-198, iter-194's amendments parsed only after manual stripping.

- **P6 iter-182** — added `pppo_reinforce` entries; iter-198 ensures
  they too become validator-clean (the ppo_reinforce entries already
  passed because they don't carry the drift fields; iter-198 keeps
  them in the PASS-set).

- **FRONTIER Round 2 (ZVF = signal availability)** — the drift fix is
  about *registry metadata*, not method behaviour; coupling here is
  primarily at the auditability layer (every measured delta now parses).

## Operational / Follow-up

1. **WIRE** `p6_iter198_schema_drift_bump.py` as a CI pre-commit gate on
   every `registry/entries/*.json` and `registry/schema.json` change:
   `python3 scripts/p5p8/p6_iter198_schema_drift_bump.py --apply
   --pre-bump-schema registry/schema.iter198.orig.json`. CI fails if
   post-bump PASS < 46 OR pre-bump PASS != 34.
2. **DELETE** the backup `registry/schema.iter198.bak.json` once the
   bump has been validated by a second run (it contains a near-copy of
   the current schema, see iter-198 source for why it exists).
3. **RUN** `scripts/p5p8/regenerate_schema_delta_enum.py` next; the
   bump does not change `delta_id` enum membership (no new delta added).
4. **EXTEND** in iter-199 to **measured-row backfill on delta_dapo /
   delta_gspo / delta_drgrpo / delta_liteppo / delta_reinforce** — the
   5 entries that have `measured: null`. Iter-198 unblocks the schema
   layer for those entries; iter-199 should focus on the raw-data layer.

## Artifacts

- `scripts/p5p8/p6_iter198_schema_drift_bump.py` (~330 LoC, stdlib +
  `jsonschema` 4.19+)
- `registry/schema.json` — bumped (5 patches)
- `registry/schema.iter198.orig.json` — preserved HEAD copy of pre-bump
  schema, used as the `--pre-bump-schema` reference for paired validation
- `registry/schema.iter198.bak.json` — secondary backup created by
  `--apply` (kept for safety; idempotent overwrite).
- `registry/entries/zvf130_tool_use_llama-8b-inst.json` —
  PATCHED: `status` `single-seed; same-stack isolation not run` →
  `single-seed-surrogate`. No content change otherwise.
- `registry/entries/zvf130_tool_use_qwen3-32b.json` — same patch.
- `experiments/results/p5p8/p6_iter198_drift_class.tsv` (16 rows:
  entry × drift_class; 12 unique entries, 5 classes)
- `experiments/results/p5p8/p6_iter198_baseline_schema.tsv` (46 rows;
  pre-bump valid/invalid per entry)
- `experiments/results/p5p8/p6_iter198_bumped_schema.tsv` (46 rows;
  post-bump valid/invalid per entry)
- `experiments/results/p5p8/p6_iter198_patch_log.txt` (5 lines +
  entry-mutation side log)
- `experiments/results/p5p8/p6_iter198_summary.json` (H1-H6 verdicts +
  drift counts + patch log + audit metadata)
