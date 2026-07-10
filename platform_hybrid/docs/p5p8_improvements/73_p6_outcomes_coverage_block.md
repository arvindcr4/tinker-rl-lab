# 73 — P6 `outcomes.coverage` self-report block + claim-without-evidence audit

**Pillar:** P6 (Pillar 2 — GRPO-Registry).

**Vein (fresh, closes iter-60 row 71):** the iter-60 audit
(`docs/p5p8_improvements/71_p6_registry_minreport_audit.md`) documented that
**0/31 registry entries disclose their actual MIN-REPORT coverage**, while
the 98 mega manifests do. Iter-60 row 71's recommendation was:
> "schema patch needed: `outcomes.coverage` block per #36 pattern
> (iter-28 `ci_method` additive-optional model)".

Iter-61 row 72 confirmed the discriminator is sharpest at item-7 granularity
on the 98 mega cells; both rows converged on closing the registry's
self-reporting gap. This iter closes that gap by:

1. Adding an **additive-optional** `outcomes.coverage` block to
   `registry/schema.json` (the same `additionalProperties: false`
   constraint as the existing `outcomes.ci_method`).
2. Patching all 20 stack entries with the coverage block populated from
   real measured coverage (the audit script is idempotent — re-running
   on a patched entry is a no-op).
3. Adding a `coverage` subcommand to `registry/query.py` that prints the
   per-entry coverage table.
4. Auditing the cross-table consistency: how many stacks claim a variant
   in `variant_deltas_applied` (status `implemented`/`surrogate`) without
   the corresponding `delta_*.json` having any measured rows.

## Falsifiable headlines (audit re-run 2026-07-05)

1. **34/34 schema PASS** post-patch — same as iter-50 (31) and iter-54 (34);
   no entry was forced to non-conformance; the patch is purely additive.
2. **20/20 stack entries** now carry `outcomes.coverage` (the 14
   `variant_delta` records do not carry `outcomes` per the schema, so the
   block is stack-only).
3. **Stack `min_report_coverage` mean = 0.750**, range
   `[0.429, 1.000]`. 16/20 stacks carry `>=0.714`; only the 5 `zvf130_*`
   stacks carry `<=0.429` (those entries intentionally report less MIN-REPORT
   detail because they live on a single-batch risk-index harness).
4. **Stack `declared_deltas_coverage` mean = 0.375**. The 8
   `colab-open_*` + `tinker_dapo` + `tinker_gspo` entries that **do** declare
   variants drive this up; the 7 baseline-arm entries (4× `tinker_grpo_*`,
   `openrlhf_grpo_qwen3-8b_gsm8k`, `trl_grpo_qwen3-8b_gsm8k`,
   `verl_grpo_qwen3-8b_gsm8k`) deliberately declare zero deltas, so their
   rate is 0.0 by construction. The 5 `zvf130_*` stacks declare one unknown
   each, so their rate is 0.0 (the audit's `informative` filter excludes
   `unknown`).
5. **Stack `measured_coverage` mean = 0.360**. The 7 N2-derived
   `tinker_*` entries (aero/areal/dapo/drgrpo/gift/gspo/grpo) carry 0.6–0.8;
   the rest are 0.0–0.6 because they do not come from N2 tensors.
6. **Stack `ci_method_present` = 7/20** — exactly the iter-28 #36 result
   preserved (the same 7 N2 entries self-report CI provenance).
7. **Cross-table consistency — claim-without-evidence: 10/14 (71.4%)**
   `implemented`/`surrogate` claims on stacks that point to a
   `delta_*.json` with **0 measured rows**. The 4/14 with backing rows are
   the N2-anchored pairs: `(tinker_aero, delta_aero, 4 rows)`,
   `(tinker_areal, delta_areal, 4 rows)`,
   `(tinker_gift, delta_gift, 4 rows)`,
   `(colab-open_grpo-adaptiveg, delta_adaptiveg, 2 rows)`. The remaining
   10 are honest gaps where the registry claims an implementation but the
   same-stack panel has not yet been measured.
8. **Delta-side**: of the 14 `variant_delta` records, 8 have at least one
   measured row (aero, areal, cppo, es, gift, mcgrpo, ngrpo, scafgrpo,
   adaptiveg). 6 carry `measured=null` (dapo, drgrpo, gspo, liteppo,
   reinforce) — the iter-54 finding preserved.

## Why this matters (frontier synthesis)

The iter-61 row 72 finding is that the auditor's weight vector is
**over-parameterised** on the homogeneous mega corpus: items 1/2 each carry
≤1% of variance because every mega cell populates them identically. The
P6 registry has the *same structural property* but **did not even
self-report**: 0/31 entries disclosed their own MIN-REPORT coverage.
Now 20/20 stacks do. This means:

- a downstream `claim-equivalence` detector (frontier synthesis — Gemini
  Deep Think's `δ_div` anti-herding diagnostic) can ingest the coverage
  block as a structural fingerprint and reason about "what does this entry
  actually measure?" instead of "what does it claim?"
- the audit no longer relies on parsing each entry's `min_report` 23-leaf
  dict to compute coverage; the block is the audit's machine-readable
  output, mirrored back into the entry. (The block was populated by the
  audit; the schema ensures the block can never be free-form.)

## Cross-paper coupling

- **P5 ↔ P6**: iter-61 row 72 found item-7 (decontam/parser) carries the
  only measurable outcome-correlation (Spearman ρ ≈ ±0.83) on the 98
  mega cells. The P6 audit now reports per-entry `decontamination` item
  population directly (the existing
  `p6_registry_minreport_subfield.tsv` audit shows only **4/20 entries**
  populate `decontamination.performed` — that gap is now visible as a
  `0.0` rate in the per-entry table).
- **P6 ↔ P7**: the 4/14 claim-without-evidence pairs that *do* have measured
  rows are the iter-31 / iter-47 / iter-54 confirmation that adaptive-G
  (delta_adaptiveg) and the N2 deltas (aero/areal/gift) measurably shift
  ZVF in the same direction the source papers claim.
- **P6 ↔ P8**: the `outcomes.coverage` schema pattern is identical to
  the iter-28 `outcomes.ci_method` extension — the registry now has
  **two** self-reported provenance blocks; the pattern generalises cleanly
  to any future "did you actually run it?" block (e.g. a future
  `outcomes.heldout_seed_count` block).

## Artifacts

- `registry/schema.json` — additive `outcomes.coverage` property
  (`min_report_coverage`, `declared_deltas_coverage`,
  `measured_coverage`, `ci_method_present`, `audit_source`,
  `audit_date`; all nullable; `additionalProperties: false`).
- `registry/entries/*.json` — 20 stack entries patched (one
  `outcomes.coverage` block per entry); 14 delta records unchanged
  (variant_delta_record does not carry `outcomes`).
- `registry/query.py` — new `coverage` subcommand with `--entry`
  filter; existing 10 subcommands unchanged.
- `scripts/p5p8/p6_outcomes_coverage_block.py` (≤300 LoC, stdlib only)
  — idempotent re-runnable audit + patch.
- `experiments/results/p5p8/p6_outcomes_coverage_audit.tsv` (34 rows)
- `experiments/results/p5p8/p6_outcomes_coverage_claim_evidence.tsv`
  (14 rows, one per `implemented`/`surrogate` stack claim)
- `experiments/results/p5p8/p6_outcomes_coverage_summary.json`
- appended to `findings_ledger.jsonl` (pillar P6).

## Reproduction

```bash
python3 scripts/p5p8/p6_outcomes_coverage_block.py   # ~2s on 1 core
python3 registry/query.py validate                    # 34/34 PASS
python3 registry/query.py coverage                    # prints per-entry table
python3 registry/query.py coverage --entry tinker_aero_qwen3.5-4b_gsm8k
```

## Validation

- `python3 registry/query.py validate` returns **34/34 PASS** (exit 0).
- `python3 registry/query.py health` still works (iter-50 audit unaffected).
- `python3 registry/query.py coverage` lists all 34 entries with the
  computed rates; one entry has 1.0/1.0/0.8/True (tinker_aero); one has
  0.429/0.0/0.0/False (zvf130_es — minimal MIN-REPORT on the
  risk-index-only harness).