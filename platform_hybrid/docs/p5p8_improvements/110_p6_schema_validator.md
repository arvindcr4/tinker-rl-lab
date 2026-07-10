# Iter 94 — P6 schema validator + measured-block coverage audit

## Pillar
P6 (Pillar 2 — GRPO-Registry)

## Vein
Brief veins (c) + (b) — a CI-style schema validator that every registry
entry must pass against `registry/schema.json`, plus a per-leaf null-
population coverage audit on the 20 stack records. Fresh vein, not in 109
prior rows.

## Deliverables

- `scripts/p5p8/p6_iter94_schema_validator.py` (~280 LoC, stdlib + jsonschema)
- `scripts/p5p8/p6_iter94_close_surrogate_gaps.py` (~80 LoC, stdlib)
- `experiments/results/p5p8/p6_iter94_validation.json`
- `experiments/results/p5p8/p6_iter94_validation.tsv`
- `experiments/results/p5p8/p6_iter94_field_coverage.tsv`
- `experiments/results/p5p8/p6_iter94_pending_gaps.tsv`
- `experiments/results/p5p8/p6_iter94_crosscheck.json`
- patched `registry/entries/{delta_dapo,delta_gspo,delta_reinforce,delta_adaptiveg}.json`
- `paper/sections/p6_iter94_schema_validator.tex`
- `paper/paper_P6_registry.pdf` rebuilds to **48 pages / 0 errors / 0 undefined citations** (was 46, +2 pages)

## Falsifiable measured headlines

### H1 — 35/35 entries pass schema validation on initial run

The validator runs `jsonschema.Draft202012Validator` against
`registry/schema.json` for every `registry/entries/*.json` and reports 0
schema violations across 20 stack + 15 variant_delta records. The
`--strict` mode exits code 0 so the script can be wired into CI as a gate
on registry mutations.

### H2 — all 4 MEDIUM gaps closed by iter 94 on real evidence

Initial run surfaced 2 MEASURED-BLOCK-MISSING (`delta_dapo`, `delta_gspo`,
both notes lacked `intentional null` marker) and 2 CITATION-INCOMPLETE
(`delta_adaptiveg`, `delta_reinforce`, both with bibkey but missing arxiv
and not flagged as transparent-placeholder). All 4 closed this iteration:

- **DAPO/GSPO**: patched with surrogate-markers that cite the existing
  `tinker_dapo_qwen3.5-4b_gsm8k` / `tinker_gspo_qwen3.5-4b_gsm8k` stack
  records' `variant_deltas_applied` status field (DAPO: 1 surrogate /
  2 absent / 2 unknown; GSPO: 1 surrogate / 1 unknown). Marker states
  explicitly that the tinker stack is a LABEL-FLIP SURROGATE only and that
  adding a measured row sourced from that stack would advertise the
  surrogate as the real algorithm.
- **REINFORCE**: patched with the canonical 1992 journal DOI surface
  (10.1007/BF00992696) since the paper predates arXiv.
- **Adaptive-G**: patched with a worktree-implementation note (live in
  `colab-open_grpo-adaptiveg_e3` and the P7 unified controller bank).

Post-iter-94: 0 pending MEDIUM gaps, 35/35 `citation_ok`, 5
`intentional_null` variant_delta entries (delta_ppo, delta_reinforce,
delta_liteppo, delta_dapo, delta_gspo) all with real provenance, not
paper-derived theory.

### H3 — stale-audit cross-check found a real drift

`registry/measured_block_audit.json` says `delta_drgrpo.measured_count=0`
but the entry actually has 3 measured rows (added in iter 74 on the
`length_bias_iter60` panel). The audit script was not re-run after iter
74's patch. `p6_iter94_crosscheck.json` flags this as a drift signal;
future audits should re-derive measured counts from live entries, not
the cached audit.

### H4 — 9 RED-FLAG leaves (>50% null rate across 20 stack records)

- `decontamination.parser_robustness_probe` (80% null)
- `decontamination.performed` (80% null)
- `loss_form.token_mask` (80% null)
- `loss_form.clip_eps_high` (75% null)
- `loss_form.clip_eps_low` (75% null)
- `loss_form.length_normalization` (75% null)
- `reference_kl.kl_beta` (65% null)
- `reference_kl.kl_estimator` (65% null)
- `loss_form.advantage_normalization` (60% null)

The 5-entry cluster `loss_form.clip_*` / `length_normalization` /
`token_mask` / `reference_kl.kl_*` is concentrated on the 5
`zvf130_<method>` single-batch risk-index harness entries where loss-form
internals are managed-by-tinker and unverifiable. The 2
`decontamination.*` leaves are universal because the field was added
later and most legacy entries pre-date it. Mean min_report coverage
across the 35-entry corpus is **0.3576**.

## Cross-paper coupling

1. **P6 iter 78 row 92** — iter 78 hand-audited every entry's reporting-
   coverage. Iter 94's validator produces the same kind of audit
   mechanically, and re-runnable on every future registry update.
2. **P6 iter 90 row 107** — iter 90 closed 5 `zvf130_*` measured-block
   gaps. Iter 94's validator would have flagged those entries as MEDIUM
   severity on the day iter 90's gap was open; a CI-style validator would
   have forced the issue earlier in the pipeline.
3. **P5 iter 89 row 106 (GIFT carries algorithm-axis variance)** — iter
   89's GIFT-isolation finding gives operational priority to GIFT's
   measured-coverage (4 rows on 2 panels = `measured_coverage=1.0`); the
   validator now reports this in the per-entry summary.
4. **FRONTIER_INSIGHTS Round 1 (Critic Degeneracy Hypothesis)** — the
   frontier synthesis argues the token-level critic is dead weight on
   sparse terminal-reward CoT. Iter 94's validator surfaces exactly the
   relevant loss-form leaves (`loss_form.clip_eps_*`,
   `length_normalization`, `advantage_normalization`) as the red-flag gap
   cluster — the frontier argument is operationally testable by
   populating those leaves for the GRPO baseline and comparing to
   `delta_ppo`.

## Operational recommendation

Every new entry must pass
`python3 scripts/p5p8/p6_iter94_schema_validator.py --strict` before
commit. The validator exits non-zero on any new SCHEMA-VIOLATION;
MEASURED-BLOCK-MISSING and CITATION-INCOMPLETE are emitted as MEDIUM/LOW
severity rows in `p6_iter94_pending_gaps.tsv` but do not block. Future
iterations should:

1. Regenerate `measured_block_audit.json` from live entries to close the
   stale-audit discrepancy on `delta_drgrpo`.
2. Backfill the 9 RED-FLAG leaves on stack records with explicit
   `reported-as-unknown` provenance for the 5 `zvf130_*` managed-loss
   leaves.
3. Promote MEASURED-BLOCK-MISSING to HIGH severity once a same-stack arm
   exists for the remaining variant.

## Reproduction

```bash
# initial run (4 MEDIUM gaps)
python3 scripts/p5p8/p6_iter94_schema_validator.py

# close 4 gaps (2 surrogate-markers + 2 transparent-placeholders)
python3 scripts/p5p8/p6_iter94_close_surrogate_gaps.py

# strict CI gate (exit 0 = no HIGH-severity gaps)
python3 scripts/p5p8/p6_iter94_schema_validator.py --strict

# paper rebuild
cd paper && pdflatex paper_P6_registry && bibtex paper_P6_registry && \
  pdflatex paper_P6_registry && pdflatex paper_P6_registry
```

## Citation provenance

The 4 patches cite real evidence:
- `tinker_dapo_qwen3.5-4b_gsm8k.json` `variant_deltas_applied[]` field
- `tinker_gspo_qwen3.5-4b_gsm8k.json` `variant_deltas_applied[]` field
- `colab-open_grpo-adaptiveg_e3.json` (worktree implementation)
- Williams 1992 Machine Learning journal DOI 10.1007/BF00992696

All four are verified worktree data — no fabricated citations.