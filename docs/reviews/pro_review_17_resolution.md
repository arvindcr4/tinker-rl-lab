# Review 17 resolution: P6 and closure pass

Source: `review_17_modal_p5p8_2.md`. This is document 17, not GitHub PR 17.
Protocol: **review17-v2**. This change executes no training, provider jobs, or
historical regeneration and edits no stored results or generated artifacts.
Corrected outputs use `p5p8/review17_v2`. Registry repair tools now emit proposed
snapshots rather than silently revising historical entries.

## Completed in this commit

| Finding | Resolution | Regression evidence |
|---|---|---|
| 1–2, first priority: estimate repair/pairing | Explicit source column and seed identities; aggregate exclusion; reject missing, duplicate, incomplete and nonfinite observations; NumPy generator resamples paired differences. Complete delta/CI/n/significance/provenance object updates. mag_mean never substitutes mean_zvf; unavailable source fails. String migration preserves original method and unknown metadata | Constant +.1 paired effect; reordered identities; absent baseline; n=2/n=4 nondegenerate bootstrap; duplicate seed; mag source mismatch; idempotent complete-object replacement |
| 3, second priority: controller attribution | Complete unique method/threshold panel, method keys and method-specific filtering; per-rollout units; explicitly null/unverified bootstrap provenance. Counterfactual predictions only. Schema-validated proposals preserve unrelated fields | Two methods, one saved rollout/eight=.125; duplicate/missing panel fails; no claimed CI |
| 4, 6: P6 paths and gates | Canonical repository locations, versioned outputs and no import-time directory creation in touched scripts. Actual JSON Schema, finite ordered interval validation, nonempty corpus, nonzero CLI error. Strict audit rationale/sign declarations and source errors fail | Numeric id/string citation rejected; empty registry fails; NaN/reversed/infinite interval fails |
| 11: P6 identities | Ledger validates complete steps 30–39 and derives n; cross-panel validates 0–39, joins by step or seed, rejects duplicate/incomplete panels. Cross-stack checks actual step identities. Adaptive-G rejects incomplete/duplicate panels | Reordered steps retain delta; two-row partial window fails; missing 40-step input fails |
| 14: registry population | Closure uses same active-delta filter for audit/mutation, rejects conflicting implications and counts applied changes only; explicit --apply required. Iter138 preserves unknown metadata and removes unmatched-control effect rows, invented seeds, hardcoded sampler/splits and asserted TSV verification; links derive from actual component | Planned DAPO cannot override implemented GSPO; active conflict fails; n=3 does not invent seed IDs or n=1 effect |
| 15: repeatability | Stable result-key upserts; ledger built from proposed snapshot; no overwriting later delta enrichments (new entries are proposals); uniquely named iter134 summary artifacts | Identical proposed update is idempotent; originals untouched |
| 17: P6 utility defects | Missing/null distinction; exact method tokens; zero scores in denominator; nested leaves already fixed by earlier integrated review and preserved | Missing field vs null; ngrpo does not match grpo; existing prior-review regressions |
| 18–19: P6 scope | Remove ZVF/risk-index interchangeability; remaining cross-dataset comparison explicitly descriptive. N2 metadata distinguishes one independent run from many steps; no independent-run inference inferred | Complete-window test checks n_independent_runs=1 |
| 22: P6 quality labels | Remove significance-count evidence promotion and effect/CI-width load-bearing labels. Inventory is quality-unassessed/interval-availability only | A precise null and a large significant effect have same unassessed quality status |
| 23: shared helpers | Small review17 contract centralizes keyed loading, finite interval validation, paired bootstrap, provenance hashes, upserts and staged schema validation | Shared helpers exercised above |

## Remaining scope and scientific limits

The companion P5 owner is implementing findings 5, 7–10, 12–13, 16 and P5 parts
of 17–23 in a separate commit. This file does **not** claim those fixes until that
commit is integrated. No corrected historical numerical claims are asserted.

Existing step bootstrap intervals remain conditional within-run descriptions, not
independent-training-run inference. Bootstrap RNG repetitions do not create new
experiments. Absent matched run manifests, mag_mean source semantics, independent
replication or keyed controller bootstrap evidence remain explicit blockers rather
than invented results. Null provenance may fail the current schema; proposal
validation deliberately blocks publication rather than filling invented values.

Tests: `tests/reviews/test_review17_p6.py` (13 synthetic tests); prior review
16/18/19/20/21/22 suites passed together (84 tests). No empirical scripts were run against stored research data.
