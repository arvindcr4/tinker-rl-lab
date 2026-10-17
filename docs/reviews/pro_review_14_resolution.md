# Partial review 14: defense demo resolution

## Scope and evidence boundary

Applied the supplied `review_14_demo_spaces.md` to integrated base `6407255d`.
The supplied review ends inside the proposed `validate_run` code at `assert len(run["`.
The missing remainder and promised final five-change list are **explicitly skipped**;
no findings have been invented from that absent text. The priority summary's generic
accessibility and duplication headings have no supplied concrete reproductions beyond
the already accessible controls, and are not treated as unspecified redesign requests.

No training, paid jobs, network jobs, stored-result rewrites, generated-data commits,
or GitHub writes were performed. Canonical and historical APIs remain separate. No
changes to excluded `Platform.jsx` or `platform_calibration.py` were necessary.
The real export remains **interim**, with no final v2 equivalence claim.

## Confirmed findings

1. **Degenerate paired samples / sign-flip overflow:** already resolved in the integrated
   `src/stats.js`. Invalid/insufficient inputs throw; constant samples use the explicitly
   documented point-mass limit, including p=.5 at a TOST boundary. Exact sign flips have
   a 20-observation limit, with unavailable exact results above that limit. Existing
   shared-fixture Python/JavaScript tests cover empty, singleton, invalid, constant,
   zero, boundary, tiny-value, and n=21/31/32/40 cases. Preserved this tested policy
   rather than silently changing the canonical scientific contract.
2. **Missing contrasts / inconsistent equivalence badge:** already resolved by the
   integrated canonical `analysisModel`, `Overview`, and `SameStack`. All four declared
   contrasts remain present, pending inference is rendered explicitly, and both views
   use the same eligible adjusted decision. Existing rendering tests exercise empty,
   single-arm, disjoint, and overlapping historical seeds without making final claims.
3. **Failed audit accepted by builder:** fixed. `grader_check` no longer invokes
   `--report-only`; execution failure, nonzero exit, malformed JSON, unsupported spec,
   malformed counts, disagreement, missing coverage, and inconsistent denominators
   abort before publication. `validate_grader_report` enforces these independently of
   process exit status. Regression tests prove an existing export is not overwritten
   on audit failure. The audit CLI retains explicit read-only diagnostic report mode;
   diagnostic use is not publication approval.
4. **Malformed boxed syntax / Python parity:** already resolved by the integrated exact
   rational v2 grader. Existing tests enforce the boxed command boundary, nesting,
   unmatched boxes, full numeric syntax, repeated decorations, nonfinite forms,
   precision, and parity with both Python implementations. The policy selects the
   last *complete* valid box; an incomplete later box does not override it. Historical
   scores remain immutable.
5. **Exhaustive audit alignment:** integrated code already records missing sidecars,
   malformed arrays, score/text type failures, and length mismatch as incomplete
   coverage. Added explicit nonempty gold validation, object validation for run and
   sidecar, and declared `n_eval`/gold agreement. Added unequal/extra/missing arrays,
   malformed records, invalid scores, empty gold, and denominator-drift regressions.

## Supplied consistency/presentation concern

The existing exporter validates manifested completion bundles with the canonical
validator, selects only eligible records through offline analysis, hashes known input
and implementation sources, and explicitly discloses missing historical provenance.
That distinction is preserved; a matching historical seed is not evidence of aligned
canonical items. The new publication audit gate closes the remaining fail-open path.

Also strengthened the browser's canonical-artifact boundary: seeds must be integer
arrays aligned with pair identities, differences must be numeric arrays, intervals
must contain the mean in order, and reported raw/adjusted probabilities must be valid.
Malformed pair/selection containers now degrade to unavailable inference instead of
throwing. Invalid inputs cannot produce a confirmatory badge. Exploratory comparisons
retain valid raw inference without requiring nonexistent Holm adjustments. Regression
fixtures cover corrupted seeds, arrays, pairs, intervals, probabilities, and the
exploratory exception. This is validation of the supplied artifact, not a second
browser-side pairing or inference implementation.

## Validation

- Node unit/parity/rendering tests: 28 passed
- Python exporter tests: 9 passed
- `npm run build`: passed
- `node tests/grade_all.mjs --json`: exit 0; 71,226 completions, 27 runs,
  complete coverage, zero disagreements
- Exporter with `--output /tmp/review14-export.json`: succeeded; analysis `interim`
- Full Chromium browser tests: blocked by the execution sandbox (local socket denied);
  escalation also failed during environment setup. No full-browser pass is claimed.
  The Node React static-render regression suite passed.
- `git diff --check`: passed

Build outputs were exercised locally and restored/excluded from the commit. Python
parity uses `/tmp/tinker-review-venv/bin/python`; browser execution additionally uses
the installed runtime Playwright module through `PYTHONPATH` and system Chromium.
