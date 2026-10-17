# TinkerRL Measurement Lab (web demo)

Offline React defense demo. Canonical inferential results are calculated by the Python analysis plan and exported as `DATA.analysis`. The browser computes descriptive arm means, ZVF summaries, advantage examples and answer re-grades. Registry comparisons and audit summaries are exported offline. No training or network calls occur in the page.

## Open and rebuild

The checked-in `tinkerrl_demo.html` and `src/data.json` are historical artifacts. Source repairs do not rewrite them. Rebuild to see the repaired presentation:

```sh
npm ci
python3 build_demo_web.py --output /tmp/samestack-demo.json
DEMO_DATA=/tmp/samestack-demo.json npm run build
```

The build writes `dist/index.html` and `tinkerrl_demo.html`. Both are self-contained and open from disk. `npm run data` explicitly replaces `src/data.json` with a new export; it never edits underlying experimental results. Use `--plan /path/to/plan.json` only with a declared versioned analysis plan. Missing historical identity is never inferred from filenames or seed numbers. New runs can be selected with `--results-dir /path/to/environment/protocol` pointing to the leaf directory containing the arm/seed records; completion markers and aggregates are not treated as runs.

Presenter controls: ← / → or 1–8 change views; P toggles larger text. A short route is Advantage lab → GRPO vs PPO-style → Real answers → Stack diff.

## Scientific contract

- The sole inferential selector is `analysis_plan.analyze_runs`. It validates the raw run records and binds run IDs, actual environment hash, shared effective-stack fingerprint, protocol hash and ordered test-item hash. Matching seed numbers alone never creates a pair in JavaScript. Manifested filesystem inputs also require the shared completion-marker/inventory validator; missing or corrupt bundles are explicitly excluded before canonical selection
- Exactly three primary contrasts stay in the Holm family, including missing comparisons with p = 1. G2−PPO-style is exploratory
- Interim or retrospective analyses do not produce confirmatory conclusions. Completed record counts are separate from planned-seed completion and final-analysis eligibility
- Both 95% difference CIs and 90% equivalence CIs are labeled. Equivalence also requires a justified declared margin and adjusted combined TOST p below alpha in a validated final analysis. The present retrospective repair cannot confer preregistration
- The directional +5 pp sensitivity test excludes a sufficiently large positive effect; it is not a two-sided practical-equivalence claim
- Historical arm summaries use each arm's own pre-score and are descriptive. They do not establish a training gain. Several factors changed simultaneously from v1 to v2; the page does not attribute a causal effect to one change
- Comparisons concern the implemented PPO-style variant, not PPO algorithms generally

The provenance footer lists full SHA-256 values for known run records, text sidecars, gold answers, analysis code, grading code, exporter, registry inputs/implementation, audit input, frontend sources and dependency lock. It reports missing inputs and unknown historical identity. A file hash does not prove an unrecorded training environment.

## Verification

```sh
node --test tests/test_*.mjs
python3 tests/test_export.py
DEMO_PAGE=dist/index.html DEMO_DATA=/tmp/samestack-demo.json python3 tests/test_demo.py
node tests/grade_all.mjs
```

- Shared fixtures check exact boxed/rational grading and Python/JavaScript statistics, including invalid, zero-variance and TOST-boundary cases; exact sign-flip enumeration is capped at 20 pairs
- Analysis/view fixtures cover absent arms, zero contrasts, one pair, disjoint/invalid provenance, duplicate seeds, interim completion, missing margins and adjustment gating
- Export tests verify the normative analysis path and immutable inputs; browser checks cover all views, small/large layouts, canonical pending/available states, grader probes and navigation
- `grade_all.mjs` is an audit against immutable historical scores. A nonzero exit can indicate genuine disagreement with the repaired grader, not a software regression. `--json --report-only` emits the discrepancies and coverage without failing the export. Never change the grader to force old scores to agree

`npm test` runs unit/contract/export/browser checks before the strict historical score audit. Python browser checks use `DEMO_PAGE`/`DEMO_DATA` when supplied and can select an installed browser with `CHROME_PATH`.
