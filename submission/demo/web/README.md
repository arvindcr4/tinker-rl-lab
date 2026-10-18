# TinkerRL Measurement Lab (web demo)

Offline React defense demo. Canonical inferential results are calculated by the Python analysis plan and exported as `DATA.analysis`. The browser computes descriptive arm means, ZVF summaries, advantage examples and answer re-grades. Registry comparisons and audit summaries are exported offline. No training or network calls occur in the page.

## Open and rebuild

The checked-in `tinkerrl_demo.html` and `src/data.json` are historical artifacts. Source repairs do not rewrite them. Rebuild to see the repaired presentation:

```sh
npm ci
python3 build_demo_web.py --output /tmp/samestack-demo.json
DEMO_DATA=/tmp/samestack-demo.json npx vite build
```

The source-only build writes the self-contained `dist/index.html`, which opens from disk. `npm run build` also replaces the historical `tinkerrl_demo.html`; use it only when that replacement is intended. `npm run data` explicitly replaces `src/data.json` with a new export; it never edits underlying experimental results. Use `--plan /path/to/plan.json` only with a declared versioned analysis plan. Missing historical identity is never inferred from filenames or seed numbers. New runs can be selected with `--results-dir /path/to/environment/protocol` pointing to the leaf directory containing the arm/seed records; completion markers and aggregates are not treated as runs.

Presenter controls: ← / → or 1–9 change views; 0 opens Submissions; P toggles larger text. The header's Shortcuts toggle turns single-key navigation off. Inputs and the mobile section selector keep their native keyboard behavior. A short route is Advantage lab → GRPO vs PPO-style → Real answers → Stack diff.

## Presentation and access

- On narrow screens, a labeled section selector exposes all ten views. Desktop section links keep the current view in sight; Previous and Next links and browser Back/Forward remain available
- A skip link bypasses the header. Route changes focus the named content region, except when selecting a section, where focus stays in the selector for repeated arrow-key use
- Outcome-pattern selection uses native buttons with pressed states. Reward controls retain generous touch targets, and status labels and long identifiers can reflow
- The page uses locally available fonts without external font requests. Reduced-motion preferences disable decorative control transitions

The source-only UI pass applies accessibility, navigation and resilient-text principles from [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill). It adds no dependency on that project's tooling and does not change the scientific analysis or historical artifacts.

The Platform calibration view retains the historical cross-platform calibration from `platform_calibration.json`, including its input hash in the provenance manifest. Its legacy inference and available-only multiplicity adjustment have not been repaired by the same-stack changes; do not treat its equivalence wording as a validated canonical same-stack conclusion.

## Conference submission history

The Overview links to `#submissions`, a separate administrative snapshot verified on 2026-10-18. `src/submissions.js` contains curated bibliographic and status metadata from organizer email notifications, independently of experimental `src/data.json` and the analysis pipeline:

- ICICC 2027 paper 243: accepted by organizer notification; camera-ready, registration and final publication remain unverified. The organizer-stated registration deadline is a date only because its timezone was not specified
- NeurIPS 2026 main-track paper 36320: rejected. A public-release form receipt does not establish the selected option or public availability
- Expandable history separates 31 workshop-paper submissions of 11 distinct titles across six workshops from one workshop-organization proposal. Counts derive from the records; rejection and desk-rejection remain distinct. AI4MetaScience paper 6 has a decision-availability notice, but its outcome remains unverified

`src/submissionFeedback.js` adds sanitized, attributed paraphrases to all 34 records. Search covers titles, venues, decisions and feedback; coverage filters distinguish retrieved reviews, committee/decision context, author responses and missing review text. Each submission has a native feedback disclosure with separate coverage notices:

- ICICC: two anonymous reviewer reports and final-paper requirements, dated to the acceptance email; no numerical review scores or posted author response were found
- NeurIPS main track: three original anonymous reviews and the initial area-chair assessment from the July 24 archive, two later reviewer comments, three posted author comments, and posting receipts whose response text is unavailable. Original posting dates are unknown; archive dates and explicit rating scales remain visible. The final paper-specific rejection rationale and latest portal state remain unverified
- Workshop submissions and proposal: general organizer context where available, with explicit missing-material notices. General selection criteria are not presented as paper-specific decision reasons

The author-response warning preserves chronology: the saved July 28 provenance audit predates the August 2 author claims about the 92.6%/92.1% control. The five-seed transfer claim is unsupported by that audit; withdrawal appears in an unposted revised-response draft. The UI does not treat historical author statements as current verified results or completed revisions, or drafts as posted responses.

Titles remain verbatim historical titles, not current scientific conclusions. Dates are displayed in UTC without local timezone shifts; submission dates use notification dates except the ICICC creation record. Source correspondence, addresses, email IDs, reviewer identities, verbatim review text and private portal links are excluded. Explicit review scores are attributed to their dated source and documented scale. No public paper availability or proceedings indexing claim is made, and the page performs no inbox or portal requests. Updating these facts requires verifying new evidence and reviewing the sanitized source diff; it does not regenerate stored experiment results or the historical HTML artifact.

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
DEMO_DATA=/tmp/samestack-demo.json npx vite build
DEMO_DATA=/tmp/samestack-demo.json python3 tests/test_demo.py
node tests/grade_all.mjs
```

- Shared fixtures check exact boxed/rational grading and Python/JavaScript statistics, including invalid, zero-variance and TOST-boundary cases; exact sign-flip enumeration is capped at 20 pairs
- Analysis/view fixtures cover absent arms, zero contrasts, one pair, disjoint/invalid provenance, duplicate seeds, interim completion, missing margins and adjustment gating
- UI structure tests cover named landmarks, navigation states, native selection controls, offline dependencies and theme text contrast
- Export tests verify the normative analysis path and immutable inputs; browser checks cover all views at 320/375/768/1024/1440px, canonical pending/available states, grader probes, keyboard focus/history, touch targets, reduced motion and text-spacing reflow
- `grade_all.mjs` is an audit against immutable historical scores. A nonzero exit can indicate genuine disagreement with the repaired grader, not a software regression. `--json --report-only` emits the discrepancies and coverage without failing the export. Never change the grader to force old scores to agree

`npm test` runs unit/contract/export checks, builds the repaired React source to `dist/index.html`, and targets that page for browser checks before the strict historical score audit. This test build does not replace the historical `tinkerrl_demo.html`; the same `DEMO_DATA` input is inherited by Vite and the browser assertions.

For direct browser checks, run the commands above from this directory after creating the temporary data export described in [Open and rebuild](#open-and-rebuild). `python3 tests/test_demo.py` defaults to `dist/index.html`; a missing page fails with build instructions before importing Playwright or launching a browser. Use `npx vite build` to build without replacing the frozen `tinkerrl_demo.html`. Set the same `DEMO_DATA` for the build and assertions, or omit it from both to check a source build using the historical `src/data.json`. An explicit `DEMO_PAGE=/path/to/built.html` overrides the page (relative paths resolve from the calling directory), and `CHROME_PATH` can select an installed browser. The entrypoint contract tests cover default selection, missing builds and overrides without importing Playwright or launching a browser.
