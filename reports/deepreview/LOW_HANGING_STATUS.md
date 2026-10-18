# DeepReview editorial pass — 18 October 2026

Editorial branch: `review-low-hanging-2026-10-18`. This report records the
editorial pass; the subsequent main integration was authorized by the user.
No additional DeepReview API calls were made.

## Minor-finding follow-up

Branch `review-minor-followup-2026-10-18` corrects two remaining P8
presentation findings: **690** now displays the TRL row and caption in
percentages, and **694** uses group size G consistently and defines the
expectation over prompts drawn from the difficulty distribution. The five
released TRL seed accuracies independently reproduce the displayed range
62--81%, mean 73.4%, and sample SD 7.0 percentage points. This is a unit
conversion, not new experimental evidence.

Seventeen stale open dispositions are verified as **already-fixed** in the
current active sources and rebuilt PDFs: P8 **691, 692, 693, 698--702, 704**;
P6 submission **780--784, 787, 788, 791**. In particular P8 now includes its
workshop-specific ethics statement, not the older shared statement. The
removed failed-spend/carbon claims are not replaced with guessed values.
The mixed-case model identifiers in P6 remain literal registry strings, and
its longer anonymous registry twin still has a stale prose CLI flag; neither
is silently normalized or claimed corrected here. Unverified statistical
tables and cost totals remain open.

P8 and all four P6 roots (submission, submission_anon, registry,
registry_anon) build using cached Tectonic resources, with no undefined
references or citations. P8 remains 13 pages; P6 submission and its anonymous
twin remain 16 pages with References starting on page 10 (body stays within
9 pages). Registry and registry_anon remain 16 and 65 pages respectively.
The rendered P8 headline table and appendix equations were inspected for
clipping, percentages, group-size notation, brackets and thresholds. P6
compiled text verifies the CLI hyphens, bibliography spelling/capitalization,
and appendix heading plus introductory sentence before its table.

All 1,204 finding IDs remain covered exactly once. Non-target disposition
records are unchanged. The peer's truncation re-evaluation text and all
empirical results are preserved. Logs, disposable PDFs and inspected captures
are under `.amp/in/artifacts/minor-followup/` in the isolated
editorial-minor worktree. No new DeepReview calls were made.

## Bibliography follow-up

Branch `review-bibliography-followup-2026-10-18` applies the remaining verified
clerical bibliography requests centrally, after the other session released
ownership. Twelve shared bibliography entries lose duplicate identifier notes,
informal annotations or inline license tags; xLAM's AI capitalization is
protected. Seven standalone spectral bibliography titles protect proper nouns.
Citation keys, title words, authors, years, venues, DOIs and URLs are unchanged.
Dataset licensing prose and all empirical claims are untouched.

Findings **663, 705 and 706** are now fixed. The current P8 bibliography no
longer contains the formerly cited xLAM, Patterson, Riddell or HybridFlow
entries; the shared xLAM correction was additionally verified in P9, which
still cites it. Findings **509 and 893** remain open: their clerical subparts
are corrected, but proposed publication metadata or author-order changes were
not independently verified. Historical request files below describe the state
before this follow-up; per-finding JSONL records contain the latest outcome.

All twelve manifest roots using the shared bibliography and the corrected
standalone spectral root build successfully with cached Tectonic resources,
with no undefined references or citations. The standalone spectral build
stages its unchanged root, edited bibliography and existing NeurIPS style
together; no missing dependency was replaced with a stub. Inspected P11,
P8, P9 and spectral reference pages show the intended formatting. Existing
font/box warnings remain. P11 now renders the independently revised fail-closed
manuscript; this is not certification of any audit result.

Assertions check that removed duplicate arXiv notes retain the same identifier
in the DOI field, protected terms survive BibTeX casing, unrelated disposition
records and review provenance stay unchanged, and all **1,204** finding IDs
remain covered exactly once. Build logs, disposable PDFs and inspected page
captures are under `.amp/in/artifacts/bibliography-followup/` in the isolated
editorial-bibliography worktree. No paid review calls were made.

## Main integration

Newer substantive P4 and Sem4 dispositions, plus the P4 numeric ledger, take
precedence over this pass's editorial scope deferrals. These three conflict
resolutions preserve the corresponding main records byte-for-byte. After
integration, the same 979 finding IDs are covered exactly once: **42 fixed,
129 already-fixed, 1 needs-compute, 807 wontfix-scope**. The original
editorial-only counts below remain a historical account of this pass.
Uncommitted work in the original main checkout is excluded from this merge.

## Scope and outcome

The 22 assigned manuscripts have 979 integer findings in `findings.sqlite`.
Their disposition JSONL files cover each ID exactly once: **32 fixed,
23 already-fixed, 924 wontfix-scope**. Scope-deferred means still open,
not resolved. Compound findings remain open when only their editorial
subpart was addressed. No empirical numbers, experiments or citations were
invented or changed. Frozen reviewed PDFs and the entire locked thesis
Chapter 6 were preserved.

The other active session owns ICICC, P1 identifiability, P2 falsification,
P6 submission, P8 workshop and P9 DNB, including their editorial fixes.
This report does not certify that session's work.

## Source delivery

Tracked manuscript edits and per-finding dispositions are on this branch.
The three conference sources live in ignored `outputs/` directories, behind
tracked aliases (including an absolute IEEE symlink). Those aliases were
not changed and the original checkout was not written through them.
Edited conference sources exist in this worktree's local `outputs/` copies.
Their portable, committed delivery is
`dispositions/low_fruit_conference_sources.patch`, not tracked source files.
Integration into another checkout requires an explicit patch application:

```sh
git apply --check reports/deepreview/dispositions/low_fruit_conference_sources.patch
git apply reports/deepreview/dispositions/low_fruit_conference_sources.patch
```

The patch passed a forward applicability check against the original checkout
and a reverse applicability check against the edited worktree. Recheck after
concurrent changes before applying; do not overwrite another session's edits.
Shared bibliography, shared-section and TikZ requests are recorded in
`dispositions/low_fruit_*_shared_requests.md`; they remain unapplied.

## Verification

- Eleven of twelve edited manuscript roots build with Tectonic; the final
  logs contain no undefined references or citations. Existing font, box and
  bibliography warnings remain.
- P11 fails at the same certified-results gate on both baseline and edited
  sources. No placeholder results were introduced to bypass it.
- P12 grows from 13 to 14 pages. P7 remains 22 pages; final RAIT and IEEE
  remain 4 and 5 pages respectively. Venue-length compliance is not certified.
- Parent inspected rendered P12 equations, ACL definition/caveat, thesis GU
  caption, and RAIT/IEEE equations. Reflowed the expanded conference equations
  after inspection found gutter overflow, rebuilt both, and inspected the
  final captures again; the equations now fit their columns.
- Disposition coverage, locked Chapter 6 byte identity, patch applicability,
  and `git diff --check` pass.

Build logs and disposable PDFs are under this worktree's
`.amp/in/editorial-builds/`, `.amp/in/theory-builds/`, and
`.amp/in/low-fruit-builds/`. Inspected captures are in
`.amp/in/artifacts/editorial-checks/`. These are ignored local artifacts,
not replacement release PDFs.

The full original 28-paper API reports remain in the original checkout at
`.amp/in/artifacts/deepreview/INDEX.md` and `ALL_REVIEWS.md`.
