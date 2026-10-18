# DeepReview editorial pass — 18 October 2026

Branch: `review-low-hanging-2026-10-18`. Local changes only; not merged,
pushed, or published. No additional DeepReview API calls were made.

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
