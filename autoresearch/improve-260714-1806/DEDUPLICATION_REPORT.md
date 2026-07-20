# Manuscript deduplication report (2026-07-14)

## Outcome

The 18 canonical manuscript roots now expand to 328 unique included source
files, down from 353 before consolidation. All roots still compile and the
final corpus contains 864 rendered pages after restoring every checked-in
figure to the affected canonical PDFs.

## Removed copies

- 23 legacy `group_size_iter*.tex` sections that repeated one reconstructed
  group-by-token grid already represented by the canonical synthesis.
- Two P3 analogy sections that did not add independent evidence.
- Eight copied TikZ sources; the canonical `paper/tikz/` versions remain.
- Anonymous and venue-local copies of introductions, related work, conclusions,
  statistical-rigor text, tool-use text, figure notes, and pipeline text.
- One venue-local bibliography; all NeurIPS variants now use the canonical
  shared bibliography.
- 154 redundant LaTeX build intermediates from the review bundle. Its
  `build/` tree now retains exactly one canonical PDF per manuscript; final
  logs remain beside the canonical TeX roots.

The working-tree diff records 44 tracked deletions and 9,842 deleted lines in
the complete paper-improvement pass. Shared text with legitimate contextual
variation was parameterized instead of copied.

## Integrity checks

- Exact duplicate groups among `.tex`, `.bib`, and `.md` manuscript files: 0.
- Missing TeX input hooks across canonical roots: 0.
- Unresolved citation keys across canonical roots: 0.
- Duplicate active labels within any canonical root: 0.
- Canonical PDFs compiling successfully: 18 of 18.

Source-level placeholder matches are dormant `\IfFileExists` fallback branches
or prose that transparently describes unavailable external evidence. A rendered
PDF scan finds zero active figure fallbacks across all 18 manuscripts. R08 emits
no result cells until the preregistered audit is executed.

The per-file evidence is recorded in `FILE_REVIEW.tsv`; review flags and corpus
totals are in `SELF_REVIEW_FLAGS.md` and `self_review_summary.json`.
