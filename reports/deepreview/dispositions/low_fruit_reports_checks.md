# Reports editorial pass: verification and handoff

Branch: `review-low-hanging-2026-10-18`. No commits, pushes, installs,
training, paid/API/model calls, or writes outside this worktree were made.
Other workers' changes on this branch were left untouched.

## Completed edits

- ACL `body.tex`: **720, 721** — exact binary-outcome ZVF definition,
  positive-stabiliser/mean-centering qualification, filtering measurement
  caveat, and consistent backend spelling.
- RAIT `body.tex`: **819–821** — explicit summation bounds and completion
  indices, positive stabiliser, distinction from variance threshold, G in
  place of temporary K in the diagnostic definition.
- IEEE `paper.tex`: **868, 869** — same self-contained indices/bounds and
  stabiliser, batch cardinality definition, supported bold indicator numeral.
- Current thesis: **974, 976, 977, 989** — Spearman label in Table 1.1,
  math-mode K/binomial and G in §2.1, and GU-complement caption in the builder.
- C1 **1013** already has the requested equation spacing in current Markdown;
  no change was needed. No historical Sem3/Sem4 material was transplanted.

Eight per-paper JSONL files cover **all 344 integer DB finding IDs exactly
once**: 11 fixed, 1 already-fixed, 332 wontfix-scope. Scientific, statistical,
provenance, removed-historical-source and shared-source requests are not
misrepresented as completed editorial fixes. The thesis TSV ledger records
the two numerical statements whose labels were clarified; values were not
changed or newly recomputed, and the historical correlation remains withdrawn.

## Source-path caveat: conference outputs are ignored

ACL/RAIT directory symlinks pointed to absent ignored output directories.
The tracked IEEE root symlink resolves to the original checkout, outside
this worktree. It was **not written through**. Necessary conference source
dependencies were copied read-only into the corresponding real local
`outputs/` directories, then only local body/root files were patched.
Bibliography/style/figure copies are byte-identical to their originals.

The three edited conference files therefore do not appear in ordinary
`git diff`, because `outputs/` is ignored. Their exact changes are preserved
in **`low_fruit_conference_sources.patch`**, with repository-relative paths.
The parent must explicitly decide how to commit these source changes; do not
assume a commit of tracked thesis/disposition files includes them. No reviewed
conference PDF was copied over or rebuilt in place.

## Builds

All four changed roots compiled successfully, exit 0, with cached resources:

1. `tectonic -X compile .amp/in/low-fruit-builds/ACL/paper_camera.tex --only-cached --keep-logs --outdir .amp/in/low-fruit-builds/ACL`
2. Same command for RAIT `paper_camera.tex`.
3. Same command for IEEE `paper.tex`.
4. `python3 .amp/in/low-fruit-builds/Thesis/build_thesis.py`; this invokes
   `tectonic -X compile` with `--only-cached --untrusted` after Pandoc.

Scratch source copies were made from **this worktree's edited sources**, not
the original staged thesis/Sem3. Thesis was rebuilt again after the final
Chapter 2/caption edits. Full logs and disposable PDFs remain in the isolated
`.amp/in/low-fruit-builds/` tree. No undefined references or citations and no
TeX errors were found in any of the four final logs. Font substitution,
box/layout and PDF-version warnings remain; this is not a layout-polish pass.
`pdftotext` confirms the IEEE indicator extracts as `1 [σg = 0]`.
`git diff --check` passes.

## Locked/frozen content

Byte comparison against branch HEAD passes for the entire Chapter 6 and the
reviewed thesis PDF:

- `ch06_results_core.md`: SHA-256
  `63ddfaf915774d0cf63671dcee713c0a4229deb3ce86f48a6001fd55f8b7173c`.
- Reviewed thesis PDF: SHA-256
  `83d8ee3a73a7073f17b80d260595b2d11a27f8d750c369b784326d94734980e0`.

Only the identified label/notation/caption changes were made outside Chapter 6;
locked provisional blocks were not edited. In particular **997–1003** remain
scope-deferred despite being cheap: explicit Chapter 6 locking takes priority.
Shared bibliography requests are recorded separately and remain unapplied.
