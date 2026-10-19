# ICICC 2027 paper 243: author-review camera-ready candidate

Prepared 18 October 2026 from the identified historical source and retained repository evidence; finalized locally on 19 October 2026. This is an **AI-assisted author-review candidate**, not a submission or a certification of venue-policy compliance.

The accepted paper-243 title is retained by the author's 19 October decision; the proposed governance retitle is withdrawn. The review response distinguishes fixed, partly addressed and unresolved concerns. Build the manuscript with `./build.sh`, open `build/paper_camera.pdf` (12 pages in the checked build), then read `REVIEW_RESPONSE_AND_CHECKLIST.md`. The key unresolved release requirements are author/mentor approval, the organizer's restrictions on AI-generated prose and references/citations, and a licensed similarity report below 15%. No payment, registration, submission, copyright signature, new experiment or stored-result change was performed.

The paper-specific final page limit and sibling-publication policies also need confirmation. DeepReview finding 754 remains open: no retained record explains why AgentDojo injection attacks were omitted. The accepted title's “Held-Out” wording is retained as requested, but item-level decontamination is not established; this mismatch remains an author decision, not a verified holdout claim.

## Contents

- `paper_camera.tex`: editable manuscript, accepted paper-243 title (author decision 2026-10-19; proposed governance retitle withdrawn)
- `refs.bib`: 41 cited scholarly/project/platform references; full metadata and citation audit included
- `llncs.cls`, `splncs04.bst`: unmodified Springer class and bibliography style supplied with the recovered working draft
- `build.sh`: local-only LaTeX build; no network or model execution
- `verify_manuscript.py`: editorial/source/PDF checks; run after `./build.sh` because it reads the generated files in `build/`
- `REVIEW_RESPONSE_AND_CHECKLIST.md`: proposed reviewer responses and human release checks
- `SOURCE_MAP.md`: exact claim-to-source/evidence-level map and arithmetic checks
- `PROVENANCE.md`: reconstruction identity, changes and template checks
- `verification/`: bibliography/model identity provenance, complete inference inventory (JSON/CSV), paired evidence inventory, empirical findings, build/layout and package checks

## Build

Requires TeX Live or MiKTeX with `latexmk`, BibTeX, pdfLaTeX, Latin Modern, TikZ, microtype, booktabs, tabularx and xurl. Run:

```sh
./build.sh
```

Generated files are not included in the checkout. The PDF is written to `build/paper_camera.pdf` and the generated bibliography to `build/paper_camera.bbl`. Run `python3 verify_manuscript.py` after the build; this also requires Python 3 and Poppler (`pdfinfo`, `pdftotext`, `pdffonts`, `pdfimages`). Separate author-review delivery archives may include a PDF and bibliography copy. Optional offline regression tests use only the standard library: `python3 -m unittest discover -s verification -p test_manuscript_checks.py`. Do not use Python `-O`; the audit explicitly rejects disabled assertions. The optional fallback in the script handles this read-only Linux environment's missing TeX database/maps by constructing only a local build cache. It does not install software or edit global settings.

The manuscript's text and layout use standard Springer LLNCS dimensions on A4. No margin, baseline or body-font shrinkage was used to reach the page limit. Numeric records, historical experiment files and v2 comparisons are not part of this source archive and were not changed. The source map intentionally distinguishes historical statements from raw evidence that is unavailable in this public checkout.
