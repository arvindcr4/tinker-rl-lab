# ICICC 2027 paper 243: author-review camera-ready candidate

Prepared 18 October 2026 from the identified historical source and retained repository evidence. This is an **AI-assisted author-review candidate**, not a submission or a certification of venue-policy compliance.

Revised against the full new critique, including a proposed title change and an exhaustive fixed/partly-addressed/unresolved response. Start with the 12-page PDF, then read `REVIEW_RESPONSE_AND_CHECKLIST.md`. The key unresolved release requirements are author/mentor approval, the organizer's restrictions on AI-generated prose and references/citations, and a licensed similarity report below 15%. No payment, registration, submission, copyright signature, new experiment or stored-result change was performed.

## Contents

- `paper_camera.tex`: editable manuscript, proposed governance-focused title; author/venue approval required
- `refs.bib`: 37 cited scholarly/project/platform references; full metadata and citation audit included
- `llncs.cls`, `splncs04.bst`: unmodified Springer class and bibliography style supplied with the recovered working draft
- `build.sh`: local-only LaTeX build; no network or model execution
- `verification/paper_camera.bbl`: generated bibliography included as a verification copy in the delivery archive
- `REVIEW_RESPONSE_AND_CHECKLIST.md`: proposed reviewer responses and human release checks
- `SOURCE_MAP.md`: exact claim-to-source/evidence-level map and arithmetic checks
- `PROVENANCE.md`: reconstruction identity, changes and template checks
- `verification/`: bibliography/model identity provenance, complete inference inventory (JSON/CSV), paired evidence inventory, empirical findings, build/layout and package checks
- `ICICC2027_Paper243_AuthorReview.pdf`: final PDF included in the delivery archive

## Build

Requires TeX Live or MiKTeX with `latexmk`, BibTeX, pdfLaTeX, Latin Modern, TikZ, microtype, booktabs, tabularx and xurl. Run:

```sh
./build.sh
```

The PDF is written to `build/paper_camera.pdf`. The optional fallback in the script handles this read-only Linux environment's missing TeX database/maps by constructing only a local build cache. It does not install software or edit global settings.

The manuscript's text and layout use standard Springer LLNCS dimensions on A4. No margin, baseline or body-font shrinkage was used to reach the page limit. Numeric records, historical experiment files and v2 comparisons are not part of this source archive and were not changed. The source map intentionally distinguishes historical statements from raw evidence that is unavailable in this public checkout.
