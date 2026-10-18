# Thesis source review revision, 18 October 2026

This is the authoritative source revision of the full M.Tech report, based on
`276f14c23b8f7e90c4f048d531afdce194c19993`. It addresses three review packets
across all nine chapters and the appendices. It is an unsigned evaluation draft,
not institutional approval or certification of the author's declarations.

## Scientific scope

The revision withdraws contaminated arithmetic estimator claims, interrupted-run
scaling fits, synthetic/imputed predictive validation and retrospective controller
rankings from empirical support. It distinguishes historical variance conventions,
cap-censored length observations, current code from unbound historical execution,
and original campaign contracts from exploratory protocol-override replacements.
The campaign is an infrastructure/harness audit. Unavailable source, receipts and
unexecuted research controls remain explicit gaps. Stored experimental records
were not changed. The complete provisional SAMESTACK-V2 block and Table 6.2 remain
byte-identical to the starting source.

## Build

Run `./build_portable.sh` here. It requires Python 3.11+, Pandoc and either Tectonic
with its populated cache or latexmk/pdfLaTeX/BibTeX with the relevant TeX packages.
The fallback supports the selected Linux image's installed TeX files using a local
`build/texmf` format/font cache; it does not install software or run network jobs.
It preserves a usable system font map when present. Emergency synthesized maps select the known-mapped mathptmx/Latin Modern profile rather than newtx. Outside that profile both newtx text/math packages must exist, otherwise mathptmx is checked and Latin Modern retained as the final fallback. The builder detects `IEEEtran` or `ieeetr` for latexmk, and rejects an explicitly requested missing style.
All figures are compiled from their `.tex` sources; the obsolete E4 histogram is
not injected. The output is
`Thesis_Report_ArvindCR_2026-10-18_REVISED_REVIEW.pdf`.

`thesis_master.tex`, `build/`, compiled figures and the PDF are derived outputs;
edit the chapter Markdown, figure TeX, bibliography and build sources instead.
A fresh build must be checked for unresolved citations, document completeness and
rendered layout. Appendices I and J and the older availability/provenance JSONs
are historical snapshots, not a new passing audit of this revision. A source-only
package can rebuild the document but cannot reproduce missing experiments.

## Publication boundary

The source revision includes no private review attachments, withheld receipt
archive, credentials or provider exports. Existing public author details are
preserved. The Library delivery has a separate source-package integrity manifest,
full PDF, chapter excerpts and point-by-point review response. File hashes establish
file identity, not historical execution authenticity or scientific validity.

## Current PDF consumers and historical review-pack boundary

`current_artifact.json` is the shared output/receipt contract for the thesis build,
`../check_pdf.py` and `tools/build_pes_review_pack.py`. From the repository root,
`python3 reports/project_finish_2026-10-11/check_pdf.py` checks the newly built PDF
and writes `validation/thesis_2026-10-18/pdf_checks.json`; `--pdf` and `--receipt`
allow explicit alternatives. An automatic rerun resets visual review to pending.
The historical `validation/pdf_checks.json` is preserved, not relabelled.

The broad October PES pack also requires its separate number-audit and deck
bindings. Those historical receipts have not been regenerated for this revision;
the pack must fail closed on stale bindings. The delivered standalone thesis
source ZIP is the current rebuildable source package, not a refreshed broad PES
review archive or a new numerical-audit PASS.
