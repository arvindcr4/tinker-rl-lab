# Current Phase 2 thesis

The canonical academic deliverable is **Thesis_Report_ArvindCR.pdf**.
Start with the [submission guide](../../../SUBMISSION.md) for the evidence
boundary, current package, and required author approvals.

## Sources

- `frontmatter.tex`: title, certificate, declarations, abstract.
- `ch*.md`: chapters and appendices listed in `build_thesis.py`.
- `references.bib`, `preamble.tex`, `figures/*.tex`, and the two PNG assets.
- `build_thesis.py`: chapter conversion, figure placement, evidence map, PDF build.

`thesis_master.tex` and the PDF are generated. Do not edit them directly.
`assemble_thesis.py` is an older Markdown/DOCX exporter with a stale abstract and
chapter order. `ch09_FULL_results_alternate.tex.md` is an unused draft. Neither
belongs to the submission build.

## Rebuild from the repository root

Python 3.11+, Pandoc, and Tectonic are required. A warm Tectonic cache permits an
offline build; first-time TeX package downloads need network access.

```bash
uv run --no-sync python outputs/PES_Phase2_Third_Review_2026-09-24/thesis/compile_figures.py --force
uv run --no-sync python outputs/PES_Phase2_Third_Review_2026-09-24/thesis/build_thesis.py
```

Always rebuild figures from their sources before a submission build; checkout
mtime order is not evidence that an ignored figure PDF is current. The package
builder does this automatically. Missing required inputs, failed conversions,
or failed PDF compilation are errors, not partial success. A failed compile
must not replace the previously published thesis PDF.

The certificate and AI declaration still need author/guide confirmation. This
is an identified university report, not a blind-review conference paper.
