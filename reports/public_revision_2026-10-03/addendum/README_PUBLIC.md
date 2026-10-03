# C1 scientific addendum: PUBLIC revision

This distinct public derivative preserves the scientific findings of the 3 October 2026 research addendum. The frozen parser result is 11/64 (17.19%); the conservative corroborated numeric witness count is 3/64 (4.69%). These are different quantities. No training intervention or training gain was measured.

## Public files

- `C1_Research_Addendum_2026-10-03_PUBLIC.pdf`: readable scientific addendum
- `C1_Research_Addendum_2026-10-03_PUBLIC.md`: editable manuscript
- `C1_Research_Addendum_2026-10-03_PUBLIC.tex`: generated LaTeX source
- `build_public_addendum.py`: self-contained Markdown-to-PDF builder
- `case_ledger_public.json`: all 64 classifications, 11 event records and 126 event-response checks
- `case_ledger_public.csv`: compact 64-case classification table
- `scientific_method_public.json`: benchmark/model revisions, generation settings and selected-case seeds

## Rebuild

Install Python 3, Pandoc and Tectonic, ensure `pandoc` and `tectonic` are on PATH, then run `python build_public_addendum.py` in this directory. Tectonic may need access to its standard TeX package bundle or a previously populated cache. Standard XDG cache/configuration environment variables are respected. The builder reads only the adjacent public Markdown and writes generated LaTeX, a PDF and intermediate files in `build/`.

## Scientific limits and source availability

The public document does not provide full execution provenance. It omits the complete raw response corpus. Response-content hashes are retained scientific identifiers, but cannot independently reproduce the response judgments without the corresponding texts. Seed values do not guarantee bitwise reproducibility across systems. All review stages were assistant-assisted and are not independent human validation. The 64-question cohort was selected using frozen exact-match zeros and deterministic truncation; it cannot estimate benchmark-wide reference-error prevalence or population recovery.

The original research records are unchanged. The public ledger retains the first/second review disagreements, unit-sensitive cases and visible-reasoning caveats. Generated build and QA intermediates are not publication inputs.
