# Tinker RL Lab derived public coverage revision

The primary document is `Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-03_PUBLIC.pdf`, a derived public revision dated 3 October 2026. It preserves scientific results and limitations while excluding private operational records and identifiers. It is not a new institutional certification.

Editable Markdown, LaTeX, included figure sources and Python builders rebuild this document. C1 case ledgers preserve scientific classifications and event accounting. P11 bindings preserve scientific units, recorded public commits and manifest checksums; execution-specific access paths are omitted. `public_source_availability.json` specifies the included files and withheld evidence categories.

Source references are historical scientific references, not claims that all cited payloads are bundled or public. A document build does not reproduce the experiments. Private inventories, account activity, raw provider exports, private notebook links, recovery wrappers and access correspondence are excluded. Original historical files were not modified.

## Rebuild

Use Python 3.11 or later, Pandoc, Tectonic 0.17.0 and a populated writable TeX cache. Put the tools on PATH. From this directory run:

    python3 compile_figures.py --force
    python3 build_thesis.py

The sources use NewTX, TikZ/PGFPlots and TeX Live packages listed in the preambles. Generated `thesis_master.tex`, build files and logs are excluded from the publication allowlist. Pagination can change with font and tool versions.
