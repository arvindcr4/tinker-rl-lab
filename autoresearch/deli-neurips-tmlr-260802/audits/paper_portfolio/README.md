> **2026-08-02 zero-GPU freeze:** canonical spine is **P11** (see
> `drafts/PORTFOLIO_DECISION.md`, `drafts/PORTFOLIO_ROSTER_DISPOSITION.md`,
> `drafts/P11_NEURIPS_OVERLAP_CHECK.md`). Other active roots are demoted;
> do not treat the 12-PDF roster as a 12-submission queue.

# Paper portfolio evidence

This folder contains two different views of the manuscript set.

- `inventory.tsv`, `source/`, `text/`, `include_map.json`, and
  `similarity.tsv` are the frozen pre-consolidation review snapshot. That
  snapshot has 18 roots, 868 PDF pages, and 329 distinct included source files.
  Several paths were live when the snapshot was taken and have since moved to
  `platform_hybrid/paper/archive/absorbed/`; the old paths and hashes are kept
  here as review provenance, not as a current checkout manifest.
- `current_manifest.tsv` reconciles those same 18 review IDs to the current
  P1-P12 queue and the six absorbed archive roots. It records current paths,
  page counts, hashes, and PDF readability after the 2026-08-02 rebuild.

The current active queue totals 486 pages. The six absorbed archives total 412
pages and are history only. U01 is readable after repair, but its LaTeX build
still has unresolved citation warnings; it is not a venue candidate.
