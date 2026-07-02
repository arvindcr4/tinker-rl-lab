# paper/ — INDEX

**Purpose:** Canonical LaTeX source for the NeurIPS 2026 paper "A Unified Benchmark for RL Post-Training of Language Models" (GRPO scaling study). Holds the main manuscript, its anonymized twin, per-section inputs, figures, and reviewer-response tooling.

**Key files:**
- `main.tex` — canonical manuscript (135 KB); `\input`s `sections/*`, embeds `tikz/*.pdf` + `figures/v2/*.pdf`
- `main_anon.tex` — anonymized twin for blind review (authors/affils stripped)
- `main.pdf` / `main_anon.pdf` / `supplement.pdf` — compiled outputs
- `references.bib` — master bibliography (57 KB); `bib_fragments/*` get concatenated in
- `ethics_statement.tex` (+`_anon`, +`ethics_wrapper.tex`) — standalone ethics section
- `neurips_checklist_update.tex`, `limitations_update.tex` — checklist + limitations blocks
- `reviewer_points.yaml` — reviewer weakness registry; markers `%REVIEW-ADDR: <id>` grepped in `.tex` for scoring
- `FIGURES.tex` / `FIGURE_AUDIT.md` — figure registry (path→regenerator script) + used/unused audit (30 used of 64)
- `expected_results.json` — expected headline metrics for audits
- `acm_main.tex` — alternate ACM-format build
- `neurips_2026.sty` / `neurips_2025.sty` / `neurips_2024.sty` — style files (2024 is a stub)

**Subfolders:**
- `sections/` — per-section + appendix `.tex` inputs, anon variants (see its INDEX.md)
- `figures/` + `figures/v2/` — plot assets + generators; v2 is the submission-quality set (see INDEX.md)
- `tikz/` — hand-drawn diagram `.tex`→`.pdf` (taxonomy, pipeline, reward flow) (see INDEX.md)
- `bib_fragments/` — extra `.bib` snippets woven into `references.bib` (see INDEX.md)

**Find it fast:**
- to edit paper text → `sections/*.tex` (not inline in main.tex)
- to see which figure a `\includegraphics` uses → `FIGURES.tex` or grep `main.tex`
- to answer a reviewer point → `reviewer_points.yaml` (find id, add marker)
- to build blind version → `main_anon.tex`
