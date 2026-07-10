# platform_local/contexts/research-engineering/platform_hybrid/docs/adr/ — INDEX

**Purpose:** Architecture decision records scoped to the research-engineering context.

**Key files:**
- `0001-figure-provenance-before-consolidation.md` — Accepted. Multiple overlapping figure scripts exist (`platform_modal/scripts/regenerate_figures.py`, `platform_modal/scripts/regenerate_measured_figures.py`, `platform_modal/scripts/make_real_figures.py`, etc.) and `platform_hybrid/paper/main.tex` references more figures than any one produces. Decision: do NOT collapse them into a single figure module until provenance is reconciled against `platform_hybrid/paper/main.tex`.

**Find it fast:**
- to know why figure scripts aren't merged → `0001-*.md`
