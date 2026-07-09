# contexts/research-engineering/docs/adr/ — INDEX

**Purpose:** Architecture decision records scoped to the research-engineering context.

**Key files:**
- `0001-figure-provenance-before-consolidation.md` — Accepted. Multiple overlapping figure scripts exist (`scripts/regenerate_figures.py`, `regenerate_measured_figures.py`, `make_real_figures.py`, etc.) and `paper/main.tex` references more figures than any one produces. Decision: do NOT collapse them into a single figure module until provenance is reconciled against `paper/main.tex`.

**Find it fast:**
- to know why figure scripts aren't merged → `0001-*.md`
