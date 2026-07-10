# paper/figures/ — INDEX

**Purpose:** Figure assets (PNG/PDF) plus their Python generators. This top level is the older/mixed set; the paper references the curated `v2/` copies for most plots. Naming: `<plot_name>.{png,pdf}` with a matching generator script.

**Key files:**
- `generate_figures.py` — main figure generator (24 KB); `gen_figures.py` — older/smaller variant
- `wave6_sensitivity.py` → `wave6_sensitivity.png/.pdf` — Wave-6 sensitivity heatmap (referenced by main.tex)
- `learning_curves.*`, `comparison_bars.*`, `performance_profiles.*`, `sensitivity_heatmap.*`, `old_trl_seeds.pdf` — core result plots (PDF versions used)
- `scaling_law_figure.png`, `scaling_params_figure.png`, `scaling_plot.png` — scaling-law plots (large PNGs)
- `zvf_heatmap.png`, `zvf_correlation.png`, `effect_sizes_forest.png`, `reward_stability.png` — ZVF / stats plots used from here
- `architecture.tex`, `pipeline.tex`, `reward_flow.tex`, `taxonomy.tex` — TikZ diagram sources (duplicated/superseded by `../tikz/`)

**Subfolders:**
- `v2/` — submission-quality regenerated figures, 300dpi PNG + vector PDF, colorblind-safe (see its INDEX.md)

**Find it fast:**
- for the figure the paper actually embeds → prefer `v2/` (main.tex uses `figures/v2/*.pdf`)
- to regenerate → `../../scripts/regenerate_figures.py` (v2) or `wave6_sensitivity.py` (wave6)
- to check used vs unused → `../FIGURE_AUDIT.md`
