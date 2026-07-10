# 0001 — Figure provenance before consolidation

## Status

Accepted

## Context

The repo has multiple overlapping figure-generation scripts:

- `platform_modal/scripts/regenerate_figures.py` writes eight v2 figures from `experiments/master_results.json`.
- `platform_modal/scripts/regenerate_measured_figures.py` writes a measured-artifact subset to `paper/figures/v2/`.
- `platform_modal/scripts/make_real_figures.py`, `platform_modal/scripts/make_paper_figures.py`, `platform_modal/scripts/generate_figures.py`, and `platform_modal/scripts/regenerate_missing_figures.py` write older or missing figures with different data sources, styles, and fallback behavior.

`paper/main.tex` references more than those eight v2 figures, including `performance_profiles`, `scaling_law_figure`, `scaling_params_figure`, `effect_sizes_forest`, `zvf_heatmap`, `zvf_correlation`, `reward_stability`, and `old_trl_seeds`.

## Decision

Do not collapse the figure scripts into a single eight-figure module until provenance is reconciled against `paper/main.tex`.

Future work should first map each figure reference in the paper to:

- the authoritative data source,
- the figure-generation module or script,
- measured vs simulated vs canonical fallback status,
- output path expected by LaTeX,
- whether the current script is still authoritative or should be retired.

Only after that map is complete should the repo consolidate into a **figure module** with **results adapters** and, if needed, a **fallback adapter**.

## Consequences

- Architecture reviews should not re-suggest “one eight-figure module” as a complete solution.
- `platform_modal/scripts/regenerate_figures.py` remains the best candidate for the core figure module, but not yet the authoritative source for every paper figure.
- Consolidation without a provenance map risks hiding placeholder data, stale scripts, or missing figure references.
