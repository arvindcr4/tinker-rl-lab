# Deep Dive: `platform_modal/scripts/zvf_discrimination_iter110.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_modal/scripts/zvf_discrimination_iter110.py` (826 lines)

## Overview
`zvf_discrimination_iter110.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **config, csv, numpy, protocol, viz** to do its work.
*Self-description:* "zvf_discrimination_iter110.py - Pillar 2 (ZVF): discrimination + iso-G sizing.  Builds on iter94/98/102/106 (cross-library calibration dashboard) and extends wi"

## Key Components
- `_strip_comments()` -- function
- `load_calibration_rows()` -- Re-use the iter102 per-row calibration table (same parser as iter106).
- `load_leadtime_rows()` -- function
- `load_groupsize_step_log()` -- For each of the 12 group-size runs (G in {2,4,8,16} x 3 seeds) with per-step ZVF and reward in groupsize_zvf_sweep.json, compute first-passa
- `_classify_row()` -- Same labels used by iter102/106 (must stay in sync).
- `_auc_roc()` -- Mann-Whitney U / AUC. Ties -> 0.5 attribution.
- `_bootstrap_auc_ci()` -- function
- `_em_distance_1d()` -- Exact 1-Wasserstein / EMD on the real line for two empirical samples.
- `_normalised_em()` -- EMD normalised by the empirical range of the full sample so the number is in roughly [0, 1] and comparable across predictors.
- `_spearman()` -- function
- `_iso_g_floor()` -- For each target saturation floor tau and each p-bin midpoint, find the minimum G in 1..G_MAX such that p^G + (1-p)^G <= tau.  Returns one ro
- `_write_tsv()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### CSV I/O
- **What**: `csv` reads/writes comma-separated records, the lingua franca for tabular data and result dumps.
- **Why used here**: Large benchmark/results files are exchanged as CSV, so importing/exporting that format is a direct requirement.
- **When**: When tabular data must be human-openable or compatible with spreadsheets/other tools.
- **Trade-offs**: CSV has no schema or types -- every field is a string, so parsing and quoting edge cases are on you.

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Data visualization
- **What**: Matplotlib/Plotly render metrics into figures, replacing dense number tables with readable curves.
- **Why used here**: The repo produces decks and figures as code so charts derive from evidence and regenerate whenever the checkout changes.
- **When**: When a comparison (scaling curve, loss trace, ablation) is clearer as a picture than a table.
- **Trade-offs**: Figures need explicit styling to stay trustworthy; a miscalled axis or log scale can misrepresent the claim.


## Related Code
- sibling `platform_modal/scripts/_reviewer_points_extract.py`
- sibling `platform_modal/scripts/anonymize.sh`
- sibling `platform_modal/scripts/build_submission.py`
- sibling `platform_modal/scripts/build_university_submission.py`
- sibling `platform_modal/scripts/contamination_check.py`
- sibling `platform_modal/scripts/ed25519-sign.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
