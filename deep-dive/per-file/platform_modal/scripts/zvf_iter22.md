# Deep Dive: `platform_modal/scripts/zvf_iter22.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_modal/scripts/zvf_iter22.py` (700 lines)

## Overview
`zvf_iter22.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **config, csv, numpy, protocol, viz** to do its work.
*Self-description:* "Iter 22 Pillar 2 ZVF elevation.  Two new empirical questions that the existing zvf_diagnostic.py does not answer:    (A) Per-library iterated bootstrap CIs on m"

## Key Components
- `load_variance_mitigation_per_seed()` -- Per-step rows grouped by (method, seed).
- `per_seed_mean_zvf()` -- function
- `per_seed_collapse_rate()` -- function
- `bootstrap_mean_ci()` -- Percentile bootstrap CI on the mean.
- `compute_library_bootstrap_ci()` -- For each method, compute mean_ZVF CI and collapse_rate CI from seeds.
- `find_first_collapse()` -- Return (peak_pass_step, first_collapse_step, lead_steps, mean_zvf_at_pass).  "peak_pass_step" = argmax(heldout_acc) over the entire trajecto
- `find_max_zvf_in_window()` -- Return (argmax step, max zvf) in [t_start, t_end]. Returns NaN if empty.
- `compute_leadtime()` -- For each (method, seed), compute collapse lead-time and ZVF-at-pass.  Also computes the local ZVF max in the [pass, pass+lead_window] window
- `write_leadtime_table()` -- function
- `aggregate_leadtime()` -- function
- `write_leadtime_summary()` -- function
- `wilcoxon_signed_rank()` -- Returns (W+, p_two_sided) by a normal approximation.  For n < 10 we just return the sum of positive ranks and a Gaussian approximation p-val
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
