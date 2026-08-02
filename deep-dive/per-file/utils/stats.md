# Deep Dive: `utils/stats.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `utils/stats.py` (374 lines)

## Overview
`stats.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, numpy, pandas, protocol, viz** to do its work.
*Self-description:* "Statistical Analysis Tooling for RL Experiments ================================================= Implements rliable-based aggregate metrics, bootstrap confiden"

## Key Components
- `load_multi_seed_results()` -- Load results from multiple seeds for a given experiment.  Expected directory structure:     results/<experiment>/seed_<N>/metrics.jsonl  Ret
- `compute_bootstrap_ci()` -- Compute bootstrap confidence interval for the mean.  Args:     scores: Array of scores (one per seed/run).     n_bootstrap: Number of bootst
- `welch_ttest()` -- Welch's t-test for comparing two algorithms. Recommended over Student's t-test when variances may differ.  Reference: Colas et al. (2019), S
- `mann_whitney_u()` -- Mann-Whitney U test (non-parametric alternative to t-test). Use when distributions may be non-normal.  Reference: Colas et al. (2019), Secti
- `plot_learning_curves_with_ci()` -- Plot learning curves with shaded confidence bands (±1 SE).  Args:     results: Dict[algorithm_name -> Dict[seed -> List[step_metrics]]]     
- `generate_results_table()` -- Generate a LaTeX table with mean ± SE and bootstrap CIs.  Args:     results: Dict[algorithm_name -> array of final scores across seeds]     
- `try_rliable_analysis()` -- Run rliable aggregate metrics if the library is available.  Reference: Agarwal et al. (2021) https://arxiv.org/abs/2108.13264
- `main()` -- function
- `compute_iqm()` -- Compute the Interquartile Mean (IQM) of a score array.  Follows Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice". D
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

### Tabular data with pandas
- **What**: pandas DataFrames hold labeled, columnar data and offer groupby/merge/agg one-liners over CSV/JSON exports.
- **Why used here**: Experiment logs and result exports aggregate nicely into tables for reporting and audits.
- **When**: When you'd otherwise hand-roll loops over rows of CSV/JSON results.
- **Trade-offs**: DataFrames are heavier than raw arrays; overuse for tiny data adds import cost and ambiguity about index semantics.

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
- sibling `utils/__init__.py`
- sibling `utils/audit_utils.py`
- sibling `utils/seed.py`
- sibling `utils/tinker_grpo.py`
- sibling `utils/verify_results.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
