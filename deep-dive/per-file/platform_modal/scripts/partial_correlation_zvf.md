# Deep Dive: `platform_modal/scripts/partial_correlation_zvf.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_modal/scripts/partial_correlation_zvf.py` (633 lines)

## Overview
`partial_correlation_zvf.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, numpy, protocol** to do its work.
*Self-description:* "Partial-correlation ablation for the ZVF diagnostic.  Computes partial corr(ZVF_t*, final_reward) while controlling for   * batch mean reward   * policy entropy"

## Key Components
- `_iter_logs()` -- function
- `_load_jsonl()` -- function
- `_extract_zvf_row()` -- Project a heterogeneous log record into our canonical schema.
- `_ols_residualize()` -- Return y - X @ beta, where beta is the OLS coefficient vector.
- `_partial_corr_pearson()` -- Return (r_partial, ci_low, ci_high, p_value) for a Pearson partial correlation.  Bootstrap CIs refit the residualizations on each resampled 
- `_normal_cdf()` -- Acklam-style normal CDF approximation; used only when scipy is missing.
- `_partial_corr_spearman()` -- Spearman partial correlation: rank-transform then run Pearson on ranks.  Returns (r_partial, ci_low, ci_high, p_value). p_value is from a pe
- `_rank_fallback()` -- Average-rank fallback when scipy is unavailable.
- `_incremental_r2()` -- Incremental R^2 from adding column x to a regression of y on the controls.  Returns R^2_full - R^2_reduced, where:     reduced: y ~ 1 + cont
- `_r2_ols()` -- R^2 of an OLS fit y ~ X. Returns 0.0 for the trivial intercept-only model.
- `_bh_adjust()` -- Benjamini-Hochberg FDR adjustment; NaN entries are passed through.
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

### Generators & lazy pipelines
- **What**: `yield` turns a function into a generator that produces values on demand instead of materializing a full list up front.
- **Why used here**: Streaming long result sets (rollouts, log lines, remote listings) one item at a time keeps memory flat regardless of dataset size.
- **When**: When you iterate over something too large to hold in memory at once.
- **Trade-offs**: Generators are single-pass and stateful; you can't rewind one, and exceptions surfaces only when you pull the next value.

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

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.


## Related Code
- sibling `platform_modal/scripts/_reviewer_points_extract.py`
- sibling `platform_modal/scripts/anonymize.sh`
- sibling `platform_modal/scripts/build_submission.py`
- sibling `platform_modal/scripts/build_university_submission.py`
- sibling `platform_modal/scripts/contamination_check.py`
- sibling `platform_modal/scripts/ed25519-sign.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
