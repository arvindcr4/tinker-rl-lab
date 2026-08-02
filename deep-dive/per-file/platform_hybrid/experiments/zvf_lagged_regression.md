# Deep Dive: `platform_hybrid/experiments/zvf_lagged_regression.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/zvf_lagged_regression.py` (619 lines)

## Overview
`zvf_lagged_regression.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **config, dataclass, numpy, protocol** to do its work.
*Self-description:* "Lagged ZVF regression: test whether ZVF predicts FUTURE reward controlling for CURRENT reward.  Addresses Reviewer Objection O8: "ZVF is tautologically correlat"

## Key Components
- `RunData` -- class
- `as_float()` -- function
- `walk_records()` -- function
- `extract_step_data()` -- Extract (step, reward, zvf, gu) tuples from a step_log.
- `load_runs()` -- function
- `ols_regression()` -- OLS with standard errors, t-statistics, p-values, and R².
- `cross_run_lagged_regression()` -- Cross-run analysis: for each (run, step t), predict R_{t+k} from R_t and ZVF_t.  This pools all (t, t+k) pairs across all runs, including th
- `within_run_lagged_regression()` -- Within-run analysis: for each run, fit R_{t+k} ~ R_t + ZVF_t, then pool coefficients. This controls for run-level confounders (model, task, 
- `partial_correlation()` -- Partial correlation between x and y controlling for z.
- `zvf_partial_correlation_analysis()` -- Compute partial correlation between ZVF_t and R_{t+k}, controlling for R_t.
- `fmt()` -- function
- `main()` -- function
- `write_markdown_report()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Data modeling with dataclasses
- **What**: `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, turning plain classes into compact value objects.
- **Why used here**: The repo models specs, results, and plans as frozen dataclasses so structural equality and hashing come for free and mutation is blocked.
- **When**: For passive data carriers -- configs, results, plans -- especially when you want `==`/hash semantics.
- **Trade-offs**: No validation by itself; frozen fields protect from mutation but not bad values (pair with pydantic for that).

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


## Related Code
- sibling `platform_hybrid/experiments/aggregate_results.py`
- sibling `platform_hybrid/experiments/analyze_lora_sparsity.py`
- sibling `platform_hybrid/experiments/archive_local_artifacts.py`
- sibling `platform_hybrid/experiments/base_instruct_paired.py`
- sibling `platform_hybrid/experiments/bfclv4_tool_use.py`
- sibling `platform_hybrid/experiments/browser_control_smoke.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
