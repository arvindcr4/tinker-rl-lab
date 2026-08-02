# Deep Dive: `platform_hybrid/experiments/zvf_predictive_validation.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/zvf_predictive_validation.py` (676 lines)

## Overview
`zvf_predictive_validation.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **config, csv, dataclass, numpy, protocol, viz** to do its work.
*Self-description:* "Retrospective early-prefix validation for ZVF/GU diagnostics.  The thesis argument being tested is narrow: if the first few logged optimization steps already sh"

## Key Components
- `RunRecord` -- class
- `walk_records()` -- function
- `as_float()` -- function
- `as_str()` -- function
- `metric_arrays()` -- function
- `record_quality_score()` -- function
- `make_record()` -- function
- `load_runs()` -- function
- `rank_average()` -- function
- `pearson()` -- function
- `spearman()` -- function
- `auc_binary()` -- function
- `bootstrap_ci()` -- function
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

### CSV I/O
- **What**: `csv` reads/writes comma-separated records, the lingua franca for tabular data and result dumps.
- **Why used here**: Large benchmark/results files are exchanged as CSV, so importing/exporting that format is a direct requirement.
- **When**: When tabular data must be human-openable or compatible with spreadsheets/other tools.
- **Trade-offs**: CSV has no schema or types -- every field is a string, so parsing and quoting edge cases are on you.

### Data visualization
- **What**: Matplotlib/Plotly render metrics into figures, replacing dense number tables with readable curves.
- **Why used here**: The repo produces decks and figures as code so charts derive from evidence and regenerate whenever the checkout changes.
- **When**: When a comparison (scaling curve, loss trace, ablation) is clearer as a picture than a table.
- **Trade-offs**: Figures need explicit styling to stay trustworthy; a miscalled axis or log scale can misrepresent the claim.

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.


## Related Code
- sibling `platform_hybrid/experiments/aggregate_results.py`
- sibling `platform_hybrid/experiments/analyze_lora_sparsity.py`
- sibling `platform_hybrid/experiments/archive_local_artifacts.py`
- sibling `platform_hybrid/experiments/base_instruct_paired.py`
- sibling `platform_hybrid/experiments/bfclv4_tool_use.py`
- sibling `platform_hybrid/experiments/browser_control_smoke.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
