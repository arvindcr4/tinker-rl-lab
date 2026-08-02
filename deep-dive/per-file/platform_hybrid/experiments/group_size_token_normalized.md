# Deep Dive: `platform_hybrid/experiments/group_size_token_normalized.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/group_size_token_normalized.py` (460 lines)

## Overview
`group_size_token_normalized.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, config, csv, protocol** to do its work.
*Self-description:* "Token-budget-normalized $G$-sweep for the GRPO group-size reconciliation appendix.  Implements the estimator defined in ``paper/sections/group_size_reconcile.te"

## Key Components
- `parse_budget()` -- Accept ``1M``, ``4m``, ``16000000``, ``1e6`` style budgets.
- `parse_budgets()` -- function
- `_iter_json_records()` -- function
- `_collect_g_sweep()` -- Keyed by G, value is list of experiment dicts.
- `_row_heldout_acc()` -- function
- `_row_token_budget()` -- Extract an explicit token budget from a run record, if present.
- `_bootstrap_ci()` -- Return (mean, lo, hi); tiny samples get a simple t-interval fallback.
- `gu_estimate()` -- Sample-variance-proxy estimator of Eq.~(eq:gu).  GU_hat(G, B) proportional to (1 - ZVF) * Var_A / (G * K * L_bar). Multiplied by GU_SCALE so
- `_canonical_budget_filter()` -- Round budget to the nearest canonical {1M, 4M, 16M, 64M} so the fallback table key-matches even if the CLI passes ``1000000``.
- `build_rows()` -- function
- `write_tsv()` -- function
- `format_budget_label()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Generators & lazy pipelines
- **What**: `yield` turns a function into a generator that produces values on demand instead of materializing a full list up front.
- **Why used here**: Streaming long result sets (rollouts, log lines, remote listings) one item at a time keeps memory flat regardless of dataset size.
- **When**: When you iterate over something too large to hold in memory at once.
- **Trade-offs**: Generators are single-pass and stateful; you can't rewind one, and exceptions surfaces only when you pull the next value.

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

### CSV I/O
- **What**: `csv` reads/writes comma-separated records, the lingua franca for tabular data and result dumps.
- **Why used here**: Large benchmark/results files are exchanged as CSV, so importing/exporting that format is a direct requirement.
- **When**: When tabular data must be human-openable or compatible with spreadsheets/other tools.
- **Trade-offs**: CSV has no schema or types -- every field is a string, so parsing and quoting edge cases are on you.

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
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
