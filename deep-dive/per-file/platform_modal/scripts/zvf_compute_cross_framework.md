# Deep Dive: `platform_modal/scripts/zvf_compute_cross_framework.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_modal/scripts/zvf_compute_cross_framework.py` (631 lines)

## Overview
`zvf_compute_cross_framework.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, numpy, protocol** to do its work.
*Self-description:* "Cross-framework ZVF (Zero-Variance Fraction) computation pipeline.  Implements the reference pseudocode from ``paper/sections/zvf_pipeline_spec.tex`` and the pe"

## Key Components
- `zvf()` -- Canonical pseudocode from zvf_pipeline_spec.tex.  ``rewards_2d`` is an iterable of iterables, shape ``[num_groups, K]``. Returns the fractio
- `batch_reward_mean()` -- function
- `_iter_json_records()` -- Yield dict records from a JSON or JSONL file.  Handles three shapes gracefully:   * JSONL (one json object per line)   * JSON with a top-lev
- `_schema_diff()` -- function
- `_flatten_keys()` -- function
- `_get_nested()` -- function
- `_as_float_list()` -- function
- `parse_trl()` -- TRL GRPOTrainer log row.  Expects ``rewards`` (flat list), ``batch_size`` and ``group_size`` OR a ``group_size``/``num_generations`` hint.
- `parse_tinker()` -- TINKER managed-runtime log row.  Expects ``rollouts`` or ``rollout`` list; each entry carries ``reward`` and ``group_id``.
- `parse_openrlhf()` -- OpenRLHF log row.  Expects per-sample ``reward_score`` and ``prompt_index`` (or ``prompt_id``) either as parallel lists or an array of dicts
- `parse_verl()` -- veRL log row.  Expects per-sample entries carrying ``uid`` (group boundary) and a ``data_source/reward`` or ``reward`` field. Accepts either
- `compute_time_series()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

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
- sibling `platform_modal/scripts/_reviewer_points_extract.py`
- sibling `platform_modal/scripts/anonymize.sh`
- sibling `platform_modal/scripts/build_submission.py`
- sibling `platform_modal/scripts/build_university_submission.py`
- sibling `platform_modal/scripts/contamination_check.py`
- sibling `platform_modal/scripts/ed25519-sign.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
