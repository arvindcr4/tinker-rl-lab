# Deep Dive: `platform_modal/scripts/zvf_diagnostic.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_modal/scripts/zvf_diagnostic.py` (1143 lines)

## Overview
`zvf_diagnostic.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, csv, protocol, viz** to do its work.
*Self-description:* "Pillar 2 cross-experiment Zero-Variance Fraction (ZVF) diagnostic.  Aggregates ZVF measurement streams from every per-experiment training run that surfaced a pe"

## Key Components
- `_stat()` -- mean, min, max with NaN-safe handling.
- `load_tinker_gsm8k()` -- Real Qwen3-8B / GSM8K rollouts, K=8 per problem, 3 seeds.
- `load_groupsize_sweep()` -- Aggregate G-sweep rows (G in {2,4,8,16}, 3 seeds each).
- `load_variance_mitigation()` -- Per-method collapse-labeled trajectories.  The TSV carries per-step zvf and heldout-acc for several methods; we aggregate to one row per (me
- `load_tool_use_diagnostics()` -- Cross-tool tool-use runs that fully collapsed (last10 == 0).  Each row has a `zvf` of 1.0 across the entire trajectory and a last10_avg of 0
- `load_scaling_law_phases()` -- Three-phase scaling-law table -- one row per model.  The scaling-law TSV does NOT carry explicit per-step ZVF; we keep these rows in the sum
- `load_drgrpo_vs_grpo()` -- Per-run mean_zvf from the Qwen2.5-0.5B DRGRPO vs GRPO study.
- `load_samestack_ppo_grpo()` -- Per-run metrics from the PPO/GRPO shared-stack diagnostic.
- `classify()` -- function
- `_pearson()` -- function
- `_spearman()` -- function
- `bootstrap_ci()` -- Percentile bootstrap for correlation.  Resamples (xs[i], ys[i]) rows to respect paired structure.
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
