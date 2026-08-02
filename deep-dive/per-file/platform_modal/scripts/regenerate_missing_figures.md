# Deep Dive: `platform_modal/scripts/regenerate_missing_figures.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_modal/scripts/regenerate_missing_figures.py` (373 lines)

## Overview
`regenerate_missing_figures.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, numpy, protocol, subprocess, viz** to do its work.
*Self-description:* "Regenerate the three figure PDF+PNG pairs that were missing from the repo.  Writes, in the exact paths main.tex expects:   - paper/figures/v2/performance_profil"

## Key Components
- `_save()` -- function
- `_load_jsonl()` -- function
- `_gaussian_smooth()` -- Pure-numpy Gaussian smoothing (no scipy.ndimage dependency).
- `fig_performance_profiles()` -- Agarwal et al. 2021-style performance profiles over 5 RL libraries.
- `_wave6_sweep()` -- function
- `fig_wave6_sensitivity()` -- 3-panel Qwen3-8B GSM8K sensitivity (temperature / rank / batch).
- `fig_old_trl_seeds()` -- 5-seed TRL GRPO reproducibility violin (Qwen2.5-0.5B / GSM8K / 125 steps / L4).
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

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

### Process orchestration (subprocess)
- **What**: `subprocess.run`/`Popen` spawns and captures external commands, letting Python drive shell steps, remote CLIs, and other tools as child processes.
- **Why used here**: Remote backends provision a box then shell out to a driver command -- subprocess is the seam between 'plan' and 'actually run elsewhere'.
- **When**: When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, secondary scripts.
- **Trade-offs**: Argument quoting/escaping and env leakage are footguns; you lose in-process debugging across the boundary.

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
