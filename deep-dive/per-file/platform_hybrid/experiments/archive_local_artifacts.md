# Deep Dive: `platform_hybrid/experiments/archive_local_artifacts.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/archive_local_artifacts.py` (778 lines)

## Overview
`archive_local_artifacts.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, config, csv, protocol, regex, subprocess, wandb** to do its work.
*Self-description:* "Archive W&B logs, Hugging Face cache snapshots, and Tinker checkpoints locally.  This script intentionally reads service tokens only from environment variables "

## Key Components
- `log()` -- function
- `write_json()` -- function
- `append_jsonl()` -- function
- `run_cmd()` -- function
- `disk_free_bytes()` -- function
- `dir_size_bytes()` -- function
- `safe_name()` -- function
- `hardlink_or_copy()` -- function
- `ignore_sensitive_cache_files()` -- function
- `copy_tree_hardlinked()` -- function
- `archive_local_wandb()` -- function
- `archive_wandb_cloud()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.

### CSV I/O
- **What**: `csv` reads/writes comma-separated records, the lingua franca for tabular data and result dumps.
- **Why used here**: Large benchmark/results files are exchanged as CSV, so importing/exporting that format is a direct requirement.
- **When**: When tabular data must be human-openable or compatible with spreadsheets/other tools.
- **Trade-offs**: CSV has no schema or types -- every field is a string, so parsing and quoting edge cases are on you.

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

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

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.


## Related Code
- sibling `platform_hybrid/experiments/aggregate_results.py`
- sibling `platform_hybrid/experiments/analyze_lora_sparsity.py`
- sibling `platform_hybrid/experiments/base_instruct_paired.py`
- sibling `platform_hybrid/experiments/bfclv4_tool_use.py`
- sibling `platform_hybrid/experiments/browser_control_smoke.py`
- sibling `platform_hybrid/experiments/build_nia_index_payload.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
