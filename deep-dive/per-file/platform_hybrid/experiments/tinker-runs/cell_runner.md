# Deep Dive: `platform_hybrid/experiments/tinker-runs/cell_runner.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/tinker-runs/cell_runner.py` (467 lines)

## Overview
`cell_runner.py` is an entry point that parses intent from the command line and dispatches to the underlying machinery. It translates `--framework/--backend` flags into a plan, then executes or dry-runs it, acting as the seam between human intent and framework-specific code.
It leans on **argparse, config, csv, regex, torch** to do its work.
*Self-description:* "cell_runner.py — THIN per-cell shim for the ZVF Pillar-1/M1 sweep.  WHAT THIS IS   The sweep orchestrator (zvf-program/sweep/run_sweep.py) shells out once per  "

## Key Components
- `_gsm8k_step_count()` -- Cheap proxy for problem difficulty: number of reasoning steps in the GSM8K reference solution. GSM8K rationales put one step per line and em
- `build_difficulty_loader()` -- Return a replacement for live_zvf_probe.load_gsm8k_examples that filters the GSM8K training pool to the requested difficulty tercile, then f
- `install_loss_arm()` -- Adjust the existing runner's loss surrogate for the requested arm.  The in-repo loop computes a per-group standardized advantage and a REINF
- `upsert_master_row()` -- Insert/replace the row for this cell in master_results.csv.  Pulls every value from `result` (the real runner's output). The row's experimen
- `parse_args()` -- function
- `main()` -- function
- `_failure_stub()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### One entry, many substrates
- **What**: Every backend eventually re-enters the local dispatch, so the CLI is both the human interface and the remote-on-box interface.

### PyTorch tensor computation & autograd
- **What**: PyTorch is the numeric engine: `torch.Tensor` holds batched GPU/CPU arrays and `torch.autograd` builds the computation graph so gradients flow from a loss back to every parameter.
- **Why used here**: TRL, transformers, vLLM and this repo's RL loops are all built on PyTorch, so using it directly avoids impedance mismatch between framework and training code.
- **When**: Anywhere gradients must reach model weights -- training, RL rollouts, LoRA adaptation, or evaluation under a different dtype.
- **Trade-offs**: Eager execution is easy to debug but slower than compiled graphs; `torch.compile`/export recover speed at the cost of traceability.

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

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.


## Related Code
- sibling `platform_hybrid/experiments/tinker-runs/build_claim_run_table.py`
- sibling `platform_hybrid/experiments/tinker-runs/campaign_v2.py`
- sibling `platform_hybrid/experiments/tinker-runs/h2h_summarize.py`
- sibling `platform_hybrid/experiments/tinker-runs/live_zvf_probe.py`
- sibling `platform_hybrid/experiments/tinker-runs/massive_campaign.py`
- sibling `platform_hybrid/experiments/tinker-runs/modal_inventory.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
