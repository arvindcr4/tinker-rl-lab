# Deep Dive: `zvf-program/audit/run_colab_e1_confirmatory.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `zvf-program/audit/run_colab_e1_confirmatory.py` (794 lines)

## Overview
`run_colab_e1_confirmatory.py` is an entry point that parses intent from the command line and dispatches to the underlying machinery. It translates `--framework/--backend` flags into a plan, then executes or dry-runs it, acting as the seam between human intent and framework-specific code.
It leans on **argparse, config, parallel, protocol, regex, subprocess** to do its work.
*Self-description:* "Launch one tracked, resumable E1 GRPO unit through the Colab CLI."

## Key Components
- `utc_now()` -- function
- `atomic_json()` -- function
- `sha256_file()` -- function
- `snapshot_sources()` -- Store one content-addressed copy of every executable provenance input.
- `fingerprint()` -- function
- `parse_args()` -- function
- `load_credentials()` -- function
- `read_json()` -- function
- `result_from_log()` -- function
- `local_command_timeout_seconds()` -- Mirror a Colab CLI --timeout locally so a lost socket cannot hang forever.
- `run_logged()` -- function
- `is_transient_remote_failure()` -- Recognize retryable provider/network failures from one command attempt.
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### One entry, many substrates
- **What**: Every backend eventually re-enters the local dispatch, so the CLI is both the human interface and the remote-on-box interface.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### Parallelism (threads / processes / futures)
- **What**: `concurrent.futures`/threading/multiprocessing run independent work concurrently, cutting wall-clock for fan-out tasks.
- **Why used here**: Rollout generation and multi-backend dispatch are embarrassingly parallel, so pools/futures are a cheap win.
- **When**: Many independent units of work that share no mutable state.
- **Trade-offs**: Threads don't speed up CPU-bound Python (GIL); processes add copy/serialization cost and need picklable arguments.

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
- sibling `zvf-program/audit/aggregate_audit.py`
- sibling `zvf-program/audit/run_colab_e1_campaign.py`
- sibling `zvf-program/audit/run_colab_e1_evaluation.py`
- sibling `zvf-program/audit/run_colab_e1_resume.py`
- sibling `zvf-program/audit/run_colab_obligations.py`
- sibling `zvf-program/audit/test_aggregate_audit.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
