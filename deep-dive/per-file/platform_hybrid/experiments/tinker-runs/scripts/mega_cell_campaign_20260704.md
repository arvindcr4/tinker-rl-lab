# Deep Dive: `platform_hybrid/experiments/tinker-runs/scripts/mega_cell_campaign_20260704.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/tinker-runs/scripts/mega_cell_campaign_20260704.py` (789 lines)

## Overview
`mega_cell_campaign_20260704.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, asyncio, config, csv, dataclass, parallel, protocol, regex, subprocess, transformers, wandb** to do its work.
*Self-description:* "500+ cell sampling-only Tinker measurement campaign.  Cell = (model, task_slice, G, temperature, seed). Each completed cell writes:   - one row to cells.tsv   -"

## Key Components
- `Cell` -- class (2 methods: model_family, id)
- `JsonWandbLogger` -- class (7 methods: __init__, start, log_cell, finish, _stash, _worker, _flush_summary_runs)
- `Campaign` -- class (10 methods: __init__, _load_existing_tokens, run, _run_with_sem, tokenizer_for, tokenizer_for_async, run_cell, _write_cell_outputs)
- `model_family()` -- function
- `ensure_dirs()` -- function
- `gsm_reward()` -- function
- `extract_code()` -- function
- `run_humaneval_test()` -- function
- `load_task_examples()` -- function
- `make_prompt()` -- function
- `score_completion()` -- function
- `metric_summary()` -- function
- `get_tokenizer()` -- function
- `discover_models()` -- function
- `load_done_ids()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Data modeling with dataclasses
- **What**: `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, turning plain classes into compact value objects.
- **Why used here**: The repo models specs, results, and plans as frozen dataclasses so structural equality and hashing come for free and mutation is blocked.
- **When**: For passive data carriers -- configs, results, plans -- especially when you want `==`/hash semantics.
- **Trade-offs**: No validation by itself; frozen fields protect from mutation but not bad values (pair with pydantic for that).

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Hugging Face Transformers (pretrained models & tokenizers)
- **What**: The `transformers` library loads pretrained checkpoints (here Qwen3-8B) and their tokenizers behind a uniform `AutoModelForCausalLM`/`AutoTokenizer` interface.
- **Why used here**: It gives one stable API over many architectures plus hosted checkpoints, which is why it is the shared backbone across every framework in this repo.
- **When**: Any task that starts from an existing LLM and adds training, serving, or eval.
- **Trade-offs**: The abstraction hides internals; subtle differences between architectures can surprise you when you rely on undocumented behavior.

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

### Asynchronous I/O (asyncio / async def)
- **What**: `async`/`await` lets one thread interleave many I/O-bound operations (network, subprocesses) instead of blocking on each.
- **Why used here**: Drivers that fan out to remote boxes or API calls benefit from overlapping wait-time; async is the idiomatic way to keep those concurrent.
- **When**: I/O-bound fan-out where you'd otherwise stall on many sequential round-trips; CPU-bound compute still needs threads/processes.
- **Trade-offs**: Async code is more invasive (an async function must await other async functions) and easier to deadlock if a blocking call sneaks in.

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
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_100_math.py`
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_100_synthetic.py`
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_100_xlam.py`
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_exp_a_baseline.py`
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_exp_d_xlam.py`
- sibling `platform_hybrid/experiments/tinker-runs/scripts/grpo_gsm8k_base.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
