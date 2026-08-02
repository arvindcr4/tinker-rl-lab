# Deep Dive: `platform_hybrid/experiments/webarena/react_eval.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/webarena/react_eval.py` (525 lines)

## Overview
`react_eval.py` is an evaluation/measurement script that quantifies outcomes and produces evidence. It turns raw run outputs into comparable metrics and receipts rather than anecdotes.
It leans on **argparse, config, dataclass, logging, protocol, transformers, wandb** to do its work.
*Self-description:* "True multi-turn ReAct eval for BrowserGym (MiniWoB / WebArena-verified).  Per episode:   1. env.reset -> obs   2. loop:        feed (goal + axtree_txt + last_ac"

## Key Components
- `StepRecord` -- class
- `EpisodeResult` -- class
- `TinkerChatSampler` -- Minimal wrapper: render chat messages -> tokens -> sc.sample -> decode. (2 methods: __init__, sample)
- `_import_benchmark()` -- Lazy-register the gym environment ids for a benchmark.
- `_make_env()` -- function
- `_axtree_to_str()` -- Flatten axtree to text with a hard char cap.
- `_goal_to_str()` -- function
- `_parse_response()` -- Extract (thought, action_str). Returns (thought, None) if no action found.
- `run_episode()` -- function
- `_parse_tasks()` -- Accept either a comma list or 'all' for benchmark's registered tasks.
- `_shard()` -- 'k/N' -> take every N-th task starting at index k.
- `_main_sync()` -- function
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Comparability over raw numbers
- **What**: Results only matter relative to a shared frozen protocol; evaluation exists to keep every framework measured against the same yardstick.

### Data modeling with dataclasses
- **What**: `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, turning plain classes into compact value objects.
- **Why used here**: The repo models specs, results, and plans as frozen dataclasses so structural equality and hashing come for free and mutation is blocked.
- **When**: For passive data carriers -- configs, results, plans -- especially when you want `==`/hash semantics.
- **Trade-offs**: No validation by itself; frozen fields protect from mutation but not bad values (pair with pydantic for that).

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

### Structured diagnostics with logging
- **What**: The `logging` module writes level-filtered messages to stderr/files, separating operational noise from real errors and leaving them toggleable at runtime.
- **Why used here**: Runs are audited, so leaving a trail of INFO/DEBUG statements lets a reviewer reconstruct what happened without rerunning GPUs.
- **When**: Anywhere you'd `print` something that matters: progress, warnings, step boundaries, and fatal errors.
- **Trade-offs**: More setup than `print`; misconfigured handler levels silently swallow the very lines you need in production.

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


## Related Code
- sibling `platform_hybrid/experiments/webarena/aggregate.py`
- sibling `platform_hybrid/experiments/webarena/push_model_card.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
