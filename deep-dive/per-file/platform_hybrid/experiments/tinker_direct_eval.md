# Deep Dive: `platform_hybrid/experiments/tinker_direct_eval.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/tinker_direct_eval.py` (209 lines)

## Overview
`tinker_direct_eval.py` is an evaluation/measurement script that quantifies outcomes and produces evidence. It turns raw run outputs into comparable metrics and receipts rather than anecdotes.
It leans on **argparse, asyncio, config, numpy, protocol, regex, transformers** to do its work.
*Self-description:* "platform_hybrid/experiments/tinker_direct_eval.py  Direct Tinker API evaluation on GSM8K (no Atropos needed). Uses Tinker sampling client for inference, local s"

## Key Components
- `extract_boxed_answer()` -- Extract answer from \boxed{{}}.
- `score_answer()` -- Binary reward: 1 if match, 0 otherwise.
- `evaluate_with_tinker()` -- Evaluate model on GSM8K using Tinker API.
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Comparability over raw numbers
- **What**: Results only matter relative to a shared frozen protocol; evaluation exists to keep every framework measured against the same yardstick.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.

### Asynchronous I/O (asyncio / async def)
- **What**: `async`/`await` lets one thread interleave many I/O-bound operations (network, subprocesses) instead of blocking on each.
- **Why used here**: Drivers that fan out to remote boxes or API calls benefit from overlapping wait-time; async is the idiomatic way to keep those concurrent.
- **When**: I/O-bound fan-out where you'd otherwise stall on many sequential round-trips; CPU-bound compute still needs threads/processes.
- **Trade-offs**: Async code is more invasive (an async function must await other async functions) and easier to deadlock if a blocking call sneaks in.

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

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### Hugging Face Transformers (pretrained models & tokenizers)
- **What**: The `transformers` library loads pretrained checkpoints (here Qwen3-8B) and their tokenizers behind a uniform `AutoModelForCausalLM`/`AutoTokenizer` interface.
- **Why used here**: It gives one stable API over many architectures plus hosted checkpoints, which is why it is the shared backbone across every framework in this repo.
- **When**: Any task that starts from an existing LLM and adds training, serving, or eval.
- **Trade-offs**: The abstraction hides internals; subtle differences between architectures can surprise you when you rely on undocumented behavior.


## Related Code
- sibling `platform_hybrid/experiments/aggregate_results.py`
- sibling `platform_hybrid/experiments/analyze_lora_sparsity.py`
- sibling `platform_hybrid/experiments/archive_local_artifacts.py`
- sibling `platform_hybrid/experiments/base_instruct_paired.py`
- sibling `platform_hybrid/experiments/bfclv4_tool_use.py`
- sibling `platform_hybrid/experiments/browser_control_smoke.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
