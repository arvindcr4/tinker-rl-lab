# Deep Dive: `zvf-program/experiments-next/eval_passk_standalone.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `zvf-program/experiments-next/eval_passk_standalone.py` (415 lines)

## Overview
`eval_passk_standalone.py` is an evaluation/measurement script that quantifies outcomes and produces evidence. It turns raw run outputs into comparable metrics and receipts rather than anecdotes.
It leans on **argparse, config, parallel, regex, subprocess, vllm** to do its work.
*Self-description:* "eval_passk_standalone.py — portable pass@k evaluator (no Modal deps).  Runs anywhere with a GPU + vllm (Lightning Studio, Colab, bare metal). Datasets: gsm8k (t"

## Key Components
- `chatml()` -- function
- `gsm8k_reward()` -- function
- `norm_math()` -- function
- `last_boxed()` -- function
- `math_reward()` -- function
- `extract_code()` -- function
- `_run_candidate()` -- function
- `pass_at_k()` -- function
- `write_checkpoint()` -- Atomically persist progress so SIGKILL cannot corrupt the checkpoint.
- `validate_resume()` -- Refuse to combine partial results from different evaluation configs.
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

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### Parallelism (threads / processes / futures)
- **What**: `concurrent.futures`/threading/multiprocessing run independent work concurrently, cutting wall-clock for fan-out tasks.
- **Why used here**: Rollout generation and multi-backend dispatch are embarrassingly parallel, so pools/futures are a cheap win.
- **When**: Many independent units of work that share no mutable state.
- **Trade-offs**: Threads don't speed up CPU-bound Python (GIL); processes add copy/serialization cost and need picklable arguments.

### Text processing with regular expressions
- **What**: `re` matches/extracts patterns in text -- parsing logs, sanitizing identifiers, or validating formats that don't warrant a full parser.
- **Why used here**: Receipts, path probes, and name munging are string-shaped; regex is the compact tool for targeted extraction.
- **When**: Small, well-defined text patterns where a parser is overkill.
- **Trade-offs**: Regex is opaque and easy to get subtly wrong; complex grammars should graduate to a real parser.

### Process orchestration (subprocess)
- **What**: `subprocess.run`/`Popen` spawns and captures external commands, letting Python drive shell steps, remote CLIs, and other tools as child processes.
- **Why used here**: Remote backends provision a box then shell out to a driver command -- subprocess is the seam between 'plan' and 'actually run elsewhere'.
- **When**: When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, secondary scripts.
- **Trade-offs**: Argument quoting/escaping and env leakage are footguns; you lose in-process debugging across the boundary.

### vLLM high-throughput inference serving
- **What**: vLLM serves LLMs with paged attention and continuous batching, turning costly rollout generation into a fast, batched operation.
- **Why used here**: RL rollouts need thousands of completions per step; vLLM's batching is what makes that tractable on limited GPU time.
- **When**: Whenever inference volume (not just latency) is the bottleneck -- exactly the RL rollout case.
- **Trade-offs**: Serving adds a process boundary and a different memory footprint; small models can be slower through vLLM than a plain HF forward pass.


## Related Code
- sibling `zvf-program/experiments-next/aggregate_seed_audits.py`
- sibling `zvf-program/experiments-next/analyze_rollout_quality.py`
- sibling `zvf-program/experiments-next/analyze_t1_ci.py`
- sibling `zvf-program/experiments-next/analyze_t1_correlated_fix.py`
- sibling `zvf-program/experiments-next/analyze_t2_floor.py`
- sibling `zvf-program/experiments-next/analyze_t3_gstar.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
