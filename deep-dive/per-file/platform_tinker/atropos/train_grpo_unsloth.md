# Deep Dive: `platform_tinker/atropos/train_grpo_unsloth.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_tinker/atropos/train_grpo_unsloth.py` (846 lines)

## Overview
`train_grpo_unsloth.py` is a training path that runs gradient-based optimization (GRPO/PPO-style) over model weights. Rollouts are generated, scored by a reward model, and their feedback is backpropagated through a policy updated toward higher reward.
It leans on **argparse, config, http, logging, numpy, parallel, peft, protocol, regex, torch, transformers, trl, wandb** to do its work.
*Self-description:* "train_grpo_unsloth.py — Drop-in Unsloth replacement for Atropos + Tinker RL.  Reads any existing configs/gsm8k_*.yaml config and runs GRPO training with Unsloth"

## Key Components
- `StatefulRewardFunction` -- A stateful reward function compatible with TRL GRPOTrainer, tracking metrics thread-safely. (3 methods: __init__, __call__, get_metrics_and_reset)
- `StepTracker` -- Accumulates per-completion scores and logs step-level metrics. (4 methods: __init__, record, flush, save_csv)
- `_setup_math_verify()` -- Import math_verify lazily so the module works without it for tests.
- `_extract_gold()` -- GSM8K gold answers are in the form '... #### 42'.
- `_score_response()` -- Return 1.0 if response contains a correct \boxed{} answer, else 0.0. Mirrors GSM8kEnv.score() exactly.
- `_generative_score_response()` -- Area 10: Generative Verifier. Uses the local LLM inference server to generate a reasoning trace and verify correctness.
- `_execution_score_response()` -- Area 5: Execution-Based Rewards. Extracts a Python block, executes it, captures stdout, and compares it to the gold answer.
- `_completion_to_text()` -- Normalize TRL completion payloads across versions.
- `build_prompt()` -- Format a GSM8K question into the chat template expected by the model. Matches the message structure in gsm8k_tinker.py.
- `load_config()` -- function
- `_param_count_B()` -- Heuristic: extract parameter count from model name.
- `load_model_and_tokenizer()` -- Load model with Unsloth. Uses 4-bit quantisation for ≥4B models to fit smaller GPUs; full BF16 for tiny models.
- `prepare_dataset()` -- function
- `_bool_env()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Why the loop is GRPO and not full RLHF
- **What**: GRPO is the reward-model-free-online variant this protocol froze: it relies on group-relative advantage, cutting the value critic and memory.

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

### HTTP client calls
- **What**: `requests`/`httpx`/`aiohttp` issue HTTP requests to APIs -- model hosting, receipt uploads (HF/W&B/GCS), or remote preflight checks.
- **Why used here**: Evidence must land on independent channels, and those channels are network APIs, so HTTP is how receipts and checkpoints actually get out.
- **When**: Any interaction with a REST endpoint: upload, download, health-check, serverless invocation.
- **Trade-offs**: Network calls fail; you need timeouts, retries, and idempotency or a transient blip becomes a lost run.

### Structured diagnostics with logging
- **What**: The `logging` module writes level-filtered messages to stderr/files, separating operational noise from real errors and leaving them toggleable at runtime.
- **Why used here**: Runs are audited, so leaving a trail of INFO/DEBUG statements lets a reviewer reconstruct what happened without rerunning GPUs.
- **When**: Anywhere you'd `print` something that matters: progress, warnings, step boundaries, and fatal errors.
- **Trade-offs**: More setup than `print`; misconfigured handler levels silently swallow the very lines you need in production.

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

### Parallelism (threads / processes / futures)
- **What**: `concurrent.futures`/threading/multiprocessing run independent work concurrently, cutting wall-clock for fan-out tasks.
- **Why used here**: Rollout generation and multi-backend dispatch are embarrassingly parallel, so pools/futures are a cheap win.
- **When**: Many independent units of work that share no mutable state.
- **Trade-offs**: Threads don't speed up CPU-bound Python (GIL); processes add copy/serialization cost and need picklable arguments.

### Parameter-Efficient Fine-Tuning (LoRA & friends)
- **What**: PEFT freezes the base weights and trains small low-rank adapter matrices (LoRA r=16 here) plus a handful of config classes (`LoraConfig`).
- **Why used here**: LoRA makes the canonical 30-step GRPO run affordable and makes checkpoints tiny and shareable -- the repo freezes `peft` into every framework path.
- **When**: When you want to adapt a large model without retraining it or shipping full copies.
- **Trade-offs**: Adapters cap what the model can learn and add a merge step before inference; k-bit quantized base weights complicate offloading.

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

### PyTorch tensor computation & autograd
- **What**: PyTorch is the numeric engine: `torch.Tensor` holds batched GPU/CPU arrays and `torch.autograd` builds the computation graph so gradients flow from a loss back to every parameter.
- **Why used here**: TRL, transformers, vLLM and this repo's RL loops are all built on PyTorch, so using it directly avoids impedance mismatch between framework and training code.
- **When**: Anywhere gradients must reach model weights -- training, RL rollouts, LoRA adaptation, or evaluation under a different dtype.
- **Trade-offs**: Eager execution is easy to debug but slower than compiled graphs; `torch.compile`/export recover speed at the cost of traceability.

### Hugging Face Transformers (pretrained models & tokenizers)
- **What**: The `transformers` library loads pretrained checkpoints (here Qwen3-8B) and their tokenizers behind a uniform `AutoModelForCausalLM`/`AutoTokenizer` interface.
- **Why used here**: It gives one stable API over many architectures plus hosted checkpoints, which is why it is the shared backbone across every framework in this repo.
- **When**: Any task that starts from an existing LLM and adds training, serving, or eval.
- **Trade-offs**: The abstraction hides internals; subtle differences between architectures can surprise you when you rely on undocumented behavior.

### HF TRL training library (PPO/GRPO-style RLHF)
- **What**: `trl` implements RLHF-style trainers (PPO, and the GRPO variant used here) that coordinate policy, reference model, reward computation, and rollout generation.
- **Why used here**: It removes the burden of writing the RL loop from scratch and is one of the five frameworks whose equivalence this repo proves.
- **When**: When you need a maintained, well-tested RLHF loop and accept its configuration model.
- **Trade-offs**: The library's opinionated defaults can fight custom research setups; equivalence testing exists precisely because each framework behaves slightly differently.

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.


## Related Code
- sibling `platform_tinker/atropos/.pre-commit-config.yaml`
- sibling `platform_tinker/atropos/eval_arenahard.py`
- sibling `platform_tinker/atropos/eval_reasoning_suite.py`
- sibling `platform_tinker/atropos/eval_toxicchat.py`
- sibling `platform_tinker/atropos/generate_slides.py`
- sibling `platform_tinker/atropos/launch_training.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
