# Deep Dive: `platform_hybrid/skyrl/run_skyrl_tinker.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/skyrl/run_skyrl_tinker.py` (785 lines)

## Overview
`run_skyrl_tinker.py` is an entry point that parses intent from the command line and dispatches to the underlying machinery. It translates `--framework/--backend` flags into a plan, then executes or dry-runs it, acting as the seam between human intent and framework-specific code.
It leans on **argparse, config, regex, torch, transformers, wandb** to do its work.
*Self-description:* "SkyRL + Hosted Tinker Bridge  Runs SkyRL-style GRPO training against the hosted Tinker API. Combines SkyRL's reward/environment framework with Tinker's GPU trai"

## Key Components
- `GSM8KEnv` -- GSM8K math word problems. (5 methods: __init__, load, make_prompt, reward, sample_batch)
- `MathEnv` -- MATH competition problems. (5 methods: __init__, load, make_prompt, reward, sample_batch)
- `ToolUseEnv` -- Tool-calling (same as grpo_tooluse_tinker.py). (5 methods: __init__, load, make_prompt, reward, sample_batch)
- `load_config()` -- function
- `load_environment()` -- Load a SkyRL gym environment or built-in task.
- `init_wandb()` -- Initialize W&B run from config.
- `log_step_wandb()` -- Log a training step to W&B.
- `log_checkpoint_wandb()` -- Log checkpoint event to W&B.
- `upload_to_hub()` -- Download LoRA weights from Tinker and push to HuggingFace Hub.
- `train()` -- function
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### One entry, many substrates
- **What**: Every backend eventually re-enters the local dispatch, so the CLI is both the human interface and the remote-on-box interface.

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

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.


## Related Code
- Stand-alone: imports nothing else local.

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
