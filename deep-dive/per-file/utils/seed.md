# Deep Dive: `utils/seed.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `utils/seed.py` (210 lines)

## Overview
`seed.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **argparse, config, numpy, torch, transformers, trl** to do its work.
*Self-description:* "Seed Management Utility for Reproducible RL Experiments ======================================================== Ensures deterministic behavior across Python, N"

## Key Components
- `set_global_seed()` -- Set random seed for reproducibility across all frameworks.  Args:     seed: Integer seed value.     deterministic_cudnn: If True, set CuDNN 
- `get_seed_from_args()` -- Parse --seed from command line arguments.
- `get_environment_info()` -- Collect environment information for reproducibility logging. Logs GPU type, CUDA version, library versions, etc.
- `log_experiment_metadata()` -- Log full experiment metadata to a JSON file for reproducibility.  Args:     experiment_name: Name of the experiment     seed: Random seed us

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

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

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

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


## Related Code
- sibling `utils/__init__.py`
- sibling `utils/audit_utils.py`
- sibling `utils/stats.py`
- sibling `utils/tinker_grpo.py`
- sibling `utils/verify_results.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
