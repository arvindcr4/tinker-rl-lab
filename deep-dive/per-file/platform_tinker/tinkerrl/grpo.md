# Deep Dive: `platform_tinker/tinkerrl/grpo.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_tinker/tinkerrl/grpo.py` (832 lines)

## Overview
`grpo.py` is a training path that runs gradient-based optimization (GRPO/PPO-style) over model weights. Rollouts are generated, scored by a reward model, and their feedback is backpropagated through a policy updated toward higher reward.
It leans on **config, dataclass, protocol, regex, torch, transformers, wandb** to do its work.
*Self-description:* "Deepened GRPO run module.  Consolidates the copy-pasted GRPO training loop that lived across ``grpo_exp_*.py``, ``grpo_100_*.py``, ``grpo_gsm8k_base.py``, and `"

## Key Components
- `TrainingExample` -- One training example: a prompt plus a task-specific target.
- `InMemoryDataset` -- A minimal dataset adapter backed by two lists of examples. (2 methods: train_examples, test_examples)
- `GRPOConfig` -- All knobs for a GRPO experiment in one value object. (1 methods: effective_save_every)
- `GRPORunResult` -- Outcome of one seed inside :func:`run_grpo`.
- `DatasetAdapter` (bases: Protocol) -- Supplies training and held-out examples. (2 methods: train_examples, test_examples)
- `RewardAdapter` (bases: Protocol) -- Scores one completion against the example's target. (1 methods: score)
- `ToolCallReward` -- Scores tool-call completions the way the original ``grpo_exp_*.py`` scripts did. (1 methods: score)
- `MathReward` -- Scores math completions: boxed answer > last number > partial credit. (1 methods: score)
- `ExactMathReward` -- Binary boxed-or-final-number reward used by held-out math benchmarks. (1 methods: score)
- `normalize_rewards()` -- Group-relative advantage normalization (mean 0, std 1).
- `make_grpo_loss_fn()` -- Return a Tinker-compatible loss closure bound to ``advantages``.
- `_decode_response()` -- function
- `_build_datum()` -- Build a ``T.Datum`` from prompt + response token ids.
- `_metric()` -- function
- `_checkpoint_path()` -- function
- `_config_fingerprint()` -- function
- `_write_checkpoint()` -- function
- `_load_checkpoint()` -- function
- `_start_wandb()` -- function
- `_publish_checkpoint()` -- function
- `_run_one_seed()` -- Execute the GRPO loop for one seed.  Pure enough to unit-test with fakes.

## Concepts & Decisions
### Why the loop is GRPO and not full RLHF
- **What**: GRPO is the reward-model-free-online variant this protocol froze: it relies on group-relative advantage, cutting the value critic and memory.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Data modeling with dataclasses
- **What**: `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, turning plain classes into compact value objects.
- **Why used here**: The repo models specs, results, and plans as frozen dataclasses so structural equality and hashing come for free and mutation is blocked.
- **When**: For passive data carriers -- configs, results, plans -- especially when you want `==`/hash semantics.
- **Trade-offs**: No validation by itself; frozen fields protect from mutation but not bad values (pair with pydantic for that).

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


## Related Code
- sibling `platform_tinker/tinkerrl/__init__.py`
- sibling `platform_tinker/tinkerrl/grpo_cli.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
