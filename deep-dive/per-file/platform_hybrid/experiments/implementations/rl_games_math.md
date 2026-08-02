# Deep Dive: `platform_hybrid/experiments/implementations/rl_games_math.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/implementations/rl_games_math.py` (225 lines)

## Overview
`rl_games_math.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, config, dataclass, logging, numpy, protocol, wandb** to do its work.
*Self-description:* "rl_games (NVIDIA) PPO Math RL Implementation ============================================= Port of Tinker Math RL to NVIDIA's rl_games library.  rl_games is des"

## Key Components
- `AlgoConfig` -- class
- `ModelConfig` -- class
- `MLPConfig` -- class
- `SpaceConfig` -- class
- `NetworkConfig` -- class
- `PPOConfig` -- class
- `RLGamesParams` -- class
- `RLGamesConfig` -- class
- `ArithmeticEnv` (bases: gym.Env) -- Arithmetic environment for rl_games. (3 methods: __init__, reset, step)
- `parse_args()` -- function
- `main()` -- Main function for rl_games training.  Note: Full rl_games integration requires rl_games package. This shows the configuration pattern.
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

### Data modeling with dataclasses
- **What**: `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations, turning plain classes into compact value objects.
- **Why used here**: The repo models specs, results, and plans as frozen dataclasses so structural equality and hashing come for free and mutation is blocked.
- **When**: For passive data carriers -- configs, results, plans -- especially when you want `==`/hash semantics.
- **Trade-offs**: No validation by itself; frozen fields protect from mutation but not bad values (pair with pydantic for that).

### Experiment tracking with Weights & Biases
- **What**: W&B records metrics, hyperparameters, and artifacts to a hosted or local run timeline, giving every training run a shareable dashboard and history.
- **Why used here**: The repo treats receipts/evidence as first-class outputs, and W&B is one of the three independent channels (HF + W&B + GCS) whose agreement is the trust signal.
- **When**: When a run's value is in its history -- comparing sweeps, auditing, or sharing results without sending weights.
- **Trade-offs**: Adds a network dependency and an external account; local-only runs must opt out or write a local fallback.

### Configuration as declarative data (YAML/JSON/TOML)
- **What**: Knobs live in YAML/JSON/TOML files or tables rather than code, so a run's intent is inspectable and diffable without reading the program.
- **Why used here**: A single frozen `CanonicalSpec` + preregistration files is the repo's whole comparability contract -- config-as-data is what makes runs hashable and testable.
- **When**: Anywhere parameters should be changeable without editing code, or compared across runs.
- **Trade-offs**: Config can drift from what the code actually reads; validation (pydantic) is what catches a key that no longer means what it says.

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Numeric arrays with NumPy
- **What**: NumPy gives dense N-d arrays and vectorized math (reductions, broadcasting) that run at C speed.
- **Why used here**: Reward computation and metrics are array operations; vectorizing over a batch is both faster and more readable than Python loops.
- **When**: Any batched numeric transform -- rewards, accuracy, aggregations across rollouts.
- **Trade-offs**: NumPy and torch each own their memory; converting between them copies unless you share storage carefully.

### Structured diagnostics with logging
- **What**: The `logging` module writes level-filtered messages to stderr/files, separating operational noise from real errors and leaving them toggleable at runtime.
- **Why used here**: Runs are audited, so leaving a trail of INFO/DEBUG statements lets a reviewer reconstruct what happened without rerunning GPUs.
- **When**: Anywhere you'd `print` something that matters: progress, warnings, step boundaries, and fatal errors.
- **Trade-offs**: More setup than `print`; misconfigured handler levels silently swallow the very lines you need in production.

### Command-line argument parsing
- **What**: `argparse` turns `sys.argv` into typed options (`--framework`, `--dry-run`) with help text and error handling for free.
- **Why used here**: Every platform entry point must be runnable by humans and by shelling-out code, so a stable, documented CLI is the contract between them.
- **When**: When a script is invoked by people, CI, or other processes and needs explicit knobs.
- **Trade-offs**: Boilerplate-heavy and positional-only; richer CLIs use click/typer for nesting and auto-generated help.


## Related Code
- `utils.seed` &rarr; local `utils/seed.py`
- sibling `platform_hybrid/experiments/implementations/cleanrl_ppo_math.py`
- sibling `platform_hybrid/experiments/implementations/d3rlpy_offline.py`
- sibling `platform_hybrid/experiments/implementations/p1_scaled_layer_freeze.py`
- sibling `platform_hybrid/experiments/implementations/p2p3_token_budget_curriculum.py`
- sibling `platform_hybrid/experiments/implementations/p4_length_bias_kl_mask.py`
- sibling `platform_hybrid/experiments/implementations/p7_zvf_pid.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
