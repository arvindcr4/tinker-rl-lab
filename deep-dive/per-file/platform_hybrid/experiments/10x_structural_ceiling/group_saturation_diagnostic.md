# Deep Dive: `platform_hybrid/experiments/10x_structural_ceiling/group_saturation_diagnostic.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `platform_hybrid/experiments/10x_structural_ceiling/group_saturation_diagnostic.py` (179 lines)

## Overview
`group_saturation_diagnostic.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **config, dataclass, protocol, torch, wandb** to do its work.
*Self-description:* "Group Saturation Diagnostic — tracks zero-variance fraction per GRPO step.  Key insight: When all G completions in a group receive the same reward, the advantag"

## Key Components
- `GroupStats` -- class (5 methods: mean, std, is_saturated, spread, erf)
- `StepDiagnostic` -- class (9 methods: mean_reward, erf, n_groups, n_saturated, zero_variance_frac, mean_group_std, effective_groups, gradient_utilization)
- `SaturationTracker` -- Accumulates per-step group saturation metrics across a training run. (4 methods: __init__, record_step, summary, save)
- `log_to_wandb()` -- Log saturation metrics to W&B if available.

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

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


## Related Code
- sibling `platform_hybrid/experiments/10x_structural_ceiling/analyze_results.py`
- sibling `platform_hybrid/experiments/10x_structural_ceiling/grpo_10x_runner.py`
- sibling `platform_hybrid/experiments/10x_structural_ceiling/gsm8k_dpo.py`
- sibling `platform_hybrid/experiments/10x_structural_ceiling/prebin_gsm8k.py`
- sibling `platform_hybrid/experiments/10x_structural_ceiling/round2_runner.py`
- sibling `platform_hybrid/experiments/10x_structural_ceiling/run_all.sh`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
