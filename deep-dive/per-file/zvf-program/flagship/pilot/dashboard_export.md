# Deep Dive: `zvf-program/flagship/pilot/dashboard_export.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 12:34 UTC &middot; source: `zvf-program/flagship/pilot/dashboard_export.py` (1229 lines)

## Overview
`dashboard_export.py` is a library module exposing reusable building blocks to the rest of the codebase. It defines types, helpers, and algorithms consumed by drivers and experiments rather than performing a single top-level action.
It leans on **config, dataclass, numpy, protocol, torch** to do its work.

## Key Components
- `DashboardExportError` (bases: RuntimeError) -- Raised when dashboard export receives invalid data or fails contract checks.
- `SpectralTrajectoryData` -- class (1 methods: to_dict)
- `EntropicGatingData` -- class (1 methods: to_dict)
- `GradientRecoveryData` -- class (1 methods: to_dict)
- `_require()` -- function
- `_to_numpy()` -- function
- `prepare_spectral_trajectory_data()` -- Prepare spectral trajectory distance matrix and mode energy payload.  spectral_coeffs: Tensor or array of shape [B, N, D] or [N, D]
- `prepare_entropic_gating_data()` -- Prepare pre and post Givens entropic gating density heatmap payload.
- `prepare_gradient_recovery_data()` -- Prepare step-wise gradient recovery curves across different GRPO conditions.
- `_render_page_wrapper()` -- Generate standalone HTML document wrapper with embedded CSS and JS.
- `export_spectral_trajectory_html()` -- Generate interactive HTML visualizer for spectral trajectory distances.
- `export_gating_density_heatmap_html()` -- Generate interactive HTML visualizer for Givens entropic gating density heatmaps.
- `export_gradient_recovery_html()` -- Generate interactive HTML visualizer for gradient norm recovery curves.
- `export_comparative_dashboard_html()` -- Generate single standalone HTML dashboard containing all three interactive visualizers.

## Concepts & Decisions
### DRY across drivers
- **What**: Shared helper modules stop five framework drivers from each re-solving the same problem in five slightly different ways.

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


## Related Code
- sibling `zvf-program/flagship/pilot/__init__.py`
- sibling `zvf-program/flagship/pilot/analysis.py`
- sibling `zvf-program/flagship/pilot/artifacts.py`
- sibling `zvf-program/flagship/pilot/benchmark_spectral_harness.py`
- sibling `zvf-program/flagship/pilot/bootstrap.py`
- sibling `zvf-program/flagship/pilot/checkpointing.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 12:34 UTC &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
