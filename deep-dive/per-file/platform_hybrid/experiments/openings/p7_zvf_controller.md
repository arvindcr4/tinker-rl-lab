# Deep Dive: `platform_hybrid/experiments/openings/p7_zvf_controller.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/experiments/openings/p7_zvf_controller.py` (441 lines)

## Overview
`p7_zvf_controller.py` is an experiment script that exercises a specific research configuration end-to-end. It wires a chosen model, dataset, algorithm, and backend into one reproducible run and records the outcome.
It leans on **argparse, config, dataclass, numpy, protocol** to do its work.
*Self-description:* "P7 --- ZVF closed-loop controller (promoted from designed-only to runnable).  Background ---------- In GRPO-style training the advantage of rollout i for a prom"

## Key Components
- `PID` -- class (2 methods: reset, __call__)
- `ZVFController` -- Closed-loop controller that keeps measured ZVF near a target band.  Actuator ``threshold`` (default): a mid-difficulty acceptance half-width (6 methods: __init__, init_estimates, observe, select, oversample_candidates, update)
- `RunTrace` -- class
- `make_population()` -- True per-prompt pass-rates p_i.  A deliberately hard case for ZVF: a U-shaped mixture (lots of near-0 'too hard' and near-1 'too easy' promp
- `analytic_zvf()` -- Expected ZVF for pass-rates p at group size g (Bernoulli rollouts).
- `rollout_group()` -- G Bernoulli rewards for one prompt.
- `group_is_dead()` -- Zero reward variance <=> all rollouts identical.
- `measure_step_zvf()` -- Run G rollouts for each prompt in idx; return (zvf, n_groups, gradient_groups).  If ctrl is given, fold observed rollouts into its pass-rate
- `_rolling_mean()` -- Causal rolling mean (window shrinks at the head).
- `steps_to_converge()` -- First step (1-based) after which the causal rolling-mean ZVF stays within `tol` of target for `hold` consecutive steps.  Smoothing is used b
- `run_baseline()` -- Uncontrolled: every step trains on a random batch of ALL prompts.
- `run_controlled()` -- Closed loop. A short warmup seeds pass-rate estimates for every prompt, then each step: select -> measure ZVF -> PID update.
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### Frozen protocol over flexibility
- **What**: Experiments intentionally give up knob freedom in exchange for equivalence -- comparability beats configurability here.

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


## Related Code
- sibling `platform_hybrid/experiments/openings/campaign.py`
- sibling `platform_hybrid/experiments/openings/curriculum_grpo.py`
- sibling `platform_hybrid/experiments/openings/groupsize_zvf.py`
- sibling `platform_hybrid/experiments/openings/hard_curriculum.py`
- sibling `platform_hybrid/experiments/openings/p1_emergence.py`
- sibling `platform_hybrid/experiments/openings/p1_freeze_flop.py`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
