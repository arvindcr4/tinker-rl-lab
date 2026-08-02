# Deep Dive: `platform_hybrid/skyrl/backends/vastai_runner.py`

> AntiVibe &middot; compact mode &middot; 2026-08-02 &middot; source: `platform_hybrid/skyrl/backends/vastai_runner.py` (494 lines)

## Overview
`vastai_runner.py` is an entry point that parses intent from the command line and dispatches to the underlying machinery. It translates `--framework/--backend` flags into a plan, then executes or dry-runs it, acting as the seam between human intent and framework-specific code.
It leans on **argparse, asyncio, config, dataclass, protocol, subprocess** to do its work.
*Self-description:* "vast.ai Runner — framework-aware GPU provisioning.  Provisions GPU instances on vast.ai and runs the chosen RL framework (trl/tinker/verl/openrlhf/skyrl) on-ins"

## Key Components
- `VastInstance` -- class
- `VastAILauncher` -- Launcher for SkyRL tx on vast.ai instances.  This class: 1. Searches for available GPU instances 2. Provisions instances 3. Installs SkyRL a (12 methods: __init__, run_command, search_instances, launch_instance, wait_for_ready, setup_instance, start_skyrl_server, generate_setup_script)
- `main()` -- function
- Has a `if __name__ == "__main__"` entry point

## Concepts & Decisions
### One entry, many substrates
- **What**: Every backend eventually re-enters the local dispatch, so the CLI is both the human interface and the remote-on-box interface.

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

### Structural subtyping with typing.Protocol
- **What**: `Protocol` describes an interface by the *attributes* something has, not by inheritance -- anything matching the shape satisfies it (duck typing with static checks).
- **Why used here**: Lets the code accept `plan`-like and `run`-like objects without forcing a class hierarchy, useful in the shim layer.
- **When**: When many small objects share behavior but have no common ancestor.
- **Trade-offs**: Runtime `isinstance` checks need `@runtime_checkable` and are shallow; static checkers are the real beneficiary.

### Process orchestration (subprocess)
- **What**: `subprocess.run`/`Popen` spawns and captures external commands, letting Python drive shell steps, remote CLIs, and other tools as child processes.
- **Why used here**: Remote backends provision a box then shell out to a driver command -- subprocess is the seam between 'plan' and 'actually run elsewhere'.
- **When**: When work is naturally a separate executable: `modal run`, `gcloud`, ssh commands, secondary scripts.
- **Trade-offs**: Argument quoting/escaping and env leakage are footguns; you lose in-process debugging across the boundary.


## Related Code
- sibling `platform_hybrid/skyrl/backends/__init__.py`
- sibling `platform_hybrid/skyrl/backends/vastai_launch.sh`

---
*Generated by AntiVibe per-file pass &middot; 2026-08-02 &middot; run `/antivibe` (or the antivibe skill) on this file for a full-mode drill-down.*
