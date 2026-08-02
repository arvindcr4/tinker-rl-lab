# Deep Dive: Unified Launcher & Experiment Matrix (platform_local + tests)

## Overview
The `platform_local/unified/` package is the single entry point for RL training in this repo: it dispatches one of five RL frameworks (`trl`, `tinker`, `verl`, `openrlhf`, `skyrl`) onto one of six compute backends (`local`, `modal`, `colab`, `vast`, `gcp`, `hfspaces`). Its job is not to implement training — it is to make 30 (framework × backend) cells reproducible from one frozen `CanonicalSpec` (Qwen3-8B / GSM8K / GRPO / 30 steps / G=8 / LoRA r=16), and to make that claim testable in CI without GPUs. The whole design pivots on splitting "what runs" (framework) from "where it runs" (backend), with `LaunchPlan` as the cheap intermediate that lets `--dry-run` and `test_unified_matrix.py` gate correctness without spend.

## Key Components
- `platform_local/unified/launcher.py:UnifiedLauncher` — orchestrator; outer dispatch on backend, inner `dispatch_framework()` for in-process paths (`_run_skyrl/_run_tinker/_run_verl/_run_openrlhf/_run_trl`).
- `platform_local/unified/launcher.py:TrainingResult` — return dataclass (`framework, model, algorithm, final_step, reward_history, loss_history`).
- `platform_local/unified/canonical.py:CanonicalSpec` — frozen dataclass; defaults are the Layer-B truth, optionally overridden by `preregistration.json` (defensive `load_spec()`).
- `platform_local/unified/backends/base.py:Backend (ABC)` + `LaunchPlan` — Strategy interface; `plan()` resolves a cell cost-free, `run()` shells out (default) or delegates to launcher.
- `platform_local/unified/backends/__init__.py` — lazy registry: `BACKENDS() -> dict[str, Backend]`, `get_backend(name)`.
- `platform_local/unified/backends/{local,colab,modal,gcp,vast,hfspaces}.py` — six strategies; `local`/`colab` go through `dispatch_framework`, others shell out to per-framework drivers.
- `platform_local/unified/__main__.py:main` — argparse CLI; `--framework/--backend` select the cell, `--dry-run` short-circuits to `LaunchPlan.format()`, `--generate-script` writes a real TRL GRPO Python file.
- `platform_local/unified/peft_utils.py` — lazy-import PEFT helpers; LoRA / prefix / p-tuning / prompt / BitFit, with k-bit prep and a custom BitFit checkpoint format.
- `platform_local/trl_integrations/{config,trainer}.py` — Pydantic-validated `TRLConfig` + `generate_trl_train_script()` used by both `--generate-script` and `_run_trl`.
- `platform_local/unified/matrix.json` — persisted 30-cell manifest (schema `tinkerrl-framework-backend-matrix-v1`); the artifact tests pin against.
- `tests/test_unified_matrix.py` — invariant gate covering manifest shape, live-vs-manifest drift, framework threading, Colab recursion, CLI dry-run, and shim delegation.

## Concepts & Decisions

### Two-axis framework × backend dispatch
- **What**: `UnifiedLauncher.run()` first dispatches on `self.backend` (where), and `local`/`colab` backends then call `launcher.dispatch_framework()` (what). Other backends shell out to a per-framework driver file.
- **Why used here**: cleanly separates two orthogonal concerns (RL algorithm library vs. compute provisioning) that were previously tangled in per-backend scripts. Each axis can grow independently.
- **Trade-offs**: 30 cells to keep coherent; risk that a backend "resolves" a cell but quietly runs the wrong framework. Mitigated by `test_each_cell_threads_its_framework`.
- **Alternatives**: a single `run(framework, backend)` monolith; or per-cell scripts with no shared abstraction. The matrix abstraction wins because it factors `CanonicalSpec` injection and dry-run uniformly.

### `LaunchPlan` as the dry-run seam
- **What**: `Backend.plan(framework, spec) -> LaunchPlan` returns `{command, driver_file, output, env, notes}` without compute. `run()` calls `plan()` first, then optionally `_execute`.
- **Why used here**: lets `--dry-run`, the matrix test, and the manifest all share one resolution path. If you can produce a `LaunchPlan`, you've proven the cell exists.
- **Trade-offs**: `plan()` and `run()` can drift if a backend overrides `run()` but forgets to keep `plan()` in sync. Tests pin them together (`test_manifest_matches_live_backends`).
- **Alternatives**: parse `--help`; mock `subprocess.run`. The `LaunchPlan` dataclass is more declarative and survives refactor.

### Canonical frozen spec as protocol
- **What**: `CanonicalSpec` is a `@dataclass(frozen=True)` with hardcoded Layer-B defaults (Qwen3-8B, GSM8K, GRPO, 30 steps, G=8, LoRA r=16, β=0, ε=0.2, seed=211). `load_spec()` only overrides from `preregistration.json` defensively.
- **Why used here**: this is *the* experiment every cell reproduces; freezing it makes cross-framework equivalence meaningful. Per-framework equivalence YAMLs live in `platform_hybrid/experiments/framework_config_dumps/` and are reachable via `spec.framework_config_path(fw)`.
- **Trade-offs**: less per-experiment flexibility; you trade knob count for comparability.
- **Alternatives**: a config tree with many YAMLs; would lose the "one protocol" invariant.

### Backend abstraction (Strategy pattern)
- **What**: `Backend` is an ABC with abstract `plan()` and a default `run()` that shells out. `LocalBackend`/`ColabBackend` override `run()` to call `launcher.dispatch_framework()` instead.
- **Why used here**: lets each backend encapsulate its provisioning idiom (Modal volumes, gcloud Spot VMs, vast.ai rentals, HF Spaces demos) while sharing the resolution contract.
- **Trade-offs**: backends with very different I/O shapes (serverless vs. notebook vs. SSH) are forced into one interface; acceptable because they all reduce to "run a command, point at a driver file."

### Test-as-invariant (the CI substitute for GPUs)
- **What**: `test_unified_matrix.py` cannot run real training, so it pins invariants: manifest well-formedness, every `plan()` returns a non-stub driver file that exists on disk, persisted `matrix.json` matches live `Backend.plan()` output, every cell references its named framework in command or driver filename, Colab does not recurse through `run_canonical.py`, every CLI cell resolves through `--dry-run`, and every `platform_<x>/run_experiment.py` shim pins its backend.
- **Why used here**: it is the only feasible CI gate for a 30-cell GPU matrix. The framework-threading test is the load-bearing check — it catches the bug class where a vast runner resolves every `--framework` to SkyRL.
- **Trade-offs**: invariants prove plumbing, not correctness of training. A cell can pass all tests and still produce wrong gradients.
- **Alternatives**: skip CI entirely; or run a tiny smoke job per cell (too expensive, flaky).

### Lazy PEFT integration
- **What**: `peft_utils.py` imports `peft` inside the function body and raises `RuntimeError("install the 'trl' extra")` if missing. `TRLModelConfig` uses Pydantic validators to enforce `4bit/8bit → use_peft=True` and mutual exclusion.
- **Why used here**: PEFT is optional for repo checkout; you can generate scripts on a laptop without `peft` installed.
- **Trade-offs**: errors surface at runtime, not import time; mitigated by Pydantic validation at config build.

## Related Code
- `platform_tinker/tinkerrl/grpo_cli.py` — the real Tinker GRPO loop, shelled by `_run_tinker`.
- `platform_hybrid/experiments/modal/modal_grpo_{trl,tinker,verl,openrlhf,skyrl}.py` — per-framework Modal H100 drivers, referenced by `ModalBackend`.
- `platform_hybrid/skyrl/backends/vastai_runner.py` — `VastAILauncher`, the vast.ai provisioner that threads `--framework` onto the instance.
- `zvf-program/next-submission/run_gcp_preflight.py` + `remote_preflight.py` — GCP A100 Spot plumbing; `trl` uses the validated `remote_preflight.py`, others clone and re-dispatch.
- `platform_{modal,colab,vast,gcp,hf_spaces}/run_experiment.py` — thin shims that pin their backend and forward to `python -m platform_local.unified`.
- `platform_local/trl_integrations/` — used by both `__main__.py` (`--generate-script`) and `_run_trl`, so the generated-script path and the in-process path cannot drift.

## Start Here
- `platform_local/unified/launcher.py` — shows the two-axis dispatch, dry-run seam, and per-framework `_run_*` methods. Read this first.
- `tests/test_unified_matrix.py` — the actual contract this subsystem is judged against; each test names an invariant.
- `platform_local/unified/backends/base.py` + `canonical.py` — the `Backend`/`LaunchPlan` interface and the frozen spec it consumes.

---
*Generated by AntiVibe (full-repo pass) · 2026-08-02*
