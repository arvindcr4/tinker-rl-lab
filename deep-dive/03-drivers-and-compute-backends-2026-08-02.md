# Deep Dive: Experiment Drivers & Compute Backends

## Overview
The "where it runs" layer is a registry of six interchangeable compute substrates (local, Modal, Colab, vast.ai, GCP Spot, HF Spaces) fronted by identical ~22-LOC entry shims. All five non-local shims (`platform_{modal,gcp,vast,colab,hf_spaces}/run_experiment.py`) prepend `--backend <name>` and delegate to `platform_local.unified.__main__`, which keeps the framework dimension (trl/tinker/verl/openrlhf/skyrl — the "what") orthogonal to the backend (the "where"). The launcher's `Backend.plan()` returns a `LaunchPlan` (command + driver + env + output) **without spending compute**, so `--dry-run`, the matrix test, and real launches flow through one path. Actual provisioning is delegated to one canonical driver per backend.

## Vendored vs project code in `platform_modal/`
Of ~537 .py files, only **two are project-authored launch code**: `run_experiment.py` (the entry shim) and `__init__.py`. The rest is a vendored experiment-scripts subtree retained for reproducibility but not invoked by the unified launcher.

- `platform_modal/scripts/berkeley/` — **~44 files, ~17k LOC** — paper-named scripts (`multiplicity_winners_curse.py`, `alphaproof_mcts_zvf.py`, `paper2verifier.py`) — **vendored — not analyzed**.
- `platform_modal/scripts/p5p8/` — **~278 files** — **vendored — not analyzed**.
- `platform_modal/scripts/figures/` — empty.
- `platform_modal/openrlhf/{__init__,config,trainer}.py` — **3 files** — abandoned shadow package sharing a name with PyPI `openrlhf`; the Modal driver comment explicitly avoids importing it and shells out to the PyPI CLI instead. **vendored — not analyzed**.

## Key Components
- `platform_local/unified/backends/base.py:Backend` — ABC; declares `plan() -> LaunchPlan` (no compute) and a default `run()` that prints + shells out.
- `platform_local/unified/backends/__init__.py:BACKENDS` — lazy registry of all six backends; `get_backend(name)` accessor.
- `platform_local/unified/canonical.py:CanonicalSpec` — frozen dataclass of matrix-truth (model, task, hyperparams); every `plan()` reads it.
- `platform_{modal,gcp,vast,colab,hf_spaces}/run_experiment.py:main` — ~22-LOC shims that prepend `--backend <name>` and call the unified launcher.
- `platform_local/unified/backends/modal.py:_PER_FW` — maps each framework to its per-framework Modal driver file.
- `platform_hybrid/experiments/modal/modal_grpo_openrlhf.py:run_openrlhf_qwen3_8b` — canonical Modal H100 driver (FastAPI reward server + subprocess to `openrlhf` CLI + W&B trace pull + HF Hub push).
- `platform_hybrid/experiments/modal_runner.py:main` — multi-seed × multi-framework sweep using `run_experiment.spawn()` + `aggregate_results.remote()`.
- `platform_hybrid/experiments/modal_batch_runner.py:run_trl_grpo` — TRL GRPO sweep; patches experiment source via string-replace before `exec`.
- `platform_hybrid/skyrl/backends/vastai_runner.py:VastAILauncher` — async-provisions vast.ai instances; `generate_setup_script()` clones repo + installs chosen framework; runs unified launcher on-box.
- `platform_local/unified/backends/gcp.py:GCPBackend` — delegates to `zvf-program/next-submission/run_gcp_preflight.py` (Spot A100 + cloud-init).
- `platform_local/unified/backends/hfspaces.py:HFSpacesBackend` — results-only; no training, calls `fetch_results.py` into a Gradio dashboard.
- `platform_hybrid/experiments/aggregate_results.py` — joins Modal/Tinker/consolidated JSONs into `master_results.{json,csv,md}`.

## Concepts & Decisions

### Plan/Run separation
- **What**: `Backend.plan()` resolves a cell into a `LaunchPlan` (concrete command, driver file, env, output path) without launching; `run()` then executes.
- **Why here**: lets `--dry-run` and the matrix test exercise every backend×framework cell for free (no GPU spend) and makes plan-testability uniform.
- **Trade-offs**: extra abstraction layer; commands baked as opaque strings (no streaming/inspection mid-flight).
- **Alternatives**: single `run()` with dry-run flags threaded through (loses uniformity); or per-backend CLIs (no central registry).

### Modal images + serverless GPU functions
- **What**: each per-framework Modal driver builds a `modal.Image` (CUDA base + pinned torch/vllm/openrlhf/flash-attn), declares `@app.function(gpu="H100", secrets=[...])`, and the local entrypoint calls `.remote()`.
- **Why here**: serverless H100s, zero instance management, image layering caches deps across runs, Volumes/Secrets handle persistence.
- **Trade-offs**: cold-start per image; pinned deps drift from local installs; vendor lock-in.
- **Alternatives**: Modal warm pools; or self-managed GPU VMs (vast/GCP), which the project already runs in parallel.

### Async fan-out for multi-seed sweeps
- **What**: `modal_runner.py` calls `run_experiment.spawn(name, file, seed)` for every cell, collects `Job` handles, then `.get()`s each as a barrier; `aggregate_results.remote()` runs after.
- **Why here**: maximizes parallel slot utilisation (10 SEEDS × 13 frameworks = 130 jobs); `.spawn` is non-blocking, `.get()` is the join, `retries=1` masks transient failures.
- **Trade-offs**: tail-latency dominated by the slowest job; no fail-fast; `.spawn` requires serializable args.
- **Alternatives**: `modal.Map` for ordered parallelism; or Ray/Dask for cross-cloud sweeps.

### Codegen'd setup script for vast.ai
- **What**: `VastAILauncher.generate_setup_script(framework)` emits a bash heredoc that apt-gets, installs `uv`, clones this repo, and runs a per-framework `uv pip install` line.
- **Why here**: vast.ai instances are bare Ubuntu; the chosen framework must be installed on-instance before training. Codegen keeps provisioning logic in Python (single source of truth) rather than a stale `.sh`.
- **Trade-offs**: ~5–10 min setup per instance; failures opaque (logged only to `/root/setup.log`); the same training command (`python -m platform_local.unified --backend local`) runs on-box, so framework bugs surface there not here.
- **Alternatives**: pre-bake a vast template image per framework; or `docker run` a pinned image (more storage, faster cold-start).

### Results-only HF Spaces backend
- **What**: `HFSpacesBackend` never trains; it runs `fetch_results.py` which pulls JSONs/traces from HF Hub + W&B + GCS into a Gradio dashboard.
- **Why here**: HF Spaces CPU tier can't host GPU RL training; results-fanout is what users actually want from a "demo" Space. Decoupling producer (Modal/GCP/vast) from consumer (Space) is cleaner than embedding dashboards in each driver.
- **Trade-offs**: requires producers to push to a stable namespace (`arvindcr4/tinker-rl-next-preflight-gcp-*`, W&B project); schema drift breaks the dashboard silently.
- **Alternatives**: host dashboard as GCS static site; or read directly from W&B via its API.

### Receipt/result persistence
- **What**: every driver writes `result.json` (success, elapsed, returncode, stdout/stderr tail, reward_trace) and commits to either a Modal Volume (`tinker-results`) or pushes to HF Hub; the aggregator joins these.
- **Why here**: serverless backends lose local state on teardown; durable receipts are the only proof a run happened, and the only input to the cross-framework comparison.
- **Trade-offs**: receipts are per-cell JSONs; no streaming of intermediate metrics; 2000-char stdout tail truncation loses long training logs.
- **Alternatives**: live W&B logging only (already runs in parallel — but no offline replay); or a centralized results DB.

## Related Code
- `platform_local/unified/launcher.py:UnifiedLauncher` owns the framework×backend matrix banner and `dispatch_framework()` — the in-process path used by local + colab backends.
- `platform_local/unified/backends/{local,modal,colab,vast,gcp,hfspaces}.py` — six `Backend` subclasses; local/colab short-circuit to `dispatch_framework`, modal/vast/gcp shell out to drivers, hfspaces is read-only.
- `zvf-program/next-submission/{run_gcp_preflight,remote_preflight}.py` — the GCP driver and validator that `GCPBackend` delegates to; emits HF/W&B/GCS receipts consumed by `hfspaces`.
- `platform_hybrid/experiments/framework_config_dumps/*.yaml` — frozen per-framework canonical configs (openrlhf/trl/verl/tinker, Qwen3-8B, GSM8K); these are the Layer-B truth `CanonicalSpec` loads.
- `platform_hybrid/experiments/results/aggregate_framework_comparison.py` — cross-framework aggregator the matrix test feeds.

## Start Here
1. `platform_local/unified/backends/base.py` (~71 LOC) — the `Backend` ABC + `LaunchPlan`; every other file in this layer keys off this contract.
2. `platform_local/unified/backends/__init__.py` + `modal.py` — the registry and one concrete backend to see the pattern.
3. `platform_hybrid/experiments/modal/modal_grpo_openrlhf.py` (~292 LOC) — canonical per-framework Modal H100 driver: image build, FastAPI reward server, subprocess to `openrlhf`, W&B trace pull, HF push, receipt write.

---
*Generated by AntiVibe (full-repo pass) · 2026-08-02*
