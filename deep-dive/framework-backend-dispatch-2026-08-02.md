# Deep Dive: Framework × Backend Dispatch (making every cell run its own framework)

**Generated**: 2026-08-02
**Files**: `launcher.py`, `backends/modal.py`, `backends/colab.py`, `backends/vast.py`, `backends/gcp.py`, `vastai_runner.py`, `run_gcp_preflight.py`, `modal_grpo_tinker.py`, `modal_grpo_skyrl.py`, `test_unified_matrix.py`

---

## Overview

This code makes a 5-framework × 6-backend experiment matrix honest. Previously each cell only needed its driver file to *exist* for the test suite to pass, so ~15 cells silently ran the wrong framework (vast ran SkyRL for everyone; gcp ran TRL for everyone; modal ran TRL for tinker/skyrl; colab infinite-recursed). The fix rests on one observation: the real per-framework training code already lived in `UnifiedLauncher.dispatch_framework()` (`_run_trl`/`_run_tinker`/`_run_verl`/`_run_openrlhf`/`_run_skyrl`). So instead of writing 15 new trainers, each remote backend is rewired to invoke that existing dispatch on its GPU box, and a new test enforces that no cell can swap frameworks again.

---

## Key Components

- `UnifiedLauncher.run()` (`launcher.py`): outer dispatch — picks in-process `dispatch_framework()` for `local` **and** `colab`, delegates other backends to their `Backend.run()`.
- `ModalBackend.plan()` (`backends/modal.py`): resolves a framework to its per-framework H100 driver via the `_PER_FW` dict.
- `VastAILauncher.generate_setup_script()` / `training_command_for()` (`vastai_runner.py`): builds a framework-aware setup script and the on-box command that runs unified dispatch for the chosen framework.
- `build_unified_entry_script()` (`run_gcp_preflight.py`): generates the Python entrypoint a fresh GCP VM runs to clone the repo and dispatch a non-TRL framework.
- `test_each_cell_threads_its_framework()` (`test_unified_matrix.py`): the invariant — every cell's plan must reference its named framework.

> Files in scope but summarized (offer to go deeper on request):
> - `backends/colab.py`: gives Colab a `run()` that delegates to `dispatch_framework()` instead of shelling back to its own entrypoint.
> - `backends/gcp.py` / `backends/vast.py`: thin `plan()` methods that thread `--framework` into the command string.
> - `modal_grpo_tinker.py` / `modal_grpo_skyrl.py`: new Modal H100 drivers mirroring the trl/verl shape, invoking `grpo_cli` and the SkyRL recipe respectively.

---

## Concepts & Decisions

### Two-axis dispatch, collapsed through one code path

- **What**: The system dispatches on two independent axes — *framework* (trl/tinker/verl/openrlhf/skyrl = "what algorithm stack runs") and *backend* (local/modal/colab/vast/gcp/hfspaces = "which GPU box it runs on"). A cell is one (framework, backend) pair.
- **Why used here**: The earlier design treated each backend as framework-aware but actually hardcoded one framework per backend. The rewrite makes the backends purely about *provisioning a box*, then runs `python -m platform_local.unified --framework <fw> --backend local` on that box — so all 25 training cells converge on the single already-tested `dispatch_framework()` code path. You trade a little startup cost (clone/install the repo on the box) for the guarantee that "vast runs verl" means the *same* verl code local runs, not a second implementation that can drift.

### Registry / Strategy pattern (`_PER_FW`)

- **What**: `modal.py` maps each framework to its driver file in a dict (`_PER_FW = {"trl": ..., "tinker": ...}`) and `plan()` is a pure lookup — no branching.
- **Why used here**: Modal is the one backend with genuinely distinct, framework-native H100 drivers (each framework's stack needs a different image). A dict lookup is the simplest way to say "one driver per framework" and makes a missing framework a loud `KeyError` instead of a silent fallback to the wrong driver — which is exactly the bug being fixed (the old code fell back to `--exp trl_grpo` for tinker/skyrl).

### Recursive entrypoint bug & fix

- **What**: `run_canonical.py` called the unified launcher with `--backend colab`; the colab backend's generic `run()` then shelled back out to `run_canonical.py` — an infinite loop if ever executed (only dry-runs worked). The fix is the one-line gate `if self.backend in ("local", "colab"): dispatch_framework()`, plus a `ColabBackend.run()` override that calls `dispatch_framework()` directly.
- **Why used here**: Colab *is* an on-box A100 runtime — it's architecturally identical to local, not a remote provisioner. Routing it through the same in-process dispatch as local both kills the recursion and correctly models what Colab actually is. The lesson: when a "backend" shares the execution model of another backend, route to that backend's path rather than adding a parallel one.

### Code generation for cloud-init (`build_unified_entry_script`)

- **What**: GCP doesn't ssh in; it hands a fresh Spot VM a startup script. `build_unified_entry_script` *generates a Python program as a string*, base64-embeds it, and the VM decodes and runs it. That generated program fetches secrets from the GCE metadata server, `git clone`s the repo, and runs unified dispatch.
- **Why used here**: The VM starts blank with no repo and no credentials in its filesystem — the only channel in is the startup script and the metadata server. Generating the entrypoint lets the trl path (validated, with HF/W&B/GCS receipt uploads) stay untouched while non-trl frameworks get a parallel generated entrypoint. The trade-off: generated code is harder to debug than code-on-disk, so it's kept minimal (clone, run, write `result.json`, exit) and reuses the same receipt uploader as trl.

### Testing the invariant, not the implementation

- **What**: `test_each_cell_threads_its_framework` asserts each plan references its framework (in the command or driver filename). It does *not* assert which file or how — just that the named framework is actually dispatched.
- **Why used here**: The original gate checked `driver_file.exists()` and a 7-cell dry-run sample — both passed while the matrix was wrong, because "a file exists" and "a plan prints" don't imply "the right framework runs." The new test encodes the actual correctness property (no silent framework swap) at the level the bug lived, so a future regression of the same shape fails fast. This is the general move: when a test passed but the system was broken, the test was checking the wrong layer — fix the *invariant*, not just the code.

---

*Generated by AntiVibe · `/antivibe full` for the extended version with resources and line-by-line walkthrough.*
