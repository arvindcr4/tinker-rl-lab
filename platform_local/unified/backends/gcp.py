"""GCP backend — A100 Spot VM, split across two launchers.

TRL keeps the validated, **frozen** preflight launcher
``zvf-program/next-submission/run_gcp_preflight.py`` (hash-anchored to the
preregistered protocol, with HF/W&B/GCS receipts). The other four frameworks
route to ``platform_gcp/run_unified_dispatch.py`` — a non-frozen launcher that
provisions the same Spot A100 shape and runs the unified launcher's in-process
dispatch for the chosen framework on the VM.

The split exists so framework dispatch can evolve without modifying the frozen
protocol file. The ``--arm grpo_g8`` cell is baseline canonical GRPO (G=8),
matching the CanonicalSpec — so GCP runs the same experiment as every other
backend.
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_FROZEN_LAUNCHER = "zvf-program/next-submission/run_gcp_preflight.py"
_UNIFIED_LAUNCHER = "platform_gcp/run_unified_dispatch.py"


class GCPBackend(Backend):
    name = "gcp"

    def plan(self, framework: str, spec) -> LaunchPlan:
        if framework == "trl":
            command = (
                f"python {_FROZEN_LAUNCHER} --task {spec.task} --arm grpo_g8 "
                f"--seed {spec.seed} --wait"
            )
            driver_file = _FROZEN_LAUNCHER
            notes = (
                "single A100 Spot (a2-highgpu-1g); validated TRL preflight "
                "(frozen, hash-anchored) with HF/W&B/GCS receipts"
            )
        else:
            command = (
                f"python {_UNIFIED_LAUNCHER} --framework {framework} "
                f"--task {spec.task} --seed {spec.seed} --wait"
            )
            driver_file = _UNIFIED_LAUNCHER
            notes = (
                f"single A100 Spot (a2-highgpu-1g); non-frozen launcher runs "
                f"unified in-process dispatch for {framework} on the VM"
            )
        return LaunchPlan(
            backend="gcp",
            framework=framework,
            command=command,
            driver_file=driver_file,
            output=(
                "HF repo arvindcr4/tinker-rl-next-preflight-gcp-* + W&B "
                "tinker-rl-lab + GCS receipts"
            ),
            env=["GCP_PROJECT", "WANDB_API_KEY", "HF_TOKEN"],
            notes=notes,
        )
