"""GCP backend — A100 Spot VM via the next-submission preflight plumbing.

Reuses ``zvf-program/next-submission/run_gcp_preflight.py`` (gcloud Spot-A100
provisioning) + ``remote_preflight.py`` (canonical TRL GRPO load/train/eval).
The ``--arm grpo_g8`` cell is baseline canonical GRPO (G=8), matching the
CanonicalSpec — so GCP runs the same experiment as every other backend.
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_LAUNCHER = "zvf-program/next-submission/run_gcp_preflight.py"
_DRIVER = "zvf-program/next-submission/remote_preflight.py"


class GCPBackend(Backend):
    name = "gcp"

    def plan(self, framework: str, spec) -> LaunchPlan:
        # The preflight path is TRL-backed (trl_sampler_adapter). Other frameworks
        # route to MODAL/LOCAL; GCP's training avenue is the TRL GRPO preflight.
        return LaunchPlan(
            backend="gcp",
            framework=framework,
            command=(
                f"python {_LAUNCHER} --task {spec.task} --arm grpo_g8 "
                f"--seed {spec.seed} --wait"
            ),
            driver_file=_DRIVER,
            output=(
                "HF repo arvindcr4/tinker-rl-next-preflight-gcp-* + W&B "
                "tinker-rl-lab + GCS receipts"
            ),
            env=["GCP_PROJECT", "WANDB_API_KEY", "HF_TOKEN"],
            notes=(
                "single A100 Spot (a2-highgpu-1g); reuse preflight provisioning "
                f"(launcher: {_LAUNCHER})"
            ),
        )
