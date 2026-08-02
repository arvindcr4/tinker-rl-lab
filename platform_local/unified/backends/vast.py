"""VAST.AI backend — rented GPUs via the existing SkyRL vast.ai provisioner.

Reuses ``platform_hybrid/skyrl/backends/vastai_runner.py`` (``VastAILauncher``).
The thin on-instance shim is ``platform_vast/run_experiment.py``.
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_DRIVER = "platform_hybrid/skyrl/backends/vastai_runner.py"


class VastBackend(Backend):
    name = "vast"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="vast",
            framework=framework,
            command=(
                f"python -m platform_hybrid.skyrl.backends.vastai_runner "
                f"--model {spec.model} --algorithm {spec.algorithm} "
                f"--instance-type a100-80gb --num-instances 1"
            ),
            driver_file=_DRIVER,
            output="/root/tinker-rl-lab/platform_tinker/atropos/checkpoints/ + /root/results_summary.json",
            env=["VAST_API_KEY", "TINKER_API_KEY", "HF_TOKEN", "WANDB_API_KEY"],
            notes="provisions a vast.ai A100; on-instance script runs the canonical GSM8K config",
        )
