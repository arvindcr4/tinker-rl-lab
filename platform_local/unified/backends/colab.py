"""COLAB backend — A100 runtime via notebooks / a parametrized .py driver.

The canonical .py entry is ``platform_colab/run_canonical.py`` (added alongside
this package); notebooks (``advanced_rl_colab.ipynb`` etc.) remain for interactive
use. Reuses ``platform_tinker/atropos`` (Unsloth+TRL) for tinker/skyrl.
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_DRIVER = "platform_colab/run_canonical.py"
_NOTEBOOK = "platform_colab/advanced_rl_colab.ipynb"


class ColabBackend(Backend):
    name = "colab"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="colab",
            framework=framework,
            command=f"python {_DRIVER} --framework {framework} --model {spec.model}",
            driver_file=_DRIVER,
            output="platform_tinker/atropos/checkpoints/ (session-ephemeral)",
            env=["HF_TOKEN", "WANDB_API_KEY"],
            notes=f"A100 runtime; interactive alt: {_NOTEBOOK}",
        )
