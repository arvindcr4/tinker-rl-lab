"""COLAB backend — A100 runtime via notebooks / a parametrized .py driver.

Colab is an on-box A100 runtime, so training runs the framework **in-process**
through ``UnifiedLauncher.dispatch_framework()`` — the same all-framework path the
local backend uses. The canonical .py entry is ``platform_colab/run_canonical.py``
(it delegates to ``python -m platform_local.unified --backend colab``); notebooks
(``advanced_rl_colab.ipynb`` etc.) remain for interactive use.
"""

from __future__ import annotations

from platform_local.unified.backends.base import InProcessBackend, LaunchPlan

_DRIVER = "platform_colab/run_canonical.py"
_NOTEBOOK = "platform_colab/advanced_rl_colab.ipynb"


class ColabBackend(InProcessBackend):
    name = "colab"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="colab",
            framework=framework,
            command=f"python {_DRIVER} --framework {framework} --model {spec.model}",
            driver_file=_DRIVER,
            output="platform_tinker/atropos/checkpoints/ (session-ephemeral)",
            env=["HF_TOKEN", "WANDB_API_KEY"],
            notes=(
                "A100 runtime; runs unified in-process dispatch "
                f"(--backend local on the box); interactive alt: {_NOTEBOOK}"
            ),
        )

    # run() is inherited from InProcessBackend (in-process dispatch_framework).
