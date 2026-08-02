"""COLAB backend — A100 runtime via notebooks / a parametrized .py driver.

Colab is an on-box A100 runtime, so training runs the framework **in-process**
through ``UnifiedLauncher.dispatch_framework()`` — the same all-framework path the
local backend uses. The canonical .py entry is ``platform_colab/run_canonical.py``
(it delegates to ``python -m platform_local.unified --backend colab``); notebooks
(``advanced_rl_colab.ipynb`` etc.) remain for interactive use.
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
            notes=(
                "A100 runtime; runs unified in-process dispatch "
                f"(--backend local on the box); interactive alt: {_NOTEBOOK}"
            ),
        )

    def run(self, framework: str, spec, *, dry_run: bool = False, launcher=None):
        plan = self.plan(framework, spec)
        print(plan.format())
        if dry_run or launcher is None:
            return None
        # In-process on the Colab box — same path as the local backend. Delegating
        # to dispatch_framework (not shelling back out to run_canonical.py) is what
        # avoids the entry-point self-recursion.
        return launcher.dispatch_framework()
