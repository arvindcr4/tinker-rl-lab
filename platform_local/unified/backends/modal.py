"""MODAL backend — serverless H100 via the per-framework Modal drivers.

Reuses one H100 driver per framework:
  platform_hybrid/experiments/modal/modal_grpo_{trl,tinker,verl,openrlhf,skyrl}.py
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_PER_FW = {
    "trl": "platform_hybrid/experiments/modal/modal_grpo_trl.py",
    "tinker": "platform_hybrid/experiments/modal/modal_grpo_tinker.py",
    "verl": "platform_hybrid/experiments/modal/modal_grpo_verl.py",
    "openrlhf": "platform_hybrid/experiments/modal/modal_grpo_openrlhf.py",
    "skyrl": "platform_hybrid/experiments/modal/modal_grpo_skyrl.py",
}


class ModalBackend(Backend):
    name = "modal"

    def plan(self, framework: str, spec) -> LaunchPlan:
        driver = _PER_FW[framework]
        return LaunchPlan(
            backend="modal",
            framework=framework,
            command=f"modal run {driver}",
            driver_file=driver,
            output="Modal volume 'tinker-results' + /home/user/workspace/elevation_outputs/",
            env=["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN", "WANDB_API_KEY"],
            notes=f"Modal H100 serverless; per-framework driver ({framework})",
        )
