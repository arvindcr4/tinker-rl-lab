"""MODAL backend — serverless H100 via the existing per-framework Modal drivers.

Reuses:
  - platform_hybrid/experiments/modal/modal_grpo_{trl,verl,openrlhf}.py  (H100 drivers)
  - platform_hybrid/experiments/modal_runner.py                          (multi-framework sweep)
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

_PER_FW = {
    "trl": "platform_hybrid/experiments/modal/modal_grpo_trl.py",
    "verl": "platform_hybrid/experiments/modal/modal_grpo_verl.py",
    "openrlhf": "platform_hybrid/experiments/modal/modal_grpo_openrlhf.py",
}
_SWEEP = "platform_hybrid/experiments/modal_runner.py"  # tinker / skyrl path


class ModalBackend(Backend):
    name = "modal"

    def plan(self, framework: str, spec) -> LaunchPlan:
        if framework in _PER_FW:
            driver = _PER_FW[framework]
            cmd = f"modal run {driver}"
            notes = "Modal H100 serverless; per-framework driver"
        else:
            driver = _SWEEP
            cmd = f"modal run {driver} --exp trl_grpo --seeds 1"
            notes = "Modal A10G sweep runner (tinker/skyrl delegate here)"
        return LaunchPlan(
            backend="modal",
            framework=framework,
            command=cmd,
            driver_file=driver,
            output="Modal volume 'tinker-results' + /home/user/workspace/elevation_outputs/",
            env=["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN", "WANDB_API_KEY"],
            notes=notes,
        )
