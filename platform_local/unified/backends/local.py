"""LOCAL backend — runs the framework in-process via UnifiedLauncher.dispatch_framework()."""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

# The repo file that actually executes training per framework (local path).
_DRIVERS = {
    "trl": "platform_local/trl_integrations/trainer.py",
    "tinker": "platform_tinker/tinkerrl/grpo_cli.py",
    "verl": "verl/trainer.py",
    "openrlhf": "platform_modal/openrlhf/trainer.py",
    "skyrl": "platform_hybrid/skyrl/configs/grpo_gsm8k.yaml",
}


class LocalBackend(Backend):
    name = "local"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="local",
            framework=framework,
            command=(
                f"python -m platform_local.unified --backend local "
                f"--framework {framework} --model {spec.model}"
            ),
            driver_file=_DRIVERS.get(framework, "?"),
            output="./checkpoints/",
            env=["TINKER_API_KEY"] if framework in ("tinker", "skyrl") else [],
            notes="in-process framework dispatch",
        )

    def run(self, framework: str, spec, *, dry_run: bool = False, launcher=None):
        plan = self.plan(framework, spec)
        print(plan.format())
        if dry_run or launcher is None:
            return None
        # Delegate to the launcher's per-framework dispatch (the _run_* methods).
        return launcher.dispatch_framework()
