"""LOCAL backend — runs the framework in-process via UnifiedLauncher.dispatch_framework()."""

from __future__ import annotations

from platform_local.unified.backends.base import InProcessBackend, LaunchPlan

# The repo file that actually executes training per framework (local path).
_DRIVERS = {
    "trl": "platform_local/trl_integrations/trainer.py",
    "tinker": "platform_tinker/tinkerrl/grpo_cli.py",
    "verl": "verl/trainer.py",
    "openrlhf": "platform_modal/openrlhf/trainer.py",
    "skyrl": "platform_hybrid/skyrl/configs/grpo_gsm8k.yaml",
}


class LocalBackend(InProcessBackend):
    name = "local"

    def plan(self, framework: str, spec) -> LaunchPlan:
        return LaunchPlan(
            backend="local",
            framework=framework,
            argv=[
                "python",
                "-m",
                "platform_local.unified",
                "--backend",
                "local",
                "--framework",
                framework,
                "--model",
                spec.model,
                "--algorithm",
                spec.algorithm,
                "--task",
                spec.task,
                "--seed",
                str(spec.seed),
                "--epochs",
                str(spec.training_steps),
            ],
            driver_file=_DRIVERS.get(framework, "?"),
            output="./checkpoints/",
            env=["TINKER_API_KEY"] if framework in ("tinker", "skyrl") else [],
            notes=f"in-process dispatch; protocol={spec.protocol}; spec_hash={spec.spec_hash}",
        )

    # run() is inherited from InProcessBackend (in-process dispatch_framework).
