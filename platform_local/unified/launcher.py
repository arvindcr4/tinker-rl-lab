import sys
import time
from dataclasses import dataclass


@dataclass
class TrainingResult:
    """Result of a training run."""

    framework: str
    model: str
    algorithm: str
    final_step: int
    reward_history: list[float]
    loss_history: list[float]
    total_time: float = 0.0


class UnifiedLauncher:
    """
    Unified launcher for all RL frameworks.

    Supports:
    - skyrl: SkyRL tx (Tinker API implementation)
    - tinker: Tinker Atropos (Atropos + Tinker API)
    - verl: Volcano Engine RL
    - openrlhf: OpenRLHF
    - trl: HuggingFace TRL
    """

    FRAMEWORKS = {
        "skyrl": "SkyRL tx (Local Tinker API)",
        "tinker": "Tinker Atropos",
        "verl": "Volcano Engine RL",
        "openrlhf": "OpenRLHF",
        "trl": "HuggingFace TRL",
    }

    # Compute backends (the "where" dimension; framework is the "what").
    BACKENDS = {
        "local": "Local GPU (in-process)",
        "modal": "Modal serverless H100",
        "colab": "Google Colab A100",
        "vast": "vast.ai rented GPUs",
        "gcp": "GCP A100 Spot preflight",
        "hfspaces": "HF Spaces (results demo + fetch)",
    }

    ALGORITHMS = {
        "grpo": "Group Relative Policy Optimization",
        "ppo": "Proximal Policy Optimization",
        "reinforce": "REINFORCE",
        "dapo": "DAPO",
        "dpo": "Direct Preference Optimization",
    }

    def __init__(self):
        self.framework = None
        self.model = None
        self.algorithm = "grpo"
        self.epochs = 20
        self.config = None
        self.use_peft = True
        self.peft_method = "lora"
        # Backend dispatch (framework × backend matrix).
        self.backend = "local"
        self.dry_run = False
        self.spec = None  # platform_local.unified.canonical.CanonicalSpec

    def print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 60)
        print("  Unified RL Training Launcher")
        print("=" * 60)
        print(f"\nAvailable Frameworks:")
        for key, desc in self.FRAMEWORKS.items():
            marker = "→" if key == self.framework else " "
            print(f"  {marker} {key:12s} - {desc}")
        print(f"\nAvailable Algorithms:")
        for key, desc in self.ALGORITHMS.items():
            marker = "→" if key == self.algorithm else " "
            print(f"  {marker} {key:12s} - {desc}")
        print(f"\nAvailable Backends:")
        for key, desc in self.BACKENDS.items():
            marker = "→" if key == self.backend else " "
            print(f"  {marker} {key:10s} - {desc}")
        print(f"\nSelected:")
        print(f"  Framework: {self.framework}")
        print(f"  Backend:  {self.backend}")
        print(f"  Model: {self.model}")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Epochs: {self.epochs}")
        if self.framework == "trl":
            tuning = self.peft_method if self.use_peft else "full fine-tuning"
            print(f"  Tuning: {tuning}")
        print("=" * 60 + "\n")

    def _resolve_spec(self):
        """Return the canonical spec, defaulting to the Layer-B frozen values."""
        if self.spec is not None:
            return self.spec
        from platform_local.unified.canonical import load_spec

        self.spec = load_spec()
        return self.spec

    def run(self):
        """Run the unified launcher.

        Outer dispatch is on backend (where to run); the local backend runs the
        selected framework in-process via :meth:`dispatch_framework`. Other
        backends delegate to their provisioning driver.
        """
        self.print_banner()
        spec = self._resolve_spec()

        # Dry-run: resolve the cell into a LaunchPlan without spending compute.
        if self.dry_run:
            from platform_local.unified.backends import get_backend

            plan = get_backend(self.backend).plan(self.framework, spec)
            print(plan.format())
            return

        start_time = time.time()

        if self.backend in ("local", "colab"):
            # Both run in-process on the GPU box they execute on (Colab is an
            # on-box A100 runtime, same as local). Going through dispatch_framework
            # here — rather than shelling back out to run_canonical.py — is what
            # breaks the colab entry's self-recursion.
            result = self.dispatch_framework()
        else:
            from platform_local.unified.backends import get_backend

            result = get_backend(self.backend).run(
                self.framework, spec, dry_run=False, launcher=self
            )

        total_time = time.time() - start_time

        if result is None:
            print(f"\n[done] {self.backend}/{self.framework} launched "
                  f"(see driver output for results); {total_time:.1f}s")
            return

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"\n  Framework: {result.framework}")
        print(f"  Model: {result.model}")
        print(f"  Final Step: {result.final_step}")
        if result.reward_history:
            print(f"  Final Reward: {result.reward_history[-1]:.4f}")
            print(f"  Peak Reward: {max(result.reward_history):.4f}")
        print(f"  Total Time: {total_time:.1f}s")
        print("=" * 60 + "\n")

    def dispatch_framework(self):
        """Inner dispatch on framework (the local backend's training path)."""
        if self.framework == "skyrl":
            return self._run_skyrl()
        elif self.framework == "tinker":
            return self._run_tinker()
        elif self.framework == "verl":
            return self._run_verl()
        elif self.framework == "openrlhf":
            return self._run_openrlhf()
        elif self.framework == "trl":
            return self._run_trl()
        print(f"Unknown framework: {self.framework}")
        print(f"Available: {', '.join(self.FRAMEWORKS.keys())}")
        sys.exit(1)

    def _run_skyrl(self) -> TrainingResult:
        """Run SkyRL tx training.

        SkyRL tx runs the Tinker API on your own GPUs behind an external SkyRL
        checkout; it is not pip-installable, so we emit the documented launch
        sequence (from README §SkyRL) and shell out when an SKYRL_CHECKOUT env
        var is present.
        """
        import os
        import subprocess

        spec = self._resolve_spec()
        print("\n[SKYRL] SkyRL tx (self-hosted Tinker API server)")
        cfg = "platform_hybrid/skyrl/configs/grpo_gsm8k.yaml"
        checkout = os.environ.get("SKYRL_CHECKOUT")
        if not checkout:
            raise NotImplementedError(
                "SkyRL tx requires an external SkyRL checkout. Set SKYRL_CHECKOUT to "
                "its path, then: `uv run --extra gpu --extra tinker -m skyrl.tinker.api` "
                f"and run the recipe at {cfg}. (The Tinker-cloud equivalent is "
                "`--framework tinker`, which runs in-process via grpo_cli.)"
            )
        subprocess.run(
            ["uv", "run", "--extra", "gpu", "--extra", "tinker", "-m", "skyrl.tinker.api"],
            cwd=checkout,
            check=True,
        )
        return TrainingResult("skyrl", spec.model, self.algorithm, spec.training_steps, [], [])

    def _run_tinker(self) -> TrainingResult:
        """Run Tinker (Atropos / GRPO-CLI) training in-process.

        Delegates to the real Tinker SDK loop in
        ``platform_tinker/tinkerrl/grpo_cli.py`` (preset gsm8k, ExactMathReward).
        """
        import subprocess

        spec = self._resolve_spec()
        cmd = [
            sys.executable,
            "-m",
            "platform_tinker.tinkerrl.grpo_cli",
            "--preset",
            spec.task,
            "--model",
            self.model or spec.model,
            "--steps",
            str(spec.training_steps),
            "--seed",
            str(spec.seed),
        ]
        print(f"\n[TINKER] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return TrainingResult(
            "tinker", self.model or spec.model, self.algorithm, spec.training_steps, [], []
        )

    def _run_verl(self) -> TrainingResult:
        """Run verl training."""
        print("\n[VERL] Starting verl training...")

        try:
            from verl.config import VERLConfig, VERLModelConfig, VERLAlgorithmConfig
            from verl.trainer import VERLTrainer
            import asyncio

            config = VERLConfig(
                model=VERLModelConfig(model_name=self.model),
                algorithm=VERLAlgorithmConfig(algorithm=self.algorithm),
            )
            trainer = VERLTrainer(config)
            asyncio.run(trainer.run())

            return TrainingResult(
                framework="verl",
                model=self.model,
                algorithm=self.algorithm,
                final_step=self.epochs,
                reward_history=[],
                loss_history=[],
            )
        except ImportError:
            raise NotImplementedError("verl is not installed. Install with: pip install verl")

    def _run_openrlhf(self) -> TrainingResult:
        """Run OpenRLHF training.

        Delegates to ``platform_modal/openrlhf/trainer.py:run_openrlhf_training``
        (subprocesses ``openrlhf.cli.train_ppo_ray`` with group-norm advantage)
        when OpenRLHF + Ray + vLLM are installed; otherwise emits the documented
        CLI command. The hosted-equivalent path is the MODAL backend
        (``modal_grpo_openrlhf.py``).
        """
        spec = self._resolve_spec()
        print("\n[OPENRLHF] OpenRLHF (Ray + vLLM, group-norm GRPO)")
        try:
            from platform_modal.openrlhf.trainer import run_openrlhf_training  # type: ignore
        except ImportError:
            raise NotImplementedError(
                "Local OpenRLHF needs `openrlhf` + Ray + vLLM. Install openrlhf or "
                "use `--backend modal` (modal_grpo_openrlhf.py). CLI equivalent: "
                "`python -m openrlhf.cli.train_ppo_ray --advantage_estimator group_norm`."
            )
        run_openrlhf_training({}, output_dir="./checkpoints/openrlhf")
        return TrainingResult("openrlhf", spec.model, self.algorithm, spec.training_steps, [], [])

    def _run_trl(self) -> TrainingResult:
        """Run TRL training.

        Generates the checkpoint-resumable GRPO script
        (``trl_integrations.trainer.generate_trl_train_script``) and executes it
        when training data is available; otherwise points the user at
        ``--generate-script`` / the MODAL & GCP TRL drivers, which carry the
        canonical GSM8K run.
        """
        import os
        import tempfile

        spec = self._resolve_spec()
        train_data = getattr(self, "train_data", []) or []
        print("\n[TRL] HuggingFace TRL GRPO")
        if not train_data:
            raise NotImplementedError(
                "Local TRL training needs --train-data (a GSM8K JSON). Generate one with "
                "`--framework trl --generate-script train_grpo.py`, or use the canonical "
                "TRL drivers: `--backend modal` (modal_grpo_trl.py) or `--backend gcp` "
                "(remote_preflight.py)."
            )
        try:
            from platform_local.trl_integrations.config import TRLConfig
            from platform_local.trl_integrations.trainer import generate_trl_train_script
        except ImportError:
            from trl_integrations.config import TRLConfig  # type: ignore
            from trl_integrations.trainer import generate_trl_train_script  # type: ignore

        config = TRLConfig(
            model={"model_name": self.model or spec.model, "use_peft": self.use_peft,
                   "peft_method": self.peft_method, "lora_rank": spec.lora_rank},
            optimizer={"learning_rate": spec.learning_rate},
            algorithm={"algorithm": self.algorithm},
            data={"train_data": train_data},
            epochs=self.epochs,
            project_name=getattr(self, "wandb_project", "unified-rl"),
        )
        script = os.path.join(tempfile.mkdtemp(), "train_trl.py")
        generate_trl_train_script(config, script)
        print(f"[TRL] generated {script}; executing")
        import subprocess

        subprocess.run([sys.executable, script], check=True)
        return TrainingResult("trl", self.model or spec.model, self.algorithm, self.epochs, [], [])
