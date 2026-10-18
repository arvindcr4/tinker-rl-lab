"""Canonical benchmark experiment spec (Layer B — the frozen protocol).

Source of truth: ``zvf-program/next-submission/preregistration.json``
  Qwen/Qwen3-8B · GSM8K (+MATH-500) · canonical GRPO · 30 steps · G=8 · LoRA r=16.

This is the **one** experiment every framework × backend reproduces. Per-framework
equivalence configs live in
``platform_hybrid/experiments/framework_config_dumps/<fw>_qwen3_8b_gsm8k.yaml``.
The legacy Layer-A toy scripts (``implementations/trl_grpo_math.py`` etc.) are
intentionally out of scope.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO_ROOT / "zvf-program" / "next-submission" / "preregistration.json"
FRAMEWORK_DUMPS = REPO_ROOT / "platform_hybrid" / "experiments" / "framework_config_dumps"

FRAMEWORKS = ("trl", "tinker", "verl", "openrlhf", "skyrl")
BACKENDS = ("local", "modal", "colab", "vast", "gcp", "hfspaces")


@dataclass(frozen=True)
class CanonicalSpec:
    """The experiment reproduced across the framework × backend matrix.

    Defaults are the Layer-B frozen values (validated 106/106 in execution-notes).
    ``load_spec()`` overrides them from ``preregistration.json`` when that file is
    present, but the defaults themselves are authoritative for the matrix.
    """

    model: str = "Qwen/Qwen3-8B"
    task: str = "gsm8k"
    algorithm: str = "grpo"
    training_steps: int = 30
    num_generations: int = 8  # group size G
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "all-linear"
    beta: float = 0.0  # canonical GRPO: no KL penalty
    # Decoupled PPO-style clip (DAPO §3.1: 0.2/0.28).  Defaults preserve the
    # historical symmetric ε=0.2; set epsilon_high explicitly to opt in.
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    max_completion_length: int = 1024
    seed: int = 211
    protocol: str = "unified-resolved-v2"

    @property
    def spec_hash(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def __post_init__(self):
        if not self.model or not self.task or not self.algorithm:
            raise ValueError("Model, task and algorithm must be nonempty")
        for key in (
            "training_steps",
            "num_generations",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
            "max_completion_length",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")

    def framework_config_path(self, framework: str) -> Path | None:
        """Per-framework equivalence config (the 4-framework dump), if present."""
        p = FRAMEWORK_DUMPS / f"{framework}_qwen3_8b_gsm8k.yaml"
        return p if p.exists() else None

    def as_dict(self) -> dict:
        return asdict(self)


def load_spec() -> CanonicalSpec:
    """Resolve represented fields from the versioned preregistration, fail closed.

    This dispatch schema is NOT the entire frozen scientific protocol. Fields
    outside it (decoder, dataset revisions, runtime locks) require adapter parity
    validation before any result can be called a canonical reproduction.
    """
    data = json.loads(PREREGISTRATION.read_text())
    if data.get("schema_version") != "aiml-next-preregistration-v1":
        raise ValueError("Unsupported preregistration schema")
    t = data["treatment"]
    return CanonicalSpec(
        model=t["model"],
        training_steps=t["training_steps"],
        num_generations=t["num_generations"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        max_completion_length=t["max_completion_length"],
        learning_rate=t["optimizer"]["learning_rate"],
        lora_rank=t["adapter"]["r"],
        lora_alpha=t["adapter"]["lora_alpha"],
        lora_target_modules=t["adapter"]["target_modules"],
        beta=t["objective"]["beta"],
        epsilon_low=t["objective"]["epsilon"],
        epsilon_high=t["objective"]["epsilon_high"],
    )


def require_supported_overrides(spec, supported=()):
    expected = load_spec().as_dict()
    unsupported = [
        key
        for key, value in spec.as_dict().items()
        if key not in supported and key != "protocol" and value != expected[key]
    ]
    if unsupported:
        raise ValueError("Backend cannot honor overrides: " + ", ".join(unsupported))
