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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
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
    epsilon: float = 0.2
    max_completion_length: int = 1024
    seed: int = 211

    def framework_config_path(self, framework: str) -> Path | None:
        """Per-framework equivalence config (the 4-framework dump), if present."""
        p = FRAMEWORK_DUMPS / f"{framework}_qwen3_8b_gsm8k.yaml"
        return p if p.exists() else None

    def as_dict(self) -> dict:
        return asdict(self)


def load_spec() -> CanonicalSpec:
    """Return the canonical spec, best-effort overriding from preregistration.json.

    Fully defensive: any structural drift in the JSON falls back to the dataclass
    defaults (which are themselves the Layer-B truth).
    """
    spec = CanonicalSpec()
    if not PREREGISTRATION.exists():
        return spec
    try:
        data = json.loads(PREREGISTRATION.read_text())
        overrides = {
            "model": data.get("included_model") or data.get("model"),
            "training_steps": data.get("training_steps"),
            "num_generations": data.get("num_generations"),
        }
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(spec, **clean)
    except Exception:
        return spec
