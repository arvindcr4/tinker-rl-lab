"""Pinned, CPU-only TRL 1.2.0 differential harness for S1 fixtures.

This module does not initialize a model or trainer and cannot launch training.
It feeds fixture log-probabilities through the loss implementation on
``GRPOTrainer._compute_loss`` and maps the result into the canonical ``Trace``
contract. Reward-to-advantage calculation is a literal transcription of TRL
1.2.0 ``GRPOTrainer._generate_and_score_completions`` because that method also
performs generation, distributed gathering, and reward-function execution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Literal, Sequence

import torch

from .fixtures import FIXTURES, ObjectiveFixture
from .reference import ATOL, RTOL, Trace, canonical_advantages, objective_trace

TRL_VERSION = "1.2.0"
TRANSFORMERS_VERSION = "5.5.4"
TRL_WHEEL_SHA256 = "f6ddfa162ac92d25973070d9e3f6cff71b32c52edc34539e4294722f9dc0a6d6"
GRPO_TRAINER_SHA256 = "4ac442a6d117838bf26fa3b786feb092c3953b2f9212081646db7fad9b356b41"
GRPO_CONFIG_SHA256 = "1c342bb4afcc56224cdec752feba1fa11d2137aa0c4fe9c15e4e03d4d711095e"
ConformanceVerdict = Literal[
    "PASS",
    "NUMERICAL_VARIATION",
    "MATERIAL_DIFFERENCE",
    "NOT_TESTED",
]


class TRLPinError(RuntimeError):
    """The imported stack does not match the preregistered TRL runtime."""


class TRLUnsupportedObjective(ValueError):
    """The requested canonical objective has no native TRL 1.2.0 mapping."""


@dataclass(frozen=True)
class TRLArmConfig:
    arm: str
    loss_type: str
    scale_rewards: str
    importance_sampling_level: str
    epsilon_low: float
    epsilon_high: float
    max_completion_length: int


@dataclass(frozen=True)
class TRLProvenance:
    trl_version: str
    transformers_version: str
    torch_version: str
    python_version: str
    platform: str
    device: str
    cuda_available: bool
    lockfile: str
    lockfile_sha256: str
    locked_wheel_url: str
    locked_wheel_sha256: str
    trainer_source: str
    trainer_source_sha256: str
    config_source: str
    config_source_sha256: str
    exercised_api: str
    advantage_source: str


@dataclass(frozen=True)
class FieldComparison:
    field: str
    agrees: bool
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class TRLDifferential:
    fixture: str
    arm: str
    actual: Trace
    expected: Trace
    config: TRLArmConfig
    provenance: TRLProvenance
    fields: tuple[FieldComparison, ...]
    formula_notes: tuple[str, ...]
    actual_semantics: str = "native_trl_1.2.0"

    @property
    def conforms(self) -> bool:
        return all(field.agrees for field in self.fields)

    @property
    def verdict(self) -> ConformanceVerdict:
        if self.conforms:
            return "PASS"
        disagreements = tuple(field for field in self.fields if not field.agrees)
        if all(
            field.max_abs_error <= 10 * ATOL or field.max_rel_error <= 10 * RTOL
            for field in disagreements
        ):
            return "NUMERICAL_VARIATION"
        return "MATERIAL_DIFFERENCE"

    def summary(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "arm": self.arm,
            "conforms": self.conforms,
            "verdict": self.verdict,
            "actual_semantics": self.actual_semantics,
            "expected_semantics": "canonical_s1_reference",
            "actual_loss": self.actual.loss.item(),
            "expected_loss": self.expected.loss.item(),
            "config": asdict(self.config),
            "provenance": asdict(self.provenance),
            "fields": [asdict(field) for field in self.fields],
            "formula_notes": list(self.formula_notes),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _locked_trl() -> tuple[Path, str, str]:
    lockfile = _repo_root() / "uv.lock"
    payload = tomllib.loads(lockfile.read_text())
    matches = [
        package
        for package in payload["package"]
        if package.get("name") == "trl" and package.get("version") == TRL_VERSION
    ]
    if len(matches) != 1:
        raise TRLPinError(f"uv.lock must contain exactly one trl=={TRL_VERSION} package")
    wheels = matches[0].get("wheels", [])
    pinned = [wheel for wheel in wheels if wheel.get("hash") == f"sha256:{TRL_WHEEL_SHA256}"]
    if len(pinned) != 1:
        raise TRLPinError("uv.lock does not contain the expected TRL 1.2.0 wheel hash")
    return lockfile, pinned[0]["url"], pinned[0]["hash"].removeprefix("sha256:")


def load_pinned_runtime() -> tuple[type[Any], TRLProvenance]:
    """Load and verify the exact stack/source pinned by the preregistration."""
    try:
        import transformers
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise TRLPinError(
            "TRL is unavailable; run this harness in an isolated "
            "`uv run --no-sync --with trl==1.2.0 --with transformers==5.5.4 ...` environment"
        ) from error

    trl_version = version("trl")
    transformers_version = transformers.__version__
    if trl_version != TRL_VERSION:
        raise TRLPinError(f"expected trl=={TRL_VERSION}, found {trl_version}")
    if transformers_version != TRANSFORMERS_VERSION:
        raise TRLPinError(
            f"expected transformers=={TRANSFORMERS_VERSION}, found {transformers_version}"
        )

    trainer_source = Path(inspect.getfile(GRPOTrainer)).resolve()
    config_source = Path(inspect.getfile(GRPOConfig)).resolve()
    trainer_hash, config_hash = _sha256(trainer_source), _sha256(config_source)
    if trainer_hash != GRPO_TRAINER_SHA256:
        raise TRLPinError(f"unexpected GRPOTrainer source hash: {trainer_hash}")
    if config_hash != GRPO_CONFIG_SHA256:
        raise TRLPinError(f"unexpected GRPOConfig source hash: {config_hash}")

    method_source = inspect.getsource(GRPOTrainer._compute_loss)
    required_fragments = (
        'if self.importance_sampling_level == "token":',
        "coef_1 = torch.exp(log_importance_weights)",
        "per_token_loss = -torch.min(per_token_loss1, per_token_loss2)",
        'elif self.loss_type in ["cispo", "dapo", "vespo"]:',
    )
    if any(fragment not in method_source for fragment in required_fragments):
        raise TRLPinError("TRL _compute_loss API/source shape differs from the pinned harness")

    lockfile, wheel_url, wheel_hash = _locked_trl()
    provenance = TRLProvenance(
        trl_version=trl_version,
        transformers_version=transformers_version,
        torch_version=torch.__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        device="cpu",
        cuda_available=torch.cuda.is_available(),
        lockfile=str(lockfile.relative_to(_repo_root())),
        lockfile_sha256=_sha256(lockfile),
        locked_wheel_url=wheel_url,
        locked_wheel_sha256=wheel_hash,
        trainer_source=str(trainer_source),
        trainer_source_sha256=trainer_hash,
        config_source=str(config_source),
        config_source_sha256=config_hash,
        exercised_api="trl.GRPOTrainer._compute_loss",
        advantage_source=f"{trainer_source}:2133-2155",
    )
    return GRPOTrainer, provenance


def _arm_config(arm: str, max_completion_length: int) -> TRLArmConfig:
    if arm == "aero":
        raise TRLUnsupportedObjective(
            "TRL 1.2.0 exposes no native AERO posterior-advantage objective"
        )
    if arm not in {"grpo", "dapo", "gspo", "drgrpo"}:
        raise ValueError(f"unknown objective arm: {arm}")
    return TRLArmConfig(
        arm=arm,
        loss_type={"dapo": "dapo", "drgrpo": "dr_grpo"}.get(arm, "grpo"),
        scale_rewards="none" if arm == "drgrpo" else "group",
        importance_sampling_level="sequence" if arm == "gspo" else "token",
        epsilon_low=0.20,
        epsilon_high=0.28 if arm == "dapo" else 0.20,
        max_completion_length=max_completion_length,
    )


def _intended_arm_config(arm: str, max_completion_length: int) -> TRLArmConfig:
    if arm not in {"grpo", "dapo", "gspo", "drgrpo", "aero"}:
        raise ValueError(f"unknown objective arm: {arm}")
    return TRLArmConfig(
        arm=arm,
        loss_type="grpo",
        scale_rewards="canonical_s1_precomputed",
        importance_sampling_level="sequence" if arm == "gspo" else "token",
        epsilon_low=0.20,
        epsilon_high=0.28 if arm == "dapo" else 0.20,
        max_completion_length=max_completion_length,
    )


def _validate_fixture(
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if rewards.ndim != 2 or logps.ndim != 2:
        raise ValueError("rewards and logps must both be rank two")
    if rewards.shape[1] < 2:
        raise ValueError("TRL group normalization requires at least two generations")
    if rewards.numel() != logps.shape[0]:
        raise ValueError("flattened rewards must match completion rows")
    if logps.shape != old_logps.shape or logps.shape != mask.shape:
        raise ValueError("logps, old_logps, and mask must have identical shapes")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("mask values must be in [0, 1]")


def _trl_advantages(rewards: torch.Tensor, config: TRLArmConfig) -> torch.Tensor:
    """TRL 1.2.0 lines 2133-2155, isolated from generation side effects."""
    means = rewards.mean(dim=1, keepdim=True)
    centered = rewards - means
    if config.scale_rewards == "none":
        return centered.flatten()
    stds = rewards.std(dim=1, correction=1, keepdim=True)
    return (centered / (stds + 1e-4)).flatten()


def _trainer_shell(
    trainer_class: type[Any],
    config: TRLArmConfig,
    per_token_logps: torch.Tensor,
) -> Any:
    trainer = object.__new__(trainer_class)
    trainer.model = torch.nn.Identity().train()
    trainer.top_entropy_quantile = 1.0
    trainer.importance_sampling_level = config.importance_sampling_level
    trainer.beta = 0.0
    trainer.loss_type = config.loss_type
    trainer.epsilon_low = config.epsilon_low
    trainer.epsilon_high = config.epsilon_high
    trainer.off_policy_mask_threshold = None
    trainer.use_vllm = False
    trainer.vllm_importance_sampling_correction = False
    trainer.current_gradient_accumulation_steps = 1
    trainer.max_completion_length = config.max_completion_length
    trainer.accelerator = SimpleNamespace(num_processes=1, gather=lambda value: value)
    trainer.args = SimpleNamespace(use_bias_correction_kl=False, delta=None)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def fixture_logps(self: Any, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        del self, args, kwargs
        return per_token_logps, torch.zeros_like(per_token_logps)

    trainer._get_per_token_logps_and_entropies = MethodType(fixture_logps, trainer)
    return trainer


def trl_trace(
    *,
    arm: str,
    rewards: torch.Tensor | Sequence[Sequence[float]],
    logps: torch.Tensor | Sequence[Sequence[float]],
    old_logps: torch.Tensor | Sequence[Sequence[float]],
    mask: torch.Tensor | Sequence[Sequence[float]],
    selected_indices: Sequence[int] | None = None,
) -> tuple[Trace, TRLArmConfig, TRLProvenance]:
    """Evaluate supplied log-probabilities through pinned TRL on CPU."""
    trainer_class, provenance = load_pinned_runtime()
    if provenance.cuda_available and torch.get_default_device().type != "cpu":
        raise TRLPinError("S1 fixture harness must run on CPU")

    rewards_t = torch.as_tensor(rewards, dtype=torch.float64, device="cpu")
    logps_t = torch.as_tensor(logps, dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    old_t = torch.as_tensor(old_logps, dtype=torch.float64, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.float64, device="cpu")
    _validate_fixture(rewards_t, logps_t, old_t, mask_t)
    config = _arm_config(arm, logps_t.shape[1])

    chosen = tuple(range(logps_t.shape[0])) if selected_indices is None else tuple(selected_indices)
    if any(index < 0 or index >= logps_t.shape[0] for index in chosen):
        raise ValueError("selected index out of range")
    selected = torch.zeros(logps_t.shape[0], dtype=torch.float64)
    selected[list(chosen)] = 1.0
    mask_t = mask_t * selected[:, None]
    if not torch.any(mask_t.sum(dim=1) > 0):
        raise ValueError("selection must retain one active completion")

    advantages = _trl_advantages(rewards_t, config)
    delta = logps_t - old_t
    if config.importance_sampling_level == "sequence":
        log_weights = (delta * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp_min(1.0)
        ratios = log_weights.exp()[:, None].expand_as(logps_t)
    else:
        ratios = delta.exp()
    clipped = ratios.clamp(1.0 - config.epsilon_low, 1.0 + config.epsilon_high)
    surrogate = -torch.minimum(
        ratios * advantages[:, None],
        clipped * advantages[:, None],
    )

    trainer = _trainer_shell(trainer_class, config, logps_t)
    inputs = {
        "prompt_ids": torch.empty((logps_t.shape[0], 0), dtype=torch.long),
        "prompt_mask": torch.empty((logps_t.shape[0], 0), dtype=torch.float64),
        "completion_ids": torch.zeros_like(logps_t, dtype=torch.long),
        "completion_mask": mask_t,
        "advantages": advantages,
        "old_per_token_logps": old_t,
        "num_items_in_batch": mask_t.sum(),
    }
    loss = trainer_class._compute_loss(trainer, trainer.model, inputs)
    gradient = torch.autograd.grad(loss, logps_t)[0].flatten()
    trace = Trace(
        arm=arm,
        advantages=advantages.detach(),
        ratios=ratios.detach(),
        mask=mask_t.detach(),
        surrogate=surrogate.detach(),
        loss=loss.detach(),
        gradient=gradient.detach(),
        selected_indices=chosen,
    )
    return trace, config, provenance


def trl_intended_trace(
    *,
    arm: str,
    rewards: torch.Tensor | Sequence[Sequence[float]],
    logps: torch.Tensor | Sequence[Sequence[float]],
    old_logps: torch.Tensor | Sequence[Sequence[float]],
    mask: torch.Tensor | Sequence[Sequence[float]],
    selected_indices: Sequence[int] | None = None,
    aero_successes: Sequence[int] | None = None,
    aero_observations: Sequence[int] | None = None,
) -> tuple[Trace, TRLArmConfig, TRLProvenance]:
    """Exercise the exact S1 treatment through TRL's pinned loss kernel."""
    trainer_class, provenance = load_pinned_runtime()
    rewards_t = torch.as_tensor(rewards, dtype=torch.float64, device="cpu")
    logps_t = torch.as_tensor(logps, dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    old_t = torch.as_tensor(old_logps, dtype=torch.float64, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.float64, device="cpu")
    _validate_fixture(rewards_t, logps_t, old_t, mask_t)
    config = _intended_arm_config(arm, logps_t.shape[1])

    chosen = tuple(range(logps_t.shape[0])) if selected_indices is None else tuple(selected_indices)
    if any(index < 0 or index >= logps_t.shape[0] for index in chosen):
        raise ValueError("selected index out of range")
    selected = torch.zeros(logps_t.shape[0], dtype=torch.float64)
    selected[list(chosen)] = 1.0
    mask_t = mask_t * selected[:, None]
    active = mask_t.sum(dim=1) > 0
    if not torch.any(active):
        raise ValueError("selection must retain one active completion")

    advantages = canonical_advantages(
        rewards_t,
        active,
        arm=arm,
        aero_successes=aero_successes,
        aero_observations=aero_observations,
    )
    delta = logps_t - old_t
    if config.importance_sampling_level == "sequence":
        log_weights = (delta * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp_min(1.0)
        ratios = log_weights.exp()[:, None].expand_as(logps_t)
    else:
        ratios = delta.exp()
    clipped = ratios.clamp(1.0 - config.epsilon_low, 1.0 + config.epsilon_high)
    surrogate = -torch.minimum(
        ratios * advantages[:, None],
        clipped * advantages[:, None],
    )

    active_logps = logps_t[active]
    active_old = old_t[active]
    active_mask = mask_t[active]
    active_advantages = advantages[active]
    trainer = _trainer_shell(trainer_class, config, active_logps)
    inputs = {
        "prompt_ids": torch.empty((active_logps.shape[0], 0), dtype=torch.long),
        "prompt_mask": torch.empty((active_logps.shape[0], 0), dtype=torch.float64),
        "completion_ids": torch.zeros_like(active_logps, dtype=torch.long),
        "completion_mask": active_mask,
        "advantages": active_advantages,
        "old_per_token_logps": active_old,
        "num_items_in_batch": active_mask.sum(),
    }
    loss = trainer_class._compute_loss(trainer, trainer.model, inputs)
    gradient = torch.autograd.grad(loss, logps_t)[0].flatten()
    trace = Trace(
        arm=arm,
        advantages=advantages.detach(),
        ratios=ratios.detach(),
        mask=mask_t.detach(),
        surrogate=surrogate.detach(),
        loss=loss.detach(),
        gradient=gradient.detach(),
        selected_indices=chosen,
    )
    return trace, config, provenance


def _field_comparison(field: str, actual: torch.Tensor, expected: torch.Tensor) -> FieldComparison:
    difference = (actual - expected).abs()
    max_abs = difference.max().item() if difference.numel() else 0.0
    denominator = expected.abs().clamp_min(ATOL)
    max_rel = (difference / denominator).max().item() if difference.numel() else 0.0
    return FieldComparison(
        field=field,
        agrees=torch.allclose(actual, expected, rtol=RTOL, atol=ATOL),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
    )


def _formula_notes(arm: str, actual_mask: torch.Tensor) -> tuple[str, ...]:
    notes: list[str] = []
    if arm in {"grpo", "dapo", "gspo"}:
        notes.append("TRL divides centered group rewards by sample_std + 1e-4; the canonical reference uses sample_std")
    if arm == "dapo":
        notes.append("TRL DAPO divides the masked token-loss sum by global active tokens; the canonical reference averages completion means")
    if arm == "drgrpo":
        notes.append("TRL DrGRPO divides by batch_size * max_completion_length; the canonical reference averages completion means")
    if arm == "gspo":
        notes.append("TRL 1.2.0 represents GSPO as loss_type='grpo' with importance_sampling_level='sequence'")
    if torch.any(actual_mask.sum(dim=1) == 0):
        notes.append("TRL computes group advantages before applying completion masks and GRPO averages zero-mask rows; the canonical reference excludes inactive rows")
    return tuple(notes)


def evaluate_fixture(fixture: ObjectiveFixture, arm: str) -> TRLDifferential:
    actual, config, provenance = trl_trace(
        arm=arm,
        rewards=fixture.rewards,
        logps=fixture.logps,
        old_logps=fixture.old_logps,
        mask=fixture.mask,
    )
    expected = objective_trace(
        arm=arm,
        rewards=fixture.rewards,
        logps=fixture.logps,
        old_logps=fixture.old_logps,
        mask=fixture.mask,
    )
    fields = tuple(
        _field_comparison(field, getattr(actual, field), getattr(expected, field))
        for field in ("advantages", "ratios", "mask", "surrogate", "loss", "gradient")
    )
    return TRLDifferential(
        fixture=fixture.name,
        arm=arm,
        actual=actual,
        expected=expected,
        config=config,
        provenance=provenance,
        fields=fields,
        formula_notes=_formula_notes(arm, actual.mask),
    )


def evaluate_intended_fixture(
    fixture: ObjectiveFixture,
    arm: str,
    *,
    selected_indices: Sequence[int] | None = None,
) -> TRLDifferential:
    actual, config, provenance = trl_intended_trace(
        arm=arm,
        rewards=fixture.rewards,
        logps=fixture.logps,
        old_logps=fixture.old_logps,
        mask=fixture.mask,
        selected_indices=selected_indices,
        aero_successes=fixture.aero_successes,
        aero_observations=fixture.aero_observations,
    )
    expected = objective_trace(
        arm=arm,
        rewards=fixture.rewards,
        logps=fixture.logps,
        old_logps=fixture.old_logps,
        mask=fixture.mask,
        selected_indices=selected_indices,
        aero_successes=fixture.aero_successes,
        aero_observations=fixture.aero_observations,
    )
    fields = tuple(
        _field_comparison(field, getattr(actual, field), getattr(expected, field))
        for field in ("advantages", "ratios", "mask", "surrogate", "loss", "gradient")
    )
    return TRLDifferential(
        fixture=fixture.name,
        arm=arm,
        actual=actual,
        expected=expected,
        config=config,
        provenance=provenance,
        fields=fields,
        formula_notes=(
            "canonical S1 advantages are injected before the pinned TRL loss kernel",
            "TRL grpo reduction is used to preserve canonical per-completion means",
        ),
        actual_semantics="intended_trl_s1_adapter",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(FIXTURES), default="base")
    parser.add_argument("--arm", choices=("grpo", "dapo", "gspo", "drgrpo"), default="grpo")
    args = parser.parse_args(argv)
    result = evaluate_fixture(FIXTURES[args.fixture], args.arm)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
