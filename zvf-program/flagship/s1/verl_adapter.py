"""Pinned, CPU-only verl 0.3.0.post1 differential harness for S1 fixtures."""

from __future__ import annotations

import hashlib
import inspect
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from .fixtures import ObjectiveFixture
from .reference import ATOL, RTOL, Trace, canonical_advantages, objective_trace

VERL_VERSION = "0.3.0.post1"
TORCH_VERSION = "2.4.0"
TRANSFORMERS_VERSION = "4.45.2"
CORE_ALGOS_SHA256 = "74d680b9186690a7788157679725bc351a5436b6419ff8c7a2b570bc4cbf13ee"
TORCH_FUNCTIONAL_SHA256 = "e2dd02383718c856053f1e70f74311b11b92ece40a034f97645c6f7db7310db6"
METADATA_SHA256 = "4d4c5ca3b2bfe5f37487292ca8407e6963ac6555bfb861c2ac78d4b39f3e0d96"

ConformanceVerdict = Literal[
    "PASS",
    "NUMERICAL_VARIATION",
    "MATERIAL_DIFFERENCE",
    "NOT_TESTED",
]


class VerlPinError(RuntimeError):
    """The imported stack does not match the frozen external verl runtime."""


class VerlUnsupportedObjective(ValueError):
    """The requested canonical objective has no native verl 0.3.0 mapping."""


@dataclass(frozen=True)
class VerlArmConfig:
    arm: str
    cliprange: float
    advantage_epsilon: float
    reduction: str
    group_id_type: str


@dataclass(frozen=True)
class VerlProvenance:
    verl_version: str
    torch_version: str
    transformers_version: str
    python_version: str
    platform: str
    device: str
    cuda_available: bool
    core_algos_source: str
    core_algos_sha256: str
    torch_functional_source: str
    torch_functional_sha256: str
    metadata_source: str
    metadata_sha256: str
    exercised_advantage_api: str
    exercised_loss_api: str


@dataclass(frozen=True)
class FieldComparison:
    field: str
    agrees: bool
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class VerlDifferential:
    fixture: str
    arm: str
    actual: Trace | None
    expected: Trace | None
    config: VerlArmConfig | None
    provenance: VerlProvenance
    fields: tuple[FieldComparison, ...]
    formula_notes: tuple[str, ...]
    not_tested_reason: str | None = None
    actual_semantics: str = "native_verl_0.3.0.post1"

    @property
    def conforms(self) -> bool:
        return self.not_tested_reason is None and all(field.agrees for field in self.fields)

    @property
    def verdict(self) -> ConformanceVerdict:
        if self.not_tested_reason is not None:
            return "NOT_TESTED"
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
            "actual_loss": None if self.actual is None else self.actual.loss.item(),
            "expected_loss": None if self.expected is None else self.expected.loss.item(),
            "config": None if self.config is None else asdict(self.config),
            "provenance": asdict(self.provenance),
            "fields": [asdict(field) for field in self.fields],
            "formula_notes": list(self.formula_notes),
            "not_tested_reason": self.not_tested_reason,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_pinned_runtime() -> tuple[Any, Any, VerlProvenance]:
    """Load and verify external verl, rejecting the repository-local wrapper."""
    try:
        import transformers
        import verl.trainer.ppo.core_algos as core_algos
        import verl.utils.torch_functional as verl_functional
    except ImportError as error:
        raise VerlPinError(
            "external verl is unavailable; run from outside the repository with "
            "`PYTHONPATH=<repo>/zvf-program/flagship` and the frozen Python 3.11 environment"
        ) from error

    versions = {
        "verl": version("verl"),
        "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__,
    }
    expected = {
        "verl": VERL_VERSION,
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }
    if versions != expected:
        raise VerlPinError(f"expected versions {expected}, found {versions}")

    core_source = Path(inspect.getsourcefile(core_algos) or "").resolve()
    functional_source = Path(inspect.getsourcefile(verl_functional) or "").resolve()
    if _repo_root() in core_source.parents or _repo_root() in functional_source.parents:
        raise VerlPinError("import resolved the repository-local verl wrapper, not the external distribution")

    dist = distribution("verl")
    metadata_source = Path(dist._path) / "METADATA"  # type: ignore[attr-defined]
    hashes = {
        "core": _sha256(core_source),
        "functional": _sha256(functional_source),
        "metadata": _sha256(metadata_source),
    }
    expected_hashes = {
        "core": CORE_ALGOS_SHA256,
        "functional": TORCH_FUNCTIONAL_SHA256,
        "metadata": METADATA_SHA256,
    }
    if hashes != expected_hashes:
        raise VerlPinError(f"unexpected external verl source hashes: {hashes}")

    advantage_source = inspect.getsource(core_algos.compute_grpo_outcome_advantage)
    loss_source = inspect.getsource(core_algos.compute_policy_loss)
    required_fragments = (
        "id2score[index[i]].append(scores[i])",
        "id2std[index[i]] + epsilon",
        "pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)",
    )
    joined_source = advantage_source + loss_source
    if any(fragment not in joined_source for fragment in required_fragments):
        raise VerlPinError("verl objective source shape differs from the frozen adapter")

    provenance = VerlProvenance(
        verl_version=versions["verl"],
        torch_version=torch.__version__,
        transformers_version=versions["transformers"],
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        device="cpu",
        cuda_available=torch.cuda.is_available(),
        core_algos_source=str(core_source),
        core_algos_sha256=hashes["core"],
        torch_functional_source=str(functional_source),
        torch_functional_sha256=hashes["functional"],
        metadata_source=str(metadata_source),
        metadata_sha256=hashes["metadata"],
        exercised_advantage_api="verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
        exercised_loss_api="verl.trainer.ppo.core_algos.compute_policy_loss",
    )
    return core_algos, verl_functional, provenance


def _validate_fixture(
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if rewards.ndim != 2 or logps.ndim != 2:
        raise ValueError("rewards and log-probabilities must be rank two")
    if logps.shape != old_logps.shape or logps.shape != mask.shape:
        raise ValueError("log-probability and mask shapes must match")
    if rewards.numel() != logps.shape[0]:
        raise ValueError("flattened rewards must have one entry per completion")
    if not torch.any(mask.sum(dim=1) > 0):
        raise ValueError("fixture must retain one active completion")


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


def verl_trace(
    *,
    arm: str,
    rewards: torch.Tensor | Sequence[Sequence[float]],
    logps: torch.Tensor | Sequence[Sequence[float]],
    old_logps: torch.Tensor | Sequence[Sequence[float]],
    mask: torch.Tensor | Sequence[Sequence[float]],
) -> tuple[Trace, VerlArmConfig, VerlProvenance]:
    """Evaluate a GRPO fixture through the pinned external verl CPU kernels."""
    core_algos, _, provenance = load_pinned_runtime()
    if arm != "grpo":
        raise VerlUnsupportedObjective(f"no native verl 0.3.0.post1 mapping for {arm}")
    if provenance.cuda_available and torch.get_default_device().type != "cpu":
        raise VerlPinError("S1 fixture harness must run on CPU")

    rewards_t = torch.as_tensor(rewards, dtype=torch.float64, device="cpu")
    logps_t = torch.as_tensor(logps, dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    old_t = torch.as_tensor(old_logps, dtype=torch.float64, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.float64, device="cpu")
    _validate_fixture(rewards_t, logps_t, old_t, mask_t)

    group_count, group_size = rewards_t.shape
    group_ids = [f"group-{group}" for group in range(group_count) for _ in range(group_size)]
    token_rewards = torch.zeros_like(logps_t)
    token_rewards[:, 0] = rewards_t.flatten()
    advantage_tokens, _ = core_algos.compute_grpo_outcome_advantage(
        token_rewards,
        mask_t,
        group_ids,
        epsilon=1e-6,
    )
    counts = mask_t.sum(dim=1).clamp_min(1.0)
    advantages = (advantage_tokens * mask_t).sum(dim=1) / counts
    loss, _, _ = core_algos.compute_policy_loss(old_t, logps_t, advantage_tokens, mask_t, 0.2)

    ratios = (logps_t - old_t).exp()
    clipped = ratios.clamp(0.8, 1.2)
    surrogate = -torch.minimum(
        ratios * advantages[:, None],
        clipped * advantages[:, None],
    )
    gradient = torch.autograd.grad(loss, logps_t)[0].flatten()
    trace = Trace(
        arm=arm,
        advantages=advantages.detach(),
        ratios=ratios.detach(),
        mask=mask_t.detach(),
        surrogate=surrogate.detach(),
        loss=loss.detach(),
        gradient=gradient.detach(),
        selected_indices=tuple(range(logps_t.shape[0])),
    )
    config = VerlArmConfig(
        arm=arm,
        cliprange=0.2,
        advantage_epsilon=1e-6,
        reduction="global_masked_token_mean",
        group_id_type="stable_python_string",
    )
    return trace, config, provenance


def verl_intended_trace(
    *,
    arm: str,
    rewards: torch.Tensor | Sequence[Sequence[float]],
    logps: torch.Tensor | Sequence[Sequence[float]],
    old_logps: torch.Tensor | Sequence[Sequence[float]],
    mask: torch.Tensor | Sequence[Sequence[float]],
    selected_indices: Sequence[int] | None = None,
    aero_successes: Sequence[int] | None = None,
    aero_observations: Sequence[int] | None = None,
) -> tuple[Trace, VerlArmConfig, VerlProvenance]:
    """Exercise the exact S1 treatment through pinned verl tensor helpers."""
    _, verl_functional, provenance = load_pinned_runtime()
    if arm not in {"grpo", "dapo", "gspo", "drgrpo", "aero"}:
        raise ValueError(f"unknown objective arm: {arm}")

    rewards_t = torch.as_tensor(rewards, dtype=torch.float64, device="cpu")
    logps_t = torch.as_tensor(logps, dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    old_t = torch.as_tensor(old_logps, dtype=torch.float64, device="cpu")
    mask_t = torch.as_tensor(mask, dtype=torch.float64, device="cpu")
    _validate_fixture(rewards_t, logps_t, old_t, mask_t)

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
    if arm == "gspo":
        log_weights = (delta * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp_min(1.0)
        ratios = log_weights.exp()[:, None].expand_as(logps_t)
    else:
        ratios = delta.exp()
    high = 0.28 if arm == "dapo" else 0.20
    clipped = ratios.clamp(0.80, 1.0 + high)
    surrogate = -torch.minimum(
        ratios * advantages[:, None],
        clipped * advantages[:, None],
    )
    per_completion = verl_functional.masked_mean(
        surrogate[active],
        mask_t[active],
        axis=-1,
    )
    loss = per_completion.mean()
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
    config = VerlArmConfig(
        arm=arm,
        cliprange=high,
        advantage_epsilon=0.0,
        reduction="per_completion_masked_mean_then_batch_mean",
        group_id_type="canonical_group_matrix",
    )
    return trace, config, provenance


def evaluate_fixture(fixture: ObjectiveFixture, arm: str) -> VerlDifferential:
    _, _, provenance = load_pinned_runtime()
    if arm != "grpo":
        reason = f"native verl 0.3.0.post1 exposes no {arm} objective kernel"
        return VerlDifferential(
            fixture=fixture.name,
            arm=arm,
            actual=None,
            expected=None,
            config=None,
            provenance=provenance,
            fields=(),
            formula_notes=(),
            not_tested_reason=reason,
        )

    actual, config, provenance = verl_trace(
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
    notes = (
        "native verl divides centered group rewards by sample_std + 1e-6; canonical uses sample_std",
        "native verl reduces the PPO surrogate with one global masked-token mean; canonical averages completion means",
        "stable Python string group IDs are required because tensor scalar IDs fail native dictionary lookup",
    )
    return VerlDifferential(
        fixture=fixture.name,
        arm=arm,
        actual=actual,
        expected=expected,
        config=config,
        provenance=provenance,
        fields=fields,
        formula_notes=notes,
    )


def evaluate_intended_fixture(
    fixture: ObjectiveFixture,
    arm: str,
    *,
    selected_indices: Sequence[int] | None = None,
) -> VerlDifferential:
    actual, config, provenance = verl_intended_trace(
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
    return VerlDifferential(
        fixture=fixture.name,
        arm=arm,
        actual=actual,
        expected=expected,
        config=config,
        provenance=provenance,
        fields=fields,
        formula_notes=(
            "canonical S1 advantages are injected before the pinned verl tensor reduction",
            "verl.utils.torch_functional.masked_mean is applied per completion before the batch mean",
        ),
        actual_semantics="intended_verl_s1_adapter",
    )
