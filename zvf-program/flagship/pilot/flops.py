from __future__ import annotations

import math
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


PROFILED_STEPS = (1, 20, 40, 60, 80, 100)
REQUIRED_TRAINING_PHASES = (
    "policy_forward",
    "optimizer_backward",
    "diagnostic_backward",
)


class FlopAccountingError(RuntimeError):
    """Profiler coverage is absent or cannot support the frozen FLOP ledger."""


class TorchPhaseProfiler:
    def __init__(self, torch_module: Any, *, enabled: bool) -> None:
        self.torch = torch_module
        self.enabled = enabled
        self.phase_flops: dict[str, float] = {}

    def __call__(self, phase: str) -> AbstractContextManager[Any]:
        if not self.enabled or phase == "optimizer_step":
            return nullcontext()
        return self._profile(phase)

    @contextmanager
    def _profile(self, phase: str) -> Iterator[None]:
        activities = [self.torch.profiler.ProfilerActivity.CPU]
        if self.torch.cuda.is_available():
            activities.append(self.torch.profiler.ProfilerActivity.CUDA)
        with self.torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as profile:
            yield
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
        flops = float(sum(float(event.flops or 0.0) for event in profile.key_averages()))
        if not math.isfinite(flops) or flops <= 0:
            raise FlopAccountingError(f"profiler reported no positive FLOPs for {phase}")
        self.phase_flops[phase] = self.phase_flops.get(phase, 0.0) + flops

    def require_training_coverage(self) -> None:
        missing = [phase for phase in REQUIRED_TRAINING_PHASES if self.phase_flops.get(phase, 0) <= 0]
        if self.enabled and missing:
            raise FlopAccountingError(f"missing profiler coverage for: {', '.join(missing)}")


@dataclass(slots=True)
class TrainingFlopLedger:
    profiled_steps: list[int] = field(default_factory=list)
    profiled_phase_flops: dict[str, float] = field(default_factory=dict)
    profiled_active_tokens: int = 0
    profiled_padded_tokens: int = 0
    total_active_tokens: int = 0
    total_padded_tokens: int = 0

    def add_step(
        self,
        *,
        step: int,
        active_tokens: int,
        padded_tokens: int,
        phase_flops: Mapping[str, float] | None,
    ) -> None:
        if step <= 0 or active_tokens <= 0 or padded_tokens < active_tokens:
            raise FlopAccountingError("invalid step token accounting")
        self.total_active_tokens += int(active_tokens)
        self.total_padded_tokens += int(padded_tokens)
        if phase_flops is None:
            return
        if step not in PROFILED_STEPS:
            raise FlopAccountingError(f"unexpected profiled training step: {step}")
        missing = [phase for phase in REQUIRED_TRAINING_PHASES if phase_flops.get(phase, 0) <= 0]
        if missing:
            raise FlopAccountingError(f"profiled step {step} is missing phases: {', '.join(missing)}")
        self.profiled_steps.append(step)
        self.profiled_active_tokens += int(active_tokens)
        self.profiled_padded_tokens += int(padded_tokens)
        for phase, value in phase_flops.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise FlopAccountingError(f"invalid FLOP value for {phase}")
            self.profiled_phase_flops[phase] = self.profiled_phase_flops.get(phase, 0.0) + float(
                value
            )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrainingFlopLedger":
        return cls(
            profiled_steps=[int(step) for step in record.get("profiled_steps", ())],
            profiled_phase_flops={
                str(phase): float(value)
                for phase, value in record.get("profiled_phase_flops", {}).items()
            },
            profiled_active_tokens=int(record.get("profiled_active_tokens", 0)),
            profiled_padded_tokens=int(record.get("profiled_padded_tokens", 0)),
            total_active_tokens=int(record.get("total_active_tokens", 0)),
            total_padded_tokens=int(record.get("total_padded_tokens", 0)),
        )

    def record(self, *, require_complete: bool) -> dict[str, Any]:
        if require_complete and tuple(self.profiled_steps) != PROFILED_STEPS:
            raise FlopAccountingError(
                f"profiled steps mismatch: expected {PROFILED_STEPS}, got {tuple(self.profiled_steps)}"
            )
        if not self.profiled_steps:
            raise FlopAccountingError("FLOP ledger contains no profiled training steps")
        if self.profiled_padded_tokens <= 0 or self.total_padded_tokens <= 0:
            raise FlopAccountingError("FLOP extrapolation has no padded-token denominator")
        scale = self.total_padded_tokens / self.profiled_padded_tokens
        extrapolated = {
            phase: value * scale for phase, value in self.profiled_phase_flops.items()
        }
        for phase in REQUIRED_TRAINING_PHASES:
            if extrapolated.get(phase, 0) <= 0:
                raise FlopAccountingError(f"extrapolated {phase} FLOPs are missing")
        return {
            "profiled_steps": list(self.profiled_steps),
            "profiled_phase_flops": dict(sorted(self.profiled_phase_flops.items())),
            "profiled_active_tokens": self.profiled_active_tokens,
            "profiled_padded_tokens": self.profiled_padded_tokens,
            "total_active_tokens": self.total_active_tokens,
            "total_padded_tokens": self.total_padded_tokens,
            "extrapolation_scale": scale,
            "policy_forward_flops": extrapolated["policy_forward"],
            "optimizer_backward_flops": extrapolated["optimizer_backward"],
            "diagnostic_backward_flops": extrapolated["diagnostic_backward"],
        }

    def final_record(self) -> dict[str, Any]:
        return self.record(require_complete=True)
