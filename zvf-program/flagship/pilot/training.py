from __future__ import annotations

import math
import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .objective import Condition, ObjectiveContractError, condition_loss


class TrainingContractError(RuntimeError):
    """A replay update cannot produce a valid, reproducible training receipt."""


PhaseContextFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    group_fingerprint: str
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    active_rows: torch.Tensor
    old_logps: torch.Tensor


@dataclass(frozen=True, slots=True)
class StepReceipt:
    step: int
    condition: Condition
    group_fingerprint: str
    selected_loss: float
    intended_loss: float
    native_loss: float
    gradient_relation: str
    gradient_cosine: float | None
    gradient_relative_l2: float | None
    intended_gradient_norm: float
    native_gradient_norm: float
    selected_gradient_norm: float
    selected_vs_intended_relation: str
    selected_vs_intended_cosine: float | None
    selected_vs_intended_relative_l2: float | None
    optimizer_update: str
    active_rows: int
    active_tokens: int
    optimizer_learning_rate: float

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _validate_batch(batch: ReplayBatch) -> None:
    tensors = (
        batch.prompt_ids,
        batch.prompt_mask,
        batch.completion_ids,
        batch.completion_mask,
        batch.old_logps,
    )
    if any(tensor.ndim != 2 or tensor.shape[0] != 8 for tensor in tensors):
        raise TrainingContractError("replay tensors must have eight rows")
    if batch.prompt_ids.shape != batch.prompt_mask.shape:
        raise TrainingContractError("prompt IDs and masks do not align")
    if not (batch.completion_ids.shape == batch.completion_mask.shape == batch.old_logps.shape):
        raise TrainingContractError("completion IDs, masks, and old log probabilities do not align")
    if batch.prompt_ids.shape[1] <= 0 or batch.completion_ids.shape[1] <= 0:
        raise TrainingContractError("prompt and completion widths must be positive")
    if batch.rewards.shape != (8,) or batch.active_rows.shape != (8,):
        raise TrainingContractError("rewards and active-row masks must have eight entries")
    if batch.active_rows.dtype != torch.bool:
        raise TrainingContractError("active_rows must be boolean")
    if not batch.group_fingerprint:
        raise TrainingContractError("replay group fingerprint is missing")


def completion_logps(model: torch.nn.Module, batch: ReplayBatch) -> torch.Tensor:
    """Extract log p(completion token | prompt and previous completion tokens)."""
    _validate_batch(batch)
    device = next(model.parameters()).device
    prompt_ids = batch.prompt_ids.to(device=device, dtype=torch.long)
    completion_ids = batch.completion_ids.to(device=device, dtype=torch.long)
    prompt_mask = batch.prompt_mask.to(device=device, dtype=torch.long)
    completion_attention = (batch.completion_mask > 0).to(device=device, dtype=torch.long)
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    attention_mask = torch.cat((prompt_mask, completion_attention), dim=1)
    # The frozen protocol runs with deterministic algorithms enabled; on
    # torch 2.7.1 + CUDA the fused SDPA kernels reject the padded 4D causal
    # mask in this configuration (`value cannot be converted to type int64_t
    # without overflow`). The math backend is deterministic and mask-agnostic,
    # so the training forward is pinned to it.
    with sdpa_kernel([SDPBackend.MATH]):
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    prompt_width = prompt_ids.shape[1]
    completion_logits = logits[:, prompt_width - 1 : -1, :]
    if completion_logits.shape[:2] != completion_ids.shape:
        raise TrainingContractError("model logits do not align with completion tokens")
    log_probs = torch.log_softmax(completion_logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)


def _trainable_parameters(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise TrainingContractError("model has no trainable parameters")
    return parameters


def _gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    # Gradient checkpointing recomputes the policy forward after
    # completion_logps() has exited its SDPA context. Keep the deterministic
    # math backend active for that recomputation as well; otherwise torch
    # 2.7.1 can route maskless Qwen3 GQA through a fused kernel and overflow
    # before the first optimizer step.
    with sdpa_kernel([SDPBackend.MATH]):
        values = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    return tuple(
        torch.zeros_like(parameter) if value is None else value
        for parameter, value in zip(parameters, values, strict=True)
    )


def _flatten(gradients: Iterable[torch.Tensor]) -> torch.Tensor:
    # Accumulate diagnostics in float64 on CPU. The optimizer still receives
    # the original gradient tensors; this conversion is receipt-only and
    # prevents low-precision dot/norm roundoff from emitting |cosine| > 1.
    return torch.cat(tuple(gradient.detach().double().reshape(-1).cpu() for gradient in gradients))


def _finite_norm(vector: torch.Tensor, *, label: str) -> float:
    norm = float(torch.linalg.vector_norm(vector))
    if not math.isfinite(norm) or norm < 0.0:
        raise TrainingContractError(f"{label} gradient norm is negative or non-finite")
    return norm


@dataclass(frozen=True, slots=True)
class _GradientComparison:
    relation: str
    cosine: float | None
    relative_l2: float | None


def _compare_gradients(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_norm: float,
    right_norm: float,
    left_zero_relation: str,
    right_zero_relation: str,
    label: str,
) -> _GradientComparison:
    """Classify a gradient pair without inventing angles for zero vectors."""
    if left_norm == 0.0 and right_norm == 0.0:
        return _GradientComparison("joint_zero", None, None)
    if left_norm == 0.0:
        return _GradientComparison(left_zero_relation, None, None)
    if right_norm == 0.0:
        return _GradientComparison(right_zero_relation, None, None)
    if torch.equal(left, right):
        # Equality is exact at the stored-vector level. Computing dot/(norm²)
        # with separate parallel reductions can still overshoot 1 by several
        # ulps on a large vector, but the mathematical diagnostics here are
        # exactly cos=1 and relative-L2=0.
        return _GradientComparison("nonzero", 1.0, 0.0)
    return _GradientComparison(
        relation="nonzero",
        cosine=_bounded_cosine(
            left,
            right,
            left_norm=left_norm,
            right_norm=right_norm,
            label=label,
        ),
        relative_l2=_relative_l2(
            left,
            right,
            reference_norm=left_norm,
            label=label,
        ),
    )


def _bounded_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_norm: float,
    right_norm: float,
    label: str,
) -> float:
    value = float(torch.dot(left, right) / (left_norm * right_norm))
    tolerance = 1e-12
    if not math.isfinite(value) or value < -1.0 - tolerance or value > 1.0 + tolerance:
        raise TrainingContractError(f"{label} cosine is outside [-1, 1]: {value}")
    return min(1.0, max(-1.0, value))


def _relative_l2(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    reference_norm: float,
    label: str,
) -> float:
    value = float(torch.linalg.vector_norm(left - right) / reference_norm)
    if not math.isfinite(value) or value < 0.0:
        raise TrainingContractError(f"{label} relative L2 is negative or non-finite: {value}")
    return value


def _assign_gradients(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if not torch.isfinite(gradient).all():
            raise TrainingContractError("selected gradient contains non-finite values")
        parameter.grad = gradient.detach().clone()


def run_replay_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batch: ReplayBatch,
    condition: Condition,
    step: int,
    max_grad_norm: float = 1.0,
    phase_context: PhaseContextFactory | None = None,
) -> StepReceipt:
    if step <= 0:
        raise TrainingContractError("training step must be positive")
    if max_grad_norm <= 0:
        raise TrainingContractError("max_grad_norm must be positive")
    parameters = _trainable_parameters(model)
    optimizer.zero_grad(set_to_none=True)
    phase = phase_context or (lambda _: nullcontext())
    with phase("policy_forward"):
        logps = completion_logps(model, batch)
        device = logps.device
        objective_inputs = {
            "rewards": batch.rewards.to(device),
            "logps": logps,
            "old_logps": batch.old_logps.to(device),
            "completion_mask": batch.completion_mask.to(device),
            "active_rows": batch.active_rows.to(device),
        }
        try:
            intended_loss, _ = condition_loss(condition="intended_full", **objective_inputs)
            native_loss, _ = condition_loss(condition="native_trl", **objective_inputs)
            selected_loss, _ = condition_loss(condition=condition, **objective_inputs)
        except ObjectiveContractError as exc:
            raise TrainingContractError(str(exc)) from exc

    with phase("optimizer_backward"):
        selected_gradients = _gradients(selected_loss, parameters, retain_graph=True)
    with phase("diagnostic_backward"):
        if condition == "intended_full":
            intended_gradients = selected_gradients
            native_gradients = _gradients(native_loss, parameters, retain_graph=False)
        elif condition == "native_trl":
            native_gradients = selected_gradients
            intended_gradients = _gradients(intended_loss, parameters, retain_graph=False)
        else:
            intended_gradients = _gradients(intended_loss, parameters, retain_graph=True)
            native_gradients = _gradients(native_loss, parameters, retain_graph=False)

    intended_vector = _flatten(intended_gradients)
    native_vector = _flatten(native_gradients)
    selected_vector = _flatten(selected_gradients)
    intended_norm = _finite_norm(intended_vector, label="intended")
    native_norm = _finite_norm(native_vector, label="native")
    selected_norm = _finite_norm(selected_vector, label="selected")
    intended_native = _compare_gradients(
        intended_vector,
        native_vector,
        left_norm=intended_norm,
        right_norm=native_norm,
        left_zero_relation="intended_zero",
        right_zero_relation="native_zero",
        label="intended-vs-native",
    )
    selected_intended = _compare_gradients(
        selected_vector,
        intended_vector,
        left_norm=selected_norm,
        right_norm=intended_norm,
        left_zero_relation="selected_zero",
        right_zero_relation="intended_zero",
        label="selected-vs-intended",
    )

    with phase("optimizer_step"):
        if selected_norm == 0.0:
            optimizer_update = "no_op_zero_gradient"
        else:
            _assign_gradients(parameters, selected_gradients)
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
            optimizer_update = "applied"
        if scheduler is not None:
            if optimizer_update == "no_op_zero_gradient":
                # PyTorch warns when the first scheduled step intentionally has
                # no optimizer.step(). Advancing last_epoch is nevertheless the
                # frozen contract, so suppress only that expected warning.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Detected call of `lr_scheduler.step\(\)` before",
                        category=UserWarning,
                    )
                    scheduler.step()
            else:
                scheduler.step()
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(learning_rates) != 1:
        raise TrainingContractError("optimizer parameter groups have inconsistent learning rates")

    return StepReceipt(
        step=step,
        condition=condition,
        group_fingerprint=batch.group_fingerprint,
        selected_loss=float(selected_loss.detach()),
        intended_loss=float(intended_loss.detach()),
        native_loss=float(native_loss.detach()),
        gradient_relation=intended_native.relation,
        gradient_cosine=intended_native.cosine,
        gradient_relative_l2=intended_native.relative_l2,
        intended_gradient_norm=intended_norm,
        native_gradient_norm=native_norm,
        selected_gradient_norm=selected_norm,
        selected_vs_intended_relation=selected_intended.relation,
        selected_vs_intended_cosine=selected_intended.cosine,
        selected_vs_intended_relative_l2=selected_intended.relative_l2,
        optimizer_update=optimizer_update,
        active_rows=int(batch.active_rows.sum()),
        active_tokens=int(batch.completion_mask.sum()),
        optimizer_learning_rate=learning_rates.pop(),
    )
