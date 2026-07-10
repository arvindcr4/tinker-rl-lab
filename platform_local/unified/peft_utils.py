"""Shared helpers for applying parameter-efficient fine-tuning methods.

The PEFT dependency is optional for the repository, so imports are intentionally
lazy.  This keeps configuration and launcher commands usable on machines that
only need to generate a training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence


SUPPORTED_PEFT_METHODS = (
    "lora",
    "prefix_tuning",
    "p_tuning",
    "prompt_tuning",
    "bitfit",
)


def _normalise_method(method: str) -> str:
    normalised = method.strip().lower().replace("-", "_")
    if normalised not in SUPPORTED_PEFT_METHODS:
        choices = ", ".join(SUPPORTED_PEFT_METHODS)
        raise ValueError(f"Unknown PEFT method {method!r}; expected one of: {choices}")
    return normalised


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")


def get_peft_config(
    method: str = "lora",
    task_type: Any = "CAUSAL_LM",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[Sequence[str]] = None,
    num_virtual_tokens: int = 32,
    encoder_hidden_size: int = 128,
) -> Any:
    """Build the PEFT configuration for ``method``.

    BitFit does not use a PEFT configuration; callers should use
    :func:`apply_bitfit` (or the higher-level :func:`apply_peft_method`).
    """

    method = _normalise_method(method)
    if method == "bitfit":
        return None

    try:
        from peft import (
            LoraConfig,
            PrefixTuningConfig,
            PromptEncoderConfig,
            PromptTuningConfig,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "PEFT is required to configure adapter tuning; install the 'trl' extra"
        ) from exc

    if method == "lora":
        _require_positive("lora_rank", lora_rank)
        _require_positive("lora_alpha", lora_alpha)
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in the range [0, 1)")
        target_modules = list(lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"])
        return LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=task_type,
            bias="none",
        )

    _require_positive("num_virtual_tokens", num_virtual_tokens)
    if method == "prefix_tuning":
        return PrefixTuningConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
        )
    if method == "p_tuning":
        _require_positive("encoder_hidden_size", encoder_hidden_size)
        return PromptEncoderConfig(
            task_type=task_type,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
        )
    return PromptTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
    )


def apply_bitfit(model: Any) -> Any:
    """Freeze the model except for parameters whose final name is ``bias``."""

    trainable_biases = 0
    for name, parameter in model.named_parameters():
        is_bias = name.rsplit(".", 1)[-1] == "bias"
        parameter.requires_grad = is_bias
        trainable_biases += int(is_bias)

    if trainable_biases == 0:
        raise ValueError("BitFit selected, but this model exposes no bias parameters")
    return model


def apply_peft_method(
    model: Any,
    *,
    method: str = "lora",
    quantized: bool = False,
    task_type: Any = "CAUSAL_LM",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[Sequence[str]] = None,
    num_virtual_tokens: int = 32,
    encoder_hidden_size: int = 128,
) -> Any:
    """Apply one supported tuning method to an already loaded model.

    Quantized models are prepared for k-bit training only when the caller
    actually loaded them in 4-bit or 8-bit mode.
    """

    method = _normalise_method(method)
    if method == "bitfit":
        return apply_bitfit(model)

    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "PEFT is required to apply adapter tuning; install the 'trl' extra"
        ) from exc

    if quantized:
        model = prepare_model_for_kbit_training(model)

    config = get_peft_config(
        method=method,
        task_type=task_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
    )
    return get_peft_model(model, config)


def get_trainable_state_dict(model: Any) -> Dict[str, Any]:
    """Return a detached CPU state dict containing only trainable parameters."""

    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_bitfit_checkpoint(
    model: Any,
    output_path: str | Path,
    *,
    base_model_name: Optional[str] = None,
) -> Path:
    """Save a compact, explicit BitFit checkpoint containing bias tensors only."""

    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = get_trainable_state_dict(model)
    if not state_dict:
        raise ValueError("Cannot save BitFit checkpoint: no trainable parameters found")
    torch.save(
        {
            "format": "bitfit-v1",
            "base_model_name": base_model_name,
            "state_dict": state_dict,
        },
        output_path,
    )
    return output_path
