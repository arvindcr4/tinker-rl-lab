"""Shared inference protocol v2: exact sampling semantics and probability alignment."""

import math
from fastapi import HTTPException
from tinker.types import SamplingParams

PROTOCOL_VERSION = "tinker-http-v2"


def sampling_params(values):
    values = dict(values)
    values.pop("n", None)
    if "max_new_tokens" in values:
        if "max_tokens" in values:
            raise HTTPException(400, "Use only one token limit")
        values["max_tokens"] = values.pop("max_new_tokens")
    supported = {"max_tokens", "temperature", "stop", "top_p", "top_k", "seed"}
    unknown = set(values) - supported
    if unknown:
        raise HTTPException(400, f"Unsupported sampling settings: {sorted(unknown)}")
    values.setdefault("max_tokens", 256)
    values.setdefault("temperature", 1.0)
    values.setdefault("stop", None)
    if not isinstance(values["max_tokens"], int) or values["max_tokens"] <= 0:
        raise HTTPException(400, "max_tokens must be positive")
    if (
        not isinstance(values["temperature"], (int, float))
        or not math.isfinite(values["temperature"])
        or values["temperature"] < 0
    ):
        raise HTTPException(400, "temperature must be finite and nonnegative")
    if "top_p" in values and not 0 < values["top_p"] <= 1:
        raise HTTPException(400, "top_p must be in (0, 1]")
    if isinstance(values["stop"], str):
        values["stop"] = [values["stop"]]
    fields = getattr(SamplingParams, "model_fields", None)
    if fields:
        unsupported = {k for k, v in values.items() if v is not None and k not in fields}
        if unsupported:
            raise HTTPException(400, f"Installed backend does not support {sorted(unsupported)}")
        values = {k: v for k, v in values.items() if k in fields}
    try:
        return SamplingParams(**values)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid sampling settings: {exc}") from exc


def finish_reason(sequence):
    reason = getattr(sequence, "stop_reason", None)
    reason = getattr(reason, "value", reason)
    if reason not in {"stop", "length"}:
        raise HTTPException(502, "Backend returned an unknown termination reason")
    return reason


def probabilities(tokens, values, *, allow_unavailable=False):
    if values is None or len(values) != len(tokens):
        raise HTTPException(502, "Backend probability array does not match token count")
    for value in values:
        if value is None and allow_unavailable:
            continue
        if value is None or not math.isfinite(value) or value > 0:
            raise HTTPException(502, "Backend returned an unavailable or invalid probability")
    return list(values)
