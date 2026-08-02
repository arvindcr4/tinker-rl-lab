"""Compute-backend registry for the unified launcher.

Six backends, each delegating to existing provisioning/drivers:
local (in-process), modal (H100), colab (A100 notebooks), vast (rented A100),
gcp (A100 Spot preflight), hfspaces (results demo + fetch).
"""
from __future__ import annotations

from platform_local.unified.backends.base import Backend, LaunchPlan

__all__ = ["Backend", "LaunchPlan", "BACKENDS", "get_backend", "BACKEND_NAMES"]

BACKEND_NAMES = ("local", "modal", "colab", "vast", "gcp", "hfspaces")
_BACKENDS: dict[str, Backend] | None = None


def _build() -> dict[str, Backend]:
    from platform_local.unified.backends import (
        colab as _colab,
        gcp as _gcp,
        hfspaces as _hf,
        local as _local,
        modal as _modal,
        vast as _vast,
    )

    instances = (
        _local.LocalBackend(),
        _modal.ModalBackend(),
        _colab.ColabBackend(),
        _vast.VastBackend(),
        _gcp.GCPBackend(),
        _hf.HFSpacesBackend(),
    )
    return {b.name: b for b in instances}


def BACKENDS() -> dict[str, Backend]:  # noqa: N802 - registry accessor
    """Lazily-built backend registry."""
    global _BACKENDS
    if _BACKENDS is None:
        _BACKENDS = _build()
    return _BACKENDS


def get_backend(name: str) -> Backend:
    registry = BACKENDS()
    if name not in registry:
        raise KeyError(f"Unknown backend {name!r}; available: {sorted(registry)}")
    return registry[name]
