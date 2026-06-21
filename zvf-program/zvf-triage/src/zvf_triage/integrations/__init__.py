"""Native framework adapters for ZVF triage.

The package ships **one working reference adapter** -- the TRL / HuggingFace
``transformers.TrainerCallback`` returned by
:meth:`zvf_triage.callback.ZVFCallback.as_trl_callback` -- and a set of
**honest stub adapters** for the other RL-post-training frameworks the ZVF
Program deck targets for upstreaming:

* :class:`~zvf_triage.integrations.verl.VERLZVFAdapter`        (verl)
* :class:`~zvf_triage.integrations.openrlhf.OpenRLHFZVFAdapter` (OpenRLHF)
* :class:`~zvf_triage.integrations.nemo_rl.NeMoRLZVFAdapter`    (NeMo-RL)

Each stub defines the adapter class signature, documents exactly which native
rollout-reward structure it will read, and raises :class:`NotImplementedError`
with an actionable message. They are scaffolds, **not** fake implementations:
nothing here silently no-ops or fabricates a decision. The generic
``ZVFCallback.on_step(rewards, group_ids)`` path already works with all of these
frameworks today -- the adapters exist to read each framework's *native* reward
layout instead of the generic log-dict shim. See ``# TODO(v0.2)`` markers.

These modules import lazily: importing :mod:`zvf_triage.integrations` never
imports verl / OpenRLHF / NeMo / torch.
"""

from __future__ import annotations

from .base import BaseZVFAdapter

__all__ = [
    "BaseZVFAdapter",
    "VERLZVFAdapter",
    "OpenRLHFZVFAdapter",
    "NeMoRLZVFAdapter",
]


def __getattr__(name: str) -> object:
    """Lazily expose the framework adapter classes without importing them eagerly.

    Keeps ``import zvf_triage.integrations`` dependency-free; the heavy framework
    imports happen only when an adapter class is actually referenced.
    """
    if name == "VERLZVFAdapter":
        from .verl import VERLZVFAdapter

        return VERLZVFAdapter
    if name == "OpenRLHFZVFAdapter":
        from .openrlhf import OpenRLHFZVFAdapter

        return OpenRLHFZVFAdapter
    if name == "NeMoRLZVFAdapter":
        from .nemo_rl import NeMoRLZVFAdapter

        return NeMoRLZVFAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
