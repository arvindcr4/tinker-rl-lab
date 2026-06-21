"""Shared base for native framework adapters.

A native adapter's only job is to translate a framework's per-step rollout
output into the generic ``(rewards, group_ids)`` pair that
:class:`zvf_triage.callback.ZVFCallback` already understands, then forward the
resulting :class:`~zvf_triage.controller.StepDecision`'s ``should_stop`` /
adaptive-``G`` signals back into the framework's trainer-control object.

The translation step (``_extract_rollout_rewards``) is framework-specific and is
left abstract here; the forwarding step is identical across frameworks and is
implemented once in :meth:`BaseZVFAdapter.process_step`.

This base class is fully implemented and import-light (numpy + the in-package
callback only). The subclasses in this package are the parts that are still
stubbed for v0.2.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..callback import ZVFCallback
from ..controller import StepDecision

ArrayLike = Union[Sequence[float], np.ndarray]


class BaseZVFAdapter:
    """Base class for native ZVF triage adapters.

    Subclasses implement :meth:`_extract_rollout_rewards` to pull the flat
    reward vector and matching group ids out of a framework-native rollout/batch
    object. Everything downstream (the ZVF state machine, adaptive ``G``,
    per-prompt drop, auto-stop, panel logging) is shared via the wrapped
    :class:`ZVFCallback`.

    Args:
        callback: an existing :class:`ZVFCallback`. If ``None`` a default one is
            constructed from ``**callback_kwargs``.
        **callback_kwargs: forwarded to :class:`ZVFCallback` when ``callback`` is
            not supplied (e.g. ``window``, ``zvf_max``, ``on_collapse``,
            ``adaptive_G``).
    """

    #: Human-readable framework name, set by each subclass.
    framework: str = "base"

    def __init__(
        self,
        callback: Optional[ZVFCallback] = None,
        **callback_kwargs: object,
    ) -> None:
        self.callback: ZVFCallback = callback or ZVFCallback(**callback_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ shared
    @property
    def should_stop(self) -> bool:
        """Whether the wrapped triage logic has requested an early stop."""
        return self.callback.should_stop

    @property
    def group_size(self) -> Optional[int]:
        """Adaptive rollout group size from the most recent decision, if any."""
        if not self.callback.decisions:
            return None
        return self.callback.decisions[-1].group_size

    def process_step(self, batch: object) -> StepDecision:
        """Translate one native ``batch`` and run it through the triage logic.

        Extracts ``(rewards, group_ids)`` via the framework-specific
        :meth:`_extract_rollout_rewards`, then forwards to
        ``ZVFCallback.on_step``. The returned :class:`StepDecision` carries the
        regime, adaptive ``G``, dropped prompts and auto-stop flag.
        """
        rewards, group_ids = self._extract_rollout_rewards(batch)
        return self.callback.on_step(rewards, group_ids)

    # ------------------------------------------------------------- framework API
    def _extract_rollout_rewards(self, batch: object) -> Tuple[ArrayLike, ArrayLike]:
        """Pull ``(rewards, group_ids)`` out of a framework-native batch object.

        Subclasses must override this. It returns a flat reward sequence and a
        same-length group-id sequence (one id per rollout, shared within a
        prompt group).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _extract_rollout_rewards()."
        )
