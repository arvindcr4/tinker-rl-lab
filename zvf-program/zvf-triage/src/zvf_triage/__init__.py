"""zvf-triage: Zero-Variance Fraction triage for GRPO-style RL fine-tuning.

Pillar 3 of the ZVF Program. A cheap, model-agnostic early-warning callback that
detects when GRPO's group-relative advantage has collapsed to zero gradient
signal, classifies the training regime (cold-start collapse / saturation /
exploitable contrast), adapts the rollout group size, drops dead prompts, and
auto-stops doomed runs.

The ZVF / GU / PTD math is reused verbatim from the ZVF Program experiments and
paper -- see ``zvf_triage.core`` for per-definition source provenance.

Quickstart::

    from zvf_triage import ZVFCallback
    cb = ZVFCallback(window=5, zvf_max=0.85, on_collapse="warm_start",
                     adaptive_G=True)
    for batch in loop:
        decision = cb.on_step(batch.rewards, batch.group_ids)
        if cb.should_stop:
            break
"""

from .core import (
    DEFAULT_VAR_EPS,
    group_uniformity,
    gu,
    peak_to_tail_drift,
    per_prompt_zvf,
    rolling_variance,
    rolling_zvf,
    stability_index,
    zvf,
)
from .controller import Regime, StepDecision, ZVFController
from .callback import ZVFCallback
from .panel import ZVFPanel

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core math
    "zvf",
    "group_uniformity",
    "gu",
    "rolling_zvf",
    "rolling_variance",
    "peak_to_tail_drift",
    "stability_index",
    "per_prompt_zvf",
    "DEFAULT_VAR_EPS",
    # controller
    "ZVFController",
    "StepDecision",
    "Regime",
    # callback + panel
    "ZVFCallback",
    "ZVFPanel",
]
