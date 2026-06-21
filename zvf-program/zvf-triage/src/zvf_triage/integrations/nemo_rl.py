"""NeMo-RL native ZVF triage adapter (stub for v0.2).

`NeMo-RL <https://github.com/NVIDIA-NeMo/RL>`_ is Apache-2.0. Its GRPO
implementation carries rollouts in a ``BatchedDataDict``; per-sample rewards land
under the ``"rewards"`` / ``"total_reward"`` keys and the prompt-group structure
follows from the GRPO config's ``num_generations_per_prompt`` (samples are laid
out in contiguous per-prompt blocks of that size).

The plan for the real adapter:

* Hook into the ``grpo_train`` loop (``nemo_rl.algorithms.grpo``) so
  ``process_step`` runs once per optimizer step with the rollout
  ``BatchedDataDict``.
* ``_extract_rollout_rewards`` reads the per-sample reward column and rebuilds
  group ids from ``num_generations_per_prompt``.
* Forward ``should_stop`` to the loop's termination check and feed the adaptive
  ``G`` back into ``num_generations_per_prompt`` for the next step.

Honest stub: instantiation works, use raises :class:`NotImplementedError`. The
generic ``ZVFCallback.on_step(rewards, group_ids)`` path already works with
NeMo-RL today if wired manually.
"""

from __future__ import annotations

from typing import Tuple

from .base import ArrayLike, BaseZVFAdapter

# TODO(v0.2): implement the NeMo-RL BatchedDataDict -> (rewards, group_ids)
# extraction and the grpo_train hook. Reads the per-sample reward column and
# rebuilds group ids from num_generations_per_prompt. Track against the NeMo-RL
# Apache-2.0 GRPO API.

_NOT_IMPLEMENTED_MSG = (
    "NeMoRLZVFAdapter is a v0.2 scaffold and is not implemented yet. "
    "For now, read per-sample rewards from your NeMo-RL BatchedDataDict and call "
    "ZVFCallback.on_step(rewards, group_ids) directly, with group_ids derived "
    "from num_generations_per_prompt. "
    "Track progress at <project repository>/issues (see pyproject [project.urls])."
)


class NeMoRLZVFAdapter(BaseZVFAdapter):
    """Native ZVF triage adapter for NeMo-RL GRPO (NOT YET IMPLEMENTED).

    See the module docstring for the intended ``BatchedDataDict`` extraction. The
    class signature is stable; the body lands in v0.2.
    """

    framework = "nemo_rl"

    def _extract_rollout_rewards(self, batch: object) -> Tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def install(self, trainer: object) -> None:
        """Register this adapter on a NeMo-RL trainer (NOT YET IMPLEMENTED).

        Will attach ``process_step`` to the ``grpo_train`` loop and wire
        ``should_stop`` / adaptive ``G`` back into ``num_generations_per_prompt``.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
