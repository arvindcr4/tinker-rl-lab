"""OpenRLHF native ZVF triage adapter (stub for v0.2).

`OpenRLHF <https://github.com/OpenRLHF/OpenRLHF>`_ is Apache-2.0. Its GRPO /
REINFORCE++ path builds ``Experience`` objects whose ``info["reward"]`` holds the
per-sample scalar reward; the prompt-group structure follows from
``n_samples_per_prompt`` (the ``--n_samples_per_prompt`` flag) -- samples are
emitted in contiguous blocks of that size, one block per prompt.

The plan for the real adapter:

* Subclass / wrap ``PPOTrainer`` (the ``openrlhf.trainer`` Ray and non-Ray
  variants) and call ``process_step`` once the experiences for a step are
  collected.
* ``_extract_rollout_rewards`` flattens ``info["reward"]`` across the experience
  list and reconstructs group ids from ``n_samples_per_prompt``.
* Forward ``should_stop`` to the training loop and the adaptive ``G`` back into
  ``n_samples_per_prompt`` for the next generation round.

Honest stub: the class instantiates but raises :class:`NotImplementedError` when
used. The generic ``ZVFCallback.on_step(rewards, group_ids)`` path already works
with OpenRLHF today if wired manually.
"""

from __future__ import annotations

from typing import Tuple

from .base import ArrayLike, BaseZVFAdapter

# TODO(v0.2): implement the OpenRLHF Experience -> (rewards, group_ids) extraction
# and the PPOTrainer hook. Reads info["reward"] and rebuilds group ids from
# n_samples_per_prompt. Track against the OpenRLHF Apache-2.0 trainer API.

_NOT_IMPLEMENTED_MSG = (
    "OpenRLHFZVFAdapter is a v0.2 scaffold and is not implemented yet. "
    "For now, gather per-sample rewards from your OpenRLHF Experience objects "
    "and call ZVFCallback.on_step(rewards, group_ids) directly, with group_ids "
    "derived from n_samples_per_prompt. "
    "Track progress at <project repository>/issues (see pyproject [project.urls])."
)


class OpenRLHFZVFAdapter(BaseZVFAdapter):
    """Native ZVF triage adapter for OpenRLHF trainers (NOT YET IMPLEMENTED).

    See the module docstring for the intended ``Experience`` extraction. The
    class signature is stable; the body lands in v0.2.
    """

    framework = "openrlhf"

    def _extract_rollout_rewards(self, batch: object) -> Tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def install(self, trainer: object) -> None:
        """Register this adapter on an OpenRLHF trainer (NOT YET IMPLEMENTED).

        Will attach ``process_step`` to the ``PPOTrainer`` step loop and wire
        ``should_stop`` / adaptive ``G`` back into ``n_samples_per_prompt``.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
