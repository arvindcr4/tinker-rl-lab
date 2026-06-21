"""verl native ZVF triage adapter (real, beta).

`verl <https://github.com/volcengine/verl>`_ is Apache-2.0. Its GRPO/PPO
trainers carry rollout rewards inside a ``DataProto`` whose ``batch`` is a
TensorDict; per-token scores live under ``batch['token_level_scores']`` (or the
KL-penalised ``batch['token_level_rewards']``), and the prompt-group structure
is recoverable from ``non_tensor_batch['uid']`` -- one uid per prompt, repeated
``actor_rollout_ref.rollout.n`` times (one per rollout). This is exactly how
verl computes group-relative GRPO advantages
(``core_algos.compute_grpo_outcome_advantage`` groups token-level scores by
``index`` == ``uid``).

This adapter reduces the per-token score tensor to one scalar reward per
sequence (sum over the response tokens, matching verl's outcome-reward
convention) and groups those scalars by ``uid``, then runs them through the
shared :class:`~zvf_triage.callback.ZVFCallback` triage logic.

The verl import is **guarded**: importing this module never imports verl, so
``import zvf_triage`` works without verl installed. ``process_step`` operates on
any object that quacks like a ``DataProto`` (``.batch`` mapping +
``.non_tensor_batch['uid']``), so it is testable without the heavy dependency.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .base import ArrayLike, BaseZVFAdapter

# Candidate keys for the per-token / per-sequence reward tensor inside a verl
# DataProto.batch, in preference order. token_level_rewards carries the KL
# penalty folded in; token_level_scores is the raw verifier score; reward_scores
# / rm_scores / rewards are seen in some verl reward-manager paths.
_REWARD_KEYS = (
    "token_level_rewards",
    "token_level_scores",
    "reward_scores",
    "rm_scores",
    "rewards",
    "scores",
    "token_level_reward",
)
_UID_KEY = "uid"


def _is_verl_available() -> bool:
    """True iff verl is importable. Never raises; never imported at module load."""
    import importlib.util

    return importlib.util.find_spec("verl") is not None


class VERLZVFAdapter(BaseZVFAdapter):
    """Native ZVF triage adapter for verl GRPO/PPO trainers (real, beta).

    ``process_step(data_proto)`` reads the rollout reward tensor and the per-
    sequence ``uid`` out of a verl ``DataProto`` (or any duck-typed equivalent),
    reduces the rewards to one scalar per sequence, groups by ``uid``, and runs
    the batch through the wrapped :class:`ZVFCallback`. The returned
    :class:`StepDecision` carries the regime, adaptive ``G``, dropped prompts and
    auto-stop flag; ``self.should_stop`` reflects auto-stop / abort-collapse.

    Args:
        callback / **callback_kwargs: forwarded to :class:`BaseZVFAdapter`.
        reward_key: override the reward tensor key (otherwise auto-detected from
            ``_REWARD_KEYS``).
        uid_key: ``non_tensor_batch`` key holding the per-sequence prompt id
            (default ``"uid"``).
    """

    framework = "verl"

    def __init__(
        self,
        callback=None,
        reward_key: Optional[str] = None,
        uid_key: str = _UID_KEY,
        **callback_kwargs: object,
    ) -> None:
        super().__init__(callback=callback, **callback_kwargs)
        self.reward_key = reward_key
        self.uid_key = uid_key
        #: Whether verl itself is importable in this environment (informational).
        self.verl_available = _is_verl_available()

    # ------------------------------------------------------------ extraction

    def _extract_rollout_rewards(self, batch: object) -> Tuple[ArrayLike, ArrayLike]:
        """Pull ``(rewards, uids)`` out of a verl ``DataProto``-like object.

        Expects ``batch.batch`` (a TensorDict / mapping of named tensors) and
        ``batch.non_tensor_batch`` (a mapping holding ``uid``). Per-sequence
        scalar reward = sum of the per-token reward tensor over the token axis
        (a no-op for an already 1-D per-sequence reward).
        """
        tensor_batch = getattr(batch, "batch", None)
        non_tensor = getattr(batch, "non_tensor_batch", None)
        if tensor_batch is None or non_tensor is None:
            raise TypeError(
                "VERLZVFAdapter.process_step expects a verl DataProto-like object "
                "with `.batch` (named reward tensors) and `.non_tensor_batch['uid']`; "
                f"got {type(batch).__name__}."
            )

        reward_tensor = self._select_reward_tensor(tensor_batch)
        rewards = self._reduce_to_per_sequence(reward_tensor)

        uids = non_tensor[self.uid_key] if self.uid_key in non_tensor else None
        if uids is None:
            raise KeyError(
                f"verl DataProto.non_tensor_batch has no {self.uid_key!r} key "
                f"(available: {list(getattr(non_tensor, 'keys', lambda: [])())}). "
                "GRPO requires per-prompt uids to group rollouts."
            )
        uids = np.asarray(uids).ravel()
        if uids.shape[0] != rewards.shape[0]:
            raise ValueError(
                f"verl rollout/uid length mismatch: {rewards.shape[0]} rewards vs "
                f"{uids.shape[0]} uids."
            )
        return rewards, uids

    def _select_reward_tensor(self, tensor_batch: Any) -> Any:
        if self.reward_key is not None:
            if not _has_key(tensor_batch, self.reward_key):
                raise KeyError(
                    f"reward_key {self.reward_key!r} not in verl DataProto.batch "
                    f"(available: {_keys(tensor_batch)})."
                )
            return tensor_batch[self.reward_key]
        for key in _REWARD_KEYS:
            if _has_key(tensor_batch, key):
                return tensor_batch[key]
        raise KeyError(
            "no recognised reward tensor in verl DataProto.batch; looked for "
            f"{list(_REWARD_KEYS)}, found {_keys(tensor_batch)}. Pass reward_key=..."
        )

    @staticmethod
    def _reduce_to_per_sequence(reward_tensor: Any) -> np.ndarray:
        """Reduce a (B, T) per-token reward tensor to a (B,) per-sequence reward.

        Sums over the last (token) axis -- verl's outcome reward is the sum of
        token-level scores along the response. Accepts torch tensors, numpy
        arrays, or lists. An already 1-D per-sequence reward passes through.
        """
        arr = _to_numpy_2d_safe(reward_tensor)
        if arr.ndim == 1:
            return arr.astype(float)
        return arr.reshape(arr.shape[0], -1).sum(axis=1).astype(float)

    # --------------------------------------------------------------- install

    def install(self, trainer: object) -> None:
        """Register a per-step ZVF hook on a verl ``RayPPOTrainer``.

        verl does not expose a formal callback registry, so this attaches a
        light wrapper around the trainer's reward-computation entry point: after
        the trainer produces the rollout ``DataProto`` for a step, the wrapper
        calls :meth:`process_step` on it and, on auto-stop / abort-collapse, sets
        a ``zvf_should_stop`` flag on the trainer that the training loop can
        check. This requires verl to be installed.

        Raises:
            ImportError: if verl is not importable.
            AttributeError: if the trainer exposes no recognised rollout hook.
        """
        if not self.verl_available:
            raise ImportError(
                "VERLZVFAdapter.install() requires verl to be installed "
                "(pip install verl). The framework-agnostic process_step / "
                "ZVFCallback.on_step path works without it."
            )

        hook_name = None
        for candidate in ("compute_advantage", "_compute_advantage", "fit"):
            if hasattr(trainer, candidate):
                hook_name = candidate
                break
        if hook_name is None:
            raise AttributeError(
                "verl trainer exposes no recognised rollout hook "
                "(compute_advantage / fit). Call process_step(data_proto) "
                "manually from your verl training loop instead."
            )

        adapter = self
        original = getattr(trainer, hook_name)

        def _wrapped(data, *args, **kwargs):  # pragma: no cover - needs real verl
            result = original(data, *args, **kwargs)
            try:
                adapter.process_step(data)
            except Exception as exc:
                # Triage must never break training, but the exception
                # must NOT be silent: a user debugging "why is ZVF
                # not adapting my G?" needs to know the adapter
                # failed. Log to stderr with the offending exception
                # and the trainer class for diagnosis, then return
                # the trainer's result unchanged.
                import sys as _sys
                import traceback as _traceback
                print(
                    f"[zvf-triage] WARN: VERLZVFAdapter.process_step raised "
                    f"{type(exc).__name__}: {exc}. Training continues; "
                    f"auto-stop / adaptive G are NOT in effect for this step. "
                    f"Trainer: {type(trainer).__name__}",
                    file=_sys.stderr,
                )
                _traceback.print_exc(file=_sys.stderr)
                return result
            setattr(trainer, "zvf_should_stop", adapter.should_stop)
            return result

        setattr(trainer, hook_name, _wrapped)
        setattr(trainer, "zvf_should_stop", False)


# --------------------------------------------------------------------- helpers


def _to_numpy_2d_safe(x: Any) -> np.ndarray:
    detach = getattr(x, "detach", None)
    if detach is not None:
        try:
            x = detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(x, dtype=float)


def _has_key(mapping: Any, key: str) -> bool:
    try:
        return key in mapping
    except TypeError:
        return hasattr(mapping, key)


def _keys(mapping: Any) -> list:
    keys = getattr(mapping, "keys", None)
    if callable(keys):
        try:
            return list(keys())
        except Exception:
            return []
    return []
