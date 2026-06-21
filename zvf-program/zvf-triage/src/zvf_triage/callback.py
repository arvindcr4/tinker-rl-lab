"""ZVF triage callback: framework-agnostic base + a lazy TRL adapter.

``ZVFCallback`` is the public entry point named in the slide deck::

    from zvf_triage import ZVFCallback
    trainer = GRPOTrainer(model, dataset, callbacks=[ZVFCallback(
        window=5, zvf_max=0.85, on_collapse="warm_start",
        adaptive_G=True, wandb_panel=True)])

It wires together :mod:`zvf_triage.core` (ZVF/GU math), the
:class:`zvf_triage.controller.ZVFController` (state machine + adaptive G +
per-prompt drop + auto-stop), and :class:`zvf_triage.panel.ZVFPanel` (logging).

The base class has *no* framework dependency: call ``on_step(rewards,
group_ids)`` from any training loop. ``torch`` / ``trl`` / ``transformers`` are
imported lazily, so importing this module never requires them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from . import core
from .controller import Regime, StepDecision, ZVFController
from .panel import ZVFPanel

ArrayLike = Union[Sequence[float], np.ndarray]

# Recognised on_collapse hook names.
_VALID_HOOKS = {"warm_start", "abort", "noop"}


class ZVFCallback:
    """Framework-agnostic ZVF triage callback.

    Args:
        window: rolling-ZVF window.
        zvf_max: high-ZVF threshold (theta_hi).
        eps_lo / eps_hi: cold-start / saturation reward bands.
        on_collapse: action on cold-start collapse: "warm_start" | "abort" |
            "noop". Drives both the emitted signal and the optional hook
            callbacks below.
        adaptive_G: enable adaptive group-size scheduling.
        G0 / Gmin / Gmax: group-size schedule bounds.
        adaptive_fn: optional f(zvf) multiplier override.
        drop_k / stop_k: per-prompt-drop and auto-stop streak lengths.
        var_eps: per-group variance tolerance.
        wandb_panel / tensorboard_panel: enable logging backends.
        panel: a pre-built :class:`ZVFPanel` (overrides the flags above).
        warm_start_fn / abort_fn / on_decision: user hooks (all optional).
    """

    def __init__(
        self,
        window: int = 5,
        zvf_max: float = 0.85,
        eps_lo: float = 0.05,
        eps_hi: float = 0.05,
        on_collapse: str = "warm_start",
        adaptive_G: bool = False,
        G0: int = 8,
        Gmin: int = 2,
        Gmax: int = 32,
        adaptive_fn: Optional[Callable[[float], float]] = None,
        drop_k: int = 3,
        stop_k: int = 3,
        var_eps: float = core.DEFAULT_VAR_EPS,
        wandb_panel: bool = False,
        tensorboard_panel: bool = False,
        panel: Optional[ZVFPanel] = None,
        warm_start_fn: Optional[Callable[[StepDecision], None]] = None,
        abort_fn: Optional[Callable[[StepDecision], None]] = None,
        on_decision: Optional[Callable[[StepDecision], None]] = None,
    ) -> None:
        if on_collapse not in _VALID_HOOKS:
            raise ValueError(
                f"on_collapse must be one of {sorted(_VALID_HOOKS)}; got {on_collapse!r}"
            )

        controller_kwargs: Dict[str, Any] = dict(
            window=window,
            zvf_max=zvf_max,
            eps_lo=eps_lo,
            eps_hi=eps_hi,
            var_eps=var_eps,
            adaptive_G=adaptive_G,
            G0=G0,
            Gmin=Gmin,
            Gmax=Gmax,
            drop_k=drop_k,
            stop_k=stop_k,
            on_collapse=on_collapse,
        )
        if adaptive_fn is not None:
            controller_kwargs["adaptive_fn"] = adaptive_fn

        self.controller = ZVFController(**controller_kwargs)
        self.on_collapse = on_collapse

        if panel is not None:
            self.panel: Optional[ZVFPanel] = panel
        elif wandb_panel:
            self.panel = ZVFPanel(backend="wandb")
        elif tensorboard_panel:
            self.panel = ZVFPanel(backend="tensorboard")
        else:
            self.panel = None

        self.warm_start_fn = warm_start_fn
        self.abort_fn = abort_fn
        self.on_decision = on_decision

        self.decisions: list[StepDecision] = []
        self.should_stop: bool = False

    # ------------------------------------------------------------------ core API

    def on_step(self, rewards: ArrayLike, group_ids: ArrayLike) -> StepDecision:
        """Process one optimizer step's rollout rewards.

        Returns the :class:`StepDecision`. Side effects: logs to the panel (if
        any), fires the configured hooks, and sets ``self.should_stop`` on an
        auto-stop or an ``on_collapse="abort"`` cold-start collapse.
        """
        decision = self.controller.step(rewards, group_ids)
        self.decisions.append(decision)

        if self.panel is not None:
            self.panel.log(decision)

        self._dispatch(decision)

        if self.on_decision is not None:
            self.on_decision(decision)

        return decision

    @property
    def dropped_prompts(self) -> List[object]:
        return self.controller.dropped_prompts

    def _dispatch(self, decision: StepDecision) -> None:
        # Auto-stop always wins.
        if decision.auto_stop:
            self.should_stop = True
            if self.abort_fn is not None:
                self.abort_fn(decision)
            return

        if decision.regime is Regime.COLD_START_COLLAPSE:
            if self.on_collapse == "abort":
                self.should_stop = True
                if self.abort_fn is not None:
                    self.abort_fn(decision)
            elif self.on_collapse == "warm_start":
                if self.warm_start_fn is not None:
                    self.warm_start_fn(decision)
            # "noop": nothing.

    def reset(self) -> None:
        self.controller.reset()
        self.decisions.clear()
        self.should_stop = False

    # --------------------------------------------------------- TRL / HF adapter

    def as_trl_callback(
        self,
        trainer: Any = None,
        reward_key: str = "rewards",
        group_key: str = "group_ids",
        num_generations: Optional[int] = None,
        zvf_log_key: str = "frac_reward_zero_std",
        reward_mean_key: str = "reward",
    ) -> Any:
        """Return a ``transformers.TrainerCallback`` wrapping this triage logic.

        Imports transformers lazily and builds a real
        :class:`transformers.TrainerCallback` matched to the TRL ``GRPOTrainer``
        logging API (verified against trl>=1.6 / transformers>=5).

        Per training step the callback's ``on_log`` hook extracts the GRPO group
        rewards and feeds them through :meth:`on_step`, then mirrors the
        triage decision back into the trainer by setting
        ``control.should_training_stop`` on auto-stop / abort-collapse.

        Reward extraction is layered (first that matches wins):

        1. **Raw per-rollout rewards** under ``reward_key`` in the log dict or
           ``kwargs``. Grouped into contiguous blocks of ``num_generations`` --
           exactly how ``GRPOTrainer`` lays out its rollouts
           (``rewards.view(-1, num_generations)``). ``num_generations`` is taken
           from this argument, then from a wired ``trainer.num_generations``,
           then from ``kwargs``. Explicit ``group_key`` ids override the block
           grouping.
        2. **TRL's native per-step zero-variance fraction** under
           ``zvf_log_key`` (default ``"frac_reward_zero_std"`` -- the metric
           ``GRPOTrainer`` already logs, equal to ZVF). When raw rewards are not
           in the log dict, this fraction *is* the batch ZVF; the adapter
           reconstructs a faithful per-group reward vector that reproduces that
           exact ZVF and batch mean (``reward_mean_key``) so the full controller
           path -- regime, adaptive G, auto-stop -- runs on real numbers.

        Args:
            trainer: optional ``GRPOTrainer`` to read ``num_generations`` from.
            reward_key: log/kwargs key holding the raw per-rollout reward vector.
            group_key: optional log/kwargs key holding explicit group ids.
            num_generations: GRPO rollouts-per-prompt (group size). Overrides the
                value discovered on ``trainer`` / in ``kwargs``.
            zvf_log_key: log key for TRL's per-step zero-std fraction.
            reward_mean_key: log key for the per-step batch mean reward.
        """
        return _make_trl_callback(
            self,
            trainer=trainer,
            reward_key=reward_key,
            group_key=group_key,
            num_generations=num_generations,
            zvf_log_key=zvf_log_key,
            reward_mean_key=reward_mean_key,
        )


def _make_trl_callback(
    zvf_cb: ZVFCallback,
    trainer: Any,
    reward_key: str,
    group_key: str,
    num_generations: Optional[int],
    zvf_log_key: str,
    reward_mean_key: str,
):
    """Build a ``transformers.TrainerCallback`` subclass instance lazily."""
    try:
        from transformers import TrainerCallback  # type: ignore  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - exercised only without transformers
        raise ImportError(
            "as_trl_callback() requires the 'trl' extra "
            "(pip install 'zvf-triage[trl]')."
        ) from exc

    class _TRLZVFCallback(TrainerCallback):  # type: ignore[misc, valid-type]
        """ZVF triage as a HuggingFace/TRL ``TrainerCallback``.

        Matches the real callback hook signatures
        (``on_train_begin``/``on_log`` -> ``(args, state, control, **kwargs)``).
        """

        def __init__(self, inner: ZVFCallback) -> None:
            self.inner = inner
            self._num_generations = num_generations
            self._trainer = trainer

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
            # Discover the GRPO group size from a wired trainer if available.
            self._resolve_num_generations(kwargs)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            logs = logs or {}
            self._resolve_num_generations(kwargs)
            extracted = self._extract_group_rewards(logs, kwargs, state)
            if extracted is None:
                return control
            rewards, group_ids = extracted
            decision = self.inner.on_step(rewards, group_ids)
            if decision.auto_stop or (
                decision.regime is Regime.COLD_START_COLLAPSE
                and self.inner.on_collapse == "abort"
            ):
                control.should_training_stop = True
            return control

        # ----------------------------------------------------------- helpers

        def _resolve_num_generations(self, kwargs: Dict[str, Any]) -> None:
            if self._num_generations:
                return
            for src in (self._trainer, kwargs.get("model")):
                g = getattr(src, "num_generations", None)
                if g:
                    self._num_generations = int(g)
                    return
            g = kwargs.get("num_generations") or kwargs.get("group_size")
            if g:
                self._num_generations = int(g)

        def _extract_group_rewards(self, logs, kwargs, state):
            """Return ``(rewards, group_ids)`` numpy arrays, or ``None``.

            Path 1: raw per-rollout rewards (block-grouped by num_generations,
            or by an explicit group_key). Path 2: reconstruct from TRL's logged
            ``frac_reward_zero_std`` (== ZVF) + mean reward.
            """
            rewards = _extract(logs, kwargs, reward_key)
            if rewards is not None:
                rewards = _to_numpy(rewards)
                group_ids = _extract(logs, kwargs, group_key)
                if group_ids is not None:
                    return rewards, _to_numpy(group_ids)
                return rewards, _infer_group_ids(rewards, self._num_generations)

            # Path 2: TRL's per-step zero-variance fraction is exactly ZVF.
            zvf_val = logs.get(zvf_log_key)
            if zvf_val is None:
                return None
            mean_reward = logs.get(reward_mean_key, 0.0)
            n_groups = self._num_generations or 8
            return _synthesize_from_zvf(float(zvf_val), float(mean_reward), n_groups)

    return _TRLZVFCallback(zvf_cb)


def _extract(logs: Dict[str, Any], kwargs: Dict[str, Any], key: str) -> Any:
    if key in logs:
        return logs[key]
    if key in kwargs:
        return kwargs[key]
    return None


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor / list / array to a flat numpy float array."""
    detach = getattr(x, "detach", None)
    if detach is not None:
        try:
            x = detach().cpu().numpy()  # torch tensor
        except Exception:
            pass
    return np.asarray(x, dtype=float).ravel()


def _infer_group_ids(rewards: np.ndarray, num_generations: Optional[int]) -> np.ndarray:
    """Block-group rewards into contiguous chunks of ``num_generations``.

    Mirrors ``GRPOTrainer``'s ``rewards.view(-1, num_generations)`` layout: the
    first ``G`` rewards are prompt 0, the next ``G`` are prompt 1, and so on.
    Falls back to one group per reward when ``num_generations`` is unknown.
    """
    g = int(num_generations or 1)
    g = max(1, g)
    n = int(rewards.shape[0])
    return np.array([i // g for i in range(n)])


def _synthesize_from_zvf(
    zvf_val: float, mean_reward: float, n_groups: int, group_size: int = 2
) -> "tuple[np.ndarray, np.ndarray]":
    """Reconstruct a per-group reward vector reproducing a given batch ZVF.

    ``GRPOTrainer`` logs ``frac_reward_zero_std`` -- the fraction of prompt
    groups whose rollout rewards have zero std -- which is exactly the ZVF the
    controller would compute. When the raw rollouts are unavailable we rebuild a
    minimal batch with ``n_groups`` groups of ``group_size`` rollouts such that:

    * round(zvf_val * n_groups) groups are zero-variance (set to ``mean_reward``),
    * the rest carry within-group contrast (split around ``mean_reward``),

    so :func:`core.zvf` on the result equals ``zvf_val`` (to grid resolution) and
    the batch mean equals ``mean_reward``. The reconstruction is exact for the
    ZVF and mean; it does not claim to recover the true reward distribution.
    """
    n_groups = max(1, int(n_groups))
    group_size = max(2, int(group_size))
    n_zero = int(round(zvf_val * n_groups))
    n_zero = min(max(n_zero, 0), n_groups)

    rewards: List[float] = []
    group_ids: List[int] = []
    for g in range(n_groups):
        if g < n_zero:
            # Zero-variance group: all rollouts identical -> counts toward ZVF.
            block = [mean_reward] * group_size
        else:
            # Contrast group: symmetric spread about the mean, nonzero variance.
            lo = max(0.0, mean_reward - 0.5)
            hi = min(1.0, mean_reward + 0.5)
            if hi <= lo:  # mean at a boundary; force a tiny spread
                lo, hi = mean_reward, mean_reward + 1e-3
            block = [lo, hi] + [mean_reward] * (group_size - 2)
        rewards.extend(block)
        group_ids.extend([g] * group_size)
    return np.asarray(rewards, dtype=float), np.asarray(group_ids)
