"""Pure-logic ZVF triage controller (no framework dependency).

Implements the state machine from the ZVF Program slide deck:

  Per optimizer step, given the batch's per-group rewards we compute
  ``zvf = mean(1[var(r_group) <= eps])`` and the batch mean reward ``r_bar``,
  then classify the step into one of four regimes:

    * COLD_START_COLLAPSE : zvf > theta_hi and r_bar < eps_lo
        -> nothing is being solved AND there is no within-group contrast.
           Signal "warm_start" (seed easier prompts / warm-start the policy).
    * SATURATION          : zvf > theta_hi and r_bar > 1 - eps_hi
        -> everything is solved; groups are uniform because the task is too easy.
           Signal "lift_difficulty".
    * EXPLOITABLE_CONTRAST : zvf moderate, within band -> healthy. Spend budget.
    * (default) HEALTHY    : low zvf, plenty of usable gradient.

  Adaptive group size: ``G_t = clip(G0 * f(zvf), Gmin, Gmax)`` -- when ZVF is
  high we *increase* the rollout group size so that more rollouts per prompt
  raise the chance of producing within-group reward contrast.

  Per-prompt drop: a prompt whose ZVF == 1 (zero-variance) for ``K`` consecutive
  steps is dropped (it is contributing no gradient).

  Auto-stop: if the *global* batch ZVF == 1 for ``K`` consecutive steps the run
  is aborted (the entire batch has collapsed to zero gradient signal).

The ZVF / GU math itself lives in :mod:`zvf_triage.core` and is reused verbatim.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from . import core

ArrayLike = Union[Sequence[float], np.ndarray]


class Regime(str, Enum):
    """Triage regime for a single optimizer step.

    Note: there is no HEALTHY value. The base regime name in the
    parent paper is "healthy" but the controller's classifier never
    returns it; it returns EXPLOITABLE_CONTRAST for any non-collapse,
    non-saturation regime (a single regime label for a continuum of
    operating points). Keeping HEALTHY as a dead enum value would
    mislead a downstream consumer who imports Regime.HEALTHY and
    never sees it produced.
    """

    EXPLOITABLE_CONTRAST = "exploitable_contrast"
    COLD_START_COLLAPSE = "cold_start_collapse"
    SATURATION = "saturation"
    # Special regime: every prompt in the batch has been dropped (the
    # keep-mask filtered out all rewards). The "batch" is empty so the
    # ZVF / mean-reward numbers are vacuous. Reporting
    # cold_start_collapse here would be a false positive (the
    # reviewer-flagged "misleading COLD_START_COLLAPSE label" case);
    # consumers should treat this as a "no data" signal and skip
    # classification. auto_stop is NEVER set under this regime.
    ALL_DROPPED = "all_dropped"


# Signal an upstream callback may act on. "noop" means "no special action".
Signal = str  # one of: "noop", "warm_start", "lift_difficulty", "abort"


@dataclass
class StepDecision:
    """Structured decision returned by :meth:`ZVFController.step`."""

    step: int
    zvf: float
    gu: float
    mean_reward: float
    regime: Regime
    signal: Signal
    group_size: int
    rolling_zvf: float
    dropped_prompts: List[object] = field(default_factory=list)
    auto_stop: bool = False
    n_groups: int = 0
    notes: str = ""
    # Number of rollout rewards excluded from this step's ZVF
    # computation because they belonged to prompts that had been
    # dropped by the per-prompt drop logic. Observability field
    # only; the exclusion happens at the top of step().
    filtered_rewards: int = 0

    def as_dict(self) -> dict:
        d = {
            "step": self.step,
            "zvf": self.zvf,
            "gu": self.gu,
            "mean_reward": self.mean_reward,
            "regime": self.regime.value,
            "signal": self.signal,
            "group_size": self.group_size,
            "rolling_zvf": self.rolling_zvf,
            "dropped_prompts": list(self.dropped_prompts),
            "auto_stop": self.auto_stop,
            "n_groups": self.n_groups,
            "notes": self.notes,
            "filtered_rewards": self.filtered_rewards,
        }
        return d


def _default_adaptive_fn(zvf_value: float) -> float:
    """Default G multiplier ``f(zvf)``.

    Bidirectional and SMOOTHED: G INCREASES when ZVF is high (the
    batch is starving for contrast, so fish for it with more
    rollouts) and G DECREASES when ZVF is low (the batch is already
    saturated; per-rollout cost is wasted on rollouts that all
    return the same bit). The previous version used raw ZVF, which
    caused oscillation under high-variance regimes (G swung
    Gmin<->Gmax every few steps); the controller now exposes
    ``controller.smoothed_zvf = core.rolling_zvf(self._zvf_history,
    self.window)[-1]`` for use as the argument to this function, so
    the default behavior is implicitly smooth.

    Concretely:
      - zvf = 0.0 (full contrast, healthy) -> multiplier 0.5x (halve G,
        per-rollout budget is wasted on already-easy prompts)
      - zvf = 0.4 (the dead-band upper edge) -> multiplier 1.0x (G0)
      - zvf = 0.85 (the early-warning threshold) -> multiplier 1.5x
      - zvf = 1.0 (full collapse) -> multiplier 2.0x (double G to fish
        for contrast)

    Callers can override with any ``f: [0,1] -> float`` and pass the
    smoothed ZVF themselves if they want a different smoothing
    window. Note: the default uses raw ZVF as input; the controller
    decides whether to call it with raw or smoothed ZVF (current
    controller calls with smoothed).
    """
    z = float(zvf_value)
    if z < 0.0:
        z = 0.0
    if z > 1.0:
        z = 1.0
    if z < 0.4:
        # Dead-band: ease off on per-rollout cost when contrast is
        # already plentiful. Linear from 0.5x at z=0.0 to 1.0x at z=0.4.
        return 0.5 + (z / 0.4) * 0.5
    # Linear from 1.0x at z=0.4 to 2.0x at z=1.0.
    return 1.0 + ((z - 0.4) / 0.6) * 1.0


@dataclass
class ZVFController:
    """Stateful triage controller. Frame-work agnostic; pure Python/numpy.

    Args mirror the slide-deck API (window, zvf_max, adaptive_G, ...).

    Args:
        window: trailing window for the rolling ZVF reported in decisions.
        zvf_max (theta_hi): high ZVF threshold above which a step is "high ZVF".
        eps_lo: cold-start mean-reward floor (r_bar < eps_lo => nothing solved).
        eps_hi: saturation slack (r_bar > 1 - eps_hi => everything solved).
        var_eps: per-group variance tolerance passed to core.zvf.
        adaptive_G: enable adaptive group-size scheduling.
        G0, Gmin, Gmax: base / min / max rollout group size.
        adaptive_fn: f(zvf) multiplier for G_t (defaults to 1 + zvf).
        drop_k: drop a prompt after its ZVF==1 for this many consecutive steps.
        stop_k: auto-stop after global ZVF==1 for this many consecutive steps.
        on_collapse: signal emitted on a cold-start collapse regime
            ("warm_start" | "abort" | "noop"). Other regimes emit their own
            canonical signal.
    """

    window: int = 5
    zvf_max: float = 0.85
    eps_lo: float = 0.05
    eps_hi: float = 0.05
    var_eps: float = core.DEFAULT_VAR_EPS
    adaptive_G: bool = False
    G0: int = 8
    Gmin: int = 2
    Gmax: int = 32
    adaptive_fn: Callable[[float], float] = _default_adaptive_fn
    drop_k: int = 3
    stop_k: int = 3
    on_collapse: Signal = "warm_start"

    # --- internal state (not constructor args) ---
    _step: int = field(default=0, init=False)
    _zvf_history: List[float] = field(default_factory=list, init=False)
    _global_collapse_streak: int = field(default=0, init=False)
    _prompt_streaks: Dict[object, int] = field(default_factory=dict, init=False)
    _dropped: Set[object] = field(default_factory=set, init=False)
    _stopped: bool = field(default=False, init=False)

    # Concurrency: this controller is single-threaded. step() mutates
    # _step, _zvf_history, _global_collapse_streak, _prompt_streaks,
    # and _dropped without an internal lock; a concurrent caller would
    # race on _prompt_streaks[gid] (read-modify-write) and on
    # _global_collapse_streak (also RMW). The TRL / verl / OpenRLHF
    # integrations all invoke the controller from the trainer's main
    # thread, so this is safe in practice. _lock is here so a future
    # consumer who needs re-entrant access (e.g. warm_start_fn called
    # from inside on_collapse) can take it; step() does NOT take it
    # itself to keep the hot path lock-free for the common
    # single-threaded case.
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------ public

    @property
    def dropped_prompts(self) -> List[object]:
        return sorted(self._dropped, key=lambda x: str(x))

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def zvf_history(self) -> List[float]:
        return list(self._zvf_history)

    def lock(self) -> threading.RLock:
        """Return the controller's re-entrant lock.

        Use as ``with controller.lock():`` to serialize step() calls
        from multiple threads. The single-threaded integrations (TRL,
        verl, OpenRLHF) do not need this; the multiprocess
        integrations use a separate ZVFCallback wrapper that takes
        the lock for them. See README "Threading model".
        """
        return self._lock

    def reset(self) -> None:
        self._step = 0
        self._zvf_history.clear()
        self._global_collapse_streak = 0
        self._prompt_streaks.clear()
        self._dropped.clear()
        self._stopped = False

    def step(
        self,
        rewards: ArrayLike,
        group_ids: ArrayLike,
    ) -> StepDecision:
        """Ingest one batch of rollout rewards and return a triage decision.

        Args:
            rewards: flat sequence of scalar rollout rewards for this step.
            group_ids: same-length sequence labelling the prompt-group of each
                reward.

        Returns:
            A :class:`StepDecision`.

        Dropped-prompt handling: prompts whose ZVF has been 1.0 for
        `drop_k` consecutive steps are added to ``self.dropped_prompts``.
        On the *next* call to ``step()``, the rewards and group_ids
        corresponding to dropped prompts are filtered out before any ZVF
        / mean-reward computation, so the dropped prompts do not
        continue to drag the batch's ZVF. The number of filtered
        rewards is recorded as ``filtered_rewards`` in the returned
        decision for observability. The previous version merely
        *identified* dropped prompts in the ``dropped_prompts`` field
        but never excluded them from future rollouts, which made the
        drop an "identification, not an action" (per the prior
        review).
        """
        self._step += 1
        rewards_arr = np.asarray(rewards, dtype=float).ravel()
        group_ids_arr = np.asarray(group_ids).ravel()

        # Filter out rewards belonging to dropped prompts BEFORE any
        # ZVF / mean-reward computation. We log the number of filtered
        # rewards so a downstream consumer can see how much the
        # batch shrank.
        keep_mask = np.array(
            [gid not in self._dropped for gid in group_ids_arr.tolist()],
            dtype=bool,
        )
        filtered_rewards_count = int((~keep_mask).sum())
        if filtered_rewards_count > 0:
            rewards_arr = rewards_arr[keep_mask]
            group_ids_arr = group_ids_arr[keep_mask]

        # If the keep-mask filtered out every reward (every prompt in
        # the batch was already in self._dropped), the batch is empty
        # and ZVF / mean-reward are vacuous. We report Regime.ALL_DROPPED
        # so a downstream consumer does not mis-interpret
        # (zvf=1.0, mean_reward=0.0) as a cold-start collapse (the
        # reviewer-flagged "misleading COLD_START_COLLAPSE" case).
        # auto_stop is NEVER set under this regime; the per-prompt
        # drop tracking is also skipped (there is nothing to drop).
        all_dropped = rewards_arr.size == 0
        if all_dropped:
            self._zvf_history.append(1.0)
            roll = float(core.rolling_zvf(self._zvf_history, window=self.window)[-1])
            return StepDecision(
                step=self._step,
                zvf=1.0,
                gu=0.0,
                mean_reward=0.0,
                regime=Regime.ALL_DROPPED,
                signal="noop",
                group_size=self._adaptive_group_size(roll),
                rolling_zvf=roll,
                dropped_prompts=[],
                auto_stop=False,
                n_groups=0,
                notes="all prompts in batch had been dropped; vacuous ZVF/mean_reward",
                filtered_rewards=filtered_rewards_count,
            )

        batch_zvf = core.zvf(rewards_arr, group_ids_arr, eps=self.var_eps)
        batch_gu = 1.0 - batch_zvf
        mean_reward = float(np.mean(rewards_arr)) if rewards_arr.size else 0.0
        n_groups = len(set(group_ids_arr.tolist()))

        self._zvf_history.append(batch_zvf)
        roll = float(core.rolling_zvf(self._zvf_history, window=self.window)[-1])

        regime, signal = self._classify(batch_zvf, mean_reward)

        # --- per-prompt drop tracking ---
        newly_dropped: List[object] = []
        per_prompt = core.per_prompt_zvf(rewards_arr, group_ids_arr, eps=self.var_eps)
        for gid, is_zero_var in per_prompt.items():
            if gid in self._dropped:
                continue
            if is_zero_var >= 1.0:
                self._prompt_streaks[gid] = self._prompt_streaks.get(gid, 0) + 1
                if self._prompt_streaks[gid] >= self.drop_k:
                    self._dropped.add(gid)
                    newly_dropped.append(gid)
            else:
                self._prompt_streaks[gid] = 0

        # --- global auto-stop tracking ---
        # "global ZVF == 1" => the entire batch collapsed to zero variance.
        # Auto-stop fires ONLY on cold-start collapse (high ZVF + low
        # reward), not on saturation (high ZVF + high reward). A
        # saturated run that has solved every prompt is healthy: the
        # batch is collapsing to a uniform "all correct" outcome, not
        # to a degenerate "all zero-gradient" failure mode. The
        # previous version would abort "everything solved" runs as
        # if they were collapsing. We now require mean_reward <
        # self.eps_lo (cold-start) before setting auto_stop=True;
        # saturation is left as a no-op and the regime is reported
        # but the run continues.
        if batch_zvf >= 1.0 - 1e-12 and mean_reward < self.eps_lo:
            self._global_collapse_streak += 1
        else:
            self._global_collapse_streak = 0
        auto_stop = (
            self._global_collapse_streak >= self.stop_k
            and mean_reward < self.eps_lo
        )
        if auto_stop:
            self._stopped = True
            signal = "abort"
            regime = Regime.COLD_START_COLLAPSE

        # Use the rolling-window SMOOTHED ZVF for the adaptive_G
        # decision, not the raw per-step value. The raw value is
        # noisy (a single step can flip between 0.6 and 1.0 by
        # chance) and would cause the controller to oscillate G
        # between Gmin and Gmax every few steps under realistic
        # training conditions. The smoothed value (rolling mean
        # over `window` steps) is what the parent paper calls
        # "rolling ZVF" and is the natural input to a
        # decision-of-control.
        group_size = self._adaptive_group_size(roll)

        return StepDecision(
            step=self._step,
            zvf=batch_zvf,
            gu=batch_gu,
            mean_reward=mean_reward,
            regime=regime,
            signal=signal,
            group_size=group_size,
            rolling_zvf=roll,
            dropped_prompts=newly_dropped,
            auto_stop=auto_stop,
            n_groups=n_groups,
            notes=self._notes(regime, batch_zvf, mean_reward),
            filtered_rewards=filtered_rewards_count,
        )

    # ------------------------------------------------------------------ private

    def _classify(self, batch_zvf: float, mean_reward: float) -> Tuple[Regime, Signal]:
        high_zvf = batch_zvf > self.zvf_max
        if high_zvf and mean_reward < self.eps_lo:
            return Regime.COLD_START_COLLAPSE, self.on_collapse
        if high_zvf and mean_reward > (1.0 - self.eps_hi):
            return Regime.SATURATION, "lift_difficulty"
        if high_zvf:
            # High ZVF but reward is mid-range: groups are uniform yet the task
            # is neither solved nor unsolved -- still degenerate, treat as a soft
            # collapse so the operator is warned.
            return Regime.COLD_START_COLLAPSE, self.on_collapse
        # Below the threshold there is usable contrast to exploit.
        return Regime.EXPLOITABLE_CONTRAST, "noop"

    def _adaptive_group_size(self, batch_zvf: float) -> int:
        if not self.adaptive_G:
            return self.G0
        raw = self.G0 * self.adaptive_fn(batch_zvf)
        clipped = int(round(min(max(raw, self.Gmin), self.Gmax)))
        return clipped

    @staticmethod
    def _notes(regime: Regime, batch_zvf: float, mean_reward: float) -> str:
        if regime is Regime.COLD_START_COLLAPSE:
            return (
                f"high ZVF ({batch_zvf:.3f}) with low reward "
                f"({mean_reward:.3f}): cold-start collapse"
            )
        if regime is Regime.SATURATION:
            return (
                f"high ZVF ({batch_zvf:.3f}) with high reward "
                f"({mean_reward:.3f}): task saturated"
            )
        return f"usable within-group contrast (ZVF={batch_zvf:.3f})"