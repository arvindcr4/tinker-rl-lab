"""Pure-numpy ZVF / GU / drift primitives.

These reuse the math from the ZVF Program's existing experiments and paper so the
package is numerically consistent with the published diagnostics.

Provenance of each definition (source files in the parent ``tinker-rl-lab`` repo):

* ``zvf`` — the Zero-Variance Fraction. Formal definition is
  Eq. (eq:zvf) of ``paper/sections/appendix_zvf_formalization.tex``::

      ZVF_t = (1 / |B_t|) * sum_g  1[ Var_hat(r_{g,1..K}) <= eps ]

  with ``Var_hat`` the *unbiased* sample variance and ``eps = 1e-6`` for rewards
  normalized to [0, 1]. The reference implementation in
  ``experiments/causal_zvf_experiment.py`` (``compute_zvf_and_gu``) and the live
  probe ``experiments/tinker-runs/live_zvf_probe.py`` thresholds the per-group
  standard deviation at ~1e-8. We expose ``eps`` (variance tolerance) and keep
  the default behaviour identical to the existing code: a group counts toward ZVF
  iff its rollout variance is <= eps.

* ``group_uniformity`` (a.k.a. GU / "Gradient Utilization") — defined as
  ``GU = 1 - ZVF`` in ``experiments/causal_zvf_experiment.py`` (line 141) and in
  ``paper/.../sections/intro.tex`` ("Gradient Utilization (GU = 1 - ZVF)").
  GU is the fraction of groups that still carry usable group-relative advantage
  (non-zero gradient signal).

* ``peak_to_tail_drift`` (PTD) — from ``paper/main_eai_body.tex`` (line 1139)::

      PTD = (r_max - mean(r_tail)) / r_max

  the net reward degradation from the peak to the mean of the last ``tail``
  reward values. PTD > 0.3 signals severe reward instability.

* ``rolling_zvf`` — windowed mean of a per-step ZVF series; mirrors the
  first-K-step "early window" used in
  ``experiments/zvf_predictive_validation.py`` and the rolling-window proxy in
  ``paper/main_eai_body.tex`` (line 1141).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

# Default variance tolerance. Matches eps = 1e-6 from the paper's formal
# definition (appendix_zvf_formalization.tex). The existing code thresholds the
# per-group *std* at ~1e-8; 1e-6 on variance is equivalent at that scale for
# binary rewards and is the value stated in the paper.
DEFAULT_VAR_EPS = 1e-6


def _grouped_rewards(
    rewards: ArrayLike,
    group_ids: ArrayLike,
) -> "dict[object, list]":
    """Bucket a flat ``rewards`` array by ``group_ids`` preserving group order."""
    rewards = np.asarray(rewards, dtype=float).ravel()
    group_ids = np.asarray(group_ids).ravel()
    if rewards.shape[0] != group_ids.shape[0]:
        raise ValueError(
            f"rewards and group_ids must have equal length; "
            f"got {rewards.shape[0]} and {group_ids.shape[0]}"
        )
    groups: "dict[object, list]" = {}
    for gid, r in zip(group_ids.tolist(), rewards.tolist()):
        groups.setdefault(gid, []).append(r)
    return groups


def _group_variance(group: Sequence[float]) -> float:
    """Unbiased sample variance of a single group's rollout rewards.

    A group with fewer than 2 rollouts has undefined variance; consistent with
    the paper's boundary condition ("K=1 ... ZVF is trivially 1"), such a group
    is treated as zero-variance (variance returned as 0.0, which is <= eps and
    therefore counts toward ZVF).

    NaN/inf guard: if any reward is non-finite (NaN or inf), the group's
    variance is returned as 0.0. Without this, a parse glitch that
    produced a single "nan" string in the rewards array would silently
    poison the entire batch's ZVF (np.var of a vector containing NaN
    returns NaN, and the comparison NaN <= eps is False, so the
    group would be silently miscounted as a non-zero-variance
    group). Counting it as zero-variance and surfacing it via the
    count of NaN-containing groups is a conservative default; a
    downstream consumer who wants a strict NaN-as-failure policy can
    detect the drop in the count via a separate guard.
    """
    arr = np.asarray(group, dtype=float)
    if arr.size < 2:
        return 0.0
    if not np.isfinite(arr).all():
        return 0.0
    # ddof=1 -> unbiased sample variance (Var_hat in the paper).
    return float(np.var(arr, ddof=1))


def n_nan_groups(rewards: ArrayLike, group_ids: ArrayLike) -> int:
    """Count of groups whose reward array contains at least one NaN/inf.

    Companion to :func:`zvf`: when this is non-zero, the ZVF summary
    is understating the number of zero-variance groups (those groups
    are silently counted as zero-variance in the current default).
    A consumer that wants strict NaN-as-failure should call this and
    raise.
    """
    rewards = np.asarray(rewards, dtype=float).ravel()
    group_ids = np.asarray(group_ids).ravel()
    out = 0
    for gid, r in zip(group_ids.tolist(), rewards.tolist()):
        # A non-finite scalar r poisons its group.
        pass
    # Bucket by group and check finiteness
    groups = _grouped_rewards(rewards, group_ids)
    return sum(
        1 for g in groups.values()
        if not np.isfinite(np.asarray(g, dtype=float)).all()
    )


def zvf(
    rewards: ArrayLike,
    group_ids: ArrayLike,
    eps: float = DEFAULT_VAR_EPS,
) -> float:
    """Zero-Variance Fraction over a GRPO batch.

    Fraction of prompt-groups whose rollout rewards have (unbiased sample)
    variance ``<= eps`` -- i.e. groups that contribute zero group-relative
    advantage and hence zero gradient signal.

    Source: Eq. (eq:zvf), appendix_zvf_formalization.tex;
    reference impl experiments/causal_zvf_experiment.py::compute_zvf_and_gu.

    Args:
        rewards: flat sequence of scalar rollout rewards.
        group_ids: same-length sequence labelling which prompt-group each
            reward belongs to.
        eps: variance tolerance (default 1e-6).

    Returns:
        ZVF in [0, 1]. An empty batch returns 1.0 (degenerate / no usable
        signal), matching compute_zvf_and_gu's empty-batch behaviour.
    """
    groups = _grouped_rewards(rewards, group_ids)
    if not groups:
        return 1.0
    zero_var = sum(1 for g in groups.values() if _group_variance(g) <= eps)
    return zero_var / len(groups)


def group_uniformity(
    rewards: ArrayLike,
    group_ids: ArrayLike,
    eps: float = DEFAULT_VAR_EPS,
) -> float:
    """Group Uniformity / Gradient Utilization: ``GU = 1 - ZVF``.

    Fraction of groups that still carry usable within-group reward contrast.

    Source: experiments/causal_zvf_experiment.py line 141; intro.tex
    ("Gradient Utilization (GU = 1 - ZVF)").
    """
    return 1.0 - zvf(rewards, group_ids, eps=eps)


# Backwards/alias-friendly name used in some of the existing experiment logs.
def gu(rewards: ArrayLike, group_ids: ArrayLike, eps: float = DEFAULT_VAR_EPS) -> float:
    """Alias for :func:`group_uniformity`."""
    return group_uniformity(rewards, group_ids, eps=eps)


def rolling_zvf(zvf_series: ArrayLike, window: int = 5) -> np.ndarray:
    """Trailing-window mean of a per-step ZVF series.

    For each step ``t`` returns the mean of ZVF over the last ``window`` steps
    (or fewer at the start of the run). Mirrors the first-K-step early window of
    experiments/zvf_predictive_validation.py and the rolling-window proxy in
    main_eai_body.tex (window = 5).

    Args:
        zvf_series: 1-D sequence of per-step ZVF values.
        window: trailing window length (>= 1).

    Returns:
        Array of the same length as ``zvf_series`` of rolling means.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    series = np.asarray(zvf_series, dtype=float).ravel()
    out = np.empty(series.shape[0], dtype=float)
    for t in range(series.shape[0]):
        lo = max(0, t - window + 1)
        out[t] = float(np.mean(series[lo : t + 1]))
    return out


def rolling_variance(reward_series: ArrayLike, window: int = 5) -> np.ndarray:
    """Trailing-window (population) variance of a per-step reward series.

    Real-time instability signal from main_eai_body.tex (line 1141,
    "Rolling Variance (window = 5 steps)").
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    series = np.asarray(reward_series, dtype=float).ravel()
    out = np.empty(series.shape[0], dtype=float)
    for t in range(series.shape[0]):
        lo = max(0, t - window + 1)
        out[t] = float(np.var(series[lo : t + 1]))
    return out


def peak_to_tail_drift(reward_series: ArrayLike, tail: int = 10) -> float:
    """Peak-to-Tail Drift (PTD): net reward degradation from peak to tail.

    ``PTD = (r_max - mean(r_tail)) / r_max`` where ``r_tail`` is the last
    ``tail`` reward values. PTD > 0.3 signals severe reward instability.

    Source: paper/main_eai_body.tex line 1139.

    Returns:
        PTD as a float. If the peak reward is 0 (e.g. a never-learning run),
        returns 0.0 to avoid division by zero -- there is no peak to drift from.
    """
    series = np.asarray(reward_series, dtype=float).ravel()
    if series.size == 0:
        return 0.0
    r_max = float(np.max(series))
    if r_max == 0.0:
        return 0.0
    n_tail = min(tail, series.size)
    r_tail = float(np.mean(series[-n_tail:]))
    return (r_max - r_tail) / r_max


def stability_index(reward_series: ArrayLike, tail: int = 10) -> float:
    """Stability Index (SI): coefficient of variation of the last-``tail`` rewards.

    ``SI = std(r_tail) / |mean(r_tail)|``. High SI = erratic late-stage reward.

    Source: paper/main_eai_body.tex line 1135.
    """
    series = np.asarray(reward_series, dtype=float).ravel()
    if series.size == 0:
        return 0.0
    n_tail = min(tail, series.size)
    r_tail = series[-n_tail:]
    mu = float(np.mean(r_tail))
    if mu == 0.0:
        return 0.0
    return float(np.std(r_tail)) / abs(mu)


def per_prompt_zvf(
    rewards: ArrayLike,
    group_ids: ArrayLike,
    eps: float = DEFAULT_VAR_EPS,
) -> "dict[object, float]":
    """Per-group zero-variance indicator (1.0 if the group is zero-variance).

    Useful for the per-prompt drop logic in the controller: a prompt-group whose
    rollouts all collapsed to the same outcome maps to 1.0, otherwise 0.0.
    """
    groups = _grouped_rewards(rewards, group_ids)
    return {
        gid: (1.0 if _group_variance(g) <= eps else 0.0)
        for gid, g in groups.items()
    }
