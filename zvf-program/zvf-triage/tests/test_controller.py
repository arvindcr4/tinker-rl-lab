"""Unit tests for the ZVF triage controller state machine."""

import numpy as np
import pytest

from zvf_triage.controller import Regime, ZVFController


def _uniform_batch(value, n_groups=4, group_size=2):
    """All groups uniform at `value` -> ZVF == 1."""
    rewards = []
    gids = []
    for g in range(n_groups):
        rewards.extend([value] * group_size)
        gids.extend([g] * group_size)
    return rewards, gids


def _mixed_batch(n_groups=4, group_size=2):
    """All groups mixed [1,0] -> ZVF == 0."""
    rewards = []
    gids = []
    for g in range(n_groups):
        rewards.extend([1.0, 0.0] + [0.0] * (group_size - 2))
        gids.extend([g] * group_size)
    return rewards, gids


def test_cold_start_collapse_high_zvf_low_reward():
    c = ZVFController(zvf_max=0.85, eps_lo=0.05)
    rewards, gids = _uniform_batch(0.0)  # ZVF=1, mean reward=0
    d = c.step(rewards, gids)
    assert d.zvf == 1.0
    assert d.regime is Regime.COLD_START_COLLAPSE
    assert d.signal == "warm_start"


def test_saturation_high_zvf_high_reward():
    c = ZVFController(zvf_max=0.85, eps_hi=0.05)
    rewards, gids = _uniform_batch(1.0)  # ZVF=1, mean reward=1
    d = c.step(rewards, gids)
    assert d.zvf == 1.0
    assert d.regime is Regime.SATURATION
    assert d.signal == "lift_difficulty"


def test_exploitable_contrast_low_zvf():
    c = ZVFController(zvf_max=0.85)
    rewards, gids = _mixed_batch()  # ZVF=0
    d = c.step(rewards, gids)
    assert d.zvf == 0.0
    assert d.regime is Regime.EXPLOITABLE_CONTRAST
    assert d.signal == "noop"


def test_on_collapse_abort_signal():
    c = ZVFController(on_collapse="abort", eps_lo=0.05)
    rewards, gids = _uniform_batch(0.0)
    d = c.step(rewards, gids)
    assert d.regime is Regime.COLD_START_COLLAPSE
    assert d.signal == "abort"


def test_adaptive_group_size_increases_with_zvf():
    # Default adaptive_fn is now BIDIRECTIONAL: G decreases when ZVF
    # is low (already saturated, per-rollout cost is wasted) and
    # increases when ZVF is high (need contrast). This test verifies
    # both directions.
    c = ZVFController(adaptive_G=True, G0=8, Gmin=2, Gmax=32)
    # ZVF=0 -> multiplier 0.5x -> G=4 (decreased from G0)
    rewards, gids = _mixed_batch()
    assert c.step(rewards, gids).group_size == 4
    # ZVF=1 -> multiplier 2.0x -> G=16 (increased from G0)
    c2 = ZVFController(adaptive_G=True, G0=8, Gmin=2, Gmax=32)
    rewards, gids = _uniform_batch(1.0)
    assert c2.step(rewards, gids).group_size == 16


def test_adaptive_group_size_clipped_to_gmax():
    c = ZVFController(adaptive_G=True, G0=20, Gmin=2, Gmax=32)
    rewards, gids = _uniform_batch(0.0)  # ZVF=1 -> 20*2=40 -> clip to 32
    assert c.step(rewards, gids).group_size == 32


def test_adaptive_monotonic_increasing_override():
    # A user who wants the OLD monotonically-increasing behavior can
    # pass adaptive_fn explicitly. This is the only way to override
    # the new bidirectional default; the default itself is no longer
    # monotonic.
    c = ZVFController(adaptive_G=True, G0=8, Gmin=2, Gmax=32,
                      adaptive_fn=lambda z: 1.0 + z)
    # ZVF=0 -> multiplier 1.0x -> G=8 (unchanged)
    rewards, gids = _mixed_batch()
    assert c.step(rewards, gids).group_size == 8
    # ZVF=1 -> multiplier 2.0x -> G=16 (doubled)
    c2 = ZVFController(adaptive_G=True, G0=8, Gmin=2, Gmax=32,
                       adaptive_fn=lambda z: 1.0 + z)
    rewards, gids = _uniform_batch(1.0)
    assert c2.step(rewards, gids).group_size == 16


def test_adaptive_disabled_returns_g0():
    c = ZVFController(adaptive_G=False, G0=8)
    rewards, gids = _uniform_batch(0.0)
    assert c.step(rewards, gids).group_size == 8


def test_per_prompt_drop_after_k_consecutive():
    # Group 0 always zero-variance; group 1 always mixed. drop_k=3.
    c = ZVFController(drop_k=3, zvf_max=0.85)
    for step in range(1, 4):
        rewards = [0.0, 0.0, 1.0, 0.0]  # g0 uniform, g1 mixed
        gids = [0, 0, 1, 1]
        d = c.step(rewards, gids)
        if step < 3:
            assert d.dropped_prompts == []
        else:
            assert 0 in d.dropped_prompts
            assert 1 not in d.dropped_prompts
    assert 0 in c.dropped_prompts
    assert 1 not in c.dropped_prompts


def test_per_prompt_streak_resets_on_contrast():
    c = ZVFController(drop_k=2)
    # zero-var, zero-var, mixed (resets), zero-var -> never reaches 2 in a row
    seqs = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    dropped_any = False
    # use drop_k=3 so the reset matters
    c = ZVFController(drop_k=3)
    for r0, r1 in seqs:
        d = c.step([r0, r1], [0, 0])
        if d.dropped_prompts:
            dropped_any = True
    assert not dropped_any


def test_auto_stop_after_k_global_collapse():
    c = ZVFController(stop_k=3, eps_lo=0.05)
    last = None
    for step in range(1, 4):
        rewards, gids = _uniform_batch(0.0)  # global ZVF == 1 every step
        last = c.step(rewards, gids)
        if step < 3:
            assert not last.auto_stop
            assert not c.stopped
        else:
            assert last.auto_stop
            assert c.stopped
            assert last.signal == "abort"


def test_auto_stop_streak_resets_on_non_collapse():
    c = ZVFController(stop_k=2)
    c.step(*_uniform_batch(0.0))            # ZVF=1, streak=1
    c.step(*_mixed_batch())                 # ZVF=0, streak reset
    d = c.step(*_uniform_batch(0.0))        # ZVF=1, streak=1 again (not 2)
    assert not d.auto_stop
    assert not c.stopped


def test_rolling_zvf_reported():
    c = ZVFController(window=2)
    d1 = c.step(*_mixed_batch())       # zvf 0
    d2 = c.step(*_uniform_batch(0.0))  # zvf 1
    assert d1.rolling_zvf == pytest.approx(0.0)
    assert d2.rolling_zvf == pytest.approx(0.5)  # mean(0,1)


def test_reset_clears_state():
    c = ZVFController(stop_k=1)
    c.step(*_uniform_batch(0.0))
    assert c.stopped
    c.reset()
    assert not c.stopped
    assert c.zvf_history == []
    assert c.dropped_prompts == []


def test_decision_as_dict_roundtrip():
    c = ZVFController()
    d = c.step(*_mixed_batch())
    dd = d.as_dict()
    assert dd["regime"] == "exploitable_contrast"
    assert dd["step"] == 1
    assert "zvf" in dd and "gu" in dd


def test_n_groups_counted():
    c = ZVFController()
    d = c.step(*_uniform_batch(0.0, n_groups=5))
    assert d.n_groups == 5
