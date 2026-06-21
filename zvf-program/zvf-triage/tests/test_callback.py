"""Unit tests for the ZVFCallback (framework-agnostic path)."""

import numpy as np
import pytest

from zvf_triage import ZVFCallback, ZVFPanel
from zvf_triage.controller import Regime


def _uniform(value, n_groups=4, group_size=2):
    rewards, gids = [], []
    for g in range(n_groups):
        rewards.extend([value] * group_size)
        gids.extend([g] * group_size)
    return rewards, gids


def _mixed(n_groups=4, group_size=2):
    rewards, gids = [], []
    for g in range(n_groups):
        rewards.extend([1.0, 0.0] + [0.0] * (group_size - 2))
        gids.extend([g] * group_size)
    return rewards, gids


def test_import_does_not_require_torch_or_trl():
    # The package imported above with only numpy installed; reaching here is the
    # assertion. Building a callback must also not import torch/trl.
    cb = ZVFCallback(window=5, zvf_max=0.85, on_collapse="warm_start", adaptive_G=True)
    assert cb is not None


def test_invalid_on_collapse_raises():
    with pytest.raises(ValueError):
        ZVFCallback(on_collapse="explode")


def test_on_step_returns_decision():
    cb = ZVFCallback()
    d = cb.on_step(*_mixed())
    assert d.regime is Regime.EXPLOITABLE_CONTRAST
    assert len(cb.decisions) == 1


def test_warm_start_hook_fires_on_collapse():
    fired = []
    cb = ZVFCallback(on_collapse="warm_start", warm_start_fn=lambda d: fired.append(d.step))
    cb.on_step(*_uniform(0.0))  # cold-start collapse
    assert fired == [1]
    assert not cb.should_stop  # warm_start does not stop training


def test_abort_hook_sets_should_stop():
    fired = []
    cb = ZVFCallback(on_collapse="abort", abort_fn=lambda d: fired.append(d.step))
    cb.on_step(*_uniform(0.0))
    assert cb.should_stop
    assert fired == [1]


def test_noop_does_not_stop_or_fire():
    fired = []
    cb = ZVFCallback(on_collapse="noop", warm_start_fn=lambda d: fired.append(1))
    cb.on_step(*_uniform(0.0))
    assert not cb.should_stop
    assert fired == []


def test_auto_stop_sets_should_stop():
    cb = ZVFCallback(on_collapse="warm_start", stop_k=2)
    cb.on_step(*_uniform(0.0))
    assert not cb.should_stop
    cb.on_step(*_uniform(0.0))  # second consecutive global collapse -> auto-stop
    assert cb.should_stop


def test_adaptive_g_via_callback():
    cb = ZVFCallback(adaptive_G=True, G0=8, Gmin=2, Gmax=32)
    d = cb.on_step(*_uniform(0.0))  # ZVF=1 -> G=16
    assert d.group_size == 16


def test_on_decision_hook_called_every_step():
    seen = []
    cb = ZVFCallback(on_decision=lambda d: seen.append(d.step))
    cb.on_step(*_mixed())
    cb.on_step(*_mixed())
    assert seen == [1, 2]


def test_memory_panel_records_history():
    panel = ZVFPanel(backend="memory")
    cb = ZVFCallback(panel=panel)
    cb.on_step(*_mixed())
    cb.on_step(*_uniform(0.0))
    assert len(panel.history) == 2
    assert panel.history[0]["zvf"] == 0.0
    assert panel.history[1]["zvf"] == 1.0


def test_dropped_prompts_exposed():
    cb = ZVFCallback(drop_k=2)
    cb.on_step([0.0, 0.0, 1.0, 0.0], [0, 0, 1, 1])
    cb.on_step([0.0, 0.0, 1.0, 0.0], [0, 0, 1, 1])
    assert 0 in cb.dropped_prompts


def test_reset():
    cb = ZVFCallback(stop_k=1)
    cb.on_step(*_uniform(0.0))
    assert cb.should_stop
    cb.reset()
    assert not cb.should_stop
    assert cb.decisions == []


def test_custom_adaptive_fn():
    cb = ZVFCallback(adaptive_G=True, G0=4, Gmax=64, adaptive_fn=lambda z: 1 + 3 * z)
    d = cb.on_step(*_uniform(0.0))  # ZVF=1 -> 4*(1+3)=16
    assert d.group_size == 16


def test_as_trl_callback_without_transformers_raises_cleanly():
    cb = ZVFCallback()
    try:
        import transformers  # noqa: F401
        has_transformers = True
    except Exception:
        has_transformers = False
    if has_transformers:
        # If transformers happens to be installed, it must build a real callback.
        tcb = cb.as_trl_callback()
        assert tcb is not None
    else:
        with pytest.raises(ImportError):
            cb.as_trl_callback()


def test_panel_log_returns_flat_metrics():
    panel = ZVFPanel(backend="memory")
    cb = ZVFCallback(panel=panel)
    d = cb.on_step(*_uniform(1.0))
    last = panel.history[-1]
    assert last["regime_code"] == 2  # saturation
    assert last["gu"] == 0.0
