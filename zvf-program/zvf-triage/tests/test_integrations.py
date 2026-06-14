"""Tests for the native-adapter scaffolds in zvf_triage.integrations.

These assert the *honest stub* contract: the framework adapters instantiate but
raise NotImplementedError when used, and the shared BaseZVFAdapter wiring works
end-to-end when a subclass provides a real extraction.
"""

import importlib

import numpy as np
import pytest

from zvf_triage import ZVFCallback
from zvf_triage.controller import Regime
from zvf_triage.integrations import (
    BaseZVFAdapter,
    NeMoRLZVFAdapter,
    OpenRLHFZVFAdapter,
    VERLZVFAdapter,
)


# A concrete adapter exercising the shared BaseZVFAdapter machinery.
class _ListAdapter(BaseZVFAdapter):
    framework = "list"

    def _extract_rollout_rewards(self, batch):
        # batch is just (rewards, group_ids) here.
        return batch[0], batch[1]


def _uniform(value, n_groups=4, group_size=2):
    rewards, gids = [], []
    for g in range(n_groups):
        rewards.extend([value] * group_size)
        gids.extend([g] * group_size)
    return rewards, gids


def _mixed(n_groups=4, group_size=2):
    rewards, gids = [], []
    for g in range(n_groups):
        rewards.extend([1.0, 0.0])
        gids.extend([g] * group_size)
    return rewards, gids


# ----------------------------------------------------------------- base adapter

def test_base_adapter_builds_callback_from_kwargs():
    a = _ListAdapter(zvf_max=0.85, adaptive_G=True, G0=8)
    assert isinstance(a.callback, ZVFCallback)
    assert a.should_stop is False
    assert a.group_size is None  # no step processed yet


def test_base_adapter_accepts_prebuilt_callback():
    cb = ZVFCallback(adaptive_G=True, G0=4)
    a = _ListAdapter(callback=cb)
    assert a.callback is cb


def test_base_adapter_process_step_runs_triage():
    a = _ListAdapter(adaptive_G=True, G0=8, Gmin=2, Gmax=32)
    d = a.process_step(_mixed())
    assert d.regime is Regime.EXPLOITABLE_CONTRAST
    assert d.group_size == 8  # ZVF=0 -> base G
    assert a.group_size == 8


def test_base_adapter_forwards_auto_stop():
    a = _ListAdapter(stop_k=2)
    a.process_step(_uniform(0.0))
    assert not a.should_stop
    a.process_step(_uniform(0.0))  # second consecutive global collapse
    assert a.should_stop


def test_base_adapter_default_extract_raises():
    a = BaseZVFAdapter(window=5)
    with pytest.raises(NotImplementedError):
        a.process_step(object())


# ------------------------------------------------------------- framework stubs

# OpenRLHF and NeMo-RL remain honest NotImplementedError stubs; verl is real.
_STUB_ADAPTERS = [OpenRLHFZVFAdapter, NeMoRLZVFAdapter]


@pytest.mark.parametrize("cls", [VERLZVFAdapter, OpenRLHFZVFAdapter, NeMoRLZVFAdapter])
def test_framework_adapter_instantiates(cls):
    a = cls(window=5, zvf_max=0.85)
    assert isinstance(a, BaseZVFAdapter)
    assert a.framework in {"verl", "openrlhf", "nemo_rl"}


@pytest.mark.parametrize("cls", _STUB_ADAPTERS)
def test_framework_adapter_extract_not_implemented(cls):
    a = cls()
    with pytest.raises(NotImplementedError):
        a.process_step(object())


@pytest.mark.parametrize("cls", _STUB_ADAPTERS)
def test_framework_adapter_install_not_implemented(cls):
    a = cls()
    with pytest.raises(NotImplementedError):
        a.install(object())


@pytest.mark.parametrize("cls", _STUB_ADAPTERS)
def test_not_implemented_message_is_actionable(cls):
    # The stub message must point users at the working on_step path.
    a = cls()
    with pytest.raises(NotImplementedError, match="on_step"):
        a.process_step(object())


# ----------------------------------------------------------- lazy import contract

def test_integrations_import_is_dependency_free():
    # Importing the package must not import verl / openrlhf / nemo / torch.
    mod = importlib.import_module("zvf_triage.integrations")
    assert hasattr(mod, "BaseZVFAdapter")
    # Attribute access lazily resolves the adapter classes via __getattr__.
    assert mod.VERLZVFAdapter is VERLZVFAdapter


def test_integrations_unknown_attr_raises():
    mod = importlib.import_module("zvf_triage.integrations")
    with pytest.raises(AttributeError):
        _ = mod.DoesNotExistAdapter


# ------------------------------------------------------------- verl (real, beta)


class _FakeDataProto:
    """Minimal stand-in for a verl ``DataProto``.

    ``batch`` is a dict of named per-token reward tensors (numpy here, but the
    adapter also accepts torch tensors / lists); ``non_tensor_batch`` carries the
    per-sequence ``uid`` that groups rollouts by prompt, exactly like verl.
    """

    def __init__(self, reward_tensor, uids, key="token_level_scores"):
        self.batch = {key: np.asarray(reward_tensor, dtype=float)}
        self.non_tensor_batch = {"uid": np.asarray(uids, dtype=object)}


def _verl_batch(per_seq_scores, uids, n_tokens=3):
    """Build a (B, n_tokens) per-token tensor whose row-sums equal per_seq_scores."""
    rows = []
    for s in per_seq_scores:
        row = [0.0] * (n_tokens - 1) + [float(s)]  # all mass on the last token
        rows.append(row)
    return np.asarray(rows, dtype=float), np.asarray(uids, dtype=object)


def test_verl_adapter_is_real_not_stub():
    # verl adapter must NOT raise NotImplementedError on a valid DataProto.
    a = VERLZVFAdapter()
    rt, uids = _verl_batch([1.0, 0.0, 1.0, 0.0], ["p0", "p0", "p1", "p1"])
    dp = _FakeDataProto(rt, uids)
    d = a.process_step(dp)
    assert d is not None
    assert d.regime is Regime.EXPLOITABLE_CONTRAST  # both groups have contrast


def test_verl_adapter_groups_by_uid_and_computes_zvf():
    # p0 = {1,1} zero-variance; p1 = {1,0} contrast -> ZVF = 0.5, GU = 0.5.
    a = VERLZVFAdapter()
    rt, uids = _verl_batch([1.0, 1.0, 1.0, 0.0], ["p0", "p0", "p1", "p1"])
    d = a.process_step(_FakeDataProto(rt, uids))
    assert d.zvf == 0.5
    assert d.gu == 0.5
    assert d.n_groups == 2


def test_verl_adapter_reduces_token_scores_by_sum():
    # Per-token scores must be summed per sequence. Group p0: row-sums {2,2}
    # (zero-var); group p1: {3,1} (contrast). ZVF over 2 groups = 0.5.
    a = VERLZVFAdapter()
    rt = np.array([[1.0, 1.0], [2.0, 0.0], [1.0, 2.0], [0.0, 1.0]])
    uids = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    d = a.process_step(_FakeDataProto(rt, uids))
    assert d.zvf == 0.5
    assert d.n_groups == 2


def test_verl_adapter_auto_stops_on_collapse_streak():
    # Every step fully collapsed (all groups zero-variance) -> ZVF == 1 ->
    # auto-stop after stop_k consecutive steps.
    a = VERLZVFAdapter(stop_k=2)
    rt, uids = _verl_batch([0.0, 0.0, 0.0, 0.0], ["p0", "p0", "p1", "p1"])
    dp = _FakeDataProto(rt, uids)
    a.process_step(dp)
    assert not a.should_stop
    a.process_step(dp)  # second consecutive global collapse
    assert a.should_stop


def test_verl_adapter_accepts_1d_per_sequence_rewards():
    # If batch already holds a 1-D per-sequence reward, pass it through.
    a = VERLZVFAdapter()
    dp = _FakeDataProto([1.0, 1.0, 1.0, 0.0], ["p0", "p0", "p1", "p1"])
    d = a.process_step(dp)
    assert d.zvf == 0.5


def test_verl_adapter_explicit_reward_key():
    a = VERLZVFAdapter(reward_key="rewards")
    dp = _FakeDataProto([1.0, 0.0, 1.0, 0.0], ["p0", "p0", "p1", "p1"], key="rewards")
    d = a.process_step(dp)
    assert d.regime is Regime.EXPLOITABLE_CONTRAST


def test_verl_adapter_missing_uid_raises_keyerror():
    a = VERLZVFAdapter()

    class _NoUID:
        batch = {"token_level_scores": np.array([[1.0], [0.0]])}
        non_tensor_batch = {}

    with pytest.raises(KeyError):
        a.process_step(_NoUID())


def test_verl_adapter_wrong_object_type_raises_typeerror():
    a = VERLZVFAdapter()
    with pytest.raises(TypeError):
        a.process_step(object())


def test_verl_adapter_no_recognised_reward_tensor_raises():
    a = VERLZVFAdapter()

    class _BadBatch:
        batch = {"something_else": np.array([[1.0]])}
        non_tensor_batch = {"uid": np.array(["p0"], dtype=object)}

    with pytest.raises(KeyError):
        a.process_step(_BadBatch())


def test_verl_adapter_install_without_verl_raises_importerror():
    # verl is not installed in this venv; install() must say so, not silently no-op.
    a = VERLZVFAdapter()
    if a.verl_available:  # pragma: no cover - verl not expected in test env
        pytest.skip("verl is installed; ImportError path not exercised")
    with pytest.raises(ImportError):
        a.install(object())


def test_verl_adapter_module_imports_without_verl():
    # Importing the verl integration must not require verl itself.
    mod = importlib.import_module("zvf_triage.integrations.verl")
    assert hasattr(mod, "VERLZVFAdapter")
