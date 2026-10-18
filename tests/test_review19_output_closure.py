"""Synthetic corrected-v4 output consumers; no stored evidence replay."""

import csv
import importlib
import json
from pathlib import Path
import random
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "platform_modal/scripts/p5p8"
sys.path.insert(0, str(SCRIPTS))


def load(pattern):
    return importlib.import_module(next(SCRIPTS.glob(pattern)).stem)


def test_shared_adaptive_policy_and_threshold_boundaries():
    a, b = load("p7_iter119*"), load("p7_iter127*")
    for z in [0.01, 0.45, 0.5, 0.51, 0.7, 0.9, 1]:
        assert a.adaptive_gstar(z) == b.adaptive_gstar(z)
    assert b.adaptive_gstar(0.5) == 8
    c, d = load("p7_iter135*"), load("p7_iter155*")
    for tau in [0.5, 0.7, 0.8]:
        values = [tau - 0.01, tau, tau + 0.01]
        assert d.fire_rate(values, tau) == sum(c.fire(z, tau) for z in values) / 3


def test_pooled_failure_interval_resamples_and_zero_exclusion():
    m = load("p7_iter107_taut*")
    rows = m.pooled_failure_shares(
        {"a": [[0, 1], [1, 0]] * 10, "b": [[0, 1], [1, 0]] * 5}, ["A", "B"], 500, random.Random(1)
    )
    for row in rows:
        assert row["ci_lo"] < row["share_obs"] < row["ci_hi"]
        assert row["ci_excludes_zero"] == int(row["ci_lo"] > 0)
    with pytest.raises(ValueError, match="no observations for pooled shares"):
        m.pooled_failure_shares({"a": []}, ["A"], 100, random.Random(1))


def test_cross_method_pairing_joins_identity_before_missing_filter(monkeypatch):
    m = load("p7_iter171*")
    monkeypatch.setattr(m, "METHODS", ["a", "b"])
    tensors = {
        "a": [
            {"step": 1, "zvf": 0.1, "reward_mean": 0.2},
            {"step": 2, "zvf": float("nan"), "reward_mean": 0.3},
            {"step": 3, "zvf": 0.9, "reward_mean": 0.4},
        ],
        "b": [
            {"step": 3, "zvf": 0.9, "reward_mean": 0.4},
            {"step": 1, "zvf": 0.1, "reward_mean": 0.2},
            {"step": 2, "zvf": 0.5, "reward_mean": 0.3},
        ],
    }
    result = m.cross_method_tost(tensors)
    assert result[0]["n_pairs"] == 2
    assert result[0]["delta"] == 0
    assert result[1]["n_pairs"] == 3
    assert result[1]["delta"] == 0
    tensors["b"].append(tensors["b"][0])
    with pytest.raises(ValueError, match="duplicate step"):
        m.cross_method_tost(tensors)


@pytest.mark.parametrize("bad", [None, float("nan"), -0.1, 1.1])
def test_missing_zvf_is_never_zero(tmp_path, monkeypatch, bad):
    m = load("p7_hybrid_unification_n2.py")
    monkeypatch.setattr(m, "N2_DIR", tmp_path)
    monkeypatch.setattr(m, "N_STEPS", 2)
    rows = [{"zvf": 0.1}, {} if bad is None else {"zvf": bad}]
    (tmp_path / "grpo_s0_tensors.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="ZVF"):
        m.load_n2_methods()


def test_iter115_main_persists_advertised_replay_and_seed_summary(tmp_path, monkeypatch):
    m = load("p7_iter115*")
    panel = {
        seed: [
            {"step": i, "zvf": z, "reward": 0.3, "mean_len": 10}
            for i, z in enumerate([0.55, 0.7, 0.9])
        ]
        for seed in [1, 2, 3]
    }
    monkeypatch.setattr(m, "load_n10", lambda: panel)
    monkeypatch.setattr(m, "OUT_DIR", tmp_path)
    m.main()
    summary = json.loads((tmp_path / "p7_iter115_summary.json").read_text())
    for key in ["controller_replay_tsv", "per_seed_summary_tsv"]:
        with (tmp_path / summary["deliverables"][key]).open() as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert rows
        assert all(row["seed"] for row in rows)
        if key == "controller_replay_tsv":
            assert "target_status" in rows[0]


@pytest.mark.parametrize('name, metric', [('p7_threshold_sweep.py', 'per_seed_metrics'),
                                         ('p7_hybrid_unification_n2.py', 'metrics'),
                                         ('p7_dualformer_n10_seed.py', 'per_seed_metrics')])
def test_saturated_deescalation_is_not_bad_headroom(name, metric):
    fn = getattr(load(name), metric)
    assert fn([4], [1.0])['headroom_bad'] == 0
    assert fn([8], [1.0])['headroom_bad'] == 0
    assert fn([16], [1.0])['headroom_bad'] == 1
    assert fn([16], [.98])['headroom_bad'] == 0


def test_unified_bank_headroom_uses_action_direction():
    m = load('p7_unified_controller_bank.py')
    assert m.n10_per_seed_metrics([1.0], .7, .9)['headroom_bad'] == 0
    assert m.n10_per_seed_metrics([1.0], .7, 1.1)['headroom_bad'] == 1


def test_iter107_taxonomy_follows_actual_incremental_action():
    m = load('p7_iter107_taut*')
    s = {'per_p': [.5], 'zvf': .8, 'pcd': .3}
    assert m.failure_classify(s, .7) == 'GUARD_BLOCKED'
    s['pcd'] = .1
    assert m.decide_gs_c3(s['per_p'], .8, .1, .7) == ([16], False)
    assert m.failure_classify(s, .7) == 'TRIGGER_NOOP'
    assert m.action_diagnostics(s, .7)['incremental_opportunity'] == 0
    s['per_p'] = [.9]
    assert m.failure_classify(s, .7) == 'USEFUL_ESCALATION'
    d = m.action_diagnostics(s, .7)
    assert d['baseline_groups'] == [4] and d['actual_groups'] == [8]
    assert d['incremental_gain'] == pytest.approx(m.iid_zvf(.9, 4)-m.iid_zvf(.9, 8))
    s['zvf'] = .6
    assert m.failure_classify(s, .7) == 'MISSED_MODELED_GAIN'
    s.update(zvf=.8, per_p=[1.0])
    assert m.failure_classify(s, .7) == 'LOW_GAIN_ESCALATION'


def test_iter107_main_serializes_action_taxonomy(tmp_path, monkeypatch):
    m = load('p7_iter107_taut*')
    tensors = {}
    for method in m.METHODS:
        for step, pcd in enumerate([.3, .1]):
            tensors[(method, step)] = [[0, 0, 0, 0, 1, 1, 1, 1]]
            tensors[(method, step, '_meta')] = {'zvf': 1.0, 'pcd': pcd, 'reward_mean': .5}
    monkeypatch.setattr(m, 'load_tensors', lambda: tensors)
    monkeypatch.setattr(m, 'N_BOOT', 40)
    monkeypatch.setattr(m, 'OUT', tmp_path)
    m.main()
    with (tmp_path / 'p7_iter107_failure_taxonomy.tsv').open() as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    assert {r['failure_class'] for r in rows} == {'GUARD_BLOCKED', 'TRIGGER_NOOP'}
    assert all(r['fired'] == '0' and float(r['incremental_gain']) == 0 for r in rows)
    result = json.loads((tmp_path / 'p7_iter107_summary.json').read_text())
    assert sum(result['failure_pooled_shares'].values()) == pytest.approx(1)


def test_iter135_all_edges_and_decline_magnitudes():
    m = load('p7_iter135*')
    rates = {t: 1-i*.1 for i, t in enumerate(m.TAU_GRID)}
    slopes = m.adjacent_rate_slopes(rates)
    assert list(slopes) == m.TAU_GRID[:-1]
    assert list(slopes.values()) == pytest.approx([-2]*7)
    assert m.decline_near_max(slopes, .7) == (2, 2, True)
    assert m.decline_near_max({.5: -4, .7: -1}, .7) == (4, 1, False)
    assert m.decline_near_max({.5: 0, .7: 0}, .7) == (0, 0, False)
    assert m.decline_near_max({.5: -2}, .7) == (2, None, False)


def test_iter135_main_exports_complete_slopes(tmp_path, monkeypatch):
    m = load('p7_iter135*')
    panel = {seed: [{'zvf': z, 'reward': .5} for z in [.525, .575, .625, .675, .725, .775, .825, .9]]
             for seed in m.SEEDS}
    monkeypatch.setattr(m, 'load_n10', lambda: panel)
    monkeypatch.setattr(m, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(m, 'N_STEPS', 8)
    m.main()
    result = json.loads((tmp_path / 'p7_iter135_summary.json').read_text())
    assert len(result['h4_pairwise_slopes']) == 7
    assert result['h4_pass'] is True
    assert result['h4_max_decline_magnitude'] == pytest.approx(2.5)
