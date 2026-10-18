"""Synthetic corrected-v4 output consumers; no stored evidence replay."""
import csv
import importlib
import json
from pathlib import Path
import random
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'platform_modal/scripts/p5p8'
sys.path.insert(0, str(SCRIPTS))


def load(pattern):
    return importlib.import_module(next(SCRIPTS.glob(pattern)).stem)


def test_shared_adaptive_policy_and_threshold_boundaries():
    a, b = load('p7_iter119*'), load('p7_iter127*')
    for z in [.01, .45, .5, .51, .7, .9, 1]:
        assert a.adaptive_gstar(z) == b.adaptive_gstar(z)
    assert b.adaptive_gstar(.5) == 8
    c, d = load('p7_iter135*'), load('p7_iter155*')
    for tau in [.5, .7, .8]:
        values = [tau-.01, tau, tau+.01]
        assert d.fire_rate(values, tau) == sum(c.fire(z, tau) for z in values) / 3


def test_pooled_failure_interval_resamples_and_zero_exclusion():
    m = load('p7_iter107_taut*')
    rows = m.pooled_failure_shares({'a': [[0, 1], [1, 0]] * 10,
                                    'b': [[0, 1], [1, 0]] * 5}, ['A', 'B'],
                                   500, random.Random(1))
    for row in rows:
        assert row['ci_lo'] < row['share_obs'] < row['ci_hi']
        assert row['ci_excludes_zero'] == int(row['ci_lo'] > 0)
    with pytest.raises(ValueError):
        m.pooled_failure_shares({'a': []}, ['A'], 100, random.Random(1))


def test_cross_method_pairing_joins_identity_before_missing_filter(monkeypatch):
    m = load('p7_iter171*')
    monkeypatch.setattr(m, 'METHODS', ['a', 'b'])
    tensors = {'a': [{'step': 1, 'zvf': .1, 'reward_mean': .2},
                     {'step': 2, 'zvf': float('nan'), 'reward_mean': .3},
                     {'step': 3, 'zvf': .9, 'reward_mean': .4}],
               'b': [{'step': 3, 'zvf': .9, 'reward_mean': .4},
                     {'step': 1, 'zvf': .1, 'reward_mean': .2},
                     {'step': 2, 'zvf': .5, 'reward_mean': .3}]}
    result = m.cross_method_tost(tensors)
    assert result[0]['n_pairs'] == 2 and result[0]['delta'] == 0
    assert result[1]['n_pairs'] == 3 and result[1]['delta'] == 0
    tensors['b'].append(tensors['b'][0])
    with pytest.raises(ValueError, match='duplicate step'):
        m.cross_method_tost(tensors)


@pytest.mark.parametrize('bad', [None, float('nan'), -0.1, 1.1])
def test_missing_zvf_is_never_zero(tmp_path, monkeypatch, bad):
    m = load('p7_hybrid_unification_n2.py')
    monkeypatch.setattr(m, 'N2_DIR', tmp_path)
    monkeypatch.setattr(m, 'N_STEPS', 2)
    rows = [{'zvf': .1}, {} if bad is None else {'zvf': bad}]
    (tmp_path / 'grpo_s0_tensors.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='ZVF'):
        m.load_n2_methods()


def test_iter115_main_persists_advertised_replay_and_seed_summary(tmp_path, monkeypatch):
    m = load('p7_iter115*')
    panel = {seed: [{'step': i, 'zvf': z, 'reward': .3, 'mean_len': 10}
                    for i, z in enumerate([.55, .7, .9])] for seed in [1, 2, 3]}
    monkeypatch.setattr(m, 'load_n10', lambda: panel)
    monkeypatch.setattr(m, 'OUT_DIR', tmp_path)
    m.main()
    summary = json.loads((tmp_path / 'p7_iter115_summary.json').read_text())
    for key in ['controller_replay_tsv', 'per_seed_summary_tsv']:
        with (tmp_path / summary['deliverables'][key]).open() as fh:
            rows = list(csv.DictReader(fh, delimiter='\t'))
        assert rows and all(row['seed'] for row in rows)
        if key == 'controller_replay_tsv':
            assert 'target_status' in rows[0]
