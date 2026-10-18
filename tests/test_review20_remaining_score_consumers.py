"""Offline synthetic consumers: no fitting, provider calls or stored-result writes."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from test_pro_review_19_regressions import SCRIPTS, functions


def score_namespace(**kwargs):
    common = functions('_p8_common', np=np)
    return functions(
        'p8_iter100_score_gradient_seed_stability', np=np, auc=common['auc_p8'],
        K_PCT=50, COST_XGB=0.0001, COST_LLM=0.001,
        **kwargs,
    )


def test_secondary_scores_change_catches_and_screening_always_charged():
    ns = score_namespace()
    raw = np.array([0.49, 0.51])
    y = np.array([1, 0])
    first = ns['eval_rule'](raw, raw, y, 'absolute-band', 1, w=0.1)
    second = ns['eval_rule'](raw, 1 - raw, y, 'absolute-band', 1, w=0.1)
    assert first['n_caught_at_K2'] == 0
    assert first['cost_per_fraud_caught'] is None
    assert second['n_caught_at_K2'] == 1
    assert second['cost_per_decision'] == pytest.approx(0.0011)


def test_main_uses_fixed_full_surrogate_paired_draws_and_computed_narrative(tmp_path):
    ns = score_namespace(
        ROOT=tmp_path, RES=tmp_path, RAW20=['raw'], ALL24=['raw', 'full'],
        SEED_ORIG=101, SEED_NEW=202, N_BOOT=30, G_THR=0.01, W=0.1,
        json=json,
    )
    y = np.array([1, 0, 1, 0])
    raw = np.array([0.48, 0.52, 0.49, 0.51])
    full = np.array([0.9, 0.1, 0.8, 0.2])
    ns['load'] = lambda _: (np.zeros((4, 2)), y)
    fits = []
    def fit(Xtr, ytr, Xte, cols, seed):
        fits.append((tuple(cols), seed))
        return full.copy() if len(cols) == 2 else raw.copy()
    ns['fit'] = fit
    original = ns['eval_rule']
    second_stages = []
    def evaluate(p_cheap, p_llm, *args, **kwargs):
        second_stages.append(p_llm.copy())
        return original(p_cheap, p_llm, *args, **kwargs)
    ns['eval_rule'] = evaluate
    outputs = {}
    ns['write_tsv'] = lambda name, rows: outputs.update({name: rows})
    ns['main']()
    assert len(fits) == 4
    assert all(np.array_equal(p, full) for p in second_stages)
    boot = outputs['p8_iter100_score_gradient_seed_stability.tsv']
    assert all(r['delta_recall_ci_lo'] == r['delta_recall_ci_hi'] == 0 for r in boot)
    summary = json.loads((tmp_path / 'p8_iter100_score_gradient_seed_stability.json').read_text())
    assert not summary['headline_seed_pass']
    assert 'failed' in summary['key_finding']
    assert '143' not in summary['key_finding']
    assert 'not measured LLM' in summary['second_stage']


def test_disagreement_import_does_not_read_fit_write_or_plot(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('analysis ran during module import')
    xgb = types.ModuleType('xgboost')
    xgb.XGBClassifier = forbidden
    mpl = types.ModuleType('matplotlib')
    mpl.use = lambda _: None
    pyplot = types.ModuleType('matplotlib.pyplot')
    pyplot.subplots = forbidden
    common = types.ModuleType('_p8_common')
    common.auc_p8 = functions('_p8_common', np=np)['auc_p8']
    paths = types.ModuleType('_paths')
    paths.REPO_ROOT = tmp_path
    for name, mod in [('xgboost', xgb), ('matplotlib', mpl),
                      ('matplotlib.pyplot', pyplot), ('_p8_common', common), ('_paths', paths)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    spec = importlib.util.spec_from_file_location('disagreement_under_test', SCRIPTS / 'p8_decision_disagreement.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert module.auc_p8(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
