"""CPU-only regression fixtures; never load research result data."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'platform_modal' / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def test_aic_columns():
    m = importlib.import_module('scaling_law_iter129')
    rows = [[0,0,0,0,0,0,1,0,1,0], [0,0,0,0,0,0,-3,1,1,1]]
    assert m.summarize_aic_wins(rows) == (1, 2, 1)


def test_overall_permutation_metadata(tmp_path, monkeypatch):
    m = importlib.import_module('scaling_law_iter109b')
    monkeypatch.setattr(m, 'RES', tmp_path)
    monkeypatch.setattr(m, 'N_PERM', 19)
    monkeypatch.setattr(m, 'N_BOOT', 19)
    anchors = [dict(name=name, lambda_3p=.1 + i*.07, R_inf=.3+i*.03,
                    rmse=.01+i*.001, valid=True, lambda_lo=.01, lambda_hi=1.0)
               for i, name in enumerate(m.MODELS_PARAMS_B)]
    (tmp_path/'scaling_law_iter109_meta.json').write_text(json.dumps({'anchors':anchors}))
    m.main()
    meta = json.loads((tmp_path/'scaling_law_iter109b_meta.json').read_text())
    import csv
    with (tmp_path/'scaling_law_iter109b_permtest.tsv').open() as f:
        row = next(csv.DictReader(f, delimiter='\t'))
    value = meta['E_permutation_test']['lambda_vs_N']['p_two_sided']
    assert value == float(row['p_two_sided'])
    assert value >= 1/20


@pytest.mark.parametrize('module', ['scaling_law_iter125', 'scaling_law_iter133'])
def test_no_fake_dip_pvalue(module):
    m = importlib.import_module(module)
    d, p = m.uniform_cdf_deviation(np.array([0., .1, .9, 1.]))
    assert np.isfinite(d) and np.isnan(p)
    assert not hasattr(m, 'hartigan_dip_bootstrap')


def test_capability_label_permutation():
    m = importlib.import_module('scaling_law_iter133')
    values=np.array([.2,.3,.8,.9])
    assert m.higher_mean_cluster(values, [1,1,2,2]).tolist() == [False,False,True,True]
    assert m.higher_mean_cluster(values, [2,2,1,1]).tolist() == [False,False,True,True]
    loc, _ = m.largest_gap_split(values)
    assert (values > sorted(values)[loc]).tolist() == [False,False,True,True]


def test_failed_fit_and_insufficient_bootstrap(monkeypatch):
    m = importlib.import_module('scaling_law_iter109')
    monkeypatch.setattr(m, 'N_BOOT', 2)
    def fail(*a, **kw): raise RuntimeError('fixture')
    monkeypatch.setattr(m, 'curve_fit', fail)
    fit=m.fit_sat_3p(np.arange(5.), np.ones(5))
    assert not fit['valid'] and fit['bootstrap_successes'] == 0
    assert np.isnan(m.bootstrap_slope([], [], n_boot=2)['mean'])


def test_flat_offset_unidentifiable():
    m=importlib.import_module('scaling_law_iter137')
    fit=m.fit_3p(np.arange(1.,11.), np.full(10,.5))
    assert not fit['identifiable'] and np.isnan(fit['t_80'])


def test_canonical_entrypoint_only_main():
    tree=ast.parse((SCRIPTS/'scaling_law_fit.py').read_text())
    block=tree.body[-1]
    assert len(block.body)==1 and isinstance(block.body[0], ast.Expr)


def test_seed_runner_no_eval_and_failed_exit(tmp_path):
    runner=SCRIPTS/'run_seeds.sh'
    marker=tmp_path/'should_not_exist'
    result=subprocess.run(['bash',str(runner),'--seed','2','--',sys.executable,'-c',
                           'import sys; print(sys.argv); sys.exit(1)',f'; touch {marker}'],capture_output=True,text=True)
    assert result.returncode == 1 and not marker.exists()
    assert 'exit_status=1' in result.stdout
    assert subprocess.run(['bash',str(runner),'--seed','oops','--','true'],capture_output=True).returncode==2


@pytest.mark.parametrize('case', ['empty', 'all_invalid', 'two_eligible', 'constant_size'])
def test_permutation_main_undefined_is_not_significant(tmp_path, monkeypatch, case):
    m = importlib.import_module('scaling_law_iter109b')
    monkeypatch.setattr(m, 'RES', tmp_path)
    monkeypatch.setattr(m, 'N_PERM', 99)
    monkeypatch.setattr(m, 'N_BOOT', 19)
    anchors = [dict(name=name, lambda_3p=.1+i*.1, R_inf=.4+i*.02,
                    rmse=.01, valid=case != 'all_invalid', lambda_lo=.01, lambda_hi=1.)
               for i, name in enumerate(list(m.MODELS_PARAMS_B)[:3])]
    if case == 'empty': anchors = []
    if case == 'two_eligible': anchors = anchors[:2]
    if case == 'constant_size':
        monkeypatch.setattr(m, 'MODELS_PARAMS_B', {a['name']: 8. for a in anchors})
    (tmp_path/'scaling_law_iter109_meta.json').write_text(json.dumps({'anchors': anchors}))
    m.main()
    result = json.loads((tmp_path/'scaling_law_iter109b_meta.json').read_text())
    for name in ('lambda_vs_N', 'R_inf_vs_N'):
        test = result['E_permutation_test'][name]
        assert test['status'] == 'not_evaluable'
        assert np.isnan(test['p_two_sided'])
        assert test['rejects_H0_at_0.05'] is False


def test_offset_comparison_optimizer_failure_is_unavailable(monkeypatch):
    m = importlib.import_module('scaling_law_iter137')
    def fail(*args, **kwargs): raise RuntimeError('optimizer failed')
    monkeypatch.setattr(m, 'curve_fit', fail)
    result = m.fit_2p(np.arange(1., 11.), np.linspace(.2, .6, 10))
    assert result['status'] == 'fit_failed'
    assert np.isnan(result['aicc'])
    assert np.isnan(result['t_80'])
    assert m.comparison_winner(result, {'aicc': 1.}) == 'not_evaluable'
    assert m.comparison_winner({'aicc': 1.}, result) == 'not_evaluable'
    assert m.comparison_winner(result, result) == 'not_evaluable'
    assert m.comparison_winner({'aicc': 2.}, {'aicc': 1.}) == '3p'


def test_versioned_scaling_roots_bind_hybrid_sources():
    root = SCRIPTS.parents[1]/'platform_hybrid'
    for n in ('109', '109b', '121', '125', '129', '133', '137'):
        m = importlib.import_module('scaling_law_iter'+n)
        output = getattr(m, 'RES', getattr(m, 'RESULTS_DIR', None))
        assert output == root/'experiments/results/review26_v2'
        if n != '109b':
            traces = getattr(m, 'TR', getattr(m, 'TRACE_DIR', None))
            assert traces == root/'experiments/tinker-runs/results'
        tree = ast.parse((SCRIPTS/f'scaling_law_iter{n}.py').read_text())
        assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'mkdir'
                       for node in tree.body)


def test_canonical_scaling_root_and_candidate_output_isolation():
    m = importlib.import_module('scaling_law_fit')
    root = SCRIPTS.parents[1]/'platform_hybrid'
    assert m.TRACE_DIR == root/'experiments/tinker-runs/results'
    assert m.RESULTS_DIR == root/'experiments/results/review26_v2'
    assert m.FIG_DIR == root/'figures/review26_v2'
    tree = ast.parse((SCRIPTS/'scaling_law_fit.py').read_text())
    assert not any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'mkdir'
                   for node in tree.body)
