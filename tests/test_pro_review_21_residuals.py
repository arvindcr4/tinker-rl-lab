"""Offline synthetic regressions; never fit models or update experiment artifacts."""
import ast
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from test_pro_review_19_regressions import functions, SCRIPTS


def test_screening_is_charged_on_every_row_and_no_catch_is_unavailable():
    ns = functions('p8_iter116_realistic_llm_cost_sweep')
    assert ns['cost_at_llm'](100, 10, 2, .01, .01) == pytest.approx((.011, .55))
    cpd, cpf = ns['cost_at_llm'](100, 10, 0, .01, .01)
    assert cpd == pytest.approx(.011)
    assert cpf is None
    assert ns['cost_delta'](None, None) is None
    with pytest.raises(ValueError, match="invalid actual cohort"):
        ns['cost_at_llm'](0, 0, 0, .01, .01)


def test_seed_cost_does_not_invent_hybrid_catches():
    ns = functions('p8_iter152_cost_seed_stability', np=np, COST_XGB=.01, G_THR=.001)
    values = ns['compute_cell'](.01, 100, 2, 10)
    assert values[:3] == pytest.approx((.01, .011, .5))
    assert values[3] is None
    assert values[4] == pytest.approx(1.1)
    assert ns['compute_cell'](.01, 100, 2, 10, caught_hybrid=4)[3] == pytest.approx(.275)
    assert ns['compute_cell'](.01, 100, 0, 10)[2:4] == (None, None)


def test_iter116_consumer_requires_corrected_provenance_and_actual_size(tmp_path):
    rules = ['xgb_only', 'gradient_band', 'absolute_band']
    cells = [dict(rate=.0144, tree='XGB-24full', rule=r, n_test=99, n_pos=1,
                  n_llm_calls=0 if r == 'xgb_only' else 1, n_pos_caught_at_K=0,
                  outcome_kind='composed_XGB24_feature_proxy') for r in rules]
    ns = functions('p8_iter116_realistic_llm_cost_sweep', RES=tmp_path, csv=csv, json=json,
                   COST_XGB=.0001, COST_LLM_SWEEP=[.001, .10], RATES=[.0144], TREES=['XGB-24full'], RULES=rules)
    ns['load'] = lambda path: cells
    ns['main']()
    with (tmp_path / 'p8_iter116_cost_llm_sweep.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert all(int(r['n_test_eff']) == 99 for r in rows)
    assert all(r['cpf_usd'] == '' and r['cost_per_catch_status'] == 'no_catches' for r in rows)
    with (tmp_path / 'p8_iter116_best_rule_per_cell.tsv').open() as f:
        assert all(r['best_cpf_rule'] == '' for r in csv.DictReader(f, delimiter='\t'))
    cells[0].pop('outcome_kind')
    with pytest.raises(ValueError, match='regenerate iter112'):
        ns['main']()


def test_sampling_counts_use_retained_negatives_and_preserve_release_rows():
    y = np.array([1] * 50 + [0] * 100)
    X = np.arange(len(y))[:, None]
    ns = functions('p8_iter160_operating_point_utility', np=np, COST_BUDGET=1, BOOT_SEED=42)
    xs, ys = ns['downsample'](X, y, 20, 4)
    assert ys.sum() == 25
    assert len(ys) == 125
    assert len(np.unique(xs)) == len(xs)
    ns = functions('p8_iter200_base_rate_stress', np=np)
    xs, ys, ids = ns['make_subsample'](X, y, y.mean(), np.random.default_rng(4))
    np.testing.assert_array_equal(ids, np.arange(len(y)))
    np.testing.assert_array_equal(xs, X)
    ns = functions('p8_pr_auc_realistic', np=np, RANDOM_SEED=42, BOOT_SEED=42)
    mask = ns['downsample_positives'](y, 0, np.random.default_rng(4))
    assert mask.sum() == 100
    assert not y[mask].any()


def test_pr_auc_main_only_draws_each_cohort_once():
    tree = ast.parse((SCRIPTS / 'p8_pr_auc_realistic.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'downsample_positives']
    assert len(calls) == 1
    source = ast.unparse(main)
    assert source.count('mask = retained_masks[tgt]') == 2
    assert 'retained_row_ids' in source


def test_disjoint_sparse_firing_cannot_pass_event_overlap():
    ns = functions('p8_iter140_sensor_feat_realistic', np=np, G_THR=.001)
    a = np.zeros(100, dtype=bool); b = a.copy()
    a[0] = True; b[1] = True
    assert (a == b).mean() == .98
    assert ns['event_overlap'](a, b) == (0, 0)
    assert ns['event_overlap'](a, a) == (1, 1)
    assert ns['event_overlap'](np.zeros(10), np.zeros(10)) == (None, None)


def test_importance_is_cohort_specific_and_tied_rank_is_undefined():
    class Model:
        def predict_proba(self, X):
            score = .5 * (X[:, 0] + X[:, 1])
            return np.column_stack([1 - score, score])
    ns = functions('p8_iter92_gain_attribution', np=np, pd=pd)
    x = np.tile([0., 1.], 50)
    a = np.column_stack([x, np.zeros(100)])
    b = a[:, ::-1]
    ia = ns['cohort_permutation_importance'](Model(), a, x, ['a', 'b'], 42)
    ib = ns['cohort_permutation_importance'](Model(), b, x, ['a', 'b'], 42)
    assert ia['a'] > ia['b']
    assert ib['b'] > ib['a']
    assert ns['spearman_rho'](list(ia.values()), list(ib.values())) == -1
    assert ns['spearman_rho']([0, 0], [0, 0]) is None


def test_summary_guards_and_stable_seed_paths():
    for name in ['p8_subgroup_alert_fairness', 'p8_single_sensor_ablation']:
        tree = ast.parse((SCRIPTS / (name + '.py')).read_text())
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'hash' for n in ast.walk(tree))
    for name in ['p8_operational_calibration', 'p8_subgroup_alert_fairness']:
        assert 'sorted(df_b["pair"].unique())' in (SCRIPTS / (name + '.py')).read_text()
    for name in ['p8_threshold_calibration', 'p8_threshold_transfer']:
        source = (SCRIPTS / (name + '.py')).read_text()
        assert '"current_result"' in source
        assert '"falsifiable_claim"' not in source
    noise = (SCRIPTS / 'p8_iter96_per_cohort_noise_audit.py').read_text()
    assert noise.count('[sig for (a, s), sig in first_unfav.items()') == 2
    ablation = (SCRIPTS / 'p8_single_sensor_ablation.py').read_text()
    assert 'r["n_features"] == len(RAW20) + 1' in ablation
    assert '"n_features": len(feats)' in ablation
    assert '"excludes_zero": "undefined_no_catches"' in ablation
    frontier = (SCRIPTS / 'p8_iter88_noise_cost_frontier.py').read_text()
    assert 'cost_at_recall=front[R],' in frontier
    assert 'if c24 is not None and c20 is not None:' in frontier


def test_ablation_infinite_cost_bootstrap_is_unavailable_not_a_tie():
    tree = ast.parse((SCRIPTS / 'p8_single_sensor_ablation.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and 'rng = random.Random' in ast.unparse(n))
    ns = functions('p8_single_sensor_ablation', hashlib=hashlib, N_BOOT=20)
    ns.update(seed=42, K_BUDGETS=[50], base_models=['XGB-24full', 'XGB-20raw'],
              scores={'XGB-24full': ([.9, .1], .01), 'XGB-20raw': ([.1, .9], .01)},
              te_labels=[0, 0], cost_boot=[])
    exec(compile(ast.Module(body=[loop], type_ignores=[]), '<bootstrap fixture>', 'exec'), ns)  # noqa: S102 - Execute checked-in analysis blocks with synthetic fixtures only.
    row = ns['cost_boot'][0]
    assert row['ci_lo'] is None
    assert row['ci_hi'] is None
    assert row['delta_cost_per_caught'] is None
    assert row['excludes_zero'] == 'undefined_no_catches'


def test_sigma_and_pair_summaries_reconcile_detail_rows():
    tree = ast.parse((SCRIPTS / 'p8_iter96_per_cohort_noise_audit.py').read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in {'amount_q0_unfav_sigma', 'vmean_q2_unfav_sigma'}]
    ns = {'first_unfav': {('Amount_q', 0): .25, ('V_mean_q', 2): .5}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<summary fixture>', 'exec'), ns)  # noqa: S102 - Execute checked-in analysis blocks with synthetic fixtures only.
    assert ns['amount_q0_unfav_sigma'] == [.25]
    assert ns['vmean_q2_unfav_sigma'] == [.5]
    df = pd.DataFrame({'pair': ['XGB-24full-XGB-20raw', 'XGB-24full-XGB-20raw'],
                       'excludes_zero': ['YES', 'NO']})
    for name in ['p8_operational_calibration', 'p8_subgroup_alert_fairness']:
        tree = ast.parse((SCRIPTS / (name + '.py')).read_text())
        comp = next(n for n in ast.walk(tree) if isinstance(n, ast.DictComp)
                    and 'df_b' in ast.unparse(n) and 'excludes_zero' in ast.unparse(n))
        result = eval(compile(ast.Expression(comp), '<pair fixture>', 'eval'), {'df_b': df})  # noqa: S307 - Evaluate a checked-in summary expression against a synthetic DataFrame.
        assert result == {'XGB-24full-XGB-20raw': 1}


def test_vstd_hypothesis_uses_actual_strata_and_direction():
    tree = ast.parse((SCRIPTS / 'p8_iter192_vmean_decile_audit.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'std_edges')
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'h2')
    full = np.full(100, .5); full[70:] = 0
    ns = dict(np=np, v_std=np.arange(100), N_DECILES=10, SEEDS=[42], yte=np.zeros(100),
              probas={('24full', 42): full, ('20raw', 42): np.full(100, .5)})
    exec(compile(ast.Module(body=main.body[start:end + 1], type_ignores=[]), '<vstd fixture>', 'exec'), ns)  # noqa: S102 - Execute checked-in analysis blocks with synthetic fixtures only.
    assert ns['h2'] is True
    assert ns['std_gaps'] == {'low': 0, 'high': -.25}
