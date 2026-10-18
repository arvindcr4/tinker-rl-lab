"""Synthetic consumer-path coverage; never replay stored research evidence."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'platform_modal/scripts/p5p8'
sys.path.insert(0, str(SCRIPTS))


def module(iteration):
    return importlib.import_module(next(SCRIPTS.glob(f'p7_iter{iteration}_*.py')).stem)


def test_salvage_through_replay_and_aggregate():
    m = module(115)
    for rule in m.RULES:
        row = m.replay_seed([{'zvf': .7}], .65, rule)[0]
        assert row['G_used'] == 16
        assert row['salvage'] == 1
        assert m.seed_summary([row])['n_salvaged'] == 1
    z = .9  # requires G64 to meet .5 target
    assert m.decide_step(z)['Gstar'] == 64
    assert m.replay_seed([{'zvf': z}], .65, 'ADAPTIVE_GSTAR')[0]['salvage'] == 1
    assert m.decide_step(1)['target_status'] == 'NO_FEASIBLE_G'
    assert m.decide_step(1)['zvf_at_Gstar'] == 1
    assert m.decide_step(1)['salvageable'] == 0


def test_concordance_main_unavailable_and_diagnostics(tmp_path, monkeypatch):
    m = module(143)
    monkeypatch.setattr(m, 'N10_DIR', tmp_path)
    monkeypatch.setattr(m, 'OUT_DIR', tmp_path)
    for seed in m.SEED_ORDER:
        (tmp_path / f'n10_grpo_s{seed}.json').write_text(json.dumps({'step_log': [
            {'step': step, 'zvf': 0.1} for step in reversed(range(m.N_STEPS))]}))
    m.main()
    result = json.loads((tmp_path / 'p7_iter143_summary.json').read_text())
    assert result['best_tau'] is None
    assert result['H1_mean_kappa_at_canonical_ge_040']['pass'] is None
    diag = result['pair_diagnostics'][0]
    assert diag['kappa'] is None and diag['raw_agreement'] == 1
    assert diag['prevalence_i'] == 0
    assert m.summarize_kappas({'a': -.2, 'b': -.8})['sd'] == pytest.approx(.3)
    assert m.summarize_kappas({'a': -.2, 'b': -.8})['cv'] is None
    path = tmp_path / f'n10_grpo_s{m.SEED_ORDER[0]}.json'
    path.write_text(json.dumps({'step_log': [{'step': 0, 'zvf': .5}]}))
    with pytest.raises(ValueError, match='expected 15 steps'):
        m.load_seeds()
    path.write_text(json.dumps({'step_log': [{'step': 0, 'zvf': .5}] * m.N_STEPS}))
    with pytest.raises(ValueError, match='duplicate step'):
        m.load_seeds()


def corpus(m):
    return [{'method': method, 'step': i, 'prompt_idx': 0, 'k': round(p * 8),
             'p_hat': p, 'step_zvf': .8}
            for method in m.METHODS for i, p in enumerate([.125, .5, .875])]


def test_oracle167_consumers_share_yield_objective():
    m = module(167)
    obs = m.compute_obs_table(corpus(m))
    for row in obs:
        assert row['oracle_score'] == row['oracle_dY']
        assert row['oracle_dY'] == max(m.per_prompt_delta_y(row['p_hat'], g) for g in m.G_MENU)
    for row in m.per_method_aggregate(obs):
        assert row['regret_abs_dY'] >= 0
        assert row['pct_oracle_abs_captured'] <= 100
    for row in m.bootstrap_axes(obs, B=40):
        assert row['pct_abs_ci95_hi'] <= 100


def test_oracle175_and_c6_target_consumers():
    m = module(175)
    rows = m.obs_table(corpus(m))
    fallback = m.obs_table([dict(corpus(m)[0], p_hat=.5)])[0]
    assert fallback['g_c6'] == 16
    assert fallback['g_dualformer_target'] == 8
    no_fire = m.obs_table([dict(corpus(m)[0], p_hat=.8, step_zvf=.1)])[0]
    assert no_fire['g_c6'] == 8 and no_fire['g_dualformer_target'] == 16
    for row in m.per_method_summary(rows):
        assert row['pct_oracle_captured'] <= 100
    for row in m.bootstrap_pct_ci(rows, 'C6_calibrated', B=40):
        assert row['ci_hi'] <= 100


def test_synthetic195_main_retires_system_verdicts(tmp_path, monkeypatch):
    m = module(195)
    monkeypatch.setattr(m, 'N2_DIR', tmp_path)
    monkeypatch.setattr(m, 'OUT', tmp_path)
    monkeypatch.setattr(m, 'N_BOOT', 40)
    for method in m.METHODS:
        rows = [{'method': method, 'step': i, 'zvf': .8 if i % 2 else .1,
                 'reward_mean': .5, 'rewards': [[0, 1] * 4] if i % 2 else [[0] * 8]}
                for i in range(10)]
        (tmp_path / f'{method}_s0_tensors.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        assert {m.step_decisions(row)['df'] for row in rows} == {0, 1}
        assert {m.step_decisions(row)['ap'] for row in rows} == {0}
    m.main()
    result = json.loads((tmp_path / 'p7_iter195_summary.json').read_text())
    assert result['verdict_status'] == 'NOT_EVALUATED_SYNTHETIC_DIAGNOSTICS'
    assert all(value is None for value in result['verdicts'].values())
    assert result['rule_identity']['ap'] == 'always_off_impossible_inequality'


@pytest.mark.parametrize('iteration', [123, 127])
def test_rng_stream_independent_of_python_hash_seed(iteration):
    name = next(SCRIPTS.glob(f'p7_iter{iteration}_*.py')).stem
    script = f'import {name} as m; import random; print([random.Random(m.SEED + m.stable_seed(x) % 10000).random() for x in ["aero", "grpo"]])'
    values = [subprocess.check_output([sys.executable, '-c', script], env={**os.environ,
              'PYTHONPATH': str(SCRIPTS), 'PYTHONHASHSEED': seed}, text=True) for seed in ['1', '987']]
    assert values[0] == values[1]
    assert 'hash(m)' not in (SCRIPTS / (name + '.py')).read_text()
