"""Synthetic-only regression tests. Extract pure functions to avoid legacy import I/O."""
import ast
import math
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'platform_modal/scripts/berkeley'


def functions(filename, *names):
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {'math': math, 'Path': Path}
    exec(compile(tree, filename, 'exec'), ns)
    return ns


def test_sign_tails_and_ties():
    sign = functions('alphaproof_mcts_zvf.py', 'sign_test')['sign_test']
    assert sign([{'a': x, 'b': 0} for x in [1, 1, 1, 1, -1]], 'a', 'b')['binom_p'] == .375
    assert sign([{'a': 0, 'b': 0}], 'a', 'b')['binom_p'] == 1
    with pytest.raises(ValueError):
        sign([{'a': math.nan, 'b': 0}], 'a', 'b')


def test_verifier_subsample_and_absolute_expectation():
    ns = functions('cove_factored_baseline.py', 'hyper_pmf', 'verifier_residual')
    f = ns['verifier_residual']
    for n in [4, 8]:
        for k in range(n+1):
            for g in [2, 4]:
                assert f(n, k, g, 0) == pytest.approx(0)
    # Balanced peers have zero expected signed noise but positive expected absolute noise.
    assert f(4, 2, 4, .5) > 0
    assert f(2, 1, 2, .25) == pytest.approx(.25)


def test_ols_independent_coefficients():
    fit = functions('eureka_reward_design_quality.py', '_fit_ols')['_fit_ols']
    x = [[0,0], [1,0], [0,1], [1,1], [2,1], [1,2]]
    y = [3 + 2*a - 5*b for a,b in x]
    result = fit(x, y)
    assert result['rss'] < 1e-20
    assert result['slope'] == pytest.approx([2,-5])
    assert result['k'] == 4


def test_exact_wilcoxon_and_sign_direction():
    ns = functions('cdh_echo_synthesis.py', 'paired_wilcoxon', 'sign_test_binom')
    assert ns['paired_wilcoxon']([1]*5, [0]*5)[2] == 1/32
    assert ns['paired_wilcoxon']([0]*5, [1]*5)[2] == 1
    assert ns['paired_wilcoxon']([0], [0])[2] == 1
    assert ns['sign_test_binom'](4,5) == 6/32
    assert ns['sign_test_binom'](0,5) == 1


def test_no_synthetic_reproduction():
    ns = functions('paper2verifier.py', '_sample_iter127_points', '_sample_iter135_points')
    for fn in ns.values():
        if callable(fn) and getattr(fn, '__name__', '').startswith('_sample'):
            with pytest.raises(ValueError, match='MISSING_MEASURED_DATA'):
                fn()


def test_loader_rejects_malformed_rewards(tmp_path):
    import json
    ns = functions('cot_decoding_zvf_substitution.py', 'load_gsm8k')
    ns.update(WORKTREE=tmp_path, G=8, json=json)
    folder = tmp_path/'experiments/results'
    folder.mkdir(parents=True)
    for rewards in ([1]*7, [1]*9, [1]*7+[math.nan], [1]*7+[.5]):
        (folder/'tinker_gsm8k_zvf_s1.json').write_text(json.dumps({'per_problem':[{'rewards':rewards}]}))
        with pytest.raises(ValueError):
            ns['load_gsm8k']('s1')


def test_corrected_policy_is_tested():
    ns = functions('iso_g_corrected_synthesis.py', 'h5_practitioner_iso_g')
    ns.update(ROOT=Path('.'), _read_tsv=lambda path: [{'T': '1', 'G_opt': '4'}])
    # Static regression also catches accidentally reintroducing an unrelated floor.
    source = (ROOT/'iso_g_corrected_synthesis.py').read_text()
    assert 'G_iso_floor = G_iso_corrected' in source


def test_capability_labels_ordered():
    f = functions('cybench_capability_tiers.py', 'cybench_tier')['cybench_tier']
    assert [f(x) for x in [.1,.3,.6,.9]] == ['C1_Low','C2_Lower','C3_Upper','C4_High']
