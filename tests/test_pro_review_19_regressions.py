"""CPU-only synthetic checks; load functions without executing analysis entry points."""
import ast
import math
import random
import statistics
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'platform_modal/scripts/p5p8'


def functions(name, **constants):
    tree = ast.parse((SCRIPTS / (name + '.py')).read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(math=math, random=random, statistics=statistics, N_BOOT=2000,
              B=2000, CI=.95, SEED=42, G_BASE=8, G_NEW=16, G_OBS=8,
              N_PROMPTS=16, TOST_BOUND=.05)
    ns.update(constants)
    exec(compile(tree, name, 'exec', flags=__import__('__future__').annotations.compiler_flag), ns)
    return ns


@pytest.mark.parametrize('name', ['p7_iter111_target_g_selection', 'p7_iter131_per_prompt_adaptive_gstar_simulation'])
def test_small_bootstrap_has_variation(name):
    ns = functions(name)
    mean, lo, hi = ns['bootstrap_ci']([0, 1])
    assert mean == .5 and lo < mean < hi


def test_four_method_bootstrap_has_variation():
    ns = functions('p7_iter159_pareto_permethod_ci')
    mean, lo, hi = ns['bootstrap_ci']([0, 1, 2, 3], 2000, 42)
    assert lo < mean < hi
    assert ns['paired_bootstrap_ci']([0]*4, 100, 42) == (0, 0, 0)


def test_append_restoration():
    lb = lambda a,b: math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    ns = functions('p7_costeff_boot', betaln=lb)
    assert ns['restore_prob'](0) == pytest.approx(8/17)
    assert ns['restore_prob'](8) == pytest.approx(8/17)
    assert ns['restore_prob'](4) == 0


def test_inverse_roundtrip_and_unsupported_domain():
    ns = functions('p7_iter123_headline_cis', BOOT=2000, RNG_SEED=42)
    for z in [.8, .05, 1, 2**-7]:
        assert ns['z_bernoulli'](ns['invert_p'](z), 8) == pytest.approx(z)
    with pytest.raises(ValueError):
        ns['invert_p'](.001)


def test_identity_gain_and_real_sample_count():
    ns = functions('p7_iter171_headline_cis')
    s = {'rewards': [[0]*4+[1]*4]*16}
    assert ns['eval_ctrl']([s], lambda s, p: [8]*len(p))[3] == [0]
    assert ns['_bci']([0,1,2])[3] == 3
