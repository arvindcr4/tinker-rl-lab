"""Exercise the v2 runner's pure statistics without importing its Modal client."""

import ast
import itertools
import math
from pathlib import Path

import pytest
from scipy import stats


@pytest.fixture(scope="module")
def helpers():
    path = (
        Path(__file__).resolve().parents[1]
        / "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
    )
    names = {"_t_cdf", "_t_ppf", "signflip_p", "paired"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MARGIN" for target in node.targets
            )
        )
    ]
    namespace = {"math": math, "itertools": itertools}
    # Only this repository's selected pure definitions; no imports or launch code.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("df", [1, 4, 30, 60])
@pytest.mark.parametrize("t", [-1e-8, -1e-16, 0.0, 1e-16, 1e-8])
def test_t_cdf_near_zero_matches_scipy(helpers, df, t):
    # The df=1 distribution has an exact Cauchy CDF. SciPy's t implementation
    # itself loses about 1e-9 near zero at df=1 in the locked SciPy release.
    expected = 0.5 + math.atan(t) / math.pi if df == 1 else stats.t.cdf(t, df)
    assert helpers["_t_cdf"](t, df) == pytest.approx(expected, abs=1e-12)


def test_paired_roundoff_near_zero_mean_does_not_crash(helpers):
    diffs = [0.078, 0.052, -0.086, 0.072, -0.043, -0.073]
    out = helpers["paired"](dict(enumerate(diffs)), dict.fromkeys(range(len(diffs)), 0.0))
    assert out["p_two_sided"] == pytest.approx(1.0)


@pytest.mark.parametrize("effect", [0.0, 0.125, -0.125])
def test_paired_zero_variance_has_valid_difference_and_equivalence_tests(helpers, effect):
    out = helpers["paired"](dict.fromkeys(range(5), effect), dict.fromkeys(range(5), 0.0))
    assert out["p_two_sided"] == (1.0 if effect == 0 else 0.0)
    assert out["ci95"] == [effect, effect]
    assert out["equivalent_at_margin"] is (abs(effect) < helpers["MARGIN"])
    if effect == 0:
        assert out["t"] == 0.0
    else:
        assert out["t"] == math.copysign(math.inf, effect)


@pytest.mark.parametrize("direction", [-1, 1])
def test_paired_at_equivalence_boundary_is_not_equivalent(helpers, direction):
    effect = direction * helpers["MARGIN"]
    out = helpers["paired"](dict.fromkeys(range(2), effect), dict.fromkeys(range(2), 0.0))
    assert out["tost_p"] == 0.5
    assert out["equivalent_at_margin"] is False


@pytest.mark.parametrize(("a", "b"), [({}, {}), ({1: 0.5}, {1: 0.5}), ({1: 0.5}, {2: 0.5})])
def test_paired_requires_two_shared_seeds(helpers, a, b):
    with pytest.raises(ValueError, match="at least two shared seeds"):
        helpers["paired"](a, b)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_paired_rejects_nonfinite_inputs_before_significance(helpers, value):
    with pytest.raises(ValueError, match="finite paired differences"):
        helpers["paired"]({1: value, 2: 0.5}, {1: 0.0, 2: 0.5})
