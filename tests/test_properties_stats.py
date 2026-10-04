"""Property-based tests for the statistical helpers the thesis numbers rest on.

Covers ``utils.stats`` (bootstrap CI, IQM, Welch, Mann-Whitney), the exact
McNemar in ``tools/check_thesis_evidence.py``, the Wilson intervals used by
the p5p8 headline scripts, the multiplicity corrections (Bonferroni/BH in
``compute_statistics.py``, BH in ``partial_correlation_zvf.py``, Holm in the
same-stack v2 runner) and the sign-flip / paired-t helpers.

Modules are imported read-only. The same-stack v2 runner imports ``modal`` at
module scope, so its pure helpers are lifted out by AST instead of importing
the module (no Modal client, no network).
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given
from hypothesis import strategies as st

from tests._shared_fakes import fast_settings
from tools import check_thesis_evidence as evidence
from utils import stats

ROOT = Path(__file__).resolve().parents[1]
FAST = fast_settings()


def _load_path(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    # compute_statistics.py calls warnings.filterwarnings("ignore") at import;
    # contain it so it cannot silence warnings for the rest of the session.
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def _lift_functions(rel: str, names: tuple[str, ...], consts: tuple[str, ...] = ()) -> dict:
    """Exec only the named top-level defs/constants of a module (no imports run)."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in consts for t in node.targets)
        )
    ]
    found = {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    missing = set(names) - found
    if missing:
        pytest.skip(f"{rel}: helpers not found: {sorted(missing)}")
    namespace: dict = {"math": math, "itertools": itertools}
    # Runs this repo's own helper defs extracted from a Modal script (importing it
    # would need the modal package); never untrusted input.
    exec(compile(ast.Module(body=keep, type_ignores=[]), rel, "exec"), namespace)  # noqa: S102
    return namespace


compute_statistics = _load_path(
    "_prop_compute_statistics", "platform_hybrid/experiments/compute_statistics.py"
)
partial_corr = _load_path("_prop_partial_corr", "platform_modal/scripts/partial_correlation_zvf.py")
# The p5p8 scripts mkdir a hardcoded absolute RES path at import; lift the
# pure helper instead of importing the module.
wilson_181 = _lift_functions(
    "platform_modal/scripts/p5p8/p5_iter181_v25_schema_rollout.py", ("wilson",)
)["wilson"]
wilson_185 = _lift_functions(
    "platform_modal/scripts/p5p8/p5_iter185_v25_cross_corpus.py", ("wilson",)
)["wilson"]
signflip_mean = _load_path("_prop_lb24", "platform_modal/scripts/length_bias_iter24.py").signflip_p
V2 = _lift_functions(
    "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py",
    ("_t_cdf", "_t_ppf", "signflip_p", "paired", "holm"),
    consts=("MARGIN",),
)

# --------------------------------------------------------------------------- strategies

scores = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=12,
)
unit_scores = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=12)
pvals = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=15)
k_n = st.integers(min_value=1, max_value=2000).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)
# Paired accuracy differences live on a k/1000 grid; the sign-flip tie
# tolerance is an absolute 1e-12 (on the sum in v2, on the mean in iter24), so
# the two implementations can disagree for sub-1e-12 structure.
small_diffs = st.lists(st.integers(-1000, 1000).map(lambda k: k / 1000), min_size=2, max_size=8)


def _spread(xs) -> bool:
    return float(np.std(xs)) > 1e-6 * (1.0 + float(np.max(np.abs(xs))))


# --------------------------------------------------------------------------- Wilson


@FAST
@given(k_n)
def test_wilson_bounds_in_unit_interval_and_contain_phat(kn):
    k, n = kn
    for wilson in (wilson_181, wilson_185):
        lo, p, hi = wilson(k, n)
        assert p == k / n
        assert 0.0 <= lo <= hi <= 1.0
        # Analytically lo <= p <= hi; allow float rounding at k in {0, n}.
        assert lo <= p + 1e-12
        assert p <= hi + 1e-12


@FAST
@given(k_n)
def test_wilson_reflection_symmetry_and_implementations_agree(kn):
    k, n = kn
    lo, p, hi = wilson_181(k, n)
    lo_r, p_r, hi_r = wilson_181(n - k, n)
    assert math.isclose(lo, 1.0 - hi_r, abs_tol=1e-12)
    assert math.isclose(hi, 1.0 - lo_r, abs_tol=1e-12)
    assert math.isclose(p, 1.0 - p_r, abs_tol=1e-12)
    assert wilson_185(k, n) == (lo, p, hi)


# --------------------------------------------------------------------------- McNemar


paired_binary = st.integers(min_value=1, max_value=40).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
    )
)


@FAST
@given(paired_binary)
def test_mcnemar_exact_p_range_counts_and_swap_symmetry(pair):
    trained, base = pair
    b, c, p = evidence.mcnemar(trained, base)
    assert b == sum(t == 1 and a == 0 for t, a in zip(trained, base))
    assert c == sum(t == 0 and a == 1 for t, a in zip(trained, base))
    assert 0.0 < p <= 1.0
    b2, c2, p2 = evidence.mcnemar(base, trained)
    assert (b2, c2) == (c, b)
    assert p2 == p
    if b == c:
        assert p == 1.0


# --------------------------------------------------------------------------- sign-flip / paired t


@FAST
@given(small_diffs)
def test_signflip_p_range_floor_and_sign_symmetry(d):
    n = len(d)
    p_sum = V2["signflip_p"](d)
    p_mean = signflip_mean(d)
    assert 2.0 / 2**n - 1e-15 <= p_sum <= 1.0
    assert p_sum == V2["signflip_p"]([-x for x in d])
    # Sum- and mean-statistic implementations are the same test.
    assert math.isclose(p_sum, p_mean, abs_tol=1e-12)


@FAST
@given(st.integers(min_value=2, max_value=8))
def test_signflip_all_zero_differences_is_one(n):
    assert V2["signflip_p"]([0.0] * n) == 1.0


@FAST
@given(
    st.floats(min_value=-30.0, max_value=30.0, allow_nan=False),
    st.integers(min_value=1, max_value=60),
)
def test_scipy_free_t_cdf_matches_scipy(t, df):
    from scipy import stats as sps

    assert math.isclose(V2["_t_cdf"](t, df), float(sps.t.cdf(t, df)), abs_tol=1e-8)


@FAST
@given(st.floats(min_value=0.6, max_value=0.99), st.integers(min_value=1, max_value=60))
def test_t_ppf_inverts_t_cdf(p, df):
    assert math.isclose(V2["_t_cdf"](V2["_t_ppf"](p, df), df), p, abs_tol=1e-7)


@FAST
@given(small_diffs)
def test_paired_p_in_unit_interval_and_ci_contains_mean(d):
    assume(_spread(d))
    a = {i: x for i, x in enumerate(d)}
    b = dict.fromkeys(range(len(d)), 0.0)
    out = V2["paired"](a, b)
    assert 0.0 <= out["p_two_sided"] <= 1.0
    lo, hi = out["ci95"]
    assert lo <= out["mean_diff"] <= hi
    assert 0.0 <= out["tost_p"] <= 1.0


def test_paired_float_noise_mean_does_not_crash():
    d = [0.078, 0.052, -0.086, 0.072, -0.043, -0.073]  # mean ~1.16e-18, not 0.0
    out = V2["paired"](dict(enumerate(d)), dict.fromkeys(range(len(d)), 0.0))
    assert out["p_two_sided"] == pytest.approx(1.0)


def test_paired_identical_arms_is_not_significant():
    a = {s: 0.5 for s in range(5)}
    out = V2["paired"](a, dict(a))
    assert out["p_two_sided"] >= 0.05


# --------------------------------------------------------------------------- multiplicity


@FAST
@given(pvals)
def test_holm_dominates_raw_monotone_and_bounded(ps):
    named = {f"h{i}": p for i, p in enumerate(ps)}
    adj = V2["holm"](named)
    bonf = compute_statistics.bonferroni(ps)
    order = sorted(named, key=named.get)
    for i, key in enumerate(order):
        assert named[key] <= adj[key] <= 1.0
        assert adj[key] <= bonf[int(key[1:])] + 1e-15
        if i:
            assert adj[order[i - 1]] <= adj[key]


@FAST
@given(pvals)
def test_bh_bounded_monotone_and_below_holm(ps):
    bh = compute_statistics.benjamini_hochberg(ps)
    holm = V2["holm"]({i: p for i, p in enumerate(ps)})
    order = np.argsort(ps, kind="mergesort")
    for i, idx in enumerate(order):
        assert ps[idx] <= bh[idx] + 1e-15
        assert bh[idx] <= 1.0
        assert bh[idx] <= holm[int(idx)] + 1e-12
        if i:
            assert bh[order[i - 1]] <= bh[idx] + 1e-15


@FAST
@given(pvals)
def test_bh_implementations_agree(ps):
    a = compute_statistics.benjamini_hochberg(ps)
    b = partial_corr._bh_adjust(ps)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- utils.stats


@FAST
@given(scores, st.floats(min_value=-100.0, max_value=100.0))
def test_bootstrap_ci_ordered_within_range_and_shift_equivariant(xs, shift):
    x = np.asarray(xs)
    mean, lo, hi = stats.compute_bootstrap_ci(x, n_bootstrap=200, rng=np.random.default_rng(0))
    assert math.isclose(mean, float(np.mean(x)), rel_tol=1e-12, abs_tol=1e-9)
    tol = 1e-9 * (1.0 + float(np.max(np.abs(x))))
    assert x.min() - tol <= lo <= hi <= x.max() + tol
    m2, lo2, hi2 = stats.compute_bootstrap_ci(
        x + shift, n_bootstrap=200, rng=np.random.default_rng(0)
    )
    tol = 1e-6 * (1.0 + abs(shift) + float(np.max(np.abs(x))))
    assert math.isclose(lo2, lo + shift, abs_tol=tol)
    assert math.isclose(hi2, hi + shift, abs_tol=tol)


@FAST
@given(scores, st.floats(min_value=0.0, max_value=0.49))
def test_iqm_within_range_and_constant_fixed_point(xs, tau):
    x = np.asarray(xs)
    iqm = stats.compute_iqm(x, tau=tau)
    tol = 1e-9 * (1.0 + float(np.max(np.abs(x))))
    assert x.min() - tol <= iqm <= x.max() + tol
    assert stats.compute_iqm(np.full(len(xs), xs[0]), tau=tau) == pytest.approx(xs[0])


@FAST
@given(unit_scores, unit_scores)
def test_welch_p_range_and_swap_antisymmetry(a, b):
    assume(_spread(a) and _spread(b))
    ab = stats.welch_ttest(a, b)
    ba = stats.welch_ttest(b, a)
    assert 0.0 <= ab["p_value"] <= 1.0
    assert math.isclose(ab["p_value"], ba["p_value"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(ab["t_statistic"], -ba["t_statistic"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(
        ab["effect_size_cohens_d"], -ba["effect_size_cohens_d"], rel_tol=1e-9, abs_tol=1e-12
    )


@FAST
@given(unit_scores, unit_scores)
def test_mann_whitney_p_range_and_swap_symmetry(a, b):
    ab = stats.mann_whitney_u(a, b)
    ba = stats.mann_whitney_u(b, a)
    assert 0.0 <= ab["p_value"] <= 1.0
    assert math.isclose(ab["p_value"], ba["p_value"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(ab["u_statistic"] + ba["u_statistic"], len(a) * len(b))


def test_mann_whitney_all_tied_returns_finite_p():
    assert math.isfinite(stats.mann_whitney_u([0.5, 0.5], [0.5, 0.5])["p_value"])


@pytest.mark.parametrize("t", [0.0, 1e-20, -1e-20, 1e-9, -1e-9, float("inf"), -float("inf"), 1e200])
def test_t_cdf_rounding_and_infinite_boundaries(t):
    from scipy.stats import t as student_t

    assert V2["_t_cdf"](t, 5) == pytest.approx(student_t.cdf(t, 5), abs=1e-14)


@pytest.mark.parametrize("p", [0.001, 0.999])
def test_t_quantile_does_not_saturate_at_fifty(p):
    from scipy.stats import t as student_t

    assert V2["_t_ppf"](p, 1) == pytest.approx(student_t.ppf(p, 1), rel=1e-8)


@pytest.mark.parametrize("delta", [-0.1, -0.02, 0.0, 0.02, 0.1])
def test_constant_paired_difference_equivalence_limits(delta):
    margin = V2["MARGIN"]
    out = V2["paired"](dict.fromkeys(range(3), delta), dict.fromkeys(range(3), 0.0))
    assert out["p_two_sided"] == (1.0 if delta == 0 else 0.0)
    assert out["equivalent_at_margin"] == (abs(delta) < margin)
    assert out["ci95"] == [delta, delta]
    if delta:
        assert math.copysign(1, out["t"]) == math.copysign(1, delta)


@pytest.mark.parametrize("delta_sign", [-1, 1])
def test_constant_paired_difference_at_equivalence_margin(delta_sign):
    delta = delta_sign * V2["MARGIN"]
    out = V2["paired"]({0: delta, 1: delta}, {0: 0.0, 1: 0.0})
    assert out["tost_p"] == 0.5
    assert not out["equivalent_at_margin"]


@pytest.mark.parametrize(
    ("a", "b"), [({}, {}), ({0: 1}, {1: 1}), ({0: 1}, {0: 1}), ({0: 1, 1: math.nan}, {0: 1, 1: 1})]
)
def test_paired_invalid_samples_raise(a, b):
    with pytest.raises(ValueError, match="paired requires"):
        V2["paired"](a, b)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_mann_whitney_nonfinite_is_rejected(invalid):
    with pytest.raises(ValueError, match="finite scores"):
        stats.mann_whitney_u([invalid], [0.0])


@pytest.mark.parametrize(("n", "m"), [(1, 1), (2, 3), (10, 20)])
def test_mann_whitney_ties_have_midrank_u_and_unit_p(n, m):
    out = stats.mann_whitney_u([0.1] * n, [0.1] * m)
    assert out["u_statistic"] == n * m / 2
    assert out["p_value"] == 1.0
    assert not out["significant_at_005"]


@pytest.mark.parametrize(("a", "b"), [(0.1, 0.1), (0.1, 0.2), (0.2, 0.1)])
def test_welch_constant_arms_have_correct_null_and_effect(a, b):
    out = stats.welch_ttest([a] * 3, [b] * 5)
    assert out["p_value"] == (1.0 if a == b else 0.0)
    assert out["significant_at_005"] == (a != b)
    expected = math.copysign(math.inf, a - b) if a != b else 0.0
    assert out["t_statistic"] == expected
    assert out["effect_size_cohens_d"] == expected


@pytest.mark.parametrize("df", [0, -1, math.nan, math.inf])
def test_t_helpers_reject_invalid_degrees_of_freedom(df):
    with pytest.raises(ValueError, match="positive finite df"):
        V2["_t_cdf"](0.0, df)
    with pytest.raises(ValueError, match="positive finite df"):
        V2["_t_ppf"](0.975, df)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_t_quantile_rejects_invalid_probability(p):
    with pytest.raises(ValueError, match="0 < p < 1"):
        V2["_t_ppf"](p, 5)


@pytest.mark.parametrize("magnitude", [1e150, 1e160, 1e200, 1e300])
def test_cauchy_extreme_tail_survives_squared_argument_overflow(magnitude):
    expected = math.atan(1 / magnitude) / math.pi
    assert V2["_t_cdf"](-magnitude, 1) == pytest.approx(expected, rel=2e-13, abs=0)


@pytest.mark.parametrize("p", [1e-100, 1e-200, 1e-300, math.nextafter(1.0, 0.0)])
def test_cauchy_extreme_quantile_matches_analytical_tail(p):
    q = min(p, 1 - p)
    expected = (1 if p > 0.5 else -1) / math.tan(math.pi * q)
    result = V2["_t_ppf"](p, 1)
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=3e-13)


def test_unrepresentable_cauchy_quantile_terminates_with_infinity():
    assert V2["_t_ppf"](math.nextafter(0.0, 1.0), 1) == -math.inf
