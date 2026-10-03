"""Property-based tests for the GRPO advantage / loss helpers.

Read-only use of ``platform_tinker.tinkerrl``: group and batch-global
advantage normalization, degenerate-group detection, truncation masking,
GAE, and the GRPO / GSPO loss closures (response-token scoping).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("hypothesis")
torch = pytest.importorskip("torch")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from platform_tinker.tinkerrl import critic, grpo  # noqa: E402

FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

rewards = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=16,
)
binary_rewards = st.lists(st.sampled_from([0.0, 1.0]), min_size=2, max_size=16)


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# --------------------------------------------------------------------------- group advantages


@FAST
@given(rewards, st.booleans())
def test_group_advantages_are_zero_mean(rs, unbiased):
    # Near-constant groups: see test_constant_groups_are_degenerate_and_near_zero.
    assume(unbiased or _std(rs) > 1e-6)
    adv = grpo.normalize_rewards(rs, unbiased=unbiased)
    assert len(adv) == len(rs)
    scale = 1.0 if not unbiased else 1.0 + max(abs(r) for r in rs)
    assert abs(sum(adv)) <= 1e-9 * len(rs) * scale


@FAST
@given(rewards)
def test_standardized_advantages_have_unit_std(rs):
    assume(_std(rs) > 1e-3)
    adv = grpo.normalize_rewards(rs)
    assert math.isclose(_std(adv), 1.0, rel_tol=1e-5)


@FAST
@given(binary_rewards.filter(lambda r: len(set(r)) == 1), st.booleans())
def test_constant_binary_groups_give_exactly_zero_and_are_degenerate(rs, unbiased):
    assert grpo.normalize_rewards(rs, unbiased=unbiased) == [0.0] * len(rs)
    assert grpo.normalize_advantages_global(rs, unbiased=unbiased) == [0.0] * len(rs)
    assert grpo.is_degenerate_group(rs)


@FAST
@given(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    st.integers(min_value=2, max_value=16),
)
def test_constant_groups_are_degenerate_and_near_zero(value, n):
    rs = [value] * n
    assert grpo.is_degenerate_group(rs)
    # Standardized mode divides float-rounding residue by epsilon; bounded, not zero.
    assert all(abs(a) <= 1e-6 for a in grpo.normalize_rewards(rs))
    assert all(abs(a) <= 1e-12 for a in grpo.normalize_rewards(rs, unbiased=True))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING (grpo.normalize_rewards): a constant non-binary group is not "
        "exactly zero: sum(rs)/n rounds (0.1*3/3 != 0.1), and the standardized "
        "path divides the ~1e-17 residue by epsilon=1e-8 -> ~-1.4e-9 advantages. "
        "Harmless in magnitude; exact zero would need math.fsum or a degenerate "
        "short-circuit."
    ),
)
def test_constant_fractional_group_is_exactly_zero():
    assert grpo.normalize_rewards([0.1, 0.1, 0.1]) == [0.0, 0.0, 0.0]


@FAST
@given(
    rewards,
    st.floats(min_value=0.1, max_value=100.0),
    st.floats(min_value=-100.0, max_value=100.0),
)
def test_standardized_advantages_invariant_to_positive_affine_rewards(rs, a, b):
    assume(_std(rs) > 1e-2)
    base = grpo.normalize_rewards(rs)
    moved = grpo.normalize_rewards([a * r + b for r in rs])
    for x, y in zip(base, moved):
        assert math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5)


@FAST
@given(
    rewards,
    st.floats(min_value=0.1, max_value=100.0),
    st.floats(min_value=-100.0, max_value=100.0),
)
def test_unbiased_advantages_shift_invariant_scale_equivariant(rs, a, b):
    base = grpo.normalize_rewards(rs, unbiased=True)
    moved = grpo.normalize_rewards([a * r + b for r in rs], unbiased=True)
    tol = 1e-9 * (1.0 + abs(a) * 10.0 + abs(b))
    for x, y in zip(base, moved):
        assert math.isclose(a * x, y, rel_tol=1e-9, abs_tol=tol)


@FAST
@given(rewards)
def test_advantages_preserve_reward_order(rs):
    adv = grpo.normalize_rewards(rs)
    for i in range(len(rs)):
        for j in range(len(rs)):
            if rs[i] < rs[j]:
                assert adv[i] <= adv[j]


@FAST
@given(rewards, st.booleans())
def test_global_normalization_matches_group_for_single_group(rs, unbiased):
    g = grpo.normalize_advantages_global(rs, unbiased=unbiased)
    n = grpo.normalize_rewards(rs, unbiased=unbiased)
    for x, y in zip(g, n):
        assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)


@FAST
@given(rewards, st.data())
def test_global_exclude_keeps_alignment_and_zero_mean_basis(rs, data):
    exclude = data.draw(st.lists(st.booleans(), min_size=len(rs), max_size=len(rs)))
    adv = grpo.normalize_advantages_global(rs, unbiased=True, exclude=exclude)
    assert len(adv) == len(rs)
    basis = [a for a, drop in zip(adv, exclude) if not drop] or adv
    assert abs(sum(basis)) <= 1e-9 * len(rs) * (1.0 + max(abs(r) for r in rs))


@FAST
@given(rewards, st.data())
def test_truncation_mask_zeroes_exactly_the_truncated(rs, data):
    adv = grpo.normalize_rewards(rs)
    lens = data.draw(st.lists(st.integers(0, 64), min_size=len(rs), max_size=len(rs)))
    masked = grpo.apply_truncation_mask(adv, lens, max_tokens=32)
    for a, m, n in zip(adv, masked, lens):
        assert m == (0.0 if n >= 32 else a)


# --------------------------------------------------------------------------- GAE


@FAST
@given(st.integers(min_value=1, max_value=12), st.data())
def test_gae_returns_identity_and_lambda_one_is_discounted_return(t, data):
    floats = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
    rs = data.draw(st.lists(floats, min_size=t, max_size=t))
    vs = data.draw(st.lists(floats, min_size=t + 1, max_size=t + 1))
    gamma = data.draw(st.floats(min_value=0.0, max_value=1.0))
    dones = [False] * (t - 1) + [True]
    adv, ret = critic.compute_gae_advantages(rs, vs, dones, gamma=gamma, lam=1.0)
    for a, r, v in zip(adv, ret, vs):
        assert math.isclose(r, a + v, rel_tol=1e-12, abs_tol=1e-12)
    # lam=1 with a terminal at the end: return_t is the discounted reward-to-go.
    g = 0.0
    for step in reversed(range(t)):
        g = rs[step] + gamma * g
        assert math.isclose(ret[step], g, rel_tol=1e-9, abs_tol=1e-9)


# --------------------------------------------------------------------------- loss closures


@st.composite
def batches(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    prompt = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    resp = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    adv = draw(
        st.lists(st.floats(min_value=-3.0, max_value=3.0, allow_nan=False), min_size=n, max_size=n)
    )
    seed = draw(st.integers(min_value=0, max_value=2**16))
    gen = torch.Generator().manual_seed(seed)
    lps = [
        (-torch.rand(p + r, generator=gen, dtype=torch.float64)).requires_grad_(True)
        for p, r in zip(prompt, resp)
    ]
    return adv, prompt, resp, lps


@FAST
@given(batches(), st.booleans(), st.floats(min_value=0.0, max_value=1.0))
def test_grpo_loss_scores_only_response_tokens(batch, use_nll, nll_coef):
    adv, prompt, resp, lps = batch
    nll_mask = [True] * len(adv) if use_nll else None
    loss_fn = grpo.make_grpo_loss_fn(adv, nll_mask=nll_mask, nll_coef=nll_coef, response_lens=resp)
    loss, metrics = loss_fn(None, lps)
    loss.backward()
    expected = sum(-a * lp[p:].sum().item() for a, p, lp in zip(adv, prompt, lps)) / len(adv)
    if use_nll and nll_coef:
        expected += nll_coef * (-sum(lp[p:].mean().item() for p, lp in zip(prompt, lps)) / len(adv))
    assert math.isclose(loss.item(), expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(metrics["grpo_loss"], loss.item(), rel_tol=1e-12, abs_tol=1e-12)
    for p, r, a, lp in zip(prompt, resp, adv, lps):
        assert torch.all(lp.grad[:p] == 0), "prompt tokens must get zero gradient"
        if not use_nll or not nll_coef:
            want = torch.full((r,), -a / len(adv), dtype=lp.dtype)
            assert torch.allclose(lp.grad[p:], want, atol=1e-12)


@FAST
@given(batches())
def test_grpo_loss_zero_advantages_zero_policy_term(batch):
    _, prompt, resp, lps = batch
    loss_fn = grpo.make_grpo_loss_fn([0.0] * len(lps), response_lens=resp)
    loss, _ = loss_fn(None, lps)
    assert loss.item() == 0.0


@FAST
@given(batches())
def test_gspo_first_epoch_scores_only_response_tokens(batch):
    adv, prompt, resp, lps = batch
    loss_fn = grpo.make_gspo_loss_fn(adv, response_lens=resp)
    loss, metrics = loss_fn(None, lps)
    # Ratio is exactly 1 on the first epoch: loss = -mean(A), nothing clipped.
    assert math.isclose(loss.item(), -sum(adv) / len(adv), rel_tol=1e-9, abs_tol=1e-12)
    assert metrics["gspo_clip_frac"] == 0.0
    loss.backward()
    for p, lp in zip(prompt, lps):
        assert torch.all(lp.grad[:p] == 0)
