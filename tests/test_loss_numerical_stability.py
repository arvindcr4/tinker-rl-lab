"""CPU value/gradient oracles for stable policy-loss kernels."""

import pytest
import torch

from platform_tinker.tinkerrl.grpo import GRPOConfig, make_grpo_loss_fn, make_gspo_loss_fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gspo_matches_direct_objective_and_gradients_in_every_finite_branch(dtype):
    cases = [(adv, drift) for adv in (-2.0, 0.0, 1.5) for drift in (-2.0, -0.2, 0.0, 0.2, 2.0)]
    new = [
        torch.full((i % 3 + 1,), -3.0 + drift, dtype=dtype, requires_grad=True)
        for i, (_, drift) in enumerate(cases)
    ]
    oracle_new = [value.detach().clone().requires_grad_() for value in new]
    old = [torch.full_like(value, -3.0) for value in new]
    ratios = [torch.exp((value - ref).mean()) for value, ref in zip(oracle_new, old)]
    oracle = -torch.stack(
        [
            torch.minimum(ratio * adv, torch.clamp(ratio, 0.9, 1.2) * adv)
            for ratio, (adv, _) in zip(ratios, cases)
        ]
    ).mean()
    loss, metrics = make_gspo_loss_fn(
        [adv for adv, _ in cases], old_logprobs=old, epsilon_low=0.1, epsilon_high=0.2
    )(None, new)
    oracle.backward()
    loss.backward()
    torch.testing.assert_close(loss, oracle)
    for actual, expected in zip(new, oracle_new):
        torch.testing.assert_close(actual.grad, expected.grad)
    expected_fraction = sum(not torch.equal(r, torch.clamp(r, 0.9, 1.2)) for r in ratios) / len(
        ratios
    )
    assert metrics["gspo_clip_frac"] == expected_fraction


@pytest.mark.parametrize("advantage", [0.0, 1.0])
def test_clipped_or_zero_advantage_overflow_never_poison_gradients(advantage):
    new = torch.tensor([-1.0], requires_grad=True)
    loss, metrics = make_gspo_loss_fn([advantage], old_logprobs=[[-1000.0]])(None, [new])
    loss.backward()
    assert loss.item() == pytest.approx(-advantage * 1.0004)
    assert torch.equal(new.grad, torch.zeros_like(new))
    assert metrics["gspo_clip_frac"] == 1.0


def test_unbounded_negative_advantage_is_rejected_not_falsely_clipped():
    new = torch.tensor([-1.0], requires_grad=True)
    stash = []
    with pytest.raises(ValueError, match="GSPO objective is non-finite"):
        make_gspo_loss_fn([-1.0], old_logprobs=[[-1000.0]], stash=stash)(None, [new])
    assert stash == []
    assert new.grad is None


@pytest.mark.parametrize("advantage", [-1.0, 0.0, 1.0])
def test_deep_underflow_preserves_signed_clipping(advantage):
    new = torch.tensor([-1000.0], requires_grad=True)
    loss, _ = make_gspo_loss_fn([advantage], old_logprobs=[[-1.0]], epsilon_low=0.1)(None, [new])
    loss.backward()
    assert loss.item() == pytest.approx(0.9 if advantage < 0 else 0.0)
    assert torch.equal(new.grad, torch.zeros_like(new))


def test_behavior_policy_is_a_detached_reference():
    old = torch.tensor([-2.0, -2.0], requires_grad=True)
    new = torch.tensor([-2.0, -2.0], requires_grad=True)
    loss, _ = make_gspo_loss_fn([1.0], old_logprobs=[old])(None, [new])
    loss.backward()
    assert old.grad is None
    torch.testing.assert_close(new.grad, torch.tensor([-0.5, -0.5]))


@pytest.mark.parametrize(("advantages", "count"), [([1.0, -1.0], 1), ([1.0], 2)])
def test_grpo_rejects_misaligned_advantages(advantages, count):
    with pytest.raises(ValueError, match="pair 1:1"):
        make_grpo_loss_fn(advantages)(None, [torch.tensor([-1.0])] * count)


@pytest.mark.parametrize("factory", [make_grpo_loss_fn, make_gspo_loss_fn])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_policy_inputs_are_rejected(factory, value):
    with pytest.raises(ValueError, match="advantages must be finite"):
        factory([value])
    with pytest.raises(ValueError, match="logprobs must be finite"):
        factory([1.0])(None, [torch.tensor([value], requires_grad=True)])


def test_large_lower_clip_epsilon_retains_disabled_lower_bound_semantics():
    new = torch.tensor([-4.0], requires_grad=True)
    loss, _ = make_gspo_loss_fn([-1.0], old_logprobs=[[-2.0]], epsilon_low=2.0)(None, [new])
    assert loss.item() == pytest.approx(torch.exp(torch.tensor(-2.0)).item())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_clip_settings_fail_at_configuration_boundary(value):
    with pytest.raises(ValueError, match="clip epsilons must be finite"):
        GRPOConfig(name="invalid", gspo_epsilon_low=value)
    with pytest.raises(ValueError, match="clip epsilons must be finite"):
        GRPOConfig(name="invalid", gspo_epsilon_high=value)


def test_grpo_nonfinite_reduction_is_rejected():
    values = torch.tensor([-3e38, -3e38], requires_grad=True)
    with pytest.raises(ValueError, match="GRPO objective is non-finite"):
        make_grpo_loss_fn([1.0])(None, [values])
    assert values.grad is None
