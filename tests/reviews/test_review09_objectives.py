"""Tiny offline numerical oracles for the implemented v3 objectives.

These are synthetic tensor evaluations, not model training or provider calls.
The local GRPO helper is a token-sum policy-gradient objective with optional NLL;
GSPO is response-mean clipped ratios. Neither implements PPO KL or framework parity.
"""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _Future, temporary_modules
from tests import test_grpo_module as grpo_tests


@pytest.mark.parametrize("advantage", [2.0, -2.0])
@pytest.mark.parametrize("ratio", [0.7, 0.9, 1.1, 1.4])
def test_gspo_value_and_gradient_match_scalar_clipping_oracle(advantage, ratio):
    old = torch.tensor([-2.0, -2.0], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor(advantage, dtype=torch.float64, requires_grad=True)
    current = torch.tensor([-2 + math.log(ratio)] * 2, dtype=torch.float64, requires_grad=True)
    loss, _ = grpo.make_gspo_loss_fn([adv], [old], epsilon_low=0.2, epsilon_high=0.2)(
        None, [current]
    )
    clipped = min(max(ratio, 0.8), 1.2)
    expected = -min(ratio * advantage, clipped * advantage)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    active = ratio * advantage <= clipped * advantage
    expected_gradient = -advantage * ratio / 2 if active else 0
    assert current.grad.tolist() == pytest.approx([expected_gradient] * 2)
    assert old.grad is None
    assert adv.grad is None


def test_bound_behavior_probabilities_do_not_change_between_epochs_or_by_caller_mutation():
    old = torch.tensor([-2.0], requires_grad=True)
    adv = torch.tensor(2.0, requires_grad=True)
    loss_fn = grpo.make_gspo_loss_fn([adv], [old], epsilon_low=0.2, epsilon_high=0.2)
    with torch.no_grad():
        old.fill_(5)
        adv.fill_(99)
    for ratio in (1.1, 1.4):
        current = torch.tensor([-2 + math.log(ratio)], requires_grad=True)
        loss, _ = loss_fn(None, [current])
        assert loss.item() == pytest.approx(-2 * min(ratio, 1.2))
        loss.backward()
    assert old.grad is None
    assert adv.grad is None


def test_grpo_unequal_responses_are_token_sums_then_response_mean_with_detached_advantages():
    adv = torch.tensor([2.0, -1.0], requires_grad=True)
    short = torch.tensor([-9.0, -0.2], requires_grad=True)
    long = torch.tensor([-9.0, -0.3, -0.4, -0.5], requires_grad=True)
    loss, _ = grpo.make_grpo_loss_fn(adv, response_lens=[1, 3])(None, [short, long])
    assert loss.item() == pytest.approx((-0.0 + 0.4 - 1.2) / 2)
    loss.backward()
    assert short.grad.tolist() == pytest.approx([0, -1])
    assert long.grad.tolist() == pytest.approx([0, 0.5, 0.5, 0.5])
    assert adv.grad is None


@pytest.mark.parametrize("rewards", [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
def test_homogeneous_rewards_have_zero_task_gradient_but_auxiliary_is_separate(rewards):
    variance = sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)
    assert variance == 0
    assert grpo.is_degenerate_group(rewards)
    advantages = grpo.normalize_rewards(rewards)
    task_lp = [
        torch.tensor([-0.5], requires_grad=True),
        torch.tensor([-1.0, -1.0], requires_grad=True),
    ]
    loss, _ = grpo.make_grpo_loss_fn(advantages)(None, task_lp)
    loss.backward()
    assert all(torch.count_nonzero(lp.grad).item() == 0 for lp in task_lp)
    aux_lp = [lp.detach().clone().requires_grad_() for lp in task_lp]
    total, metrics = grpo.make_grpo_loss_fn(advantages, nll_mask=[True, True], nll_coef=0.4)(
        None, aux_lp
    )
    total.backward()
    assert metrics["nll_loss"] == pytest.approx(0.75)
    assert aux_lp[0].grad.tolist() == pytest.approx([-0.2])
    assert aux_lp[1].grad.tolist() == pytest.approx([-0.1, -0.1])


def test_two_epoch_integration_executes_loss_closure_with_fixed_sampler_probabilities(tmp_path):
    helper = grpo_tests.TestTrackingFailClosed()
    runtime = helper._fake_runtime()
    prototype = runtime["tinker"].ServiceClient().training_client
    training_class = type(prototype)
    sampling_class = type(prototype.create_sampling_client("synthetic"))
    seen = []

    def sample(self, *args, **kwargs):
        return _Future(
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(tokens=[1, 2], logprobs=[-1.0, -1.0]),
                    SimpleNamespace(tokens=[0, 2], logprobs=[-1.0, -1.0]),
                ]
            )
        )

    runtime["tokenizer"].decode = lambda tokens, **kwargs: "1" if tokens[0] == 1 else "0"

    def forward(self, *, data, loss_fn, loss_type_input):
        epoch = self.forward_count % 2
        self.forward_count += 1
        current = [
            torch.tensor([-1.0 + 0.0001 + epoch * 0.0005] * 2, requires_grad=True),
            torch.tensor([-1.0] * 2, requires_grad=True),
        ]
        loss, metrics = loss_fn(data, current)
        loss.backward()
        seen.append((loss.item(), metrics["gspo_clip_frac"], current[0].grad.tolist()))
        return _Future(SimpleNamespace(metrics=metrics))

    with (
        temporary_modules(helper._fake_modules(runtime)),
        patch.object(sampling_class, "sample", sample),
        patch.object(training_class, "forward_backward_custom", forward),
        patch.object(grpo, "_publish_checkpoint", side_effect=helper._fake_success_receipt),
    ):
        cfg = grpo.GRPOConfig(
            name="objective-v3",
            steps=1,
            save_every=1,
            group_size=2,
            batch_size=1,
            checkpoint_dir=str(tmp_path),
            gspo_enabled=True,
            gspo_update_epochs=2,
        )
        grpo._run_one_seed(
            cfg,
            helper._dataset(),
            grpo.ExactMathReward(),
            runtime["tokenizer"],
            logger=lambda _: None,
        )
    assert len(seen) == 2
    assert seen[0][1] == 0
    assert seen[1][1] == 0.5
    assert any(value != 0 for value in seen[0][2])
    assert seen[1][2] == [0, 0]
    assert runtime["holder"]["training_client"].optim_count == 2
    assert cfg.protocol_version == "grpo-on-policy-v3"


@pytest.mark.parametrize("factory", [grpo.make_grpo_loss_fn, grpo.make_gspo_loss_fn])
def test_python_advantage_snapshot_preserves_float64_precision(factory):
    advantage = 0.0699605178105589
    logprobs = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    loss, _ = factory([advantage], response_lens=[1])(None, [logprobs])
    expected = advantage if factory is grpo.make_grpo_loss_fn else -advantage
    assert loss.item() == pytest.approx(expected, rel=1e-14, abs=1e-15)


def test_grpo_rejects_unused_or_missing_advantages():
    for advantages in ([], [1.0, 2.0]):
        with pytest.raises(ValueError, match="pair 1:1"):
            grpo.make_grpo_loss_fn(advantages, response_lens=[1])(None, [torch.tensor([-1.0])])
