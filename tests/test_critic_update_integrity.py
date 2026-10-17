"""CPU critic updates reject nonfinite loss/gradients before Adam state changes."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from platform_tinker.tinkerrl.critic import PromptValueCritic, train_critic_step, value_loss_mse


def critic_and_optimizer():
    torch.manual_seed(7)
    critic = PromptValueCritic(vocab_buckets=16, embed_dim=4, hidden_dim=8)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    optimizer.step = Mock(wraps=optimizer.step)
    return critic, optimizer


@pytest.mark.parametrize("target", [float("nan"), float("inf"), 1e308, 1e20])
def test_invalid_critic_target_or_loss_cannot_change_parameters_or_optimizer(target):
    critic, optimizer = critic_and_optimizer()
    before = deepcopy(critic.state_dict())
    with pytest.raises(ValueError, match="critic .*finite"):
        train_critic_step(critic, optimizer, [[1, 2]], [target])
    optimizer.step.assert_not_called()
    assert optimizer.state == {}
    for name, value in critic.state_dict().items():
        assert torch.equal(value, before[name])


def test_nonfinite_backward_gradient_is_cleared_before_optimizer():
    critic, optimizer = critic_and_optimizer()
    before = deepcopy(critic.state_dict())
    hook = critic.mlp[-1].bias.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    try:
        with pytest.raises(ValueError, match="critic gradients must be finite"):
            train_critic_step(critic, optimizer, [[1, 2]], [1.0])
    finally:
        hook.remove()
    optimizer.step.assert_not_called()
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in critic.parameters())
    for name, value in critic.state_dict().items():
        assert torch.equal(value, before[name])


def test_ordinary_critic_update_matches_existing_arithmetic_exactly():
    critic, optimizer = critic_and_optimizer()
    reference = deepcopy(critic)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    prompts, targets = [[1, 2], [3]], [1.0, 0.0]
    expected_losses = []
    for _ in range(2):
        reference_optimizer.zero_grad()
        loss = value_loss_mse(reference(prompts), torch.tensor(targets, dtype=torch.float32))
        loss.backward()
        reference_optimizer.step()
        expected_losses.append(loss.item())
    actual = train_critic_step(critic, optimizer, prompts, targets, steps=2)
    assert actual == sum(expected_losses) / 2
    for name, value in critic.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name])
    assert optimizer.step.call_count == 2


def test_critic_failure_cannot_publish_successful_run_receipt(tmp_path, monkeypatch):
    from platform_tinker.tinkerrl import grpo
    from tests._shared_fakes import _config, _example, _Tokenizer
    from tests.test_grpo_caliber_fixes import PROMPT, _runtime

    class LargeFiniteReward:
        def score(self, _response, _example):
            return 1e20

    with _runtime(monkeypatch) as holder:
        with pytest.raises(ValueError, match="critic loss must be finite"):
            grpo._run_one_seed(
                _config(tmp_path, name="critic-overflow", critic_enabled=True),
                grpo.InMemoryDataset(train=[_example()]),
                LargeFiniteReward(),
                _Tokenizer(tokens=PROMPT, decode_first_only=True),
                logger=lambda _: None,
            )
    # Actor-before-critic ordering is unchanged; that actor update is not
    # rolled back. The failed critic cannot certify successful completion.
    assert holder["optim_calls"] == 1
    assert not (tmp_path / "critic-overflow_seed42.json").exists()
    assert not list(tmp_path.glob("*.pt"))
