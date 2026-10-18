"""Hash binding does not make poisoned critic/Adam state numerically valid."""

import hashlib
import io

import pytest
import torch

from platform_tinker.tinkerrl import grpo
from platform_tinker.tinkerrl.critic import train_critic_step
from tests.test_critic_update_integrity import critic_and_optimizer


def trained_state():
    critic, optimizer = critic_and_optimizer()
    train_critic_step(critic, optimizer, [[1, 2]], [1.0])
    optimizer.step.reset_mock()
    return critic, optimizer


def poison(model, optimizer, kind):
    if kind == "model":
        next(iter(model.values())).view(-1)[0] = float("nan")
    else:
        next(iter(optimizer["state"].values()))["exp_avg_sq"].fill_(float("inf"))


@pytest.mark.parametrize("kind", ["model", "optimizer"])
def test_nonfinite_state_cannot_be_published(tmp_path, kind):
    critic, optimizer = trained_state()
    poison(critic.state_dict(), optimizer.state_dict(), kind)
    config = grpo.GRPOConfig(name="bad-state", checkpoint_dir=str(tmp_path))
    with pytest.raises(ValueError, match="critic state must be finite"):
        grpo._save_critic(config, config.seed, critic, optimizer, step=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["model", "optimizer"])
@pytest.mark.parametrize("bound", [False, True])
def test_nonfinite_state_is_rejected_even_when_its_hash_matches(tmp_path, kind, bound):
    critic, optimizer = trained_state()
    config = grpo.GRPOConfig(name="bad-state", checkpoint_dir=str(tmp_path))
    payload = {
        "model": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": 1,
        "config_sha256": grpo._critic_config_hash(config, config.seed),
    }
    poison(payload["model"], payload["optimizer"], kind)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    path = grpo._critic_checkpoint_path(config, config.seed, digest if bound else None)
    path.write_bytes(content)
    receipt = {"critic_state_sha256": digest} if bound else {}
    with pytest.raises(ValueError, match="critic state must be finite"):
        grpo._load_critic_checkpoint(config, config.seed, 1, receipt)
    assert path.read_bytes() == content


def test_existing_poisoned_adam_state_stops_before_another_update():
    critic, optimizer = trained_state()
    poison(critic.state_dict(), optimizer.state_dict(), "optimizer")
    before = {name: value.clone() for name, value in critic.state_dict().items()}
    with pytest.raises(ValueError, match="critic state must be finite"):
        train_critic_step(critic, optimizer, [[1, 2]], [1.0])
    optimizer.step.assert_not_called()
    assert all(torch.equal(value, before[name]) for name, value in critic.state_dict().items())


def test_newly_poisoned_optimizer_state_aborts_before_next_iteration():
    critic, optimizer = trained_state()
    original = optimizer.step

    def poison_after_update():
        original()
        poison(critic.state_dict(), optimizer.state_dict(), "optimizer")

    optimizer.step = poison_after_update
    with pytest.raises(ValueError, match="critic state must be finite"):
        train_critic_step(critic, optimizer, [[1, 2]], [1.0], steps=3)
    # Detection is after this completed update; no rollback is claimed.
    assert original.call_count == 1


def test_nested_scalar_state_and_shared_cycles_are_checked_without_mutation():
    from platform_tinker.tinkerrl.critic import require_finite_critic_state

    state = {"finite": torch.tensor([1.0]), "groups": [{"lr": 0.1}]}
    state["alias"] = state
    require_finite_critic_state(state)
    state["groups"][0]["lr"] = float("nan")
    with pytest.raises(ValueError, match="critic state must be finite"):
        require_finite_critic_state(state)
    assert state["alias"] is state
