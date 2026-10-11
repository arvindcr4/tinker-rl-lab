"""A hash-valid critic sidecar must also fit the configured model and Adam."""

import hashlib
import io
from unittest.mock import patch

import pytest
import torch

from platform_tinker.tinkerrl import grpo
from platform_tinker.tinkerrl.critic import restore_critic_state, train_critic_step
from tests._shared_fakes import _config, _example, _Tokenizer


def _receipt(tmp_path, mutation):
    config = _config(tmp_path, steps=2, critic_enabled=True, critic_hidden_dim=8)
    critic = grpo.PromptValueCritic(
        hidden_dim=config.critic_hidden_dim, max_prompt_tokens=config.max_prompt_tokens
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    train_critic_step(critic, optimizer, [[1, 2]], [1.0])
    payload = {
        "model": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": 1,
        "config_sha256": grpo._critic_config_hash(config, config.seed),
    }
    mutation(payload)
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = grpo._critic_checkpoint_path(config, config.seed, digest)
    sidecar.write_bytes(data)
    path = grpo._checkpoint_path(config, config.seed)
    grpo._write_checkpoint(
        path,
        {
            "config": grpo._config_fingerprint(config, config.seed),
            "status": "started",
            "step": 1,
            "train_state_path": "tinker://synthetic/actor-step-1",
            "reward_trace": [0.5],
            "critic_state_sha256": digest,
        },
    )
    return config, path, sidecar, payload


def _mutate(payload, kind):
    state = next(iter(payload["optimizer"]["state"].values()))
    if kind == "learning_rate":
        payload["optimizer"]["param_groups"][0]["lr"] = 0.5
    elif kind == "moment_shape":
        state["exp_avg"] = torch.zeros(3)
    elif kind == "negative_variance":
        state["exp_avg_sq"].fill_(-1)
    elif kind == "fractional_step":
        state["step"] = torch.tensor(0.5)
    elif kind == "boolean_step":
        state["step"] = torch.tensor(True)
    elif kind == "complex_model":
        key = next(iter(payload["model"]))
        payload["model"][key] = payload["model"][key].to(torch.complex64) + 1j
    elif kind == "complex_moment":
        state["exp_avg"] = state["exp_avg"].to(torch.complex64) + 1j
    elif kind == "model_cast_overflow":
        key = next(iter(payload["model"]))
        payload["model"][key] = torch.full_like(payload["model"][key], 1e100, dtype=torch.float64)
    elif kind == "moment_cast_overflow":
        state["exp_avg"] = torch.full_like(state["exp_avg"], 1e100, dtype=torch.float64)
    elif kind == "missing_moment":
        state.pop("exp_avg_sq")
    elif kind == "parameter_order":
        payload["optimizer"]["param_groups"][0]["params"].reverse()
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "learning_rate",
        "moment_shape",
        "negative_variance",
        "fractional_step",
        "boolean_step",
        "complex_model",
        "complex_moment",
        "model_cast_overflow",
        "moment_cast_overflow",
        "missing_moment",
        "parameter_order",
    ],
)
def test_bad_but_hash_valid_state_stops_before_tracking(tmp_path, kind):
    config, path, sidecar, _ = _receipt(tmp_path, lambda payload: _mutate(payload, kind))
    before = path.read_bytes(), sidecar.read_bytes()
    data = grpo.InMemoryDataset(train=[_example()])
    with patch.object(
        grpo, "_start_wandb", side_effect=AssertionError("tracking started")
    ) as tracking:
        with pytest.raises(ValueError, match="critic restore"):
            grpo._run_one_seed(config, data, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert (path.read_bytes(), sidecar.read_bytes()) == before


@pytest.mark.parametrize("kind", ["ordinary", "no_updates", "legacy_defaults"])
def test_supported_optimizer_states_restore_without_changing_options(tmp_path, kind):
    def adjust(payload):
        if kind == "no_updates":
            payload["optimizer"]["state"].clear()
        if kind == "legacy_defaults":
            group = payload["optimizer"]["param_groups"][0]
            group.pop("foreach")
            group.pop("fused")

    config, _, _, payload = _receipt(tmp_path, adjust)
    critic = grpo.PromptValueCritic(
        hidden_dim=config.critic_hidden_dim, max_prompt_tokens=config.max_prompt_tokens
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    expected_groups = optimizer.state_dict()["param_groups"]
    restore_critic_state(critic, optimizer, payload)
    assert optimizer.state_dict()["param_groups"] == expected_groups
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, payload["model"][key], rtol=0, atol=0)
    actual = optimizer.state_dict()["state"]
    assert actual.keys() == payload["optimizer"]["state"].keys()
    for key, state in actual.items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, payload["optimizer"]["state"][key][name], rtol=0, atol=0
            )
