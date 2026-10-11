"""Actual CPU actor/critic optimizers must survive a joint mid-run restore."""

from copy import deepcopy
import json
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("tinker")
from tinker import types as T

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config
from tests.test_cpu_optimizer_smoke import CPUClient
from tests.test_grpo_coverage import _runtime


def _same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _same_state(left, right)
    else:
        assert actual == expected


def test_real_actor_and_critic_resume_preserve_adam_and_pretraining_draws(tmp_path, monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("CPU critic smoke must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    client = CPUClient()
    initial_actor = client.policy.detach().clone()
    config = _config(
        tmp_path,
        name="critic-cpu",
        seed=7,
        steps=3,
        batch_size=2,
        save_every=1,
        lr=0.03,
        critic_enabled=True,
        critic_hidden_dim=8,
        critic_lr=0.01,
        critic_updates_per_step=1,
        critic_pretrain_batches=2,
        max_prompt_tokens=8,
    )
    dataset = grpo.InMemoryDataset(
        train=[
            grpo.TrainingExample(prompt=str(index), target=target)
            for index, target in enumerate(("all", "1", "none"))
        ]
    )
    encoded = []

    def encode(prompt, **_kwargs):
        encoded.append(prompt)
        return [int(prompt)] * (int(prompt) + 2)

    tokenizer = SimpleNamespace(encode=encode, decode=lambda ids, **_kwargs: str(ids[0]))
    reward = SimpleNamespace(score=lambda text, ex: float(ex.target == "all" or text == ex.target))
    service = SimpleNamespace(
        create_lora_training_client=lambda **_kwargs: client,
        create_training_client_from_state_with_optimizer=client.restore,
    )

    def run():
        with _runtime(monkeypatch):
            with patch.dict(sys.modules, {"tinker.types": T}):
                with (
                    patch.object(sys.modules["tinker"], "types", T),
                    patch.object(sys.modules["tinker"], "ServiceClient", return_value=service),
                ):
                    return grpo._run_one_seed(
                        config, dataset, reward, tokenizer, logger=lambda _: None
                    )

    periodic = {}
    write = grpo._write_checkpoint

    def capture(path, payload):
        if payload.get("status") == "started" and payload.get("step") == 1:
            periodic.update(deepcopy(payload))
        write(path, payload)

    with patch.object(grpo, "_write_checkpoint", side_effect=capture):
        uninterrupted = run()
    path = grpo._checkpoint_path(config, config.seed)
    final_receipt = json.loads(path.read_text())
    final_critic = grpo._load_critic_checkpoint(config, config.seed, config.steps, final_receipt)
    final_actor = client.policy.detach().clone()
    final_actor_optimizer = deepcopy(client.optimizer.state_dict())
    original_order = list(encoded)
    assert (
        len(original_order) == (config.critic_pretrain_batches + config.steps) * config.batch_size
    )
    assert not torch.equal(initial_actor, final_actor)
    assert final_critic["optimizer"]["state"]

    # Real, digest-bound critic bytes and both Adam states come from step one.
    grpo._write_checkpoint(path, periodic)
    encoded.clear()
    resumed = run()
    resumed_critic = grpo._load_critic_checkpoint(
        config, config.seed, config.steps, json.loads(path.read_text())
    )
    _same_state(client.policy, final_actor)
    _same_state(client.optimizer.state_dict(), final_actor_optimizer)
    _same_state(resumed_critic, final_critic)
    prefix = (config.critic_pretrain_batches + 1) * config.batch_size
    assert encoded == original_order[prefix:]
    assert resumed.reward_trace == uninterrupted.reward_trace
    assert resumed.resumed_from_step == 1
    assert len(client.gradients) == config.steps - 1
