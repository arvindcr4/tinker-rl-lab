"""Provider checkpoint references must be valid and stable before reuse."""

import json
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer, _Future
from tests.test_grpo_coverage import _runtime


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("checkpoint reference tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.mark.parametrize(
    "phase",
    [
        "initial",
        "behavior",
        "periodic_actor",
        "periodic_second_actor",
        "periodic_sampler",
        "final_actor",
        "final_sampler",
    ],
)
def test_bad_checkpoint_response_stops_before_export_or_further_updates(
    tmp_path, monkeypatch, phase
):
    config = _config(tmp_path, steps=2, save_every=1 if phase.startswith("periodic") else 99)
    dataset = grpo.InMemoryDataset(train=[_example()])
    with _runtime(monkeypatch) as holder:
        original_service = sys.modules["tinker"].ServiceClient

        class BrokenService(original_service):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                original_sampler = self.client.save_weights_for_sampler
                original_state = self.client.save_state

                def sampler(name):
                    fail = (
                        (phase == "initial" and name.endswith("_step_0"))
                        or (phase == "behavior" and name.startswith("behavior_seed"))
                        or (phase == "periodic_sampler" and name.startswith("step_seed"))
                        or (phase == "final_sampler" and name.endswith("_final"))
                    )
                    return _Future(SimpleNamespace(path="   ")) if fail else original_sampler(name)

                def state(name, overwrite):
                    fail = (
                        phase in {"periodic_actor", "periodic_second_actor"}
                        and name.startswith("state_seed")
                        and (phase != "periodic_second_actor" or name.endswith("_2"))
                    ) or (phase == "final_actor" and name.endswith("_final"))
                    return (
                        _Future(SimpleNamespace(path=None))
                        if fail
                        else original_state(name, overwrite)
                    )

                self.client.save_weights_for_sampler = sampler
                self.client.save_state = state

        with patch.object(sys.modules["tinker"], "ServiceClient", BrokenService):
            with patch.object(
                grpo, "_prepare_hf_revision", wraps=grpo._prepare_hf_revision
            ) as publish:
                with pytest.raises(RuntimeError, match="checkpoint path"):
                    grpo._run_one_seed(
                        config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
                    )
    assert publish.call_count == (
        0 if phase == "initial" else 2 if phase == "periodic_second_actor" else 1
    )
    expected_updates = {
        "initial": 0,
        "behavior": 0,
        "periodic_actor": 1,
        "periodic_sampler": 1,
    }.get(phase, 2)
    assert holder["events"].count("optim") == expected_updates
    path = grpo._checkpoint_path(config, config.seed)
    if phase == "periodic_second_actor":
        saved = json.loads(path.read_text())
        assert saved["step"] == 1
        assert saved["train_state_path"] == "tinker://state/state_seed42_1"
        assert saved["reward_trace"] == [0.5]
    else:
        assert not path.exists()


def test_export_rejects_nonstrings_blank_or_option_like_references_before_hub(tmp_path):
    config = _config(tmp_path)
    for value in (None, "", "   ", True, 123, {}, "--help", "tinker://synthetic\nline"):
        with patch.object(grpo, "_preflight_hf", side_effect=AssertionError("Hub accessed")) as hub:
            with pytest.raises(RuntimeError, match="checkpoint path"):
                grpo._publish_checkpoint(config, config.seed, value, lambda _: None)
        hub.assert_not_called()


def test_callback_mutation_cannot_rebind_saved_sampler_reference(tmp_path, monkeypatch):
    config = _config(tmp_path, steps=1, save_every=1)
    dataset = grpo.InMemoryDataset(train=[_example()])
    responses, published, sampled = [], [], []
    actor_responses, actor_paths, persisted_actor_paths = [], [], []
    original_write = grpo._write_checkpoint

    def capture(path, payload):
        if payload.get("status") == "started":
            persisted_actor_paths.append(payload["train_state_path"])
        original_write(path, payload)

    original_publish = grpo._publish_checkpoint
    with _runtime(monkeypatch):
        original_service = sys.modules["tinker"].ServiceClient

        class MutableService(original_service):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                save = self.client.save_weights_for_sampler
                create = self.client.create_sampling_client
                save_state = self.client.save_state

                def state(name, overwrite):
                    response = save_state(name, overwrite).result()
                    actor_responses.append(response)
                    actor_paths.append(response.path)
                    return _Future(response)

                def weights(name):
                    if name.startswith("step_seed") or name.endswith("_final"):
                        actor_responses[-1].path = "tinker://synthetic/mutated-actor"
                    response = save(name).result()
                    responses.append(response)
                    return _Future(response)

                def sampling(model_path):
                    sampled.append(model_path)
                    return create(model_path)

                self.client.save_state = state
                self.client.save_weights_for_sampler = weights
                self.client.create_sampling_client = sampling

        def publish(config, seed, path, logger, **kwargs):
            published.append(path)
            receipt = original_publish(config, seed, path, logger, **kwargs)
            responses[-1].path = "tinker://synthetic/mutated-after-export"
            return receipt

        with patch.object(sys.modules["tinker"], "ServiceClient", MutableService):
            with (
                patch.object(grpo, "_publish_checkpoint", side_effect=publish),
                patch.object(grpo, "_write_checkpoint", side_effect=capture),
            ):
                result = grpo._run_one_seed(
                    config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
                )
    assert persisted_actor_paths == actor_paths[:1]
    assert [path for path in sampled if "/behavior_seed" not in path] == published
    assert len([path for path in sampled if "/behavior_seed" in path]) == config.steps
    assert result.sampler_path == published[-1]
    receipt = json.loads(grpo._checkpoint_path(config, config.seed).read_text())
    assert receipt["result"]["sampler_path"] == published[-1]
    assert receipt["train_state_path"] == actor_paths[-1]
