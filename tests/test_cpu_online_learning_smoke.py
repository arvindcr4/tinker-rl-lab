"""Closed-loop sampling and GRPO on a random tiny CPU LM, not a quality benchmark."""

from copy import deepcopy
import json
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("tinker")
pytest.importorskip("transformers")
from tinker import types as T
from transformers import Qwen3Config, Qwen3ForCausalLM

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _Future
from tests.test_cpu_causal_lm_smoke import TinyCausalLMClient
from tests.test_cpu_critic_resume_smoke import _same_state
from tests.test_grpo_coverage import _runtime
from tests.test_tinker_sdk_loss_integration import execute_bridge


class OnlineCPUClient(TinyCausalLMClient):
    model_id = "synthetic-online-cpu"

    def __init__(self, model):
        super().__init__(model)
        self.optimizer = None
        self.snapshots = {}
        self.states = {}
        self.updates = 0

    def save_weights_for_sampler(self, name):
        path = "tinker://synthetic/" + name
        self.snapshots[path] = deepcopy(self.model)
        return _Future(SimpleNamespace(path=path))

    def save_state(self, name, overwrite):
        assert overwrite is True
        path = "tinker://synthetic/state/" + name
        self.states[path] = {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
        }
        return _Future(SimpleNamespace(path=path))

    def restore(self, path):
        state = self.states[path]
        self.model = Qwen3ForCausalLM(deepcopy(self.model.config)).cpu().eval()
        self.model.load_state_dict(state["model"])
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        self.updates = 0
        return self

    def create_sampling_client(self, model_path):
        model = self.snapshots[model_path]

        def sample(prompt, num_samples, sampling_params):
            # This environment terminates after one categorical action. Its
            # action probabilities come from the frozen causal LM snapshot.
            assert sampling_params.max_tokens == 1
            assert sampling_params.temperature == 1.0
            assert sampling_params.top_p == 1.0
            assert sampling_params.seed is not None
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([prompt.to_ints()]), use_cache=False).logits[
                    0, -1
                ]
                logprobs = logits.log_softmax(-1)
            generator = torch.Generator().manual_seed(sampling_params.seed)
            sampled = torch.multinomial(
                logprobs.exp(), num_samples, replacement=True, generator=generator
            )
            return _Future(
                SimpleNamespace(
                    sequences=[
                        T.SampledSequence(
                            tokens_np=np.array([int(value)]),
                            logprobs_np=np.array([float(logprobs[value])]),
                            stop_reason="stop",
                        )
                        for value in sampled
                    ]
                )
            )

        return SimpleNamespace(sample=sample)

    def forward_backward_custom(self, *, data, loss_fn, loss_type_input):
        assert loss_type_input == "logprobs"
        self.model.zero_grad(set_to_none=True)
        return _Future(execute_bridge(self, data, loss_fn))

    def optim_step(self, params):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                eps=params.eps,
            )
        self.optimizer.step()
        self.updates += 1
        return _Future(SimpleNamespace())


@pytest.mark.parametrize("target", [2, 3])
def test_runner_learns_from_its_own_cpu_causal_lm_samples(tmp_path, monkeypatch, target):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("online CPU smoke must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = (
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=4,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=16,
                    attention_dropout=0.0,
                    tie_word_embeddings=True,
                )
            )
            .cpu()
            .eval()
        )

    def target_probability():
        with torch.no_grad():
            return float(
                model(input_ids=torch.tensor([[0]]), use_cache=False)
                .logits[0, -1]
                .softmax(-1)[target]
            )

    before = target_probability()
    client = OnlineCPUClient(model)
    service = SimpleNamespace(
        create_lora_training_client=lambda **_kwargs: client,
        create_training_client_from_state_with_optimizer=client.restore,
    )
    config = _config(
        tmp_path,
        name="tiny-online",
        seed=7,
        seed_sampling=True,
        steps=20,
        batch_size=1,
        group_size=8,
        lr=0.02,
        save_every=1,
        temperature=1.0,
        top_p=1.0,
        max_prompt_tokens=4,
        max_response_tokens=1,
    )
    dataset = grpo.InMemoryDataset(
        train=[grpo.TrainingExample(prompt="synthetic", target=str(target))]
    )
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: [0], decode=lambda ids, **_kwargs: str(ids[0])
    )

    def run():
        with (
            _runtime(monkeypatch),
            patch.dict(sys.modules, {"tinker.types": T}),
            patch.object(sys.modules["tinker"], "types", T),
            patch.object(sys.modules["tinker"], "ServiceClient", return_value=service),
        ):
            return grpo._run_one_seed(
                config, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
            )

    periodic = {}
    write = grpo._write_checkpoint

    def capture(path, payload):
        if payload.get("status") == "started" and payload.get("step") == 10:
            periodic.update(deepcopy(payload))
        write(path, payload)

    with patch.object(grpo, "_write_checkpoint", side_effect=capture):
        result = run()
    after = target_probability()
    assert before < 0.4
    assert after > 0.9
    assert after - before > 0.6
    assert client.updates == config.steps
    assert len(result.reward_trace) == config.steps
    assert result.avg_last10 > result.avg_first5
    assert len(client.snapshots) == config.steps + 2
    final_state = client.states["tinker://synthetic/state/seed7_final"]
    assert final_state["optimizer"]["state"]
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, final_state["model"][name], rtol=0, atol=0)
    receipt = json.loads(grpo._checkpoint_path(config, config.seed).read_text())
    assert receipt["status"] == "completed"
    assert receipt["result"]["reward_trace"] == result.reward_trace

    # Restart from the actual step-ten actor/Adam state and committed receipt.
    # Seeded model sampling must reproduce the remaining trajectory exactly.
    expected_final = deepcopy(final_state)
    path = grpo._checkpoint_path(config, config.seed)
    grpo._write_checkpoint(path, periodic)
    saved_tenth = deepcopy(client.states[periodic["train_state_path"]])
    resumed = run()
    assert client.updates == 10
    assert resumed.resumed_from_step == 10
    _same_state(client.states[periodic["train_state_path"]], saved_tenth)
    assert resumed.reward_trace == result.reward_trace
    _same_state(client.model.state_dict(), expected_final["model"])
    _same_state(client.optimizer.state_dict(), expected_final["optimizer"])

    # An optimizer response can fail after its update was applied. Recover only
    # from the last durable receipt, discarding uncommitted actor/Adam changes.
    grpo._write_checkpoint(path, periodic)
    optimizer_step = client.optim_step

    def applied_then_failed(params):
        response = optimizer_step(params)
        if client.updates == 2:
            raise RuntimeError("synthetic optimizer response lost after application")
        return response

    with patch.object(client, "optim_step", side_effect=applied_then_failed):
        with pytest.raises(RuntimeError, match="response lost after application"):
            run()
    durable = json.loads(path.read_text())
    assert durable["status"] == "started"
    assert durable["step"] == 11
    assert durable["train_state_path"].endswith("state_seed7_11")
    assert "result" not in durable
    assert client.updates == 2
    recovered = run()  # also proves the failed run released its local lock
    assert recovered.resumed_from_step == 11
    assert client.updates == 9
    assert recovered.reward_trace == result.reward_trace
    _same_state(client.model.state_dict(), expected_final["model"])
    _same_state(client.optimizer.state_dict(), expected_final["optimizer"])
