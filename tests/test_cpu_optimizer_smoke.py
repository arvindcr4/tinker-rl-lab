"""Real runner + SDK custom-loss bridge + Adam, with only CPU transports.

Balanced synthetic responses are fixed fixtures, not model-quality evaluation.
"""

import json
from copy import deepcopy
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("tinker")
from tinker import types as T

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _Future
from tests.test_grpo_coverage import _runtime
from tests.test_tinker_sdk_loss_integration import ReadyFuture, TransportlessClient, execute_bridge


class CPUClient(TransportlessClient):
    model_id = "synthetic-cpu-model"

    def __init__(self):
        super().__init__([])
        self.policy = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1]))
        self.prompt = torch.nn.Parameter(torch.tensor([0.4, -0.1, 0.2]))
        self.optimizer = None
        self.snapshots = {}
        self.states = {}
        self.gradients = []
        self.probabilities = [float(self.policy.softmax(0)[1].detach())]

    def rows(self, data):
        result = []
        for datum in data:
            targets = datum.loss_fn_inputs["target_tokens"].to_torch().long()
            result.append(
                torch.cat(
                    [
                        self.prompt.log_softmax(0)[targets[:-1]],
                        self.policy.log_softmax(0)[targets[-1:]],
                    ]
                )
            )
        return result

    async def forward_async(self, data, loss_fn, loss_fn_config):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        rows = self.rows(data)
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="synthetic_cpu",
                loss_fn_outputs=[
                    {
                        "logprobs": T.TensorData(
                            data=row.detach().tolist(), dtype="float32", shape=[row.numel()]
                        )
                    }
                    for row in rows
                ],
                metrics={},
            )
        )

    async def forward_backward_async(self, data, loss_fn, loss_fn_config):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        self.policy.grad = self.prompt.grad = None
        loss = sum(
            -(row * datum.loss_fn_inputs["weights"].to_torch()).sum()
            for row, datum in zip(self.rows(data), data)
        )
        loss.backward()
        self.gradients.append(self.policy.grad.detach().clone())
        assert torch.equal(self.prompt.grad, torch.zeros_like(self.prompt))
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="synthetic_cpu", loss_fn_outputs=[], metrics={}
            )
        )

    def forward_backward_custom(self, *, data, loss_fn, loss_type_input):
        assert loss_type_input == "logprobs"
        return _Future(execute_bridge(self, data, loss_fn))

    def optim_step(self, params):
        assert isinstance(params, T.AdamParams)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                [self.policy, self.prompt],
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                eps=params.eps,
            )
        self.optimizer.step()
        self.probabilities.append(float(self.policy.softmax(0)[1].detach()))
        return _Future(None)

    def save_weights_for_sampler(self, name):
        path = f"tinker://synthetic/{name}/{len(self.snapshots)}"
        self.snapshots[path] = self.policy.log_softmax(0).detach().clone()
        return _Future(SimpleNamespace(path=path))

    def create_sampling_client(self, model_path):
        old = self.snapshots[model_path]

        def sample(prompt, num_samples, sampling_params):
            assert isinstance(prompt, T.ModelInput)
            assert isinstance(sampling_params, T.SamplingParams)
            assert num_samples == 2
            return _Future(
                T.SampleResponse(
                    sequences=[
                        T.SampledSequence(
                            stop_reason="stop",
                            tokens_np=np.array([token]),
                            logprobs_np=np.array([float(old[token])], dtype=np.float32),
                        )
                        for token in (1, 2)
                    ]
                )
            )

        return SimpleNamespace(sample=sample)

    def save_state(self, name, overwrite):
        path = f"tinker://synthetic-state/{name}"
        self.states[path] = (
            self.policy.detach().clone(),
            self.prompt.detach().clone(),
            deepcopy(self.optimizer.state_dict()),
        )
        return _Future(SimpleNamespace(path=path))

    def restore(self, path):
        policy, prompt, state = self.states[path]
        with torch.no_grad():
            self.policy.copy_(policy)
            self.prompt.copy_(prompt)
        self.optimizer.load_state_dict(deepcopy(state))
        self.gradients = []
        self.probabilities = [float(self.policy.softmax(0)[1].detach())]
        return self


@pytest.mark.parametrize(
    ("mode", "rewarded_token"), [("grpo", 1), ("nll", 1), ("gspo", 1), ("grpo", 2)]
)
def test_runner_sdk_and_adam_match_independent_cpu_objective(
    tmp_path, monkeypatch, mode, rewarded_token
):
    def blocked(*_args, **_kwargs):
        raise AssertionError("CPU smoke must not use a network socket")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    client = CPUClient()
    original_prompt = client.prompt.detach().clone()
    expected = torch.nn.Parameter(client.policy.detach().clone())
    config = _config(
        tmp_path,
        name=mode,
        steps=3,
        save_every=1,
        lr=0.03,
        gspo_enabled=mode == "gspo",
        gspo_update_epochs=2,
        nll_aux_enabled=mode == "nll",
        nll_aux_coef=0.2,
    )
    oracle = torch.optim.Adam(
        [expected], lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps
    )
    expected_gradients = []
    advantages = torch.tensor([0.5 / (0.5 + 1e-8), -0.5 / (0.5 + 1e-8)])
    if rewarded_token == 2:
        advantages = -advantages
    for _ in range(config.steps):
        old = expected.log_softmax(0)[1:].detach().clone()
        for _ in range(config.gspo_update_epochs if mode == "gspo" else 1):
            oracle.zero_grad()
            lp = expected.log_softmax(0)[1:]
            if mode == "gspo":
                ratio = (lp - old).exp()
                loss = -torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1 - config.gspo_epsilon_low, 1 + config.gspo_epsilon_high)
                    * advantages,
                ).mean()
            else:
                loss = -(lp * advantages).mean()
                if mode == "nll":
                    loss = loss - config.nll_aux_coef * lp[rewarded_token - 1]
            loss.backward()
            expected_gradients.append(expected.grad.detach().clone())
            oracle.step()
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: [0, 0, 0], decode=lambda tokens, **_kwargs: str(tokens[0])
    )
    dataset = grpo.InMemoryDataset(
        train=[grpo.TrainingExample(prompt="synthetic", target=str(rewarded_token))]
    )

    class TokenReward:
        def score(self, text, example):
            return float(text == example.target)

    reward = TokenReward()

    def run():
        service = SimpleNamespace(
            create_lora_training_client=lambda **_kwargs: client,
            create_training_client_from_state_with_optimizer=client.restore,
        )
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
        result = run()
    assert len(client.gradients) == len(expected_gradients)
    for actual, expected_gradient in zip(client.gradients, expected_gradients):
        torch.testing.assert_close(actual, expected_gradient, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(client.policy, expected, rtol=1e-5, atol=1e-7)
    assert torch.equal(client.prompt, original_prompt)
    direction = 1 if rewarded_token == 1 else -1
    assert all(
        direction * (b - a) > 0 for a, b in zip(client.probabilities, client.probabilities[1:])
    )
    assert result.reward_trace == [0.5] * config.steps
    receipt = json.loads(grpo._checkpoint_path(config, config.seed).read_text())
    assert receipt["status"] == "completed"
    assert receipt["step"] == 3

    # Restore the actual step-one Adam state and receipt, then finish the same
    # budget. This checks optimizer momentum as well as weight restoration.
    grpo._write_checkpoint(grpo._checkpoint_path(config, config.seed), periodic)
    resumed = run()
    torch.testing.assert_close(client.policy, expected, rtol=1e-5, atol=1e-7)
    assert torch.equal(client.prompt, original_prompt)
    assert resumed.resumed_from_step == 1
    assert resumed.reward_trace == result.reward_trace
    remaining = (config.steps - 1) * (config.gspo_update_epochs if mode == "gspo" else 1)
    assert len(client.gradients) == remaining
