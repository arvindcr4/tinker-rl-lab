"""Execute the real SDK custom-loss bridge with transport-free CPU fixtures."""

import asyncio
import socket
from types import SimpleNamespace

import pytest
import torch
import numpy as np

pytest.importorskip("tinker", reason="Tinker SDK extra is not installed")

from tinker import types as T
from tinker.lib.public_interfaces.training_client import TrainingClient

from platform_tinker.tinkerrl.grpo import (
    GRPOConfig,
    TrainingExample,
    _build_datum,
    _datum_response_len,
    _sample_scored_group,
    make_grpo_loss_fn,
    make_gspo_loss_fn,
)


class ReadyFuture:
    def __init__(self, value):
        self.value = value

    async def result_async(self, timeout=None):
        return self.value


class TransportlessClient:
    """Only the two transport methods are fake; the SDK bridge is executed."""

    holder = None
    _get_custom_loss_forward_data = TrainingClient._get_custom_loss_forward_data

    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.forward_data = None
        self.backward_data = None

    async def forward_async(self, data, loss_fn, loss_fn_config):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        self.forward_data = data
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="synthetic_cpu",
                loss_fn_outputs=[
                    {"logprobs": T.TensorData(data=values, dtype="float32", shape=[len(values)])}
                    for values in self.logprobs
                ],
                metrics={},
            )
        )

    async def forward_backward_async(self, data, loss_fn, loss_fn_config):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        self.backward_data = data
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="synthetic_cpu",
                loss_fn_outputs=[],
                metrics={"synthetic_transport": 1.0},
            )
        )


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden_connect(*_args, **_kwargs):
        raise AssertionError("SDK integration must never contact a network service")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


def execute_bridge(client, data, loss_fn):
    async def run():
        future = await TrainingClient.forward_backward_custom_async(
            client, data, loss_fn, loss_type_input="logprobs"
        )
        return await future.result_async()

    return asyncio.run(run())


def run_bridge(data, logprobs, loss_fn):
    client = TransportlessClient(logprobs)
    result = execute_bridge(client, data, loss_fn)
    assert result.metrics["synthetic_transport"] == 1
    for datum in client.forward_data:
        assert all(value == 0 for value in datum.loss_fn_inputs["weights"].data)
    return client, result


@pytest.mark.parametrize("mode", ["grpo", "grpo-nll", "gspo", "gspo-clipped"])
def test_real_sdk_bridge_preserves_response_masking_and_gradient_sign(mode):
    prompts, responses = [[10, 11, 12], [30]], [[20, 21], [40, 41, 42]]
    data = [
        _build_datum(prompt, response) for prompt, response in zip(prompts, responses, strict=True)
    ]
    lengths = [
        _datum_response_len(prompt, response)
        for prompt, response in zip(prompts, responses, strict=True)
    ]
    assert data[0].model_input.to_ints() == [10, 11, 12, 20]
    assert data[0].loss_fn_inputs["target_tokens"].data == [11, 12, 20, 21]
    logprobs = [[-9.0, -8.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]
    if mode.startswith("gspo"):
        old = [[-2.0, -3.0], [-4.0, -5.0, -6.0]]
        if mode == "gspo-clipped":
            old = [[-3.0, -4.0], [-3.0, -4.0, -5.0]]
        loss_fn = make_gspo_loss_fn(
            [1.0, -1.0],
            old_logprobs=old,
            response_lens=lengths,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        expected = [[0.0, 0.0, 0.25, 0.25], [-1 / 6] * 3]
        if mode == "gspo-clipped":
            expected = [[0.0] * 4, [0.0] * 3]
    else:
        loss_fn = make_grpo_loss_fn(
            [1.0, -1.0],
            response_lens=lengths,
            nll_mask=[True, False],
            nll_coef=0.2 if mode == "grpo-nll" else 0.0,
        )
        response_weight = 0.6 if mode == "grpo-nll" else 0.5
        expected = [[0.0, 0.0, response_weight, response_weight], [-0.5] * 3]
    client, result = run_bridge(data, logprobs, loss_fn)
    for original, backward, weights in zip(data, client.backward_data, expected, strict=True):
        assert backward.model_input == original.model_input
        assert backward.loss_fn_inputs["target_tokens"] == original.loss_fn_inputs["target_tokens"]
        actual = backward.loss_fn_inputs["weights"].to_torch()
        torch.testing.assert_close(actual, torch.tensor(weights), rtol=1e-6, atol=1e-7)
        assert torch.isfinite(actual).all()
    metric = "gspo_loss" if mode.startswith("gspo") else "grpo_loss"
    assert metric in result.metrics


def test_real_sdk_bridge_emits_zero_gradients_for_an_empty_response():
    data = [_build_datum([10, 11, 12], [])]
    loss_fn = make_grpo_loss_fn([1.0], response_lens=[0])
    client, result = run_bridge(data, [[-3.0, -4.0]], loss_fn)
    assert client.backward_data[0].loss_fn_inputs["weights"].data == [0.0, 0.0]
    assert result.metrics["grpo_loss"] == 0.0


def test_nonfinite_sdk_forward_output_never_reaches_backward_transport():
    data = [_build_datum([10], [20])]
    client = TransportlessClient([[float("nan")]])
    with pytest.raises(ValueError, match="response logprobs must be finite"):
        execute_bridge(client, data, make_grpo_loss_fn([1.0], response_lens=[1]))
    assert client.forward_data is not None
    assert client.backward_data is None


def test_sampling_controls_and_metadata_use_real_sdk_models():
    config = GRPOConfig(
        name="sdk-contract",
        group_size=2,
        seed_sampling=True,
        temperature=0.75,
        top_p=0.9,
        max_response_tokens=4,
        wandb_enabled=False,
        require_wandb=False,
        hf_enabled=False,
        require_hf=False,
    )
    sequences = [
        T.SampledSequence(
            stop_reason="stop",
            tokens_np=np.array([20]),
            logprobs_np=np.array([-0.1], dtype=np.float32),
        ),
        T.SampledSequence(
            stop_reason="length",
            tokens_np=np.array([21, 22, 23, 24]),
            logprobs_np=np.array([-0.2] * 4, dtype=np.float32),
        ),
    ]

    def sample(prompt, num_samples, sampling_params):
        assert isinstance(prompt, T.ModelInput)
        assert prompt.to_ints() == [10, 11, 12]
        assert isinstance(sampling_params, T.SamplingParams)
        assert num_samples == 2
        assert sampling_params.seed == 17
        assert sampling_params.temperature == 0.75
        assert sampling_params.top_p == 0.9
        assert sampling_params.max_tokens == 4
        return SimpleNamespace(result=lambda: T.SampleResponse(sequences=sequences))

    tokenizer = SimpleNamespace(
        encode=lambda prompt, **kwargs: [10, 11, 12],
        decode=lambda tokens, **kwargs: "42" if tokens == [20] else "0",
    )
    reward = SimpleNamespace(score=lambda text, example: float(text == "42"))
    metadata = {}
    result = _sample_scored_group(
        tokenizer,
        SimpleNamespace(sample=sample),
        T,
        config,
        reward,
        TrainingExample(prompt="synthetic question", target="42"),
        seed=17,
        meta=metadata,
    )
    assert result == ([10, 11, 12], [[20], [21, 22, 23, 24]], [1.0, 0.0], [1, 4])
    assert metadata["truncated"] == [False, True]
    assert metadata["sampler_logprobs"][0] == pytest.approx([-0.1])
    assert metadata["sampler_logprobs"][1] == pytest.approx([-0.2] * 4)
    assert metadata["identical_group"] is False


@pytest.mark.parametrize("mode", ["grpo", "gspo"])
def test_real_sdk_await_cannot_rebind_rollout_inputs(mode):
    lengths, mask, old = [1], [True], [[-1.0]]
    if mode == "grpo":
        loss_fn = make_grpo_loss_fn([1.0], nll_mask=mask, nll_coef=0.5, response_lens=lengths)
        expected = [0.0, 1.5]
    else:
        loss_fn = make_gspo_loss_fn([1.0], old_logprobs=old, response_lens=lengths)
        expected = [0.0, 1.0]

    class MutatingForwardClient(TransportlessClient):
        async def forward_async(self, data, loss_fn, loss_fn_config):
            await asyncio.sleep(0)
            lengths[0] = 2
            mask[0] = False
            old[0][0] = -10.0
            return await super().forward_async(data, loss_fn, loss_fn_config)

    client = MutatingForwardClient([[-9.0, -1.0]])
    execute_bridge(client, [_build_datum([10, 11], [20])], loss_fn)
    assert client.backward_data[0].loss_fn_inputs["weights"].data == pytest.approx(expected)
