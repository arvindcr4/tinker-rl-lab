"""Two-token CPU trajectories and an independent multi-epoch GSPO objective.

Random full-parameter toy LM; transport/persistence doubles, no pretrained
weights, LoRA implementation, provider service or benchmark-quality claim.
"""

from copy import deepcopy
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
from tests.test_cpu_online_learning_smoke import OnlineCPUClient
from tests.test_grpo_coverage import _runtime


class TwoTokenCPUClient(OnlineCPUClient):
    def __init__(self, model):
        super().__init__(model)
        self.rollouts = []

    def create_sampling_client(self, model_path):
        model = self.snapshots[model_path]

        def sample(prompt, num_samples, sampling_params):
            assert sampling_params.max_tokens == 2
            assert sampling_params.temperature == sampling_params.top_p == 1.0
            generator = torch.Generator().manual_seed(sampling_params.seed)
            sequences = []
            with torch.no_grad():
                for _ in range(num_samples):
                    prefix = list(prompt.to_ints())
                    tokens, logprobs = [], []
                    for _ in range(2):
                        lp = (
                            model(input_ids=torch.tensor([prefix]), use_cache=False)
                            .logits[0, -1]
                            .log_softmax(-1)
                        )
                        token = int(torch.multinomial(lp.exp(), 1, generator=generator))
                        tokens.append(token)
                        logprobs.append(float(lp[token]))
                        prefix.append(token)
                    sequences.append(
                        T.SampledSequence(
                            tokens_np=np.array(tokens),
                            logprobs_np=np.array(logprobs),
                            stop_reason="stop",
                        )
                    )
            self.rollouts.extend(deepcopy(sequences))
            return _Future(SimpleNamespace(sequences=sequences))

        return SimpleNamespace(sample=sample)


def test_sampled_two_token_gspo_epochs_match_independent_clipped_ratio_oracle(
    tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("multi-epoch CPU GSPO must stay offline")

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
    oracle = deepcopy(model)
    client = TwoTokenCPUClient(model)
    config = _config(
        tmp_path,
        name="cpu-gspo-epochs",
        seed=7,
        seed_sampling=True,
        steps=1,
        group_size=8,
        batch_size=1,
        lr=0.02,
        save_every=1,
        temperature=1.0,
        top_p=1.0,
        max_prompt_tokens=4,
        max_response_tokens=2,
        gspo_enabled=True,
        gspo_update_epochs=3,
        gspo_epsilon_low=0.2,
        gspo_epsilon_high=0.2,
    )
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="synthetic", target="2")])
    tokenizer = SimpleNamespace(encode=lambda *_a, **_k: [0], decode=lambda ids, **_k: str(ids[-1]))
    service = SimpleNamespace(create_lora_training_client=lambda **_: client)
    with (
        _runtime(monkeypatch),
        patch.dict(sys.modules, {"tinker.types": T}),
        patch.object(sys.modules["tinker"], "types", T),
        patch.object(sys.modules["tinker"], "ServiceClient", return_value=service),
    ):
        grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
        )
    assert len(client.rollouts) == config.group_size
    assert client.updates == config.gspo_update_epochs
    rewards = np.array([float(row.tokens[-1] == 2) for row in client.rollouts], dtype=np.float64)
    assert 0 < rewards.sum() < len(rewards)
    advantages = torch.tensor(
        (rewards - rewards.mean()) / (rewards.std() + 1e-8), dtype=torch.float32
    )

    def selected_logprobs(network):
        result = []
        for sequence in client.rollouts:
            tokens = list(sequence.tokens)
            logits = network(input_ids=torch.tensor([[0, tokens[0]]]), use_cache=False).logits[0]
            result.append(
                logits.log_softmax(-1).gather(-1, torch.tensor(tokens).unsqueeze(-1)).squeeze(-1)
            )
        return torch.stack(result)

    with torch.no_grad():
        old = selected_logprobs(oracle)
    torch.testing.assert_close(old, torch.tensor([list(row.logprobs) for row in client.rollouts]))
    optimizer = torch.optim.Adam(
        oracle.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps
    )
    clipped_after_first = False
    for epoch in range(config.gspo_update_epochs):
        optimizer.zero_grad(set_to_none=True)
        ratios = (selected_logprobs(oracle) - old).mean(-1).exp()
        clipped = ratios.clamp(1 - config.gspo_epsilon_low, 1 + config.gspo_epsilon_high)
        if epoch:
            selected_clip = ((advantages > 0) & (ratios > 1 + config.gspo_epsilon_high)) | (
                (advantages < 0) & (ratios < 1 - config.gspo_epsilon_low)
            )
            clipped_after_first |= bool(selected_clip.any())
        loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
        loss.backward()
        optimizer.step()
    assert clipped_after_first
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, oracle.state_dict()[name], rtol=2e-5, atol=2e-6)
    actual_state = client.optimizer.state_dict()["state"]
    expected_state = optimizer.state_dict()["state"]
    for key, state in expected_state.items():
        for field, value in state.items():
            torch.testing.assert_close(actual_state[key][field], value, rtol=2e-5, atol=2e-7)
