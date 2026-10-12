"""Offline training -> final-policy evaluation -> receipt -> cached replay.

Random CPU toy model and in-memory transports, not a provider/benchmark result.
"""

from dataclasses import asdict
import hashlib
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

from platform_tinker.tinkerrl import grpo, grpo_cli
from tests._shared_fakes import _config, _Future
from tests.test_cpu_online_learning_smoke import OnlineCPUClient
from tests.test_grpo_coverage import _runtime


class EvaluatingCPUClient(OnlineCPUClient):
    def __init__(self, model):
        super().__init__(model)
        self.evaluation_requests = []

    def create_sampling_client(self, model_path):
        training_sampler = super().create_sampling_client(model_path)
        model = self.snapshots[model_path]

        def sample(prompt, num_samples, sampling_params):
            if sampling_params.temperature == 1.0:
                return training_sampler.sample(prompt, num_samples, sampling_params)
            assert num_samples == sampling_params.max_tokens == 1
            assert sampling_params.temperature == 0.1
            assert sampling_params.top_p == 0.95
            self.evaluation_requests.append((model_path, sampling_params.seed))
            if len(self.evaluation_requests) == 2:
                raise RuntimeError("synthetic held-out transport failure")
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([prompt.to_ints()]), use_cache=False).logits[
                    0, -1
                ]
                probabilities = (logits / sampling_params.temperature).softmax(-1)
                ordered, indices = probabilities.sort(descending=True)
                keep = ordered.cumsum(-1) - ordered < sampling_params.top_p
                filtered = ordered * keep
                filtered /= filtered.sum()
                generator = torch.Generator().manual_seed(sampling_params.seed)
                token = int(indices[torch.multinomial(filtered, 1, generator=generator)])
            return _Future(
                SimpleNamespace(
                    sequences=[T.SampledSequence(tokens_np=np.array([token]), stop_reason="stop")]
                )
            )

        return SimpleNamespace(sample=sample)


def test_cpu_training_final_evaluation_and_cached_receipt(tmp_path, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU workflow must stay offline")

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
    client = EvaluatingCPUClient(model)
    config = _config(
        tmp_path,
        name="cpu-heldout-workflow",
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
        evaluate_heldout=True,
    )
    dataset = grpo.InMemoryDataset(
        train=[grpo.TrainingExample(prompt="train", target="2")],
        test=[grpo.TrainingExample(prompt=f"heldout-{i}", target="2") for i in range(4)],
    )
    tokenizer = SimpleNamespace(encode=lambda *_a, **_k: [0], decode=lambda ids, **_k: str(ids[0]))
    service = SimpleNamespace(create_lora_training_client=lambda **_kwargs: client)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    arguments = ["--preset", "math100", "--reward", "gsm8k", "--json-config", str(config_path)]
    assert grpo_cli.main([*arguments, "--check-config"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert (
        inspection["config_sha256"]
        == hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    )
    assert inspection["effective"]["evaluate_heldout"] is True
    monkeypatch.setenv("TINKER_API_KEY", "synthetic-offline-placeholder")
    monkeypatch.setattr(grpo_cli, "_build_dataset", lambda *_a: dataset)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: tokenizer)
    with (
        _runtime(monkeypatch),
        patch.dict(sys.modules, {"tinker.types": T}),
        patch.object(sys.modules["tinker"], "types", T),
        patch.object(sys.modules["tinker"], "ServiceClient", return_value=service) as provider,
    ):
        assert grpo_cli.main(arguments) == 0
        assert "heldout_reward: 0.750" in capsys.readouterr().out
        path = grpo._checkpoint_path(config, config.seed)
        result = grpo.GRPORunResult(**json.loads(path.read_text())["result"])
        assert client.updates == 20
        assert result.heldout_reward == 0.75
        assert client.evaluation_requests == [
            (result.sampler_path, grpo._sampling_seed(7, 20, i)) for i in range(4)
        ]
        evaluation = result.heldout_evaluation
        assert evaluation["status"] == "partial_failure"
        assert evaluation["total_n"] == 4
        assert evaluation["failed_n"] == 1
        assert evaluation["input_source"]["status"] == "fingerprinted"
        path = grpo._checkpoint_path(config, config.seed)
        before = path.read_bytes()
        stored = json.loads(before)
        assert stored["config"] == json.loads(json.dumps(grpo._config_fingerprint(config, 7)))
        assert stored["result"]["heldout_evaluation"] == evaluation
        provider.reset_mock()
        replay = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
        )
        provider.assert_not_called()
        assert replay == result
        assert len(client.evaluation_requests) == 4
        assert path.read_bytes() == before
