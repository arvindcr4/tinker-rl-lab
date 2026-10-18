"""Real tiny causal LM, SDK loss bridge and Adam; no pretrained weights or network."""

from copy import deepcopy
import socket

import pytest
import torch

pytest.importorskip("tinker")
pytest.importorskip("transformers")
from tinker import types as T
from transformers import Qwen3Config, Qwen3ForCausalLM

from platform_tinker.tinkerrl import grpo
from tests.test_tinker_sdk_loss_integration import ReadyFuture, TransportlessClient, execute_bridge


def _rows(model, data):
    rows = []
    for datum in data:
        inputs = torch.tensor([datum.model_input.to_ints()], dtype=torch.long)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch().long()
        logits = model(input_ids=inputs, use_cache=False).logits[0]
        rows.append(logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return rows


class TinyCausalLMClient(TransportlessClient):
    def __init__(self, model):
        super().__init__([])
        self.model = model

    async def forward_async(self, data, loss_fn, loss_fn_config):
        assert loss_fn == "cross_entropy"
        assert loss_fn_config is None
        self.forward_data = data
        with torch.no_grad():
            rows = _rows(self.model, data)
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="offline_tiny_causal_lm",
                loss_fn_outputs=[
                    {
                        "logprobs": T.TensorData(
                            data=row.tolist(), dtype="float32", shape=[row.numel()]
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
        self.backward_data = data
        rows = _rows(self.model, data)
        weighted_nll = sum(
            -(row * datum.loss_fn_inputs["weights"].to_torch()).sum()
            for row, datum in zip(rows, data, strict=True)
        )
        weighted_nll.backward()
        return ReadyFuture(
            T.ForwardBackwardOutput(
                loss_fn_output_type="offline_tiny_causal_lm", loss_fn_outputs=[], metrics={}
            )
        )


@pytest.mark.parametrize("mode", ["grpo", "grpo_nll", "gspo"])
def test_real_causal_lm_update_matches_direct_objective_and_improves_rewarded_odds(
    monkeypatch, mode
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("tiny causal LM smoke must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            attention_dropout=0.0,
            attention_bias=False,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(config).cpu().eval()
    reference = deepcopy(model)
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    data = [grpo._build_datum([10, 11, 12], response) for response in ([20, 21], [22, 23])]
    initial_rows = _rows(model, data)
    before_odds = float((initial_rows[0][-2:].sum() - initial_rows[1][-2:].sum()).detach())
    if mode == "gspo":
        loss_fn = grpo.make_gspo_loss_fn(
            [1.0, -1.0],
            old_logprobs=[row[-2:].detach() for row in initial_rows],
            response_lens=[2, 2],
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
    else:
        loss_fn = grpo.make_grpo_loss_fn(
            [1.0, -1.0],
            response_lens=[2, 2],
            nll_mask=[True, False],
            nll_coef=0.2 if mode == "grpo_nll" else 0.0,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-3)
    client = TinyCausalLMClient(model)
    execute_bridge(client, data, loss_fn)
    direct_loss, _ = loss_fn(None, _rows(reference, data))
    direct_loss.backward()
    for datum in client.backward_data:
        weights = datum.loss_fn_inputs["weights"].to_torch()
        assert torch.equal(weights[:2], torch.zeros(2))
        assert torch.isfinite(weights).all()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
    optimizer.step()
    reference_optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_parameters, model.parameters(), strict=True)
    )
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    after_rows = _rows(model, data)
    after_odds = float((after_rows[0][-2:].sum() - after_rows[1][-2:].sum()).detach())
    assert after_odds > before_odds
    assert optimizer.state
    assert all("exp_avg" in state for state in optimizer.state.values())
    for actual, expected in zip(
        optimizer.state.values(), reference_optimizer.state.values(), strict=True
    ):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-7)
