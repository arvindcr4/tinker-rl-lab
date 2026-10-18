"""A genuine committed-checkpoint interruption of the offline loop fixture.

Only synthetic scalar actor tensors and a tiny CPU critic are evaluated. No
provider, model download, GPU job, or stored experiment evidence is touched.
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _Future, temporary_modules
from tests import test_grpo_module as fixtures


class Tokenizer:
    def encode(self, prompt, **kwargs):
        return [int(prompt.split("-")[-1]) + 1]

    def decode(self, tokens, **kwargs):
        return "1" if tokens[0] == 1 else "0"


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_nested(a, b)
    else:
        assert left == right


def execute(tmp_path, interrupted=False, mutation=None):
    helper = fixtures.TestTrackingFailClosed()
    runtime = helper._fake_runtime()
    service_class = runtime["tinker"].ServiceClient
    prototype = service_class().training_client
    training_class = type(prototype)
    sampling_class = type(prototype.create_sampling_client("test"))
    states, observations, losses, torch_draws = {}, [], [], []
    original_init = training_class.__init__
    original_checkpoint = grpo._write_checkpoint
    stop = [interrupted]

    def init(self):
        original_init(self)
        self.actor = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
        self.actor_optimizer = torch.optim.Adam([self.actor], lr=0.01)

    def forward(self, *, data, loss_fn, **kwargs):
        self.actor_optimizer.zero_grad()
        # This synthetic observation advances Torch RNG at every actual step,
        # and affects both objective and update. Restarting at the initial RNG
        # position cannot accidentally reproduce the continuation trace.
        scale = 0.8 + 0.4 * torch.rand((), dtype=torch.float64)
        torch_draws.append(scale.item())
        logprobs = [
            (self.actor * scale).expand(len(d.loss_fn_inputs["target_tokens"].data)) for d in data
        ]
        loss, metrics = loss_fn(data, logprobs)
        loss.backward()
        losses.append(loss.item())
        self.forward_count += 1
        return _Future(SimpleNamespace(metrics=metrics))

    def optim(self, params):
        self.actor_optimizer.step()
        self.optim_count += 1
        return _Future(None)

    def save(self, name, overwrite):
        path = "synthetic://state/" + name
        states[path] = {
            "actor": self.actor.detach().clone(),
            "optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "forward_count": self.forward_count,
            "optim_count": self.optim_count,
        }
        return _Future(SimpleNamespace(path=path))

    def restore(self, path):
        state = states[path]
        client = self.training_client
        with torch.no_grad():
            client.actor.copy_(state["actor"])
        client.actor_optimizer.load_state_dict(state["optimizer"])
        client.forward_count = state["forward_count"]
        client.optim_count = state["optim_count"]
        return client

    def sample(self, prompt, **kwargs):
        observations.append(list(prompt))
        return _Future(
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(tokens=[1, 2, 3], logprobs=[-1.0] * 3),
                    SimpleNamespace(tokens=[0], logprobs=[-1.0]),
                ]
            )
        )

    def checkpoint(path, payload):
        original_checkpoint(path, payload)
        if stop[0] and payload.get("status") == "started" and payload["step"] == 2:
            stop[0] = False
            raise RuntimeError("synthetic interruption after committed checkpoint")

    cfg = grpo.GRPOConfig(
        name="real-interruption",
        steps=4,
        save_every=1,
        group_size=2,
        batch_size=1,
        checkpoint_dir=str(tmp_path),
        critic_enabled=True,
        critic_hidden_dim=4,
        critic_updates_per_step=1,
    )
    dataset = grpo.InMemoryDataset(
        train=[grpo.TrainingExample(prompt=f"prompt-{i}", target="1") for i in range(5)]
    )
    tokenizer = Tokenizer()
    with (
        temporary_modules(helper._fake_modules(runtime)),
        patch.object(training_class, "__init__", init),
        patch.object(training_class, "forward_backward_custom", forward),
        patch.object(training_class, "optim_step", optim),
        patch.object(training_class, "save_state", save),
        patch.object(service_class, "create_training_client_from_state_with_optimizer", restore),
        patch.object(sampling_class, "sample", sample),
        patch.object(grpo, "_publish_checkpoint", side_effect=helper._fake_success_receipt),
        patch.object(grpo, "_write_checkpoint", checkpoint),
    ):
        if interrupted:
            with pytest.raises(RuntimeError, match="synthetic interruption"):
                grpo._run_one_seed(
                    cfg, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
                )
            receipt_path = tmp_path / "real-interruption_seed42.json"
            receipt = json.loads(receipt_path.read_text())
            assert receipt["step"] == receipt["critic_state"]["step"] == 2
            if mutation:
                mutation(receipt)
                receipt_path.write_text(json.dumps(receipt))
            result = grpo._run_one_seed(
                cfg, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
            )
        else:
            result = grpo._run_one_seed(
                cfg, dataset, grpo.ExactMathReward(), tokenizer, logger=lambda _: None
            )
        client = runtime["holder"]["training_client"]
    final = json.loads((tmp_path / "real-interruption_seed42.json").read_text())
    critic = torch.load(final["critic_state"]["path"], weights_only=True)
    return {
        "actor": client.actor.detach(),
        "optimizer": client.actor_optimizer.state_dict(),
        "critic": critic,
        "prompts": observations,
        "losses": losses,
        "rewards": result.reward_trace,
        "steps": (client.forward_count, client.optim_count),
        "data_rng_state": final["data_rng_state"],
        "torch_draws": torch_draws,
        "torch_rng_state": torch.get_rng_state(),
    }


def test_genuine_interruption_matches_uninterrupted_parameters_optimizer_rng_and_trace(tmp_path):
    full = execute(tmp_path / "full")
    resumed = execute(tmp_path / "resumed", interrupted=True)
    # Each artifact is config-bound to its own checkpoint directory; numerical
    # state and random streams must nevertheless match bit for bit.
    assert full["critic"].pop("config_sha256") != resumed["critic"].pop("config_sha256")
    equal_nested(full, resumed)
    assert len(full["prompts"]) == 4
    assert len(set(map(tuple, full["prompts"]))) > 1
    assert full["steps"] == (4, 4)
    assert full["actor"].item() != -1
    assert len(set(full["torch_draws"])) == 4


def test_genuine_interruption_detects_disabled_torch_rng_restoration(tmp_path):
    full = execute(tmp_path / "full")
    with patch.object(grpo.torch, "set_rng_state", lambda state: None):
        resumed = execute(tmp_path / "resumed", interrupted=True)
    assert full["torch_draws"][:2] == resumed["torch_draws"][:2]
    assert full["torch_draws"][2:] != resumed["torch_draws"][2:]
    assert full["losses"][2:] != resumed["losses"][2:]
    with pytest.raises(AssertionError):
        equal_nested(full, resumed)


@pytest.mark.parametrize("step", [1, 3])
def test_resume_rejects_stale_or_newer_critic_sidecar(tmp_path, step):
    with pytest.raises(RuntimeError, match="step mismatch"):
        execute(
            tmp_path,
            interrupted=True,
            mutation=lambda receipt: receipt["critic_state"].update(step=step),
        )


def test_resume_rejects_different_protocol_identity(tmp_path):
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        execute(
            tmp_path,
            interrupted=True,
            mutation=lambda receipt: receipt["config"].update(protocol_version="other-protocol"),
        )
