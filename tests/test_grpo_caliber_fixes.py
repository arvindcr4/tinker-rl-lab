"""Regression tests for the 2026-10-03 code-caliber audit fixes in tinkerrl.grpo.

Covers: response-only policy loss (prompt-token logprobs get no gradient),
GSPO ratio against sampler logprobs, held-out failure counting, resume
backfill with dataclass defaults, seeded sampling, stop_reason truncation,
dynamic-sampling reward/trace logging and the CLI reward lookup.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from platform_tinker.tinkerrl import grpo, grpo_cli
from platform_tinker.tinkerrl.grpo import (
    ExactMathReward,
    GRPOConfig,
    InMemoryDataset,
    MathReward,
    TrainingExample,
    make_grpo_loss_fn,
    make_gspo_loss_fn,
)

PROMPT = [11, 12, 13, 14, 15]
PROMPT_LP = -5.0  # trainer logprob on prompt positions
RESP_LP = -1.0  # trainer logprob on response positions


def _example(target="1", prompt="q"):
    return TrainingExample(prompt=prompt, target=target)


def _with_prefix(n_prompt: int, resp: list) -> torch.Tensor:
    """Fake Tinker logprobs for prompt+response targets: P-1 prompt + R response."""
    return torch.tensor([PROMPT_LP] * (n_prompt - 1) + list(resp), requires_grad=True)


# ---------------------------------------------------------------- loss closures


def test_grpo_loss_ignores_prompt_tokens():
    lp = _with_prefix(50, [-1.0, -2.0, -3.0])
    loss, _ = make_grpo_loss_fn([2.0], response_lens=[3])(None, [lp])
    loss.backward()
    assert loss.item() == pytest.approx(-2.0 * -6.0)
    assert torch.all(lp.grad[:49] == 0)
    assert torch.all(lp.grad[49:] == -2.0)


def test_grpo_loss_without_mask_still_sums_everything():
    # response_lens=None keeps the old contract for response-only callers.
    lp = _with_prefix(3, [-1.0])
    loss, _ = make_grpo_loss_fn([1.0])(None, [lp])
    assert loss.item() == pytest.approx(-(2 * PROMPT_LP - 1.0) * -1.0 * -1.0)


def test_nll_aux_is_response_token_mean():
    lp = _with_prefix(10, [-1.0, -3.0])
    loss, metrics = make_grpo_loss_fn([0.0], nll_mask=[True], nll_coef=1.0, response_lens=[2])(
        None, [lp]
    )
    loss.backward()
    assert metrics["nll_loss"] == pytest.approx(2.0)
    assert torch.all(lp.grad[:9] == 0)


def test_gspo_loss_ignores_prompt_tokens_and_uses_old_logprobs():
    lp = _with_prefix(20, [-1.0, -1.0])
    loss_fn = make_gspo_loss_fn(
        [1.0], old_logprobs=[[-1.0, -1.0]], response_lens=[2], epsilon_low=0.2, epsilon_high=0.2
    )
    loss, metrics = loss_fn(None, [lp])
    loss.backward()
    assert metrics["gspo_clip_frac"] == 0.0
    assert loss.item() == pytest.approx(-1.0)
    assert torch.all(lp.grad[:19] == 0)
    assert torch.all(lp.grad[19:] != 0)


def test_gspo_stale_sampler_clips():
    # Behaviour policy (sampler) put -2.0 on each token, trainer now -1.0:
    # s = exp(1) ~ 2.72, far outside [1-3e-4, 1+4e-4] -> clipped.
    lp = _with_prefix(8, [-1.0, -1.0, -1.0])
    loss, metrics = make_gspo_loss_fn([1.0], old_logprobs=[[-2.0, -2.0, -2.0]], response_lens=[3])(
        None, [lp]
    )
    loss.backward()
    assert metrics["gspo_clip_frac"] == 1.0
    assert loss.item() == pytest.approx(-(1.0 + 4e-4))
    # Clipped branch is constant -> no gradient anywhere.
    assert torch.all(lp.grad == 0)


def test_response_lens_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        make_grpo_loss_fn([1.0], response_lens=[5])(None, [torch.zeros(3, requires_grad=True)])
    with pytest.raises(ValueError, match="pair 1:1"):
        make_grpo_loss_fn([1.0], response_lens=[1, 1])(None, [torch.zeros(3, requires_grad=True)])


def test_datum_response_len_matches_build_datum_targets():
    assert grpo._datum_response_len([1, 2], [3, 4, 5]) == 3
    assert grpo._datum_response_len([], [3, 4, 5]) == 2


# ---------------------------------------------------------------- fake runtime


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Tokenizer:
    def encode(self, _prompt, add_special_tokens=False):
        return list(PROMPT)

    def decode(self, tokens, skip_special_tokens=True):
        return "1" if list(tokens)[:1] == [1] else "0"


class _SeededParams(SimpleNamespace):
    """Mimics pydantic SamplingParams: exposes ``model_fields`` incl. seed."""

    model_fields = {"max_tokens": None, "seed": None, "temperature": None, "top_p": None}


def _seq(tokens, logprobs=None, stop_reason=None):
    seq = SimpleNamespace(tokens=list(tokens))
    if logprobs is not None:
        seq.logprobs = list(logprobs)
    if stop_reason is not None:
        seq.stop_reason = stop_reason
    return seq


@contextlib.contextmanager
def _runtime(monkeypatch, *, sample_fn=None, seeded=False):
    """Fake W&B/Hub/Tinker.  forward_backward_custom runs the real loss closure
    on fake logprobs that carry a prompt prefix, and records the gradients."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("HF_PUSH", raising=False)
    holder = {"params": [], "grads": [], "metrics": [], "logs": []}
    run = SimpleNamespace(id="wandb-cal", mode="online", summary={}, logs=holder["logs"])
    run.log = lambda payload: holder["logs"].append(payload) or True
    run.finish = Mock()

    class SamplingClient:
        def sample(self, prompt, num_samples=1, sampling_params=None):
            holder["params"].append(sampling_params)
            if sample_fn is not None:
                return _Future(sample_fn(prompt, num_samples, sampling_params))
            return _Future(SimpleNamespace(sequences=[_seq([1, 7]), _seq([2, 7])]))

    class TrainingClient:
        model_id = "tinker-cal"

        def save_weights_for_sampler(self, name):
            return _Future(SimpleNamespace(path=f"tinker://sampler/{name}"))

        def create_sampling_client(self, model_path):
            return SamplingClient()

        def forward_backward_custom(self, data, loss_fn, loss_type_input="logprobs"):
            lps = []
            for datum in data:
                n_targets = len(datum.loss_fn_inputs["target_tokens"].data)
                n_prompt_targets = len(PROMPT) - 1
                values = [PROMPT_LP] * n_prompt_targets + [RESP_LP] * (n_targets - n_prompt_targets)
                lps.append(torch.tensor(values, requires_grad=True))
            loss, metrics = loss_fn(data, lps)
            loss.backward()
            holder["grads"].append([lp.grad.clone() for lp in lps])
            holder["metrics"].append(dict(metrics))
            return _Future(SimpleNamespace(metrics=dict(metrics)))

        def optim_step(self, _params):
            return _Future(None)

        def save_state(self, name, overwrite):
            return _Future(SimpleNamespace(path=f"tinker://state/{name}"))

    class ServiceClient:
        def __init__(self, **_kwargs):
            self.client = TrainingClient()

        def create_lora_training_client(self, **_kwargs):
            return self.client

        def create_training_client_from_state_with_optimizer(self, path):
            return self.client

    class HfApi:
        info_count = 0

        def __init__(self, **_kwargs):
            pass

        def whoami(self, **_kwargs):
            return {"name": "owner"}

        def model_info(self, _repo_id, revision):
            type(self).info_count += 1
            return SimpleNamespace(sha=f"{type(self).info_count:040x}")

        def create_repo(self, **_kwargs):
            return None

        def create_branch(self, **_kwargs):
            return None

    wandb = types.ModuleType("wandb")
    wandb.init = Mock(return_value=run)
    hf = types.ModuleType("huggingface_hub")
    hf.HfApi = HfApi
    tinker = types.ModuleType("tinker")
    tinker.ServiceClient = ServiceClient
    tt = types.ModuleType("tinker.types")
    tt.ModelInput = SimpleNamespace(from_ints=lambda values: values)
    tt.TensorData = lambda **kwargs: SimpleNamespace(**kwargs)
    tt.Datum = lambda **kwargs: SimpleNamespace(**kwargs)
    tt.SamplingParams = _SeededParams if seeded else (lambda **kwargs: SimpleNamespace(**kwargs))
    tt.AdamParams = lambda **kwargs: SimpleNamespace(**kwargs)
    tinker.types = tt
    modules = {"wandb": wandb, "huggingface_hub": hf, "tinker": tinker, "tinker.types": tt}
    with patch.dict(sys.modules, modules, clear=False):
        with patch.object(grpo.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            yield holder


def _config(tmp_path, **kwargs):
    base = dict(name="cal", steps=1, group_size=2, batch_size=1, checkpoint_dir=str(tmp_path))
    base.update(kwargs)
    return GRPOConfig(**base)


def _run(config, dataset=None, logs=None):
    dataset = dataset or InMemoryDataset(train=[_example()])
    return grpo._run_one_seed(
        config,
        dataset,
        ExactMathReward(),
        _Tokenizer(),
        logger=(logs.append if logs is not None else (lambda _m: None)),
    )


# ------------------------------------------------- (1) prompt mask, every path


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"mask_truncated_responses": True, "max_response_tokens": 3},
        {"global_advantage_normalization": True},
        {"critic_enabled": True, "critic_updates_per_step": 0},
        {"nll_aux_enabled": True, "nll_aux_coef": 0.5},
        {"gspo_enabled": True},
    ],
    ids=["per-group", "per-group-trunc", "global", "critic", "nll-aux", "gspo"],
)
def test_training_paths_put_no_gradient_on_prompt_tokens(tmp_path, monkeypatch, kwargs):
    with _runtime(monkeypatch) as holder:
        _run(_config(tmp_path, **kwargs))
    assert holder["grads"], "loss closure never ran"
    n_prompt_targets = len(PROMPT) - 1
    for grads in holder["grads"]:
        for grad in grads:
            assert torch.all(grad[:n_prompt_targets] == 0), grad
    # Something still trains on the response tokens (not every path is a no-op).
    assert any(
        torch.any(grad[n_prompt_targets:] != 0) for grads in holder["grads"] for grad in grads
    )


# ---------------------------------------------------- (2) GSPO vs sampler lps


def test_gspo_uses_stale_sampler_logprobs_and_clips(tmp_path, monkeypatch):
    def sample_fn(_prompt, _n, _sp):
        # Sampler (behaviour policy) logprob -2 per token; trainer says -1.
        return SimpleNamespace(
            sequences=[_seq([1, 7], logprobs=[-2.0, -2.0]), _seq([2, 7], logprobs=[-2.0, -2.0])]
        )

    with _runtime(monkeypatch, sample_fn=sample_fn) as holder:
        _run(_config(tmp_path, gspo_enabled=True))
    assert holder["metrics"][0]["gspo_clip_frac"] > 0
    step_log = next(item for item in holder["logs"] if "train/gspo_clip_frac" in item)
    assert step_log["train/gspo_clip_frac"] == 1.0
    assert step_log["train/gspo_sampler_old"] == 1.0


def test_gspo_without_sampler_logprobs_falls_back_and_logs(tmp_path, monkeypatch):
    logs = []
    with _runtime(monkeypatch) as holder:
        _run(_config(tmp_path, gspo_enabled=True), logs=logs)
    assert holder["metrics"][0]["gspo_clip_frac"] == 0.0
    assert any("GSPO falls back" in message for message in logs)
    step_log = next(item for item in holder["logs"] if "train/gspo_sampler_old" in item)
    assert step_log["train/gspo_sampler_old"] == 0.0


# --------------------------------------------------- (3) held-out failures


def test_heldout_failures_score_zero_and_are_logged(tmp_path, monkeypatch):
    calls = {"heldout": 0}

    def sample_fn(_prompt, _n, sp):
        if sp.temperature == 0.1:
            calls["heldout"] += 1
            if calls["heldout"] == 1:
                raise RuntimeError("sampler blip")
            return SimpleNamespace(sequences=[_seq([1])])
        return SimpleNamespace(sequences=[_seq([1, 7]), _seq([2, 7])])

    dataset = InMemoryDataset(train=[_example()], test=[_example(), _example()])
    logs = []
    with _runtime(monkeypatch, sample_fn=sample_fn) as holder:
        result = _run(_config(tmp_path, evaluate_heldout=True), dataset=dataset, logs=logs)
    assert result.heldout_reward == 0.5
    summary = next(item for item in holder["logs"] if "test/failed_n" in item)
    assert summary["test/failed_n"] == 1.0
    assert summary["test/total_n"] == 2.0
    assert summary["test/reward_n"] == 2.0
    assert any("held-out example 0 failed" in message for message in logs)


# --------------------------------------------------- (4) seeded sampling


def test_sampling_is_seeded_per_step_and_index(tmp_path, monkeypatch):
    with _runtime(monkeypatch, seeded=True) as holder:
        _run(
            _config(tmp_path, steps=2, batch_size=2, seed_sampling=True),
            dataset=InMemoryDataset(train=[_example(), _example(prompt="r")]),
        )
    seeds = [params.seed for params in holder["params"]]
    assert len(seeds) == 4
    assert len(set(seeds)) == 4
    assert seeds[0] == grpo._sampling_seed(42, 0, 0)
    assert seeds[3] == grpo._sampling_seed(42, 1, 1)


def test_sampling_seed_skipped_when_params_have_no_seed_field(tmp_path, monkeypatch):
    with _runtime(monkeypatch) as holder:
        _run(_config(tmp_path))
    assert all(not hasattr(params, "seed") for params in holder["params"])


def test_sampling_is_unseeded_by_default(tmp_path, monkeypatch):
    with _runtime(monkeypatch, seeded=True) as holder:
        _run(_config(tmp_path))
    assert all(getattr(params, "seed", None) is None for params in holder["params"])


# ------------------------------------------- (7) stop_reason truncation


def test_truncation_uses_stop_reason_when_available():
    config = GRPOConfig(name="t", group_size=3, max_response_tokens=2)
    sampler = SimpleNamespace(
        sample=lambda *a, **k: _Future(
            SimpleNamespace(
                sequences=[
                    _seq([1, 2], stop_reason="stop"),  # EOS exactly at budget
                    _seq([1, 2], stop_reason="length"),
                    _seq([1, 2]),  # no stop_reason -> length heuristic
                ]
            )
        )
    )
    types_mod = SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ModelInput=SimpleNamespace(from_ints=lambda values: values),
    )
    meta = {}
    grpo._sample_scored_group(
        _Tokenizer(), sampler, types_mod, config, ExactMathReward(), _example(), meta=meta
    )
    assert meta["truncated"] == [False, True, True]
    assert meta["sampler_logprobs"] == [None, None, None]


# --------------------------------- (6) dynamic sampling reward + trace


def test_dynamic_sampling_logs_all_groups_and_keeps_trace_aligned(tmp_path, monkeypatch):
    # Every group is all-correct -> degenerate -> every step skipped.
    def sample_fn(_prompt, _n, _sp):
        return SimpleNamespace(sequences=[_seq([1, 7]), _seq([1, 8])])

    config = _config(tmp_path, steps=3, dynamic_sampling=True, dynamic_sampling_max_resamples=1)
    with _runtime(monkeypatch, sample_fn=sample_fn) as holder:
        result = _run(config)
    assert result.reward_trace == [1.0, 1.0, 1.0]
    skipped = [item for item in holder["logs"] if item.get("train/skipped_step") == 1.0]
    assert len(skipped) == 3
    assert skipped[0]["train/reward"] == 1.0
    assert skipped[0]["train/skipped_degenerate_groups"] == 2.0
    assert not holder["grads"]


def test_dynamic_sampling_reward_includes_degenerate_groups(tmp_path, monkeypatch):
    calls = {"n": 0}

    def sample_fn(_prompt, _n, _sp):
        calls["n"] += 1
        if calls["n"] == 1:  # degenerate all-correct group, then a mixed one
            return SimpleNamespace(sequences=[_seq([1, 7]), _seq([1, 8])])
        return SimpleNamespace(sequences=[_seq([1, 7]), _seq([2, 7])])

    config = _config(tmp_path, dynamic_sampling=True, dynamic_sampling_max_resamples=1)
    with _runtime(monkeypatch, sample_fn=sample_fn) as holder:
        result = _run(config)
    step_log = next(item for item in holder["logs"] if "train/reward_trained" in item)
    assert step_log["train/reward"] == pytest.approx(0.75)
    assert step_log["train/reward_trained"] == pytest.approx(0.5)
    assert result.reward_trace == [pytest.approx(0.75)]


# --------------------------------------------- (5) resume backfill


def _write_stored(config, stored):
    path = grpo._checkpoint_path(config, config.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": stored, "step": 3, "reward_trace": []}))


def test_resume_backfills_missing_keys_with_field_defaults(tmp_path):
    old = GRPOConfig(name="bf", checkpoint_dir=str(tmp_path))
    stored = json.loads(json.dumps(grpo._config_fingerprint(old, old.seed)))
    del stored["gspo_enabled"]
    _write_stored(old, stored)

    # Old receipt predates gspo_enabled -> it ran under the default (False).
    assert grpo._load_checkpoint(old, old.seed)["config"]["gspo_enabled"] is False
    switched = GRPOConfig(name="bf", checkpoint_dir=str(tmp_path), gspo_enabled=True)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        grpo._load_checkpoint(switched, switched.seed)


def test_resume_backfill_handles_default_factory_fields(tmp_path):
    config = GRPOConfig(name="bf2", checkpoint_dir=str(tmp_path))
    stored = json.loads(json.dumps(grpo._config_fingerprint(config, config.seed)))
    factory_keys = [
        f.name
        for f in grpo.fields(GRPOConfig)
        if f.default is grpo.MISSING and f.default_factory is not grpo.MISSING
    ]
    for key in factory_keys:
        stored.pop(key, None)
    _write_stored(config, stored)
    assert grpo._load_checkpoint(config, config.seed) is not None


# --------------------------------------------------- (8) CLI reward lookup


def test_cli_unknown_reward_fails(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(grpo_cli, "run_grpo", Mock(side_effect=AssertionError("ran")))
    monkeypatch.setattr(grpo_cli, "_build_dataset", Mock(side_effect=AssertionError("loaded")))
    with pytest.raises(SystemExit, match="Unknown reward: 'not-a-real-reward'"):
        grpo_cli.main(["--preset", "tooluse_synth", "--reward", "not-a-real-reward"])


def test_cli_reward_follows_dataset(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    seen = {}

    def fake_run(_config, _dataset, reward):
        seen["reward"] = reward
        return []

    monkeypatch.setattr(grpo_cli, "run_grpo", fake_run)
    monkeypatch.setattr(grpo_cli, "_build_dataset", lambda *_a, **_k: object())
    assert grpo_cli.main(["--preset", "tooluse_synth", "--dataset", "gsm8k"]) == 0
    assert isinstance(seen["reward"], ExactMathReward)


def test_explicit_reward_overrides_the_preset(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    seen = {}

    def fake_run(_config, _dataset, reward):
        seen["reward"] = reward
        return []

    monkeypatch.setattr(grpo_cli, "run_grpo", fake_run)
    monkeypatch.setattr(grpo_cli, "_build_dataset", lambda *_a, **_k: object())
    assert grpo_cli.main(["--preset", "tooluse_synth", "--reward", "math100"]) == 0
    assert isinstance(seen["reward"], MathReward)
