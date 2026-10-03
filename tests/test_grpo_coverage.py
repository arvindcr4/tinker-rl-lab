"""Statement-coverage tests for branches the main GRPO suites do not select."""

from __future__ import annotations

import contextlib
import json
import runpy
import sys
import types
from dataclasses import asdict

import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from platform_tinker.tinkerrl import grpo, grpo_cli
from platform_tinker.tinkerrl.grpo import (
    ExactMathReward,
    GRPOConfig,
    GRPORunResult,
    InMemoryDataset,
    MathReward,
    PatchReward,
    StrictToolCallReward,
)
from platform_tinker.tinkerrl.grpo_cli import _build_dataset, _parse_args, build_config
from tests._shared_fakes import HfApi, _Future, _Tokenizer, _config, _example


def _dataset(*examples, test=()):
    return InMemoryDataset(train=list(examples), test=list(test))


def _receipt(step=1, **overrides):
    repo = "owner/model"
    revision = f"rev-{step}"
    sha = "c" * 40
    receipt = {
        "step": step,
        "repo_id": repo,
        "revision": revision,
        "commit_sha": sha,
        "repo_url": f"https://huggingface.co/{repo}",
        "revision_url": f"https://huggingface.co/{repo}/tree/{revision}",
        "commit_url": f"https://huggingface.co/{repo}/commit/{sha}",
        "source_path": "tinker://source",
    }
    receipt.update(overrides)
    return receipt


def _write_prior(config, **payload):
    stored = json.loads(json.dumps(grpo._config_fingerprint(config, config.seed)))
    body = {"config": stored, "step": 0}
    body.update(payload)
    grpo._write_checkpoint(grpo._checkpoint_path(config, config.seed), body)


def _quiet_run():
    run = SimpleNamespace(id="wandb-cov", mode="online", summary={}, logs=[])

    def log(payload):
        run.logs.append(payload)
        return True

    run.log = log
    run.finish = Mock()
    return run


class _Seq:
    def __init__(self, tokens):
        self.tokens = list(tokens)


@contextlib.contextmanager
def _runtime(monkeypatch, *, metrics=None, sample_fn=None, install_transformers=False):
    """Install fake W&B, Hub, and Tinker modules for one seed."""
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("HF_PUSH", raising=False)
    holder = {"events": []}
    metrics = dict(metrics or {"grpo_loss": 0.25})
    run = _quiet_run()
    holder["run"] = run

    class SamplingClient:
        def sample(self, prompt, num_samples=1, sampling_params=None):
            holder["events"].append(("sample", num_samples, sampling_params, prompt))
            if sample_fn is not None:
                return _Future(sample_fn(prompt, num_samples, sampling_params))
            return _Future(SimpleNamespace(sequences=[_Seq([1]), _Seq([2])]))

    class TrainingClient:
        model_id = "tinker-run-1"

        def save_weights_for_sampler(self, name):
            holder["events"].append(("save_sampler", name))
            return _Future(SimpleNamespace(path=f"tinker://sampler/{name}"))

        def create_sampling_client(self, model_path):
            return SamplingClient()

        def forward_backward_custom(self, **_kwargs):
            holder["events"].append("forward")
            return _Future(SimpleNamespace(metrics=dict(metrics)))

        def optim_step(self, _params):
            holder["events"].append("optim")
            return _Future(None)

        def save_state(self, name, overwrite):
            return _Future(SimpleNamespace(path=f"tinker://state/{name}"))

    class ServiceClient:
        def __init__(self, **kwargs):
            holder["service_kwargs"] = kwargs
            self.client = TrainingClient()

        def create_lora_training_client(self, **kwargs):
            holder["lora_kwargs"] = kwargs
            return self.client

        def create_training_client_from_state_with_optimizer(self, path):
            holder["resume_path"] = path
            return self.client

    wandb = types.ModuleType("wandb")
    wandb.init = Mock(return_value=run)

    hf = types.ModuleType("huggingface_hub")
    hf.HfApi = HfApi
    tinker = types.ModuleType("tinker")
    tinker.ServiceClient = ServiceClient
    tinker_types = types.ModuleType("tinker.types")
    tinker_types.ModelInput = SimpleNamespace(from_ints=lambda values: values)
    tinker_types.TensorData = lambda **kwargs: SimpleNamespace(**kwargs)
    tinker_types.Datum = lambda **kwargs: SimpleNamespace(**kwargs)
    tinker_types.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    tinker_types.AdamParams = lambda **kwargs: SimpleNamespace(**kwargs)
    tinker.types = tinker_types
    modules = {
        "wandb": wandb,
        "huggingface_hub": hf,
        "tinker": tinker,
        "tinker.types": tinker_types,
    }
    if install_transformers:

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model, **kwargs):
                holder["pretrained"] = (model, kwargs)
                return _Tokenizer()

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer
        modules["transformers"] = transformers

    with patch.dict(sys.modules, modules, clear=False):
        with patch.object(grpo.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
            yield holder


def test_pending_suite_receipt_is_unfrozen():
    receipt = grpo._pending_suite_receipt()
    assert receipt["frozen"] is False
    assert set(receipt) == {
        "frozen",
        "split",
        "hash",
        "license",
        "runtime",
        "decontamination",
        "source",
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wandb_enabled": False}, "cannot be disabled"),
        ({"wandb_group": None}, "wandb_group"),
        ({"wandb_mode": "offline"}, "online mode"),
        ({"wandb_tags": ()}, "tags must be configured"),
    ],
)
def test_tracking_rejects_incomplete_wandb_setup(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GRPOConfig(name="track", **kwargs).validate_tracking()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"campaign_status": ""}, "status is required when supplied"),
        ({"budget_status": "  "}, "budget status is required when supplied"),
        ({"budget_status": "nope"}, "authorized budget status"),
        ({"paid_jobs_may_launch": False}, "disables paid jobs"),
        ({"authorized_budget_usd": True}, "must be numeric"),
        ({"authorized_budget_usd": 0}, "must be positive"),
        ({"maximum_usd": float("nan")}, "must be numeric"),
        ({"maximum_usd": -1}, "must be positive"),
        ({"authorized_budget_usd": 5, "maximum_usd": 1}, "exceeds"),
        ({"paid_jobs_may_launch": True}, "launchable campaign status"),
        (
            {"paid_jobs_may_launch": True, "campaign_status": "ready"},
            "authorized_budget_usd and maximum_usd",
        ),
    ],
)
def test_campaign_gate_rejects_incomplete_receipts(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GRPOConfig(name="gate", **kwargs).validate_campaign_gate()


def test_sample_scored_group_truncates_long_prompts():
    seen = {}

    class Tok:
        def encode(self, _prompt, add_special_tokens=False):
            return list(range(10))

        def decode(self, _tokens, skip_special_tokens=True):
            return "1"

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            seen["prompt"] = prompt
            seen["params"] = sampling_params
            return _Future(SimpleNamespace(sequences=[_Seq([9, 9])]))

    types_mod = SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ModelInput=SimpleNamespace(from_ints=lambda values: values),
    )
    config = GRPOConfig(name="trunc", max_prompt_tokens=3, group_size=1, max_response_tokens=4)
    prompt_ids, resp_ids, rewards, lengths = grpo._sample_scored_group(
        Tok(),
        Sampler(),
        types_mod,
        config,
        ExactMathReward(),
        _example(),
    )

    assert prompt_ids == [0, 1, 2]
    assert seen["prompt"] == [0, 1, 2]
    assert seen["params"].max_tokens == 4
    assert resp_ids == [[9, 9]]
    assert rewards == [1.0]
    assert lengths == [2]


def test_wandb_finalizers_fail_closed_on_missing_or_broken_runs():
    with pytest.raises(RuntimeError, match="final status is inadmissible"):
        grpo._finish_wandb(None, success=True)
    with pytest.raises(RuntimeError, match="failure status is inadmissible"):
        grpo._mark_wandb_failure(None, RuntimeError("x"))

    class Boom(dict):
        def __setitem__(self, key, value):
            raise RuntimeError(f"locked {key}")

    broken = SimpleNamespace(summary=Boom(), finish=lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match="summary status update failed"):
        grpo._finish_wandb(broken, success=False)
    with pytest.raises(RuntimeError, match="failure status update failed"):
        grpo._mark_wandb_failure(broken, RuntimeError("boom"))

    no_finish = SimpleNamespace(summary={})
    with pytest.raises(RuntimeError, match="no finish method"):
        grpo._finish_wandb(no_finish, success=True)

    created = SimpleNamespace()
    assert grpo._wandb_summary(created) == {}
    assert created.summary == {}

    class Locked:
        @property
        def summary(self):
            return None

    with pytest.raises(RuntimeError, match="summary is unavailable"):
        grpo._wandb_summary(Locked())

    with pytest.raises(RuntimeError, match="no log method"):
        grpo._wandb_log(SimpleNamespace(), {"a": 1})

    def exploding(_payload):
        raise RuntimeError("log down")

    with pytest.raises(RuntimeError, match="W&B log failed"):
        grpo._wandb_log(SimpleNamespace(log=exploding), {"a": 1})
    with pytest.raises(RuntimeError, match="log was rejected"):
        grpo._wandb_log(SimpleNamespace(log=lambda _payload: False), {"a": 1})


def test_start_wandb_rejects_disabled_env_import_and_bad_runs(monkeypatch):
    config = GRPOConfig(name="wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(RuntimeError, match="WANDB_MODE=online"):
        grpo._start_wandb(config, 0)

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DISABLED", "yes")
    with pytest.raises(RuntimeError, match="WANDB_DISABLED"):
        grpo._start_wandb(config, 0)

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    with patch.dict(sys.modules, {"wandb": None}):
        with pytest.raises(RuntimeError, match="W&B dependency is unavailable"):
            grpo._start_wandb(config, 0)

    missing = types.ModuleType("wandb")
    missing.init = lambda **_kwargs: None
    with patch.dict(sys.modules, {"wandb": missing}):
        with pytest.raises(RuntimeError, match="no live run"):
            grpo._start_wandb(config, 0)

    offline = SimpleNamespace(id="id", mode="offline", summary={}, finish=Mock())
    offline_mod = types.ModuleType("wandb")
    offline_mod.init = lambda **_kwargs: offline
    with patch.dict(sys.modules, {"wandb": offline_mod}):
        with pytest.raises(RuntimeError, match="non-online"):
            grpo._start_wandb(config, 0)
    assert offline.summary["status"] == "failed"
    offline.finish.assert_called_once_with(exit_code=1)

    disabled = SimpleNamespace(id="id", mode="online", disabled=True, summary={}, finish=Mock())
    disabled_mod = types.ModuleType("wandb")
    disabled_mod.init = lambda **_kwargs: disabled
    with patch.dict(sys.modules, {"wandb": disabled_mod}):
        with pytest.raises(RuntimeError, match="disabled run"):
            grpo._start_wandb(config, 0)
    disabled.finish.assert_called_once_with(exit_code=1)


def test_huggingface_helpers_cover_import_auth_and_receipt_failures(monkeypatch):
    config = GRPOConfig(name="hf")
    monkeypatch.delenv("HF_PUSH", raising=False)
    monkeypatch.delenv("HF_REPO_OWNER", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    with patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(RuntimeError, match="Hugging Face dependency is unavailable"):
            grpo._make_hf_api()

    class TokenFreeApi:
        def __init__(self, *args, **kwargs):
            if kwargs:
                raise TypeError("token is not a parameter")
            self.ok = True

    token_free = types.ModuleType("huggingface_hub")
    token_free.HfApi = TokenFreeApi
    with patch.dict(sys.modules, {"huggingface_hub": token_free}):
        api, _token = grpo._make_hf_api()
    assert api.ok is True

    class ExplodingApi:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("hub down")

    exploding = types.ModuleType("huggingface_hub")
    exploding.HfApi = ExplodingApi
    with patch.dict(sys.modules, {"huggingface_hub": exploding}):
        with pytest.raises(RuntimeError, match="API initialization failed"):
            grpo._make_hf_api()

    monkeypatch.setenv("HF_PUSH", "off")
    with pytest.raises(RuntimeError, match="cannot be disabled"):
        grpo._preflight_hf(config)
    with pytest.raises(RuntimeError, match="cannot be disabled"):
        grpo._publish_checkpoint(config, 1, "tinker://x", lambda _message: None)
    monkeypatch.delenv("HF_PUSH", raising=False)
    with pytest.raises(RuntimeError, match="empty Tinker"):
        grpo._publish_checkpoint(config, 1, "", lambda _message: None)

    class WhoamiApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self, *args, **kwargs):
            if "token" in kwargs:
                raise TypeError("no token kw")
            return WhoamiApi.identity

    whoami_mod = types.ModuleType("huggingface_hub")
    whoami_mod.HfApi = WhoamiApi
    with patch.dict(sys.modules, {"huggingface_hub": whoami_mod}):
        WhoamiApi.identity = {"user": {"name": "nested-owner"}}
        assert grpo._preflight_hf(config) == "nested-owner"
        WhoamiApi.identity = {"user": "string-owner"}
        assert grpo._preflight_hf(config) == "string-owner"
        WhoamiApi.identity = {}
        with pytest.raises(RuntimeError, match="no identity"):
            grpo._preflight_hf(config)
        WhoamiApi.identity = {"id": 1}
        with pytest.raises(RuntimeError, match="no owner"):
            grpo._preflight_hf(config)

    class InfoApi:
        payload = {"sha": " deadbeef "}

        def __init__(self, token=None):
            pass

        def model_info(self, _repo_id, revision):
            return dict(InfoApi.payload)

        def create_repo(self, **_kwargs):
            raise RuntimeError("create failed")

        def create_branch(self, **_kwargs):
            return None

    info_mod = types.ModuleType("huggingface_hub")
    info_mod.HfApi = InfoApi
    with patch.dict(sys.modules, {"huggingface_hub": info_mod}):
        assert grpo._verify_hf_revision("owner/model", "rev") == "deadbeef"
        InfoApi.payload = {"sha": "  "}
        with pytest.raises(RuntimeError, match="no commit SHA"):
            grpo._verify_hf_revision("owner/model", "rev")
        with pytest.raises(RuntimeError, match="revision preparation failed"):
            grpo._prepare_hf_revision(config, "owner/model", "rev")

    monkeypatch.setenv("HF_REPO_OWNER", "hub-owner")
    prefixed = GRPOConfig(name="n", hf_repo_prefix="Someone/Pretty Name")
    repo_id = grpo._checkpoint_repo_id(prefixed, "fallback-owner", 3, 4, "tinker://path")
    assert repo_id.startswith("hub-owner/")
    assert "pretty-name" in repo_id

    with pytest.raises(RuntimeError, match="receipt missing"):
        grpo._require_checkpoint_receipt(["not-a-dict"], step=1)
    incomplete = _receipt()
    incomplete["source_path"] = ""
    with pytest.raises(RuntimeError, match="receipt incomplete"):
        grpo._require_checkpoint_receipt(incomplete, step=1)
    bad_repo = _receipt(repo_url="https://example.invalid/owner/model")
    with pytest.raises(RuntimeError, match="invalid repo URL"):
        grpo._require_checkpoint_receipt(bad_repo, step=1)
    bad_commit = _receipt(commit_url="https://example.invalid/commit")
    with pytest.raises(RuntimeError, match="invalid commit URL"):
        grpo._require_checkpoint_receipt(bad_commit, step=1)

    with patch.object(grpo, "_prepare_hf_revision"):
        with patch.object(grpo.subprocess, "run", side_effect=FileNotFoundError("missing")):
            with pytest.raises(RuntimeError, match="Tinker CLI is unavailable"):
                grpo._publish_checkpoint(
                    config, 1, "tinker://source", lambda _message: None, step=4, hf_owner="owner"
                )
        with patch.object(
            grpo.subprocess, "run", side_effect=RuntimeError("secret-token-should-not-leak")
        ):
            with pytest.raises(RuntimeError, match=r"export failed for step 5$") as raised:
                grpo._publish_checkpoint(
                    config, 1, "tinker://source", lambda _message: None, step=5, hf_owner="owner"
                )
    assert "secret-token" not in str(raised.value)


def test_campaign_log_records_budget_and_heldout_suites():
    config = GRPOConfig(
        name="camp",
        campaign_status="authorized",
        budget_status="authorized",
        heldout_suite_ids=("suite-a",),
    )
    run = _quiet_run()
    grpo._log_campaign_metadata(run, config)
    assert run.logs[0]["campaign/budget_status"] == "authorized"
    assert run.logs[0]["campaign/heldout_suite_ids"] == ["suite-a"]
    assert run.summary["budget_status"] == "authorized"
    assert run.summary["heldout_suite_ids"] == ["suite-a"]


def test_empty_dataset_fails_before_wandb(tmp_path):
    class Empty:
        def train_examples(self):
            return []

        def test_examples(self):
            return []

    with pytest.raises(ValueError, match="0 training examples"):
        grpo._run_one_seed(_config(tmp_path), Empty(), ExactMathReward(), tokenizer=_Tokenizer())


def _completed_config(tmp_path, result):
    config = _config(tmp_path, steps=1)
    _write_prior(
        config,
        status="completed",
        step=1,
        result=json.loads(json.dumps(asdict(result))),
    )
    return config


def test_completed_receipt_with_checkpoints_returns_without_tinker(tmp_path, monkeypatch):
    result = GRPORunResult(
        seed=42,
        run_id="tinker-done",
        sampler_path="tinker://final",
        reward_trace=[0.5],
        checkpoint_receipts=[_receipt(step="final")],
    )
    config = _completed_config(tmp_path, result)
    run = _quiet_run()
    monkeypatch.setattr(grpo, "_start_wandb", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(grpo, "_preflight_hf", lambda _config: "owner")
    with patch.dict(sys.modules, {"tinker": None}):
        completed = grpo._run_one_seed(config, _dataset(_example()), ExactMathReward(), object())

    assert completed.run_id == "tinker-done"
    assert run.summary["tinker_run_id"] == "tinker-done"
    assert run.summary["checkpoint_receipts"][0]["commit_sha"] == "c" * 40
    run.finish.assert_called_once_with(exit_code=0)


def test_completed_legacy_urls_are_retained(tmp_path, monkeypatch):
    result = GRPORunResult(
        seed=42,
        run_id="tinker-legacy",
        checkpoint_urls=["https://huggingface.co/legacy/tree/old"],
    )
    config = _completed_config(tmp_path, result)
    run = _quiet_run()
    monkeypatch.setattr(grpo, "_start_wandb", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(grpo, "_preflight_hf", lambda _config: "owner")
    completed = grpo._run_one_seed(config, _dataset(_example()), ExactMathReward(), object())

    assert completed.checkpoint_urls == ["https://huggingface.co/legacy/tree/old"]
    assert run.summary["checkpoint_urls"] == ["https://huggingface.co/legacy/tree/old"]
    assert "checkpoint_commit_shas" not in run.summary


def test_completed_receipt_without_run_id_fails_closed(tmp_path, monkeypatch):
    result = GRPORunResult(seed=42, run_id="")
    config = _completed_config(tmp_path, result)
    run = _quiet_run()
    monkeypatch.setattr(grpo, "_start_wandb", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(grpo, "_preflight_hf", lambda _config: "owner")
    with pytest.raises(RuntimeError, match="no nonempty Tinker run ID"):
        grpo._run_one_seed(config, _dataset(_example()), ExactMathReward(), object())
    assert run.summary["status"] == "failed"


def test_tinker_and_tokenizer_imports_fail_closed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    dataset = _dataset(_example())
    run = _quiet_run()
    monkeypatch.setattr(grpo, "_start_wandb", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(grpo, "_preflight_hf", lambda _config: "owner")
    with patch.dict(sys.modules, {"tinker": None}):
        with pytest.raises(RuntimeError, match="Tinker dependency is unavailable"):
            grpo._run_one_seed(config, dataset, ExactMathReward(), tokenizer=_Tokenizer())

    tinker = types.ModuleType("tinker")
    tinker_types = types.ModuleType("tinker.types")
    tinker.types = tinker_types
    with patch.dict(
        sys.modules, {"tinker": tinker, "tinker.types": tinker_types, "transformers": None}
    ):
        with pytest.raises(RuntimeError, match="Tokenizer dependency is unavailable"):
            grpo._run_one_seed(config, dataset, ExactMathReward(), tokenizer=None)


def test_resume_uses_saved_optimizer_state(tmp_path, monkeypatch):
    config = _config(tmp_path, steps=1)
    _write_prior(config, status="started", step=1, train_state_path="tinker://state/resume")
    logs = []
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config,
            _dataset(_example()),
            ExactMathReward(),
            _Tokenizer(),
            logger=logs.append,
        )
    assert result.resumed_from_step == 1
    assert any("Resuming from step 1" in message for message in logs)


def test_per_group_shaping_logs_nll_zero_loss_and_skips(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        mask_truncated_responses=True,
        nll_aux_enabled=True,
        nll_aux_coef=0.1,
        dynamic_sampling=True,
    )
    with _runtime(monkeypatch, metrics={"grpo_loss": 0.0, "nll_loss": 0.2}) as holder:
        result = grpo._run_one_seed(
            config, _dataset(_example()), ExactMathReward(), _Tokenizer(), logger=lambda _m: None
        )
    assert result.zero_loss_steps == 1
    logged = next(item for item in holder["run"].logs if "train/nll_loss" in item)
    assert logged["train/nll_loss"] == 0.2
    assert logged["train/nll_frac"] == 0.5
    assert logged["train/skipped_degenerate_groups"] == 0.0


def test_critic_masks_truncated_responses_out_of_the_fit(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        critic_enabled=True,
        critic_updates_per_step=0,
        mask_truncated_responses=True,
        max_response_tokens=1,
    )
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, _dataset(_example()), ExactMathReward(), _Tokenizer(), logger=lambda _m: None
        )
    assert result.reward_trace
    assert Path(str(tmp_path)).joinpath(f"{config.name}_seed{config.seed}.critic.pt").exists()


def test_fresh_run_pins_revisions_and_scores_heldout(tmp_path, monkeypatch):
    calls = {"heldout": 0}

    def sample_fn(prompt, num_samples, sampling_params):
        assert len(prompt) == 2
        if sampling_params.temperature == 0.1:
            calls["heldout"] += 1
            if calls["heldout"] == 1:
                raise RuntimeError("sampler blip")
            return SimpleNamespace(sequences=[_Seq([1])])
        return SimpleNamespace(sequences=[_Seq([1]), _Seq([2])])

    config = _config(
        tmp_path,
        evaluate_heldout=True,
        max_prompt_tokens=2,
        model_revision="model-rev",
        dataset_revision="data-rev",
    )
    dataset = _dataset(_example(), test=(_example(), _example("1")))
    with _runtime(monkeypatch, sample_fn=sample_fn, install_transformers=True) as holder:
        result = grpo._run_one_seed(
            config, dataset, ExactMathReward(), tokenizer=None, logger=lambda _m: None
        )

    assert holder["pretrained"] == (
        config.model,
        {"trust_remote_code": True, "revision": "model-rev"},
    )
    metadata = holder["lora_kwargs"]["user_metadata"]
    assert metadata["model_revision"] == "model-rev"
    assert metadata["dataset_revision"] == "data-rev"
    assert calls["heldout"] == 2
    # The sampler blip scores 0 and stays in the denominator.
    assert result.heldout_reward == 0.5
    summary = next(item for item in holder["run"].logs if "test/failed_n" in item)
    assert summary["test/failed_n"] == 1.0
    assert summary["test/total_n"] == 2.0
    assert summary["test/reward_n"] == 2.0
    assert summary["test/reward"] == 0.5


def test_wandb_failure_receipt_cannot_be_finalized(tmp_path, monkeypatch):
    class Boom(dict):
        def __setitem__(self, key, value):
            raise RuntimeError(f"locked {key}")

    run = SimpleNamespace(id="w", mode="online", summary=Boom(), finish=Mock())
    run.log = lambda _payload: True
    monkeypatch.setattr(grpo, "_start_wandb", lambda *_args, **_kwargs: run)
    with pytest.raises(RuntimeError, match="could not be finalized"):
        grpo._run_one_seed(_config(tmp_path), _dataset(_example()), ExactMathReward(), _Tokenizer())


def test_gsm8k_loader_skips_rows_without_a_final_answer():
    rows = {
        "train": [
            {"question": "What?", "answer": "work #### 1,234"},
            {"question": "Skip", "answer": "no marker"},
        ],
        "test": [{"question": "Held?", "answer": "#### 7"}],
    }

    def load_dataset(name, subset, split):
        assert (name, subset) == ("openai/gsm8k", "main")
        return rows[split]

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = load_dataset
    with patch.dict(sys.modules, {"datasets": datasets}):
        dataset = grpo.make_gsm8k_dataset(seed=1)

    assert [example.target for example in dataset.train_examples()] == ["1234"]
    assert [example.target for example in dataset.test_examples()] == ["7"]


def test_xlam_loader_skips_malformed_rows():
    rows = [
        {"tools": "[]", "answers": "[]", "query": "empty"},
        {"tools": "[]", "answers": {"name": "x"}, "query": "not-a-list"},
        {"tools": "[]", "answers": "NOPE", "query": "bad-json"},
        {"tools": [], "answers": [{"name": "", "arguments": {}}], "query": "no-tool"},
        {
            "tools": [],
            "answers": [{"name": "lookup", "arguments": '{"city": "Paris"}'}],
            "query": "weather?",
        },
        None,
    ]
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: rows
    with patch.dict(sys.modules, {"datasets": datasets}):
        dataset = grpo.make_xlam_dataset(seed=1)

    assert len(dataset.train_examples()) == 1
    assert dataset.train_examples()[0].target == {
        "tool": "lookup",
        "arguments": {"city": "Paris"},
    }


def _api_content(tool="search"):
    return (
        "**Available Tools**\n"
        f"1. Name: {tool}\nParameters: query: str\n"
        "**Output Format**\nJSON\n[USER] Find account 7"
    )


def test_api_bank_and_swe_gym_prompts_fail_closed():
    with pytest.raises(ValueError, match="not a literal"):
        grpo._api_bank_prompt("[", "search")
    with pytest.raises(ValueError, match="exactly one message"):
        grpo._api_bank_prompt(repr([]), "search")
    with pytest.raises(ValueError, match="exactly one message"):
        grpo._api_bank_prompt(repr([{"content": "x"}, {"content": "y"}]), "search")
    with pytest.raises(ValueError, match="missing tools"):
        grpo._api_bank_prompt(repr([{"content": "hello"}]), "search")
    duplicated = (
        "**Available Tools**\n1. Name: search\n2. Name: search\n**Output Format**\n[USER] hi"
    )
    with pytest.raises(ValueError, match="exactly once"):
        grpo._api_bank_prompt(repr([{"content": duplicated}]), "search")

    with pytest.raises(ValueError, match="not JSON"):
        grpo._api_bank_target("{")
    with pytest.raises(ValueError, match="missing name"):
        grpo._api_bank_target(json.dumps({"name": "", "parameters": {}}))
    with pytest.raises(ValueError, match="missing name"):
        grpo._api_bank_target(json.dumps(["not-an-object"]))

    with pytest.raises(ValueError, match="missing repo"):
        grpo._swe_gym_prompt({"problem_statement": "x", "repo": "o/r", "base_commit": "zz"})


def _pavlov_files(root, e4_text='{"task_id":"eval-e4"}\n'):
    e1 = root / "outputs/e1_swe_bench_pro/hf_dataset/data"
    e1.mkdir(parents=True)
    (e1 / "test-00000-of-00001.parquet").touch()
    (root / "outputs/e2_frontier_swe/frontier-swe/tasks").mkdir(parents=True)
    e4 = root / "outputs/e4_banker_toolbench/official_repo_ff6db552/native-data"
    e4.mkdir(parents=True)
    (e4 / "tasks.jsonl").write_text(e4_text)


def _swe_row(instance_id="swe-1"):
    return {
        "repo": "owner/repo",
        "instance_id": instance_id,
        "base_commit": "a" * 40,
        "problem_statement": "Fix it",
        "hints_text": "",
        "FAIL_TO_PASS": [],
        "PASS_TO_PASS": [],
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@\n-a\n+b\n",
    }


def _api_row(index, prompt=None):
    return {
        "prompt": prompt
        if prompt is not None
        else repr([{"role": "user", "content": _api_content()}]),
        "ground_truth": json.dumps({"name": "search", "parameters": {"query": "account 7"}}),
        "extra_info": json.dumps({"index": index}),
    }


class _Rows(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def test_pavlov_loader_rejects_missing_overlap_and_bad_counts(tmp_path):
    with pytest.raises(RuntimeError, match="decontamination input is missing"):
        grpo.make_pavlov_non_xlam_dataset(repo_root=tmp_path)

    def load(e1_rows, swe_rows, api_rows):
        def load_dataset(name, **_kwargs):
            if name == "parquet":
                return _Rows(e1_rows)
            if name == "SWE-Gym/SWE-Gym":
                return _Rows(swe_rows)
            if name == "Simu-Env/API-Bank-RLVR":
                return {"train": _Rows(api_rows), "validation": _Rows([])}
            raise AssertionError(name)

        datasets = types.ModuleType("datasets")
        datasets.load_dataset = load_dataset
        return datasets

    root = tmp_path / "overlap-swe"
    _pavlov_files(root)
    datasets = load(
        [{"instance_id": "swe-1", "repo": "eval/repo", "base_commit": "f" * 40}], [_swe_row()], []
    )
    with patch.dict(sys.modules, {"datasets": datasets}):
        with pytest.raises(RuntimeError, match="SWE-Gym contamination"):
            grpo.make_pavlov_non_xlam_dataset(repo_root=root)

    root = tmp_path / "overlap-api"
    _pavlov_files(root, e4_text='\n{"task_id": "7"}\n')
    datasets = load(
        [{"instance_id": "eval-e1", "repo": "eval/repo", "base_commit": "f" * 40}],
        [_swe_row()],
        [_api_row(7)],
    )
    with patch.dict(sys.modules, {"datasets": datasets}):
        with pytest.raises(RuntimeError, match="API-Bank contamination"):
            grpo.make_pavlov_non_xlam_dataset(repo_root=root)

    root = tmp_path / "short-mix"
    _pavlov_files(root, e4_text='\n{"task_id": "other"}\n')
    datasets = load(
        [{"instance_id": "eval-e1", "repo": "eval/repo", "base_commit": "f" * 40}],
        [_swe_row()],
        [_api_row(1, prompt="[]"), _api_row(2)],
    )
    with patch.dict(sys.modules, {"datasets": datasets}):
        with pytest.raises(RuntimeError, match="unexpected train/test counts"):
            grpo.make_pavlov_non_xlam_dataset(repo_root=root)


def test_gspo_rejects_empty_logprobs_and_shape_mismatch():
    with pytest.raises(ValueError, match="no token logprobs"):
        grpo.make_gspo_loss_fn([1.0])(None, [torch.tensor([])])
    with pytest.raises(ValueError, match="token-for-token"):
        grpo.make_gspo_loss_fn([1.0], old_logprobs=[torch.tensor([-0.5, -0.5])])(
            None, [torch.tensor([-0.5])]
        )


def test_reward_parsers_cover_rejected_shapes():
    strict = StrictToolCallReward()
    assert strict.score("", _example(target={"tool": "search", "arguments": {}})) == 0.0
    assert strict.score("   ", _example(target={"tool": "search", "arguments": {}})) == 0.0
    assert strict.score("{}", _example(target={})) == 0.0
    assert (
        strict.score(
            '{"tool":"search","arguments":{}}',
            _example(target={"tool": "search"}),
        )
        == 1.0
    )
    assert (
        strict.score(
            '{"tool":"search","arguments":{"a":1}}',
            _example(target={"tool": "search", "arguments": ["nope"]}),
        )
        == 0.0
    )
    nested = {"meta": {"b": 1, "a": [2]}}
    assert (
        strict.score(
            json.dumps({"tool": "search", "arguments": nested}),
            _example(target={"tool": "search", "arguments": nested}),
        )
        == 1.0
    )

    math_reward = MathReward()
    assert math_reward.score(r"\boxed{abc}", _example("42")) == 0.3
    assert math_reward.score(r"\boxed{3.140}", _example("3.14")) == 1.0
    assert math_reward.score("value 10", _example("abc")) == 0.0
    assert math_reward.score("approximately 3.140", _example("3.14")) == 1.0
    assert ExactMathReward().score(r"\boxed{abc}", _example("abc")) == 1.0

    patch = PatchReward()
    assert patch.score("   \n", _example(target="diff --git a/a.py b/a.py\n")) == 0.0
    assert patch.score("diff --git a/a.py b/a.py", _example(target="")) == 0.0


def test_cli_dataset_preset_and_flag_branches(tmp_path):
    synth = _parse_args(["--preset", "tooluse_synth", "--dataset", "not-a-dataset"])
    with pytest.raises(SystemExit, match="Unknown dataset"):
        _build_dataset(synth, build_config(_parse_args(["--preset", "tooluse_synth"])))

    xlam_args = _parse_args(["--preset", "tooluse_xlam"])
    xlam_config = build_config(xlam_args)
    assert xlam_config.dataset_revision is None
    with patch.dict(grpo_cli.DATASET_FACTORIES, {"tooluse_xlam": lambda **kwargs: kwargs}):
        assert _build_dataset(xlam_args, xlam_config) == {"seed": xlam_config.seed}

    math_args = _parse_args(["--preset", "math100"])
    with patch.dict(grpo_cli.DATASET_FACTORIES, {"math100": lambda **kwargs: ("bare", kwargs)}):
        assert _build_dataset(math_args, build_config(math_args)) == ("bare", {})

    flags = build_config(
        _parse_args(
            ["--preset", "tooluse_synth", "--evaluate-heldout", "--no-resume", "--hf-public"]
        )
    )
    assert flags.evaluate_heldout is True
    assert flags.resume is False
    assert flags.hf_public is True

    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"name": "from-json", "steps": 3}))
    loaded = build_config(_parse_args(["--json-config", str(path)]))
    assert loaded.name == "from-json"
    assert loaded.steps == 3

    unknown = _parse_args(["--preset", "tooluse_synth"])
    unknown.preset = "missing"
    with pytest.raises(SystemExit, match="Unknown preset"):
        build_config(unknown)


def test_main_reports_a_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert grpo_cli.main(["--preset", "tooluse_synth"]) == 1
    assert "TINKER_API_KEY" in capsys.readouterr().err


def test_main_checks_the_api_key_before_the_reward_name(monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert grpo_cli.main(["--preset", "tooluse_synth", "--reward", "not-a-real-reward"]) == 1
    assert "TINKER_API_KEY" in capsys.readouterr().err


def test_main_prints_a_completed_run(monkeypatch, capsys):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    result = GRPORunResult(
        seed=7,
        run_id="run-7",
        sampler_path="tinker://sampler",
        reward_trace=[0.0, 0.5],
        avg_first5=0.1,
        avg_last10=0.2,
        peak_reward=0.5,
        zero_loss_steps=1,
        zero_reward_steps=2,
        heldout_reward=0.25,
    )
    bare = GRPORunResult(
        seed=8,
        run_id="run-8",
        sampler_path="tinker://other",
        reward_trace=[0.2],
    )
    monkeypatch.setattr(grpo_cli, "run_grpo", lambda *_args, **_kwargs: [result, bare])
    monkeypatch.setattr(grpo_cli, "_build_dataset", lambda *_args, **_kwargs: object())
    assert grpo_cli.main(["--preset", "tooluse_synth", "--reward", "tooluse_synth"]) == 0
    printed = capsys.readouterr().out
    assert "[grpo_cli] Seed 7 done." in printed
    assert "run_id        : run-7" in printed
    assert "sampler       : tinker://sampler" in printed
    assert "avg_first5    : 0.100" in printed
    assert "avg_last10    : 0.200" in printed
    assert "peak_reward   : 0.500" in printed
    assert "zero_loss     : 1/100" in printed
    assert "zero_reward   : 2/100" in printed
    assert "heldout_reward: 0.250" in printed
    assert "reward_trace  : [0.0, 0.5]" in printed
    tail = printed.split("[grpo_cli] Seed 8 done.", 1)[1]
    assert "run-8" in tail
    assert "heldout_reward" not in tail
    assert "reward_trace  : [0.2]" in tail


def test_redact_error_strips_secrets_and_token_shapes(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "plain-secret-value")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hub-secret-value")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret-value")
    message = grpo._redact_error(
        RuntimeError(
            "plain-secret-value hub-secret-value wandb-secret-value "
            "hf_Abcd1234efgh wandb_Abcd1234 sk-Abcd1234 "
            "ghp_Abcd1234 github_pat_Abcd1234 Bearer abc.def-XYZ "
            "still visible"
        )
    )
    assert "plain-secret-value" not in message
    assert "hub-secret-value" not in message
    assert "wandb-secret-value" not in message
    assert "hf_Abcd1234efgh" not in message
    assert "wandb_Abcd1234" not in message
    assert "sk-Abcd1234" not in message
    assert "ghp_Abcd1234" not in message
    assert "github_pat_Abcd1234" not in message
    assert "abc.def-XYZ" not in message
    assert "Bearer [redacted]" in message
    assert "still visible" in message
    assert message.count("[redacted]") >= 7


def test_redact_error_empty_message_uses_the_class_name():
    class Blank(Exception):
        def __str__(self) -> str:
            return ""

    assert grpo._redact_error(Blank()) == "Blank"


def test_help_overrides_returns_zero(monkeypatch):
    real = grpo_cli._parse_args

    def wrapped(argv=None):
        if argv == ["--help"]:
            return SimpleNamespace()
        return real(argv)

    monkeypatch.setattr(grpo_cli, "_parse_args", wrapped)
    assert grpo_cli.main(["--help-overrides"]) == 0


def test_module_guard_exits_with_main_status(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["grpo_cli"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("platform_tinker.tinkerrl.grpo_cli", run_name="__main__")
    assert raised.value.code == 1
