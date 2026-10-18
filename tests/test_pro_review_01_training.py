"""Offline synthetic regressions for review 01. Never launches training or a service."""

import asyncio
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf import trainer
from platform_modal.openrlhf import objective_contract as objective
from platform_modal.openrlhf.reward import reward_func, smoke_test


def test_public_default_is_real_and_simulation_is_explicit(monkeypatch):
    real = Mock(return_value={"sentinel": "real"})
    monkeypatch.setattr(trainer, "run_openrlhf_training", real)
    config = OpenRLHFConfig()
    assert trainer.run(config) == {"sentinel": "real"}
    assert asyncio.run(trainer.OpenRLHFTrainer(config).run()) == {"sentinel": "real"}
    fixture = trainer.run(config, dry_run=True)
    assert "reward_trace" not in fixture
    assert fixture["data_kind"] == "synthetic_fixture"
    with pytest.raises(ValueError, match="completed empirical"):
        trainer.require_empirical_result(fixture)
    assert fixture == trainer.run(config, dry_run=True)
    assert (
        fixture["fixture_rewards"]
        != trainer.run(config.model_copy(update={"seed": 43}), dry_run=True)["fixture_rewards"]
    )


def test_missing_or_wrong_backend_never_falls_back(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "version", Mock(side_effect=importlib.metadata.PackageNotFoundError)
    )
    with pytest.raises(RuntimeError, match="Real execution"):
        trainer.run(OpenRLHFConfig())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.8.4")
    with pytest.raises(RuntimeError, match="Unsupported OpenRLHF"):
        trainer.validate_backend()


def command(config, tmp_path):
    return trainer.build_command(config, tmp_path, tmp_path / "data.jsonl", "/immutable/model")


@pytest.mark.parametrize(
    ("section", "field", "value", "flag"),
    [
        (None, "seed", 7, "--seed"),
        ("algorithm", "temperature", 0.7, "--temperature"),
        ("algorithm", "epsilon", 0.1, "--eps_clip"),
        ("algorithm", "sample_num", 32, "--n_samples_per_prompt"),
        ("data", "max_response_length", 2048, "--generate_max_len"),
        ("optimizer", "learning_rate", 0.0002, "--actor_learning_rate"),
        (None, "project_name", "test-project", "--wandb_project"),
    ],
)
def test_requested_settings_change_actual_command(section, field, value, flag, tmp_path):
    config = OpenRLHFConfig()
    setattr(getattr(config, section) if section else config, field, value)
    args = command(config, tmp_path)
    assert args[args.index(flag) + 1] == str(value)
    assert "--ref_pretrain" not in args
    assert Path(args[args.index("--remote_rm_url") + 1]).is_file()


def test_work_counters_and_budget_are_distinct(tmp_path):
    config = OpenRLHFConfig(train_batch_size=8, max_steps=240)
    config.algorithm.sample_num = 8
    budget = trainer.effective_config(config)["budget"]
    assert budget["rollout_rounds"] == 30
    assert budget["optimizer_updates"] == 240
    assert budget["prompt_presentations"] == 240
    args = command(config, tmp_path)
    assert args[args.index("--max_samples") + 1] == "240"
    config.max_steps = 248
    assert trainer.effective_config(config)["budget"]["rollout_rounds"] == 31
    config.max_steps = 30
    with pytest.raises(ValueError, match="divisible"):
        trainer.effective_config(config)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.algorithm, "algorithm", "dapo"),
        lambda c: setattr(c, "num_gpus", 2),
        lambda c: setattr(c, "max_steps", 0),
        lambda c: setattr(c.optimizer, "eps", 1e-4),
        lambda c: setattr(c.data, "val_data", ["unimplemented.json"]),
        lambda c: setattr(c, "eval_interval", 5),
    ],
)
def test_unsupported_settings_fail_before_launch(mutate):
    config = OpenRLHFConfig()
    mutate(config)
    with pytest.raises(ValueError, match="algorithm|GPUs|gpus|budget|Unsupported|Held-out"):
        trainer.effective_config(config)


def test_cli_preflight_uses_help_without_training_arguments(tmp_path, monkeypatch):
    args = command(OpenRLHFConfig(), tmp_path)
    probe = Mock(
        return_value=types.SimpleNamespace(stdout=" ".join(x for x in args if x.startswith("--")))
    )
    monkeypatch.setattr(trainer.subprocess, "run", probe)
    trainer.cli_preflight(args, {}, tmp_path)
    assert probe.call_args.args[0] == [sys.executable, "-m", "openrlhf.cli.train_ppo_ray", "--help"]
    probe.return_value.stdout = "--pretrain"
    with pytest.raises(RuntimeError, match="rejects flags"):
        trainer.cli_preflight(args, {}, tmp_path)


def mock_runtime(monkeypatch, *, code=0, observations=None):
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-secret-never-log")
    monkeypatch.setattr(trainer, "validate_backend", lambda: None)
    monkeypatch.setattr(trainer, "runtime_metadata", lambda: {"platform": "synthetic"})
    monkeypatch.setattr(
        trainer, "prepare_inputs", lambda c, d, b: (d / "prompts.jsonl", "/model", {})
    )
    monkeypatch.setattr(trainer, "cli_preflight", lambda *a: None)
    process = Mock(return_value=types.SimpleNamespace(returncode=code))
    monkeypatch.setattr(trainer.subprocess, "run", process)
    if observations is None:
        observations = [{"step": i, "reward": 0.5} for i in range(30)]
    reader = Mock(return_value=observations)
    monkeypatch.setattr(trainer, "read_rewards", reader)
    return process, reader


def test_exit_17_never_becomes_real_result(tmp_path, monkeypatch, capsys):
    process, reader = mock_runtime(monkeypatch, code=17)
    with pytest.raises(subprocess.CalledProcessError):
        trainer.run_openrlhf_training(OpenRLHFConfig(), str(tmp_path))
    reader.assert_not_called()
    assert not list(tmp_path.rglob("result.json"))
    manifest = json.loads(next(tmp_path.rglob("attempt.json")).read_text())
    assert manifest["state"] == "failed"
    assert manifest["last10_avg"] is None
    assert process.call_args.kwargs["check"] is True
    assert process.call_args.kwargs["timeout"] == 7200
    assert "synthetic-secret-never-log" not in capsys.readouterr().out
    assert "synthetic-secret-never-log" not in json.dumps(manifest)


def test_attempt_ids_isolate_stale_results(tmp_path, monkeypatch):
    mock_runtime(monkeypatch)
    first = trainer.run_openrlhf_training(OpenRLHFConfig(), str(tmp_path))
    second = trainer.run_openrlhf_training(OpenRLHFConfig(), str(tmp_path))
    assert first["attempt_id"] != second["attempt_id"]
    assert len(list(tmp_path.rglob("result.json"))) == 2
    assert first["observed_counters"]["optimizer_updates"] is None


def test_missing_history_is_incomplete_never_zero(tmp_path, monkeypatch):
    _, reader = mock_runtime(monkeypatch)
    reader.side_effect = RuntimeError("incomplete")
    with pytest.raises(RuntimeError, match="incomplete"):
        trainer.run_openrlhf_training(OpenRLHFConfig(), str(tmp_path))
    manifest = json.loads(next(tmp_path.rglob("attempt.json")).read_text())
    assert manifest["state"] == "incomplete"
    assert manifest["reward_trace"] is None
    assert not list(tmp_path.rglob("result.json"))


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"_step": 0, "train/reward": None}],
        [{"_step": 0, "train/reward": float("nan")}],
        [{"_step": 0, "train/reward": 0.5}] * 2,
        [{"_step": 2, "train/reward": 0.5}],
    ],
)
def test_bad_reward_histories_are_rejected(rows):
    with pytest.raises(RuntimeError):
        trainer.validate_history(rows, 1)


def test_history_reads_exact_id_and_upstream_key(monkeypatch):
    child = types.SimpleNamespace(
        id="unique",
        state="finished",
        scan_history=Mock(return_value=[{"_step": 1, "train/reward": 0.75}]),
    )
    api = types.SimpleNamespace(default_entity="owner", run=Mock(return_value=child))
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: api))
    assert trainer.read_rewards(OpenRLHFConfig(project_name="project"), "unique", 1) == [
        {"step": 1, "reward": 0.75}
    ]
    api.run.assert_called_once_with("owner/project/unique")
    assert child.scan_history.call_args.kwargs["keys"] == ["_step", "train/reward"]


def test_native_prompt_template_and_missing_label(tmp_path):
    tokenizer = types.SimpleNamespace(apply_chat_template=Mock(return_value="native-prompt"))
    path = tmp_path / "data.jsonl"
    trainer._prepare_gsm8k_500_jsonl(
        str(path),
        tokenizer=tokenizer,
        revision="commit",
        rows=[{"question": "1+1?", "answer": "reason #### 2"}],
    )
    data = json.loads(path.read_text())
    assert data["prompt"] == "native-prompt"
    assert data["label"] == "2"
    assert data["example_id"] == "openai/gsm8k@commit:train:0"
    assert tokenizer.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    with pytest.raises(ValueError, match="ground truth"):
        trainer._prepare_gsm8k_500_jsonl(
            str(path),
            tokenizer=tokenizer,
            revision="commit",
            rows=[{"question": "?", "answer": "missing"}],
        )


def test_actual_callback_correct_wrong_malformed_truncated_multibox():
    smoke_test()
    assert reward_func([r"prompt \boxed{2}"], ["prompt "], ["2"]).tolist() == [1]
    with pytest.raises(ValueError, match="lengths differ"):
        reward_func([], ["p"], [])


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_zero_contrast_has_zero_task_gradient(reward):
    advantages = objective.group_advantages(torch.full((2, 2), reward))
    log_probs = torch.zeros((4, 3), requires_grad=True)
    loss = objective.policy_loss(
        log_probs, torch.zeros_like(log_probs), advantages.flatten(), torch.ones_like(log_probs)
    )
    loss.backward()
    assert torch.isfinite(advantages).all()
    assert torch.equal(log_probs.grad, torch.zeros_like(log_probs))


def test_sample_sd_and_masked_length_weighting():
    advantages = objective.group_advantages(torch.tensor([[1.0, 0.0]]))
    assert advantages.tolist()[0] == pytest.approx([1 / np.sqrt(2), -1 / np.sqrt(2)])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    values = torch.tensor([[2.0, 999.0], [4.0, 4.0]])
    assert objective.sequence_mean(values, mask).item() == 3
    values[0, 1] = -999
    assert objective.sequence_mean(values, mask).item() == 3


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected"),
    [(1.5, 1.0, -1.2), (0.5, -1.0, 0.8), (0.5, 1.0, -0.5), (1.5, -1.0, 1.5)],
)
def test_clipped_surrogate_signs(ratio, advantage, expected):
    current = torch.tensor([[np.log(ratio)]])
    assert objective.policy_loss(
        current, torch.zeros_like(current), torch.tensor([advantage]), torch.ones_like(current)
    ).item() == pytest.approx(expected)


def test_kl_equal_reference_and_zero_contrast_nonzero_kl():
    log_probs = torch.zeros((2, 3), requires_grad=True)
    assert objective.kl_loss(log_probs, log_probs.detach(), torch.ones_like(log_probs)).item() == 0
    kl = objective.kl_loss(log_probs, torch.ones_like(log_probs), torch.ones_like(log_probs))
    assert kl.item() > 0
    kl.backward()
    assert torch.count_nonzero(log_probs.grad) == 6


def test_masked_extreme_padding_logits_do_not_change_loss_or_gradients():
    logits = torch.tensor([[0.0, 1e6]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0]])
    task_loss = objective.policy_loss(logits, torch.zeros_like(logits), torch.tensor([1.0]), mask)
    total = task_loss + objective.kl_loss(logits, torch.zeros_like(logits), mask)
    total.backward()
    assert total.item() == -1
    assert logits.grad[0, 1] == 0
    assert torch.isfinite(logits.grad).all()


def test_malformed_numeric_commas_are_not_accepted():
    assert reward_func([r"p\boxed{1,2}"], ["p"], ["12"]).tolist() == [0]
    assert reward_func([r"p\boxed{1,200}"], ["p"], ["1200"]).tolist() == [1]


def aggregate_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "platform_hybrid/experiments/results/aggregate_framework_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("review01_framework_aggregate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empirical_aggregator_never_launches_training_for_missing_result(tmp_path, monkeypatch):
    aggregate = aggregate_module()
    monkeypatch.setattr(aggregate, "OPENRLHF_RESULT", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="never launches training"):
        aggregate._load_openrlhf()


@pytest.mark.parametrize(
    ("kind", "state", "mode"),
    [
        ("synthetic_fixture", "completed", "dryrun"),
        ("empirical", "failed", "real"),
        ("empirical", "incomplete", "real"),
        (None, None, "real"),
    ],
)
def test_empirical_aggregator_rejects_fixture_failure_and_unverified_legacy(kind, state, mode):
    aggregate = aggregate_module()
    with pytest.raises(ValueError, match="completed empirical"):
        aggregate._validate_measured(
            {"data_kind": kind, "state": state, "mode": mode, "reward_trace": [0.5]}
        )


def test_framework_aggregate_preserves_actual_config_without_imputation(tmp_path, monkeypatch):
    aggregate = aggregate_module()
    source = tmp_path / "master.json"
    source.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "campaign_v2_w1_qwen3-8b-base",
                        "data_kind": "empirical",
                        "state": "completed",
                        "reward_trace": [0.2, 0.4],
                        "seed": 17,
                        "group_size": 32,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(aggregate, "MASTER", source)
    result = aggregate._load_tinker()
    assert result["seed"] == 17
    assert result["group_size"] == 32
    assert "model" not in result
    assert result["last10_avg"] == pytest.approx(0.3)


def test_saved_framework_summary_is_recomputed_from_observations(tmp_path):
    aggregate = aggregate_module()
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            {
                "data_kind": "empirical",
                "state": "completed",
                "mode": "real",
                "reward_trace": [0.2] * 10,
                "last10_avg": 0.99,
                "peak_reward": 0.99,
            }
        )
    )
    result = aggregate._load_saved(path, "OpenRLHF")
    assert result["last10_avg"] == pytest.approx(0.2)
    assert result["peak_reward"] == 0.2
    assert json.loads(path.read_text())["last10_avg"] == 0.99


def test_requested_wandb_entity_overrides_inherited_destination(tmp_path, monkeypatch):
    process, _ = mock_runtime(monkeypatch)
    monkeypatch.setenv("WANDB_ENTITY", "other-team")
    result = trainer.run_openrlhf_training(
        OpenRLHFConfig(wandb_entity="requested-team"), str(tmp_path)
    )
    assert process.call_args.kwargs["env"]["WANDB_ENTITY"] == "requested-team"
    assert result["wandb_entity"] == "requested-team"


def test_exact_run_lookup_honors_inherited_entity_when_not_configured(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "inherited-team")
    child = types.SimpleNamespace(
        id="unique",
        state="finished",
        scan_history=Mock(return_value=[{"_step": 1, "train/reward": 0.75}]),
    )
    api = types.SimpleNamespace(default_entity="other-owner", run=Mock(return_value=child))
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: api))
    trainer.read_rewards(OpenRLHFConfig(project_name="project"), "unique", 1)
    api.run.assert_called_once_with("inherited-team/project/unique")
