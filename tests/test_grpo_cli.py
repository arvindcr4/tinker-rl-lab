from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from platform_tinker.tinkerrl.grpo_cli import (
    _build_dataset,
    _parse_args,
    build_config,
    legacy_main,
)


def test_baseline_preset_preserves_legacy_experiment_configuration():
    config = build_config(_parse_args(["--preset", "tooluse_baseline"]))

    assert config.name == "A_baseline"
    assert config.steps == 200
    assert config.group_size == 8
    assert config.batch_size == 4
    assert config.num_seeds == 5
    assert config.seed == 0
    assert config.resume is True


def test_legacy_gsm8k_flags_translate_to_canonical_interface():
    with patch("platform_tinker.tinkerrl.grpo_cli.main", return_value=0) as canonical_main:
        assert (
            legacy_main(
                "gsm8k",
                ["--rank", "16", "--group", "4", "--batch", "2", "--tag", "run"],
            )
            == 0
        )

    canonical_main.assert_called_once_with(
        [
            "--preset",
            "gsm8k",
            "--lora-rank",
            "16",
            "--group-size",
            "4",
            "--batch-size",
            "2",
            "--name",
            "run",
        ]
    )


def test_checkpoint_manifest_rejects_config_drift(tmp_path):
    config = grpo.GRPOConfig(name="resume", checkpoint_dir=str(tmp_path))
    path = grpo._checkpoint_path(config, config.seed)
    path.write_text(
        json.dumps(
            {
                "status": "started",
                "config": {**grpo._config_fingerprint(config, config.seed), "steps": 999},
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible checkpoint"):
        grpo._load_checkpoint(config, config.seed)


def test_pavlov_preset_carries_complete_campaign_metadata():
    config = build_config(_parse_args(["--preset", "pavlov_xlam"]))

    assert len(config.training_suite_ids) == 12
    assert len(config.primary_evaluation_suite_ids) == 14
    assert len(config.declared_domains) == 16
    assert len(config.training_domain_union) == 16
    assert len(config.primary_evaluation_domain_union) == 16
    assert len(config.domain_tags) == 14
    assert len(config.heldout_suite_ids) == 6
    assert set(config.heldout_suite_ids).issubset(config.primary_evaluation_suite_ids)
    assert "openreward_train" in config.training_suite_ids
    assert "frontiermath_eval" in config.primary_evaluation_suite_ids
    assert "computer_use" in config.domain_tags
    assert config.wandb_project == "tinker-rl-lab-pavlov"
    assert config.wandb_group == "pavlov-tinker-18usd-20260809"
    assert config.wandb_entity == "arvindcr4-pes-university"
    assert config.hf_public is True
    assert config.campaign_status == "authorized"
    assert config.dataset_revision == "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
    assert config.model_revision == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert config.authorized_budget_usd == 18.0
    assert config.maximum_usd == 18.0
    config.validate_campaign_gate()
    immutable = grpo._immutable_config(config, config.seed)
    assert immutable["dataset_revision"] == config.dataset_revision
    assert immutable["model_revision"] == config.model_revision


def test_pavlov_xlam_dataset_factory_receives_pinned_revision():
    config = build_config(_parse_args(["--preset", "pavlov_xlam"]))
    factory = patch("platform_tinker.tinkerrl.grpo_cli.DATASET_FACTORIES", {
        "pavlov_xlam": lambda **kwargs: kwargs,
    })
    with factory:
        dataset = _build_dataset(_parse_args(["--preset", "pavlov_xlam"]), config)

    assert dataset == {
        "seed": config.seed,
        "revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    }


def test_pavlov_portfolio_preset_is_exact_and_excludes_xlam():
    config = build_config(_parse_args(["--preset", "pavlov_portfolio"]))

    assert config.training_suite_ids == ("api_bank_rlvr_train", "swe_gym_train")
    assert len(config.primary_evaluation_suite_ids) == 14
    assert config.evaluate_heldout is False
    assert config.dataset_revision == grpo.PAVLOV_NON_XLAM_DATASET_REVISION
    assert config.authorized_budget_usd == 16.5
    assert "xlam" not in json.dumps(grpo._immutable_config(config, config.seed)).lower()


def test_pavlov_portfolio_dataset_factory_receives_seed_only():
    config = build_config(_parse_args(["--preset", "pavlov_portfolio"]))
    factories = {"pavlov_portfolio": lambda **kwargs: kwargs}
    with patch("platform_tinker.tinkerrl.grpo_cli.DATASET_FACTORIES", factories):
        dataset = _build_dataset(_parse_args(["--preset", "pavlov_portfolio"]), config)

    assert dataset == {"seed": 809}


def test_dataset_revision_cli_override_is_carried_into_config():
    config = build_config(
        _parse_args(
            [
                "--preset",
                "tooluse_xlam",
                "--dataset-revision",
                "rev-123",
                "--model-revision",
                "model-rev-123",
            ]
        )
    )
    assert config.dataset_revision == "rev-123"
    assert config.model_revision == "model-rev-123"


def test_no_wandb_escape_hatch_is_removed():
    with pytest.raises(SystemExit):
        _parse_args(["--preset", "tooluse_synth", "--no-" + "wandb"])


def test_json_config_rejects_disabled_tracking_before_run(tmp_path):
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps({"name": "disabled", "wandb_project": None}))

    with pytest.raises(ValueError, match="W&B tracking is mandatory"):
        build_config(_parse_args(["--json-config", str(path)]))


def test_json_config_rejects_contradictory_campaign_gate(tmp_path):
    path = tmp_path / "blocked-campaign.json"
    path.write_text(
        json.dumps(
            {
                "name": "blocked",
                "campaign_status": "draft-awaiting-budget-cap",
                "budget_status": "AUTHORIZED_TINKER_ONLY",
                "paid_jobs_may_launch": True,
                "authorized_budget_usd": 18.0,
                "maximum_usd": 18.0,
            }
        )
    )

    with pytest.raises(ValueError, match="not launchable"):
        build_config(_parse_args(["--json-config", str(path)]))
