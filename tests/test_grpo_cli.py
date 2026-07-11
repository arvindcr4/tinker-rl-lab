from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from platform_tinker.tinkerrl.grpo_cli import (
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
