"""Fresh starts must not overwrite a different experiment's colliding path."""

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


def test_sanitized_name_collision_fails_before_tracking(tmp_path, monkeypatch):
    original = _config(tmp_path, name="experiment/a", resume=False)
    replacement = _config(tmp_path, name="experiment a", resume=False)
    path = grpo._checkpoint_path(original, original.seed)
    assert path == grpo._checkpoint_path(replacement, replacement.seed)
    grpo._write_checkpoint(
        path, {"config": grpo._config_fingerprint(original, original.seed), "step": 1}
    )
    before = path.read_bytes()
    with _runtime(monkeypatch), patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(ValueError, match="checkpoint identity"):
            grpo._run_one_seed(
                replacement,
                grpo.InMemoryDataset(train=[_example()]),
                grpo.ExactMathReward(),
                _Tokenizer(),
                logger=lambda _: None,
            )
    tracking.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "stored",
    [None, {}, {"name": "fresh"}, {"name": "fresh", "seed": True}, {"name": "fresh", "seed": 43}],
)
def test_fresh_restart_rejects_unproven_existing_identity(tmp_path, stored):
    config = _config(tmp_path, name="fresh", resume=False)
    path = grpo._checkpoint_path(config, config.seed)
    path.write_text(json.dumps({"config": stored}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="checkpoint identity"):
        grpo._load_checkpoint(config, config.seed)
    assert path.read_bytes() == before


def test_explicit_same_identity_restart_can_change_configuration(tmp_path):
    config = _config(tmp_path, name="fresh", resume=False)
    previous = _config(tmp_path, name="fresh", steps=5)
    grpo._write_checkpoint(
        grpo._checkpoint_path(config, config.seed),
        {"config": grpo._config_fingerprint(previous, previous.seed)},
    )
    assert grpo._load_checkpoint(config, config.seed) is None
