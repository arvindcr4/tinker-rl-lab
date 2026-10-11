"""Budget and arithmetic configuration fail before external work starts."""

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from platform_tinker.tinkerrl import grpo


def test_integer_budgets_reject_coercion_and_invalid_bounds():
    minima = {
        "steps": 0,
        "lora_rank": 0,
        "group_size": 1,
        "batch_size": 1,
        "max_prompt_tokens": 1,
        "max_response_tokens": 1,
        "num_seeds": 1,
        "dynamic_sampling_max_resamples": 0,
        "gspo_update_epochs": 1,
        "critic_hidden_dim": 1,
        "critic_updates_per_step": 0,
        "critic_pretrain_batches": 0,
        "save_every": 1,
    }
    for field, minimum in minima.items():
        for value in (True, False, 1.5, "2", None, minimum - 1):
            if field == "save_every" and value is None:
                continue
            with pytest.raises(ValueError, match=field):
                grpo.GRPOConfig(name="invalid", **{field: value})
    for value in (True, 1.5, "42", None):
        with pytest.raises(ValueError, match="seed"):
            grpo.GRPOConfig(name="invalid", seed=value)


def test_numeric_arithmetic_rejects_nonfinite_and_non_numeric_values():
    fields = (
        "lr",
        "temperature",
        "top_p",
        "beta1",
        "beta2",
        "eps",
        "nll_aux_coef",
        "nll_aux_min_reward",
        "critic_lr",
        "gspo_epsilon_low",
        "gspo_epsilon_high",
    )
    for field in fields:
        for value in (float("nan"), float("inf"), -float("inf"), True, "0.5", None, 10**400):
            with pytest.raises(ValueError, match=field):
                grpo.GRPOConfig(name="invalid", **{field: value})


def test_existing_zero_modes_and_default_save_cadence_are_preserved():
    config = grpo.GRPOConfig(
        name="valid",
        steps=0,
        lora_rank=0,
        group_size=1,
        temperature=0,
        dynamic_sampling_max_resamples=0,
        critic_updates_per_step=0,
        critic_pretrain_batches=0,
        nll_aux_coef=0,
        gspo_epsilon_low=0,
        gspo_epsilon_high=0,
    )
    assert config.effective_save_every() == 10
    assert replace(config, steps=200).effective_save_every() == 50
    assert replace(config, save_every=1).effective_save_every() == 1


def test_mutated_config_fails_before_dataset_tracking_or_seed_run():
    config = grpo.GRPOConfig(name="mutated")
    dataset = Mock()
    reward = Mock()
    config.num_seeds = 0
    with patch.object(grpo, "_run_one_seed") as run_seed:
        with pytest.raises(ValueError, match="num_seeds"):
            grpo.run_grpo(config, dataset, reward)
        run_seed.assert_not_called()
    config.num_seeds = 1
    config.critic_lr = float("nan")
    with patch.object(grpo.GRPOConfig, "validate_tracking") as tracking:
        with pytest.raises(ValueError, match="critic_lr"):
            grpo._run_one_seed(config, dataset, reward, tokenizer=None)
        tracking.assert_not_called()
    assert dataset.mock_calls == []
    assert reward.mock_calls == []
