import json
from types import SimpleNamespace

import pytest
from platform_hybrid.skyrl.reward_contract import (
    gsm8k_reward,
    math_reward,
    tool_reward,
    refresh_sampler,
)


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        (r"\boxed{\frac{1}{3}}", r"\boxed{\frac{1}{2}}", 0),
        (r"\boxed{\frac{2}{4}}", r"\boxed{0.5}", 1),
        (r"\boxedX{42}", r"\boxed{42}", 0),
    ],
)
def test_math_exact(prediction, gold, expected):
    assert math_reward(prediction, {"answer": gold}) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is -42", 0),
        ("#### -42", 0),
        ("42", 1),
        ("#### 42", 1),
        ("42.0,000", 0),
        ("The answer is 42", 1),
    ],
)
def test_signed_final_answers(text, expected):
    assert gsm8k_reward(text, {"answer": "42"}) == expected


def test_tool_values_and_whole_json_are_required():
    gold = {"tool": "get_weather", "args": {"city": "Tokyo", "units": "metric"}}
    wrong = json.dumps({"tool": "get_weather", "arguments": {"city": "Paris", "units": "nonsense"}})
    assert tool_reward(wrong, gold) == 0
    valid = json.dumps({"tool": "get_weather", "arguments": gold["args"]})
    assert tool_reward(valid, gold) == 1
    assert tool_reward("prefix " + valid, gold) == 0
    assert tool_reward('{"tool":"bad","tool":"get_weather","arguments":{}}', gold) == 0


def test_sampler_tracks_every_update_independent_of_checkpoint_cadence():
    class Client:
        version = 0

        def save_weights_for_sampler(self, name):
            version = self.version
            return SimpleNamespace(result=lambda: SimpleNamespace(path=str(version)))

        def create_sampling_client(self, model_path):
            return int(model_path)

    client = Client()
    versions = []
    for completed in range(1, 12):
        client.version = completed
        sampler, _ = refresh_sampler(client, completed)
        versions.append(sampler)
    assert versions == list(range(1, 12))


def test_duplicate_yaml_cannot_erase_training_settings():
    from platform_hybrid.skyrl.reward_contract import load_unique_yaml

    with pytest.raises(ValueError, match="duplicate YAML"):
        load_unique_yaml("trainer:\n  policy: original\ntrainer:\n  logger: wandb\n")


@pytest.mark.parametrize("name", ["grpo_math.yaml", "grpo_gsm8k.yaml"])
def test_released_skyrl_config_retains_model_optimizer_and_logging(name):
    from pathlib import Path
    from platform_hybrid.skyrl.reward_contract import load_unique_yaml

    path = Path(__file__).resolve().parents[2] / "platform_hybrid/skyrl/configs" / name
    config = load_unique_yaml(path.read_text())
    assert config["trainer"]["policy"]["model"]["path"]
    assert config["trainer"]["policy"]["optimizer_config"]["lr"]
    assert config["trainer"]["algorithm"]["advantage_estimator"] == "grpo"
    assert config["trainer"]["logger"] == "wandb"


@pytest.mark.parametrize(
    ("arguments", "parameters"),
    [
        ({"x": 1}, {"x": True}),
        ({"x": True}, {"x": 1}),
        ({"x": 1}, {"x": "1"}),
        ({"x": 1}, {"x": 2}),
        ({"x": 1}, {"x": float("nan")}),
        ({"x": float("nan")}, {"x": float("nan")}),
        ({"x": [1]}, {"x": [True]}),
    ],
)
def test_tool_aliases_must_agree_under_exact_json_types(arguments, parameters):
    response = json.dumps(dict(tool="calc", arguments=arguments, parameters=parameters))
    assert tool_reward(response, {"tool": "calc", "args": arguments}) == 0


def test_identical_tool_aliases_remain_valid():
    args = {"x": [1, True, "1"], "y": {"a": 2}}
    response = json.dumps(dict(tool="calc", name="calc", arguments=args, parameters=args))
    assert tool_reward(response, {"tool": "calc", "args": args}) == 1
