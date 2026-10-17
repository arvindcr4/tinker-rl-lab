"""Offline synthetic regressions for supplied Pro review slice 03.

No generated Python is executed and no model, dataset, or service is fetched.
"""

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ATROPOS = ROOT / "platform_tinker/atropos"
sys.path.insert(0, str(ATROPOS))
from tinker_atropos.config import TinkerAtroposConfig, apply_overrides
from tinker_atropos.safe_execution import (
    arithmetic_value,
    evaluate_python,
    IsolatedEvaluatorUnavailable,
)
from tinker_atropos.grading import tool_correctness, extract_json_object
from tinker_atropos.dataset import DatasetPreprocessor
from platform_tinker.tinkerrl import grpo


def load_functions(relative, names, namespace=None):
    """Load repository helper definitions without optional environment imports."""
    tree = ast.parse((ATROPOS / relative).read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    ns = {"re": re, "Optional": object, "Dict": dict, "List": list, "Tuple": tuple}
    ns.update(namespace or {})
    # Postponed annotations avoid requiring the optional environment package.
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(tree),
            "<repository helpers>",
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize(
    "expr,expected", [("sqrt(9)", 3.0), ("2*(3+4)", 14), ("round(1.25,1)", 1.2), ("-2**3", -8)]
)
def test_numeric_ast(expr, expected):
    assert arithmetic_value(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "abs.__self__.__import__('math').sqrt(9)",
        "().__class__",
        "1e309",
        "9**99999",
        "[1]*1000000",
        "sum(x for x in [1])",
    ],
)
def test_arithmetic_rejects_traversal_and_resources(expr):
    # Parsing a string is safe; it is never passed to Python eval/exec.
    assert arithmetic_value(expr) is None


@pytest.mark.parametrize("code", ["import os\nos._exit(0)", 'print("PASS")', "def f(): return 42"])
def test_generated_execution_fails_closed_before_code_is_run(code):
    with pytest.raises(IsolatedEvaluatorUnavailable):
        evaluate_python(code=code, tests="def check(f): assert f() == 42", entry_point="f")


@pytest.mark.parametrize(
    "key,value,path",
    [
        ("num_steps", 1, ("env", "total_steps")),
        ("batch_size", 8, ("env", "batch_size")),
        ("group_size", 2, ("env", "group_size")),
        ("learning_rate", 0.1, ("tinker", "learning_rate")),
        ("lora_rank", 4, ("tinker", "lora_rank")),
        ("use_lora", False, ("tinker", "use_lora")),
        ("use_wandb", False, ("env", "use_wandb")),
        ("wandb_project", "test", ("tinker", "wandb_project")),
        ("wandb_group", "grp", ("tinker", "wandb_group")),
    ],
)
def test_every_cli_override(key, value, path):
    resolved = apply_overrides(TinkerAtroposConfig(), {key: value}).model_dump()
    assert resolved[path[0]][path[1]] == value


def test_model_identity_and_unsupported_config():
    cfg = apply_overrides(TinkerAtroposConfig(), {"base_model": "OVERRIDE/MODEL"})
    assert cfg.base_model == cfg.tinker_model == cfg.openai[0].model_name == "OVERRIDE/MODEL"
    with pytest.raises(ValueError):
        TinkerAtroposConfig(num_steps=1)
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"unknown": 1})
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"num_steps": 0})


def test_legacy_help_uses_process_arguments(monkeypatch):
    from platform_tinker.tinkerrl import grpo_cli

    seen = []
    monkeypatch.setattr(sys, "argv", ["wrapper.py", "--help"])
    monkeypatch.setattr(grpo_cli, "main", lambda argv: seen.extend(argv) or 0)
    assert grpo_cli.legacy_main("xlam") == 0
    assert seen == ["--preset", "xlam", "--help"]


@pytest.mark.parametrize(
    "response,target,expected",
    [
        (r"Intermediate \boxed{42}. Final answer: \boxed{7}.", "42", 0),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", 1),
        (r"\boxed{42}. Verified in 7 steps", "42", 1),
        ("0", 0, 1),
        ("There are 42 examples and 7 trials", "7", 0),
        (r"\boxed{42.001}", "42", 0),
    ],
)
def test_authoritative_math(response, target, expected):
    assert grpo.ExactMathReward().score(response, grpo.TrainingExample("q", target)) == expected


@pytest.mark.parametrize(
    "args",
    [{"path": "/TMP/A"}, {"path": False}, {"path": {"x": False}}, {"path": "/tmp/a", "extra": 1}],
)
def test_tool_full_credit_requires_values_types_and_exact_keys(args):
    target = {"tool": "read", "arguments": {"path": "/tmp/a"}}
    import json

    assert (
        grpo.ToolCallReward().score(
            json.dumps({"tool": "read", "arguments": args}), grpo.TrainingExample("q", target)
        )
        < 1
    )
    assert not tool_correctness(
        {"tool": "read", "arguments": args}, "read", target["arguments"]
    ).correct


def test_nested_bool_zero_and_null_tool():
    target = {"tool": "x", "arguments": {"a": {"b": 0}}}
    assert (
        grpo.StrictToolCallReward().score(
            '{"tool":"x","arguments":{"a":{"b":false}}}', grpo.TrainingExample("q", target)
        )
        < 1
    )
    assert not tool_correctness({"tool": None, "args": {}}, "x", {}).correct
    assert extract_json_object('Call {"tool":"x","args":{"a":{"b":2}}} done')["args"]["a"]["b"] == 2
    assert extract_json_object('{"tool":"x","tool":"y"}') is None


def distillation_batch():
    return {
        "batch": [
            {
                "tokens": [[1, 2, 3], [1, 2, 4]],
                "inference_logprobs": [[1, 1, -2], [1, 1, -3]],
                "scores": [1, 1],
                "distill_token_ids": [[[1], [2], [3]], [[1], [2], [4]]],
                "distill_logprobs": [[[None], [None], [-1]], [[None], [None], [-1]]],
                "masks": [[-100, -100, 3], [-100, -100, 4]],
            }
        ]
    }


def test_tied_distillation_and_override():
    batch = distillation_batch()
    # Unrequested prompt values are masked, never teacher evidence.
    for row in batch["batch"][0]["distill_logprobs"]:
        row[0][0] = row[1][0] = 0.0
    pre = DatasetPreprocessor("unused")
    datums, _, distil, zvf = pre.pad_data_to_good_offset(batch)
    assert distil and zvf == 1 and len(datums) == 2
    assert datums[0].loss_fn_inputs["advantages"].to_torch().tolist() == [0, 1]
    batch["batch"][0]["overrides"] = [{"set_advantage_to_zero": True}, {}]
    datums, *_ = pre.pad_data_to_good_offset(batch)
    assert datums[0].loss_fn_inputs["advantages"].to_torch().tolist() == [0, 0]
    batch["batch"][0]["distill_token_ids"][0][-1] = [999]
    with pytest.raises(ValueError, match="Teacher token IDs"):
        pre.pad_data_to_good_offset(batch)


def test_missing_teacher_is_not_positive_reinforcement():
    batch = distillation_batch()
    batch["batch"][0]["distill_logprobs"][0][-1] = [None]
    with pytest.raises((ValueError, TypeError)):
        DatasetPreprocessor("unused").pad_data_to_good_offset(batch)


def fake_trainer():
    tok = NS(
        encode=lambda text, **kw: [1, 2],
        decode=lambda ids, **kw: "answer",
        apply_chat_template=lambda *a, **kw: "prompt",
    )
    seq = NS(tokens=[3, 4], logprobs=[-1.0, -2.0], stop_reason="length")
    sampler = NS(
        sample_async=AsyncMock(return_value=NS(sequences=[seq])),
        compute_logprobs_async=AsyncMock(return_value=[None, -1.0]),
    )
    return NS(
        tokenizer=tok,
        current_sampling_client=sampler,
        base_model="mock",
        policy_version="snapshot-2",
    )


def test_resolved_config_readiness_generate_datum_integration():
    from fastapi.testclient import TestClient
    from tinker_atropos.api import create_app

    cfg = apply_overrides(TinkerAtroposConfig(), {"num_steps": 1, "group_size": 2})
    trainer = fake_trainer()
    client = TestClient(create_app(trainer))
    assert cfg.num_steps == 1 and client.get("/health").json()["ready"]
    response = client.post(
        "/generate", json={"input_ids": [1, 2], "sampling_params": {"seed": 7, "top_p": 0.8}}
    )
    assert response.status_code == 200
    meta = response.json()["meta_info"]
    assert meta["finish_reason"] == "length" and meta["policy_version"] == "snapshot-2"
    params = trainer.current_sampling_client.sample_async.call_args.kwargs["sampling_params"]
    assert params.seed == 7 and params.top_p == 0.8 and params.stop is None
    batch = {
        "batch": [
            {
                "tokens": [[1, 2, 3, 4], [1, 2, 5, 6]],
                "inference_logprobs": [[1, 1, -1, -2], [1, 1, -2, -1]],
                "scores": [1, 0],
            }
        ]
    }
    datums, *_ = DatasetPreprocessor("unused").pad_data_to_good_offset(batch)
    assert len(datums) == 2 and datums[0].loss_fn_inputs["advantages"].to_torch().tolist() == [
        0,
        0.5,
        0.5,
    ]
    trainer.current_sampling_client.sample_async.return_value.sequences[0].logprobs = [-1.0]
    assert client.post("/generate", json={"input_ids": [1, 2]}).status_code == 502
    assert (
        client.post(
            "/generate", json={"input_ids": [1, 2], "sampling_params": {"unsupported": 1}}
        ).status_code
        == 400
    )
    assert client.post("/generate", json={}).status_code == 400
    assert (
        client.post("/logprobs", json={"input_ids": [1, 2]}).json()["logprobs"][0]["logprob"]
        is None
    )


def test_inactive_batch_does_not_update_optimizer():
    from tinker_atropos.trainer import TinkerAtroposTrainer

    trainer = TinkerAtroposTrainer(TinkerAtroposConfig())
    trainer.dataset.get_data = Mock(return_value=([], False))
    trainer.training_client = NS(optim_step_async=AsyncMock())
    result = asyncio.run(trainer.train_step(0))
    assert result["no_update"]
    trainer.training_client.optim_step_async.assert_not_called()


def test_training_failure_propagates():
    from tinker_atropos.trainer import TinkerAtroposTrainer

    trainer = TinkerAtroposTrainer(apply_overrides(TinkerAtroposConfig(), {"num_steps": 1}))
    trainer.setup = AsyncMock()
    trainer.train_step = AsyncMock(side_effect=RuntimeError("failed step"))
    with pytest.raises(RuntimeError, match="failed step"):
        asyncio.run(trainer.run())
    assert trainer.run_status == "failed"


def test_critic_files_are_immutable_and_step_identified(tmp_path):
    config = grpo.GRPOConfig(name="test", checkpoint_dir=str(tmp_path))
    critic = torch.nn.Linear(2, 1)
    first = grpo._save_critic(config, 42, critic, step=1)
    original = Path(first["path"]).read_bytes()
    second = grpo._save_critic(config, 42, critic, step=2)
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == original
    assert torch.load(second["path"], weights_only=True)["step"] == 2


def test_manifest_changes_with_actual_examples_and_grader():
    tok = fake_trainer().tokenizer
    train = [grpo.TrainingExample("q", 1)]
    a = grpo._experiment_manifest(grpo.InMemoryDataset(train), grpo.MathReward(), train, [], tok)
    b = grpo._experiment_manifest(
        grpo.InMemoryDataset(train), grpo.ExactMathReward(), train, [], tok
    )
    assert a != b
    train2 = [grpo.TrainingExample("q", 2)]
    c = grpo._experiment_manifest(grpo.InMemoryDataset(train2), grpo.MathReward(), train2, [], tok)
    assert a != c


def test_duplicate_finish_cannot_count_as_retrieval():
    ns = load_functions(
        "tinker_atropos/environments/multihop_react_tinker.py",
        ["_execute_react_trace"],
        {"_bm25_search": lambda *a: "obs"},
    )
    f = ns["_execute_react_trace"]
    assert f('Action: finish("Paris")\nAction: finish("Paris")', {}) == ("Paris", 0, 1)
    assert f('Action: finish("Paris")\nAction: search("Paris")', {}) == ("Paris", 0, 1)


def test_multilabel_toxicity_is_not_inverted():
    labels = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}
    import math

    ns = load_functions(
        "eval_toxicchat.py", ["toxicity_score"], {"TOXIC_LABELS": labels, "math": math}
    )
    assert (
        ns["toxicity_score"](
            [{"label": k, "score": 0.99 if k == "insult" else 0.01} for k in labels]
        )
        == 0.99
    )
    with pytest.raises(ValueError):
        ns["toxicity_score"]([{"label": "insult", "score": 0.99}])


def test_arena_role_and_budget():
    ns = load_functions("eval_arenahard.py", ["arena_messages"])
    assert ns["arena_messages"]([{"content": "question"}]) == [
        {"role": "user", "content": "question"}
    ]
    from tinker_atropos.budgets import budget_steps

    assert budget_steps(128 * 2048 * 10, 2048, 16, 8) == 10
    with pytest.raises(ValueError):
        budget_steps(10, 2048, 16, 8)


def test_interaction_conditions_on_observation_and_masks_it():
    from tinker_atropos.interaction import interactive_rollout

    class Managed:
        index = 0

        async def chat_completion(self, **kwargs):
            self.index += 1
            if self.index == 1:
                self.node = NS(tokens=[1, 2], masked_tokens=[-100, 2], logprobs=[1, -0.5])
                text = 'Action: search("Paris")'
            else:
                assert kwargs["messages"][-1]["content"] == "Observation: authoritative fact"
                self.node = NS(
                    tokens=[1, 2, 3, 4],
                    masked_tokens=[-100, -100, -100, 4],
                    logprobs=[1, 1, 1, -0.2],
                )
                text = 'Action: finish("Paris")'
            return NS(choices=[NS(message=NS(content=text))])

        def get_state(self):
            return {"nodes": [self.node]}

    @asynccontextmanager
    async def managed_server(**kwargs):
        yield Managed()

    text, node = asyncio.run(
        interactive_rollout(
            NS(managed_server=managed_server),
            None,
            [],
            lambda trace, text: ("authoritative fact", "finish" in text),
            max_tokens=20,
        )
    )
    assert node.masked_tokens == [-100, 2, -100, 4]
    assert node.logprobs == [1, -0.5, 1, -0.2]
    assert "Observation: authoritative fact" in text


def test_math_verify_gold_first_and_last_complete_box():
    from tinker_atropos.grading import authoritative_math_answer, verify_answer

    calls = []

    def asymmetric(gold, prediction):
        calls.append((gold, prediction))
        return gold == "interval-reference" and prediction == "inequality-answer"

    assert verify_answer("interval-reference", "inequality-answer", asymmetric)
    assert calls == [("interval-reference", "inequality-answer")]
    assert (
        authoritative_math_answer(r"\boxed{42} final \boxed{\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"
    )


def test_browsergym_raw_interactive_ids_survive_url_and_screenshot(monkeypatch):
    import types

    helper = types.ModuleType("browsergym.utils.obs")
    helper.flatten_axtree_to_str = lambda tree: "[42] button Submit"
    helper.flatten_dom_to_str = lambda tree: '<button bid="42">Submit</button>'
    monkeypatch.setitem(sys.modules, "browsergym.utils.obs", helper)
    ns = load_functions(
        "tinker_atropos/environments/browsergym_tinker.py", ["_stringify_observation", "_shorten"]
    )
    _, text = ns["_stringify_observation"](
        {"url": "http://local", "screenshot": "pixels", "axtree_object": {"nodes": []}}, 1000
    )
    assert "[42] button Submit" in text


def test_clarification_dependent_glaive_record_is_rejected():
    ns = load_functions(
        "tinker_atropos/environments/tool_use_tinker.py",
        ["_build_glaive_examples"],
        {"_extract_json_objects": lambda text: [{"name": "book"}], "SYSTEM_PROMPT": "system"},
    )
    record = {
        "system": "tools",
        "chat": 'USER: Book a flight\nASSISTANT: Where?\nUSER: Tokyo\nASSISTANT: <functioncall> {"name":"book","arguments":\'{"destination":"Tokyo"}\'} <|endoftext|>',
    }
    assert ns["_build_glaive_examples"]([record]) == []


def test_nested_alternative_tool_call_and_numeric_boundaries():
    import json

    ns = load_functions("train_grpo_humaneval.py", ["_score_tool_call"], {"json": json})
    score = ns["_score_tool_call"]
    predicate = lambda args: args == {"expression": "17*23"}
    assert (
        score(
            '{"tool":"calculator","args":{"expression":"17*23"}} Answer: 391',
            "calculator",
            predicate,
            "391",
        )
        == 1
    )
    assert (
        score(
            '{"tool":"calculator","args":{"expression":"17*23"}} Answer: 3910',
            "calculator",
            predicate,
            "391",
        )
        == 0
    )


def test_batch_http_errors_and_bounded_empty_polling(monkeypatch):
    import requests

    pre = DatasetPreprocessor("http://unused.invalid")
    response = NS(
        raise_for_status=Mock(side_effect=requests.HTTPError("401")),
        json=Mock(return_value={"error": "unauthorized"}),
    )
    monkeypatch.setattr("tinker_atropos.dataset.requests.get", Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        pre.get_batch.__wrapped__(pre)
    response.json.assert_not_called()
    pre.get_batch = Mock(return_value={"batch": None})
    with pytest.raises(TimeoutError):
        pre.get_data(timeout=0)


def test_sampler_refresh_is_independent_of_publication(tmp_path):
    from tests.test_grpo_module import TestTrackingFailClosed, _temporary_modules
    from unittest.mock import patch

    helper = TestTrackingFailClosed()
    runtime = helper._fake_runtime()
    config = grpo.GRPOConfig(
        name="fresh-policy",
        steps=3,
        save_every=3,
        group_size=2,
        batch_size=1,
        checkpoint_dir=str(tmp_path),
    )
    with (
        _temporary_modules(helper._fake_modules(runtime)),
        patch.object(grpo, "_publish_checkpoint", side_effect=helper._fake_success_receipt),
    ):
        grpo._run_one_seed(
            config,
            helper._dataset(),
            grpo.ExactMathReward(),
            runtime["tokenizer"],
            logger=lambda _: None,
        )
    events = runtime["events"]
    behavior = [
        event
        for event in events
        if isinstance(event, tuple)
        and event[0] == "save_sampler"
        and event[1].startswith("behavior_")
    ]
    assert len(behavior) == 3
    names = [event[1] for event in behavior]
    assert names == ["behavior_seed42_0", "behavior_seed42_1", "behavior_seed42_2"]


def test_completed_resume_rejects_changed_task(tmp_path):
    from tests.test_grpo_module import TestTrackingFailClosed, _temporary_modules
    from unittest.mock import patch

    helper = TestTrackingFailClosed()
    runtime = helper._fake_runtime()
    config = grpo.GRPOConfig(
        name="manifest-task",
        steps=1,
        save_every=1,
        group_size=2,
        batch_size=1,
        checkpoint_dir=str(tmp_path),
    )
    with (
        _temporary_modules(helper._fake_modules(runtime)),
        patch.object(grpo, "_publish_checkpoint", side_effect=helper._fake_success_receipt),
    ):
        grpo._run_one_seed(
            config,
            helper._dataset(),
            grpo.ExactMathReward(),
            runtime["tokenizer"],
            logger=lambda _: None,
        )
        changed = grpo.InMemoryDataset(train=[grpo.TrainingExample("q", 99)])
        with pytest.raises(ValueError, match="experiment inputs"):
            grpo._run_one_seed(
                config, changed, grpo.ExactMathReward(), runtime["tokenizer"], logger=lambda _: None
            )


@pytest.mark.parametrize(
    "wrapper",
    ["grpo_100_math.py", "grpo_100_xlam.py", "grpo_exp_a_baseline.py", "grpo_tooluse_tinker.py"],
)
def test_compatibility_wrapper_help_exits_without_training(wrapper):
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "platform_tinker" / wrapper), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0 and "usage:" in result.stdout


@pytest.mark.parametrize(
    "path", sorted((ATROPOS / "configs").glob("*.yaml")), ids=lambda path: path.name
)
def test_supplied_configuration_schema_is_complete(path):
    cfg = TinkerAtroposConfig.from_yaml(path)
    if path.name.startswith("browsergym_"):
        assert cfg.browsergym_env_ids
        assert cfg.browsergym_max_browser_steps == (4 if "miniwob" in path.name else 8)


@pytest.fixture
def actual_tool_use_module(monkeypatch):
    """Import the actual environment with optional integrations stubbed out."""
    import importlib.util
    from types import ModuleType

    dataset_module = ModuleType("datasets")
    dataset_module.load_dataset = Mock(side_effect=AssertionError("no dataset access permitted"))
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)
    for name in ("atroposlib", "atroposlib.envs", "atroposlib.envs.base"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    base = sys.modules["atroposlib.envs.base"]
    base.BaseEnv = object
    base.BaseEnvConfig = base.APIServerConfig = NS
    base.ScoredDataGroup = dict
    spec = importlib.util.spec_from_file_location(
        "review03_actual_tool_use",
        ATROPOS / "tinker_atropos/environments/tool_use_tinker.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reward_version", ["v1", "v2"])
@pytest.mark.parametrize(
    "response,expected",
    [
        ('{"tool":"lookup","arguments":{"query":"France"}}', 1.0),
        (' \n {"tool":"lookup","arguments":{"query":"France"}}\t ', 1.0),
        (
            '{"tool":"lookup","arguments":{"query":"France"}}\n{"tool":"other","arguments":{}}',
            0.0,
        ),
        ('{"tool":"lookup","arguments":{"query":"France"}} trailing answer', 0.0),
        ('Call {"tool":"lookup","arguments":{"query":"France"}}', 0.0),
        ('```json\n{"tool":"lookup","arguments":{"query":"France"}}\n```', 0.0),
        ('[{"tool":"lookup","arguments":{"query":"France"}}]', 0.0),
        ('{"tool":"lookup","tool":"other","arguments":{"query":"France"}}', 0.0),
        ("null", 0.0),
        ('{"tool":"lookup","arguments":{"query":NaN}}', 0.0),
    ],
)
def test_actual_tool_environment_requires_single_whole_json_object(
    actual_tool_use_module, monkeypatch, reward_version, response, expected
):
    module = actual_tool_use_module
    monkeypatch.setattr(module, "_REWARD_VERSION", reward_version)
    env = object.__new__(module.ToolUseEnv)
    env.config = NS(group_size=1, max_token_length=128)
    env.percent_correct_buffer = []
    example = {
        "system": "Respond with exactly one JSON object.",
        "user": "Look up France",
        "expected_tool": "lookup",
        "expected_args": {"query": "France"},
    }
    # Exercise both real environment scoring paths, not a copied helper.
    env.server = NS(
        chat_completion=AsyncMock(return_value=NS(choices=[NS(message=NS(content=response))]))
    )
    assert asyncio.run(env.rollout_and_score_eval(example))["score"] == expected
    rollout = {
        **example,
        "messages": [{"role": "assistant", "content": response}],
        "tokens": [1, 2],
        "masked_tokens": [-100, 2],
        "logprobs": [-1.0, -2.0],
    }
    assert asyncio.run(env.score([rollout]))["scores"] == [expected]


@pytest.mark.parametrize("boxed", [False, True])
@pytest.mark.parametrize(
    "candidate,target,expected",
    [
        ("9007199254740993", "9007199254740992", 0.0),
        ("9007199254740992", "9007199254740993", 0.0),
        ("-9007199254740993", "-9007199254740992", 0.0),
        ("9007199254740993", "9007199254740993", 1.0),
        ("9007199254740993.00000000000000000001", "9007199254740993", 0.0),
        (
            "123456789012345678901234567890.12345678901234567891",
            "123456789012345678901234567890.12345678901234567890",
            0.0,
        ),
        (
            "123456789012345678901234567890.12345678901234567890",
            "123456789012345678901234567890.1234567890123456789",
            1.0,
        ),
        ("9.007199254740993e15", "9007199254740993", 1.0),
        ("1/3", "2/6", 1.0),
        ("1/3", "0.3333333333333333333333333333333333", 0.0),
        (r"\frac{1}{2}", "0.5", 1.0),
        ("1/0", "1/0", 0.0),
        ("1e10001", "1e10001", 0.0),
    ],
)
def test_exact_math_never_rounds_distinct_numbers(candidate, target, expected, boxed):
    response = rf"\boxed{{{candidate}}}" if boxed else candidate
    assert grpo.ExactMathReward().score(response, grpo.TrainingExample("q", target)) == expected


@pytest.mark.parametrize("boxed", [False, True])
@pytest.mark.parametrize(
    "value", ["NaN", "nan", "sNaN", "NaN123", "Inf", "-Inf", "+Infinity", "-Infinity"]
)
def test_exact_math_rejects_nonfinite_even_identical_targets(value, boxed):
    response = rf"\boxed{{{value}}}" if boxed else value
    assert grpo.ExactMathReward().score(response, grpo.TrainingExample("q", value)) == 0.0
    assert grpo.ExactMathReward().score(response, grpo.TrainingExample("q", "0")) == 0.0


@pytest.mark.parametrize(
    "candidate,target,tolerance,expected",
    [
        ("9007199254740993", "9007199254740992", 0.01, 0.3),
        ("9007199254740993.01", "9007199254740993", 0.01, 1.0),
        ("9007199254740993.01000000000000001", "9007199254740993", 0.01, 0.3),
        ("42.005", "42", 0.01, 1.0),
        ("42.02", "42", 0.01, 0.3),
    ],
)
def test_shaped_math_absolute_tolerance_uses_exact_decimal_difference(
    candidate, target, tolerance, expected
):
    reward = grpo.MathReward(numeric_tolerance=tolerance)
    assert reward.score(rf"\boxed{{{candidate}}}", grpo.TrainingExample("q", target)) == expected
