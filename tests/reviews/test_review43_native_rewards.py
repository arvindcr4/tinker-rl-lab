"""Offline native-reward integrity tests; never contact a service."""

import json
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
from platform_hybrid.experiments.webarena import aggregate, react_eval as re


@pytest.mark.parametrize(
    "bad", [float("inf"), float("-inf"), float("nan"), 2, -0.5, True, "1", None]
)
def test_invalid_native_reward_stays_scoreless_after_prior_success(tmp_path, monkeypatch, bad):
    rewards = iter([1, bad])
    env = NS(
        reset=lambda **k: ({"goal": "x"}, {}),
        close=lambda: None,
        step=lambda action: ({}, next(rewards), False, False, {}),
    )
    monkeypatch.setattr(re, "_import_benchmark", lambda _: None)
    monkeypatch.setattr(re, "_make_env", lambda _: env)
    monkeypatch.setattr(re, "_axtree_to_str", lambda *a: "tree")
    result = re.run_episode(
        NS(sample=lambda _: "Action: click('1')"), "task", 42, 3, 100, "webarena"
    )
    assert result.score is None
    assert result.total_reward is None
    assert "invalid native reward" in result.error
    result.evaluation_id = "test"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(asdict(result), allow_nan=False))
    with pytest.raises(ValueError, match="invalid score"):
        aggregate.parse_results([path])


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -1, True, 4])
def test_aggregate_rejects_corrupt_native_total(tmp_path, total):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(
            dict(
                env_id="task",
                seed=42,
                evaluation_id="test",
                score=1,
                num_steps=3,
                valid_action_count=3,
                total_reward=total,
            )
        )
    )
    with pytest.raises(ValueError, match="invalid total reward"):
        aggregate.parse_results([path])


def test_driver_persists_and_resumes_scoreless_native_error(tmp_path, monkeypatch):
    monkeypatch.setattr(re, "_import_benchmark", lambda _: None)
    monkeypatch.setattr(re, "_parse_tasks", lambda *a: ["task"])
    monkeypatch.setattr(re, "TinkerChatSampler", lambda **k: None)
    monkeypatch.setattr(
        re,
        "run_episode",
        lambda *a, **k: re.EpisodeResult(
            "task", a[2], None, None, False, False, 0, 0, [], 0, "invalid native reward"
        ),
    )
    args = NS(
        benchmark="webarena",
        tasks="x",
        model="model",
        seed=42,
        max_steps=3,
        max_axtree_chars=100,
        action_tokens=12,
        temperature=0.0,
        concurrency=1,
        shard="0/1",
        limit=0,
        wandb_project=None,
        out=str(tmp_path / "rows"),
        resume=False,
        hf_repo=None,
    )
    assert re._main_sync(args) == 0
    args.resume = True
    assert re._main_sync(args) == 0
    rows = list(map(json.loads, (tmp_path / "rows").read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["score"] is None
    assert rows[0]["error"]
