"""CORE native fixtures only: no real capsule/model/container execution or scores."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import public_core_bench_native as runner


@pytest.fixture(scope="module")
def prepared():
    path = runner.DEFAULT_SETUP / "prepared"
    return path, runner.prepared_manifest(runner.DEFAULT_SETUP, path)


@pytest.fixture
def model():
    return {"base_model": copy.deepcopy(runner.ACTOR_BASE), "adapter": copy.deepcopy(runner.ACTOR_ADAPTER),
            "adapter_loaded": True, "vision": True, "served_model": "fixture-only-actor", "deployment_receipt_sha256": "a" * 64}


@pytest.fixture
def wandb():
    return {"mode": "online", "run_id": "fixture", "url": "https://wandb.ai/fixture/p/runs/fixture", "initialized_before_model_work": True}


@pytest.fixture
def agent():
    return {"entrypoint": "agent.sh", "files_sha256": {"agent.sh": "b" * 64}, "tool_interfaces": ["shell", "file_read", "vision"],
            "sampling": {"temperature": 0, "max_tokens": 1024}, "max_model_calls": 100, "timeout_seconds": 8100,
            "all_model_calls_use_same_actor": True, "gold_visible_to_agent": False}


@pytest.fixture
def runtime():
    return {"platform": "local", "source_revision": runner.REVISION, "benchmark_level": runner.LEVEL,
            "image_digest": "sha256:" + "c" * 64, "gold_dataset_not_mounted": True, "gpu_available": False}


def environment_for(task, root):
    environment = root / "environment"
    (environment / task["capsule_id"] / "results").mkdir(parents=True)
    (environment / task["capsule_id"] / "code").mkdir()
    (environment / "task.txt").write_text(task["task_txt"])
    return environment


def asset_for(task):
    return {"capsule_id": task["capsule_id"], "archive_url": task["archive_url"], "archive_sha256": "d" * 64,
            "extracted_tree_sha256": "e" * 64, "license_receipt_sha256": "f" * 64,
            "uses_gpu": False, "registry_link": "registry.codeocean.com/published/fixture:v1"}


def test_exact_hard45_test_denominator_and_no_gold(prepared):
    manifest = prepared[1]
    assert manifest["benchmark_level"] == "codeocean_hard"
    assert len(manifest["tasks"]) == 45
    assert sum(len(task["questions"]) for task in manifest["tasks"]) == 79
    assert manifest["protocol"]["task_limit"] is None
    assert manifest["protocol"]["no_gpu"] is False
    assert manifest["protocol"]["include_correct_result_paths"] is False
    assert all("results" not in task and task["archive_sha256"] is None and task["uses_gpu"] is None for task in manifest["tasks"])


def test_easier_level_or_partial_test_manifest_rejected(prepared):
    easier = copy.deepcopy(prepared[1])
    easier["benchmark_level"] = "codeocean_easy"
    with pytest.raises(runner.ContractError, match="HARD"):
        runner.validate_manifest(easier)
    partial = copy.deepcopy(prepared[1])
    partial["expected_total"] = 44
    partial["tasks"].pop()
    with pytest.raises(runner.ContractError, match="45hard"):
        runner.validate_manifest(partial)


def test_exact_upstream_hard_environment_removals_and_prompt(prepared, tmp_path, monkeypatch):
    task = prepared[1]["tasks"][0]
    row = runner.dataset(runner.DEFAULT_SETUP)[0]
    source = runner.DEFAULT_SETUP / "source"
    cls = next(n for n in ast.parse((source / "benchmark/benchmark.py").read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CodeOceanBenchmark")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__setup_task_environment")
    namespace = {"json": json, "os": os, "shutil": shutil}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "native_hard_setup_fixture", "exec"), namespace)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "docker").mkdir()
    shutil.copyfile(source / "benchmark/benchmark_prompts.json", tmp_path / "benchmark/benchmark_prompts.json")
    shutil.copyfile(source / "docker/Dockerfile", tmp_path / "docker/Dockerfile")
    agent_dir = tmp_path / "fixture_agent"
    agent_dir.mkdir()
    capsule = tmp_path / "fixture_data" / task["capsule_id"]
    for directory in ("results", "environment", "code"):
        (capsule / directory).mkdir(parents=True)
    for file in ("results/original.txt", "REPRODUCING.md", "environment/Dockerfile", "code/run", "code/run.sh"):
        (capsule / file).write_text("fixture only")
    native_self = SimpleNamespace(experiment_name="fixture", timestamp="fixture", agent_dir=str(agent_dir),
                                  dataset_dir=str(capsule.parent), benchmark_level=runner.LEVEL, include_correct_result_paths=False)
    native_task = SimpleNamespace(capsule_id=task["capsule_id"], task_prompt=row["task_prompt"], results=row["results"],
                                  registry_link="registry.codeocean.com/published/fixture:v1")
    monkeypatch.chdir(tmp_path)
    namespace["__setup_task_environment"](native_self, native_task)
    environment = tmp_path / "benchmark/temp_envs/fixture" / f'{task["capsule_id"]}-fixture' / "environment"
    inspected = runner.inspect_hard_environment(environment, task)
    assert inspected["hard_removals_verified"] is True
    assert (environment / "task.txt").read_text() == task["task_txt"]


@pytest.mark.parametrize("leak", ["results/original.txt", "REPRODUCING.md", "environment", "code/run", "code/run.sh"])
def test_hard_environment_rejects_easier_assets(prepared, tmp_path, leak):
    task = prepared[1]["tasks"][0]
    environment = environment_for(task, tmp_path)
    target = environment / task["capsule_id"] / leak
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fixture leaked hint")
    with pytest.raises(runner.ContractError):
        runner.inspect_hard_environment(environment, task)


def test_native_numeric_string_list_and_vision_rules():
    native = runner.load_native_evaluator(runner.DEFAULT_SETUP)
    gold = [{"value": value, "text": "Example", "fig list": [1, 2]} for value in (10, 12, 14)]
    correct = native.eval_result_json(gold, {"value": "12%", "text": "EXAMPLE", "fig list": [1, 2]})
    assert correct == {"correct_written_answers": 2, "correct_vision_answers": 1, "total_written_questions": 2, "total_vision_questions": 1}
    wrong = native.eval_result_json(gold, {"value": 1000, "text": "wrong", "fig list": [2, 1]})
    assert wrong["correct_written_answers"] == 0 and wrong["correct_vision_answers"] == 0
    assert native.eval_result_json(gold, {})["total_written_questions"] == 2


def test_native_aggregate_requires_all_answers(tmp_path):
    native = runner.load_native_evaluator(runner.DEFAULT_SETUP)
    rows = [{"capsule_id": f"fixture-{i}", "correct_written_answers": 1, "total_written_questions": 1,
             "correct_vision_answers": 1 if i == 0 else 0, "total_vision_questions": 1} for i in range(45)]
    path = tmp_path / "fixture_native_results.json"
    path.write_text(json.dumps({"capsule_results": rows}))
    native.score_results(str(path), llm_summary=False)
    summary = json.loads(path.read_text())["summary"]
    assert summary["correct_tasks"] == 1 and summary["total_tasks"] == 45
    assert summary["correct_written_tasks"] == 45


def test_actor_model_and_wandb_gates(prepared, model, wandb, agent):
    with pytest.raises(runner.ContractError, match="W&B"):
        runner.create_contract(prepared[1], model, {**wandb, "mode": "offline"}, agent)
    wrong = {**model, "base_model": {**model["base_model"], "revision": "a" * 40}}
    with pytest.raises(runner.ContractError, match="exact portfolio"):
        runner.create_contract(prepared[1], wrong, wandb, agent)
    with pytest.raises(runner.ContractError, match="call budget"):
        runner.create_contract(prepared[1], model, wandb, {**agent, "max_model_calls": None})


def test_reservation_blocks_unknown_attempt_and_environment_drift(prepared, model, wandb, agent, runtime, tmp_path):
    task = prepared[1]["tasks"][0]
    environment = environment_for(task, tmp_path)
    args = (runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", task["capsule_id"], model, wandb, agent, asset_for(task), runtime, environment)
    assert runner.reserve_task(*args)["status"] == "RESERVED_NOT_EXECUTED"
    with pytest.raises(runner.ContractError, match="unresolved attempt"):
        runner.reserve_task(*args)
    (environment / task["capsule_id"] / "code/new_file.py").write_text("fixture")
    with pytest.raises(runner.ContractError, match="collision"):
        runner.reserve_task(*args)


def test_partial_score_never_calls_native_grader(prepared, model, wandb, agent, runtime, tmp_path, monkeypatch):
    task = prepared[1]["tasks"][0]
    environment = environment_for(task, tmp_path)
    runner.reserve_task(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", task["capsule_id"], model, wandb, agent,
                        asset_for(task), runtime, environment)
    monkeypatch.setattr(runner, "load_native_evaluator", lambda *args: pytest.fail("partial run cannot invoke full scoring"))
    receipt = runner.score(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", tmp_path / "partial.json")
    assert receipt["score"] is None and receipt["native_summary"] is None
    assert receipt["expected_total"] == 45 and receipt["native_terminal_capsules"] == 0


def test_native_ingest_requires_actual_artifacts_and_preserves_one_terminal(prepared, model, wandb, agent, runtime, tmp_path):
    task = prepared[1]["tasks"][0]
    environment = environment_for(task, tmp_path)
    run_dir = tmp_path / "run"
    reserve = runner.reserve_task(runner.DEFAULT_SETUP, prepared[0], run_dir, task["capsule_id"], model, wandb, agent, asset_for(task), runtime, environment)
    artifacts = tmp_path / "fixture_artifacts"
    artifacts.mkdir()
    hashes = {}
    for name in ("agent_trace.log", "raw_model_requests.jsonl", "raw_model_responses.jsonl", "container.log"):
        (artifacts / name).write_text("fixture only: no model calls")
        hashes[name] = runner.digest((artifacts / name).read_bytes())
    result = {key: task[key] for key in ("capsule_id", "field", "language", "capsule_title")}
    result.update({"result_report": {}, "correct_written_answers": 0, "correct_vision_answers": 0,
                   "total_written_questions": sum("fig" not in q for q in task["questions"]),
                   "total_vision_questions": sum("fig" in q for q in task["questions"])})
    native_path = tmp_path / "fixture_native.json"
    native_path.write_bytes(runner.canonical({"capsule_results": [result]}))
    execution = {"capsule_id": task["capsule_id"], "submission_sha256": reserve["submission_sha256"], "benchmark_level": runner.LEVEL,
                 "source_revision": runner.REVISION, "terminal": True, "outcome": "agent_failed", "started_at": "fixture", "finished_at": "fixture",
                 "model_provenance_sha256": runner.fingerprint(model), "native_results_sha256": runner.digest(native_path.read_bytes()),
                 "artifacts_sha256": hashes, "report_status": "missing_or_invalid"}
    args = (runner.DEFAULT_SETUP, prepared[0], run_dir, task["capsule_id"], native_path, execution, artifacts)
    assert runner.ingest_native(*args)["score"] is None
    assert runner.ingest_native(*args)["score"] is None
    execution["outcome"] = "timeout"
    with pytest.raises(runner.ContractError, match="collision"):
        runner.ingest_native(*args)
    # The bridge retains the original bytes; later external log changes cannot
    # destroy the captured evidence, and mutations of retained evidence fail.
    (artifacts / "container.log").write_text("external artifact later changed")
    receipt = runner.score(runner.DEFAULT_SETUP, prepared[0], run_dir, tmp_path / "partial_retained.json")
    assert receipt["native_terminal_capsules"] == 1 and receipt["score"] is None
    retained = run_dir / "tasks" / task["capsule_id"] / "artifacts/container.log"
    retained.write_text("retained artifact tampered")
    with pytest.raises(runner.ContractError, match="retained execution artifact drift"):
        runner.score(runner.DEFAULT_SETUP, prepared[0], run_dir, tmp_path / "must_not_score.json")
