"""Native-source integration tests; fixtures are never benchmark results."""
import asyncio
import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import public_portfolio_native as runner


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    # Do not rewrite older preparation receipts when the native policy changes.
    path = tmp_path_factory.mktemp("labbench-native-prepared")
    manifest, keys = runner.build_manifest(runner.DEFAULT_SETUP)
    runner.write_once(path / "manifest.json", manifest)
    runner.write_once(path / "answer_key.json", keys)
    return path, manifest, keys


@pytest.fixture(scope="module")
def native(prepared):
    return runner.load_native(runner.DEFAULT_SETUP)


@pytest.fixture
def provenance():
    return {"base_model": {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0"},
            "adapter": {"repo_id": "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6",
                        "revision": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"},
            "adapter_loaded": True, "vision": True, "served_model": "fixture-model",
            "deployment_receipt_sha256": "a" * 64, "deployment_verified_at": "fixture-only"}


@pytest.fixture
def wandb():
    return {"mode": "online", "run_id": "fixture-only", "url": "https://wandb.ai/fixture/fixture/runs/fixture-only",
            "initialized_before_model_work": True}


@pytest.fixture
def sampling():
    return {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": 809, "n": 1, "extra_body": {}}


@pytest.fixture
def contract(prepared, provenance, wandb, sampling):
    return runner.create_contract(prepared[1], provenance, wandb, sampling, "http://localhost:9999/v1")


def test_native_full_denominator_and_images(prepared):
    manifest = prepared[1]
    assert len(manifest["tasks"]) == 1967
    assert sum(runner.EXPECTED_COUNTS.values()) == 1967
    assert sum(bool(t["images"]) for t in manifest["tasks"]) == 425
    assert sum(len(t["images"]) for t in manifest["tasks"]) == 443
    assert manifest["source"]["revision"] == runner.SOURCE_REVISION


def test_native_validator_rejects_partial_split(prepared):
    partial = copy.deepcopy(prepared[1])
    partial["tasks"].pop()
    partial["expected_total"] = 1966
    with pytest.raises(runner.ContractError, match="1967"):
        runner.validate_manifest(partial)


def test_duplicate_tasks_are_rejected(prepared):
    duplicated = copy.deepcopy(prepared[1])
    duplicated["tasks"][0] = duplicated["tasks"][1]
    with pytest.raises(runner.ContractError, match="duplicate"):
        runner.validate_manifest(duplicated)


def test_generation_payload_has_no_gold_metadata(prepared, contract):
    task = prepared[1]["tasks"][0]
    payload = runner.request_payload(task, contract, runner.DEFAULT_SETUP)
    assert set(payload) == {"model", "messages", "temperature", "top_p", "max_tokens", "seed", "n", "stream"}
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": task["prompt"]}]}]
    assert all(field not in task for field in ("ideal", "target_choice", "unsure_choice", "key-passage", "canary"))


def test_image_payload_contains_verified_actual_images(prepared, contract):
    task = next(t for t in prepared[1]["tasks"] if t["images"])
    payload = runner.request_payload(task, contract, runner.DEFAULT_SETUP)
    content = payload["messages"][0]["content"]
    assert len(content) == 1 + len(task["images"])
    assert content[1]["image_url"]["url"].startswith("data:image/")


def test_online_wandb_and_loaded_adapter_are_required(prepared, provenance, wandb, sampling):
    for field, value in (("mode", "offline"), ("initialized_before_model_work", False)):
        wrong = {**wandb, field: value}
        with pytest.raises(runner.ContractError):
            runner.create_contract(prepared[1], provenance, wrong, sampling, "http://localhost/v1")
    for field in ("vision", "adapter_loaded"):
        with pytest.raises(runner.ContractError):
            runner.create_contract(prepared[1], {**provenance, field: False}, wandb, sampling, "http://localhost/v1")


def test_sampling_cannot_resample_or_override_prompt(prepared, provenance, wandb, sampling):
    for changed in ({**sampling, "n": 2}, {**sampling, "extra_body": {"messages": []}}):
        with pytest.raises(runner.ContractError):
            runner.create_contract(prepared[1], provenance, wandb, changed, "http://localhost/v1")


def transport_fixture(counter, content="[ANSWER]A[/ANSWER]"):
    def transport(url, payload, api_key, timeout):
        counter.append(payload)
        return 200, json.dumps({"model": payload["model"], "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}).encode()
    return transport


def native_evaluator_for_task(native, task):
    inp = native.AgentInput(id=task["instance_id"], question=task["question"], choices=task["choices"])
    instance = SimpleNamespace(id=inp.id, get_input_output=lambda: (inp, "A", "B"))
    evaluator = object.__new__(native.Evaluator)
    evaluator.eval = native.Eval.CloningScenarios
    evaluator.eval_set = [("fixture", instance)]
    return evaluator


def test_resume_uses_original_response_no_resampling(prepared, contract, tmp_path):
    calls = []
    task = prepared[1]["tasks"][0]
    assert runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls))
    assert not runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls))
    assert len(calls) == 1


def test_uncertain_transport_blocks_retry(prepared, contract, tmp_path):
    calls = []
    def interrupted(*args):
        calls.append(1)
        raise TimeoutError("fixture transport lost after submission")
    task = prepared[1]["tasks"][0]
    with pytest.raises(runner.ContractError, match="resampling blocked"):
        runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=interrupted)
    with pytest.raises(runner.ContractError, match="ambiguous prior request"):
        runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=interrupted)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["prompt", "model", "sampling"])
def test_resume_rejects_model_prompt_sampling_collision(prepared, contract, tmp_path, change):
    task = copy.deepcopy(prepared[1]["tasks"][0])
    calls = []
    runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls))
    altered = copy.deepcopy(contract)
    if change == "prompt":
        task["prompt"] += " altered"
        task["prompt_sha256"] = runner.digest(task["prompt"].encode())
    elif change == "model":
        altered["provenance"]["served_model"] = "wrong-model"
    else:
        altered["sampling"]["temperature"] = 0.9
    with pytest.raises(runner.ContractError, match="collision"):
        runner.perform_task(task, altered, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls))
    assert len(calls) == 1


@pytest.mark.parametrize("text,answer,error", [
    ("reasoning\n[ANSWER]B[/ANSWER]", "B", None),
    ("A", "A", None),
    ("[ANSWER]A,B[/ANSWER]", "A", None),
    ("ordinary malformed output", None, None),
])
def test_actual_native_answer_parser(native, prepared, text, answer, error):
    parsed = asyncio.run(runner.parse_native(native, prepared[1]["tasks"][0], text, True))
    assert parsed["agent_output"] == answer
    assert parsed["parser_error"] == error
    assert parsed["unanswerable"] == bool(error)


def test_empty_output_upstream_raises_and_adapter_blocks_without_grade(native, prepared, contract, tmp_path, monkeypatch):
    task = prepared[1]["tasks"][0]

    class NativeEmptyAgent(native.BaseZeroShotAgent):
        async def get_completion(self, text_prompt, figs):
            return ""

    inp = native.AgentInput(id=task["instance_id"], question=task["question"], choices=task["choices"])
    # The same upstream parser fails; it does not return an incorrect/unsure grade.
    with pytest.raises(TypeError):
        asyncio.run(NativeEmptyAgent(use_cot=True).run_task(inp))
    with pytest.raises(TypeError):
        asyncio.run(native_evaluator_for_task(native, task).score_agent(NativeEmptyAgent(use_cot=True).run_task))
    with pytest.raises(runner.ContractError, match="native parser raised TypeError") as blocked:
        asyncio.run(runner.parse_native(native, task, "", True))
    assert isinstance(blocked.value.__cause__, TypeError)

    runner.write_once(tmp_path / "run_contract.json", contract)
    calls = []
    runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls, ""))
    # Preparation has already run actual native source validation above.
    monkeypatch.setattr(runner, "verify_manifest", lambda *args: None)
    receipt = runner.score(runner.DEFAULT_SETUP, prepared[0], tmp_path, tmp_path / "blocked_receipt.json")
    assert receipt["evaluated"] == 0
    assert receipt["score"] is None and receipt["metrics_all"] is None
    assert receipt["blocked_tasks"][0]["task_id"] == task["task_id"]
    assert "native parser raised TypeError" in receipt["blocked_tasks"][0]["reason"]
    directory = tmp_path / "tasks" / task["category"] / task["instance_id"]
    assert not (directory / "native_grade.json").exists()
    assert (directory / "response.json").exists()
    assert not runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls, ""))
    assert len(calls) == 1


def test_actual_native_unanswerable_error_retains_abstention_behavior(native, prepared, contract, tmp_path, monkeypatch):
    task = prepared[1]["tasks"][0]

    async def native_abstention(self, inp):
        raise native.UnanswerableError("explicit native agent abstention fixture")

    async def upstream_abstention(inp):
        raise native.UnanswerableError("explicit native agent abstention fixture")

    upstream = asyncio.run(native_evaluator_for_task(native, task).score_agent(upstream_abstention))
    upstream_result = next(iter(upstream["results"].values()))
    assert upstream_result["correct"] is False and upstream_result["sure"] is False

    monkeypatch.setattr(native.BaseZeroShotAgent, "run_task", native_abstention)
    calls = []
    runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=transport_fixture(calls))
    directory = tmp_path / "tasks" / task["category"] / task["instance_id"]
    result = runner.grade_response(native, task, prepared[2]["answers"][task["task_id"]],
                                   runner.read_json(directory / "response.json"), contract,
                                   runner.read_json(directory / "request.json"), True)
    assert result["unanswerable"] is True
    assert result["correct"] is False and result["sure"] is False
    assert result["correct"] == upstream_result["correct"] and result["sure"] == upstream_result["sure"]
    assert result["parser_error"] is None


def test_native_metrics_and_no_partial_as_full(native, prepared):
    # Deliberately synthetic full fixture; this result is never persisted as an experiment.
    rows = [{"task_id": t["task_id"], "category": t["category"], "correct": i % 2 == 0, "sure": True}
            for i, t in enumerate(prepared[1]["tasks"])]
    partial = runner.summarize(native, prepared[1], rows[:-1], [])
    assert partial["score"] is None and partial["metrics_all"] is None
    assert partial["expected_total"] == 1967 and partial["missing"] == 1
    complete = runner.summarize(native, prepared[1], rows, [])
    assert complete["metrics_all"] == native.Evaluator.compute_metrics(rows)
    assert complete["score"] == 984 / 1967
    assert complete["metrics_all"]["n_total"] == 1967
    assert complete["categories"]["TableQA"]["expected_total"] == 244


def test_malformed_envelope_not_graded_or_retried(native, prepared, contract, tmp_path):
    task = prepared[1]["tasks"][0]
    runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=lambda *args: (200, b"not-json"))
    directory = tmp_path / "tasks" / task["category"] / task["instance_id"]
    with pytest.raises(runner.ContractError, match="malformed transport"):
        runner.grade_response(native, task, prepared[2]["answers"][task["task_id"]], runner.read_json(directory / "response.json"),
                              contract, runner.read_json(directory / "request.json"), True)
    assert not runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path,
                                   transport=lambda *args: pytest.fail("must not resample"))


def test_response_model_mismatch_blocks_native_grade(native, prepared, contract, tmp_path):
    task = prepared[1]["tasks"][0]
    runner.perform_task(task, contract, runner.DEFAULT_SETUP, tmp_path, transport=lambda *args: (200, b'{"model":"wrong","choices":[]}'))
    directory = tmp_path / "tasks" / task["category"] / task["instance_id"]
    with pytest.raises(runner.ContractError, match="served model mismatch"):
        runner.grade_response(native, task, prepared[2]["answers"][task["task_id"]], runner.read_json(directory / "response.json"),
                              contract, runner.read_json(directory / "request.json"), True)


def test_batch_export_ingest_resume_and_collision(prepared, provenance, wandb, sampling, tmp_path, monkeypatch):
    # Source is verified by the module fixture; avoid three redundant 233 MB rereads.
    monkeypatch.setattr(runner, "verify_manifest", lambda *args: None)
    output = tmp_path / "batch.jsonl"
    result = runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", provenance, wandb, sampling,
                                    "batch://fixture/native", output, max_new=1)
    assert result["count"] == 1 and result["score"] is None
    request = json.loads(output.read_text())
    assert "ideal" not in request and "target_choice" not in request
    with pytest.raises(runner.ContractError, match="already exists"):
        runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", provenance, wandb, sampling,
                               "batch://fixture/native", output, max_new=1)
    raw = b'{"model":"fixture-model","choices":[{"message":{"content":"[ANSWER]A[/ANSWER]"}}]}'
    response = {k: request[k] for k in ("task_id", "batch_id", "contract_sha256", "request_sha256")}
    response.update({"http_status": 200, "started_at": "fixture-start", "received_at": "fixture-end",
                     "raw_body_base64": base64.b64encode(raw).decode()})
    response_path = tmp_path / "responses.jsonl"
    response_path.write_bytes(runner.canonical(response))
    assert runner.ingest_responses(prepared[0], tmp_path / "run", response_path)["count"] == 1
    assert runner.ingest_responses(prepared[0], tmp_path / "run", response_path)["count"] == 1
    response["raw_body_base64"] = base64.b64encode(b"different sample").decode()
    response_path.write_bytes(runner.canonical(response))
    with pytest.raises(runner.ContractError, match="collision"):
        runner.ingest_responses(prepared[0], tmp_path / "run", response_path)


def test_batch_duplicate_response_ids_rejected(prepared, tmp_path):
    # Duplicate protection is also exercised without touching any provider.
    assert runner.fingerprint({"a": 1}) == runner.fingerprint({"a": 1})
    with pytest.raises(runner.ContractError, match="collision"):
        runner.write_once(tmp_path / "immutable.json", {"a": 1})
        runner.write_once(tmp_path / "immutable.json", {"a": 2})
