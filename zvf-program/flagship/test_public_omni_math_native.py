"""Native Omni-MATH integration checks; all model responses here are fixtures."""
import base64
import copy
import json

import pytest

import public_omni_math_native as runner


@pytest.fixture(scope="module")
def prepared():
    path = runner.DEFAULT_SETUP / "prepared"
    manifest, keys = runner.verified_prepared(runner.DEFAULT_SETUP, path)
    return path, manifest, keys


@pytest.fixture(scope="module")
def tokenizer():
    return runner.load_tokenizer(runner.DEFAULT_SETUP)


@pytest.fixture(scope="module")
def scorer():
    return runner.load_scorer(runner.DEFAULT_SETUP)


@pytest.fixture
def wandb():
    return {"mode": "online", "run_id": "fixture", "url": "https://wandb.ai/fixture/p/runs/fixture",
            "initialized_before_model_work": True}


@pytest.fixture
def actor():
    return {"served_model": "fixture-actor", "model_loaded": True, "adapter_loaded": True,
            "base_model": {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0"},
            "adapter": {"repo_id": "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6",
                        "revision": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"},
            "deployment_receipt_sha256": "a" * 64, "deployment_verified_at": "fixture"}


@pytest.fixture
def judge():
    return {"served_model": "fixture-judge", "model_loaded": True,
            "model": {"repo_id": runner.JUDGE_REPO, "revision": runner.JUDGE_REVISION},
            "deployment_receipt_sha256": "b" * 64, "deployment_verified_at": "fixture"}


def response_from_request(request, text, model=None):
    if request["kind"] == "actor":
        choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "text": text, "finish_reason": "stop"}
    raw = json.dumps({"model": model or request["payload"]["model"], "choices": [choice]}).encode()
    response = {k: request[k] for k in ("task_id", "kind", "batch_id", "contract_sha256", "request_sha256")}
    response.update({"http_status": 200, "raw_body_base64": base64.b64encode(raw).decode(),
                     "started_at": "fixture-start", "received_at": "fixture-end"})
    return response


def save_response(path, row):
    path.write_bytes(runner.canonical(row))
    return path


def export_actor(prepared, actor, wandb, tmp_path, max_new=1):
    batch = tmp_path / "actor-batch.jsonl"
    runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", actor, wandb, batch, max_new=max_new)
    return [json.loads(line) for line in batch.read_text().splitlines()]


def test_full_split_denominator_retains_native_duplicates(prepared):
    manifest, keys = prepared[1:]
    assert len(manifest["tasks"]) == 4428
    rows = list(keys["rows"].values())
    assert len({r["problem"] for r in rows}) == 4406
    assert len({runner.fingerprint(r) for r in rows}) == 4424
    assert len({t["task_id"] for t in manifest["tasks"]}) == 4428


def test_actor_prompt_contains_only_problem_and_official_system(prepared):
    for task in prepared[1]["tasks"]:
        row = prepared[2]["rows"][task["task_id"]]
        assert task["messages"] == [{"role": "system", "content": runner.SYSTEM_PROMPT}, {"role": "user", "content": row["problem"]}]
        assert not {"answer", "solution", "reference_answer"} & set(task)


def test_shrunk_denominator_and_row_id_collision_rejected(prepared):
    manifest = copy.deepcopy(prepared[1])
    manifest["tasks"].pop()
    manifest["expected_total"] = 4427
    with pytest.raises(runner.ContractError, match="4428"):
        runner.validate_manifest(manifest)
    manifest = copy.deepcopy(prepared[1])
    manifest["tasks"][0] = manifest["tasks"][1]
    with pytest.raises(runner.ContractError, match="duplicate"):
        runner.validate_manifest(manifest)


def test_online_wandb_and_separate_pinned_judge(prepared, actor, judge, wandb):
    with pytest.raises(runner.ContractError, match="W&B"):
        runner.create_contract(prepared[1], "actor", actor, {**wandb, "mode": "offline"})
    with pytest.raises(runner.ContractError, match="base-only"):
        runner.create_contract(prepared[1], "actor", {**actor, "adapter_loaded": False}, wandb)
    changed = {**judge, "model": {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "a" * 40}}
    with pytest.raises(runner.ContractError, match="only pinned native Omni-Judge"):
        runner.create_contract(prepared[1], "judge", changed, wandb)


def test_native_actor_and_judge_sampling_cannot_be_overridden(prepared, actor, judge, wandb):
    assert runner.create_contract(prepared[1], "actor", actor, wandb)["sampling"]["max_tokens"] == 2048
    assert runner.create_contract(prepared[1], "judge", judge, wandb)["sampling"]["max_tokens"] == 300
    with pytest.raises(runner.ContractError, match="override"):
        runner.create_contract(prepared[1], "actor", actor, wandb, {"n": 2})
    with pytest.raises(runner.ContractError, match="judge accepts no"):
        runner.create_contract(prepared[1], "judge", judge, wandb, {"chat_template_kwargs": {"enable_thinking": False}})


def test_native_judge_context_uses_actual_tokenizer(tokenizer, prepared, judge, wandb):
    row = next(iter(prepared[2]["rows"].values()))
    contract = runner.create_contract(prepared[1], "judge", judge, wandb)
    payload = runner.judge_payload(row, "FIXTURE STUDENT SOLUTION", contract, tokenizer)
    assert payload["prompt"] == tokenizer.get_context(row["problem"], row["answer"], "FIXTURE STUDENT SOLUTION")
    assert payload["prompt"].startswith("<|begin_of_text|>")
    assert payload["prompt"].endswith("## Student Final Answer")
    assert payload["stop"] == [tokenizer.eos_token, "<|eot_id|>"]
    assert "messages" not in payload


def test_native_report_parser_behavior_and_malformed_gate(scorer):
    valid = "## Student Final Answer\n42\n\n## Equivalence Judgement\nTRUE\n\n## Justification\nSame."
    assert scorer.parse_report(valid)["Equivalence Judgement"] == "TRUE"
    rows = [{"omni_judge": valid} for _ in range(4428)]
    runner.validate_native_reports(rows, scorer)
    rows[-1] = {"omni_judge": "malformed no report headings"}
    with pytest.raises(runner.ContractError, match="skip a malformed"):
        runner.validate_native_reports(rows, scorer)


def test_native_full_scorer_denominator_exact(scorer, tmp_path):
    rows = [{"domain": ["fixture"], "difficulty": 1, "source": "fixture", "problem": "fixture", "answer": "fixture",
             "model_generation": "fixture", "omni_judge": f"## Equivalence Judgement\n{'TRUE' if i == 0 else 'FALSE'}"} for i in range(4428)]
    score, stdout, _ = runner.native_score(rows, scorer, tmp_path)
    assert score == 1 / 4428
    assert stdout.startswith("Total Accuracy:")
    with pytest.raises(runner.ContractError, match="4428"):
        runner.native_score(rows[:-1], scorer, tmp_path / "partial")


def test_export_reserves_once_and_skips_uncertain_samples(prepared, actor, wandb, tmp_path):
    first = export_actor(prepared, actor, wandb, tmp_path)[0]
    second_path = tmp_path / "second.jsonl"
    runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", actor, wandb, second_path, max_new=1)
    second = json.loads(second_path.read_text())
    assert first["task_id"] != second["task_id"]
    with pytest.raises(runner.ContractError, match="already exported"):
        runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", actor, wandb, second_path, max_new=1)


def test_resume_rejects_model_and_sampling_mismatch(prepared, actor, wandb, tmp_path):
    export_actor(prepared, actor, wandb, tmp_path)
    with pytest.raises(runner.ContractError, match="collision"):
        runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", {**actor, "served_model": "other"}, wandb,
                               tmp_path / "second.jsonl", max_new=1)
    with pytest.raises(runner.ContractError, match="collision"):
        runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", actor, wandb,
                               tmp_path / "second.jsonl", max_new=1, extra_body={"seed": 200})


def test_actor_response_idempotence_and_changed_sample_collision(prepared, actor, wandb, tmp_path):
    request = export_actor(prepared, actor, wandb, tmp_path)[0]
    response_path = save_response(tmp_path / "response.jsonl", response_from_request(request, "ACTOR FIXTURE"))
    assert runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", response_path)["count"] == 1
    assert runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", response_path)["count"] == 1
    save_response(response_path, response_from_request(request, "SECOND DRAW"))
    with pytest.raises(runner.ContractError, match="collision"):
        runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", response_path)


def test_duplicate_response_task_rejected(prepared, actor, wandb, tmp_path):
    request = export_actor(prepared, actor, wandb, tmp_path)[0]
    response = response_from_request(request, "fixture")
    path = tmp_path / "duplicates.jsonl"
    path.write_bytes(runner.canonical(response) * 2)
    with pytest.raises(runner.ContractError, match="duplicate"):
        runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", path)


def test_actor_to_judge_to_partial_score_keeps_full_score_null(prepared, actor, judge, wandb, tmp_path):
    request = export_actor(prepared, actor, wandb, tmp_path)[0]
    actor_response = save_response(tmp_path / "actor-response.jsonl", response_from_request(request, "ACTOR FIXTURE"))
    runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", actor_response)
    judge_batch = tmp_path / "judge-batch.jsonl"
    runner.export_requests(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "judge", judge, wandb, judge_batch, max_new=1)
    judge_request = json.loads(judge_batch.read_text())
    assert judge_request["task_id"] == request["task_id"]
    assert judge_request["api_path"] == "/v1/completions"
    assert "ACTOR FIXTURE" in judge_request["payload"]["prompt"]
    judge_response = save_response(tmp_path / "judge-response.jsonl", response_from_request(judge_request,
                                   "42\n\n## Equivalence Judgement\nTRUE\n\n## Justification\nFixture only."))
    runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "judge", judge_response)
    receipt = runner.score(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", tmp_path / "partial.json")
    assert receipt["actor_samples"] == 1 and receipt["native_judged_samples"] == 1
    assert receipt["score"] is None and receipt["expected_total"] == 4428


def test_wrong_model_response_is_preserved_but_not_validated(prepared, actor, wandb, tmp_path):
    request = export_actor(prepared, actor, wandb, tmp_path)[0]
    response = save_response(tmp_path / "wrong.jsonl", response_from_request(request, "fixture", model="wrong"))
    runner.ingest_responses(runner.DEFAULT_SETUP, prepared[0], tmp_path / "run", "actor", response)
    contract = runner.read_json(tmp_path / "run/actor/contract.json")
    with pytest.raises(runner.ContractError, match="served model mismatch"):
        runner.checked_response(tmp_path / "run", "actor", request["task_id"], contract)
