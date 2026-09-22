import ast
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).with_name("public_balrog_native.py")
spec = importlib.util.spec_from_file_location("balrog_bridge_test", PATH)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


@pytest.fixture
def manifest():
    return b.prepare()


def observation(text="An actual captured observation fixture"):
    return {"text": {"long_term_context": text, "short_term_context": "Inventory fixture"}, "image": None}


def contract(tmp_path):
    actor = {**b.ACTOR, "served_model_id": "pavlov-public-portfolio-bf16"}
    wandb = {"mode": "online", "initialized_before_model_work": True, "run_id": "fixture",
             "url": "https://wandb.ai/fixture/project/runs/fixture"}
    runtime = {"cpu_image_digest": "sha256:" + "f" * 64, "packages_manifest_sha256": "a" * 64,
        "assets_manifest_sha256": "b" * 64, "actor_provenance_sha256": "c" * 64,
        "dependency_revisions": b.DEPENDENCY_REVISIONS, "sdk_retries": 2}
    return b.reserve_run(b.DEFAULT_SETUP, tmp_path, actor, wandb, runtime)


def context():
    return {"seed": 809, "initial_state_sha256": "a" * 64, "env_max_steps": 64,
            "native_env_config_sha256": b.fingerprint(b.prepare()["native_config"]["envs"]),
            "worker_rng_state_sha256": "b" * 64, "game_assets_sha256": "c" * 64}


def response(row, content="go forward"):
    raw = b.canonical({"model": row["payload"]["model"], "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    return {"task_id": row["task_id"], "step": row["step"], "contract_sha256": row["contract_sha256"],
            "request_sha256": row["request_sha256"], "http_status": 200, "raw_body_base64": base64.b64encode(raw).decode(),
            "started_at": "2026-09-05T01:00:00Z", "received_at": "2026-09-05T01:00:01Z"}


def test_full_denominators_and_call_caps(manifest):
    assert len(manifest["episodes"]) == len({x["episode_id"] for x in manifest["episodes"]}) == 255
    assert sum(x["task_configurations"] for x in manifest["environments"].values()) == 58
    assert {k:v["max_agent_steps"] for k,v in manifest["environments"].items()} == {
        "babyai": 5120, "babaisai": 12000, "textworld": 2400, "crafter": 20000, "nle": 500000, "minihack": 4000}
    assert manifest["request_limits"]["logical_generations"] == 543520
    assert manifest["request_limits"]["sdk_create_invocations"] == 2717600
    assert manifest["request_limits"]["http_attempts"] is None
    assert manifest["request_limits"]["logical_step_completion_token_ceiling"] == 4452515840
    assert manifest["score"] is None


def test_static_baba_default_and_babyai_navigation_formula_are_in_pinned_source():
    root = b.DEFAULT_SETUP / "dependency_sources"
    tree = ast.parse((root / "nacloos--baba-is-ai/source/baba/grid.py").read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "BabaIsYouEnv")
    init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
    assert ast.literal_eval(defaults["max_steps"]) == 100
    room = (root / "BartekCupial--Minigrid/source/minigrid/envs/babyai/core/roomgrid_level.py").read_text()
    assert "self.max_steps = num_navs * nav_time_maze" in room
    assert "nav_time_room = self.room_size**2" in room
    assert "nav_time_maze = nav_time_room * self.num_rows * self.num_cols" in room


def test_selected_baba_and_babyai_registrations_do_not_override_derived_limits(manifest):
    root = b.DEFAULT_SETUP / "dependency_sources"
    source = (root / "nacloos--baba-is-ai/source/baba/envs.py").read_text()
    tree = ast.parse(source)
    registered = {call.args[0].value for node in tree.body if isinstance(node, ast.ClassDef)
                  for call in node.decorator_list if isinstance(call, ast.Call)
                  and isinstance(call.func, ast.Name) and call.func.id == "register"}
    assert set(manifest["native_config"]["tasks"]["babaisai_tasks"]).issubset(registered)
    assert "max_steps" not in source  # selected subclasses inherit base100; config passes no override.
    path = root / "BartekCupial--Minigrid/registration_inspection.py"
    receipt = b.read(path.with_name("registration_inspection_receipt.json"))
    assert b.digest(path.read_bytes()) == receipt["sha256"]
    calls = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "register"]
    selected = next(node for node in calls if any(k.arg == "id" and isinstance(k.value, ast.Constant)
                    and k.value.value == "BabyAI-MixedTrainLocal-v0" for k in node.keywords))
    assert {k.arg for k in selected.keywords} == {"id", "entry_point"}


def test_native_messages_and_sampling_are_preserved():
    payload = b.build_native_request(b.DEFAULT_SETUP, "fixture-model", "Native instruction fixture", [observation()], [])
    assert payload["temperature"] == 1.0 and payload["max_tokens"] == 8192
    assert set(payload) == {"messages", "model", "temperature", "max_tokens"}
    assert payload["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "Native instruction fixture"}]}
    assert "Current Observation:" in payload["messages"][-1]["content"][0]["text"]
    assert "one of the above actions" in payload["messages"][-1]["content"][0]["text"]


def test_native_history_keeps_sixteen_observations_and_sanitizes_previous_actions():
    obs = [observation(f"observation{i}") for i in range(40)]
    payload = b.build_native_request(b.DEFAULT_SETUP, "fixture", "instruction", obs, ["go forward 123!"] * 39)
    texts = [x["content"][0]["text"] for x in payload["messages"]]
    assert len(payload["messages"]) == 33
    assert sum("observation" in x for x in texts) == 16
    assert not any("observation23" in x for x in texts)
    assert all(x["content"][0]["text"] == "go forward " for x in payload["messages"] if x["role"] == "assistant")


def test_image_mode_and_mismatched_history_are_rejected():
    with pytest.raises(b.ContractError, match="history length"):
        b.build_native_request(b.DEFAULT_SETUP, "fixture", "instructions", [observation()], ["go forward"])
    obs = observation()
    obs["image"] = "not-supported-in-native-text-mode"
    with pytest.raises(b.ContractError, match="image mode"):
        b.build_native_request(b.DEFAULT_SETUP, "fixture", "instructions", [obs], [])


def test_run_requires_actual_checkpoint_and_online_wandb(tmp_path):
    with pytest.raises(b.ContractError, match="checkpoint"):
        b.reserve_run(b.DEFAULT_SETUP, tmp_path, {**b.ACTOR, "hf_commit": "d" * 40}, {}, {})
    with pytest.raises(b.ContractError, match="online W&B"):
        b.reserve_run(b.DEFAULT_SETUP, tmp_path, {**b.ACTOR, "served_model_id": "fixture"}, {"mode": "offline"}, {})


def test_export_is_idempotent_and_changed_prompt_collides(tmp_path, manifest):
    contract(tmp_path)
    ident = manifest["episodes"][0]["episode_id"]
    row = b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation()], context())
    assert row == b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation()], context())
    with pytest.raises(b.ContractError, match="collision"):
        b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation("changed")], context())


def test_ingest_retains_raw_response_and_native_action_parser(tmp_path, manifest):
    contract(tmp_path)
    ident = manifest["episodes"][0]["episode_id"]
    row = b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation()], context())
    envelope = response(row, " go forward 12! ")
    parsed = b.ingest_response(b.DEFAULT_SETUP, tmp_path, envelope)
    assert parsed["native_candidate_action"] == "go forward "
    assert parsed["raw_completion"] == " go forward 12! "
    assert parsed["score"] is None
    assert parsed == b.ingest_response(b.DEFAULT_SETUP, tmp_path, envelope)


def test_unknown_response_and_model_mismatch_are_blocked(tmp_path, manifest):
    contract(tmp_path)
    row = b.export_turn(b.DEFAULT_SETUP, tmp_path, manifest["episodes"][0]["episode_id"], "instructions", [observation()], context())
    bad = response(row)
    bad["request_sha256"] = "0" * 64
    with pytest.raises(b.ContractError, match="prompt/model"):
        b.ingest_response(b.DEFAULT_SETUP, tmp_path, bad)
    bad = response(row)
    obj = json.loads(base64.b64decode(bad["raw_body_base64"]))
    obj["model"] = "base-only-substitute"
    bad["raw_body_base64"] = base64.b64encode(b.canonical(obj)).decode()
    assert b.ingest_response(b.DEFAULT_SETUP, tmp_path, bad)["status"] == "BLOCKED_NATIVE_RESPONSE"


def test_empty_response_is_not_a_native_success(tmp_path, manifest):
    contract(tmp_path)
    row = b.export_turn(b.DEFAULT_SETUP, tmp_path, manifest["episodes"][0]["episode_id"], "instructions", [observation()], context())
    assert b.ingest_response(b.DEFAULT_SETUP, tmp_path, response(row, ""))["status"] == "BLOCKED_NATIVE_RESPONSE"


def test_resume_requires_all_prior_responses_and_unchanged_observations(tmp_path, manifest):
    contract(tmp_path)
    ident = manifest["episodes"][0]["episode_id"]
    row = b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation()], context())
    with pytest.raises(FileNotFoundError):
        b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation(), observation("next")], context())
    b.ingest_response(b.DEFAULT_SETUP, tmp_path, response(row))
    with pytest.raises(b.ContractError, match="past observation"):
        b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation("changed"), observation("next")], context())
    next_row = b.export_turn(b.DEFAULT_SETUP, tmp_path, ident, "instructions", [observation(), observation("next")], context())
    assert next_row["step"] == 1


def test_partial_and_duplicate_episodes_never_produce_full_score(tmp_path, manifest):
    contract(tmp_path)
    result = b.score_complete_native(b.DEFAULT_SETUP, tmp_path, [], tmp_path / "partial.json")
    assert result["score"] is None and result["expected_episodes"] == 255
    row = {"episode_id": manifest["episodes"][0]["episode_id"]}
    with pytest.raises(b.ContractError, match="duplicate"):
        b.score_complete_native(b.DEFAULT_SETUP, tmp_path, [row, row], tmp_path / "duplicate.json")


def test_full_inventory_alone_cannot_become_a_score(tmp_path, manifest):
    c = contract(tmp_path)
    rows = [{"episode_id": e["episode_id"], "contract_sha256": b.fingerprint(c), "status": "FAILED"}
            for e in manifest["episodes"]]
    with pytest.raises(b.ContractError, match="failed episode"):
        b.score_complete_native(b.DEFAULT_SETUP, tmp_path, rows, tmp_path / "full.json")


def test_native_summary_is_environment_macro_average_on_synthetic_logs(tmp_path, manifest):
    for episode in manifest["episodes"]:
        path = tmp_path / episode["native_json_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task": episode["task"], "progression": float(episode["environment"] == "nle"),
                                    "num_steps": 1, "input_tokens": 10, "output_tokens": 1}))
    summary = b.native_symbols().collect_and_summarize_results(tmp_path)
    assert summary["average_progress"] == pytest.approx(100 / 6)
    assert sum(e["episodes_played"] for e in summary["environments"].values()) == 255
    assert summary["total_input_tokens"] == 2550
