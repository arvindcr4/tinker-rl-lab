#!/usr/bin/env python3
"""Pinned BALROG protocol, passive native request bridge and complete-suite receipt gate.

This module does not instantiate games, start model clients, or send network requests.
The transport/runtime owner must retain the native game state and every actual attempt.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, defaultdict, deque, namedtuple
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETUP = ROOT / "outputs/public_portfolio_2026-09-05/balrog_setup"
REVISION = "b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5"
TREE_SHA256 = "6adab1d9375abb4f37dec81883878d9a0d3b0a66c4b526dafe1437567a1ade1b"
CONFIG_SHA256 = "f6af7a94e782a3d9ef88cec2f5b5f2b6ff8ca9e3e1ca5701161f43611a11705d"
COUNTS = {"babyai": (5, 10), "babaisai": (40, 3), "textworld": (3, 10),
          "crafter": (1, 10), "nle": (1, 5), "minihack": (8, 5)}
DEPENDENCY_REVISIONS = {
    "BartekCupial/Minigrid": "cf73dd148e51276bd675a37e3bfb0bf2b42329b2",
    "nacloos/baba-is-ai": "33c2ca184a6b41bba03a910574b7111c07792003",
    "balrog-ai/minihack": "3ecb6da4eadcac7a4dbc5f8e801ccf636e34d7ec",
    "balrog-ai/TextWorld": "1d56f4765e7b28f04a68eff0bce5a6fe6ecd1a9f",
}
ACTOR = {
    "model_id": "Qwen/Qwen3.6-35B-A3B",
    "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "hf_repo": "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6",
    "hf_commit": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8",
}


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(canonical(value))


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    data = canonical(value)
    if path.exists():
        require(path.read_bytes() == data, f"immutable collision: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)


def verify_sources(setup):
    setup = Path(setup)
    require(digest((setup / "source_tree.json").read_bytes()) == TREE_SHA256, "BALROG source tree drift")
    tree = read(setup / "source_tree.json")
    require(tree["sha"] == REVISION and not tree["truncated"], "incomplete source tree")
    selected = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"
                and x["path"] != "SECRETS" and not x["path"].startswith("docs/imgs/")}
    hashes = {}
    for rel, record in selected.items():
        path = setup / "source" / rel
        require(path.is_file() and not path.is_symlink(), f"missing source: {rel}")
        data = path.read_bytes()
        require(len(data) == record["size"] and
                hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == record["sha"], f"source drift: {rel}")
        hashes[rel] = digest(data)
    require(hashes["balrog/config/config.yaml"] == CONFIG_SHA256, "native config drift")
    dependencies = read(setup / "dependency_sources/resolved_revisions.json")
    require({r["repository"]: r["revision"] for r in dependencies} == DEPENDENCY_REVISIONS,
            "resolved dependency refs drift")
    dep_hashes = {}
    for file in read(setup / "dependency_sources/file_receipts.json"):
        require(file["revision"] == DEPENDENCY_REVISIONS[file["repository"]], "dependency revision drift")
        rel = file["repository"].replace("/", "--") + "/source/" + file["path"]
        data = (setup / "dependency_sources" / rel).read_bytes()
        require(digest(data) == file["sha256"] and len(data) == file["size"], "dependency source drift")
        dep_hashes[rel] = digest(data)
    return {"repository": "https://github.com/balrog-ai/BALROG", "revision": REVISION,
            "tree_sha256": TREE_SHA256, "files_sha256": hashes, "license": "MIT",
            "prospective_dependency_revisions": DEPENDENCY_REVISIONS, "dependency_files_sha256": dep_hashes,
            "dependency_boundary": "Current default refs resolved from native unpinned requirements; not a historical dependency lock or built runtime proof."}


def step_limit(env, task):
    if env == "babyai":
        return {"goto": 64, "pickup": 64, "open": 128, "putnext": 128, "pick_up_seq_go_to": 128}[task.rsplit("/", 1)[-1]]
    return {"babaisai": 100, "nle": 100000, "minihack": 100, "crafter": 2000, "textworld": 80}[env]


def prepare(setup=DEFAULT_SETUP):
    setup = Path(setup)
    source = verify_sources(setup)
    config = yaml.safe_load((setup / "source/balrog/config/config.yaml").read_text())
    require(config["agent"]["type"] == "naive" and config["agent"]["max_image_history"] == 0, "default agent drift")
    require(config["eval"]["max_steps_per_episode"] is None and config["envs"]["env_kwargs"]["seed"] is None,
            "native default caps/seeds changed")
    episodes, environments = [], {}
    for env in config["envs"]["names"].split("-"):
        tasks = config["tasks"][env + "_tasks"]
        repetitions = config["eval"]["num_episodes"][env]
        require((len(tasks), repetitions) == COUNTS[env], f"native denominator drift: {env}")
        upper = 0
        for task in tasks:
            cap = step_limit(env, task)
            upper += cap * repetitions
            for episode_index in range(repetitions):
                episodes.append({"episode_id": f"{env}/{task}/episode-{episode_index:02d}", "environment": env,
                    "task": task, "episode_index": episode_index, "max_agent_steps": cap,
                    "seed": None, "seed_status": "native time/PID-derived seed must be captured before first model call",
                    "native_json_relative_path": f"{env}/{task}/{task}_run_{episode_index:02d}.json"})
        environments[env] = {"task_configurations": len(tasks), "episodes_per_task": repetitions,
                             "episodes": len(tasks) * repetitions, "max_agent_steps": upper}
    require(len(episodes) == 255 and sum(v["task_configurations"] for v in environments.values()) == 58,
            "full native denominator mismatch")
    total = sum(x["max_agent_steps"] for x in episodes)
    require(total == 543520, "full step ceiling drift")
    manifest = {"schema": "public-balrog-native-v1", "suite_id": "balrog_eval", "source": source,
        "native_config": config, "environments": environments, "episodes": episodes,
        "actor": ACTOR, "status": "PROTOCOL_PREPARED_RUNTIME_AND_ASSETS_UNPROVISIONED",
        "game_calls": 0, "model_calls": 0, "score": None,
        "request_limits": {"logical_generations": total, "generations_per_agent_step": 1,
            "outer_attempts_per_generation": config["client"]["max_retries"],
            "sdk_create_invocations": total * config["client"]["max_retries"],
            "http_attempts": None, "http_limit_reason": "OpenAI SDK version/default retries are not pinned by BALROG; freeze and inspect before launch.",
            "conditional_http_attempts_if_sdk_retries_2": total * 5 * 3,
            "per_response_max_tokens": 8192, "logical_step_completion_token_ceiling": total * 8192,
            "input_token_ceiling": None, "cpu_wall_time_ceiling": None,
            "cpu_limit_reason": "BabyAI construction/mission rejection loops are unbounded; native environment creation occurs before the evaluator's finite step loop."},
        "metric": {"native_entrypoint": "balrog.utils.collect_and_summarize_results",
            "overall": "100 * arithmetic mean of six per-environment means of episode progression",
            "missing_native_progression_behavior": "native summary defaults missing progression to zero; evidence gate rejects missing progression instead",
            "completion_requirement": "exactly all 255 declared episodes, native end logs plus CSV and model request receipts; no failed/missing/duplicate episode"},
        "runtime_requirements": {"platform": "Linux x86_64 CPU workers; upstream Docker uses Ubuntu22.04/Python3.10",
            "gpu_for_games_required": False, "model_endpoint": "separately provisioned exact actor using native vllm client path",
            "worker_count": 16, "native_client_name_for_local_endpoint": "vllm",
            "client_warning": "native client_name=openai ignores base_url and uses standard OpenAI API; do not use it for the local actor",
            "seed_warning": "logged seed alone is insufficient: make_env precedes reset seeding; capture worker RNG, initial state and actual TextWorld game path/hash",
            "required_assets": ["Boxoban levels (native unpinned master.zip)", "TextWorld tw_games (native mutable Google Drive archive)"],
            "freeze_needed": ["all Python wheel/source versions and hashes", "Linux image digest/system libraries", "level archives and individual game-file hashes",
                              "worker/process/task scheduling and RNG/realized state evidence", "actor runtime provenance and online W&B before first request"]}}
    write_once(setup / "manifest.json", manifest)
    return manifest


def native_symbols(setup=DEFAULT_SETUP):
    """Compile exact inspected native definitions without importing games or SDK clients."""
    setup = Path(setup)
    verify_sources(setup)
    env = {"__builtins__": __builtins__, "deque": deque, "List": List, "Optional": Optional,
           "copy": copy, "re": re, "namedtuple": namedtuple, "json": json, "math": math,
           "os": os, "defaultdict": defaultdict, "Path": Path}
    selections = {
        "balrog/prompt_builder/history.py": {"Message", "HistoryPromptBuilder"},
        "balrog/agents/base.py": {"BaseAgent"},
        "balrog/agents/naive.py": {"NaiveAgent"},
        "balrog/client.py": {"LLMResponse", "LLMClientWrapper", "OpenAIWrapper"},
        "balrog/utils.py": {"collect_and_summarize_results"},
    }
    for rel, names in selections.items():
        tree = ast.parse((setup / "source" / rel).read_text())
        selected = [n for n in tree.body if getattr(n, "name", None) in names or
                    isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        require(len(selected) == len(names), f"native AST definitions missing: {rel}")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(setup / "source" / rel), "exec"), env)
    return SimpleNamespace(**env)


class _Captured(BaseException):
    pass


def build_native_request(setup, served_model, instruction_prompt, observations, previous_completions):
    """Replay saved text observations/completions through unmodified native history + naïve agent.

    Observations must be actual wrapper outputs, with image=None for default text mode.
    No game or model is called. No shortening of context, cap, or native action parsing.
    """
    require(len(observations) == len(previous_completions) + 1, "observation/response history length mismatch")
    native = native_symbols(setup)
    payloads = []
    index = 0
    wrapper = object.__new__(native.OpenAIWrapper)
    wrapper.alternate_roles = False

    class ReplayClient:
        def generate(self, messages):
            nonlocal index
            payload = {"messages": wrapper.convert_messages(messages), "model": served_model,
                       "max_tokens": 8192, "temperature": 1.0}
            payloads.append(payload)
            if index == len(previous_completions):
                raise _Captured()
            content = previous_completions[index]
            index += 1
            require(isinstance(content, str) and bool(content), "previous native response must contain content")
            return native.LLMResponse(served_model, content.strip(), "stop", 0, 0, None)

    builder = native.HistoryPromptBuilder(max_text_history=16, max_image_history=0, max_cot_history=1)
    builder.update_instruction_prompt(instruction_prompt)
    agent = native.NaiveAgent(lambda: ReplayClient(), builder)
    prev_action = None
    for observation in observations:
        require(observation.get("image") is None, "default text bridge does not accept image mode")
        try:
            response = agent.act(copy.deepcopy(observation), prev_action=prev_action)
            prev_action = response.completion
        except _Captured:
            return payloads[-1]
    raise ContractError("native request was not captured")


def validate_actor(actor):
    require(all(actor.get(key) == value for key, value in ACTOR.items()), "exact actor checkpoint mismatch")
    require(isinstance(actor.get("served_model_id"), str) and actor["served_model_id"], "served model ID required")


def reserve_run(setup, run_dir, actor, wandb, runtime):
    manifest = prepare(setup)
    validate_actor(actor)
    require(wandb.get("mode") == "online" and wandb.get("initialized_before_model_work") is True
            and wandb.get("run_id") and wandb.get("url", "").startswith("https://wandb.ai/"), "online W&B receipt required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", runtime.get("cpu_image_digest", "")) is not None,
            "runtime freeze missing immutable cpu_image_digest")
    for key in ("packages_manifest_sha256", "assets_manifest_sha256", "actor_provenance_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", runtime.get(key, "")) is not None, f"runtime freeze missing {key}")
    require(runtime.get("dependency_revisions") == DEPENDENCY_REVISIONS, "runtime dependency pins differ")
    require(isinstance(runtime.get("sdk_retries"), int) and runtime["sdk_retries"] >= 0, "SDK retry count must be captured")
    contract = {"schema": "public-balrog-runtime-contract-v1", "manifest_sha256": fingerprint(manifest),
                "runner_sha256": digest(Path(__file__).read_bytes()), "actor": actor, "wandb": wandb, "runtime": runtime}
    write_once(Path(run_dir) / "contract.json", contract)
    return contract


def load_contract(setup, run_dir):
    contract = read(Path(run_dir) / "contract.json")
    require(contract == reserve_run(setup, run_dir, contract["actor"], contract["wandb"], contract["runtime"]), "run contract drift")
    return contract


def export_turn(setup, run_dir, episode_id, instruction_prompt, observations, context):
    """Reserve exactly one logical turn; retries are separate runtime attempts, never new turns."""
    contract = load_contract(setup, run_dir)
    manifest = read(Path(setup) / "manifest.json")
    tasks = {x["episode_id"]: x for x in manifest["episodes"]}
    require(episode_id in tasks, "episode outside full native manifest")
    step = len(observations) - 1
    require(0 <= step < tasks[episode_id]["max_agent_steps"], "native episode step cap exceeded")
    require(isinstance(context.get("seed"), int) and 0 <= context["seed"] < 2**32, "realized native seed missing")
    require(context.get("native_env_config_sha256") == fingerprint(manifest["native_config"]["envs"]),
            "episode environment config differs from native defaults")
    require(context.get("env_max_steps") == tasks[episode_id]["max_agent_steps"], "realized native step limit differs")
    for field in ("initial_state_sha256", "worker_rng_state_sha256", "game_assets_sha256"):
        require(re.fullmatch("[0-9a-f]{64}", context.get(field, "")) is not None, f"episode evidence missing {field}")
    episode_dir = Path(run_dir) / "episodes" / digest(episode_id.encode())
    write_once(episode_dir / "context.json", {"episode_id": episode_id, "instruction_prompt": instruction_prompt, **context})
    previous = []
    for index in range(step):
        previous_request = read(episode_dir / f"turn-{index:06d}/request.json")
        require(previous_request["observations_sha256"] == fingerprint(observations[:index + 1]), "past observation history changed")
        receipt = read(episode_dir / f"turn-{index:06d}/response.json")
        require(receipt["status"] == "NATIVE_RESPONSE_CAPTURED", "previous turn lacks a successful native response")
        previous.append(receipt["raw_completion"])
    payload = build_native_request(setup, contract["actor"]["served_model_id"], instruction_prompt, observations, previous)
    row = {"task_id": episode_id, "episode_id": episode_id, "step": step, "api_path": "/v1/chat/completions",
           "batch_id": f"balrog-{digest(episode_id.encode())[:16]}-{step:06d}", "contract_sha256": fingerprint(contract),
           "request_sha256": fingerprint(payload), "observations_sha256": fingerprint(observations),
           "payload": payload, "native_outer_attempt_limit": 5,
           "resume_rule": "Reuse this claim; do not regenerate completed or uncertain turns. Native retry attempts require separate raw attempt evidence."}
    write_once(episode_dir / f"turn-{step:06d}/request.json", row)
    return row


def ingest_response(setup, run_dir, envelope):
    """Persist an already returned successful HTTP response; makes no request/retry."""
    contract = load_contract(setup, run_dir)
    episode_id, step = envelope["task_id"], envelope["step"]
    directory = Path(run_dir) / "episodes" / digest(episode_id.encode()) / f"turn-{step:06d}"
    request = read(directory / "request.json")
    require(envelope["contract_sha256"] == fingerprint(contract) == request["contract_sha256"], "response contract mismatch")
    require(envelope["request_sha256"] == request["request_sha256"], "response prompt/model mismatch")
    require(envelope.get("http_status") == 200, "unsuccessful HTTP attempts must be retained by runtime, not promoted")
    require(envelope.get("started_at") and envelope.get("received_at"), "raw response timestamps required")
    raw = base64.b64decode(envelope["raw_body_base64"], validate=True)
    obj = json.loads(raw)
    write_once(directory / "raw_envelope.json", envelope)
    try:
        require(obj.get("model") == contract["actor"]["served_model_id"], "returned model mismatch")
        require(len(obj["choices"]) == 1, "one native choice required")
        content = obj["choices"][0]["message"]["content"]
        require(isinstance(content, str) and bool(content), "empty native response")
        usage = obj["usage"]
        native = native_symbols(setup)
        value = native.LLMResponse(obj["model"], content.strip(), obj["choices"][0]["finish_reason"],
                                   usage["prompt_tokens"], usage["completion_tokens"], None)
        action = native.NaiveAgent._extract_final_answer(None, value).completion
        result = {"status": "NATIVE_RESPONSE_CAPTURED", "raw_completion": content, "native_candidate_action": action,
                  "native_usage": usage, "request_sha256": request["request_sha256"], "raw_response_sha256": digest(raw),
                  "score": None}
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        result = {"status": "BLOCKED_NATIVE_RESPONSE", "reason": str(exc), "raw_response_sha256": digest(raw), "score": None}
    write_once(directory / "response.json", result)
    return result


def score_complete_native(setup, run_dir, episode_evidence, output):
    """Gate source-native aggregation on all 255 complete immutable episode receipts."""
    manifest = prepare(setup)
    contract = load_contract(setup, run_dir)
    expected = {x["episode_id"]: x for x in manifest["episodes"]}
    ids = [x["episode_id"] for x in episode_evidence]
    require(len(ids) == len(set(ids)), "duplicate episode evidence")
    require(set(ids).issubset(expected), "extra episode evidence")
    if set(ids) != set(expected):
        result = {"status": "INCOMPLETE_NATIVE_EPISODES", "score": None, "completed_receipts": len(ids),
                  "expected_episodes": 255, "missing_episodes": sorted(set(expected) - set(ids))}
        write_once(output, result)
        return result
    logs = []
    for evidence in episode_evidence:
        task = expected[evidence["episode_id"]]
        require(evidence.get("contract_sha256") == fingerprint(contract), "episode contract mismatch")
        require(evidence.get("status") == "NATIVE_EPISODE_COMPLETE", "failed episode cannot be scored")
        for field in ("native_json", "trajectory_csv", "attempts_manifest", "initial_state_receipt"):
            artifact = evidence[field]
            data = Path(artifact["path"]).read_bytes()
            require(digest(data) == artifact["sha256"], f"episode artifact drift: {field}")
        log = read(evidence["native_json"]["path"])
        require(log["task"] == task["task"] and 0 < log["num_steps"] <= task["max_agent_steps"], "episode task/step mismatch")
        require(isinstance(log["seed"], int) and 0 <= log["seed"] < 2**32, "missing native seed")
        require(isinstance(log.get("progression"), (int, float)) and math.isfinite(log["progression"]), "missing native progression")
        require(log["agent"] == manifest["native_config"]["agent"], "native agent config mismatch")
        require(log["client"]["model_id"] == contract["actor"]["served_model_id"] and log["client"]["client_name"] == "vllm",
                "episode actor transport mismatch")
        require(log["client"]["generate_kwargs"] == manifest["native_config"]["client"]["generate_kwargs"], "sampling protocol mismatch")
        require(log["client"]["max_retries"] == 5 and log["client"]["alternate_roles"] is False, "native client protocol mismatch")
        require(log["num_steps"] == task["max_agent_steps"] or log.get("done") is True, "early nonterminal episode")
        episode_dir = Path(run_dir) / "episodes" / digest(evidence["episode_id"].encode())
        context = read(episode_dir / "context.json")
        initial = read(evidence["initial_state_receipt"]["path"])
        require(initial == context and context["seed"] == log["seed"], "initial state/seed mismatch")
        attempts = read(evidence["attempts_manifest"]["path"])
        require(attempts["episode_id"] == evidence["episode_id"] and attempts["contract_sha256"] == fingerprint(contract),
                "attempt ledger episode/contract mismatch")
        require([t["step"] for t in attempts["turns"]] == list(range(log["num_steps"])), "incomplete request ledger")
        totals = Counter()
        for turn in attempts["turns"]:
            turn_dir = episode_dir / f"turn-{turn['step']:06d}"
            request, response = read(turn_dir / "request.json"), read(turn_dir / "response.json")
            require(request["contract_sha256"] == fingerprint(contract) and turn["request_sha256"] == request["request_sha256"],
                    "attempt request mismatch")
            require(response["status"] == "NATIVE_RESPONSE_CAPTURED" and turn["successful_response_sha256"] == response["raw_response_sha256"],
                    "attempt successful-response mismatch")
            raw_envelope = read(turn_dir / "raw_envelope.json")
            require(digest(base64.b64decode(raw_envelope["raw_body_base64"], validate=True)) == response["raw_response_sha256"],
                    "raw response evidence drift")
            outer = turn["outer_attempts"]
            require(1 <= len(outer) <= 5 and [x["outer_attempt"] for x in outer] == list(range(1, len(outer) + 1)),
                    "outer retry ledger exceeds native protocol")
            for attempt in outer:
                sdk = attempt["sdk_attempts"]
                require(1 <= len(sdk) <= contract["runtime"]["sdk_retries"] + 1, "SDK retry ledger exceeds runtime freeze")
                for event in sdk:
                    require(event["request_sha256"] == request["request_sha256"] and event.get("started_at") and event.get("received_at"),
                            "underlying HTTP attempt evidence missing")
                    require(event["outcome"] in ("response", "transport_error"), "attempt outcome missing")
                    if event["outcome"] == "response":
                        raw = event["raw_response"]
                        require(digest(Path(raw["path"]).read_bytes()) == raw["sha256"], "underlying HTTP response hash mismatch")
            require(outer[-1]["sdk_attempts"][-1].get("raw_response", {}).get("sha256") == response["raw_response_sha256"],
                    "last attempt is not the captured successful response")
            totals.update(response["native_usage"])
        require(totals["prompt_tokens"] == log["input_tokens"] and totals["completion_tokens"] == log["output_tokens"],
                "native token totals do not match captured turns")
        logs.append((task, log))
    native = native_symbols(setup)
    with tempfile.TemporaryDirectory(prefix="balrog-native-summary-") as temporary:
        for task, log in logs:
            path = Path(temporary) / task["native_json_relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical(log))
        summary = native.collect_and_summarize_results(temporary)
    result = {"status": "FULL_NATIVE_SUITE_AGGREGATED", "score": summary["average_progress"],
              "native_summary": summary, "episodes": 255, "task_configurations": 58,
              "contract_sha256": fingerprint(contract), "episode_evidence_sha256": fingerprint(episode_evidence)}
    write_once(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    args = parser.parse_args()
    manifest = prepare(args.setup)
    print(json.dumps({"status": manifest["status"], "task_configurations": 58, "episodes": 255,
                      "logical_generation_ceiling": manifest["request_limits"]["logical_generations"],
                      "manifest_sha256": fingerprint(manifest), "score": None}, indent=2))


if __name__ == "__main__":
    main()
