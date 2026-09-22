#!/usr/bin/env python3
"""Prepare CORE-Bench HARD45 and bind native container executions to receipts.

No model calls, capsule downloads, container starts, or agent executions occur in
this bridge. Actual native evaluator replay is an explicit final `score` command.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile
import types

import public_portfolio_native as durable
from public_portfolio_native import ContractError, canonical, digest, fingerprint, now, read_json, require, write_once

SCHEMA = "core-bench-hard45-native-v1"
REVISION = "e32a2980e72fe6eb04ee04eb749458f570625663"
LEVEL = "codeocean_hard"
EXPECTED_TOTAL = 45
EXPECTED_QUESTIONS = 79
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "outputs/public_portfolio_2026-09-05/core_setup"
FILES = {
    "core_test.json": "b93d7fda4cba27a074c9717d89c7a0ffa5e0e197322cc27bcd1d3d8a5bedf8d4",
    "source/LICENSE": "fa800fc3033ad315e2bf2949afca5c6bcb9f19ad6331d1db996e7dfca1e0eeb8",
    "source/README.md": "8c0f6086d0aad6871b516e441ad28f0428991779446cc2370f91bfe4ecef73ba",
    "source/main.py": "3d205be67b83ffa835380a691dcf32e0b21dfab1f9b5565979249628cb011332",
    "source/benchmark/benchmark.py": "a227b001a3696a9d5a28dbcd238dfcad6a89ea8b563db5d0ea38e5a06db05f06",
    "source/benchmark/benchmark_prompts.json": "e70b63c40671f3924a8d71aac0e1c64a5dae3c8cf0bf391471c488581fff53b8",
    "source/benchmark/evaluations.py": "88a132c1196bfd9f1f8cf32d23e64cb6e0b26d98437ab5ff72f5e77ef56b53bb",
    "source/benchmark/dataset/core_test.json.gpg": "cebf204bc8fd0b2e1b6e65ab762b7edbf906313ff3718780492633aeb4d972f2",
    "source/docker/Dockerfile": "2d352174caa6d1fae762f38bb4aac3741dab83000bd7d2209ccdf17760d80bad",
    "source/requirements.txt": "c2d7db5719bc5966125fde9e6d2aa2fb4e03f7c5de26e8fcc95954e48d948050",
    "source/agents/AutoGPT-CORE/coreagent_hard_gpt4o.sh": "a3e28bb2cadb0b76a1cd9dc6db8571ce2afe39cfdd66935b903248f28d72f600",
}
GRADE_KEYS = {"correct_written_answers", "correct_vision_answers", "total_written_questions", "total_vision_questions"}
ACTOR_BASE = {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0"}
ACTOR_ADAPTER = {"repo_id": "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6",
                 "revision": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"}


def sha_field(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), f"{name}: SHA256 required")


def write_bytes_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"immutable raw artifact collision: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=".raw-artifact-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            require(path.read_bytes() == data, f"immutable raw artifact collision: {path}")
    finally:
        os.unlink(temporary_name)


def verify_sources(setup):
    for name, expected in FILES.items():
        require(digest((setup / name).read_bytes()) == expected, f"pinned CORE source drift: {name}")
    return {"repository": "https://github.com/siegelz/core-bench", "revision": REVISION, "files_sha256": FILES,
            "code_license": "MIT", "test_decryption": "official public password; decrypted bytes SHA256 pinned",
            "capsule_assets": "not acquired by preparation; per-capsule archive and license receipts required before reservation"}


def native_validator(setup):
    tree = ast.parse((setup / "source/benchmark/benchmark.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CodeOceanTask")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__validate_json")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "native_CodeOceanTask_validator", "exec"), namespace)
    return namespace["__validate_json"]


def dataset(setup):
    require(digest((setup / "core_test.json").read_bytes()) == FILES["core_test.json"], "native test split hash mismatch")
    rows = read_json(setup / "core_test.json")
    require(len(rows) == EXPECTED_TOTAL and len({r["capsule_id"] for r in rows}) == EXPECTED_TOTAL, "HARD split must have45unique test capsules")
    validate = native_validator(setup)
    for row in rows:
        validate(None, row)
        require(re.fullmatch(r"capsule-\d+", row["capsule_id"]), "unsafe capsule ID")
        require(len(row["results"]) == 3, "native reference-run count drift")
    require(sum(len(r["results"][0]) for r in rows) == EXPECTED_QUESTIONS, "native question count drift")
    return rows


def task_prompt(setup, row):
    template = read_json(setup / "source/benchmark/benchmark_prompts.json")[LEVEL]
    require("{registry_link}" not in template, "hard prompt changed; registry metadata must be resolved natively")
    # Exact native __setup_task_environment substitutions, including dict_keys repr.
    return template.replace("{task_prompt}", row["task_prompt"]).replace("{json_fields}", str(row["results"][0].keys()))


def build_manifest(setup):
    source = verify_sources(setup)
    rows = dataset(setup)
    tasks = []
    for index, row in enumerate(rows):
        prompt = task_prompt(setup, row)
        tasks.append({"capsule_id": row["capsule_id"], "row_index": index, "source_row_sha256": fingerprint(row),
                      "field": row["field"], "language": row["language"], "capsule_title": row["capsule_title"],
                      "capsule_doi": row["capsule_doi"], "questions": list(row["results"][0]),
                      "task_txt": prompt, "task_txt_sha256": digest(prompt.encode()),
                      "archive_url": f'https://corebench.cs.princeton.edu/capsules/{row["capsule_id"]}.tar.gz',
                      "archive_sha256": None, "uses_gpu": None})
    return {"schema": SCHEMA, "suite_id": "core_bench_eval", "benchmark_level": LEVEL,
            "expected_total": EXPECTED_TOTAL, "expected_questions": EXPECTED_QUESTIONS,
            "split": "all45released test capsules, hard level only; train/easy/medium excluded", "source": source,
            "protocol": {"include_correct_result_paths": False, "no_gpu": False, "task_limit": None,
                         "native_local_timeout_seconds": 8100, "attempts_per_capsule": 1,
                         "hard_removals": ["results/*", "REPRODUCING.md", "environment/", "code/run", "code/run.sh"],
                         "native_report_fallback": "missing or invalid JSON becomes{} after an actual terminal native execution",
                         "metric": "native correct_tasks/total_tasks; all written and vision questions must pass"}, "tasks": tasks}


def validate_manifest(manifest):
    require(manifest.get("schema") == SCHEMA and manifest.get("benchmark_level") == LEVEL, "native HARD level required")
    require(manifest.get("expected_total") == EXPECTED_TOTAL and manifest.get("expected_questions") == EXPECTED_QUESTIONS, "denominator must be45hard capsules/79questions")
    tasks = manifest["tasks"]
    require(len(tasks) == EXPECTED_TOTAL and len({t["capsule_id"] for t in tasks}) == EXPECTED_TOTAL, "incomplete/duplicate capsule manifest")
    for index, task in enumerate(tasks):
        require(task["row_index"] == index and re.fullmatch(r"capsule-\d+", task["capsule_id"]), "invalid capsule order/ID")
        require(digest(task["task_txt"].encode()) == task["task_txt_sha256"], "native prompt hash drift")
        require(not {"results", "answer", "answers", "ground_truth"} & set(task), "gold values leaked to actor manifest")


def prepared_manifest(setup, prepared):
    manifest = read_json(prepared / "manifest.json")
    validate_manifest(manifest)
    require(manifest == build_manifest(setup), "manifest/source/prompt/protocol mismatch")
    return manifest


def prepare(setup, output):
    manifest = build_manifest(setup)
    validate_manifest(manifest)
    write_once(output / "manifest.json", manifest)
    write_once(output / "native_ground_truth.json", dataset(setup))
    write_once(output / "source_receipt.json", manifest["source"])
    return {"status": "PREPARED_NO_CAPSULES_EXECUTED", "expected_total": EXPECTED_TOTAL,
            "expected_questions": EXPECTED_QUESTIONS, "manifest_sha256": fingerprint(manifest), "score": None}


def create_contract(manifest, model, wandb, agent):
    validate_manifest(manifest)
    require(model.get("base_model") == ACTOR_BASE and model.get("adapter") == ACTOR_ADAPTER, "exact portfolio actor HF pins required")
    require(model.get("adapter_loaded") is True and model.get("vision") is True and model.get("served_model"), "verified adapter and vision actor runtime required")
    sha_field(model.get("deployment_receipt_sha256"), "actor deployment receipt")
    durable.validate_wandb(wandb)
    files = agent.get("files_sha256", {})
    require(files and agent.get("entrypoint") in files, "agent bundle and entrypoint hashes required")
    for name, sha in files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe agent file path")
        sha_field(sha, "agent file")
    require(agent.get("tool_interfaces") and agent.get("sampling") and isinstance(agent.get("max_model_calls"), int)
            and agent["max_model_calls"] > 0, "declared agent tools, sampling and call budget required")
    require(agent.get("timeout_seconds") == 8100, "preserve native hard agent timeout8100seconds")
    require(agent.get("all_model_calls_use_same_actor") is True, "undeclared planner/vision/helper models are forbidden")
    require(agent.get("gold_visible_to_agent") is False, "gold must remain outside agent container")
    return {"schema": SCHEMA, "manifest_sha256": fingerprint(manifest), "benchmark_level": LEVEL,
            "model": model, "wandb": wandb, "agent": agent,
            "runner_sha256": digest(Path(__file__).read_bytes()), "durability_helper_sha256": digest(Path(durable.__file__).read_bytes())}


def inspect_hard_environment(environment, task):
    capsule = environment / task["capsule_id"]
    require(capsule.is_dir() and (capsule / "results").is_dir(), "native hard capsule/results directory required")
    require(not any((capsule / "results").iterdir()), "hard capsule must begin with empty results")
    for relative in ("REPRODUCING.md", "environment", "code/run", "code/run.sh"):
        require(not (capsule / relative).exists() and not (capsule / relative).is_symlink(), f"native hard removal not applied: {relative}")
    require(not (environment / "correct_result_paths.txt").exists(), "correct result path hints forbidden")
    require(digest((environment / "task.txt").read_bytes()) == task["task_txt_sha256"], "task.txt differs from native hard prompt")
    files, symlinks = {}, {}
    for path in sorted(environment.rglob("*")):
        if path.is_symlink():
            require(path.resolve().is_relative_to(environment.resolve()), "initial environment symlink escapes capsule environment")
            relative = str(path.relative_to(environment))
            symlinks[relative] = os.readlink(path)
            files[relative] = digest(b"SYMLINK\0" + os.readlink(path).encode())
            continue
        if not path.is_file():
            continue
        require(path.name not in {"core_test.json", "native_ground_truth.json", "reference_rows.json"}, "gold dataset exposed in agent environment")
        files[str(path.relative_to(environment))] = digest(path.read_bytes())
    return {"files_sha256": files, "symlinks": symlinks, "manifest_sha256": fingerprint(files), "hard_removals_verified": True,
            "task_txt_sha256": task["task_txt_sha256"]}


def reserve_task(setup, prepared, run_dir, capsule_id, model, wandb, agent, asset, runtime, environment):
    manifest = prepared_manifest(setup, prepared)
    task = next((t for t in manifest["tasks"] if t["capsule_id"] == capsule_id), None)
    require(task is not None, "capsule outside pinned HARD45test split")
    contract = create_contract(manifest, model, wandb, agent)
    write_once(run_dir / "run_contract.json", contract)
    require(asset.get("capsule_id") == capsule_id and asset.get("archive_url") == task["archive_url"], "capsule asset/source mismatch")
    for key in ("archive_sha256", "extracted_tree_sha256", "license_receipt_sha256"):
        sha_field(asset.get(key), key)
    require(isinstance(asset.get("uses_gpu"), bool), "GPU need must be inspected from real capsule REPRODUCING.md")
    require(isinstance(asset.get("registry_link"), str) and asset["registry_link"].startswith("registry.codeocean.com/published/"), "native capsule registry metadata unresolved")
    require(runtime.get("platform") in {"local", "azure"}, "native local/Azure container platform required")
    require(runtime.get("source_revision") == REVISION and runtime.get("benchmark_level") == LEVEL, "native harness revision/level mismatch")
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", runtime.get("image_digest", "")), "resolved immutable execution image digest required")
    require(runtime.get("gold_dataset_not_mounted") is True, "gold dataset must stay outside execution container")
    require(runtime.get("gpu_available") is True or asset["uses_gpu"] is False, "required capsule GPU missing")
    initial = inspect_hard_environment(environment, task)
    submission = {"capsule_id": capsule_id, "contract_sha256": fingerprint(contract), "task": task,
                  "asset_receipt": asset, "runtime_receipt": runtime, "initial_environment": initial}
    directory = run_dir / "tasks" / capsule_id
    if (directory / "submission.json").exists():
        require(read_json(directory / "submission.json") == submission, "resume model/prompt/asset/environment collision")
        require((directory / "terminal.json").exists(), "existing unresolved attempt; recover original execution, never launch another")
        return {"status": "ALREADY_TERMINAL_NO_REEXECUTION", "capsule_id": capsule_id, "score": None}
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with (directory / "attempt.claim").open("x") as claim:
            claim.write(now())
            claim.flush()
            os.fsync(claim.fileno())
    except FileExistsError as exc:
        raise ContractError("concurrent/unresolved capsule claim; no reexecution") from exc
    write_once(directory / "submission.json", submission)
    return {"status": "RESERVED_NOT_EXECUTED", "capsule_id": capsule_id, "submission_sha256": fingerprint(submission), "score": None}


def ingest_native(setup, prepared, run_dir, capsule_id, native_result_path, execution, artifacts):
    manifest = prepared_manifest(setup, prepared)
    contract = read_json(run_dir / "run_contract.json")
    require(contract == create_contract(manifest, contract["model"], contract["wandb"], contract["agent"]), "run contract drift")
    directory = run_dir / "tasks" / capsule_id
    submission = read_json(directory / "submission.json")
    require(execution.get("capsule_id") == capsule_id and execution.get("submission_sha256") == fingerprint(submission), "execution/submission mismatch")
    require(execution.get("benchmark_level") == LEVEL and execution.get("source_revision") == REVISION, "native hard source receipt missing")
    require(execution.get("terminal") is True and execution.get("outcome") in {"completed", "timeout", "agent_failed"}, "actual terminal native execution required")
    require(execution.get("started_at") and execution.get("finished_at"), "actual execution timestamps required")
    require(execution.get("model_provenance_sha256") == fingerprint(contract["model"]), "execution actor model mismatch")
    raw = native_result_path.read_bytes()
    require(execution.get("native_results_sha256") == digest(raw), "native result file hash mismatch")
    obj = json.loads(raw)
    results = obj.get("capsule_results", [])
    matching = [r for r in results if r.get("capsule_id") == capsule_id]
    require(len(matching) == 1, "native result must identify exactly one matching capsule")
    result = matching[0]
    require(GRADE_KEYS <= set(result), "native evaluator counts missing")
    require(isinstance(result.get("result_report"), dict), "native report must use native missing/invalid fallback{}")
    task = submission["task"]
    require(all(result.get(key) == task[key] for key in ("field", "language", "capsule_title")), "native result capsule metadata mismatch")
    required = {"agent_trace.log", "raw_model_requests.jsonl", "raw_model_responses.jsonl", "container.log"}
    expected_hashes = execution.get("artifacts_sha256", {})
    require(required <= set(expected_hashes), "raw agent/model/container artifacts required")
    for name, expected in expected_hashes.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe artifact path")
        sha_field(expected, "artifact hash")
        data = (artifacts / name).read_bytes()
        require(digest(data) == expected, f"execution artifact drift: {name}")
        write_bytes_once(directory / "artifacts" / name, data)
    report_status = execution.get("report_status")
    if report_status == "present":
        require("report.json" in expected_hashes, "report artifact missing")
        # Native evaluation mutates percentage/numeric values; compare after native replay at final scoring.
        report = read_json(artifacts / "report.json")
        require(isinstance(report, dict), "native report JSON must be an object")
    else:
        require(report_status == "missing_or_invalid" and result["result_report"] == {}, "native missing report fallback mismatch")
        report = {}
    terminal = {"capsule_id": capsule_id, "submission_sha256": fingerprint(submission), "native_result": result,
                "original_report": report, "execution_receipt": execution, "native_results_sha256": digest(raw)}
    write_bytes_once(directory / "native_result_file.json", raw)
    write_once(directory / "terminal.json", terminal)
    return {"status": "NATIVE_RECEIPT_INGESTED_NOT_AGGREGATED", "capsule_id": capsule_id, "score": None}


def load_native_evaluator(setup):
    import numpy as np
    from scipy.stats import t
    require(importlib.metadata.version("numpy") == "1.26.4" and importlib.metadata.version("scipy") == "1.13.1", "use pinned native numpy1.26.4/scipy1.13.1 environment")
    path = setup / "source/benchmark/evaluations.py"
    require(digest(path.read_bytes()) == FILES["source/benchmark/evaluations.py"], "native evaluator code drift")
    tree = ast.parse(path.read_text())
    names = {"eval_result_file", "eval_result_json", "score_results"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "native scoring definitions missing")
    # Exact native functions; no OpenAI import/client or optional LLM summaries.
    namespace = {"json": json, "np": np, "math": math, "t": t, "Dict": dict, "os": os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def score(setup, prepared, run_dir, output):
    manifest = prepared_manifest(setup, prepared)
    contract = read_json(run_dir / "run_contract.json")
    require(contract == create_contract(manifest, contract["model"], contract["wandb"], contract["agent"]), "contract drift")
    terminals, missing = [], []
    for task in manifest["tasks"]:
        directory = run_dir / "tasks" / task["capsule_id"]
        if not (directory / "terminal.json").exists():
            missing.append(task["capsule_id"])
            continue
        terminal = read_json(directory / "terminal.json")
        submission = read_json(directory / "submission.json")
        require(terminal["submission_sha256"] == fingerprint(submission) and submission["contract_sha256"] == fingerprint(contract), "terminal/submission contract drift")
        require(terminal["capsule_id"] == task["capsule_id"], "terminal capsule ID drift")
        require(digest((directory / "native_result_file.json").read_bytes()) == terminal["native_results_sha256"], "retained native result artifact drift")
        for name, expected in terminal["execution_receipt"]["artifacts_sha256"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe retained artifact path")
            require(digest((directory / "artifacts" / name).read_bytes()) == expected, "retained execution artifact drift")
        terminals.append(terminal)
    receipt = {"schema": SCHEMA, "suite_id": "core_bench_eval", "experiment": "E2 replacement", "benchmark_level": LEVEL,
               "expected_total": EXPECTED_TOTAL, "expected_questions": EXPECTED_QUESTIONS, "native_terminal_capsules": len(terminals),
               "missing_capsules": missing, "score": None, "native_summary": None,
               "status": "INCOMPLETE_HARD45", "manifest_sha256": fingerprint(manifest), "run_contract_sha256": fingerprint(contract),
               "claim_boundary": "CORE-Bench public HARD45 only; no easy/medium/train aggregate or original E2 claim."}
    if not missing:
        native = load_native_evaluator(setup)
        native_rows = [{**copy.deepcopy(t["native_result"]), "result_report": copy.deepcopy(t["original_report"])} for t in terminals]
        staging = run_dir / "native_replay"
        staging.mkdir(parents=True, exist_ok=True)
        source_file = staging / "input.json"
        write_once(source_file, {"capsule_results": native_rows})
        scored_file = staging / "scored.json"
        # Always recompute with native code; an existing scored artifact is not trusted as a cache.
        fd, temporary_name = tempfile.mkstemp(prefix="native-replay-", suffix=".json", dir=staging)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(source_file.read_bytes())
            native.eval_result_file(str(temporary), str(setup / "core_test.json"), llm_summary=False, verbose=False)
            scored = read_json(temporary)
            write_once(scored_file, scored)
        finally:
            temporary.unlink(missing_ok=True)
        require(scored["summary"]["total_tasks"] == EXPECTED_TOTAL and scored["summary"]["total_questions"] == EXPECTED_QUESTIONS, "native denominator drift")
        for saved, replay in zip(terminals, scored["capsule_results"]):
            require(saved["capsule_id"] == replay["capsule_id"], "native replay capsule order drift")
            require(all(saved["native_result"][key] == replay[key] for key in GRADE_KEYS), "ingested native counts disagree with exact evaluator replay")
        receipt.update({"status": "COMPLETE_NATIVE_HARD45", "score": scored["summary"]["correct_tasks"] / EXPECTED_TOTAL,
                        "native_summary": scored["summary"], "native_scored_artifact_sha256": digest(scored_file.read_bytes()),
                        "native_evaluator_sha256": FILES["source/benchmark/evaluations.py"]})
    write_once(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    reserve = sub.add_parser("reserve")
    for key in ("prepared", "run-dir", "model", "wandb", "agent", "asset", "runtime", "initial-environment"):
        reserve.add_argument(f"--{key}", type=Path, required=True)
    reserve.add_argument("--capsule-id", required=True)
    ingest = sub.add_parser("ingest-native")
    for key in ("prepared", "run-dir", "native-results", "execution-receipt", "artifacts"):
        ingest.add_argument(f"--{key}", type=Path, required=True)
    ingest.add_argument("--capsule-id", required=True)
    scorer = sub.add_parser("score")
    for key in ("prepared", "run-dir", "output"):
        scorer.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.setup, args.output)
        elif args.command == "reserve":
            result = reserve_task(args.setup, args.prepared, args.run_dir, args.capsule_id, read_json(args.model), read_json(args.wandb),
                                  read_json(args.agent), read_json(args.asset), read_json(args.runtime), args.initial_environment)
        elif args.command == "ingest-native":
            result = ingest_native(args.setup, args.prepared, args.run_dir, args.capsule_id, args.native_results,
                                   read_json(args.execution_receipt), args.artifacts)
        else:
            result = score(args.setup, args.prepared, args.run_dir, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "score": None, "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
