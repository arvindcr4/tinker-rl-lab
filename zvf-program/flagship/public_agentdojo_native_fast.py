#!/usr/bin/env python3
"""Pinned AgentDojo v1.2.2 DEFAULT: 97 benign utility episodes.

fetch-sources, prepare, overlap, and summarize never invoke a model. The run
command requires an immutable actor identity and a live online W&B run before
constructing the native OpenAI-compatible pipeline. Native task, environment,
tool loop, trace logger and utility evaluator remain unchanged. This module
does not evaluate prompt-injection security or claim held-out performance.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request
import urllib.parse
import uuid
from datetime import datetime, timezone

REVISION = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
REPOSITORY = "https://github.com/ethz-spylab/agentdojo"
VERSION = "v1.2.2"
SOURCE_MANIFEST_SHA256 = "e0032399790567f9c19314fd304bd9e9034f48a1905565ca4ccf704d6c6d2c05"
COUNTS = {"banking": 16, "slack": 21, "travel": 20, "workspace": 40}
TOTAL = sum(COUNTS.values())
SCHEMA = "public-agentdojo-native-v1"
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "outputs/public_portfolio_2026-09-05/agentdojo_setup"
PROTOCOL = {"benchmark_version": VERSION, "attack": None, "defense": None,
            "metric": "benign_task_utility", "episode_count": TOTAL,
            "tool_output_format": "yaml", "system_message": "native_default",
            "security_evaluation": False, "heldout_claim": False}


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def runtime_episode_boundary():
    """The separate fast runtime may request a stop before a new episode intent."""
    url = os.environ.get("PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL")
    if not url:
        return None
    base_url = validate_loopback_endpoint(os.environ["OPENAI_COMPATIBLE_BASE_URL"])
    require(url == base_url + "/runtime/episode-boundary", "episode boundary must use the same private loopback gateway")
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + os.environ["OPENAI_COMPATIBLE_API_KEY"]})
    with urllib.request.urlopen(request, timeout=15) as response:
        decision = json.loads(response.read())
    require(type(decision.get("may_start_episode")) is bool, "invalid runtime episode-boundary decision")
    if decision["may_start_episode"]:
        return None
    reason = decision.get("reason")
    return reason if isinstance(reason, str) and reason else "RUNTIME_STOP_REQUESTED"


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(canonical(value))


def validate_loopback_endpoint(url):
    require(isinstance(url, str) and url == url.strip(), "endpoint cannot contain surrounding whitespace")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
        host = parsed.hostname
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        raise ContractError("invalid private loopback endpoint") from None
    require(parsed.scheme == "http" and loopback and port is not None and 1 <= port <= 65535,
            "endpoint must be HTTP loopback with an explicit valid port")
    require(parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment
            and parsed.path in {"/v1", "/v1/"}, "endpoint must have only the /v1 route and no credentials/query/fragment")
    return url.rstrip("/")


def logical_endpoint_identity(identity, runtime_path=None):
    """Bind transport semantics and code, never the ephemeral local TCP port."""
    limits = identity.get("runtime_limits")
    require(isinstance(limits, dict) and bool(limits), "fast native run requires explicit runtime_limits")
    if runtime_path is None:
        local = Path(__file__).resolve().parents[2] / ".codex-run/public_colab_runtime_fast.py"
        runtime_path = local if local.is_file() else Path("/root/public_colab_runtime_fast.py")
    runtime_path = Path(runtime_path)
    require(runtime_path.is_file(), "exact private actor runtime source unavailable for binding")
    return {"transport": "private_guarded_loopback_openai_chat_v1", "api_path": "/v1",
            "episode_boundary_protocol": "authenticated_pre_intent_get_v1",
            "model_identity_sha256": fingerprint(identity), "runtime_limits_sha256": fingerprint(limits),
            "actor_runtime_sha256": digest(runtime_path.read_bytes()),
            "native_wrapper_sha256": digest(Path(__file__).read_bytes())}


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    """Create once using a permanent claim and same-directory atomic rename.

    Modal Volumes do not support hardlinks. A claim is never removed: after an
    interrupted unpublished write, recovery must reconcile it rather than guess
    that another publisher may safely replace the intended artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    require(not path.is_symlink(), f"immutable artifact cannot be symlinked: {path}")
    if path.exists():
        require(path.read_bytes() == data, f"immutable artifact differs: {path}")
        fsync_directory(path.parent)
        return
    claim = path.with_name("." + path.name + ".publish.claim")
    try:
        with claim.open("xb") as handle:
            handle.write(canonical({"target": path.name, "content_sha256": digest(data),
                                    "state": "PUBLICATION_CLAIMED_ONCE"}))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        require(not path.is_symlink(), f"immutable artifact cannot be symlinked: {path}")
        require(path.exists(), f"unfinished publication claim;no automatic retry: {path}")
        require(path.read_bytes() == data, f"immutable artifact differs: {path}")
        fsync_directory(path.parent)
        return
    fsync_directory(path.parent)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # All cooperating publishers require the same exclusive claim. Never
        # overwrite a path that appeared through another publication protocol.
        require(not path.exists() and not path.is_symlink(), f"artifact appeared after exclusive claim: {path}")
        os.rename(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def source_manifest(tree):
    require(tree.get("truncated") is False, "upstream source tree is truncated")
    files = [{"path": item["path"], "git_blob_sha1": item["sha"], "size": item["size"]}
             for item in tree["tree"] if item["type"] == "blob" and
             (item["path"].startswith("src/") or item["path"] in
              {"pyproject.toml", "uv.lock", "LICENSE", "README.md", ".python-version"})]
    result = {"revision": REVISION, "files": sorted(files, key=lambda item: item["path"])}
    require(fingerprint(result) == SOURCE_MANIFEST_SHA256, "pinned source inventory mismatch")
    return result


def git_blob_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch_sources(setup):
    """Fetch 117 pinned source/config blobs (~1.4 MB); omit archived model runs."""
    setup = Path(setup)
    setup.mkdir(parents=True, exist_ok=True)
    manifest_path = setup / "selected_source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require(fingerprint(manifest) == SOURCE_MANIFEST_SHA256, "source inventory drift")
    else:
        request = urllib.request.Request(
            f"https://api.github.com/repos/ethz-spylab/agentdojo/git/trees/{REVISION}?recursive=1",
            headers={"User-Agent": SCHEMA})
        with urllib.request.urlopen(request, timeout=40) as response:
            manifest = source_manifest(json.load(response))
        write_once(manifest_path, manifest)
    root = setup / "source"

    def fetch(item):
        path = root / item["path"]
        require(path.resolve().is_relative_to(root.resolve()), "unsafe source path")
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/ethz-spylab/agentdojo/{REVISION}/{item['path']}"
            with urllib.request.urlopen(url, timeout=40) as response:
                data = response.read(item["size"] + 1)
            require(len(data) == item["size"] and git_blob_hash(data) == item["git_blob_sha1"],
                    f"pinned source blob mismatch: {item['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
        require(not path.is_symlink() and git_blob_hash(data) == item["git_blob_sha1"],
                f"source drift: {item['path']}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, manifest["files"]))
    result = verify_sources(setup)
    write_once(setup / "source_receipt.json", result)
    return result


def verify_sources(setup):
    setup = Path(setup)
    manifest = read_json(setup / "selected_source_manifest.json")
    require(fingerprint(manifest) == SOURCE_MANIFEST_SHA256, "source inventory drift")
    root = setup / "source"
    hashes = {}
    for item in manifest["files"]:
        path = root / item["path"]
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked source: {item['path']}")
        data = path.read_bytes()
        require(len(data) == item["size"] and git_blob_hash(data) == item["git_blob_sha1"],
                f"source drift: {item['path']}")
        hashes[item["path"]] = digest(data)
    actual = {str(path.relative_to(root)) for path in (root / "src").rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    require(actual == {path for path in hashes if path.startswith("src/")}, "extra native source file")
    return {"repository": REPOSITORY, "revision": REVISION,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "files_sha256": hashes,
            "license": "MIT", "model_calls": 0}


def load_native(setup):
    verify_sources(setup)
    root = (Path(setup) / "source" / "src").resolve()
    existing = sys.modules.get("agentdojo")
    if existing is not None:
        require(Path(existing.__file__).resolve().is_relative_to(root), "different agentdojo package loaded")
    else:
        sys.path.insert(0, str(root))
    suites = importlib.import_module("agentdojo.task_suite.load_suites").get_suites(VERSION)
    require({name: len(suite.user_tasks) for name, suite in suites.items()} == COUNTS,
            "native task inventory differs from exact 97-task default")
    return suites


def inventory(suites):
    result = []
    for name in sorted(suites):
        suite = suites[name]
        for task_id, task in sorted(suite.user_tasks.items(), key=lambda pair: int(pair[0].rsplit("_", 1)[1])):
            environment = suite.load_and_inject_default_environment({})
            environment = task.init_environment(environment)
            result.append({"evaluation_id": f"{name}/{task_id}", "suite_name": name,
                           "task_id": task_id, "prompt_sha256": digest(task.PROMPT.encode()),
                           "initial_environment_sha256": fingerprint(environment.model_dump(mode="json")),
                           "native_class": f"{type(task).__module__}.{type(task).__name__}"})
    require(len(result) == TOTAL and len({row["evaluation_id"] for row in result}) == TOTAL,
            "duplicate or missing native task IDs")
    return result


def prepare(setup):
    source = verify_sources(setup)
    tasks = inventory(load_native(setup))
    manifest = {"schema_version": SCHEMA, "source": source, "protocol": PROTOCOL,
                "tasks": tasks, "task_inventory_sha256": fingerprint(tasks),
                "status": "PREPARED_NOT_EXECUTED", "score": None, "model_calls": 0}
    write_once(Path(setup) / "manifest.json", manifest)
    return manifest


def validate_identity(identity):
    for key in ("model_id", "model_revision", "hf_repo", "hf_commit", "served_model_id"):
        require(isinstance(identity.get(key), str) and bool(identity[key].strip()), f"actor identity missing {key}")
    for key in ("model_revision", "hf_commit"):
        require(re.fullmatch(r"[0-9a-f]{40}", identity[key]) is not None, f"actor {key} must be immutable")
    return fingerprint(identity)


def overlap_report(manifest, training=None, identity_sha256=None):
    """Report actual supplied inventory matches; never infer held-out status."""
    result = {"schema_version": SCHEMA, "heldout_claim": False,
              "contract_declares_agentdojo_training": True,
              "declaration_is_training_consumption_evidence": False,
              "status": "TRAINING_INVENTORY_ABSENT", "matching_evaluation_ids": [],
              "matching_prompt_hashes": [], "training_inventory_sha256": None}
    if training is None:
        return result
    require(isinstance(training.get("tasks"), list), "training manifest must contain tasks list")
    result["training_inventory_sha256"] = fingerprint(training)
    ids = {row["evaluation_id"] for row in manifest["tasks"]}
    prompts = {row["prompt_sha256"] for row in manifest["tasks"]}
    supplied_ids, supplied_prompts = set(), set()
    for task in training["tasks"]:
        require(isinstance(task, dict), "malformed training task inventory row")
        if task.get("evaluation_id"):
            supplied_ids.add(task["evaluation_id"])
        if task.get("suite_name") and task.get("task_id"):
            supplied_ids.add(f"{task['suite_name']}/{task['task_id']}")
        if task.get("prompt_sha256"):
            supplied_prompts.add(task["prompt_sha256"])
    result["matching_evaluation_ids"] = sorted(ids & supplied_ids)
    result["matching_prompt_hashes"] = sorted(prompts & supplied_prompts)
    bound = identity_sha256 is not None and training.get("model_identity_sha256") == identity_sha256
    complete = training.get("complete") is True and bound
    result.update({"declared_complete_and_model_bound": complete, "training_task_count": len(training["tasks"])})
    result["status"] = ("OVERLAP_FOUND" if result["matching_evaluation_ids"] or result["matching_prompt_hashes"]
                        else "NO_OVERLAP_IN_SUPPLIED_COMPLETE_MANIFEST" if complete
                        else "NO_MATCH_IN_INCOMPLETE_OR_UNBOUND_MANIFEST")
    return result


def trace_path(output, row):
    return Path(output) / "native" / "openai-compatible" / row["suite_name"] / row["task_id"] / "none" / "none.json"


def validate_native_trace(path, row):
    data = read_json(path)
    require(data.get("suite_name") == row["suite_name"] and data.get("user_task_id") == row["task_id"],
            "native trace belongs to another task")
    require(data.get("benchmark_version") == VERSION, "native trace version mismatch")
    # Native TraceLogger serializes attack_type="none" as JSON null.
    require(data.get("attack_type") is None and data.get("injection_task_id") is None,
            "native trace is not benign default")
    require(data.get("pipeline_name") == "openai-compatible" and data.get("injections") == {},
            "native trace pipeline/injections mismatch")
    require(type(data.get("utility")) is bool, "native trace lacks final utility")
    require(type(data.get("security")) is bool and data["security"] is True,
            "native benign trace lacks final default result")
    require(isinstance(data.get("duration"), (int, float)) and data["duration"] >= 0,
            "native trace did not finalize")
    require(isinstance(data.get("messages"), list) and len(data["messages"]) >= 2,
            "native trace lacks complete trajectory")
    return data


def task_result(output, row, run_hash):
    path = trace_path(output, row)
    data = validate_native_trace(path, row)
    return {"schema_version": SCHEMA, "evaluation_id": row["evaluation_id"],
            "task_fingerprint": fingerprint(row), "run_manifest_sha256": run_hash,
            "native_trace_path": str(path.resolve()), "native_trace_sha256": digest(path.read_bytes()),
            "utility": data["utility"], "native_error": data.get("error"),
            "metric": "benign_task_utility", "security_evaluation": False, "heldout_claim": False}


def bind_task_header(pipeline, original_client, evaluation_id, run_hash):
    """Give a metering proxy episode context without changing native prompts/tools.

    AgentPipeline.from_config's native default shares elements[2] (OpenAILLM)
    with its ToolsExecutionLoop. OpenAI.with_options changes transport headers
    only. It does not add a sampling cap or make a model request.
    """
    require(len(pipeline.elements) == 4 and hasattr(pipeline.elements[2], "client"),
            "unexpected native default pipeline layout")
    pipeline.elements[2].client = original_client.with_options(default_headers={
        "X-Public-Task-ID": evaluation_id, "X-Public-Run-ID": run_hash})


def summarize(output):
    output = Path(output)
    run = read_json(output / "run_manifest.json")
    run_hash = fingerprint(run)
    rows = run["tasks"]
    require(len(rows) == TOTAL and run.get("protocol") == PROTOCOL, "invalid run coverage/protocol")
    require(len({row["evaluation_id"] for row in rows}) == TOTAL, "duplicate evaluation IDs")
    require(run.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256,
            "wrong pinned source inventory")
    require(run.get("task_inventory_sha256") == fingerprint(rows), "task inventory drift")
    for row in rows:
        require(row["suite_name"] in COUNTS and re.fullmatch(r"user_task_\d+", row["task_id"])
                and row["evaluation_id"] == f"{row['suite_name']}/{row['task_id']}", "invalid native task ID")
    results, missing = [], []
    for row in rows:
        path = output / "receipts" / row["suite_name"] / (row["task_id"] + ".json")
        if not path.exists():
            missing.append(row["evaluation_id"])
            continue
        result = read_json(path)
        require(result == task_result(output, row, run_hash), f"task receipt or trace drift: {row['evaluation_id']}")
        results.append(result)
    finished = not missing
    utility_passes = sum(row["utility"] for row in results)
    report = {"schema_version": SCHEMA, "suite_id": "agentdojo_eval", "protocol": PROTOCOL,
              "status": "COMPLETE_NATIVE_BENIGN_UTILITY" if finished else "PARTIAL_NATIVE_BENIGN_UTILITY",
              "score": utility_passes / TOTAL if finished else None, "score_denominator": TOTAL,
              "completed_episodes": len(results), "utility_passes": utility_passes,
              "native_handled_error_count": sum(bool(row["native_error"]) for row in results),
              "missing_evaluation_ids": missing, "run_manifest_sha256": run_hash,
              "model_identity": run["model_identity"], "decontamination": run["decontamination"],
              "declared_inference_limits": run["model_identity"].get("runtime_limits", {}),
              "claim_boundary": "Native default benign task utility only;no prompt-injection security or held-out claim."}
    # Snapshots are immutable and keyed by their content so partials remain auditable.
    write_once(output / "summaries" / (fingerprint(report) + ".json"), report)
    return report


@contextlib.contextmanager
def output_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError("another AgentDojo worker owns this output") from error
        yield


def run(args):
    setup, output = Path(args.setup), Path(args.output)
    manifest = prepare(setup)
    identity = read_json(args.model_identity)
    identity_hash = validate_identity(identity)
    require(os.environ.get("OPENAI_COMPATIBLE_BASE_URL"), "OPENAI_COMPATIBLE_BASE_URL is absent")
    require(os.environ.get("OPENAI_COMPATIBLE_API_KEY"), "OPENAI_COMPATIBLE_API_KEY is absent")
    actual_url = os.environ["OPENAI_COMPATIBLE_BASE_URL"]
    normalized_url = validate_loopback_endpoint(actual_url)
    boundary_url = os.environ.get("PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL")
    require(boundary_url == normalized_url + "/runtime/episode-boundary",
            "episode boundary must use the same private loopback gateway")
    endpoint_identity = logical_endpoint_identity(identity)
    training = read_json(args.training_manifest) if args.training_manifest else None
    run_manifest = {"schema_version": SCHEMA, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                    "protocol": PROTOCOL, "tasks": manifest["tasks"], "model_identity": identity,
                    "task_inventory_sha256": manifest["task_inventory_sha256"],
                    "logical_endpoint_identity": endpoint_identity,
                    "logical_endpoint_identity_sha256": fingerprint(endpoint_identity),
                    "decontamination": overlap_report(manifest, training, identity_hash),
                    "wandb": {"entity": args.wandb_entity, "project": args.wandb_project, "id": args.wandb_run_id}}
    with output_lock(output):
        write_once(output / "run_manifest.json", run_manifest)
        run_hash = fingerprint(run_manifest)
        write_once(output / "runtime_invocations" / (uuid.uuid4().hex + ".json"), {
            "recorded_at": now(), "run_manifest_sha256": run_hash,
            "logical_endpoint_identity_sha256": fingerprint(endpoint_identity),
            "actual_endpoint": actual_url,
            "actual_episode_boundary": boundary_url,
            "actual_endpoint_sha256": digest(actual_url.encode()),
            "actual_episode_boundary_sha256": digest(boundary_url.encode()) if boundary_url else None,
            "scientific_identity_excludes_ephemeral_port": True})
        # This is the only online tracking/provider section in this module.
        import wandb
        tracking = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                              id=args.wandb_run_id, resume="allow", mode="online",
                              config={"suite": "agentdojo_eval", "protocol": PROTOCOL,
                                      "source_revision": REVISION, "model_identity": identity})
        active_task_id = None
        try:
            require(tracking is not None and tracking.settings.mode == "online" and tracking.id == args.wandb_run_id,
                    "live online W&B initialization required before native pipeline construction")
            write_once(output / "wandb_online_receipt.json",
                       {"mode": "online", "run_id": tracking.id, "url": tracking.url,
                        "run_manifest_sha256": run_hash, "initialized_before_model_work": True})
            suites = load_native(setup)
            pipeline_module = importlib.import_module("agentdojo.agent_pipeline.agent_pipeline")
            benchmark = importlib.import_module("agentdojo.benchmark")
            logging_module = importlib.import_module("agentdojo.logging")
            pipeline = pipeline_module.AgentPipeline.from_config(pipeline_module.PipelineConfig(
                llm="openai-compatible", model_id=identity["served_model_id"], defense=None,
                system_message_name=None, system_message=None, tool_output_format=None))
            original_client = pipeline.elements[2].client
            completed_now = 0
            with logging_module.OutputLogger(str(output / "native")):
                for row in manifest["tasks"]:
                    receipt = output / "receipts" / row["suite_name"] / (row["task_id"] + ".json")
                    intent = output / "intents" / row["suite_name"] / (row["task_id"] + ".json")
                    binding = {"run_manifest_sha256": run_hash, "task_fingerprint": fingerprint(row)}
                    if receipt.exists():
                        require(read_json(receipt) == task_result(output, row, run_hash), "completed task drift")
                        continue
                    if intent.exists():
                        require(read_json(intent) == binding, "task intent identity drift")
                        # Recover only an already finalized native trace. An ambiguous task
                        # must not be silently resampled after a crash or provider timeout.
                        require(trace_path(output, row).exists(), f"ambiguous task intent;no resampling: {row['evaluation_id']}")
                        write_once(receipt, task_result(output, row, run_hash))
                        continue
                    if args.max_tasks is not None and completed_now >= args.max_tasks:
                        break
                    stop_reason = runtime_episode_boundary()
                    if stop_reason:
                        stop_path = os.environ.get("PUBLIC_RUNTIME_EPISODE_STOP_RECEIPT")
                        require(stop_path, "runtime stop receipt path required")
                        write_once(Path(stop_path), {"reason": stop_reason, "recorded_at": now(),
                            "next_unstarted_evaluation_id": row["evaluation_id"], "intent_written": False,
                            "completed_this_invocation": completed_now, "score": None})
                        break
                    require(not trace_path(output, row).exists(), "native trace has no bound task intent")
                    bind_task_header(pipeline, original_client, row["evaluation_id"], run_hash)
                    write_once(intent, binding)
                    active_task_id = row["evaluation_id"]
                    suite = suites[row["suite_name"]]
                    benchmark.run_task_without_injection_tasks(
                        suite, pipeline, suite.get_user_task_by_id(row["task_id"]),
                        logdir=output / "native", force_rerun=False, benchmark_version=VERSION)
                    result = task_result(output, row, run_hash)
                    write_once(receipt, result)
                    active_task_id = None
                    completed_now += 1
                    tracking.log({"native_benign_utility": int(result["utility"]),
                                  "evaluation_id": row["evaluation_id"]})
            report = summarize(output)
            tracking.summary.update({"completed_episodes": report["completed_episodes"],
                                     "suite_score": report["score"], "status": report["status"]})
            return report
        except BaseException as error:
            # Budget exhaustion/provider interruption is operational evidence,
            # never an invented false utility result. Completed receipts survive.
            report = summarize(output)
            failure = {"schema_version": SCHEMA, "recorded_at": now(),
                       "exception_type": f"{type(error).__module__}.{type(error).__name__}",
                       "active_evaluation_id": active_task_id,
                       "completed_episodes": report["completed_episodes"],
                       "score": report["score"], "run_manifest_sha256": run_hash,
                       "resampling_performed": False,
                       "detail": "See native trace and guarded runtime receipts;unfinalized episode remains missing."}
            write_once(output / "failures" / (fingerprint(failure) + ".json"), failure)
            raise
        finally:
            if tracking is not None:
                tracking.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-sources")
    sub.add_parser("prepare")
    overlap = sub.add_parser("overlap")
    overlap.add_argument("--training-manifest", type=Path)
    overlap.add_argument("--model-identity", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--model-identity", type=Path, required=True)
    launch.add_argument("--training-manifest", type=Path)
    launch.add_argument("--wandb-entity", required=True)
    launch.add_argument("--wandb-project", required=True)
    launch.add_argument("--wandb-run-id", required=True)
    launch.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.command == "fetch-sources":
        result = fetch_sources(args.setup)
    elif args.command == "prepare":
        result = prepare(args.setup)
    elif args.command == "summarize":
        result = summarize(args.output)
    elif args.command == "overlap":
        identity_hash = validate_identity(read_json(args.model_identity)) if args.model_identity else None
        result = overlap_report(prepare(args.setup), read_json(args.training_manifest) if args.training_manifest else None,
                                identity_hash)
    else:
        require(args.max_tasks is None or args.max_tasks > 0, "max-tasks must be positive")
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
