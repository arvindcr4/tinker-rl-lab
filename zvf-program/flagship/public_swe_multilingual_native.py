"""Offline SWE-bench Multilingual preparation and native receipt bridge.

This module never calls a model, starts Docker, downloads container layers, or
reruns a terminal attempt. `plan` writes a command for a separately authorized
runtime; `ingest` verifies that runtime's existing native artifacts.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

DATASET = "SWE-bench/SWE-bench_Multilingual"
DATASET_REVISION = "846e647b9f33c0b51b739d005d13d85493c9af09"
SOURCE_REVISION = "7a21e05772954cc81471ae19d56f436cecf43c54"
PARQUET_SHA256 = "92abca7cb527b41a9f66d03a26ce441ff7319e3a49f985998fd56be4bb9b08b2"
SOURCE_MANIFEST_SHA256 = "77547723b9520046a3619be8df0a45c745913e3038b3cb48e3f8654e64e97fd8"
COUNT = 300
ACTOR_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
GENERATION_STATUSES = {"GENERATED", "GENERATION_FAILED"}
IDENTITY_FIELDS = {"model_id", "model_revision", "hf_repo", "hf_commit", "served_model_id"}
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "outputs/public_portfolio_2026-09-05/swe_multilingual_setup"


class BoundaryError(ValueError):
    """Incomplete, changed, or unbound evidence cannot produce a suite score."""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def object_hash(value: Any) -> str:
    return digest(canonical(value))


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise BoundaryError(f"Refusing to replace existing artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def write(path: Path, value: Any) -> None:
    immutable(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def write_lines(path: Path, values: list[dict]) -> None:
    immutable(path, b"".join(canonical(value) + b"\n" for value in values))


def check_hash(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path.read_bytes()) != expected:
        raise BoundaryError(f"Artifact absent or changed: {path}")


def indexed(rows: list[dict], expected: set[str] | None = None) -> dict[str, dict]:
    result = {}
    for row in rows:
        iid = row.get("instance_id")
        if not isinstance(iid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", iid) or iid in result:
            raise BoundaryError("Missing, unsafe, or duplicate instance ID")
        result[iid] = row
    if expected is not None and set(result) != expected:
        raise BoundaryError("Task coverage differs from the complete pinned inventory")
    return result


def verify_sources(setup: Path) -> dict:
    check_hash(setup / "source_manifest.json", SOURCE_MANIFEST_SHA256)
    manifest = read(setup / "source_manifest.json")
    if manifest["revision"] != SOURCE_REVISION:
        raise BoundaryError("Native source revision mismatch")
    for row in manifest["files"]:
        check_hash(setup / "source" / row["path"], row["sha256"])
    wanted = {row["path"] for row in manifest["files"] if row["path"].endswith(".py")}
    found = {str(path.relative_to(setup / "source")) for path in (setup / "source").rglob("*.py")}
    if found != wanted:
        raise BoundaryError("Unexpected executable Python source in the pinned harness")
    return manifest


def native(setup: Path):
    verify_sources(setup)
    source = (setup / "source").resolve()
    sys.path.insert(0, str(source))
    package = importlib.import_module("swebench")
    if Path(package.__file__).resolve().parent != source / "swebench":
        raise BoundaryError("A different SWE-bench installation was already imported")
    utils = importlib.import_module("swebench.harness.utils")
    grading = importlib.import_module("swebench.harness.grading")
    return utils, grading


def actor_task(raw: dict) -> dict:
    """Allowlist excludes reference patch, tests, image, and evaluator fields."""
    return {key: raw[key] for key in ACTOR_FIELDS}


def build_actor_request(task: dict, sources: dict[str, str], served_model: str,
                        *, max_tokens: int, temperature: float, seed: int) -> dict:
    """Adapt the original E1 patch-only prompt to an OpenAI-compatible actor.

    `sources` must be independently collected at the pinned base commit; this
    function is only a serialization interface, not a new repository solver.
    """
    if set(task) != set(ACTOR_FIELDS) or not isinstance(sources, dict):
        raise BoundaryError("Unexpected actor-visible task fields")
    if max_tokens < 1 or not 0 <= temperature <= 2 or not served_model:
        raise BoundaryError("Explicit finite inference parameters are required")
    for path, content in sources.items():
        if not isinstance(content, str) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise BoundaryError("Invalid base-commit source context")
    context = "\n\n".join(f"===== {path} =====\n{content}" for path, content in sorted(sources.items()))
    prompt = f"""Solve this frozen SWE-bench Multilingual task in repository {task['repo']}.

Return only a valid unified git diff beginning with `diff --git`. Do not emit
analysis, explanations, planning, ellipses, abbreviated context, or Markdown
fences. Every hunk header must have concrete line ranges. Make the smallest
complete production fix, preserve project style, and ensure the patch applies
to base commit {task['base_commit']}.

PROBLEM STATEMENT
{task['problem_statement']}

FROZEN BASE-COMMIT SOURCE CONTEXT
{context}
"""
    return {"model": served_model, "messages": [
        {"role": "system", "content": "You are a deterministic source-code patch generator. Reply with the requested patch only."},
        {"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95, "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False}}


def prepare(setup: Path) -> dict:
    check_hash(setup / "test.parquet", PARQUET_SHA256)
    utils, _ = native(setup)
    import pyarrow.parquet as pq
    raw = pq.read_table(setup / "test.parquet").to_pylist()
    if len(raw) != COUNT:
        raise BoundaryError("Pinned split must contain exactly 300 rows")
    indexed(raw)
    tasks, specs, actors = [], [], []
    registry = importlib.import_module("swebench.harness.log_parsers").PARSER_REGISTRY
    for row in raw:
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise BoundaryError("Task base commit is not immutable")
        spec = utils.make_test_spec(row)
        if spec.log_parser not in registry or not spec.FAIL_TO_PASS:
            raise BoundaryError("Native parser or expected failing tests unavailable")
        native_spec = {**asdict(spec), "eval_script": spec.eval_script}
        public = actor_task(row)
        tasks.append({"instance_id": row["instance_id"], "repo": row["repo"],
                      "base_commit": row["base_commit"], "image": row["image"],
                      "raw_task_sha256": object_hash(row), "actor_task_sha256": object_hash(public),
                      "test_spec_sha256": object_hash(native_spec),
                      "eval_script_sha256": digest(spec.eval_script.encode()),
                      "test_patch_sha256": digest(row["test_patch"].encode()),
                      "reference_patch_sha256": digest(row["patch"].encode()),
                      "fail_to_pass_count": len(spec.FAIL_TO_PASS),
                      "pass_to_pass_count": len(spec.PASS_TO_PASS), "log_parser": spec.log_parser})
        specs.append(native_spec)
        actors.append(public)
    prepared = setup / "prepared"
    write_lines(prepared / "native_dataset.jsonl", raw)
    write_lines(prepared / "native_test_specs.jsonl", specs)
    write_lines(prepared / "actor_tasks.jsonl", actors)
    files = {name: digest((prepared / name).read_bytes()) for name in
             ["native_dataset.jsonl", "native_test_specs.jsonl", "actor_tasks.jsonl"]}
    manifest = {"schema_version": "public-swe-multilingual-manifest-v1", "suite_id": "swe_bench_multilingual_eval",
                "dataset": DATASET, "dataset_revision": DATASET_REVISION, "split": "test",
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "parquet_sha256": PARQUET_SHA256, "task_count": COUNT,
                "tasks": tasks, "task_inventory_sha256": object_hash(tasks), "files_sha256": files,
                "native_default_protocol": {"metric": "resolved_fraction", "samples_per_task": 1,
                                            "workers": 4, "timeout_seconds": 1800, "split": "test"},
                "image_digest_status": "NOT_RESOLVED_FROM_NATIVE_MUTABLE_TAGS",
                "training_overlap_status": "UNVERIFIED_NO_CONSUMED_TASK_INVENTORY", "heldout_claim": False,
                "status": "NATIVE_DATA_AND_TEST_SPECS_PREPARED_NO_EXECUTION", "score": None}
    write(prepared / "manifest.json", manifest)
    return {key: manifest[key] for key in ["status", "task_count", "task_inventory_sha256", "score"]}


def load_prepared(setup: Path) -> tuple[dict, dict[str, dict]]:
    verify_sources(setup)
    check_hash(setup / "test.parquet", PARQUET_SHA256)
    manifest = read(setup / "prepared/manifest.json")
    if (manifest.get("dataset_revision"), manifest.get("source_revision"), manifest.get("task_count")) != (DATASET_REVISION, SOURCE_REVISION, COUNT):
        raise BoundaryError("Prepared identity is not the pinned 300-task suite")
    for name, expected in manifest["files_sha256"].items():
        check_hash(setup / "prepared" / name, expected)
    tasks = indexed(manifest["tasks"])
    if len(tasks) != COUNT or object_hash(manifest["tasks"]) != manifest["task_inventory_sha256"]:
        raise BoundaryError("Prepared task inventory changed")
    raw = indexed(lines(setup / "prepared/native_dataset.jsonl"), set(tasks))
    for iid, row in raw.items():
        if object_hash(row) != tasks[iid]["raw_task_sha256"]:
            raise BoundaryError("Raw task hash mismatch")
        if tasks[iid]["actor_task_sha256"] != object_hash(actor_task(row)) or any(tasks[iid][key] != row[key] for key in ["repo", "base_commit", "image"]):
            raise BoundaryError("Actor task projection or source identity changed")
    # Rebuilding from the hard-pinned parquet prevents coordinated manifest edits.
    import pyarrow.parquet as pq
    official = indexed(pq.read_table(setup / "test.parquet").to_pylist(), set(tasks))
    if raw != official:
        raise BoundaryError("Prepared data differs from the pinned released rows")
    return manifest, raw


def validate_identity(value: dict) -> None:
    if not IDENTITY_FIELDS <= value.keys() or any(not value[key] for key in IDENTITY_FIELDS):
        raise BoundaryError("Complete actor identity is required")
    for key in ["model_revision", "hf_commit"]:
        if not re.fullmatch(r"[0-9a-f]{40}", value[key]):
            raise BoundaryError("Actor revisions must be immutable commits")


def collect_attempts(manifest: dict, attempt_root: Path, identity: dict) -> tuple[list[dict], list[dict]]:
    """Read original E1-style per-task generation files with added suite binding.

    Genuine completed invalid model responses become empty native predictions.
    Lost artifacts, provider errors, and ambiguous sampling intents stay blocked.
    """
    validate_identity(identity)
    predictions, receipts = [], []
    expected = {row["instance_id"] for row in manifest["tasks"]}
    present = {path.parent.name for path in attempt_root.glob("*/generation.json")}
    if present != expected:
        raise BoundaryError("All 300 unique terminal generation receipts are required")
    for task in manifest["tasks"]:
        iid = task["instance_id"]
        directory = attempt_root / iid
        receipt = read(directory / "generation.json")
        if receipt.get("instance_id") != iid or receipt.get("status") not in GENERATION_STATUSES:
            raise BoundaryError("Missing or nonterminal generation status")
        if receipt.get("sample_started") is not True or receipt.get("sample_completed") is not True:
            raise BoundaryError("Sampling outcome is incomplete; no implicit retry or empty patch")
        if receipt.get("task_inventory_sha256") != manifest["task_inventory_sha256"] or receipt.get("actor_task_sha256") != task["actor_task_sha256"]:
            raise BoundaryError("Generation is not bound to the frozen multilingual actor task")
        if receipt.get("model_identity_sha256") != object_hash(identity):
            raise BoundaryError("Generation actor identity mismatch")
        if not receipt.get("wandb_run_id") or receipt.get("wandb_mode") != "online":
            raise BoundaryError("Generation lacks online W&B evidence")
        if not receipt.get("started_at") or not receipt.get("finished_at"):
            raise BoundaryError("Generation timing evidence is absent")
        check_hash(directory / "generation_response.txt", receipt.get("response_sha256", ""))
        source = read(directory / "source_context.json")
        if source.get("base_commit") != task["base_commit"] or source.get("repo") != task["repo"]:
            raise BoundaryError("Source context not bound to the task base commit")
        if receipt.get("source_sha256") != object_hash(source["files"]):
            raise BoundaryError("Generation source context changed")
        request_path = directory / "generation_request.json"
        check_hash(request_path, receipt.get("request_sha256", ""))
        request = read(request_path)
        actor = {key: source["actor_task"][key] for key in ACTOR_FIELDS}
        if object_hash(actor) != task["actor_task_sha256"] or set(source["actor_task"]) != set(ACTOR_FIELDS):
            raise BoundaryError("Source context embeds a different actor task")
        expected_request = build_actor_request(actor, source["files"], identity["served_model_id"],
                                               max_tokens=request["max_tokens"], temperature=request["temperature"], seed=request["seed"])
        if request != expected_request:
            raise BoundaryError("Actor request contains changed prompt or unexpected evaluation fields")
        patch = receipt.get("patch") or ""
        if not isinstance(patch, str):
            raise BoundaryError("Patch must be a string")
        if receipt["status"] == "GENERATED":
            if not patch.strip() or receipt.get("patch_sha256") != digest(patch.encode()):
                raise BoundaryError("Generated patch absent or changed")
            if patch.strip() not in (directory / "generation_response.txt").read_text():
                raise BoundaryError("Candidate patch is absent from the retained model response")
        elif patch:
            raise BoundaryError("Failed generation cannot secretly contain a patch")
        predictions.append({"instance_id": iid, "model_patch": patch,
                            "model_name_or_path": identity["served_model_id"]})
        receipts.append({"instance_id": iid, "generation_sha256": digest((directory / "generation.json").read_bytes()),
                         "response_sha256": receipt["response_sha256"], "status": receipt["status"],
                         "patch_sha256": digest(patch.encode()), "wandb_run_id": receipt["wandb_run_id"]})
    return predictions, receipts


def validate_images(images: dict, tasks: dict[str, dict]) -> dict[str, dict]:
    rows = indexed(images.get("images", []), set(tasks))
    for iid, row in rows.items():
        original = tasks[iid]["image"]
        repository = original.rsplit(":", 1)[0]
        if row.get("source_tag") != original or row.get("platform") != "linux/amd64":
            raise BoundaryError("Image source tag or native architecture differs")
        image = row.get("image", "")
        if not re.fullmatch(re.escape(repository) + r"@sha256:[0-9a-f]{64}", image):
            raise BoundaryError("All native image references must use immutable digests")
        if row.get("manifest_sha256") != image.rsplit(":", 1)[1] or not row.get("resolved_at"):
            raise BoundaryError("Image digest requires a dated registry manifest receipt")
        try:
            registry_raw = base64.b64decode(row["registry_manifest_base64"], validate=True)
            registry_manifest = json.loads(registry_raw)
        except (KeyError, ValueError, TypeError) as exc:
            raise BoundaryError("Raw registry manifest bytes are required") from exc
        if digest(registry_raw) != row["manifest_sha256"] or registry_manifest.get("schemaVersion") != 2:
            raise BoundaryError("Registry manifest bytes disagree with image digest")
    return rows


def plan(setup: Path, output: Path, attempts: Path, identity_path: Path,
         images_path: Path, run_id: str, workers: int = 4, timeout: int = 1800) -> dict:
    manifest, raw = load_prepared(setup)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or workers < 1 or timeout < 1:
        raise BoundaryError("Safe unique run ID and positive native runtime limits required")
    identity = read(identity_path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity.get("served_model_id", "")):
        raise BoundaryError("Use a filesystem-safe served model alias")
    predictions, attempts_index = collect_attempts(manifest, attempts, identity)
    images_value = read(images_path)
    images = validate_images(images_value, raw)
    output = output.resolve()
    if (output / "logs/run_evaluation" / run_id).exists():
        raise BoundaryError("Existing native run directory: refuse stale cache or automatic re-evaluation")
    runtime_rows = [{**row, "image": images[iid]["image"]} for iid, row in raw.items()]
    write_lines(output / "runtime_dataset.jsonl", runtime_rows)
    write_lines(output / "predictions.jsonl", predictions)
    write(output / "attempt_receipts.json", attempts_index)
    write(output / "model_identity.json", identity)
    write(output / "image_manifest.json", images_value)
    command = [str((setup / ".venv/bin/swebench").resolve()), "eval", str(output / "runtime_dataset.jsonl"),
               "-p", str(output / "predictions.jsonl"), "--run-id", run_id, "--split", "test",
               "-j", str(workers), "--timeout", str(timeout), "--report-dir", str(output / "reports")]
    artifact_names = ["runtime_dataset.jsonl", "predictions.jsonl", "attempt_receipts.json", "model_identity.json", "image_manifest.json"]
    result = {"schema_version": "public-swe-multilingual-plan-v1", "status": "READY_FOR_SEPARATE_NATIVE_RUNTIME",
              "run_id": run_id, "cwd": str(output), "argv": command,
              "task_count": COUNT, "task_inventory_sha256": manifest["task_inventory_sha256"],
              "source_revision": SOURCE_REVISION, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
              "dataset_revision": DATASET_REVISION, "model_identity_sha256": object_hash(identity),
              "files_sha256": {name: digest((output / name).read_bytes()) for name in artifact_names},
              "workers": workers, "timeout_seconds": timeout, "score": None,
              "materialization": "Only image references replaced by digest from registry receipts; task/test fields unchanged",
              "execution_receipt_required": {"plan_sha256": "sha256 of native_plan.json", "returncode": 0,
                                              "started_at": "UTC timestamp", "finished_at": "UTC timestamp",
                                              "wandb_mode": "online", "wandb_run_id": "actual native runtime tracking run",
                                              "native_artifacts": "map relative artifact path to SHA256, including aggregate/report/log/patch/eval/run metadata",
                                              "image_observations": "map attempted nonempty instance ID to actual container image digest"}}
    write(output / "native_plan.json", result)
    immutable(output / "native_command.txt", ("cd " + shlex.quote(str(output)) + "\n" + shlex.join(command) + "\n").encode())
    return result


def ingest(setup: Path, output: Path, execution_path: Path) -> dict:
    """Validate native output, recomputing reports from retained native test logs."""
    manifest, raw = load_prepared(setup)
    runtime_plan = read(output / "native_plan.json")
    if runtime_plan.get("task_inventory_sha256") != manifest["task_inventory_sha256"] or runtime_plan.get("source_revision") != SOURCE_REVISION:
        raise BoundaryError("Native plan differs from pinned task/source identity")
    for name, expected in runtime_plan["files_sha256"].items():
        check_hash(output / name, expected)
    execution = read(execution_path)
    if execution.get("plan_sha256") != digest((output / "native_plan.json").read_bytes()):
        raise BoundaryError("Execution receipt is not bound to this immutable plan")
    if execution.get("wandb_mode") != "online" or not execution.get("wandb_run_id"):
        raise BoundaryError("Online runtime W&B receipt required")
    if not execution.get("started_at") or not execution.get("finished_at"):
        raise BoundaryError("Runtime timing receipt required")
    artifacts = execution.get("native_artifacts", {})
    for name, expected in artifacts.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise BoundaryError("Native artifact path escapes the runtime output")
        check_hash(output / name, expected)
    def artifact(name: str) -> Path:
        if name not in artifacts:
            raise BoundaryError(f"Unrecorded native artifact: {name}")
        return output / name
    predictions = indexed(lines(output / "predictions.jsonl"), set(raw))
    attempt_index = indexed(read(output / "attempt_receipts.json"), set(raw))
    for iid, attempt in attempt_index.items():
        if attempt.get("status") not in GENERATION_STATUSES or not attempt.get("wandb_run_id"):
            raise BoundaryError("Native result lacks all terminal actor attempt receipts")
        if attempt.get("patch_sha256") != digest(predictions[iid]["model_patch"].encode()):
            raise BoundaryError("Native predictions differ from terminal actor attempts")
        for field in ["generation_sha256", "response_sha256"]:
            if not re.fullmatch(r"[0-9a-f]{64}", attempt.get(field, "")):
                raise BoundaryError("Actor attempt index lacks immutable generation/response hashes")
    identity = read(output / "model_identity.json")
    validate_identity(identity)
    if object_hash(identity) != runtime_plan["model_identity_sha256"]:
        raise BoundaryError("Runtime model identity drift")
    alias = identity["served_model_id"]
    image_rows = validate_images(read(output / "image_manifest.json"), raw)
    runtime_rows = indexed(lines(output / "runtime_dataset.jsonl"), set(raw))
    for iid, row in runtime_rows.items():
        if row != {**raw[iid], "image": image_rows[iid]["image"]}:
            raise BoundaryError("Runtime data changed beyond image digest materialization")
        if predictions[iid].get("model_name_or_path") != alias:
            raise BoundaryError("Prediction actor alias drift")
    run_id = runtime_plan["run_id"]
    metadata = read(artifact(f"logs/run_evaluation/{run_id}/run.json"))
    if metadata.get("dataset") != str((output / "runtime_dataset.jsonl").resolve()) or metadata.get("split") != "test" or metadata.get("task_repo") is not None:
        raise BoundaryError("Native metadata names a different dataset or task repository")
    report_name = f"reports/{alias}.{run_id}.json"
    aggregate = read(artifact(report_name))
    if aggregate.get("total_instances") != COUNT or aggregate.get("submitted_instances") != COUNT or set(aggregate.get("submitted_ids", [])) != set(raw):
        raise BoundaryError("Native aggregate does not cover all 300 patch attempts")
    for field in ["submitted", "completed", "empty_patch", "resolved", "unresolved", "error", "infra_failure", "ambiguous_failure"]:
        ids = aggregate.get(field + "_ids", [])
        if len(ids) != len(set(ids)) or not set(ids) <= set(raw) or aggregate.get(field + "_instances") != len(ids):
            raise BoundaryError("Native aggregate counts or task IDs are inconsistent")
    if aggregate.get("incomplete_ids"):
        raise BoundaryError("Native harness declares missing model predictions")
    utils, grading = native(setup)
    outcomes, pending = [], []
    for iid, row in raw.items():
        pred = predictions[iid]
        prefix = f"logs/run_evaluation/{run_id}/{alias}/{iid}/"
        if pred["model_patch"] == "":
            if iid not in aggregate.get("empty_patch_ids", []):
                raise BoundaryError("Empty prediction not accounted for by native report")
            outcomes.append({"instance_id": iid, "status": "NATIVE_EMPTY_PATCH", "resolved": False})
            continue
        observed = execution.get("image_observations", {}).get(iid)
        if observed != image_rows[iid]["image"]:
            pending.append({"instance_id": iid, "reason": "actual container image digest unverified"})
            continue
        report_path = output / (prefix + "report.json")
        if not report_path.is_file():
            # A native patch-application rejection is a final model outcome.
            # Other native errors remain unresolved infrastructure evidence.
            log_name = prefix + "run_instance.log"
            patch_name = prefix + "patch.diff"
            if log_name in artifacts and patch_name in artifacts:
                log = artifact(log_name).read_text()
                if ">>>>> Patch Apply Failed" in log and iid in aggregate.get("error_ids", []):
                    if artifact(patch_name).read_text() != pred["model_patch"]:
                        raise BoundaryError("Rejected patch bytes differ from submitted model patch")
                    outcomes.append({"instance_id": iid, "status": "NATIVE_PATCH_APPLY_FAILED", "resolved": False})
                    continue
            pending.append({"instance_id": iid, "reason": "native final report absent"})
            continue
        saved = read(artifact(prefix + "report.json"))
        spec = utils.make_test_spec(runtime_rows[iid])
        if artifact(prefix + "patch.diff").read_text() != pred["model_patch"]:
            raise BoundaryError("Native patch differs from prediction")
        if artifact(prefix + "eval.sh").read_text() != spec.eval_script:
            raise BoundaryError("Executed test script differs from native pinned specification")
        artifact(prefix + "run_instance.log")
        recomputed = grading.get_eval_report(spec, pred, str(artifact(prefix + "test_output.txt")), include_tests_status=True)
        if saved != recomputed:
            raise BoundaryError("Saved native report disagrees with pinned grader on retained log")
        outcome = recomputed[iid]
        if iid not in aggregate.get("completed_ids", []) or type(outcome.get("resolved")) is not bool:
            raise BoundaryError("Native task report is not accounted for by aggregate")
        if outcome.get("infra_failure") is True or iid in aggregate.get("infra_failure_ids", []):
            pending.append({"instance_id": iid, "reason": "native grader identifies infrastructure failure"})
            continue
        outcomes.append({"instance_id": iid, "status": "NATIVE_REPORT_VERIFIED", "resolved": outcome["resolved"]})
    resolved = {row["instance_id"] for row in outcomes if row["resolved"]}
    if resolved != set(aggregate.get("resolved_ids", [])) or len(resolved) != aggregate.get("resolved_instances"):
        raise BoundaryError("Native aggregate resolved IDs disagree with verified native evidence")
    complete = len(outcomes) == COUNT and not pending and execution.get("returncode") == 0
    result = {"schema_version": "public-swe-multilingual-result-v1", "status": "NATIVE_FULL_SUITE_COMPLETE" if complete else "PARTIAL_NATIVE_EVIDENCE",
              "task_count": COUNT, "completed_attempts": len(outcomes), "resolved_count": len(resolved),
              "score": len(resolved) / COUNT if complete else None,
              "metric": "resolved_fraction", "outcomes": outcomes, "pending": pending,
              "native_report_sha256": digest((output / report_name).read_bytes()),
              "execution_receipt_sha256": digest(execution_path.read_bytes()),
              "task_inventory_sha256": manifest["task_inventory_sha256"], "source_revision": SOURCE_REVISION,
              "heldout_claim": False, "training_overlap_status": manifest["training_overlap_status"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("plan")
    for name in ["output", "attempts", "model-identity", "image-manifest"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800)
    p = sub.add_parser("ingest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution-receipt", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.setup)
    elif args.command == "plan":
        result = plan(args.setup, args.output, args.attempts, args.model_identity, args.image_manifest, args.run_id, args.workers, args.timeout)
    else:
        result = ingest(args.setup, args.output, args.execution_receipt)
        write(args.result, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
