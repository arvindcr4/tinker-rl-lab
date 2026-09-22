#!/usr/bin/env python3
"""Pinned LAB-Bench public evaluation with one durable request per question.

No provider work occurs in fetch-sources, prepare, or score. See native_setup/README.md.
The native LAB-Bench task classes, prompt, parser and scorer execute unchanged.
Only the parser's required definitions are loaded from its pinned wheel, avoiding
unrelated provider/model imports. Native parser exceptions block grading.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import collections
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import string
import sys
import tarfile
import tempfile
import types
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone


SOURCE_REVISION = "998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838"
SOURCE_URL = "https://github.com/Future-House/LAB-Bench"
ARCHIVE_URL = f"https://codeload.github.com/Future-House/LAB-Bench/tar.gz/{SOURCE_REVISION}"
ARCHIVE_SHA256 = "de92965813c88b0fc6e405bb55ad970501b37f761ddbfe67afca74f700922936"
WHEEL_URL = "https://files.pythonhosted.org/packages/7b/06/38e13bb3a0b9ea0a37621eec5e67e0434c6a2dc9f49bd2287a865d923acb/chembench-0.3.0-py3-none-any.whl"
WHEEL_SHA256 = "bf8bb8ec91c12e8b10ba2db26b88d0228ac464d425ac6729a6e12b7dae2d71ee"
EXPECTED_COUNTS = {"CloningScenarios": 33, "DbQA": 520, "FigQA": 181,
                   "LitQA2": 199, "ProtocolQA": 108, "SeqQA": 600,
                   "SuppQA": 82, "TableQA": 244}
EXPECTED_TOTAL = 1967
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "outputs/public_portfolio_2026-09-05/native_setup"
SCHEMA = "lab-bench-public-native-v1"


class ContractError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fingerprint(obj) -> str:
    return digest(canonical(obj))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path: Path, obj):
    """Publish fully fsynced bytes without overwriting even under competing workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(obj)
    if path.exists():
        require(path.read_bytes() == data, f"immutable artifact collision: {path}")
        return
    fd, tmp = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            require(path.read_bytes() == data, f"immutable artifact collision: {path}")
    finally:
        os.unlink(tmp)


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch_sources(setup: Path):
    """Download only pinned public benchmark code/data; no model/provider calls."""
    setup.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in [
        ("LAB-Bench-998a8e0.tar.gz", ARCHIVE_URL, ARCHIVE_SHA256),
        ("chembench-0.3.0-py3-none-any.whl", WHEEL_URL, WHEEL_SHA256),
    ]:
        target = setup / filename
        if not target.exists():
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            require(digest(data) == expected, f"source download hash mismatch: {filename}")
            with target.open("xb") as f:
                f.write(data)
        require(digest(target.read_bytes()) == expected, f"source hash mismatch: {filename}")
    root = setup / f"LAB-Bench-{SOURCE_REVISION}"
    if not root.exists():
        with tarfile.open(setup / "LAB-Bench-998a8e0.tar.gz") as archive:
            for member in archive.getmembers():
                require(member.isfile() or member.isdir(), "unsupported archive member")
                require((setup / member.name).resolve().is_relative_to(setup.resolve()), "unsafe archive path")
            archive.extractall(setup, filter="data")
    return verify_sources(setup)


def verify_sources(setup: Path):
    archive_path = setup / "LAB-Bench-998a8e0.tar.gz"
    require(digest(archive_path.read_bytes()) == ARCHIVE_SHA256, "LAB-Bench archive drift")
    root = setup / f"LAB-Bench-{SOURCE_REVISION}"
    hashes = {}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            rel = Path(member.name).relative_to(root.name)
            data = archive.extractfile(member).read()
            local = root / rel
            require(local.is_file() and not local.is_symlink(), f"source file missing or symlink: {rel}")
            expected = digest(data)
            require(digest(local.read_bytes()) == expected, f"LAB-Bench source drift: {rel}")
            hashes[str(rel)] = expected
    # Added Python or dataset files must not silently enter a pinned source import.
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    require(actual == set(hashes), "unexpected file in LAB-Bench source tree")
    wheel = setup / "chembench-0.3.0-py3-none-any.whl"
    require(digest(wheel.read_bytes()) == WHEEL_SHA256, "chembench wheel drift")
    return {"repository": SOURCE_URL, "revision": SOURCE_REVISION,
            "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
            "files_sha256": hashes, "license": "CC-BY-SA-4.0",
            "license_sha256": hashes["LICENSE"], "chembench_version": "0.3.0",
            "chembench_wheel_url": WHEEL_URL, "chembench_wheel_sha256": WHEEL_SHA256}


def _definitions(name, code, wanted, namespace):
    """Execute exact upstream AST definitions, without importing unused frameworks."""
    parsed = ast.parse(code, filename=name)
    selected = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    found = {n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else n.targets[0].id for n in selected}
    require(found == set(wanted), f"native definition missing: {set(wanted) - found}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), name, "exec"), namespace)


def load_native(setup: Path):
    root = (setup / f"LAB-Bench-{SOURCE_REVISION}").resolve()
    existing = sys.modules.get("labbench")
    if existing is not None:
        require(getattr(existing, "_portfolio_source", None) == str(root), "another labbench package is already loaded")
        return existing
    with zipfile.ZipFile(setup / "chembench-0.3.0-py3-none-any.whl") as wheel:
        require(digest((setup / "chembench-0.3.0-py3-none-any.whl").read_bytes()) == WHEEL_SHA256, "parser wheel hash mismatch")
        codes = {name: wheel.read(f"chembench/{name}.py").decode() for name in ("constant", "utils", "prompter")}
    # The selected parser has no extractor: its LLM fallback is unreachable.
    env = {"re": re, "logger": logging.getLogger("native_chembench")}
    _definitions("chembench/constant.py", codes["constant"], {"COT_PROMPT", "MCQ_REGEX_TEMPLATE_1", "LATEX_ENV_REGEX"}, env)
    names = {"run_regex", "create_multiple_choice_regex", "remove_ce", "remove_math", "remove_smiles", "remove_rxnsmiles", "remove_pu", "passthrough", "post_process_prompts"}
    _definitions("chembench/utils.py", codes["utils"], names, env)
    _definitions("chembench/prompter.py", codes["prompter"], {"prepare_mcq_answer"}, env)
    require(not any(k == "chembench" or k.startswith("chembench.") for k in sys.modules), "another chembench parser is loaded")
    for name in ("chembench", "chembench.constant", "chembench.utils", "chembench.prompter"):
        mod = types.ModuleType(name)
        mod.__dict__.update(env)
        sys.modules[name] = mod
    package = types.ModuleType("labbench")
    package.__path__ = [str(root / "labbench")]
    package._portfolio_source = str(root)
    sys.modules["labbench"] = package
    # Explicit public mode prevents environment or .env configuration changing the split.
    os.environ["PUBLIC_RELEASE"] = "True"
    utils = importlib.import_module("labbench.utils")
    utils.PUBLIC_RELEASE = True
    for name in ("AgentInput", "BaseEvalInstance", "EvalSet", "get_data_sources", "randomize_choices"):
        setattr(package, name, getattr(utils, name))
    evaluator = importlib.import_module("labbench.evaluator")
    zero_shot = importlib.import_module("labbench.zero_shot")
    for name in ("Eval", "Evaluator", "UnanswerableError"):
        setattr(package, name, getattr(evaluator, name))
    package.BaseZeroShotAgent = zero_shot.BaseZeroShotAgent
    package.zero_shot = zero_shot
    package.utils = utils
    return package


def native_instances(native, root: Path, category):
    """Strict version of native EvalSet ingestion: native validators may not skip rows."""
    module_name = f"labbench_portfolio_task_{category}"
    spec = importlib.util.spec_from_file_location(module_name, root / category / "task.py")
    task_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = task_module
    spec.loader.exec_module(task_module)
    for filename in task_module.MCQ_SOURCES:
        require(filename.endswith("-public.jsonl") and "openanswer" not in filename, "non-public MCQ source")
        for line_no, line in enumerate(Path(filename).read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row:
                continue
            yield Path(filename).stem, line_no, row, task_module.EvalInstance(**row)


def native_prompt(native, question, choices, use_cot):
    return native.zero_shot.post_process_prompts(native.zero_shot.MCQ_INSTRUCT_TEMPLATE.format(
        question=question, answers="\n".join(choices), cot="\n" + native.zero_shot.COT_PROMPT if use_cot else ""))


def runtime_versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("pydantic", "Pillow", "datasets", "python-dotenv", "tqdm"):
        versions[name] = importlib.metadata.version(name)
    return versions


def build_manifest(setup: Path, seed=809, use_cot=True):
    source = verify_sources(setup)
    native = load_native(setup)
    root = setup / f"LAB-Bench-{SOURCE_REVISION}"
    tasks, keys = [], {}
    for category in sorted(EXPECTED_COUNTS):
        for subset, line_no, row, instance in native_instances(native, root, category):
            task_id = f"{category}/{instance.id}"
            require(task_id not in keys, f"duplicate native task: {task_id}")
            # Stable per-task seeding retains native shuffle while allowing category resumes.
            rng_state = random.getstate()
            random.seed(f"{SCHEMA}:{seed}:{task_id}", version=2)
            try:
                inp, answer, unsure = instance.get_input_output()
            finally:
                random.setstate(rng_state)
            images = []
            for img in inp.figures or []:
                path = Path(img.filename).resolve()
                require(path.is_relative_to(root.resolve()), "figure outside pinned benchmark")
                mime, encoded = native.utils.encode_image(img)
                images.append({"path": str(path.relative_to(root.resolve())), "source_sha256": digest(path.read_bytes()),
                               "mime_type": mime, "encoded_sha256": digest(base64.b64decode(encoded))})
                img.close()
            prompt = native_prompt(native, inp.question, inp.choices, use_cot)
            tasks.append({"task_id": task_id, "instance_id": str(instance.id), "category": category, "subset": subset,
                          "source_line": line_no, "source_row_sha256": fingerprint(row), "question": inp.question,
                          "choices": inp.choices, "prompt": prompt, "prompt_sha256": digest(prompt.encode()), "images": images})
            # Gold metadata is kept separately and is never passed to the generation transport.
            keys[task_id] = {"target_choice": answer, "unsure_choice": unsure}
    manifest = {"schema": SCHEMA, "suite_id": "lab_bench_public_eval", "source": source,
                "split": "all released public MCQ files; private and open-answer files excluded",
                "expected_counts": EXPECTED_COUNTS, "expected_total": EXPECTED_TOTAL,
                "protocol": {"agent": "native zero-shot", "use_cot": use_cot, "choice_seed": seed,
                             "choice_randomization": "native randomize_choices with stable per-task Python seed v2",
                             "tools": [], "parser_exception_policy": "block grading; retain original sample without resampling"},
                "native_dependency_loader": "exact AST definitions from pinned chembench wheel; no extractor fallback",
                "runtime_versions": runtime_versions(), "tasks": sorted(tasks, key=lambda t: t["task_id"])}
    validate_manifest(manifest)
    return manifest, {"schema": SCHEMA, "manifest_sha256": fingerprint(manifest), "answers": keys}


def validate_manifest(manifest):
    require(manifest.get("schema") == SCHEMA, "manifest schema mismatch")
    require(manifest.get("expected_total") == EXPECTED_TOTAL and manifest.get("expected_counts") == EXPECTED_COUNTS,
            "declared denominator must be the full 1967-task public split")
    tasks = manifest["tasks"]
    require(len(tasks) == EXPECTED_TOTAL, "incomplete manifest: expected 1967 tasks")
    require(dict(collections.Counter(t["category"] for t in tasks)) == EXPECTED_COUNTS, "category denominator mismatch")
    require(len({t["task_id"] for t in tasks}) == EXPECTED_TOTAL, "duplicate manifest task ID")
    require([t["task_id"] for t in tasks] == sorted(t["task_id"] for t in tasks), "manifest task order drift")
    for t in tasks:
        require(str(uuid.UUID(t["instance_id"])) == t["instance_id"], "unsafe task instance ID")
        require(t["task_id"] == f'{t["category"]}/{t["instance_id"]}', "invalid task ID")
        require(set(t).isdisjoint({"ideal", "target_choice", "unsure_choice", "key-passage", "canary"}), "gold metadata leaked into task")
        require(digest(t["prompt"].encode()) == t["prompt_sha256"], "prompt hash drift")


def verify_manifest(setup, manifest, answer_key=None):
    validate_manifest(manifest)
    regenerated, regenerated_keys = build_manifest(setup, manifest["protocol"]["choice_seed"], manifest["protocol"]["use_cot"])
    require(manifest == regenerated, "manifest/source/prompt/runtime mismatch; use the pinned preparation environment")
    if answer_key is not None:
        require(answer_key == regenerated_keys, "answer key differs from native pinned source")


def prepare(setup, output, seed=809, use_cot=True):
    manifest, keys = build_manifest(setup, seed, use_cot)
    write_once(output / "manifest.json", manifest)
    write_once(output / "answer_key.json", keys)
    write_once(output / "source_receipt.json", manifest["source"])
    return {"stage": "PREPARED_NOT_EVALUATED", "score": None, "expected_total": EXPECTED_TOTAL,
            "manifest_sha256": fingerprint(manifest), "output": str(output)}


def validate_provenance(provenance):
    for field in ("base_model", "adapter"):
        obj = provenance.get(field, {})
        require(isinstance(obj.get("repo_id"), str) and "/" in obj["repo_id"], f"immutable HF {field} repo_id required")
        require(re.fullmatch(r"[a-f0-9]{40}", obj.get("revision", "")), f"immutable HF {field} revision required")
    require(provenance.get("adapter_loaded") is True, "actual adapter-loaded runtime evidence required; base-only forbidden")
    require(provenance.get("vision") is True, "full LAB-Bench needs a verified multimodal endpoint")
    require(provenance.get("served_model"), "served model name required")
    require(re.fullmatch(r"[a-f0-9]{64}", provenance.get("deployment_receipt_sha256", "")), "deployment receipt SHA256 required")
    require(provenance.get("deployment_verified_at"), "deployment verification timestamp required")


def validate_wandb(receipt):
    require(receipt.get("mode") == "online", "online W&B metadata required before model calls")
    require(receipt.get("initialized_before_model_work") is True, "W&B must precede model calls")
    require(receipt.get("run_id") and receipt.get("url", "").startswith("https://wandb.ai/"), "W&B run ID and URL required")


def create_contract(manifest, provenance, wandb, sampling, base_url):
    validate_manifest(manifest)
    validate_provenance(provenance)
    validate_wandb(wandb)
    require(set(sampling) == {"temperature", "top_p", "max_tokens", "seed", "n", "extra_body"}, "sampling fields must be explicit")
    require(sampling["n"] == 1, "exactly one sample per question required")
    require(isinstance(sampling["max_tokens"], int) and sampling["max_tokens"] > 0, "positive max_tokens required")
    require(isinstance(sampling["temperature"], (int, float)) and sampling["temperature"] >= 0, "invalid temperature")
    require(isinstance(sampling["top_p"], (int, float)) and 0 < sampling["top_p"] <= 1, "invalid top_p")
    require(isinstance(sampling["seed"], int), "explicit sampling seed required")
    require(isinstance(sampling["extra_body"], dict), "extra_body must be an object")
    require(not set(sampling["extra_body"]) & {"model", "messages", "n", "temperature", "top_p", "max_tokens", "seed", "stream"}, "extra_body cannot override locked request fields")
    parsed = urllib.parse.urlsplit(base_url)
    require(parsed.scheme in {"http", "https", "batch"} and parsed.netloc and not parsed.username and not parsed.query and not parsed.fragment, "invalid endpoint URL")
    require(parsed.scheme in {"https", "batch"} or parsed.hostname in {"127.0.0.1", "localhost", "::1"}, "remote endpoints must use HTTPS")
    return {"schema": SCHEMA, "manifest_sha256": fingerprint(manifest), "provenance": provenance,
            "wandb": wandb, "sampling": sampling, "base_url": base_url.rstrip("/"),
            "runner_sha256": digest(Path(__file__).read_bytes())}


def request_payload(task, contract, setup):
    from PIL import Image
    content = [{"type": "text", "text": task["prompt"]}]
    root = setup / f"LAB-Bench-{SOURCE_REVISION}"
    native = load_native(setup)
    for image in task["images"]:
        path = root / image["path"]
        require(path.resolve().is_relative_to(root.resolve()), "image path escapes pinned source")
        require(digest(path.read_bytes()) == image["source_sha256"], "source image drift")
        with Image.open(path) as img:
            mime, encoded = native.utils.encode_image(img)
        require(mime == image["mime_type"] and digest(base64.b64decode(encoded)) == image["encoded_sha256"], "encoded image drift")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "auto"}})
    sampling = contract["sampling"]
    return {"model": contract["provenance"]["served_model"], "messages": [{"role": "user", "content": content}],
            **{k: sampling[k] for k in ("temperature", "top_p", "max_tokens", "seed", "n")},
            "stream": False, **sampling["extra_body"]}


def http_completion(url, payload, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url + "/chat/completions", data=canonical(payload), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def validate_response(response, contract, request_record):
    require(response["contract_sha256"] == fingerprint(contract), "response contract collision")
    require(response["request_sha256"] == fingerprint(request_record), "response/request collision")
    raw = base64.b64decode(response["raw_body_base64"], validate=True)
    require(digest(raw) == response["raw_body_sha256"], "raw response hash mismatch")
    return raw


def perform_task(task, contract, setup, run_dir, api_key="", timeout=600, transport=http_completion):
    """An intent without a response is ambiguous and is never automatically resampled."""
    payload = request_payload(task, contract, setup)
    record = {"task_id": task["task_id"], "contract_sha256": fingerprint(contract),
              "prompt_sha256": task["prompt_sha256"], "payload": payload}
    task_dir = run_dir / "tasks" / task["category"] / task["instance_id"]
    request_path, response_path = task_dir / "request.json", task_dir / "response.json"
    task_dir.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        require(read_json(request_path) == record, f"model/prompt/sampling collision: {task['task_id']}")
        require(response_path.exists(), f"ambiguous prior request: {task['task_id']}; recover the original response, never resample")
        validate_response(read_json(response_path), contract, record)
        return False
    require(not response_path.exists(), "orphan response without request")
    # The exclusive claim occurs before any model call and remains after interruptions.
    claim = task_dir / "attempt.claim"
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ContractError(f"unresolved/concurrent claim: {task['task_id']}") from exc
    with os.fdopen(fd, "w") as f:
        f.write(now())
        f.flush()
        os.fsync(f.fileno())
    write_once(request_path, record)
    started = now()
    try:
        status, raw = transport(contract["base_url"], payload, api_key, timeout)
    except Exception as exc:
        write_once(task_dir / "transport_error.json", {"type": type(exc).__name__, "at": now(),
                   "state": "AMBIGUOUS_NO_AUTOMATIC_RETRY", "contract_sha256": fingerprint(contract)})
        raise ContractError(f"transport interrupted for {task['task_id']}; request retained, resampling blocked") from exc
    write_once(response_path, {"task_id": task["task_id"], "contract_sha256": fingerprint(contract),
               "request_sha256": fingerprint(record), "http_status": status, "started_at": started, "received_at": now(),
               "raw_body_base64": base64.b64encode(raw).decode(), "raw_body_sha256": digest(raw)})
    return True


def generate(setup, prepared, run_dir, provenance, wandb, sampling, base_url, api_key, max_new=None, category=None, timeout=600):
    require(not base_url.startswith("batch:"), "use export-requests for a batch transport")
    manifest = read_json(prepared / "manifest.json")
    verify_manifest(setup, manifest)
    contract = create_contract(manifest, provenance, wandb, sampling, base_url)
    write_once(run_dir / "run_contract.json", contract)
    n_new = 0
    for task in manifest["tasks"]:
        if category and task["category"] != category:
            continue
        if max_new is not None and n_new >= max_new:
            break
        if perform_task(task, contract, setup, run_dir, api_key, timeout):
            n_new += 1
            print(json.dumps({"event": "saved_response", "task_id": task["task_id"], "new_responses": n_new}), flush=True)
    return {"stage": "GENERATION_BATCH_FINISHED", "new_responses": n_new, "score": None, "expected_total": EXPECTED_TOTAL}


def export_requests(setup, prepared, run_dir, provenance, wandb, sampling, batch_uri, output, max_new=None, category=None):
    """Reserve a single immutable batch. Export never contacts a model.

    Execute a batch only once. Unknown execution state must be recovered using
    its original job, never by rerunning a batch or deleting its reservations.
    """
    require(batch_uri.startswith("batch://"), "batch transport identifier must start batch://")
    manifest = read_json(prepared / "manifest.json")
    verify_manifest(setup, manifest)
    contract = create_contract(manifest, provenance, wandb, sampling, batch_uri)
    write_once(run_dir / "run_contract.json", contract)
    require(not output.exists(), "batch file already exists; execute/recover that original batch, do not export it again")
    # Claim output before reserving questions. A crashed export is repairable by
    # reading its original task records, but cannot silently create a new batch.
    batch_id = fingerprint({"contract": fingerprint(contract), "output": str(output.resolve())})
    write_once(run_dir / "batches" / f"{batch_id}.json", {"batch_id": batch_id, "output": str(output.resolve()),
               "contract_sha256": fingerprint(contract), "state": "RESERVED_ONCE", "created_at": now()})
    exported = []
    for task in manifest["tasks"]:
        if category and task["category"] != category:
            continue
        if max_new is not None and len(exported) >= max_new:
            break
        directory = run_dir / "tasks" / task["category"] / task["instance_id"]
        if (directory / "request.json").exists() or (directory / "attempt.claim").exists():
            # Existing reservations, including ambiguous requests, never enter a new batch.
            continue
        payload = request_payload(task, contract, setup)
        record = {"task_id": task["task_id"], "contract_sha256": fingerprint(contract),
                  "prompt_sha256": task["prompt_sha256"], "payload": payload}
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory / "attempt.claim").open("x") as claim:
                claim.write(batch_id)
                claim.flush()
                os.fsync(claim.fileno())
        except FileExistsError as exc:
            raise ContractError(f"concurrent task reservation: {task['task_id']}") from exc
        write_once(directory / "request.json", record)
        write_once(directory / "batch_assignment.json", {"batch_id": batch_id})
        exported.append({"task_id": task["task_id"], "batch_id": batch_id,
                         "contract_sha256": fingerprint(contract), "request_sha256": fingerprint(record), "payload": payload})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        for row in exported:
            stream.write(canonical(row))
        stream.flush()
        os.fsync(stream.fileno())
    write_once(output.with_suffix(output.suffix + ".receipt.json"), {"batch_id": batch_id, "count": len(exported),
               "requests_sha256": digest(output.read_bytes()), "contract_sha256": fingerprint(contract),
               "stage": "EXPORTED_NOT_EVALUATED", "score": None})
    return {"stage": "EXPORTED_NOT_EVALUATED", "count": len(exported), "batch_id": batch_id, "output": str(output), "score": None}


def ingest_responses(prepared, run_dir, input_path):
    """Import exact raw envelopes from a once-executed GPU batch, without scoring."""
    manifest = read_json(prepared / "manifest.json")
    validate_manifest(manifest)
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    contract = read_json(run_dir / "run_contract.json")
    require(contract["manifest_sha256"] == fingerprint(manifest), "ingest manifest mismatch")
    require(contract["base_url"].startswith("batch://"), "ingest requires batch contract")
    seen = set()
    count = 0
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row["task_id"]
        require(task_id in tasks and task_id not in seen, "duplicate or unknown batch task")
        seen.add(task_id)
        task = tasks[task_id]
        directory = run_dir / "tasks" / task["category"] / task["instance_id"]
        require(read_json(directory / "batch_assignment.json")["batch_id"] == row["batch_id"], "batch assignment mismatch")
        record = read_json(directory / "request.json")
        require(row["contract_sha256"] == fingerprint(contract) and row["request_sha256"] == fingerprint(record), "batch response request/model/prompt mismatch")
        require(row.get("started_at") and row.get("received_at"), "actual batch execution timestamps required")
        raw = base64.b64decode(row["raw_body_base64"], validate=True)
        require(isinstance(row["http_status"], int), "HTTP-compatible status required")
        response = {"task_id": task_id, "contract_sha256": fingerprint(contract), "request_sha256": fingerprint(record),
                    "http_status": row["http_status"], "started_at": row["started_at"], "received_at": row["received_at"],
                    "raw_body_base64": row["raw_body_base64"], "raw_body_sha256": digest(raw)}
        validate_response(response, contract, record)
        write_once(directory / "response.json", response)
        count += 1
    return {"stage": "RESPONSES_INGESTED_NOT_SCORED", "count": count, "score": None}


async def parse_native(native, task, raw_text, use_cot):
    class ReplayAgent(native.BaseZeroShotAgent):
        async def get_completion(self, text_prompt, figs):
            require(text_prompt == task["prompt"], "native replay prompt mismatch")
            return raw_text
    agent = ReplayAgent(use_cot=use_cot)
    inp = native.AgentInput(id=task["instance_id"], question=task["question"], choices=task["choices"])
    try:
        answer = await agent.run_task(inp)
    except native.UnanswerableError:
        # Preserve only an actual native agent abstention, never a parser failure.
        return {"agent_output": None, "parser_error": None, "unanswerable": True}
    except (TypeError, ValueError, IndexError) as exc:
        raise ContractError(
            f"native parser raised {type(exc).__name__}; no native grade; original sample retained without resampling"
        ) from exc
    return {"agent_output": answer, "parser_error": None, "unanswerable": False}


def grade_response(native, task, answer_key, response, contract, record, use_cot):
    raw = validate_response(response, contract, record)
    require(response["http_status"] == 200, "non-successful HTTP response is not an evaluated sample")
    try:
        obj = json.loads(raw)
        require(obj.get("model") == contract["provenance"]["served_model"], "response served model mismatch")
        choices = obj["choices"]
        require(len(choices) == 1, "response must have exactly one completion")
        require(choices[0].get("index", 0) == 0, "unexpected response choice index")
        message = choices[0]["message"]
        require(message.get("role", "assistant") == "assistant", "invalid completion role")
        text = message["content"]
        require(isinstance(text, str), "response content must be text")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ContractError("malformed transport envelope; original sample retained without resampling") from exc
    parsed = asyncio.run(parse_native(native, task, text, use_cot))
    target, unsure = answer_key["target_choice"], answer_key["unsure_choice"]
    return {"task_id": task["task_id"], "category": task["category"], "subset": task["subset"],
            **parsed, "correct": not parsed["unanswerable"] and parsed["agent_output"] == target,
            "sure": not parsed["unanswerable"] and parsed["agent_output"] != unsure,
            "target_choice": target, "unsure_choice": unsure,
            "raw_text": text, "finish_reason": choices[0].get("finish_reason"), "usage": obj.get("usage"),
            "response_sha256": fingerprint(response)}


def summarize(native, manifest, results, blocked):
    validate_manifest(manifest)
    expected = {t["task_id"] for t in manifest["tasks"]}
    ids = [r["task_id"] for r in results]
    require(len(ids) == len(set(ids)) and set(ids) <= expected, "duplicate or unexpected scored task")
    categories = {}
    for category, count in EXPECTED_COUNTS.items():
        rows = [r for r in results if r["category"] == category]
        complete = len(rows) == count
        subsets = {}
        for subset, subset_count in collections.Counter(t["subset"] for t in manifest["tasks"] if t["category"] == category).items():
            subset_rows = [r for r in rows if r.get("subset") == subset]
            subsets[subset] = {"expected_total": subset_count, "evaluated": len(subset_rows),
                               "metrics": native.Evaluator.compute_metrics(subset_rows) if len(subset_rows) == subset_count else None}
        categories[category] = {"status": "COMPLETE" if complete else "INCOMPLETE", "expected_total": count,
                                "evaluated": len(rows), "metrics": native.Evaluator.compute_metrics(rows) if complete else None,
                                "subsets": subsets}
    complete = len(results) == EXPECTED_TOTAL and not blocked
    return {"schema": SCHEMA, "suite_id": "lab_bench_public_eval", "experiment": "E8 replacement",
            "status": "COMPLETE_PUBLIC_SPLIT" if complete else "INCOMPLETE_PUBLIC_SPLIT",
            "expected_total": EXPECTED_TOTAL, "evaluated": len(results), "missing": EXPECTED_TOTAL - len(results),
            "score": native.Evaluator.compute_metrics(results)["accuracy"] if complete else None,
            "score_metric": "native micro accuracy across all 1967 public MCQs",
            "metrics_all": native.Evaluator.compute_metrics(results) if complete else None,
            "categories": categories, "blocked_tasks": blocked,
            "claim_boundary": "Public LAB-Bench zero-shot replacement only; no claim about private LAB-Bench, LifeSciBench, original E8, or agent tool-use ability."}


def score(setup, prepared, run_dir, output):
    manifest, keys = read_json(prepared / "manifest.json"), read_json(prepared / "answer_key.json")
    verify_manifest(setup, manifest, keys)
    contract = read_json(run_dir / "run_contract.json")
    require(contract == create_contract(manifest, contract["provenance"], contract["wandb"], contract["sampling"], contract["base_url"]), "run contract drift")
    native = load_native(setup)
    results, blocked = [], []
    expected_paths = {str(Path(t["category"]) / t["instance_id"]) for t in manifest["tasks"]}
    for path in (run_dir / "tasks").glob("*/*/request.json"):
        require(str(path.parent.relative_to(run_dir / "tasks")) in expected_paths, "unexpected saved task")
    for task in manifest["tasks"]:
        directory = run_dir / "tasks" / task["category"] / task["instance_id"]
        if not (directory / "response.json").exists():
            if (directory / "request.json").exists() or (directory / "attempt.claim").exists():
                blocked.append({"task_id": task["task_id"], "reason": "ambiguous request; no resampling"})
            continue
        record = read_json(directory / "request.json")
        expected_record = {"task_id": task["task_id"], "contract_sha256": fingerprint(contract),
                           "prompt_sha256": task["prompt_sha256"], "payload": request_payload(task, contract, setup)}
        require(record == expected_record, "saved request differs from manifest/model/sampling")
        try:
            result = grade_response(native, task, keys["answers"][task["task_id"]], read_json(directory / "response.json"),
                                    contract, record, manifest["protocol"]["use_cot"])
        except ContractError as exc:
            blocked.append({"task_id": task["task_id"], "reason": str(exc)})
            continue
        write_once(directory / "native_grade.json", result)
        results.append(result)
    receipt = summarize(native, manifest, results, blocked)
    receipt.update({"manifest_sha256": fingerprint(manifest), "run_contract_sha256": fingerprint(contract),
                    "source_revision": SOURCE_REVISION, "native_scorer_sha256": manifest["source"]["files_sha256"]["labbench/evaluator.py"],
                    "provenance": contract["provenance"], "wandb": contract["wandb"], "sampling": contract["sampling"],
                    "graded_responses_sha256": fingerprint(results)})
    write_once(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-sources")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--choice-seed", type=int, default=809)
    prep.add_argument("--no-cot", action="store_true")
    gen = sub.add_parser("generate")
    for name in ("prepared", "run-dir", "provenance", "wandb", "sampling"):
        gen.add_argument(f"--{name}", type=Path, required=True)
    gen.add_argument("--base-url", required=True, help="OpenAI-compatible API base including /v1")
    gen.add_argument("--api-key-env", default="OPENAI_API_KEY")
    gen.add_argument("--max-new", type=int, help="batch limit only; the full denominator remains 1967")
    gen.add_argument("--category", choices=sorted(EXPECTED_COUNTS))
    gen.add_argument("--timeout", type=float, default=600)
    export = sub.add_parser("export-requests")
    for name in ("prepared", "run-dir", "provenance", "wandb", "sampling", "output"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--batch-uri", required=True, help="Stable executor identifier, e.g. batch://modal/pavlov-labbench")
    export.add_argument("--max-new", type=int)
    export.add_argument("--category", choices=sorted(EXPECTED_COUNTS))
    ingest = sub.add_parser("ingest-responses")
    for name in ("prepared", "run-dir", "input"):
        ingest.add_argument(f"--{name}", type=Path, required=True)
    scorer = sub.add_parser("score")
    for name in ("prepared", "run-dir", "output"):
        scorer.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch-sources":
            result = fetch_sources(args.setup)
            result = {"stage": "SOURCE_VERIFIED_NOT_EVALUATED", "source_revision": SOURCE_REVISION, "score": None}
        elif args.command == "prepare":
            result = prepare(args.setup, args.output, args.choice_seed, not args.no_cot)
        elif args.command == "generate":
            require(args.max_new is None or args.max_new >= 0, "max-new cannot be negative")
            result = generate(args.setup, args.prepared, args.run_dir, read_json(args.provenance), read_json(args.wandb),
                              read_json(args.sampling), args.base_url, os.getenv(args.api_key_env, ""), args.max_new, args.category, args.timeout)
        elif args.command == "export-requests":
            require(args.max_new is None or args.max_new >= 0, "max-new cannot be negative")
            result = export_requests(args.setup, args.prepared, args.run_dir, read_json(args.provenance), read_json(args.wandb),
                                     read_json(args.sampling), args.batch_uri, args.output, args.max_new, args.category)
        elif args.command == "ingest-responses":
            result = ingest_responses(args.prepared, args.run_dir, args.input)
        else:
            result = score(args.setup, args.prepared, args.run_dir, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "score": None, "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
