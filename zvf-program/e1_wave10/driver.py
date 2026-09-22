"""E1 wave10 execution driver for the sealed v6 recovery selection.

Built offline; NOT executed in this workspace. Default mode is validation
only (sealed-selection, preserved-context, identity and plan checks). The
lead-only ``--execute`` mode performs the v6 sequence: local generation
against a live OpenAI-compatible actor endpoint, then native evaluation by
invoking the swebench eval command shape emitted by the flagship ``plan()``
as a subprocess under ``bounded_procgroup.run_child`` (v6 process-group
contract; UNKNOWN/no-replay on any unresolved finalization).

Lane evidence boundary: request construction and the per-attempt receipt
schema are delegated to ``zvf-program/flagship/public_swe_multilingual_native.py``
(``build_actor_request`` / ``collect_attempts``); this driver never invents
prompt or receipt fields.

ADAPTATION POINTS (sibling member e1-runtime recovers
``zvf-program/e1_wave10/recovered/public_colab_runtime_fast.py``; its exact
serve flags were not available at build time, so endpoint facts come from
actor05_session/deployment.json):
  * ENDPOINT_DEFAULT -- the recovered runtime is expected to forward
    127.0.0.1:18015 -> container :8000 exactly like the actor05/actor12
    sessions (deployment.json "endpoint": http://127.0.0.1:18015/v1). Verify
    the tunnel/port before dispatch and pass --endpoint if it differs.
  * GENERATION_PARAMS -- model pavlov-public-portfolio-bf16, max_tokens 8192,
    temperature 0, seed 809, top_p 0.95 (fixed inside build_actor_request),
    chat_template_kwargs {"enable_thinking": false}.
  * WANDB_RUN_ID_DEFAULT -- receipts record the actor session's online W&B
    run; confirm against the new actor session ledger and pass --wandb-run-id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
FLAGSHIP_DIR = REPO_ROOT / "zvf-program" / "flagship"
sys.path.insert(0, str(FLAGSHIP_DIR))

import public_swe_multilingual_native as psml  # noqa: E402

import bounded_procgroup as bpg  # noqa: E402

FINISH_DIR = REPO_ROOT / "outputs/PES_Phase2_Review_2026-09-12/finish"
REQUEST_PATH = FINISH_DIR / "e1_recovery_v6/resource_request_v6.json"
REQUEST_SHA256 = "7adc6e2616f61bc7ef4d08533422a5df83ed1253f2fea507ce831890b2bb972a"
CONTEXTS_PATH = FINISH_DIR / "e1_recovery_v1/contexts_v1.json"
CONTEXTS_SHA256 = "a9a0c69cbf845b759eb9d4584ea295bb9f045e905695c12fb95544d2126c3dff"
ATTEMPTS_ROOT = FINISH_DIR / "e1_completion/continuation/wave10/attempts"
MODEL_IDENTITY_PATH = FINISH_DIR / "e1_completion/model_identity.json"
DEPLOYMENT_PATH = FINISH_DIR / "e1_completion/actor05_session/deployment.json"
SETUP_DIR = REPO_ROOT / "outputs/public_portfolio_2026-09-05/swe_multilingual_setup"
TASK_INVENTORY_SHA256 = "765d75f9384450871730901fb73933d9e68ef9415f6badcb603e116aa0829e2f"

# Sealed 16-task v6 selection: google__gson-* x9, hashicorp__terraform-* x5,
# immutable-js x2 (resource_request_v6.json selection.task_ids).
SEALED_TASK_IDS = [
    "google__gson-1014", "google__gson-1093", "google__gson-1100",
    "google__gson-2024", "google__gson-2061", "google__gson-2134",
    "google__gson-2158", "google__gson-2311", "google__gson-2479",
    "hashicorp__terraform-34580", "hashicorp__terraform-34814",
    "hashicorp__terraform-34900", "hashicorp__terraform-35543",
    "hashicorp__terraform-35611",
    "immutable-js__immutable-js-2005", "immutable-js__immutable-js-2006",
]

# ADAPTATION POINT: endpoint facts from actor05_session/deployment.json.
ENDPOINT_DEFAULT = "http://127.0.0.1:18015/v1/chat/completions"
GENERATION_PARAMS = {"max_tokens": 8192, "temperature": 0.0, "seed": 809}
WANDB_RUN_ID_DEFAULT = "actor-rpc-public0919-e1-actor11"
WANDB_MODE = "online"
REQUEST_TIMEOUT_SECONDS = 190  # budget_audit.per_pair_request_timeout
NATIVE_WORKERS = 4
NATIVE_TIMEOUT_SECONDS = 1800
NATIVE_OUTER_WALL_SECONDS = 7500
RUN_ID_DEFAULT = "e1multilingual0919wave10"
SELECTION_LABEL = ("sealed v6 recovery selection (resource_request_v6.json): "
                   "gson x9, terraform x5, immutable-js x2")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_line(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode() + b"\n"


def pretty(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode()


def write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"Refusing to replace existing artifact: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def extract_patch(content: str) -> tuple[str, str] | None:
    """Return (patch, extraction_reason) for a valid unified diff, else None.

    Mirrors the original E1 convention recorded in generation.json
    ("extraction_reason": "valid unified diff"). Fence recovery only accepts a
    block that is itself a diff; anything else is a genuine failed generation.
    """
    text = content.strip()
    if text.startswith("diff --git"):
        return text, "valid unified diff"
    for match in re.finditer(r"```(?:diff)?[ \t]*\n(.*?)```", content, re.S):
        body = match.group(1).strip()
        if body.startswith("diff --git"):
            return body, "valid unified diff inside markdown fence"
    return None


def load_selection(request_path: Path) -> list[str]:
    raw = request_path.read_bytes()
    if sha256_bytes(raw) != REQUEST_SHA256:
        raise RuntimeError(f"Sealed v6 request hash mismatch: {request_path}")
    request = json.loads(raw)
    task_ids = request["selection"]["task_ids"]
    if task_ids != SEALED_TASK_IDS or request["selection"]["count"] != len(SEALED_TASK_IDS):
        raise RuntimeError("Sealed task selection does not match the embedded v6 list")
    return task_ids


def load_wave_plan(task_ids: list[str],
                   attempts_root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Load preserved per-task source contexts and digest-pinned images.

    Images come from the sealed contexts_v1.json (sha256-pinned by
    resource_request_v6.contexts_manifest): the pinned suite's raw rows carry
    mutable :latest tags, so the registry digests observed during context
    collection are the immutable per-task image references. Each digest is
    cross-checked against the attempt's image_inspect.json RepoDigests.
    """
    raw = CONTEXTS_PATH.read_bytes()
    if sha256_bytes(raw) != CONTEXTS_SHA256:
        raise RuntimeError(f"Sealed contexts manifest hash mismatch: {CONTEXTS_PATH}")
    contexts_doc = json.loads(raw)
    rows = psml.indexed(contexts_doc["tasks"])
    if set(rows) != set(task_ids):
        raise RuntimeError("Sealed contexts do not cover exactly the 16 selected tasks")
    contexts, images = {}, {}
    for iid in task_ids:
        source = psml.read(attempts_root / iid / "source_context.json")
        if set(source) != {"actor_task", "base_commit", "files", "repo"}:
            raise RuntimeError(f"Unexpected source context fields: {iid}")
        if set(source["actor_task"]) != set(psml.ACTOR_FIELDS):
            raise RuntimeError(f"Source context actor task projection changed: {iid}")
        if source["base_commit"] != rows[iid]["base_commit"]:
            raise RuntimeError(f"Source context base commit drift: {iid}")
        contexts[iid] = source
        image = rows[iid]["image"]
        if "@sha256:" not in image:
            raise RuntimeError(f"Contexts image is not digest-pinned: {iid}")
        inspect_rows = psml.read(attempts_root / iid / "image_inspect.json")
        repo_digests = inspect_rows.get("RepoDigests") or []
        if repo_digests and image not in repo_digests:
            raise RuntimeError(f"Image digest disagrees with inspect receipt: {iid}")
        images[iid] = image
    # Cross-check the preserved actor task projection against the prepared
    # 300-task actor projection when available (pure JSONL read, no pyarrow).
    prepared = SETUP_DIR / "prepared/actor_tasks.jsonl"
    if prepared.is_file():
        prepared_rows = psml.indexed(psml.lines(prepared))
        for iid in task_ids:
            if psml.object_hash(contexts[iid]["actor_task"]) != psml.object_hash(prepared_rows[iid]):
                raise RuntimeError(f"Preserved context disagrees with prepared actor task: {iid}")
    return contexts, images


def load_native_rows(task_ids: list[str]) -> dict[str, dict]:
    """Raw pinned rows for the wave from the prepared 300-task dataset."""
    rows = psml.indexed(psml.lines(SETUP_DIR / "prepared/native_dataset.jsonl"))
    return {iid: rows[iid] for iid in task_ids}


def post_chat_completion(endpoint: str, request_bytes: bytes, api_key: str | None,
                          timeout: float) -> tuple[dict, dict]:
    """POST one actor request; returns (body, meta).

    Transport failures raise; the caller must abort the wave rather than
    fabricate a GENERATION_FAILED receipt (provider errors stay blocked under
    the lane evidence boundary).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=request_bytes, headers=headers,
                                 method="POST")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status, reason = response.status, response.reason
            response_headers = [[k, v] for k, v in response.getheaders()]
            body_bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:4000].decode("utf-8", "replace")
        raise RuntimeError(f"Actor endpoint HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Actor endpoint transport failure: {exc}") from exc
    try:
        body = json.loads(body_bytes)
    except ValueError as exc:
        raise RuntimeError("Actor endpoint returned non-JSON body") from exc
    meta = {"id": body.get("id"), "status": status, "reason": reason,
            "headers": response_headers, "elapsed_seconds": time.time() - started}
    return body, meta


def generate_one(iid: str, source: dict, identity: dict, args, attempts_root: Path,
                 log) -> dict:
    """One generation attempt; writes receipts in the attempts/*/ schema.

    Receipt fields mirror outputs/.../e1_completion/attempts/*/generation.json
    exactly (instance_id, status GENERATED/GENERATION_FAILED, sample_started,
    sample_completed, task_inventory_sha256, actor_task_sha256,
    model_identity_sha256, wandb_run_id/mode, started_at/finished_at,
    response_sha256, source_sha256, request_sha256, patch, patch_sha256, plus
    the recorded finish_reason/extraction_reason/usage/deployment_ref).
    """
    directory = attempts_root / iid
    actor = {key: source["actor_task"][key] for key in psml.ACTOR_FIELDS}
    request = psml.build_actor_request(actor, source["files"],
                                       identity["served_model_id"],
                                       max_tokens=args.max_tokens,
                                       temperature=args.temperature,
                                       seed=args.seed)
    request_bytes = canonical_line(request)
    request_path = directory / "generation_request.json"
    write_immutable(request_path, request_bytes)

    started_at = time.time()
    body, meta = post_chat_completion(args.endpoint, request_bytes, args.api_key,
                                      args.request_timeout)
    finished_at = time.time()
    write_immutable(directory / "generation_http_meta.json", pretty(meta))
    write_immutable(directory / "generation_http_response.json",
                    pretty(body))
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    content = message.get("content")
    finish_reason = choices[0].get("finish_reason") if choices else None
    extracted = extract_patch(content) if isinstance(content, str) else None
    if extracted is None:
        status, patch = "GENERATION_FAILED", ""
        extraction_reason = ("no valid unified diff in completed response"
                             if choices else "empty choices in completed response")
    else:
        patch, extraction_reason = extracted
        status = "GENERATED"
    response_path = directory / "generation_response.txt"
    response_bytes = (content or "").encode()
    write_immutable(response_path, response_bytes)
    if status == "GENERATED" and patch.strip() not in response_bytes.decode("utf-8", "replace"):
        raise RuntimeError(f"Candidate patch absent from retained response: {iid}")

    receipt = {
        "instance_id": iid,
        "status": status,
        "sample_started": True,
        "sample_completed": True,
        "task_inventory_sha256": TASK_INVENTORY_SHA256,
        "actor_task_sha256": psml.object_hash(actor),
        "model_identity_sha256": psml.object_hash(identity),
        "wandb_run_id": args.wandb_run_id,
        "wandb_mode": WANDB_MODE,
        "started_at": started_at,
        "finished_at": finished_at,
        "response_sha256": sha256_bytes(response_bytes),
        "source_sha256": psml.object_hash(source["files"]),
        "request_sha256": sha256_bytes(request_bytes),
        "patch": patch,
        "patch_sha256": sha256_bytes(patch.encode()),
        "finish_reason": finish_reason,
        "extraction_reason": extraction_reason,
        "usage": body.get("usage") or {},
        "deployment_ref": {"path": str(args.deployment),
                           "sha256": sha256_bytes(Path(args.deployment).read_bytes())},
    }
    write_immutable(directory / "generation.json", pretty(receipt))
    write_immutable(directory / "generation_intent.json", pretty({
        "at": started_at, "replay_forbidden": True, "endpoint": args.endpoint,
        "request_ref": {"path": str(request_path),
                        "sha256": receipt["request_sha256"]},
        "deployment_ref": receipt["deployment_ref"]}))
    log(f"{iid} {status} {int((finished_at - started_at) * 1000)}")
    return receipt


def build_native_argv(args, output_dir: Path) -> list[str]:
    """The swebench eval command shape emitted by flagship plan().

    plan() emits: <setup>/.venv/bin/swebench eval <runtime_dataset.jsonl>
    -p <predictions.jsonl> --run-id <id> --split test -j <workers>
    --timeout <seconds> --report-dir <output>/reports. This driver points
    every path at the wave's 16-task materialization.
    """
    swebench = args.swebench_bin or str((SETUP_DIR / ".venv/bin/swebench").resolve())
    return [swebench, "eval", str(output_dir / "wave10_native_dataset.jsonl"),
            "-p", str(output_dir / "wave10_predictions.jsonl"),
            "--run-id", args.run_id, "--split", "test",
            "-j", str(args.workers), "--timeout", str(args.timeout),
            "--report-dir", str(output_dir / "reports")]


def native_worker() -> int:
    """Hidden worker mode: run the eval argv inside the owned process group.

    The swebench child inherits this worker's process group (no new session),
    so the controller's killpg contains it. Exits 0 iff the result receipt was
    written; the eval returncode is data inside the receipt, never the exit
    status, so an infra-failed eval cannot masquerade as a lost worker.
    """
    flags = {}
    for name in ("--native-result-path", "--native-stdout", "--native-stderr"):
        flags[name] = sys.argv[sys.argv.index(name) + 1]
    argv = sys.argv[sys.argv.index("--") + 1:]
    stdout_path, stderr_path = Path(flags["--native-stdout"]), Path(flags["--native-stderr"])
    result_path = Path(flags["--native-result-path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            completed = subprocess.run(argv, stdout=out, stderr=err, check=False)
            returncode = completed.returncode
        except OSError as exc:
            err.write(f"native worker spawn failure: {exc}\n".encode())
            returncode = None
    bpg.write_durable(result_path, {
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": time.time(),
        "argv": argv,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    })
    return 0


def run_native_phase(args, output_dir: Path, log) -> dict:
    eval_argv = build_native_argv(args, output_dir)
    receipt_dir = output_dir / "native_group"
    result_path = receipt_dir / "native_result.json"
    worker_argv = [sys.executable, str(Path(__file__).resolve()), "--native-worker",
                   "--native-result-path", str(result_path),
                   "--native-stdout", str(receipt_dir / "native_stdout.txt"),
                   "--native-stderr", str(receipt_dir / "native_stderr.txt"),
                   "--"] + eval_argv
    deadline_epoch = time.time() + args.native_outer_wall
    outcome = bpg.run_child(worker_argv, result_path=result_path,
                            receipt_dir=receipt_dir,
                            deadline_epoch=deadline_epoch)
    if outcome["status"] != "SUCCESS":
        log(f"NATIVE_UNKNOWN {outcome['unknown_reason']}")
        return {"status": "UNKNOWN", "no_replay": True,
                "reason": outcome["unknown_reason"],
                "group_finalization": outcome["group_finalization"]}
    result = outcome["result"]
    log(f"NATIVE_EXIT {result.get('returncode')}")
    return {"status": "EXITED", "returncode": result.get("returncode"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
            "argv": result.get("argv"),
            "group_finalization": outcome["group_finalization"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="lead-only paid execution; default validates without dispatch")
    parser.add_argument("--endpoint", default=ENDPOINT_DEFAULT)
    parser.add_argument("--api-key", default=None,
                        help="optional bearer token for the actor endpoint")
    parser.add_argument("--request", type=Path, default=REQUEST_PATH)
    parser.add_argument("--attempts-root", type=Path, default=ATTEMPTS_ROOT)
    parser.add_argument("--model-identity", type=Path, default=MODEL_IDENTITY_PATH)
    parser.add_argument("--deployment", type=Path, default=DEPLOYMENT_PATH)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "run")
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--wandb-run-id", default=WANDB_RUN_ID_DEFAULT)
    parser.add_argument("--max-tokens", type=int,
                        default=GENERATION_PARAMS["max_tokens"])
    parser.add_argument("--temperature", type=float,
                        default=GENERATION_PARAMS["temperature"])
    parser.add_argument("--seed", type=int, default=GENERATION_PARAMS["seed"])
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--workers", type=int, default=NATIVE_WORKERS)
    parser.add_argument("--timeout", type=int, default=NATIVE_TIMEOUT_SECONDS)
    parser.add_argument("--native-outer-wall", type=float,
                        default=NATIVE_OUTER_WALL_SECONDS)
    parser.add_argument("--swebench-bin", default=None)
    # Hidden worker mode (internal; invoked by run_native_phase under the
    # bounded process group; everything after "--" is the eval argv).
    parser.add_argument("--native-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_known_args(argv)


def main() -> int:
    args, extra = parse_args()
    if args.native_worker:
        if extra:
            print(f"native worker: unrecognized pre-separator arguments: {extra}",
                  file=sys.stderr)
            return 2
        return native_worker()
    if extra:
        print(f"unrecognized arguments: {extra}", file=sys.stderr)
        return 2

    task_ids = load_selection(args.request)
    identity = psml.read(args.model_identity)
    psml.validate_identity(identity)
    contexts, images = load_wave_plan(task_ids, args.attempts_root)
    native_rows = load_native_rows(task_ids)

    plan_summary = {
        "status": "VALIDATED_NOT_DISPATCHED" if not args.execute else "EXECUTING",
        "run_id": args.run_id,
        "selection": SELECTION_LABEL,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_inventory_sha256": TASK_INVENTORY_SHA256,
        "model_identity_sha256": psml.object_hash(identity),
        "endpoint": args.endpoint,
        "generation_params": {"max_tokens": args.max_tokens,
                              "temperature": args.temperature,
                              "seed": args.seed, "top_p": 0.95,
                              "chat_template_kwargs": {"enable_thinking": False}},
        "wandb_run_id": args.wandb_run_id, "wandb_mode": WANDB_MODE,
        "native": {"workers": args.workers, "timeout": args.timeout,
                   "outer_wall": args.native_outer_wall},
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan_summary, indent=2, sort_keys=True))
    if not args.execute:
        return 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "wave10.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        log_file.write(message + "\n")
        log_file.flush()

    predictions, wave_tasks = [], []
    for iid in task_ids:
        receipt = generate_one(iid, contexts[iid], identity, args,
                               args.attempts_root, log)
        predictions.append({"instance_id": iid,
                            "model_name_or_path": identity["served_model_id"],
                            "model_patch": receipt["patch"]})
        wave_tasks.append({"actor": contexts[iid]["actor_task"], "image": images[iid]})

    write_immutable(output_dir / "wave10_predictions.jsonl",
                    b"".join(canonical_line(row) for row in predictions))
    write_immutable(output_dir / "wave10.json", pretty({
        "denominator": 300, "run_id": args.run_id, "selection": SELECTION_LABEL,
        "task_inventory_sha256": TASK_INVENTORY_SHA256,
        "sealed_request_ref": {"path": str(args.request),
                               "sha256": REQUEST_SHA256},
        "tasks": wave_tasks}))
    # Materialize the runtime dataset exactly as plan() does: raw pinned rows
    # with image references replaced by observed immutable digests.
    runtime_rows = [{**native_rows[iid], "image": images[iid]} for iid in task_ids]
    write_immutable(output_dir / "wave10_native_dataset.jsonl",
                    b"".join(canonical_line(row) for row in runtime_rows))

    native_receipt = run_native_phase(args, output_dir, log)
    write_immutable(output_dir / "wave10_native_execution.json",
                    pretty(native_receipt))
    log_file.close()
    return 0 if native_receipt.get("status") == "EXITED" else 3


if __name__ == "__main__":
    sys.exit(main())
