"""Run the public E1 SWE-bench Pro subset through native Modal Sandboxes.

The runner deliberately evaluates one frozen task.  It never labels that result
as the 731-task suite score.  The model receives only the task statement,
requirements, and base-checkout source context; the dataset gold patch and test
patch are removed before inference.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from modal.exception import SandboxFilesystemNotFoundError


APP_NAME = "pavlov-e1-swe-bench-pro"
RUN_DATE = "2026-08-16"
INSTANCE_ID = "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
IMAGE_URI = (
    "jefzda/sweap-images:"
    "nodebb.nodebb-NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5"
)
IMAGE_DIGEST = "sha256:e49637ebe82a479ca43b4663525955bc9cdd58c457140ee31c20958d621d3cf7"
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SAMPLER_PATH = (
    "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/"
    "sampler_weights/seed809_final"
)
HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-"
    "tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
HF_REVISION = "checkpoint-seed809-stepfinal-9f777c4018b6"
HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"

USD_PER_M_PREFILL = 0.54
USD_PER_M_SAMPLE = 1.335
MAX_TINKER_USD = 0.20

SOURCE_SLICES: tuple[tuple[str, int | None, int | None], ...] = (
    ("public/language/en-GB/admin/manage/users.json", None, None),
    ("public/language/en-GB/error.json", None, None),
    ("public/openapi/components/schemas/UserObject.yaml", 600, 710),
    ("src/controllers/admin/users.js", 130, 215),
    ("src/database/mongo/main.js", 1, 150),
    ("src/database/postgres/main.js", 70, 175),
    ("src/database/redis/main.js", 1, 115),
    ("src/socket.io/admin/user.js", 1, 125),
    ("src/user/delete.js", 120, 175),
    ("src/user/email.js", 1, 230),
    ("src/views/admin/manage/users.tpl", 80, 145),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_stream(stream: Any) -> str:
    if stream is None or not hasattr(stream, "read"):
        return ""
    value = stream.read()
    return value if isinstance(value, str) else str(value or "")


def _lookup_app() -> modal.App:
    return modal.App.lookup(APP_NAME, create_if_missing=True)


def _new_sandbox(*, timeout: int) -> modal.Sandbox:
    return modal.Sandbox.create(
        image=modal.Image.from_registry(IMAGE_URI),
        app=_lookup_app(),
        timeout=timeout,
        cpu=(2.0, 4.0),
        memory=(8 * 1024, 30 * 1024),
    )


def _write_sandbox_file(sandbox: modal.Sandbox, path: str, content: str) -> None:
    parent = str(Path(path).parent)
    sandbox.filesystem.make_directory(parent)
    sandbox.filesystem.write_text(content, path)


def _read_sandbox_text_optional(sandbox: modal.Sandbox, path: str) -> str:
    try:
        return sandbox.filesystem.read_text(path)
    except SandboxFilesystemNotFoundError:
        return ""


def _snapshot_sources(base_commit: str) -> dict[str, str]:
    sandbox = _new_sandbox(timeout=20 * 60)
    try:
        reset = sandbox.exec(
            "bash",
            "-lc",
            f"cd /app && git reset --hard {shlex.quote(base_commit)} && git clean -fd",
        )
        reset.wait()
        if reset.returncode != 0:
            raise RuntimeError("official E1 image could not reset to the frozen base commit")
        result: dict[str, str] = {}
        for relative, start, end in SOURCE_SLICES:
            quoted = shlex.quote(relative)
            if start is None:
                command = f"cd /app && cat {quoted}"
            else:
                command = f"cd /app && sed -n '{start},{end}p' {quoted}"
            process = sandbox.exec("bash", "-lc", command)
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"failed to read frozen source context: {relative}")
            result[relative] = _read_stream(process.stdout)
        return result
    finally:
        sandbox.terminate()


def _extract_diff(response: str) -> str:
    match = re.search(r"(?m)^diff --git ", response)
    if match is None:
        return ""
    patch = response[match.start() :].strip()
    patch = re.sub(r"\n```(?:[a-zA-Z0-9_-]+)?\s*$", "", patch).strip()
    patch = patch + "\n" if patch else ""
    valid, _ = _validate_unified_diff(patch)
    return patch if valid else ""


def _validate_unified_diff(patch: str) -> tuple[bool, str]:
    """Reject prose, fenced output, and placeholder hunks before evaluation."""

    if not patch.startswith("diff --git "):
        return False, "patch must begin with a diff --git header"
    if "```" in patch:
        return False, "patch contains a Markdown fence"

    blocks = re.split(r"(?m)(?=^diff --git )", patch.rstrip("\n"))
    blocks = [block for block in blocks if block]
    if not blocks:
        return False, "patch contains no file diff"

    concrete_hunk = re.compile(
        r"(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$"
    )
    for block in blocks:
        lines = block.splitlines()
        header = lines[0]
        if not re.fullmatch(r"diff --git a/\S+ b/\S+", header):
            return False, f"invalid file header: {header}"
        if not re.search(r"(?m)^--- (?:a/\S+|/dev/null)$", block):
            return False, f"missing old-file marker for {header}"
        if not re.search(r"(?m)^\+\+\+ (?:b/\S+|/dev/null)$", block):
            return False, f"missing new-file marker for {header}"
        if concrete_hunk.search(block) is None:
            return False, f"missing concrete hunk header for {header}"

        in_hunk = False
        metadata_prefixes = (
            "index ",
            "new file mode ",
            "deleted file mode ",
            "old mode ",
            "new mode ",
            "similarity index ",
            "rename from ",
            "rename to ",
            "--- ",
            "+++ ",
            "Binary files ",
        )
        for line in lines[1:]:
            if concrete_hunk.fullmatch(line):
                in_hunk = True
                continue
            if not in_hunk:
                if line.startswith(metadata_prefixes):
                    continue
                return False, f"invalid patch metadata line: {line}"
            if line.startswith("@@ "):
                return False, f"invalid hunk header: {line}"
            if line and not line.startswith((" ", "+", "-", "\\")):
                return False, f"invalid patch line: {line}"

    if not re.search(r"(?m)^[+-](?![+-]{2} )", patch):
        return False, "patch contains no changed lines"
    return True, "valid unified diff"


def _build_prompt(task: dict[str, Any], sources: dict[str, str]) -> str:
    source_text = "\n\n".join(
        f"===== {path} =====\n{content}" for path, content in sorted(sources.items())
    )
    return f"""You are solving one frozen SWE-bench Pro task in the NodeBB repository.

Return only a valid unified git diff beginning with `diff --git`. Do not include
explanations, planning text, ellipses, abbreviated context, or Markdown fences.
Every hunk header must contain real line ranges such as `@@ -12,7 +12,9 @@`;
placeholder headers such as `@@ ... @@` are invalid. Include every changed file
in full unified-diff form. Make the smallest complete production fix. Preserve
the public APIs and project style. The patch must apply to base commit
{task['base_commit']} and satisfy the task requirements.

PROBLEM STATEMENT
{task['problem_statement']}

REQUIREMENTS
{task['requirements']}

FROZEN BASE-CHECKOUT SOURCE CONTEXT
{source_text}
"""


generation_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface-hub==1.27.0",
    "jinja2==3.1.6",
    "tinker==0.24.1",
    "transformers==5.5.4",
    "wandb==0.21.0",
)
secret = modal.Secret.from_name("pavlov-e1-e14")
app = modal.App(APP_NAME, include_source=True)


@app.function(
    image=generation_image,
    secrets=[secret],
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    retries=0,
)
def generate_patch(
    task: dict[str, Any],
    sources: dict[str, str],
    *,
    seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Generate exactly one candidate; no retry is allowed for pass@1."""

    if any(key in task for key in ("patch", "test_patch")):
        raise RuntimeError("gold-bearing fields reached the model boundary")
    if task.get("instance_id") != INSTANCE_ID:
        raise RuntimeError("unexpected E1 instance")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.model_info(HF_REPO, revision=HF_REVISION)
    if info.sha != HF_COMMIT:
        raise RuntimeError(f"immutable HF checkpoint drift: {info.sha}")

    prompt = _build_prompt(task, sources)

    import tinker
    import tinker.types as T
    import wandb
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompt_ids = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        add_special_tokens=False,
    )
    projected_usd = (
        len(prompt_ids) / 1e6 * USD_PER_M_PREFILL
        + max_tokens / 1e6 * USD_PER_M_SAMPLE
    )
    if projected_usd > MAX_TINKER_USD:
        raise RuntimeError(
            f"projected Tinker spend ${projected_usd:.6f} exceeds ${MAX_TINKER_USD:.2f}"
        )

    run = wandb.init(
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group="pavlov-e1-e14-modal-20260816",
        job_type="primary-evaluation-subset",
        name=f"e1_swe_bench_pro_nodebb_modal_seed{seed}",
        tags=["e1", "swe_bench_pro", "subset", "pass@1", "modal"],
        mode="online",
        config={
            "suite_id": "swe_bench_pro_eval",
            "suite_role": "primary_eval",
            "scope": "one_task_subset_not_suite_score",
            "instance_id": INSTANCE_ID,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "evaluator_revision": EVALUATOR_REVISION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": SAMPLER_PATH,
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "hf_commit": HF_COMMIT,
            "samples_per_problem": 1,
            "sampling_retries": 0,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "projected_tinker_usd": projected_usd,
        },
        reinit=True,
    )
    if run is None or not getattr(run, "id", None):
        raise RuntimeError("W&B online initialization failed before Tinker")

    service = tinker.ServiceClient(
        user_metadata={
            "campaign": "pavlov-e1-e14-modal",
            "suite_id": "swe_bench_pro_eval",
            "wandb_run_id": run.id,
        }
    )
    sampler = service.create_sampling_client(model_path=SAMPLER_PATH)
    response = sampler.sample(
        T.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=T.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            seed=seed,
        ),
    ).result()
    tokens = list(response.sequences[0].tokens)
    response_text = tokenizer.decode(tokens, skip_special_tokens=True)
    extracted_patch_match = re.search(r"(?m)^diff --git ", response_text)
    extracted_patch = (
        response_text[extracted_patch_match.start() :].strip() + "\n"
        if extracted_patch_match is not None
        else ""
    )
    patch_valid, patch_validation_reason = _validate_unified_diff(extracted_patch)
    patch = extracted_patch if patch_valid else ""
    estimated_usd = (
        len(prompt_ids) / 1e6 * USD_PER_M_PREFILL
        + len(tokens) / 1e6 * USD_PER_M_SAMPLE
    )
    run.log(
        {
            "generation/prompt_tokens": len(prompt_ids),
            "generation/response_tokens": len(tokens),
            "generation/patch_extracted": int(bool(patch)),
            "generation/patch_structurally_valid": int(patch_valid),
            "cost/estimated_actual_usd": estimated_usd,
        },
        step=1,
    )
    run.summary.update(
        {
            "status": "GENERATED" if patch else "GENERATION_FAILED",
            "candidate_patch_sha256": _sha256_text(patch) if patch else None,
        }
    )
    run_id = run.id
    run_url = run.url
    run.finish(exit_code=0 if patch else 1)
    return {
        "patch": patch,
        "response_text": response_text,
        "patch_sha256": _sha256_text(patch) if patch else None,
        "patch_validation_reason": patch_validation_reason,
        "response_sha256": _sha256_text(response_text),
        "prompt_sha256": _sha256_text(prompt),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(tokens),
        "estimated_tinker_usd": round(estimated_usd, 6),
        "wandb_run_id": run_id,
        "wandb_url": run_url,
        "generated_at": _utc_now(),
    }


@app.function(
    image=generation_image,
    secrets=[secret],
    cpu=1.0,
    memory=2048,
    timeout=5 * 60,
    retries=0,
)
def fetch_generation_evidence(run_id: str) -> str:
    """Read immutable generation metadata for evaluator-only recovery."""

    import wandb

    run = wandb.Api().run(
        f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run_id}"
    )
    return json.dumps(
        {
            "config": dict(run.config),
            "summary": dict(run.summary),
            "state": run.state,
            "url": run.url,
        },
        default=str,
        sort_keys=True,
    )


def _evaluate_candidate(
    *,
    task: dict[str, Any],
    patch: str,
    run_script: str,
    parser_script: str,
) -> dict[str, Any]:
    sandbox = _new_sandbox(timeout=60 * 60)
    try:
        _write_sandbox_file(sandbox, "/workspace/patch.diff", patch)
        _write_sandbox_file(sandbox, "/workspace/run_script.sh", run_script)
        _write_sandbox_file(sandbox, "/workspace/parser.py", parser_script)

        selected = ast.literal_eval(task["selected_test_files_to_run"])
        selected_arg = ",".join(selected)
        before_last = task["before_repo_set_cmd"].strip().splitlines()[-1]
        command = "\n".join(
            [
                "set -e",
                "cd /app",
                f"git reset --hard {shlex.quote(task['base_commit'])}",
                "git clean -fd",
                f"git checkout {shlex.quote(task['base_commit'])}",
                "git apply --check /workspace/patch.diff",
                "git apply -v /workspace/patch.diff",
                before_last,
                (
                    "bash /workspace/run_script.sh "
                    f"{shlex.quote(selected_arg)} > /workspace/stdout.log "
                    "2> /workspace/stderr.log"
                ),
                (
                    "python /workspace/parser.py /workspace/stdout.log "
                    "/workspace/stderr.log /workspace/output.json"
                ),
            ]
        )
        process = sandbox.exec("bash", "-lc", command)
        process.wait()
        stdout = _read_stream(process.stdout)
        stderr = _read_stream(process.stderr)

        output: dict[str, Any] = {"tests": []}
        output_text = _read_sandbox_text_optional(sandbox, "/workspace/output.json")
        if output_text:
            output = json.loads(output_text)
        test_stdout = _read_sandbox_text_optional(sandbox, "/workspace/stdout.log")
        test_stderr = _read_sandbox_text_optional(sandbox, "/workspace/stderr.log")

        passed = {
            item.get("name")
            for item in output.get("tests", [])
            if item.get("status") == "PASSED"
        }
        fail_to_pass = set(ast.literal_eval(task["fail_to_pass"]))
        pass_to_pass = set(ast.literal_eval(task["pass_to_pass"]))
        required = fail_to_pass | pass_to_pass
        resolved = process.returncode == 0 and required <= passed
        return {
            "resolved": resolved,
            "entryscript_returncode": process.returncode,
            "parsed_test_count": len(output.get("tests", [])),
            "passed_test_count": len(passed),
            "required_test_count": len(required),
            "fail_to_pass_count": len(fail_to_pass),
            "pass_to_pass_count": len(pass_to_pass),
            "missing_required_tests": sorted(required - passed),
            "process_stdout_tail": stdout[-4000:],
            "process_stderr_tail": stderr[-4000:],
            "test_stdout_sha256": _sha256_text(test_stdout),
            "test_stderr_sha256": _sha256_text(test_stderr),
            "test_stdout_tail": test_stdout[-8000:],
            "test_stderr_tail": test_stderr[-8000:],
        }
    finally:
        sandbox.terminate()


@app.local_entrypoint()
def main(
    seed: int = 1817,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    resume: bool = False,
) -> None:
    root = _repo_root()
    run_dir = root / f"outputs/modal_e1_e14/{RUN_DATE}/e1_swe_bench_pro/seed{seed}"
    if resume and not run_dir.is_dir():
        raise RuntimeError(f"cannot resume missing E1 attempt: {run_dir}")
    if not resume and run_dir.exists():
        raise RuntimeError(f"refusing to overwrite an existing E1 attempt: {run_dir}")
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    row_path = root / "outputs/e1_swe_bench_pro/selected_sample_nodebb.jsonl"
    task_with_gold = json.loads(row_path.read_text(encoding="utf-8"))
    task = {
        key: value
        for key, value in task_with_gold.items()
        if key not in {"patch", "test_patch"}
    }
    if set(task_with_gold) - set(task) != {"patch", "test_patch"}:
        raise RuntimeError("unexpected E1 gold-field sanitization result")

    sources = _snapshot_sources(str(task["base_commit"]))
    response_path = run_dir / "generation_response.txt"
    candidate_path = run_dir / "candidate_patch.json"
    if resume:
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        if len(candidate_payload) != 1:
            raise RuntimeError("recovery candidate must contain exactly one patch")
        candidate = candidate_payload[0]
        if candidate.get("instance_id") != INSTANCE_ID:
            raise RuntimeError("recovery candidate has the wrong instance")
        response_text = response_path.read_text(encoding="utf-8")
        patch = str(candidate.get("patch") or "")
        if _extract_diff(response_text) != patch:
            raise RuntimeError("saved response and candidate patch disagree")
        run_id = str(candidate["generation_run_id"])
        evidence = json.loads(fetch_generation_evidence.remote(run_id))
        config = evidence["config"]
        summary = evidence["summary"]
        expected_config = {
            "instance_id": INSTANCE_ID,
            "model_revision": MODEL_REVISION,
            "hf_commit": HF_COMMIT,
            "sampler_path": SAMPLER_PATH,
            "seed": seed,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(f"W&B recovery config mismatch for {key}")
        if summary.get("status") != "GENERATED":
            raise RuntimeError("W&B recovery run is not a completed generation")
        generation = {
            "patch": patch,
            "response_text": response_text,
            "patch_sha256": _sha256_text(patch),
            "response_sha256": _sha256_text(response_text),
            "prompt_sha256": _sha256_text(_build_prompt(task, sources)),
            "prompt_tokens": int(summary["generation/prompt_tokens"]),
            "response_tokens": int(summary["generation/response_tokens"]),
            "estimated_tinker_usd": float(summary["cost/estimated_actual_usd"]),
            "wandb_run_id": run_id,
            "wandb_url": evidence["url"],
        }
    else:
        generation = generate_patch.remote(
            task,
            sources,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_path.write_text(str(generation["response_text"]), encoding="utf-8")
    patch = str(generation.get("patch") or "")
    if not patch:
        receipt: dict[str, Any] = {
            "schema_version": "pavlov-modal-e1-swe-bench-pro-subset-v1",
            "recorded_at": _utc_now(),
            "lane": "E1",
            "suite_id": "swe_bench_pro_eval",
            "suite_role": "primary_eval",
            "status": "GENERATION_FAILED",
            "score": None,
            "is_model_score": False,
            "scope": "one_task_subset_not_suite_score",
            "claim_boundary": (
                "This pass@1 attempt produced no unified diff and therefore no model "
                "score. It is not the 731-task suite score."
            ),
            "instance_id": INSTANCE_ID,
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": "test",
                "observed_license_state": "absent_at_pinned_revision",
                "proceeding_under": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
            },
            "evaluator": {
                "revision": EVALUATOR_REVISION,
                "image": IMAGE_URI,
                "image_digest": IMAGE_DIGEST,
                "native_architecture": "amd64",
                "invoked": False,
                "reason": "candidate response contained no unified diff",
            },
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampler_path": SAMPLER_PATH,
                "hf_repo": HF_REPO,
                "hf_revision": HF_REVISION,
                "hf_commit": HF_COMMIT,
            },
            "sampling": {
                "samples_per_problem": 1,
                "retries": 0,
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_tokens": generation["prompt_tokens"],
                "response_tokens": generation["response_tokens"],
                "attempt_final": True,
            },
            "candidate": {
                "patch_sha256": None,
                "response_path": str(response_path.relative_to(root)),
                "response_sha256": generation["response_sha256"],
                "prompt_sha256": generation["prompt_sha256"],
            },
            "wandb": {
                "run_id": generation["wandb_run_id"],
                "url": generation["wandb_url"],
                "mode": "online",
                "initialized_before_tinker": True,
            },
            "cost": {
                "estimated_tinker_usd": generation["estimated_tinker_usd"],
                "cap_usd": MAX_TINKER_USD,
            },
        }
        receipt["receipt_sha256"] = _sha256_text(_stable_json(receipt))
        receipt_path = run_dir / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "score": receipt["score"],
                    "scope": receipt["scope"],
                    "instance_id": INSTANCE_ID,
                    "wandb": receipt["wandb"],
                    "cost": receipt["cost"],
                    "receipt": str(receipt_path),
                },
                indent=2,
            )
        )
        return

    if not resume:
        candidate_payload = [
            {
                "instance_id": INSTANCE_ID,
                "patch": patch,
                "model_revision": HF_COMMIT,
                "generation_run_id": generation["wandb_run_id"],
                "prefix": f"modal-{generation['wandb_run_id']}",
            }
        ]
        candidate_path.write_text(
            json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    scripts = root / "outputs/e1_swe_bench_pro/evaluator/run_scripts" / INSTANCE_ID
    evaluation = _evaluate_candidate(
        task=task,
        patch=patch,
        run_script=(scripts / "run_script.sh").read_text(encoding="utf-8"),
        parser_script=(scripts / "parser.py").read_text(encoding="utf-8"),
    )
    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-e1-swe-bench-pro-subset-v1",
        "recorded_at": _utc_now(),
        "lane": "E1",
        "suite_id": "swe_bench_pro_eval",
        "suite_role": "primary_eval",
        "status": "SCORED",
        "score": 1.0 if evaluation["resolved"] else 0.0,
        "is_model_score": True,
        "scope": "one_task_subset_not_suite_score",
        "claim_boundary": (
            "This is one frozen SWE-bench Pro task. It is not the 731-task suite score."
        ),
        "instance_id": INSTANCE_ID,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "test",
            "observed_license_state": "absent_at_pinned_revision",
            "proceeding_under": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
        },
        "evaluator": {
            "revision": EVALUATOR_REVISION,
            "image": IMAGE_URI,
            "image_digest": IMAGE_DIGEST,
            "native_architecture": "amd64",
            "official_run_script_timeout_seconds": 120,
        },
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": SAMPLER_PATH,
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "hf_commit": HF_COMMIT,
        },
        "sampling": {
            "samples_per_problem": 1,
            "retries": 0,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_tokens": generation["prompt_tokens"],
            "response_tokens": generation["response_tokens"],
        },
        "candidate": {
            "path": str(candidate_path.relative_to(root)),
            "patch_sha256": generation["patch_sha256"],
            "response_sha256": generation["response_sha256"],
            "prompt_sha256": generation["prompt_sha256"],
        },
        "recovery": {
            "evaluator_only_resume": resume,
            "additional_sampling": False if resume else None,
        },
        "evaluation": evaluation,
        "wandb": {
            "run_id": generation["wandb_run_id"],
            "url": generation["wandb_url"],
            "mode": "online",
            "initialized_before_tinker": True,
        },
        "cost": {
            "estimated_tinker_usd": generation["estimated_tinker_usd"],
            "cap_usd": MAX_TINKER_USD,
        },
    }
    receipt["receipt_sha256"] = _sha256_text(_stable_json(receipt))
    receipt_path = run_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "score": receipt["score"],
                "scope": receipt["scope"],
                "instance_id": INSTANCE_ID,
                "wandb": receipt["wandb"],
                "cost": receipt["cost"],
                "receipt": str(receipt_path),
            },
            indent=2,
        )
    )
