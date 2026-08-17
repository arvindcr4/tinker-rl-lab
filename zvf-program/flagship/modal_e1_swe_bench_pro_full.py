"""Run the exact 731-task SWE-bench Pro E1 campaign on Modal.

The full campaign is intentionally separate from the historical one-task
runner.  It pins the dataset, evaluator, model, checkpoint, and every native
evaluation image; hides gold patches and verifier fields from inference; makes
one pass@1 sampling call per task; and supports evaluator-only resumption.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import modal
from modal.exception import SandboxFilesystemNotFoundError


APP_NAME = "pavlov-e1-swe-bench-pro-full"
SANDBOX_APP_NAME = "pavlov-e1-swe-bench-pro-full-native"
RUN_DATE = "2026-08-16"
EXPECTED_TASK_COUNT = 731
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
DOCKER_REPOSITORY = "jefzda/sweap-images"

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
MAX_TASK_TINKER_USD = 0.10
MAX_SUITE_PROJECTED_USD = 50.0
MAX_SOURCE_CHARS = 96_000
MAX_SOURCE_FILES = 24
MAX_FILE_CHARS = 18_000
GENERATION_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "requirements",
    "interface",
    "repo_language",
    "dockerhub_tag",
)
TERMINAL_GENERATION_STATUSES = {
    "GENERATED",
    "GENERATION_FAILED",
    "GENERATION_ARTIFACT_LOST",
}
INTERRUPTED_ATTEMPTS = (
    {
        "index": 231,
        "instance_id": (
            "instance_protonmail__webclients-"
            "6e165e106d258a442ae849cdf08260329cb92d39"
        ),
        "wandb_run_id": "thw56nqq",
        "wandb_url": (
            "https://wandb.ai/arvindcr4-pes-university/"
            "tinker-rl-lab-pavlov/runs/thw56nqq"
        ),
        "source_sha256": (
            "95641e5e10863ced8ca37f5a770477157bca815c09e0f26893cd50661001e51c"
        ),
        "prompt_sha256": (
            "fd4f91383ee15ba1b5a0758256de3326c72de382a78f7458a3ddde096bf6c98d"
        ),
        "projected_tinker_usd": 0.02658012,
        "original_wandb_status": None,
        "original_candidate_patch_sha256": None,
        "prompt_tokens": None,
        "response_tokens": None,
        "estimated_tinker_usd": None,
        "sample_started": None,
        "sample_completed": None,
        "failure_class": "client_interrupted_before_candidate_receipt",
    },
    {
        "index": 236,
        "instance_id": (
            "instance_tutao__tutanota-"
            "fbdb72a2bd39b05131ff905780d9d4a2a074de26-"
            "vbc0d9ba8f0071fbe982809910959a6ff8884dbbf"
        ),
        "wandb_run_id": "5rfa1b8x",
        "wandb_url": (
            "https://wandb.ai/arvindcr4-pes-university/"
            "tinker-rl-lab-pavlov/runs/5rfa1b8x"
        ),
        "source_sha256": (
            "fda760be70c1b743b3a7ff5379b84229dc2597d1819029145820c7e95a1c0e1b"
        ),
        "prompt_sha256": (
            "8ab6c7769e983caa0cd718b31886961ed9dc26aa5505f390fa4514e8205368d0"
        ),
        "projected_tinker_usd": 0.0133809,
        "original_wandb_status": "GENERATED",
        "original_candidate_patch_sha256": (
            "12c438b584dd401d07167dc94500330660cdb4b7c927aca77d19fecfb54e1746"
        ),
        "prompt_tokens": 4527,
        "response_tokens": 8192,
        "estimated_tinker_usd": 0.0133809,
        "sample_started": True,
        "sample_completed": True,
        "failure_class": "candidate_artifact_lost_after_verified_generation",
    },
    {
        "index": 237,
        "instance_id": (
            "instance_element-hq__element-web-"
            "459df4583e01e4744a52d45446e34183385442d6-vnan"
        ),
        "wandb_run_id": "0b2jh94x",
        "wandb_url": (
            "https://wandb.ai/arvindcr4-pes-university/"
            "tinker-rl-lab-pavlov/runs/0b2jh94x"
        ),
        "source_sha256": (
            "40d6de702c7a99406b75d842d671b9473d76f7c8c18ffcdbe7dfd2784c8aea22"
        ),
        "prompt_sha256": (
            "d78dea315be5083a683fa1b7b2359a5d54847c7f3ab034e820f4dbfa613d2dfe"
        ),
        "projected_tinker_usd": 0.0244077,
        "original_wandb_status": "GENERATED",
        "original_candidate_patch_sha256": (
            "5ffbd46ddd5b6b0653481c70cbe64a9a219c9d86986236d272ddf2d6a6aa3252"
        ),
        "prompt_tokens": 24947,
        "response_tokens": 8192,
        "estimated_tinker_usd": 0.0244077,
        "sample_started": True,
        "sample_completed": True,
        "failure_class": "candidate_artifact_lost_after_verified_generation",
    },
    {
        "index": 524,
        "instance_id": (
            "instance_element-hq__element-web-"
            "b7fea97bb68c6628a644580076f840109132f074-vnan"
        ),
        "wandb_run_id": "b8sji586",
        "wandb_url": (
            "https://wandb.ai/arvindcr4-pes-university/"
            "tinker-rl-lab-pavlov/runs/b8sji586"
        ),
        "source_sha256": (
            "ae87a6a65cc138e80335ccc283dc2c1fe0b2fc42feecea818085421b1330ebc5"
        ),
        "prompt_sha256": (
            "c519c966a2de1b20f0b693f73be6046d9facf347c1108777a3b329c4674acbd1"
        ),
        "projected_tinker_usd": 0.0,
        "original_wandb_status": None,
        "original_candidate_patch_sha256": None,
        "prompt_tokens": None,
        "response_tokens": None,
        "estimated_tinker_usd": 0.0,
        "estimated_modal_gpu_usd": 0.117286,
        "modal_gpu_seconds": 169.0,
        "generation_backend": "modal_gpu_vllm_merged_peft",
        "gpu_type": "A100-80GB",
        "vllm_version": "0.19.0",
        "hf_commit": HF_COMMIT,
        "started_at": "2026-08-17T06:28:33Z",
        "finished_at": "2026-08-17T06:31:22Z",
        "original_modal_app_id": "ap-uMSaO2eOuGzvlkHRjPOx9K",
        "sample_started": True,
        "sample_completed": False,
        "failure_class": "modal_app_stopped_for_nonoverlapping_parallel_handoff",
    },
)
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".rb",
    ".rs",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_stream(stream: Any) -> str:
    if stream is None or not hasattr(stream, "read"):
        return ""
    value = stream.read()
    return value if isinstance(value, str) else str(value or "")


def _sanitize_task(row: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in GENERATION_FIELDS if key not in row]
    if missing:
        raise RuntimeError(f"dataset row missing generation fields: {missing}")
    task = {key: row[key] for key in GENERATION_FIELDS}
    forbidden = {
        "patch",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
        "selected_test_files_to_run",
        "before_repo_set_cmd",
    }
    if forbidden & set(task):
        raise RuntimeError("gold or verifier fields crossed the model boundary")
    return task


def _validate_unified_diff(patch: str) -> tuple[bool, str]:
    if not patch.startswith("diff --git "):
        return False, "patch must begin with a diff --git header"
    if "```" in patch:
        return False, "patch contains a Markdown fence"
    blocks = [
        block
        for block in re.split(r"(?m)(?=^diff --git )", patch.rstrip("\n"))
        if block
    ]
    if not blocks:
        return False, "patch contains no file diff"
    concrete_hunk = re.compile(
        r"(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$"
    )
    for block in blocks:
        lines = block.splitlines()
        if not re.fullmatch(r"diff --git a/\S+ b/\S+", lines[0]):
            return False, f"invalid file header: {lines[0]}"
        if not re.search(r"(?m)^--- (?:a/\S+|/dev/null)$", block):
            return False, f"missing old-file marker for {lines[0]}"
        if not re.search(r"(?m)^\+\+\+ (?:b/\S+|/dev/null)$", block):
            return False, f"missing new-file marker for {lines[0]}"
        if concrete_hunk.search(block) is None:
            return False, f"missing concrete hunk header for {lines[0]}"
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


def _extract_diff(response: str) -> tuple[str, str]:
    starts = [match.start() for match in re.finditer(r"(?m)^diff --git ", response)]
    if not starts:
        return "", "response contains no diff --git header"
    candidates: list[str] = []
    for start in starts:
        tail = response[start:]
        stop_markers = [
            match.start()
            for match in re.finditer(r"(?m)^```|^<\|im_end\|>|^I will ", tail)
            if match.start() > 0
        ]
        cut_points = sorted(set([len(tail), *stop_markers]), reverse=True)
        candidates.extend(tail[:point].strip() + "\n" for point in cut_points)
    for candidate in sorted(candidates, key=len, reverse=True):
        valid, reason = _validate_unified_diff(candidate)
        if valid:
            return candidate, reason
    _, reason = _validate_unified_diff(candidates[-1])
    return "", reason


def _build_prompt(task: dict[str, Any], sources: dict[str, str]) -> str:
    if set(task) != set(GENERATION_FIELDS):
        raise RuntimeError("unexpected fields reached prompt construction")
    source_text = "\n\n".join(
        f"===== {path} =====\n{content}" for path, content in sorted(sources.items())
    )
    return f"""Solve this frozen SWE-bench Pro task in repository {task['repo']}.

Return only a valid unified git diff beginning with `diff --git`. Do not emit
analysis, explanations, planning, ellipses, abbreviated context, or Markdown
fences. Every hunk header must have concrete line ranges. Make the smallest
complete production fix, preserve project style, and ensure the patch applies
to base commit {task['base_commit']}.

PROBLEM STATEMENT
{task['problem_statement']}

REQUIREMENTS
{task['requirements']}

PUBLIC INTERFACE DESCRIPTION
{task['interface']}

FROZEN BASE-COMMIT SOURCE CONTEXT
{source_text}
"""


def _path_candidates(text: str) -> list[str]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])"
        r"([A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+"
        r"(?:\.[A-Za-z0-9]+)?)"
    )
    result: list[str] = []
    for match in pattern.finditer(text):
        path = match.group(1).strip("`'\".,:;()[]{} ")
        if path.startswith(("http://", "https://")) or ".." in Path(path).parts:
            continue
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if path and path not in result:
            result.append(path)
    return result


def _search_terms(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_.:-]{3,})`", text))
    candidates.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", text))
    candidates.extend(re.findall(r"\b([A-Za-z]+[A-Z][A-Za-z0-9]+)\b", text))
    candidates.extend(re.findall(r"\b([a-z][a-z0-9]+_[a-z0-9_]{3,})\b", text))
    stop = {
        "function",
        "method",
        "return",
        "returns",
        "class",
        "public",
        "private",
        "should",
        "must",
        "true",
        "false",
    }
    result: list[str] = []
    for candidate in candidates:
        value = candidate.strip("`'\".,:;()[]{} ")
        if len(value) < 4 or value.lower() in stop or value in result:
            continue
        result.append(value)
    return result[:12]


def _score_inventory_path(path: str, terms: Iterable[str]) -> int:
    lowered = path.lower()
    score = 0
    for term in terms:
        value = term.lower().replace("::", "/").replace(".", "/")
        pieces = [piece for piece in re.split(r"[/_-]+", value) if len(piece) >= 3]
        score += sum(2 for piece in pieces if piece in lowered)
        if value in lowered:
            score += 5
    if "/test" in lowered or lowered.startswith("test"):
        score -= 1
    return score


def _slice_source(content: str, terms: Iterable[str]) -> str:
    if len(content) <= MAX_FILE_CHARS:
        return content
    lines = content.splitlines(keepends=True)
    lowered_terms = [term.lower() for term in terms if len(term) >= 4]
    hits = [
        index
        for index, line in enumerate(lines)
        if any(term in line.lower() for term in lowered_terms)
    ]
    if not hits:
        half = MAX_FILE_CHARS // 2
        return content[:half] + "\n... SOURCE TRUNCATED ...\n" + content[-half:]
    selected: set[int] = set()
    for hit in hits[:16]:
        selected.update(range(max(0, hit - 35), min(len(lines), hit + 36)))
    output: list[str] = []
    last = -2
    size = 0
    for index in sorted(selected):
        line = lines[index]
        marker = "\n... NONMATCHING LINES OMITTED ...\n" if index > last + 1 else ""
        if size + len(marker) + len(line) > MAX_FILE_CHARS:
            break
        if marker:
            output.append(marker)
            size += len(marker)
        output.append(line)
        size += len(line)
        last = index
    return "".join(output)


def _lookup_sandbox_app() -> modal.App:
    return modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True)


def _snapshot_task_sources(task: dict[str, Any], immutable_uri: str) -> dict[str, Any]:
    sandbox = modal.Sandbox.create(
        image=modal.Image.from_registry(immutable_uri),
        app=_lookup_sandbox_app(),
        timeout=20 * 60,
        cpu=(1.0, 2.0),
        memory=(4 * 1024, 16 * 1024),
    )
    try:
        reset = sandbox.exec(
            "bash",
            "-lc",
            " && ".join(
                [
                    "cd /app",
                    f"git reset --hard {shlex.quote(str(task['base_commit']))}",
                    "git clean -fd",
                    f"git checkout {shlex.quote(str(task['base_commit']))}",
                ]
            ),
        )
        reset.wait()
        if reset.returncode != 0:
            raise RuntimeError(f"could not reset {task['instance_id']} to base commit")

        inventory_process = sandbox.exec("git", "-C", "/app", "ls-files", "-z")
        inventory_process.wait()
        if inventory_process.returncode != 0:
            raise RuntimeError(f"could not list files for {task['instance_id']}")
        inventory = [
            path
            for path in _read_stream(inventory_process.stdout).split("\0")
            if path and Path(path).suffix.lower() in SOURCE_EXTENSIONS
        ]
        inventory_set = set(inventory)
        task_text = "\n".join(
            str(task.get(field) or "")
            for field in ("problem_statement", "requirements", "interface")
        )
        terms = _search_terms(task_text)
        selected: list[str] = [
            path for path in _path_candidates(task_text) if path in inventory_set
        ]

        if terms:
            grep_args: list[str] = []
            for term in terms[:8]:
                grep_args.extend(["-e", term])
            grep_process = sandbox.exec(
                "git", "-C", "/app", "grep", "-I", "-l", *grep_args, "--"
            )
            grep_process.wait()
            if grep_process.returncode in {0, 1}:
                for path in _read_stream(grep_process.stdout).splitlines():
                    if path in inventory_set and path not in selected:
                        selected.append(path)

        ranked = sorted(
            inventory,
            key=lambda path: (-_score_inventory_path(path, terms), len(path), path),
        )
        for path in ranked:
            if len(selected) >= MAX_SOURCE_FILES:
                break
            if _score_inventory_path(path, terms) <= 0 and selected:
                break
            if path not in selected:
                selected.append(path)

        sources: dict[str, str] = {}
        source_chars = 0
        for path in selected[:MAX_SOURCE_FILES]:
            try:
                content = sandbox.filesystem.read_text(f"/app/{path}")
            except (SandboxFilesystemNotFoundError, UnicodeDecodeError):
                continue
            sliced = _slice_source(content, terms)
            remaining = MAX_SOURCE_CHARS - source_chars
            if remaining <= 0:
                break
            if len(sliced) > remaining:
                sliced = sliced[:remaining]
            if sliced:
                sources[path] = sliced
                source_chars += len(sliced)
        if not sources:
            raise RuntimeError(f"no base-commit source context for {task['instance_id']}")
        return {
            "instance_id": task["instance_id"],
            "base_commit": task["base_commit"],
            "image": immutable_uri,
            "files": sources,
            "file_count": len(sources),
            "source_chars": source_chars,
            "source_sha256": _sha256_text(_stable_json(sources)),
            "search_terms": terms,
            "recorded_at": _utc_now(),
        }
    finally:
        sandbox.terminate()


def _docker_token() -> str:
    query = urllib.parse.urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{DOCKER_REPOSITORY}:pull",
        }
    )
    with urllib.request.urlopen(
        "https://auth.docker.io/token?" + query, timeout=30
    ) as response:
        return str(json.load(response)["token"])


def _resolve_image_digest(tag: str, token: str) -> str:
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{DOCKER_REPOSITORY}/manifests/{tag}",
        method="HEAD",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": ", ".join(
                [
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        digest = str(response.headers.get("Docker-Content-Digest") or "")
        media_type = str(response.headers.get("Content-Type") or "")
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"registry returned no immutable digest for {tag}")
    if "manifest.v2+json" not in media_type and "image.manifest.v1+json" not in media_type:
        raise RuntimeError(f"registry returned unexpected manifest type for {tag}: {media_type}")
    return digest


def _resolve_image_manifest(
    tasks: list[dict[str, Any]], *, workers: int
) -> dict[str, Any]:
    token = _docker_token()
    records: dict[str, dict[str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_resolve_image_digest, str(task["dockerhub_tag"]), token): task
            for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_task), start=1
        ):
            task = future_to_task[future]
            digest = future.result()
            tag_uri = f"{DOCKER_REPOSITORY}:{task['dockerhub_tag']}"
            records[str(task["instance_id"])] = {
                "instance_id": str(task["instance_id"]),
                "tag": str(task["dockerhub_tag"]),
                "tag_uri": tag_uri,
                "digest": digest,
                "immutable_uri": f"{DOCKER_REPOSITORY}@{digest}",
            }
            if completed % 50 == 0 or completed == len(tasks):
                print(f"resolved {completed}/{len(tasks)} immutable image digests")
    images = [records[str(task["instance_id"])] for task in tasks]
    return {
        "schema_version": "pavlov-e1-image-manifest-v1",
        "registry": "registry-1.docker.io",
        "repository": DOCKER_REPOSITORY,
        "architecture": "linux/amd64",
        "count": len(images),
        "images": images,
        "manifest_sha256": _sha256_text(_stable_json(images)),
        "resolved_at": _utc_now(),
    }


generation_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "huggingface-hub==1.27.0",
    "jinja2==3.1.6",
    "tinker==0.24.1",
    "transformers==5.5.4",
    "wandb==0.21.0",
)
secret = modal.Secret.from_name("pavlov-e1-e14")
app = modal.App(APP_NAME, include_source=True)
_TOKENIZER: Any = None
_HF_VERIFIED = False
_TINKER_SERVICE: Any = None
_TINKER_SAMPLER: Any = None


def _verified_tokenizer() -> Any:
    global _TOKENIZER, _HF_VERIFIED
    if _TOKENIZER is not None and _HF_VERIFIED:
        return _TOKENIZER
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    info = HfApi(token=os.environ["HF_TOKEN"]).model_info(
        HF_REPO, revision=HF_REVISION
    )
    if info.sha != HF_COMMIT:
        raise RuntimeError(f"immutable HF checkpoint drift: {info.sha}")
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    _HF_VERIFIED = True
    return _TOKENIZER


def _shared_tinker_sampler(tinker_module: Any, wandb_run_id: str) -> Any:
    """Reuse one Tinker session per Modal container."""

    global _TINKER_SERVICE, _TINKER_SAMPLER
    if _TINKER_SAMPLER is None:
        _TINKER_SERVICE = tinker_module.ServiceClient(
            user_metadata={
                "campaign": "pavlov-e1-swe-bench-pro-full",
                "suite_id": "swe_bench_pro_eval",
                "first_wandb_run_id": wandb_run_id,
            }
        )
        _TINKER_SAMPLER = _TINKER_SERVICE.create_sampling_client(
            model_path=SAMPLER_PATH
        )
    return _TINKER_SAMPLER


@app.function(
    image=generation_image,
    secrets=[secret],
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    retries=0,
    max_containers=12,
)
def generate_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate one pass@1 candidate with W&B online before the Tinker call."""

    task = dict(payload["task"])
    sources = dict(payload["sources"])
    seed = int(payload["seed"])
    index = int(payload["index"])
    max_tokens = int(payload["max_tokens"])
    temperature = float(payload["temperature"])
    pre_sampling_recovery = payload.get("pre_sampling_recovery")
    instance_id = str(task.get("instance_id") or "")
    started_at = _utc_now()
    if set(task) != set(GENERATION_FIELDS):
        return {
            "instance_id": instance_id,
            "index": index,
            "status": "INFRA_ERROR",
            "phase": "model_boundary",
            "sample_started": False,
            "error": "unexpected fields reached remote generation",
            "started_at": started_at,
            "finished_at": _utc_now(),
        }

    run: Any = None
    sample_started = False
    response_text = ""
    try:
        tokenizer = _verified_tokenizer()
        prompt = _build_prompt(task, sources)
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic source-code patch generator. "
                        "Reply with the requested patch only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        projected_usd = (
            len(prompt_ids) / 1e6 * USD_PER_M_PREFILL
            + max_tokens / 1e6 * USD_PER_M_SAMPLE
        )
        if projected_usd > MAX_TASK_TINKER_USD:
            raise RuntimeError(
                f"projected Tinker spend ${projected_usd:.6f} exceeds "
                f"${MAX_TASK_TINKER_USD:.2f}"
            )

        import tinker
        import tinker.types as T
        import wandb

        run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group=f"e1-swe-bench-pro-full-seed{seed}",
            job_type="primary-evaluation-exact-suite",
            name=f"e1_full_{index:04d}_seed{seed}",
            tags=["e1", "swe_bench_pro", "exact-suite", "pass@1", "modal"],
            mode="online",
            config={
                "suite_id": "swe_bench_pro_eval",
                "suite_role": "primary_eval",
                "scope": "exact_731_task_test_split",
                "task_index": index,
                "instance_id": instance_id,
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
                "thinking_enabled": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "source_sha256": _sha256_text(_stable_json(sources)),
                "prompt_sha256": _sha256_text(prompt),
                "projected_tinker_usd": projected_usd,
                "pre_sampling_recovery": bool(pre_sampling_recovery),
                "prior_pre_sampling_wandb_run_id": (
                    pre_sampling_recovery.get("wandb_run_id")
                    if isinstance(pre_sampling_recovery, dict)
                    else None
                ),
            },
            reinit=True,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("W&B online initialization failed before Tinker")

        sampler = _shared_tinker_sampler(tinker, str(run.id))
        sample_started = True
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
        patch, validation_reason = _extract_diff(response_text)
        estimated_usd = (
            len(prompt_ids) / 1e6 * USD_PER_M_PREFILL
            + len(tokens) / 1e6 * USD_PER_M_SAMPLE
        )
        status = "GENERATED" if patch else "GENERATION_FAILED"
        run.log(
            {
                "generation/prompt_tokens": len(prompt_ids),
                "generation/response_tokens": len(tokens),
                "generation/patch_structurally_valid": int(bool(patch)),
                "cost/estimated_actual_usd": estimated_usd,
            },
            step=1,
        )
        run.summary.update(
            {
                "status": status,
                "candidate_patch_sha256": _sha256_text(patch) if patch else None,
                "patch_validation_reason": validation_reason,
            }
        )
        run_id = str(run.id)
        run_url = str(run.url)
        run.finish(exit_code=0 if patch else 1)
        run = None
        return {
            "instance_id": instance_id,
            "index": index,
            "status": status,
            "phase": "complete",
            "sample_started": True,
            "sample_completed": True,
            "patch": patch,
            "patch_sha256": _sha256_text(patch) if patch else None,
            "patch_validation_reason": validation_reason,
            "response_text": response_text,
            "response_sha256": _sha256_text(response_text),
            "prompt_sha256": _sha256_text(prompt),
            "source_sha256": _sha256_text(_stable_json(sources)),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(tokens),
            "projected_tinker_usd": round(projected_usd, 8),
            "estimated_tinker_usd": round(estimated_usd, 8),
            "wandb_run_id": run_id,
            "wandb_url": run_url,
            "pre_sampling_recovery": pre_sampling_recovery,
            "started_at": started_at,
            "finished_at": _utc_now(),
        }
    except Exception as exc:
        run_id = str(getattr(run, "id", "") or "")
        run_url = str(getattr(run, "url", "") or "")
        if run is not None:
            try:
                run.summary.update({"status": "INFRA_ERROR", "error_type": type(exc).__name__})
                run.finish(exit_code=1)
            except Exception:
                pass
        return {
            "instance_id": instance_id,
            "index": index,
            "status": "INFRA_ERROR",
            "phase": "sampling" if sample_started else "pre_sampling",
            "sample_started": sample_started,
            "sample_completed": False,
            "response_text": response_text,
            "response_sha256": _sha256_text(response_text) if response_text else None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "wandb_run_id": run_id or None,
            "wandb_url": run_url or None,
            "pre_sampling_recovery": pre_sampling_recovery,
            "started_at": started_at,
            "finished_at": _utc_now(),
        }


def _load_exact_dataset(root: Path, raw_path: Path) -> list[dict[str, Any]]:
    python = root / "outputs/e1_swe_bench_pro/runtime/venv/bin/python"
    code = (
        "import json\n"
        "from datasets import load_dataset\n"
        f"rows=load_dataset({DATASET_ID!r},split='test',revision={DATASET_REVISION!r})\n"
        "for row in rows:\n"
        " print(json.dumps(dict(row),sort_keys=True,ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_path.write_text(completed.stdout, encoding="utf-8")
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(rows) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TASK_COUNT} exact-suite tasks, found {len(rows)}"
        )
    instance_ids = [str(row["instance_id"]) for row in rows]
    if len(set(instance_ids)) != EXPECTED_TASK_COUNT:
        raise RuntimeError("exact dataset contains duplicate instance IDs")
    return rows


def _validate_existing_dataset(
    raw_path: Path, sanitized_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    tasks = [
        json.loads(line)
        for line in sanitized_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != EXPECTED_TASK_COUNT or len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError("saved dataset artifacts are not the exact 731-task split")
    if [_sanitize_task(row) for row in rows] != tasks:
        raise RuntimeError("saved sanitized manifest disagrees with raw dataset")
    return rows, tasks


def _source_contexts(
    tasks: list[dict[str, Any]],
    images_by_id: dict[str, str],
    tasks_dir: Path,
    *,
    workers: int,
) -> None:
    pending: list[tuple[dict[str, Any], Path]] = []
    for task in tasks:
        path = tasks_dir / str(task["instance_id"]) / "source_context.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (
                saved.get("base_commit") != task["base_commit"]
                or saved.get("image") != images_by_id[str(task["instance_id"])]
            ):
                raise RuntimeError(f"source receipt drift for {task['instance_id']}")
        else:
            pending.append((task, path))
    if not pending:
        print("all 731 base-commit source contexts already present")
        return
    print(f"capturing {len(pending)} base-commit source contexts")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(
                _snapshot_task_sources,
                task,
                images_by_id[str(task["instance_id"])],
            ): (task, path)
            for task, path in pending
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_item), start=1
        ):
            task, path = future_to_item[future]
            snapshot = future.result()
            _write_json(path, snapshot)
            if completed % 10 == 0 or completed == len(pending):
                print(f"captured {completed}/{len(pending)} source contexts")


def _recover_interrupted_attempts(
    tasks: list[dict[str, Any]], tasks_dir: Path
) -> None:
    """Record interrupted candidates as unresolved without resampling them."""

    for recovery in INTERRUPTED_ATTEMPTS:
        index = int(recovery["index"])
        task = tasks[index]
        instance_id = str(recovery["instance_id"])
        if str(task["instance_id"]) != instance_id:
            raise RuntimeError(
                f"interrupted recovery task-order drift at index {index}"
            )
        task_dir = tasks_dir / instance_id
        generation_path = task_dir / "generation.json"
        if generation_path.is_file():
            saved = json.loads(generation_path.read_text(encoding="utf-8"))
            if saved.get("status") not in TERMINAL_GENERATION_STATUSES:
                raise RuntimeError(
                    f"nonterminal interrupted recovery for {instance_id}"
                )
            continue

        snapshot = json.loads(
            (task_dir / "source_context.json").read_text(encoding="utf-8")
        )
        sources = dict(snapshot["files"])
        source_sha256 = _sha256_text(_stable_json(sources))
        prompt_sha256 = _sha256_text(_build_prompt(task, sources))
        if source_sha256 != recovery["source_sha256"]:
            raise RuntimeError(f"interrupted source receipt drift for {instance_id}")
        if prompt_sha256 != recovery["prompt_sha256"]:
            raise RuntimeError(f"interrupted prompt receipt drift for {instance_id}")

        receipt = {
            "instance_id": instance_id,
            "index": index,
            "status": "GENERATION_ARTIFACT_LOST",
            "phase": "client_interruption_recovery",
            "sample_started": recovery["sample_started"],
            "sample_completed": recovery["sample_completed"],
            "patch": "",
            "patch_sha256": None,
            "response_sha256": None,
            "prompt_sha256": prompt_sha256,
            "source_sha256": source_sha256,
            "prompt_tokens": recovery["prompt_tokens"],
            "response_tokens": recovery["response_tokens"],
            "projected_tinker_usd": recovery["projected_tinker_usd"],
            "estimated_tinker_usd": recovery["estimated_tinker_usd"],
            "wandb_run_id": recovery["wandb_run_id"],
            "wandb_url": recovery["wandb_url"],
            "original_wandb_status": recovery["original_wandb_status"],
            "original_candidate_patch_sha256": recovery[
                "original_candidate_patch_sha256"
            ],
            "failure_class": recovery["failure_class"],
            "additional_sampling_performed": False,
            "recovered_at": _utc_now(),
        }
        for optional_key in (
            "estimated_modal_gpu_usd",
            "modal_gpu_seconds",
            "generation_backend",
            "gpu_type",
            "vllm_version",
            "hf_commit",
            "started_at",
            "finished_at",
            "original_modal_app_id",
        ):
            if optional_key in recovery:
                receipt[optional_key] = recovery[optional_key]
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "generation_response.txt").write_text("", encoding="utf-8")
        _write_json(generation_path, receipt)


def _archive_pre_sampling_attempt(
    task_dir: Path, generation_path: Path, saved: dict[str, Any]
) -> dict[str, Any]:
    """Archive a proven pre-sampling failure so only that task can retry."""

    run_id = str(saved.get("wandb_run_id") or "without_wandb_run")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    attempts_dir = task_dir / "generation_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    archived_generation = attempts_dir / f"pre_sampling_{safe_run_id}.json"
    if archived_generation.exists():
        raise RuntimeError(
            f"pre-sampling archive already exists for {task_dir.name}: {run_id}"
        )
    response_path = task_dir / "generation_response.txt"
    if response_path.exists():
        archived_response = attempts_dir / f"pre_sampling_{safe_run_id}_response.txt"
        if archived_response.exists():
            raise RuntimeError(
                f"pre-sampling response archive exists for {task_dir.name}: {run_id}"
            )
        response_path.replace(archived_response)
    generation_path.replace(archived_generation)
    return {
        "wandb_run_id": saved.get("wandb_run_id"),
        "wandb_url": saved.get("wandb_url"),
        "error_type": saved.get("error_type"),
        "error": saved.get("error"),
        "archived_generation": str(archived_generation),
    }


def _generation_inputs(
    tasks: list[dict[str, Any]],
    tasks_dir: Path,
    *,
    seed: int,
    max_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], float]:
    pending: list[dict[str, Any]] = []
    projected_total = 0.0
    for index, task in enumerate(tasks):
        task_dir = tasks_dir / str(task["instance_id"])
        generation_path = task_dir / "generation.json"
        pre_sampling_recovery: dict[str, Any] | None = None
        if generation_path.is_file():
            saved = json.loads(generation_path.read_text(encoding="utf-8"))
            status = saved.get("status")
            if status in TERMINAL_GENERATION_STATUSES:
                projected_total += float(saved.get("projected_tinker_usd") or 0.0)
                continue
            if (
                status == "INFRA_ERROR"
                and saved.get("phase") == "pre_sampling"
                and saved.get("sample_started") is False
            ):
                pre_sampling_recovery = _archive_pre_sampling_attempt(
                    task_dir, generation_path, saved
                )
            else:
                raise RuntimeError(
                    f"nonterminal saved generation for {task['instance_id']}: "
                    f"{status}"
                )
        snapshot = json.loads(
            (task_dir / "source_context.json").read_text(encoding="utf-8")
        )
        sources = dict(snapshot["files"])
        prompt = _build_prompt(task, sources)
        conservative_prompt_tokens = len(prompt) / 3.0
        projected_total += (
            conservative_prompt_tokens / 1e6 * USD_PER_M_PREFILL
            + max_tokens / 1e6 * USD_PER_M_SAMPLE
        )
        pending.append(
            {
                "index": index,
                "task": task,
                "sources": sources,
                "seed": seed,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "pre_sampling_recovery": pre_sampling_recovery,
            }
        )
    return pending, projected_total


def _run_generation_map(pending: list[dict[str, Any]], tasks_dir: Path) -> None:
    if not pending:
        print("all 731 pass@1 generations already present")
        return
    print(f"starting {len(pending)} pass@1 Tinker generations")
    completed = 0
    for result in generate_candidate.map(
        pending, order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, BaseException):
            raise RuntimeError(f"Modal generation map failed without a task receipt: {result}")
        instance_id = str(result["instance_id"])
        task_dir = tasks_dir / instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        response_text = str(result.pop("response_text", "") or "")
        (task_dir / "generation_response.txt").write_text(
            response_text, encoding="utf-8"
        )
        _write_json(task_dir / "generation.json", result)
        completed += 1
        if completed % 10 == 0 or completed == len(pending):
            print(f"saved {completed}/{len(pending)} generation receipts")


def _build_candidates(
    tasks: list[dict[str, Any]], tasks_dir: Path, candidates_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    for task in tasks:
        generation = json.loads(
            (tasks_dir / str(task["instance_id"]) / "generation.json").read_text(
                encoding="utf-8"
            )
        )
        status = str(generation.get("status") or "")
        if status not in TERMINAL_GENERATION_STATUSES:
            raise RuntimeError(f"nonterminal generation for {task['instance_id']}: {status}")
        generations.append(generation)
        if status == "GENERATED":
            patch = str(generation.get("patch") or "")
            valid, reason = _validate_unified_diff(patch)
            if not valid:
                raise RuntimeError(
                    f"saved patch failed validation for {task['instance_id']}: {reason}"
                )
            candidates.append(
                {
                    "instance_id": task["instance_id"],
                    "model_patch": patch,
                    "model_revision": HF_COMMIT,
                    "generation_run_id": generation["wandb_run_id"],
                    "prefix": f"modal-{generation['wandb_run_id']}",
                }
            )
    _write_json(candidates_path, candidates)
    return candidates, generations


def _run_evaluator(
    root: Path,
    raw_path: Path,
    candidates_path: Path,
    image_manifest_path: Path,
    evaluation_dir: Path,
    *,
    workers: int,
) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    wrapper = root / "zvf-program/flagship/e1_swe_bench_pro_full_eval.py"
    command = [
        "uv",
        "run",
        "--no-project",
        "--with",
        "modal==1.5.4",
        "--with",
        "pandas==3.0.5",
        "--with",
        "tqdm==4.70.0",
        "python",
        str(wrapper),
        "--image_manifest_path",
        str(image_manifest_path),
        "--raw_sample_path",
        str(raw_path),
        "--patch_path",
        str(candidates_path),
        "--output_dir",
        str(evaluation_dir),
        "--dockerhub_username",
        "jefzda",
        "--scripts_dir",
        str(evaluator_dir / "run_scripts"),
        "--num_workers",
        str(workers),
        "--block_network",
    ]
    print(f"starting native evaluation with {workers} workers")
    completed = subprocess.run(command, cwd=evaluator_dir, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"native evaluator exited {completed.returncode}")


def _aggregate_receipt(
    *,
    root: Path,
    run_dir: Path,
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    image_manifest: dict[str, Any],
    seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    eval_results_path = run_dir / "evaluation/eval_results.json"
    eval_results: dict[str, bool] = (
        json.loads(eval_results_path.read_text(encoding="utf-8"))
        if candidates
        else {}
    )
    candidate_ids = {str(item["instance_id"]) for item in candidates}
    if set(eval_results) != candidate_ids:
        missing = sorted(candidate_ids - set(eval_results))
        extra = sorted(set(eval_results) - candidate_ids)
        raise RuntimeError(
            f"evaluator coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    generation_by_id = {str(item["instance_id"]): item for item in generations}
    full_results: dict[str, bool] = {}
    for task in tasks:
        instance_id = str(task["instance_id"])
        generation = generation_by_id[instance_id]
        full_results[instance_id] = (
            bool(eval_results[instance_id])
            if generation["status"] == "GENERATED"
            else False
        )
    if len(full_results) != EXPECTED_TASK_COUNT:
        raise RuntimeError("aggregate denominator is not the exact 731-task split")
    resolved = sum(full_results.values())
    score = resolved / EXPECTED_TASK_COUNT
    full_results_path = run_dir / "full_eval_results.json"
    _write_json(full_results_path, full_results)

    raw_path = run_dir / "dataset_test_731.jsonl"
    sanitized_path = run_dir / "generation_manifest_731.jsonl"
    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    run_scripts = evaluator_dir / "run_scripts"
    scripts_digest = hashlib.sha256()
    for task in tasks:
        instance_id = str(task["instance_id"])
        for filename in ("run_script.sh", "parser.py"):
            path = run_scripts / instance_id / filename
            scripts_digest.update(instance_id.encode())
            scripts_digest.update(filename.encode())
            scripts_digest.update(path.read_bytes())

    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-e1-swe-bench-pro-full-v1",
        "recorded_at": _utc_now(),
        "lane": "E1",
        "suite_id": "swe_bench_pro_eval",
        "suite_role": "primary_eval",
        "status": "SCORED",
        "score": score,
        "score_percent": score * 100.0,
        "is_model_score": True,
        "scope": "exact_731_task_test_split",
        "claim_boundary": (
            "Exact pinned 731-task SWE-bench Pro test split, pass@1. Generation "
            "failures and interrupted candidate-artifact losses are unresolved in "
            "the denominator; artifact losses were not resampled. Dataset license "
            "metadata is absent at the pinned revision; local evaluation proceeds "
            "under the recorded owner risk acceptance and does not authorize "
            "redistribution."
        ),
        "coverage": {
            "expected_tasks": EXPECTED_TASK_COUNT,
            "attempted_generations": len(generations),
            "valid_generated_patches": len(candidates),
            "generation_failures": sum(
                item["status"] == "GENERATION_FAILED" for item in generations
            ),
            "generation_artifact_losses": sum(
                item["status"] == "GENERATION_ARTIFACT_LOST"
                for item in generations
            ),
            "native_evaluations": len(eval_results),
            "resolved": resolved,
            "unresolved": EXPECTED_TASK_COUNT - resolved,
            "complete": True,
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "test",
            "count": EXPECTED_TASK_COUNT,
            "raw_materialization_path": str(raw_path.relative_to(root)),
            "raw_materialization_sha256": _sha256_file(raw_path),
            "generation_manifest_path": str(sanitized_path.relative_to(root)),
            "generation_manifest_sha256": _sha256_file(sanitized_path),
            "task_order_sha256": _sha256_text(
                "\n".join(str(row["instance_id"]) for row in rows) + "\n"
            ),
            "observed_license_state": "absent_at_pinned_revision",
            "claimed_spdx": None,
            "proceeding_under": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
            "decision": "owner_risk_acceptance_2026-08-09",
        },
        "evaluator": {
            "repository_revision": EVALUATOR_REVISION,
            "native_architecture": "linux/amd64",
            "official_scripts_count": EXPECTED_TASK_COUNT,
            "official_scripts_sha256": scripts_digest.hexdigest(),
            "wrapper_path": "zvf-program/flagship/e1_swe_bench_pro_full_eval.py",
            "wrapper_sha256": _sha256_file(
                root / "zvf-program/flagship/e1_swe_bench_pro_full_eval.py"
            ),
            "network_blocked": True,
            "image_manifest_path": str(
                (run_dir / "image_manifest.json").relative_to(root)
            ),
            "image_manifest_sha256": image_manifest["manifest_sha256"],
            "digest_pinned_image_count": image_manifest["count"],
            "raw_results_path": str(
                (run_dir / "evaluation/eval_results.json").relative_to(root)
            ),
            "full_results_path": str(full_results_path.relative_to(root)),
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
            "thinking_enabled": False,
            "wandb_mode": "online",
            "wandb_initialized_before_each_tinker_call": True,
            "artifact_loss_resampling_performed": False,
        },
        "cost": {
            "estimated_tinker_usd": round(
                sum(float(item.get("estimated_tinker_usd") or 0.0) for item in generations),
                8,
            ),
            "tasks_with_unknown_actual_tinker_cost": sum(
                item.get("estimated_tinker_usd") is None for item in generations
            ),
            "max_task_tinker_usd": MAX_TASK_TINKER_USD,
            "suite_preflight_cap_usd": MAX_SUITE_PROJECTED_USD,
            "modal_compute": "approved_by_owner_not_estimated_in_receipt",
        },
        "artifacts": {
            "candidates_path": str((run_dir / "candidates.json").relative_to(root)),
            "task_receipts_dir": str((run_dir / "tasks").relative_to(root)),
        },
    }
    receipt["receipt_sha256"] = _sha256_text(_stable_json(receipt))
    return receipt


@app.local_entrypoint()
def main(
    seed: int = 1818,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    source_workers: int = 16,
    digest_workers: int = 24,
    evaluator_workers: int = 20,
    resume: bool = False,
) -> None:
    root = _repo_root()
    run_dir = root / f"outputs/modal_e1_e14/{RUN_DATE}/e1_swe_bench_pro_full/seed{seed}"
    if run_dir.exists() and not resume:
        raise RuntimeError(f"refusing to overwrite existing exact-suite attempt: {run_dir}")
    if resume and not run_dir.is_dir():
        raise RuntimeError(f"cannot resume missing exact-suite attempt: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    _write_json(
        state_path,
        {
            "status": "PREPARING",
            "updated_at": _utc_now(),
            "seed": seed,
            "dataset_revision": DATASET_REVISION,
            "expected_tasks": EXPECTED_TASK_COUNT,
        },
    )

    raw_path = run_dir / "dataset_test_731.jsonl"
    sanitized_path = run_dir / "generation_manifest_731.jsonl"
    if raw_path.is_file() and sanitized_path.is_file():
        rows, tasks = _validate_existing_dataset(raw_path, sanitized_path)
    else:
        rows = _load_exact_dataset(root, raw_path)
        tasks = [_sanitize_task(row) for row in rows]
        sanitized_path.write_text(
            "".join(
                json.dumps(task, sort_keys=True, ensure_ascii=False) + "\n"
                for task in tasks
            ),
            encoding="utf-8",
        )

    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    evaluator_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=evaluator_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if evaluator_commit != EVALUATOR_REVISION:
        raise RuntimeError(f"evaluator revision drift: {evaluator_commit}")
    missing_scripts = [
        task["instance_id"]
        for task in tasks
        if not all(
            (
                evaluator_dir
                / "run_scripts"
                / str(task["instance_id"])
                / filename
            ).is_file()
            for filename in ("run_script.sh", "parser.py")
        )
    ]
    if missing_scripts:
        raise RuntimeError(f"missing native evaluator scripts: {missing_scripts[:5]}")
    if not (root / "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md").is_file():
        raise RuntimeError("recorded dataset license-risk acceptance is missing")

    image_manifest_path = run_dir / "image_manifest.json"
    if image_manifest_path.is_file():
        image_manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
        if image_manifest.get("count") != EXPECTED_TASK_COUNT:
            raise RuntimeError("saved image manifest is incomplete")
    else:
        image_manifest = _resolve_image_manifest(tasks, workers=digest_workers)
        _write_json(image_manifest_path, image_manifest)
    images_by_id = {
        str(item["instance_id"]): str(item["immutable_uri"])
        for item in image_manifest["images"]
    }
    if set(images_by_id) != {str(task["instance_id"]) for task in tasks}:
        raise RuntimeError("image manifest task coverage mismatch")

    _source_contexts(
        tasks, images_by_id, tasks_dir, workers=max(1, source_workers)
    )
    if resume:
        _recover_interrupted_attempts(tasks, tasks_dir)
    pending, projected_total = _generation_inputs(
        tasks,
        tasks_dir,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if projected_total > MAX_SUITE_PROJECTED_USD:
        raise RuntimeError(
            f"conservative suite projection ${projected_total:.4f} exceeds "
            f"${MAX_SUITE_PROJECTED_USD:.2f}"
        )
    preflight = {
        "schema_version": "pavlov-modal-e1-full-preflight-v1",
        "status": "READY",
        "recorded_at": _utc_now(),
        "task_count": EXPECTED_TASK_COUNT,
        "dataset_revision": DATASET_REVISION,
        "evaluator_revision": EVALUATOR_REVISION,
        "model_revision": MODEL_REVISION,
        "hf_commit": HF_COMMIT,
        "digest_pinned_images": len(images_by_id),
        "source_contexts": EXPECTED_TASK_COUNT,
        "pending_generations": len(pending),
        "conservative_projected_tinker_usd": round(projected_total, 6),
        "suite_projection_cap_usd": MAX_SUITE_PROJECTED_USD,
        "license_risk_acceptance": "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
    }
    preflight["receipt_sha256"] = _sha256_text(_stable_json(preflight))
    if resume:
        preflight_path = (
            run_dir
            / f"resume_preflight_{preflight['receipt_sha256'][:12]}.json"
        )
    else:
        preflight_path = run_dir / "preflight_receipt.json"
    _write_json(preflight_path, preflight)
    _write_json(
        state_path,
        {
            "status": "GENERATING",
            "updated_at": _utc_now(),
            "pending_generations": len(pending),
            "projected_tinker_usd": projected_total,
        },
    )
    print(json.dumps(preflight, indent=2, sort_keys=True))

    _run_generation_map(pending, tasks_dir)
    candidates_path = run_dir / "candidates.json"
    candidates, generations = _build_candidates(tasks, tasks_dir, candidates_path)
    _write_json(
        state_path,
        {
            "status": "EVALUATING",
            "updated_at": _utc_now(),
            "terminal_generations": len(generations),
            "valid_candidates": len(candidates),
        },
    )
    if candidates:
        _run_evaluator(
            root,
            raw_path,
            candidates_path,
            image_manifest_path,
            run_dir / "evaluation",
            workers=max(1, evaluator_workers),
        )
    else:
        (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "evaluation/eval_results.json", {})

    receipt = _aggregate_receipt(
        root=root,
        run_dir=run_dir,
        rows=rows,
        tasks=tasks,
        generations=generations,
        candidates=candidates,
        image_manifest=image_manifest,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _write_json(run_dir / "receipt.json", receipt)
    _write_json(
        state_path,
        {
            "status": "SCORED",
            "updated_at": _utc_now(),
            "score": receipt["score"],
            "receipt": str((run_dir / "receipt.json").relative_to(root)),
        },
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scope": receipt["scope"],
                "score": receipt["score"],
                "score_percent": receipt["score_percent"],
                "coverage": receipt["coverage"],
                "cost": receipt["cost"],
                "receipt": str(run_dir / "receipt.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
