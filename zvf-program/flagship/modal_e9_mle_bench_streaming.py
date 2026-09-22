"""Run one native MLE-bench competition at a time on Modal.

This is deliberately a bounded pilot runner.  It downloads and prepares one
competition in ephemeral storage, asks the pinned Tinker checkpoint for one
Python solution, executes that solution as an unprivileged user, invokes the
native MLE-bench grader, returns immutable artifacts, and then destroys the
ephemeral dataset.  A single-competition result is never promoted to an E9
suite score.
"""

import base64
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "pavlov-e9-mle-bench-streaming"
BRIDGE_BASE_URL = "https://arvindcr4--pavlov-tinker-openai-bridge-tinkeropenaibridge-web.modal.run"
BRIDGE_API_BASE = f"{BRIDGE_BASE_URL}/v1"
MODEL_ALIAS = "pavlov-qwen36-tinker"
MLE_BENCH_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
NATIVE_GRADER_PANDAS_VERSION = "2.2.2"
DEFAULT_COMPETITION = "spooky-author-identification"
MERGED_VLLM_PILOT_COMPETITIONS = {
    "alaska2-image-steganalysis",
    "h-and-m-personalized-fashion-recommendations",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "imet-2020-fgvc7",
    "predict-volcanic-eruptions-ingv-oe",
    "vesuvius-challenge-ink-detection",
}
MERGED_VLLM_PROMPT_TEMPLATE_VERSION = "e9_hard_bounded_collections_v7"
MERGED_VLLM_BACKEND = "modal_gpu_vllm_merged_peft_e9"
MERGED_VLLM_GPU_TYPE = "A100-80GB"
MERGED_VLLM_GPU_RATE_USD_PER_SECOND = 0.000694
MERGED_VLLM_GPU_TIMEOUT_SECONDS = 1800
MERGED_VLLM_CPU_RATE_USD_PER_SECOND = 0.00007016
MERGED_VLLM_GPU_GENERATION_ENABLED = False
MERGED_VLLM_CPU_TIMEOUT_SECONDS = 7200
MERGED_VLLM_MAX_MODEL_LEN = 49_152
MERGED_VLLM_MAX_TOKENS = 4096
MERGED_VLLM_SEED = 809
MERGED_VLLM_TEMPERATURE = 0.2
MERGED_VLLM_TOP_P = 0.95
MERGED_VLLM_POINTER_PATH = Path("/cache/e1-qwen36-seed809-merged-pointer.json")
if modal.is_local():
    SOURCE_ROOT = Path(__file__).resolve().parents[2] / "outputs/e9_mle_bench/mle-bench-source"
    CORE_PATH = Path(__file__).with_name("e9_mle_bench_streaming.py")
    MERGED_VLLM_ARM_PATH = Path(__file__).with_name("e9_merged_vllm_arm.py")
    MERGE_RECEIPT_PATH = (
        Path(__file__).resolve().parents[2]
        / "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/gpu_merge_receipt.json"
    )
else:
    SOURCE_ROOT = Path("/mlebench")
    CORE_PATH = Path("/root/e9_mle_bench_streaming.py")
    MERGED_VLLM_ARM_PATH = Path("/root/e9_merged_vllm_arm.py")
    MERGE_RECEIPT_PATH = Path("/root/e9_gpu_merge_receipt.json")

image = (
    modal.Image.from_dockerfile(
        SOURCE_ROOT / "environment/Dockerfile",
        context_dir=SOURCE_ROOT,
        build_args={"INSTALL_HEAVY_DEPENDENCIES": "false"},
    )
    .run_commands(
        "/opt/conda/bin/conda run -n agent pip install "
        "numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.1 scipy==1.14.1 "
        "xgboost==2.1.1 pillow==10.4.0 py7zr==0.22.0 "
        "opencv-python-headless==4.10.0.84"
    )
    .run_commands(
        f"/opt/conda/bin/conda run -n mleb pip install pandas=={NATIVE_GRADER_PANDAS_VERSION}"
    )
    .uv_pip_install("cryptography==44.0.3")
    .entrypoint([])
    .add_local_file(str(CORE_PATH), "/root/e9_mle_bench_streaming.py")
    .add_local_file(str(MERGED_VLLM_ARM_PATH), "/root/e9_merged_vllm_arm.py")
)

merged_vllm_gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.19.0",
        "transformers==4.57.6",
        "huggingface_hub[hf_xet]>=0.34.0,<1",
        "wandb==0.21.0",
        "cryptography==44.0.3",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(str(CORE_PATH), "/root/e9_mle_bench_streaming.py")
    .add_local_file(str(MERGED_VLLM_ARM_PATH), "/root/e9_merged_vllm_arm.py")
    .add_local_file(str(MERGE_RECEIPT_PATH), "/root/e9_gpu_merge_receipt.json")
)

merged_vllm_preflight_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "huggingface_hub[hf_xet]>=0.34.0,<1", "wandb==0.21.0", "cryptography==44.0.3"
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(str(MERGED_VLLM_ARM_PATH), "/root/e9_merged_vllm_arm.py")
    .add_local_file(str(MERGE_RECEIPT_PATH), "/root/e9_gpu_merge_receipt.json")
)

app = modal.App(APP_NAME, include_source=True)
bridge_secret = modal.Secret.from_name("pavlov-tinker-bridge-auth")
kaggle_secret = modal.Secret.from_name("pavlov-kaggle")
merged_vllm_secret = modal.Secret.from_name("pavlov-e1-e14")
merged_vllm_signing_secret = modal.Secret.from_name("pavlov-e9-allocation-attestation")
merged_vllm_cache = modal.Volume.from_name("pavlov-e1-qwen36-hf-cache", create_if_missing=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_v3_signing_keypair(v3_contract: dict[str, Any]) -> None:
    """Fail closed before preparation if the signing secret cannot authenticate its public key."""

    private_seed = os.environ.get("E9_ALLOCATION_ATTESTATION_PRIVATE_KEY_SEED_BASE64", "")
    public_key = v3_contract.get("public_key_base64")
    if not isinstance(public_key, str) or not private_seed:
        raise RuntimeError("merged-vLLM v3 signing key material is unavailable")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        seed = base64.b64decode(private_seed, validate=True)
        supplied_public = base64.b64decode(public_key, validate=True)
        derived_public = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except Exception as exc:
        raise RuntimeError("merged-vLLM v3 signing key material is invalid") from exc
    if len(seed) != 32 or supplied_public != derived_public:
        raise RuntimeError("merged-vLLM v3 signing key pair does not match")


def _run(
    command: list[str], *, timeout: int, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"command timed out after {timeout} seconds: {command[0]}\n"
            f"stdout tail:\n{stdout[-4000:]}\nstderr tail:\n{stderr[-4000:]}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr[-4000:]
        stdout = completed.stdout[-4000:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command[0]}\n"
            f"stdout tail:\n{stdout}\nstderr tail:\n{stderr}"
        )
    return completed


def _request_json(url: str, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"bridge returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("bridge returned a non-object response")
    return result


def _public_preview(public_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(public_dir.rglob("*")):
        if not path.is_file() or len(parts) >= 24:
            continue
        relative = path.relative_to(public_dir)
        size = path.stat().st_size
        parts.append(f"FILE {relative} ({size} bytes)")
        if size <= 2_000_000 and path.suffix.lower() in {".csv", ".txt", ".md", ".json"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(text[:3000])
    return "\n\n".join(parts)[:45_000]


def _agent_prompt(competition_id: str, public_dir: Path) -> str:
    description = (public_dir / "description.md").read_text(encoding="utf-8", errors="replace")
    return f"""You are the ML coding agent for the MLE-bench competition
`{competition_id}`. Write one complete, compact Python program. Your response
must start immediately with a Python import statement: no analysis, planning,
thinking tags, prose, or Markdown fences. Keep the program under 120 lines and
ensure it is syntactically complete.

Runtime contract:
- Read public competition files only from /home/data.
- Write the final CSV exactly to /home/submission/submission.csv.
- The environment includes numpy, pandas, scipy, scikit-learn, xgboost, Pillow,
  and py7zr.
- Do not access the network, environment variables, /private, or any path
  outside /home/data, /home/code, and /home/submission.
- Make the program deterministic and robust to the observed file names.
- Treat the public file preview as recursive path evidence. Image datasets may
  nest training files under a label, class, or hotel-ID directory; derive the
  exact path pattern from metadata columns and preview examples, and verify
  discovered train and test collections are nonempty before feature building.
- When an image collection is very large, use a deterministic bounded subset
  that preserves label coverage (for example one or a fixed number per class)
  so the complete program, submission, and native grade fit the runtime limit.
- This bound is mandatory, not advisory: process at most 8,000 training images
  total and at most 2,000 per observed class. Use at most 64 compact numeric
  features per image. Never flatten resized pixels into a large dense design
  matrix for an unbounded collection.
- Large recursive collections may instead contain one tabular sensor or time-
  series file per train/test example. Never concatenate all raw rows. Compute
  compact deterministic per-file statistics, bound only the training examples
  if necessary, and still produce one prediction for every sample-submission
  row in its exact order.
- Never require a standalone test metadata table unless the preview proves one
  exists. When test examples are files, derive their IDs and exact order from
  the sample submission, then match those IDs to recursively discovered file
  stems. Fail explicitly on an empty glob instead of indexing its first item.
- If training labels are encoded by parent directory names rather than a
  metadata table, build the target from those directory names and preserve
  coverage of every observed class in any deterministic training subset.
- Detect whether the target is scalar, multiclass, multilabel, or multioutput
  before fitting. Never pass a 2-D multilabel indicator matrix to a single-
  output estimator; use an estimator or wrapper that supports that target.
- For segmentation, detection, or other structured submissions, preserve the
  exact sample row IDs and encoding syntax. Prefer a deterministic valid-format
  baseline over a program that times out, exhausts memory, or emits no CSV;
  never read private labels or invent sample rows. Do not allocate or retain
  full-resolution masks for training or test images, and never iterate in
  Python over every pixel of a whole-slide image. For RLE submissions, build
  sparse runs directly or emit the exact valid empty-mask encoding from the
  sample schema when a bounded image method cannot finish safely. Never train
  a per-pixel classifier from whole-slide arrays, open a full-resolution image
  merely to derive a baseline, or condition CSV creation on both pixel classes
  being observed. The program must always write the schema-valid CSV.
- For recommendation submissions whose target is a fixed-length string of item
  IDs, do not inspect image directories. Read only the necessary transaction
  columns in bounded chunks, derive a deterministic popularity list, and emit
  the exact required number and formatting of item IDs for every sample row.
- Public tabular files may be single-file `.csv.zip` archives. Discover both
  `.csv` and `.csv.zip` names (for example with `*.csv*`); pandas can read a
  single-file ZIP directly.
- Train on the supplied training data and predict the supplied test rows.
- Inspect the train and test schemas independently. Never assume a feature
  column present in training has the same name, or is present at all, in test;
  explicitly align only columns that actually exist in each frame.
- Locate the supplied sample submission even when its filename has a suffix
  (for example sample_submission_null.csv), and match its columns and row order
  exactly.

COMPETITION DESCRIPTION
{description[:18_000]}

PUBLIC FILE PREVIEW
{_public_preview(public_dir)}
"""


def _grade(
    data_dir: Path, competition_id: str, submission: Path, output_path: Path
) -> dict[str, Any]:
    script = (
        "import json,sys; from pathlib import Path; "
        "from mlebench.registry import registry; from mlebench.grade import grade_csv; "
        "competition=registry.set_data_dir(Path(sys.argv[1])).get_competition(sys.argv[2]); "
        "report=grade_csv(Path(sys.argv[3]), competition).to_dict(); "
        "Path(sys.argv[4]).write_text(json.dumps(report,sort_keys=True),encoding='utf-8')"
    )
    _run(
        [
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "mleb",
            "python",
            "-c",
            script,
            str(data_dir),
            competition_id,
            str(submission),
            str(output_path),
        ],
        timeout=600,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    return result


def _cpu_prepare_failure_result(
    *,
    competition_id: str,
    reservation_id: str,
    launch_nonce: str,
    reservation: dict[str, Any],
    elapsed_seconds: float,
    error: Exception,
    recorded_at_utc: str,
) -> dict[str, Any]:
    from e9_merged_vllm_arm import build_cpu_preparation_failure_receipt_v1

    receipt = build_cpu_preparation_failure_receipt_v1(
        competition_id=competition_id,
        reservation_id=reservation_id,
        launch_nonce=launch_nonce,
        reservation=reservation,
        prepare_elapsed_seconds=elapsed_seconds,
        estimated_cpu_usd=round(elapsed_seconds * MERGED_VLLM_CPU_RATE_USD_PER_SECOND, 9),
        error_type=type(error).__name__,
        error_message=str(error) or repr(error),
        recorded_at_utc=recorded_at_utc,
    )
    return {"receipt": receipt}


@app.function(
    image=image,
    secrets=[kaggle_secret],
    cpu=4.0,
    memory=8192,
    timeout=3600,
    max_containers=1,
    single_use_containers=True,
)
def regrade_saved_submission(
    competition_id: str,
    submission_bytes: bytes,
    source_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Re-run only the native grader for an immutable saved submission."""

    sys.path.insert(0, "/root")
    from e9_mle_bench_streaming import (
        build_run_receipt,
        validate_submission_regrade_provenance,
    )

    submission_sha256 = hashlib.sha256(submission_bytes).hexdigest()
    provenance = validate_submission_regrade_provenance(
        competition_id=competition_id,
        submission_sha256=submission_sha256,
        source_receipt=source_receipt,
    )
    run_id = f"{competition_id}-regrade-{uuid.uuid4().hex[:12]}"
    run_root = Path("/tmp/e9-streaming") / run_id
    data_dir = run_root / "data"
    submission_dir = run_root / "submission"
    artifacts_dir = run_root / "artifacts"
    for directory in (data_dir, submission_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "submission.csv"
    grade_path = artifacts_dir / "native_grade.json"
    submission_path.write_bytes(submission_bytes)
    started = time.monotonic()
    try:
        pandas_version = _run(
            [
                "/opt/conda/bin/conda",
                "run",
                "-n",
                "mleb",
                "python",
                "-c",
                "import pandas; print(pandas.__version__)",
            ],
            timeout=60,
        ).stdout.strip()
        if pandas_version != NATIVE_GRADER_PANDAS_VERSION:
            raise RuntimeError(
                "native grader pandas version drift: "
                f"{pandas_version} != {NATIVE_GRADER_PANDAS_VERSION}"
            )
        prepare = _run(
            [
                "/opt/conda/bin/conda",
                "run",
                "-n",
                "mleb",
                "mlebench",
                "prepare",
                "-c",
                competition_id,
                "--data-dir",
                str(data_dir),
            ],
            timeout=3600,
        )
        native_grade = _grade(
            data_dir,
            competition_id,
            submission_path,
            grade_path,
        )
        receipt = build_run_receipt(
            competition_id=competition_id,
            native_grade=native_grade,
            bridge_health=provenance,
        )
        source_artifacts = source_receipt.get("artifacts") or {}
        receipt.update(
            {
                "run_id": run_id,
                "mle_bench_commit": MLE_BENCH_COMMIT,
                "agent": "pavlov_tinker_single_call_ml_agent_v1",
                "sample_reused": True,
                "seed": source_receipt.get("seed", 809),
                "deterministic_repair": source_receipt.get("deterministic_repair"),
                "verifier_repair": ("pinned_mleb_pandas_2_2_2_for_text_normalization_issue_150"),
                "native_grader_pandas_version": pandas_version,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "bridge_budget": {
                    "mode": "saved_submission_native_regrade_no_generation",
                    "remaining_before_usd": None,
                    "remaining_after_usd": None,
                    "charged_this_run_usd": 0.0,
                    "maximum_usd": None,
                },
                "artifacts": {
                    "solution_sha256": source_artifacts.get("solution_sha256"),
                    "submission_sha256": submission_sha256,
                    "native_grade_sha256": _sha256(grade_path),
                },
                "prepare_log_tail": (prepare.stdout + prepare.stderr)[-3000:],
                "regrade_provenance": provenance,
            }
        )
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "receipt": receipt,
            "submission_bytes": submission_bytes,
            "native_grade_json": native_grade,
        }
    finally:
        shutil.rmtree(run_root, ignore_errors=True)


@app.function(
    image=merged_vllm_preflight_image,
    secrets=[merged_vllm_secret],
    volumes={"/cache": merged_vllm_cache},
    cpu=2.0,
    memory=8192,
    timeout=900,
    max_containers=1,
    single_use_containers=True,
)
def preflight_merged_vllm_allocation() -> dict[str, Any]:
    """Verify paid-GPU provenance prerequisites on CPU before dispatching a GPU class."""

    import wandb
    from huggingface_hub import HfApi

    sys.path.insert(0, "/root")
    from e9_merged_vllm_arm import (
        ARM_ID,
        EXPECTED_ADAPTER_COMMIT,
        EXPECTED_BASE_COMMIT,
        build_gpu_allocation_gate,
        validate_merge_receipt_and_shards,
    )

    run = wandb.init(
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group=ARM_ID,
        job_type="e9-merged-vllm-pre-allocation-provenance",
        name="e9_merged_vllm_seed809_pre_allocation",
        tags=["e9", "mle_bench", "merged_vllm", "pre_allocation", "pilot"],
        mode="online",
        config={
            "arm_id": ARM_ID,
            "base_commit": EXPECTED_BASE_COMMIT,
            "adapter_commit": EXPECTED_ADAPTER_COMMIT,
        },
        reinit=True,
    )
    if run is None or not getattr(run, "id", None):
        raise RuntimeError("W&B online initialization failed before E9 GPU allocation")
    initialized_at = _utc_now()
    try:
        merge_receipt = json.loads(
            Path("/root/e9_gpu_merge_receipt.json").read_text(encoding="utf-8")
        )
        merged_root = Path(str(merge_receipt.get("merged_path") or ""))
        checkpoint_verification = validate_merge_receipt_and_shards(
            merge_receipt, merged_root=merged_root
        )
        api = HfApi(token=os.environ["HF_TOKEN"])
        base_info = api.model_info(str(merge_receipt["base_model"]), revision=EXPECTED_BASE_COMMIT)
        adapter_info = api.model_info(
            str(merge_receipt["adapter_repo"]), revision=EXPECTED_ADAPTER_COMMIT
        )
        if base_info.sha != EXPECTED_BASE_COMMIT:
            raise RuntimeError("base-model revision drift before E9 GPU allocation")
        if adapter_info.sha != EXPECTED_ADAPTER_COMMIT:
            raise RuntimeError("adapter revision drift before E9 GPU allocation")
        run.finish(exit_code=0)
        server_run = wandb.Api().run(f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}")
        if str(getattr(server_run, "id", "")) != str(run.id):
            raise RuntimeError("W&B server did not confirm the pre-allocation run")
        checkpoint_verification.update(
            {
                "base_model": str(merge_receipt["base_model"]),
                "base_commit": base_info.sha,
                "adapter_repo": str(merge_receipt["adapter_repo"]),
                "adapter_commit": adapter_info.sha,
            }
        )
        return build_gpu_allocation_gate(
            wandb_receipt={
                "mode": "online",
                "run_id": str(run.id),
                "url": str(run.url),
                "initialized_at": initialized_at,
                "server_confirmed": True,
                "server_run_path": (f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}"),
            },
            checkpoint_verification=checkpoint_verification,
            preflight_completed_at=_utc_now(),
        )
    except Exception:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise


@app.cls(
    image=merged_vllm_gpu_image,
    gpu=MERGED_VLLM_GPU_TYPE,
    secrets=[merged_vllm_secret],
    volumes={"/cache": merged_vllm_cache},
    cpu=8.0,
    memory=65_536,
    timeout=MERGED_VLLM_GPU_TIMEOUT_SECONDS,
    scaledown_window=5 * 60,
    max_containers=1,
)
class E9MergedVllmGenerator:
    """One immutable merged-checkpoint generator for the separate E9 arm."""

    allocation_gate_json: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import wandb
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        sys.path.insert(0, "/root")
        from e9_merged_vllm_arm import (
            ARM_ID,
            EXPECTED_ADAPTER_COMMIT,
            EXPECTED_BASE_COMMIT,
            require_online_wandb_before_paid_load,
            validate_gpu_allocation_gate,
            validate_merge_receipt_and_shards,
        )

        try:
            allocation_gate = validate_gpu_allocation_gate(json.loads(self.allocation_gate_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("GPU allocation gate parameter is malformed") from exc

        run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group=ARM_ID,
            job_type="e9-merged-vllm-model-load",
            name="e9_merged_vllm_seed809_model_load",
            tags=["e9", "mle_bench", "merged_vllm", "model_load", "pilot"],
            mode="online",
            config={
                "arm_id": ARM_ID,
                "base_commit": EXPECTED_BASE_COMMIT,
                "adapter_commit": EXPECTED_ADAPTER_COMMIT,
                "gpu_type": MERGED_VLLM_GPU_TYPE,
                "vllm_version": "0.19.0",
                "max_model_len": MERGED_VLLM_MAX_MODEL_LEN,
            },
            reinit=True,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("W&B online initialization failed before E9 GPU load")
        initialized_at = _utc_now()
        server_run = wandb.Api().run(f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}")
        if str(getattr(server_run, "id", "")) != str(run.id):
            raise RuntimeError("W&B server did not confirm the E9 GPU load run")
        wandb_receipt = {
            "mode": "online",
            "run_id": str(run.id),
            "url": str(run.url),
            "initialized_at": initialized_at,
            "server_confirmed": True,
            "server_run_path": f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}",
        }
        paid_load_started_at = _utc_now()
        require_online_wandb_before_paid_load(
            wandb_receipt, paid_phase_started_at=paid_load_started_at
        )
        started = time.perf_counter()
        try:
            merge_receipt = json.loads(
                Path("/root/e9_gpu_merge_receipt.json").read_text(encoding="utf-8")
            )
            merged_root = Path(str(merge_receipt.get("merged_path") or ""))
            checkpoint_verification = validate_merge_receipt_and_shards(
                merge_receipt, merged_root=merged_root
            )
            api = HfApi(token=os.environ["HF_TOKEN"])
            base_info = api.model_info(
                str(merge_receipt["base_model"]), revision=EXPECTED_BASE_COMMIT
            )
            adapter_info = api.model_info(
                str(merge_receipt["adapter_repo"]), revision=EXPECTED_ADAPTER_COMMIT
            )
            if base_info.sha != EXPECTED_BASE_COMMIT:
                raise RuntimeError("base-model revision drift before E9 GPU load")
            if adapter_info.sha != EXPECTED_ADAPTER_COMMIT:
                raise RuntimeError("adapter revision drift before E9 GPU load")
            checkpoint_verification.update(
                {
                    "base_model": str(merge_receipt["base_model"]),
                    "base_commit": base_info.sha,
                    "adapter_repo": str(merge_receipt["adapter_repo"]),
                    "adapter_commit": adapter_info.sha,
                }
            )
            if checkpoint_verification != allocation_gate["checkpoint_verification"]:
                raise RuntimeError(
                    "GPU allocation gate checkpoint evidence changed after allocation"
                )
            require_online_wandb_before_paid_load(
                allocation_gate["wandb_receipt"], paid_phase_started_at=paid_load_started_at
            )
            from vllm import LLM

            self.tokenizer = AutoTokenizer.from_pretrained(merged_root, local_files_only=True)
            self.llm = LLM(
                model=str(merged_root),
                tokenizer=str(merged_root),
                dtype="bfloat16",
                max_model_len=MERGED_VLLM_MAX_MODEL_LEN,
                max_num_seqs=1,
                gpu_memory_utilization=0.98,
                enforce_eager=True,
                language_model_only=True,
                trust_remote_code=False,
            )
            gpu_runtime = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            load_seconds = time.perf_counter() - started
            load_usd = load_seconds * MERGED_VLLM_GPU_RATE_USD_PER_SECOND
            self.load_receipt = {
                "status": "READY",
                "generation_backend": MERGED_VLLM_BACKEND,
                "gpu_type": MERGED_VLLM_GPU_TYPE,
                "gpu_runtime": gpu_runtime,
                "vllm_version": "0.19.0",
                "load_seconds": round(load_seconds, 6),
                "estimated_modal_gpu_usd": round(load_usd, 9),
                "wandb_receipt": wandb_receipt,
                "paid_load_started_at": paid_load_started_at,
                "allocation_gate_sha256": allocation_gate["allocation_gate_sha256"],
                "checkpoint_verification": checkpoint_verification,
            }
            run.log(
                {
                    "gpu/load_seconds": load_seconds,
                    "cost/estimated_modal_gpu_usd": load_usd,
                },
                step=1,
            )
            run.summary.update(
                {
                    "status": "READY",
                    "gpu_runtime": gpu_runtime,
                    "verified_shard_count": checkpoint_verification["verified_shard_count"],
                }
            )
            run.finish(exit_code=0)
        except Exception as exc:
            run.summary.update({"status": "INFRA_ERROR", "error_type": type(exc).__name__})
            run.finish(exit_code=1)
            raise

    @modal.method()
    def generate_program(
        self,
        prompt: str,
        competition_id: str,
        allocation_attestation_sha256: str | None = None,
    ) -> dict[str, Any]:
        import wandb
        from vllm import SamplingParams

        sys.path.insert(0, "/root")
        from e9_merged_vllm_arm import ARM_ID, require_online_wandb_before_paid_load

        if competition_id not in MERGED_VLLM_PILOT_COMPETITIONS:
            raise RuntimeError("competition is not in the prospective merged-vLLM pilot set")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("E9 merged-vLLM prompt is absent")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group=ARM_ID,
            job_type="e9-merged-vllm-native-pilot-generation",
            name=f"e9_merged_vllm_{competition_id}_seed{MERGED_VLLM_SEED}",
            tags=["e9", "mle_bench", "merged_vllm", competition_id, "pilot"],
            mode="online",
            config={
                "arm_id": ARM_ID,
                "competition_id": competition_id,
                "seed": MERGED_VLLM_SEED,
                "temperature": MERGED_VLLM_TEMPERATURE,
                "top_p": MERGED_VLLM_TOP_P,
                "max_tokens": MERGED_VLLM_MAX_TOKENS,
                "thinking_enabled": False,
                "prompt_sha256": prompt_sha256,
                "prompt_template_version": MERGED_VLLM_PROMPT_TEMPLATE_VERSION,
            },
            reinit=True,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("W&B online initialization failed before E9 GPU sample")
        initialized_at = _utc_now()
        server_run = wandb.Api().run(f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}")
        if str(getattr(server_run, "id", "")) != str(run.id):
            raise RuntimeError("W&B server did not confirm the E9 GPU generation run")
        wandb_receipt = {
            "mode": "online",
            "run_id": str(run.id),
            "url": str(run.url),
            "initialized_at": initialized_at,
            "server_confirmed": True,
            "server_run_path": f"arvindcr4-pes-university/tinker-rl-lab-pavlov/{run.id}",
        }
        generation_started_at = _utc_now()
        require_online_wandb_before_paid_load(
            wandb_receipt, paid_phase_started_at=generation_started_at
        )
        rendered = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Return only compact executable Python. Start with an import "
                        "statement. Never emit analysis, thinking tags, prose, or "
                        "Markdown. Keep the complete program under 120 lines."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(prompt_ids) + MERGED_VLLM_MAX_TOKENS > MERGED_VLLM_MAX_MODEL_LEN:
            raise RuntimeError("E9 prompt plus output exceeds the pinned GPU context")
        started = time.perf_counter()
        try:
            output = self.llm.generate(
                [rendered],
                SamplingParams(
                    max_tokens=MERGED_VLLM_MAX_TOKENS,
                    temperature=MERGED_VLLM_TEMPERATURE,
                    top_p=MERGED_VLLM_TOP_P,
                    seed=MERGED_VLLM_SEED,
                ),
                use_tqdm=False,
            )[0]
            seconds = time.perf_counter() - started
            token_ids = list(output.outputs[0].token_ids)
            response_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            response_sha256 = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
            generation_usd = seconds * MERGED_VLLM_GPU_RATE_USD_PER_SECOND
            total_gpu_usd = float(self.load_receipt["estimated_modal_gpu_usd"]) + generation_usd
            run.log(
                {
                    "generation/prompt_tokens": len(prompt_ids),
                    "generation/response_tokens": len(token_ids),
                    "gpu/generation_seconds": seconds,
                    "cost/estimated_modal_gpu_usd": generation_usd,
                },
                step=1,
            )
            run.summary.update({"status": "GENERATED", "response_sha256": response_sha256})
            run.finish(exit_code=0)
            return {
                "generation_mode": "fresh_merged_vllm_generation",
                "generation_backend": MERGED_VLLM_BACKEND,
                "competition_id": competition_id,
                "seed": MERGED_VLLM_SEED,
                "temperature": MERGED_VLLM_TEMPERATURE,
                "top_p": MERGED_VLLM_TOP_P,
                "max_tokens": MERGED_VLLM_MAX_TOKENS,
                "thinking_enabled": False,
                "prompt_sha256": prompt_sha256,
                "prompt_template_version": MERGED_VLLM_PROMPT_TEMPLATE_VERSION,
                "prompt_tokens": len(prompt_ids),
                "response_text": response_text,
                "response_sha256": response_sha256,
                "response_tokens": len(token_ids),
                "generation_started_at": generation_started_at,
                "generation_finished_at": _utc_now(),
                "generation_seconds": round(seconds, 6),
                "estimated_generation_gpu_usd": round(generation_usd, 9),
                "estimated_total_gpu_usd": round(total_gpu_usd, 9),
                "wandb_receipt": wandb_receipt,
                "load_receipt": dict(self.load_receipt),
                "checkpoint_verification": dict(self.load_receipt["checkpoint_verification"]),
                "allocation_attestation_sha256": allocation_attestation_sha256,
            }
        except Exception as exc:
            run.summary.update({"status": "INFRA_ERROR", "error_type": type(exc).__name__})
            run.finish(exit_code=1)
            raise


@app.function(
    image=image,
    secrets=[kaggle_secret],
    cpu=0.25,
    memory=512,
    timeout=120,
)
def probe_kaggle_rules(competition_id: str) -> dict[str, Any]:
    """Verify download access without accepting terms or downloading the archive."""

    import base64

    if competition_id not in MERGED_VLLM_PILOT_COMPETITIONS:
        raise RuntimeError("rules probe competition is outside the prospective pilot set")
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not username or not key:
        raise RuntimeError("Kaggle credential is absent")
    token = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/competitions/data/download-all/" + competition_id,
        headers={"Authorization": f"Basic {token}", "Range": "bytes=0-0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read(1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return {
            "competition_id": competition_id,
            "accepted": False,
            "http_status": exc.code,
            "blocked_on_rules": exc.code in {401, 403},
            "error": detail,
        }
    return {
        "competition_id": competition_id,
        "accepted": status in {200, 206},
        "http_status": status,
        "bytes_read": 1,
        "probe_only": True,
    }


@app.function(
    image=image,
    secrets=[bridge_secret, kaggle_secret, merged_vllm_signing_secret],
    cpu=4.0,
    memory=8192,
    timeout=7200,
    max_containers=1,
    single_use_containers=True,
)
def run_one(
    competition_id: str = DEFAULT_COMPETITION,
    authorized_total_usd: float = 55.91445263,
    solution_code_override: str | None = None,
    replay_provenance: dict[str, Any] | None = None,
    generation_backend: str = "tinker",
    merged_arm_reservation: dict[str, Any] | None = None,
    merged_vllm_allocation_gate: dict[str, Any] | None = None,
    merged_vllm_reservation_id: str | None = None,
    merged_vllm_launch_nonce: str | None = None,
    merged_vllm_v3_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, "/root")
    from e9_mle_bench_streaming import (
        LaunchGateError,
        build_run_receipt,
        extract_python_code,
        identify_known_deterministic_repair,
        preapply_replay_runtime_repair,
        repair_known_submission_alignment,
        repair_known_runtime_error,
        validate_bridge_gate,
        validate_replay_provenance,
    )

    if generation_backend not in {"tinker", "merged_vllm"}:
        raise LaunchGateError("unsupported E9 generation backend")
    # The remote boundary must reject a frozen separate arm before any long
    # MLE-bench preparation, dataset download, or GPU construction.
    if generation_backend == "merged_vllm" and not MERGED_VLLM_GPU_GENERATION_ENABLED:
        raise LaunchGateError("merged-vLLM GPU generation is frozen at the remote boundary")
    api_key = os.environ.get("TINKER_BRIDGE_API_KEY", "")
    if generation_backend == "merged_vllm":
        from e9_merged_vllm_arm import validate_gpu_allocation_gate

        if competition_id not in MERGED_VLLM_PILOT_COMPETITIONS:
            raise LaunchGateError("competition is not in the prospective merged-vLLM pilot set")
        if solution_code_override is not None or replay_provenance is not None:
            raise LaunchGateError("merged-vLLM arm forbids archived solution replay")
        if not isinstance(merged_arm_reservation, dict):
            raise LaunchGateError("merged-vLLM combined reservation is absent")
        if not isinstance(merged_vllm_reservation_id, str) or not isinstance(
            merged_vllm_launch_nonce, str
        ):
            raise LaunchGateError("merged-vLLM launches require v3 reservation and nonce bindings")
        if merged_arm_reservation.get("reservation_status") != "RESERVED_PRE_LAUNCH":
            raise LaunchGateError("merged-vLLM reservation is not pre-launch")
        if not isinstance(merged_vllm_allocation_gate, dict):
            raise LaunchGateError("merged-vLLM GPU allocation gate is absent")
        if not isinstance(merged_vllm_v3_contract, dict):
            raise LaunchGateError("merged-vLLM v3 provenance contract is absent")
        _validate_v3_signing_keypair(merged_vllm_v3_contract)
        allocation_gate = validate_gpu_allocation_gate(merged_vllm_allocation_gate)
        health_before = None
        remaining_before = None
    elif solution_code_override is None:
        if not api_key:
            raise RuntimeError("TINKER_BRIDGE_API_KEY is absent")
        health_before = _request_json(f"{BRIDGE_BASE_URL}/health", api_key)
        remaining_before = validate_bridge_gate(
            health_before, authorized_total_usd=authorized_total_usd
        )
    else:
        if replay_provenance is None:
            raise LaunchGateError("archived solution replay provenance is absent")
        normalized_override = extract_python_code(solution_code_override)
        override_sha256 = hashlib.sha256((normalized_override + "\n").encode()).hexdigest()
        health_before = validate_replay_provenance(
            competition_id=competition_id,
            solution_sha256=override_sha256,
            source_receipt=replay_provenance,
        )
        remaining_before = None

    run_id = f"{competition_id}-{uuid.uuid4().hex[:12]}"
    run_root = Path("/tmp/e9-streaming") / run_id
    data_dir = run_root / "data"
    code_dir = run_root / "code"
    submission_dir = run_root / "submission"
    artifacts_dir = run_root / "artifacts"
    for directory in (data_dir, code_dir, submission_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    try:
        try:
            prepare = _run(
                [
                    "/opt/conda/bin/conda",
                    "run",
                    "-n",
                    "mleb",
                    "mlebench",
                    "prepare",
                    "-c",
                    competition_id,
                    "--data-dir",
                    str(data_dir),
                ],
                timeout=3600,
            )
            public_dir = data_dir / competition_id / "prepared/public"
            private_dir = data_dir / competition_id / "prepared/private"
            if not public_dir.is_dir() or not private_dir.is_dir():
                raise RuntimeError("native prepare did not create public/private datasets")
        except Exception as exc:
            if generation_backend != "merged_vllm":
                raise
            elapsed_seconds = time.monotonic() - started
            return _cpu_prepare_failure_result(
                competition_id=competition_id,
                reservation_id=merged_vllm_reservation_id or "",
                launch_nonce=merged_vllm_launch_nonce or "",
                reservation=merged_arm_reservation or {},
                elapsed_seconds=elapsed_seconds,
                error=exc,
                recorded_at_utc=_utc_now(),
            )

        deterministic_repair = None
        generation: dict[str, Any] | None = None
        if solution_code_override is None:
            prompt = _agent_prompt(competition_id, public_dir)
            if generation_backend == "merged_vllm":
                from e9_merged_vllm_arm import build_allocation_attestation_v2

                required_contract = {
                    "public_key_base64",
                    "prompt_builder_sha256",
                    "deployment_revision",
                    "source_bundle_sha256",
                }
                if any(field not in merged_vllm_v3_contract for field in required_contract):
                    raise LaunchGateError("merged-vLLM v3 provenance contract is incomplete")
                issued_at = _utc_now()
                expires_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=300)
                ).isoformat(timespec="seconds")
                allocation_attestation = build_allocation_attestation_v2(
                    competition_id=competition_id,
                    reservation_id=merged_vllm_reservation_id or "",
                    launch_nonce=merged_vllm_launch_nonce or "",
                    prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    prompt_builder_sha256=merged_vllm_v3_contract["prompt_builder_sha256"],
                    deployment_revision=merged_vllm_v3_contract["deployment_revision"],
                    source_bundle_sha256=merged_vllm_v3_contract["source_bundle_sha256"],
                    wandb_receipt=allocation_gate["wandb_receipt"],
                    checkpoint_verification=allocation_gate["checkpoint_verification"],
                    reservation=merged_arm_reservation or {},
                    issued_at_utc=issued_at,
                    expires_at_utc=expires_at,
                    private_key_seed_base64=os.environ.get(
                        "E9_ALLOCATION_ATTESTATION_PRIVATE_KEY_SEED_BASE64", ""
                    ),
                )
                generation = E9MergedVllmGenerator(
                    allocation_gate_json=json.dumps(
                        allocation_gate, sort_keys=True, separators=(",", ":")
                    )
                ).generate_program.remote(
                    prompt, competition_id, allocation_attestation["attestation_sha256"]
                )
                response_text = generation["response_text"]
            else:
                completion = _request_json(
                    f"{BRIDGE_API_BASE}/chat/completions",
                    api_key,
                    {
                        "model": MODEL_ALIAS,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Return only compact executable Python. Start with an "
                                    "import statement. Never emit analysis or Markdown. "
                                    "Use compact vectorized features instead of long "
                                    "handwritten feature lists; stay under 120 lines."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4096,
                        "seed": 809,
                        "enable_thinking": False,
                    },
                )
                response_text = completion["choices"][0]["message"]["content"]
            solution_code = extract_python_code(response_text)
            if generation is not None:
                generation["program"] = solution_code
                generation["program_sha256"] = hashlib.sha256(
                    (solution_code + "\n").encode("utf-8")
                ).hexdigest()
            sample_reused = False
        else:
            solution_code = extract_python_code(solution_code_override)
            sample_reused = True
            preapplied_solution_code = preapply_replay_runtime_repair(
                solution_code, replay_provenance or {}
            )
            if preapplied_solution_code != solution_code:
                solution_code = preapplied_solution_code
                deterministic_repair = identify_known_deterministic_repair(solution_code)
            aligned_solution_code = repair_known_submission_alignment(solution_code)
            if aligned_solution_code != solution_code:
                solution_code = aligned_solution_code
                if "for i in range(row['seq_length']):" in solution_code:
                    deterministic_repair = (
                        "selected_stanford_per_position_features_targets_and_full_test_length"
                    )
                elif "'breed' not in s.columns" in solution_code:
                    deterministic_repair = "selected_dog_breed_sample_submission_not_labels"
                elif "cv2.findContours" in solution_code and "unicode_map" in solution_code:
                    deterministic_repair = identify_known_deterministic_repair(solution_code)
                elif "'labels' in os.path.basename(f).lower()" in solution_code:
                    deterministic_repair = identify_known_deterministic_repair(solution_code)
                elif (
                    "pd.read_csv(os.path.join(DATA, 'train.csv'))" in solution_code
                    and "pd.read_csv(os.path.join(DATA, 'test.csv'))" in solution_code
                ):
                    deterministic_repair = identify_known_deterministic_repair(solution_code)
                else:
                    deterministic_repair = (
                        identify_known_deterministic_repair(solution_code)
                        or "normalized_archive_member_clip_names_to_basenames"
                    )
            if deterministic_repair is None:
                deterministic_repair = identify_known_deterministic_repair(solution_code)
        solution_path = code_dir / "solution.py"
        solution_path.write_text(solution_code + "\n", encoding="utf-8")

        # Match the upstream privacy boundary: generated code is nonroot and
        # cannot read the held-out labels in the owner-only private directory.
        for link, target in (
            (Path("/home/data"), public_dir),
            (Path("/home/code"), code_dir),
            (Path("/home/submission"), submission_dir),
        ):
            if link.is_symlink() or link.is_file():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
            link.symlink_to(target, target_is_directory=True)
        os.chmod(private_dir, 0o700)
        os.chmod(data_dir / competition_id / "prepared", 0o755)
        os.chmod(public_dir, 0o555)
        for path in public_dir.rglob("*"):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(code_dir, 0o777)
        os.chmod(submission_dir, 0o777)
        os.chmod(solution_path, 0o644)

        agent_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": "/home/nonroot",
            "PYTHONHASHSEED": "809",
        }
        agent_command = [
            "runuser",
            "-u",
            "nonroot",
            "--",
            "/opt/conda/bin/conda",
            "run",
            "-n",
            "agent",
            "python",
            str(solution_path),
        ]
        agent_failure = None
        agent_timeout_seconds = 1800 if sample_reused else 900
        try:
            agent = _run(agent_command, timeout=agent_timeout_seconds, env=agent_env)
        except RuntimeError as exc:
            try:
                solution_code = repair_known_runtime_error(solution_code, str(exc))
            except LaunchGateError:
                agent = None
                agent_failure = str(exc)
            else:
                solution_path.write_text(solution_code + "\n", encoding="utf-8")
                if "for frame_name in ('train', 'test'):" in solution_code:
                    deterministic_repair = (
                        "fixed_champs_type_one_hot_assignment_and_removed_invalid_eval_set"
                    )
                elif "y_train = (train['target'].values >= 0.5).astype(int)" in solution_code:
                    deterministic_repair = (
                        "thresholded_continuous_toxicity_target_for_logistic_regression"
                    )
                elif "row_copy[target] = row[target][i]" in solution_code:
                    deterministic_repair = "selected_stanford_per_position_features_and_targets"
                elif "test_scan_matches = glob.glob(os.path.join(" in solution_code:
                    deterministic_repair = (
                        "fixed_uw_train_and_test_scan_paths_and_vector_feature_mapping"
                    )
                elif "scan_matches = glob.glob(os.path.join(" in solution_code:
                    deterministic_repair = "fixed_uw_scan_paths_and_vector_feature_mapping"
                elif (
                    "test['sentence_id'].astype(str) + '_' + "
                    "test['token_id'].astype(str)" in solution_code
                ):
                    deterministic_repair = (
                        "guarded_russian_text_rules_and_constructed_official_submission_ids"
                    )
                elif "if not isinstance(text, str):" in solution_code:
                    deterministic_repair = "guarded_russian_text_rules_against_non_string_values"
                elif "ru_train*.csv*" in solution_code:
                    deterministic_repair = (
                        "expanded_russian_csv_globs_to_include_single_file_zip_archives"
                    )
                elif "drop(columns=[id_col], errors='ignore').reset_index()" in solution_code:
                    deterministic_repair = "dropped_duplicate_id_column_before_index_reset"
                elif "quadratic_weighted_kappa" in str(exc):
                    deterministic_repair = "removed_unused_unavailable_sklearn_metric_import"
                elif "No module named 'cv2'" in str(exc):
                    deterministic_repair = "replaced_unavailable_cv2_grayscale_reads_with_pillow"
                elif "No module named 'librosa'" in str(exc):
                    deterministic_repair = "removed_redundant_unguarded_librosa_import"
                elif "Expected 2 fields in line 2, saw 40" in str(exc):
                    deterministic_repair = "skipped_mlsp_segment_feature_header"
                elif "legacy multi-label data representation" in str(exc):
                    deterministic_repair = "used_multilabel_binarizer_for_mlsp_targets"
                elif "Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]" in str(exc):
                    deterministic_repair = "zero_based_xgboost_labels_and_restored_predictions"
                elif "['combinations'] not in index" in str(exc):
                    deterministic_repair = "used_native_plant_pathology_multiple_diseases_label"
                elif "Expected 2D array, got 1D array instead" in str(exc) and "array=[]" in str(
                    exc
                ).replace(" ", ""):
                    deterministic_repair = "normalized_dog_breed_image_ids_to_filename_stems"
                elif "index 5789880 is out of bounds" in str(exc):
                    deterministic_repair = "converted_one_based_pixel_coordinates_to_zero_based"
                elif "UnicodeDecodeError" in str(exc) and "extract_single_7z" in solution_code:
                    deterministic_repair = "extracted_single_file_7z_inputs_before_loading"
                elif "invalid literal for int() with base 10: 'rec_id'" in str(exc):
                    if "rec2file[int(parts[0])]" in str(exc):
                        deterministic_repair = "skipped_mlsp_filename_mapping_header"
                    else:
                        deterministic_repair = "skipped_mlsp_bird_label_header_before_integer_parse"
                else:
                    deterministic_repair = "added_missing_scipy_sparse_hstack_import"
                try:
                    agent = _run(
                        agent_command,
                        timeout=agent_timeout_seconds,
                        env=agent_env,
                    )
                except RuntimeError as repaired_exc:
                    agent = None
                    agent_failure = str(repaired_exc)
        submission = submission_dir / "submission.csv"
        grade_path = artifacts_dir / "native_grade.json"
        if agent_failure is None:
            native_grade = _grade(
                data_dir,
                competition_id,
                submission,
                grade_path,
            )
        else:
            native_grade = {
                "competition_id": competition_id,
                "score": None,
                "valid_submission": False,
                "submission_exists": submission.is_file(),
                "agent_execution_failed": True,
                "grading_error": "sampled agent program exited nonzero",
                "agent_error_tail": agent_failure[-3000:],
            }
            grade_path.write_text(json.dumps(native_grade, sort_keys=True), encoding="utf-8")
        elapsed_seconds = time.monotonic() - started
        artifacts = {
            "solution_sha256": _sha256(solution_path),
            "submission_sha256": (_sha256(submission) if submission.is_file() else None),
            "native_grade_sha256": _sha256(artifacts_dir / "native_grade.json"),
        }
        common_receipt_fields = {
            "run_id": run_id,
            "mle_bench_commit": MLE_BENCH_COMMIT,
            "sample_reused": sample_reused,
            "seed": 809,
            "deterministic_repair": deterministic_repair,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "artifacts": artifacts,
            "prepare_log_tail": (prepare.stdout + prepare.stderr)[-3000:],
            "agent_log_tail": (
                (agent.stdout + agent.stderr)[-3000:]
                if agent is not None
                else agent_failure[-3000:]
            ),
        }
        if generation_backend == "merged_vllm":
            from e9_merged_vllm_arm import build_task_receipt_v3

            if generation is None:
                raise LaunchGateError("merged-vLLM generation receipt is absent")
            native_grade = {**native_grade, "competition_id": competition_id}
            grade_path.write_text(json.dumps(native_grade, sort_keys=True), encoding="utf-8")
            artifacts["native_grade_sha256"] = _sha256(grade_path)
            receipt = build_task_receipt_v3(
                competition_id=competition_id,
                native_grade=native_grade,
                generation=generation,
                checkpoint_verification=generation["checkpoint_verification"],
                reservation=merged_arm_reservation or {},
                allocation_gate=allocation_gate,
                allocation_attestation=allocation_attestation,
                allocation_attestation_public_key_base64=merged_vllm_v3_contract[
                    "public_key_base64"
                ],
                reservation_id=merged_vllm_reservation_id or "",
                launch_nonce=merged_vllm_launch_nonce or "",
                prompt=prompt,
                prompt_builder_sha256=merged_vllm_v3_contract["prompt_builder_sha256"],
                deployment_revision=merged_vllm_v3_contract["deployment_revision"],
                source_bundle_sha256=merged_vllm_v3_contract["source_bundle_sha256"],
            )
            actual_gpu_usd = float(generation["estimated_total_gpu_usd"])
            actual_cpu_usd = elapsed_seconds * MERGED_VLLM_CPU_RATE_USD_PER_SECOND
            receipt.update(
                {
                    **common_receipt_fields,
                    "agent": "pavlov_merged_vllm_single_call_ml_agent_v1",
                    "merged_vllm_budget": {
                        "mode": "fresh_merged_vllm_generation_and_native_grade",
                        "estimated_modal_gpu_usd": round(actual_gpu_usd, 9),
                        "estimated_modal_cpu_usd": round(actual_cpu_usd, 9),
                        "estimated_combined_usd": round(actual_gpu_usd + actual_cpu_usd, 9),
                        "reservation": dict(merged_arm_reservation or {}),
                    },
                }
            )
            receipt["generation_prompt"] = prompt
            receipt.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = hashlib.sha256(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            health_after = None
            remaining_after = None
        else:
            if sample_reused:
                health_after = health_before
                remaining_after = None
            else:
                health_after = _request_json(f"{BRIDGE_BASE_URL}/health", api_key)
                remaining_after = validate_bridge_gate(
                    health_after, authorized_total_usd=authorized_total_usd
                )
            receipt = build_run_receipt(
                competition_id=competition_id,
                native_grade=native_grade,
                bridge_health=health_after,
            )
            receipt.update(
                {
                    **common_receipt_fields,
                    "agent": "pavlov_tinker_single_call_ml_agent_v1",
                    "bridge_budget": {
                        "mode": (
                            "deterministically_repaired_archived_solution_zero_generation_replay"
                            if sample_reused and deterministic_repair
                            else "archived_solution_zero_generation_replay"
                            if sample_reused
                            else "live_tinker_generation"
                        ),
                        "remaining_before_usd": remaining_before,
                        "remaining_after_usd": remaining_after,
                        "charged_this_run_usd": (
                            0.0 if sample_reused else round(remaining_before - remaining_after, 9)
                        ),
                        "maximum_usd": (
                            None if sample_reused else health_after["budget"]["maximum_usd"]
                        ),
                    },
                }
            )
        if sample_reused:
            receipt["replay_provenance"] = dict(health_after)
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "receipt": receipt,
            "solution_code": solution_code,
            "submission_csv": (
                submission.read_text(encoding="utf-8", errors="replace")
                if submission.is_file()
                else None
            ),
            "native_grade_json": native_grade,
        }
    finally:
        # Ephemeral streaming cleanup.  Refuse to remove anything outside the
        # exact run root even if a future refactor changes a path unexpectedly.
        expected_parent = Path("/tmp/e9-streaming").resolve()
        resolved = run_root.resolve()
        if resolved.parent == expected_parent and resolved.name == run_id:
            shutil.rmtree(resolved, ignore_errors=True)


@app.local_entrypoint(name="probe-vllm-pilot")
def probe_vllm_pilot(
    competition_id: str = "hotel-id-2021-fgvc8",
) -> None:
    """Print the no-download rule-access gate for the prospective pilot."""

    result = probe_kaggle_rules.remote(competition_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("accepted") is not True:
        raise RuntimeError("competition download access is not accepted")


@app.local_entrypoint()
def main(
    competition_id: str = DEFAULT_COMPETITION,
    authorized_total_usd: float = 55.91445263,
    output_dir: str = "outputs/e9_mle_bench/modal_streaming",
    solution_path: str = "",
    submission_path: str = "",
) -> None:
    from e9_mle_bench_streaming import (
        extract_python_code,
        validate_replay_provenance,
        validate_submission_regrade_provenance,
    )

    if solution_path and submission_path:
        raise ValueError("solution_path and submission_path are mutually exclusive")
    if submission_path:
        source_submission = Path(submission_path)
        submission_bytes = source_submission.read_bytes()
        source_receipt_path = source_submission.with_name("receipt.json")
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        submission_sha256 = hashlib.sha256(submission_bytes).hexdigest()
        validate_submission_regrade_provenance(
            competition_id=competition_id,
            submission_sha256=submission_sha256,
            source_receipt=source_receipt,
        )
        result = regrade_saved_submission.remote(
            competition_id,
            submission_bytes,
            source_receipt,
        )
        destination = Path(output_dir) / result["receipt"]["run_id"]
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "receipt.json").write_text(
            json.dumps(result["receipt"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "submission.csv").write_bytes(result["submission_bytes"])
        (destination / "native_grade.json").write_text(
            json.dumps(result["native_grade_json"], sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"output_dir": str(destination), "receipt": result["receipt"]},
                indent=2,
            )
        )
        return

    solution_override = None
    replay_provenance = None
    if solution_path:
        source_solution = Path(solution_path)
        solution_override = source_solution.read_text(encoding="utf-8")
        source_receipt_path = source_solution.with_name("receipt.json")
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        normalized_solution = extract_python_code(solution_override)
        normalized_solution_sha256 = hashlib.sha256(
            (normalized_solution + "\n").encode()
        ).hexdigest()
        validate_replay_provenance(
            competition_id=competition_id,
            solution_sha256=normalized_solution_sha256,
            source_receipt=source_receipt,
        )
        replay_provenance = source_receipt
    result = run_one.remote(
        competition_id,
        authorized_total_usd,
        solution_override,
        replay_provenance,
    )
    destination = Path(output_dir) / result["receipt"]["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "receipt.json").write_text(
        json.dumps(result["receipt"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "solution.py").write_text(result["solution_code"] + "\n", encoding="utf-8")
    if result["submission_csv"] is not None:
        (destination / "submission.csv").write_text(result["submission_csv"], encoding="utf-8")
    (destination / "native_grade.json").write_text(
        json.dumps(result["native_grade_json"], sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(destination), "receipt": result["receipt"]}, indent=2))


def _validated_solution_replay_inputs(
    competition_id: str, solution_path: str
) -> tuple[str, dict[str, Any], Path, Path, str]:
    """Load and fail-closed validate an archived solution before remote replay."""

    from e9_mle_bench_streaming import extract_python_code, validate_replay_provenance

    if not solution_path:
        raise ValueError("solution_path is required for a zero-generation replay")
    source_solution = Path(solution_path)
    solution_override = source_solution.read_text(encoding="utf-8")
    source_receipt_path = source_solution.with_name("receipt.json")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    normalized_solution = extract_python_code(solution_override)
    normalized_solution_sha256 = hashlib.sha256((normalized_solution + "\n").encode()).hexdigest()
    validate_replay_provenance(
        competition_id=competition_id,
        solution_sha256=normalized_solution_sha256,
        source_receipt=source_receipt,
    )
    return (
        solution_override,
        source_receipt,
        source_solution,
        source_receipt_path,
        normalized_solution_sha256,
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Replace one artifact atomically within its destination directory."""

    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _materialize_run_result(result: dict[str, Any], output_dir: str) -> Path:
    """Idempotently persist one returned result without altering its payload."""

    destination = Path(output_dir) / result["receipt"]["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    expected = {
        "receipt.json": (json.dumps(result["receipt"], indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        "solution.py": (result["solution_code"] + "\n").encode("utf-8"),
        "native_grade.json": json.dumps(result["native_grade_json"], sort_keys=True).encode(
            "utf-8"
        ),
    }
    if result["submission_csv"] is not None:
        expected["submission.csv"] = result["submission_csv"].encode("utf-8")
    elif (destination / "submission.csv").exists():
        raise ValueError("existing materialization has an unexpected submission.csv")
    for name, payload in expected.items():
        path = destination / name
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"existing materialized artifact differs: {path}")
            continue
        _write_bytes_atomic(path, payload)
    return destination


def _validate_returned_replay_result(
    result: dict[str, Any], launch_receipt: dict[str, Any]
) -> None:
    """Bind a returned result to the immutable replay launch contract."""

    from e9_mle_bench_streaming import validate_replay_provenance

    if not isinstance(result, dict) or not isinstance(result.get("receipt"), dict):
        raise ValueError("remote replay returned no receipt object")
    receipt = result["receipt"]
    competition_id = str(launch_receipt["competition_id"])
    if receipt.get("competition_id") != competition_id:
        raise ValueError("remote receipt competition does not match launch receipt")
    run_id = receipt.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id.startswith(f"{competition_id}-")
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        or "\\" in run_id
    ):
        raise ValueError("remote receipt run_id is not a safe competition-scoped basename")
    stored_receipt_sha256 = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    expected_receipt_sha256 = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored_receipt_sha256 != expected_receipt_sha256:
        raise ValueError("remote receipt self-hash is invalid")
    source_receipt_path = Path(str(launch_receipt["source_receipt"]))
    if _sha256(source_receipt_path) != launch_receipt["source_receipt_sha256"]:
        raise ValueError("launch source receipt file hash drifted before collection")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    expected_provenance = validate_replay_provenance(
        competition_id=competition_id,
        solution_sha256=str(launch_receipt["source_solution_sha256"]),
        source_receipt=source_receipt,
    )
    if receipt.get("replay_provenance") != expected_provenance:
        raise ValueError("remote receipt replay provenance does not match launch source")
    if receipt.get("sample_reused") is not True:
        raise ValueError("remote replay unexpectedly performed a new sample")
    bridge_budget = receipt.get("bridge_budget")
    if not isinstance(bridge_budget, dict) or bridge_budget.get("charged_this_run_usd") != 0.0:
        raise ValueError("remote replay receipt does not prove zero Tinker charge")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("remote receipt artifact hashes are absent")
    solution_code = result.get("solution_code")
    if not isinstance(solution_code, str):
        raise ValueError("remote result solution_code is absent")
    solution_sha256 = hashlib.sha256((solution_code + "\n").encode()).hexdigest()
    if artifacts.get("solution_sha256") != solution_sha256:
        raise ValueError("remote solution payload does not match receipt")
    native_grade = result.get("native_grade_json")
    native_grade_bytes = json.dumps(native_grade, sort_keys=True).encode("utf-8")
    if artifacts.get("native_grade_sha256") != hashlib.sha256(native_grade_bytes).hexdigest():
        raise ValueError("remote native grade payload does not match receipt")
    submission_csv = result.get("submission_csv")
    submission_sha256 = (
        hashlib.sha256(submission_csv.encode("utf-8")).hexdigest()
        if isinstance(submission_csv, str)
        else None
    )
    if artifacts.get("submission_sha256") != submission_sha256:
        raise ValueError("remote submission payload does not match receipt")


def _returned_control_metadata(
    result: dict[str, Any], launch_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Derive collection control metadata from immutable source and returned receipts."""

    source_receipt_path = Path(str(launch_receipt["source_receipt"]))
    if _sha256(source_receipt_path) != launch_receipt["source_receipt_sha256"]:
        raise ValueError("launch source receipt file hash drifted before collection")
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    returned_receipt = result["receipt"]
    scientific_status = returned_receipt.get("status")
    coverage_increment = int(
        scientific_status == "NATIVE_SINGLE_COMPETITION_GRADED"
        and source_receipt.get("status") != "NATIVE_SINGLE_COMPETITION_GRADED"
    )
    return {
        "scientific_status": scientific_status,
        "competition_score": returned_receipt.get("competition_score"),
        "score": returned_receipt.get("score"),
        "coverage_increment": coverage_increment,
    }


def _validate_spawn_budget_and_queue(*, output_dir: str, spend_ledger_path: str) -> dict[str, Any]:
    """Fail closed on duplicate launches or insufficient remaining Modal budget."""

    pending_dir = Path(output_dir) / "_pending"
    unresolved: list[str] = []
    if pending_dir.is_dir():
        for path in sorted(pending_dir.glob("*.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("status") == "REMOTE_CALL_SPAWNED_UNCOLLECTED":
                unresolved.append(str(path))
    if unresolved:
        raise RuntimeError(
            "another recoverable E9 call is unresolved; collect it before spawning: "
            + ", ".join(unresolved)
        )
    ledger_path = Path(spend_ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    remaining = ledger.get("remaining_authorized_incremental_spend_usd")
    worst_case_modal_usd = round(7200 * 0.00007016, 9)
    reservations_path = pending_dir / "spend_reservations.json"
    reservations = (
        json.loads(reservations_path.read_text(encoding="utf-8"))
        if reservations_path.is_file()
        else {"schema_version": "e9-modal-spend-reservations-v1", "reservations": []}
    )
    existing_reserved = round(
        sum(
            float(item.get("reserved_usd", 0.0))
            for item in reservations.get("reservations", [])
            if item.get("reconciled_to_spend_ledger") is not True
        ),
        9,
    )
    available_after_reservations = (
        round(float(remaining) - existing_reserved, 9)
        if isinstance(remaining, (int, float))
        else None
    )
    if (
        ledger.get("within_cap") is not True
        or not isinstance(remaining, (int, float))
        or available_after_reservations is None
        or available_after_reservations < worst_case_modal_usd
    ):
        raise RuntimeError("campaign ledger cannot cover one worst-case E9 replay")
    return {
        "spend_ledger": str(ledger_path),
        "spend_ledger_sha256": _sha256(ledger_path),
        "remaining_authorized_incremental_spend_usd": remaining,
        "existing_unreconciled_reservations_usd": existing_reserved,
        "available_after_existing_reservations_usd": available_after_reservations,
        "worst_case_modal_replay_usd": worst_case_modal_usd,
        "unresolved_launches_before_spawn": 0,
    }


@app.local_entrypoint(name="spawn-replay")
def spawn_replay(
    competition_id: str = DEFAULT_COMPETITION,
    authorized_total_usd: float = 55.91445263,
    output_dir: str = "outputs/e9_mle_bench/modal_streaming",
    solution_path: str = "",
    spend_ledger_path: str = "outputs/e1_e14_incremental_spend_ledger_2026-08-29.json",
) -> None:
    """Spawn a replay and persist its FunctionCall ID before the client exits."""

    (
        solution_override,
        replay_provenance,
        source_solution,
        source_receipt_path,
        normalized_solution_sha256,
    ) = _validated_solution_replay_inputs(competition_id, solution_path)
    pending_dir = Path(output_dir) / "_pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    lock_path = pending_dir / ".spawn.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        budget_gate = _validate_spawn_budget_and_queue(
            output_dir=output_dir, spend_ledger_path=spend_ledger_path
        )
        reservations_path = pending_dir / "spend_reservations.json"
        reservations = (
            json.loads(reservations_path.read_text(encoding="utf-8"))
            if reservations_path.is_file()
            else {
                "schema_version": "e9-modal-spend-reservations-v1",
                "reservations": [],
            }
        )
        reservation = {
            "reservation_id": uuid.uuid4().hex,
            "competition_id": competition_id,
            "status": "PROVISIONAL_BEFORE_REMOTE_SPAWN",
            "created_at_epoch": time.time(),
            "reserved_usd": budget_gate["worst_case_modal_replay_usd"],
            "reconciled_to_spend_ledger": False,
        }
        reservations["reservations"].append(reservation)
        _write_json_atomic(reservations_path, reservations)
        try:
            deployed_run_one = modal.Function.from_name(APP_NAME, "run_one")
            function_call = deployed_run_one.spawn(
                competition_id,
                authorized_total_usd,
                solution_override,
                replay_provenance,
            )
        except Exception as exc:
            reservation.update(
                {
                    "status": "SPAWN_FAILED_RELEASED",
                    "reserved_usd": 0.0,
                    "spawn_error_type": type(exc).__name__,
                    "spawn_error_message": str(exc) or repr(exc),
                }
            )
            _write_json_atomic(reservations_path, reservations)
            raise
        launch_receipt = {
            "schema_version": "e9-modal-spawn-replay-v1",
            "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
            "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
            "score": None,
            "coverage_increment": 0,
            "competition_id": competition_id,
            "spawn_target": {
                "app_name": APP_NAME,
                "function_name": "run_one",
                "deployment_required": True,
            },
            "function_call_id": function_call.object_id,
            "created_at_epoch": time.time(),
            "authorized_total_usd": authorized_total_usd,
            "output_dir": output_dir,
            "source_solution": str(source_solution),
            "source_solution_sha256": normalized_solution_sha256,
            "source_receipt": str(source_receipt_path),
            "source_receipt_sha256": _sha256(source_receipt_path),
            "budget_gate": budget_gate,
            "spend_reservation_id": reservation["reservation_id"],
            "claim_boundary": (
                "A spawned FunctionCall is not a valid submission, native grade, or score. "
                "Only collect-replay may materialize and expose the returned native receipt."
            ),
        }
        launch_path = pending_dir / f"{competition_id}-{function_call.object_id}.json"
        _write_json_atomic(launch_path, launch_receipt)
        reservation.update(
            {
                "status": "REMOTE_CALL_RESERVED_UNRECONCILED",
                "function_call_id": function_call.object_id,
                "launch_receipt": str(launch_path),
            }
        )
        _write_json_atomic(reservations_path, reservations)
    print(
        json.dumps(
            {
                "launch_receipt": str(launch_path),
                "function_call_id": function_call.object_id,
                "status": launch_receipt["status"],
            },
            indent=2,
        )
    )


@app.local_entrypoint(name="collect-replay")
def collect_replay(
    launch_receipt_path: str,
    timeout_seconds: float = 5.0,
) -> None:
    """Retrieve a spawned call by ID and materialize its immutable result."""

    launch_path = Path(launch_receipt_path)
    launch_receipt = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch_receipt.get("schema_version") != "e9-modal-spawn-replay-v1":
        raise ValueError("unsupported E9 spawn receipt schema")
    function_call_id = str(launch_receipt.get("function_call_id") or "")
    if not function_call_id.startswith("fc-"):
        raise ValueError("spawn receipt function_call_id is invalid")
    if launch_receipt.get("status") == "REMOTE_RESULT_MATERIALIZED":
        destination = Path(str(launch_receipt["materialized_output_dir"]))
        solution_code = (destination / "solution.py").read_text(encoding="utf-8")
        if solution_code.endswith("\n"):
            solution_code = solution_code[:-1]
        result = {
            "receipt": json.loads((destination / "receipt.json").read_text(encoding="utf-8")),
            "solution_code": solution_code,
            "submission_csv": (
                (destination / "submission.csv").read_text(encoding="utf-8")
                if (destination / "submission.csv").is_file()
                else None
            ),
            "native_grade_json": json.loads(
                (destination / "native_grade.json").read_text(encoding="utf-8")
            ),
        }
        _validate_returned_replay_result(result, launch_receipt)
        control_metadata = _returned_control_metadata(result, launch_receipt)
        collection_receipt = json.loads(
            (destination / "collection_receipt.json").read_text(encoding="utf-8")
        )
        if (
            collection_receipt.get("status") != "REMOTE_RESULT_MATERIALIZED"
            or collection_receipt.get("function_call_id") != function_call_id
            or collection_receipt.get("remote_receipt_sha256")
            != _sha256(destination / "receipt.json")
        ):
            raise ValueError("local collection metadata does not bind to launch receipt")
        if any(collection_receipt.get(field) != value for field, value in control_metadata.items()):
            collection_receipt.update(control_metadata)
            _write_json_atomic(destination / "collection_receipt.json", collection_receipt)
        if any(launch_receipt.get(field) != value for field, value in control_metadata.items()):
            launch_receipt.update(control_metadata)
            _write_json_atomic(launch_path, launch_receipt)
        print(
            json.dumps(
                {
                    "status": "REMOTE_RESULT_ALREADY_MATERIALIZED_AND_REVALIDATED",
                    "output_dir": str(destination),
                    "collection_receipt": collection_receipt,
                },
                indent=2,
            )
        )
        return
    function_call = modal.FunctionCall.from_id(function_call_id)
    try:
        result = function_call.get(timeout=timeout_seconds)
    except TimeoutError:
        raise
    except Exception as exc:
        failure_receipt = {
            "schema_version": "e9-modal-collect-replay-v1",
            "status": "REMOTE_CALL_FAILED_UNMATERIALIZED",
            "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
            "score": None,
            "coverage_increment": 0,
            "function_call_id": function_call_id,
            "competition_id": launch_receipt["competition_id"],
            "observed_at_epoch": time.time(),
            "error_type": type(exc).__name__,
            "error_message": str(exc) or repr(exc),
            "claim_boundary": (
                "A remote exception without a returned native receipt is not a valid "
                "submission, native grade, or score."
            ),
        }
        launch_receipt.update(
            {
                "status": failure_receipt["status"],
                "scientific_status": failure_receipt["scientific_status"],
                "score": None,
                "coverage_increment": 0,
                "remote_error_observed_at_epoch": failure_receipt["observed_at_epoch"],
                "remote_error_type": failure_receipt["error_type"],
                "remote_error_message": failure_receipt["error_message"],
            }
        )
        _write_json_atomic(launch_path, launch_receipt)
        print(json.dumps({"launch_receipt": str(launch_path), **failure_receipt}, indent=2))
        raise RuntimeError(
            f"remote call {function_call_id} failed without a materialized receipt"
        ) from exc
    try:
        _validate_returned_replay_result(result, launch_receipt)
    except Exception as exc:
        rejection_receipt = {
            "schema_version": "e9-modal-collect-replay-v1",
            "status": "REMOTE_RESULT_REJECTED_UNBOUND",
            "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
            "score": None,
            "coverage_increment": 0,
            "function_call_id": function_call_id,
            "competition_id": launch_receipt["competition_id"],
            "observed_at_epoch": time.time(),
            "error_type": type(exc).__name__,
            "error_message": str(exc) or repr(exc),
            "claim_boundary": (
                "A remote payload was returned but failed immutable launch binding; "
                "it is not accepted as a native receipt or score."
            ),
        }
        launch_receipt.update(
            {
                "status": rejection_receipt["status"],
                "scientific_status": rejection_receipt["scientific_status"],
                "score": None,
                "coverage_increment": 0,
                "rejected_at_epoch": rejection_receipt["observed_at_epoch"],
                "rejection_error_type": rejection_receipt["error_type"],
                "rejection_error_message": rejection_receipt["error_message"],
            }
        )
        _write_json_atomic(launch_path, launch_receipt)
        print(json.dumps({"launch_receipt": str(launch_path), **rejection_receipt}, indent=2))
        raise RuntimeError(
            f"remote result {function_call_id} failed immutable launch binding"
        ) from exc
    destination = _materialize_run_result(result, str(launch_receipt["output_dir"]))
    remote_receipt_path = destination / "receipt.json"
    control_metadata = _returned_control_metadata(result, launch_receipt)
    collection_receipt = {
        "schema_version": "e9-modal-collect-replay-v1",
        "status": "REMOTE_RESULT_MATERIALIZED",
        "function_call_id": function_call_id,
        "competition_id": launch_receipt["competition_id"],
        "collected_at_epoch": time.time(),
        "output_dir": str(destination),
        "remote_run_id": result["receipt"]["run_id"],
        "remote_receipt_sha256": _sha256(remote_receipt_path),
        **control_metadata,
        "claim_boundary": (
            "The remote receipt controls scientific status. Collection itself does not "
            "promote a partial competition result to an E9 suite score."
        ),
    }
    _write_json_atomic(destination / "collection_receipt.json", collection_receipt)
    launch_receipt.update(
        {
            "status": "REMOTE_RESULT_MATERIALIZED",
            "collected_at_epoch": collection_receipt["collected_at_epoch"],
            "materialized_output_dir": str(destination),
            "remote_receipt_sha256": collection_receipt["remote_receipt_sha256"],
            **control_metadata,
        }
    )
    _write_json_atomic(launch_path, launch_receipt)
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "collection_receipt": collection_receipt,
                "receipt": result["receipt"],
            },
            indent=2,
        )
    )
