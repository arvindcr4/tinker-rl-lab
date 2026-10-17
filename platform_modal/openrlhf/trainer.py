"""Fail-closed, versioned OpenRLHF 0.8.0 GRPO launcher.

Simulation is opt-in and has a different schema. This driver does not claim
objective equivalence with another framework. No checkpoint is uploaded here.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .config import OpenRLHFConfig

PROTOCOL = "openrlhf-grpo-v2-candidate"
BACKEND_VERSION = "0.8.0"
GSM8K_SYSTEM_PROMPT = "You are a math assistant. Solve step by step, then give your final answer inside \\boxed{}."
OBJECTIVE = {
    "advantage_estimator": "group_norm", "reward_std_correction": 1,
    "advantage_epsilon": 1e-8, "token_reduction": "masked_mean_per_response_then_batch_mean",
    "kl_estimator": "k3", "kl_application": "actor_loss_only",
    "reference_policy": "frozen_initial_policy", "reference_update": "none",
    "backend_parity": "requires_exact_image_smoke_test",
}


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_empirical_result(result: dict) -> dict:
    if result.get("data_kind") != "empirical" or result.get("state") != "completed":
        raise ValueError("Only completed empirical results may enter measured analyses")
    return result


def run_simulation(config: OpenRLHFConfig) -> dict:
    """Uncalibrated random fixture, never an empirical training result."""
    rng = np.random.default_rng(config.seed)
    return {"schema": "openrlhf-simulation-fixture-v1", "data_kind": "synthetic_fixture",
            "mode": "dryrun", "seed": config.seed,
            "fixture_rewards": rng.uniform(size=config.max_steps or config.epochs).tolist()}


class OpenRLHFTrainer:
    def __init__(self, config: OpenRLHFConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    async def setup(self) -> None:
        if not self.dry_run:
            validate_backend()

    async def run(self) -> dict:
        if self.dry_run:
            return run_simulation(self.config)
        return await asyncio.to_thread(run_openrlhf_training, self.config)


def validate_backend() -> None:
    try:
        version = importlib.metadata.version("openrlhf")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Real execution requires openrlhf==0.8.0; simulation is opt-in") from exc
    if version != BACKEND_VERSION:
        raise RuntimeError(f"Unsupported OpenRLHF {version}; required {BACKEND_VERSION}")


def effective_config(config: OpenRLHFConfig) -> dict:
    """Reject treatments this driver cannot implement; derive integer work budgets."""
    numeric = [config.algorithm.gamma, config.algorithm.lam, config.algorithm.epsilon,
               config.algorithm.kl_coef, config.algorithm.temperature, config.learning_rate,
               config.optimizer.beta1, config.optimizer.beta2, config.optimizer.eps,
               config.optimizer.weight_decay, config.gpu_memory_utilization]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Numeric experiment settings must be finite")
    if config.algorithm.algorithm.lower() != "grpo":
        raise ValueError("This driver supports only algorithm=grpo")
    unsupported = {
        "gamma": (config.algorithm.gamma, 1.0), "lam": (config.algorithm.lam, 0.95),
        "optimizer.eps": (config.optimizer.eps, 1e-8),
        "num_workers": (config.num_workers, 1), "num_actors": (config.num_actors, 1),
        "critic_num_gpus": (config.critic_num_gpus, 1),
        "inference_backend": (config.inference_backend, "vllm"), "env_class": (config.env_class, "gsm8k"),
        "diagnostic_metric": (config.algorithm.diagnostic_metric, "zvf"),
        "eval_interval": (config.eval_interval, -1),
    }
    for field, (value, default) in unsupported.items():
        if value != default:
            raise ValueError(f"Unsupported {field}={value!r}")
    if config.data.val_data or config.data.test_data:
        raise ValueError("Held-out evaluation is not implemented by this driver")
    if config.model.load_in_8bit or config.model.load_in_4bit:
        raise ValueError("Quantized actor training is not implemented by this driver")
    if config.actor_num_gpus != config.num_gpus:
        raise ValueError("actor_num_gpus must match num_gpus")
    if config.vllm_num_engines * config.tensor_parallel_size != config.num_gpus:
        raise ValueError("Colocation requires actor GPUs == vLLM engines * tensor parallel size")
    positive = (config.epochs, config.train_batch_size, config.micro_batch_size,
                config.rollout_batch_size, config.algorithm.sample_num, config.num_gpus,
                config.tensor_parallel_size, config.vllm_num_engines, config.save_interval,
                config.data.max_prompt_length, config.data.max_response_length)
    if any(x <= 0 for x in positive) or config.algorithm.sample_num < 2:
        raise ValueError("Batch, length, topology and checkpoint settings must be positive; group size >=2")
    if config.max_steps is None or config.max_steps <= 0:
        raise ValueError("Set max_steps to an explicit positive optimizer-update budget")
    if config.train_batch_size % (config.micro_batch_size * config.num_gpus):
        raise ValueError("Global training batch must divide evenly across micro-batches and actor GPUs")
    completions_per_round = config.rollout_batch_size * config.algorithm.sample_num
    if completions_per_round % config.train_batch_size:
        raise ValueError("Each rollout must contain a whole number of global training batches")
    updates_per_round = completions_per_round // config.train_batch_size * config.epochs
    if config.max_steps % updates_per_round:
        raise ValueError("max_steps must be divisible by optimizer updates per rollout")
    rounds = config.max_steps // updates_per_round
    if not (0 < config.gpu_memory_utilization < 1) or config.wall_timeout_s <= 0:
        raise ValueError("Invalid memory utilization or wall timeout")
    if not (0 < config.algorithm.epsilon < 1) or config.algorithm.temperature <= 0:
        raise ValueError("Invalid clipping epsilon or sampling temperature")
    if config.algorithm.kl_coef < 0 or config.learning_rate <= 0:
        raise ValueError("KL coefficient must be nonnegative and learning rate positive")
    if not 0 <= config.optimizer.beta1 < 1 or not 0 <= config.optimizer.beta2 < 1 or config.optimizer.weight_decay < 0:
        raise ValueError("Invalid Adam betas or weight decay")
    if config.seed < 0:
        raise ValueError("Seed must be nonnegative")
    result = config.to_dict()
    result.update(protocol=PROTOCOL, backend_version=BACKEND_VERSION, validation_status="unvalidated_candidate_protocol",
                  objective={**OBJECTIVE, "clip_bounds": [1-config.algorithm.epsilon, 1+config.algorithm.epsilon],
                             "kl_coefficient": config.algorithm.kl_coef, "epochs_per_rollout": config.epochs},
                  budget={"rollout_rounds": rounds, "optimizer_updates": config.max_steps,
                          "prompt_presentations": rounds * config.rollout_batch_size,
                          "generated_completions": rounds * completions_per_round,
                          "generated_tokens": None, "counter_source": "configured_not_observed"})
    return result


def build_command(config: OpenRLHFConfig, run_dir: Path, data_path: Path, model_path: str) -> list[str]:
    effective = effective_config(config)
    values = {
        "pretrain": model_path, "prompt_data": str(data_path.resolve()),
        "input_key": "prompt", "label_key": "label", "advantage_estimator": "group_norm",
        "n_samples_per_prompt": config.algorithm.sample_num,
        "train_batch_size": config.train_batch_size, "micro_train_batch_size": config.micro_batch_size,
        "rollout_batch_size": config.rollout_batch_size, "micro_rollout_batch_size": 1,
        "max_epochs": config.epochs, "num_episodes": 1,
        "prompt_max_len": config.data.max_prompt_length, "generate_max_len": config.data.max_response_length,
        "actor_learning_rate": config.learning_rate, "init_kl_coef": config.algorithm.kl_coef,
        "eps_clip": config.algorithm.epsilon, "temperature": config.algorithm.temperature,
        "l2": config.optimizer.weight_decay, "kl_estimator": "k3",
        "save_steps": config.save_interval, "logging_steps": 1, "eval_steps": -1,
        "max_samples": effective["budget"]["prompt_presentations"],
        "save_path": str(run_dir / "export"), "ckpt_path": str(run_dir / "checkpoints"),
        # OpenRLHF only calls wandb.login when WANDB_API_KEY is absent.
        # Require environment auth; never put a credential into argv or configuration.
        "use_wandb": "enabled-via-environment", "wandb_project": config.project_name,
        "wandb_run_name": config.run_name or run_dir.name,
        "vllm_num_engines": config.vllm_num_engines,
        "vllm_tensor_parallel_size": config.tensor_parallel_size,
        "vllm_gpu_memory_utilization": config.gpu_memory_utilization,
        "actor_num_gpus_per_node": config.num_gpus, "ref_num_gpus_per_node": config.num_gpus,
        "remote_rm_url": str(Path(__file__).with_name("reward.py").resolve()), "seed": config.seed,
    }
    cmd = [sys.executable, "-m", "openrlhf.cli.train_ppo_ray"]
    for name, value in values.items():
        cmd += ["--" + name, str(value)]
    cmd += ["--adam_betas", str(config.optimizer.beta1), str(config.optimizer.beta2)]
    cmd += ["--use_kl_loss", "--colocate_all_models", "--bf16"]
    if config.model.use_flash_attention:
        cmd += ["--flash_attn"]
    return cmd


def cli_preflight(cmd: list[str], env: dict, cwd: Path) -> None:
    """CPU-only CLI availability check before the training subprocess is started."""
    probe = subprocess.run(cmd[:3] + ["--help"], cwd=cwd, env=env, check=True,
                           capture_output=True, text=True, timeout=120)
    accepted = set(re.findall(r"--[a-zA-Z0-9_]+", probe.stdout))
    missing = {arg for arg in cmd[3:] if arg.startswith("--")} - accepted
    if missing:
        raise RuntimeError(f"Pinned backend CLI rejects flags: {sorted(missing)}")


def _prepare_gsm8k_500_jsonl(out_path: str, *, tokenizer, revision: str | None, rows=None) -> str:
    """Native tokenizer formatting with exact identities; never substitute empty labels."""
    if not revision or revision in {"main", "master"}:
        raise ValueError("A pinned GSM8K dataset revision is required")
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset("openai/gsm8k", "main", split="train[:500]", revision=revision)
    out = Path(out_path)
    records = []
    for index, row in enumerate(rows):
        matches = re.findall(r"####\s*([-\d,.]+)", row["answer"])
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous GSM8K ground truth at example {index}")
        label = matches[0].replace(",", "").strip()
        from .reward import parse_number
        if parse_number(label) is None:
            raise ValueError(f"Invalid GSM8K ground truth at example {index}")
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": GSM8K_SYSTEM_PROMPT},
             {"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True)
        records.append({"prompt": prompt, "label": label,
                        "example_id": f"openai/gsm8k@{revision}:train:{index}"})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in records))
    return str(out)


def prepare_inputs(config: OpenRLHFConfig, run_dir: Path, budget: dict) -> tuple[Path, str, dict]:
    from transformers import AutoTokenizer
    if config.model.model_path:
        model_path = str(Path(config.model.model_path).resolve())
        if not Path(model_path).is_dir():
            raise FileNotFoundError(model_path)
    else:
        if not config.model.revision or config.model.revision in {"main", "master"}:
            raise ValueError("Pin model.revision or supply an immutable local model_path")
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(config.model_name, revision=config.model.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_path = run_dir / "prompts.jsonl"
    if config.data.train_data:
        records = []
        for source in config.data.train_data:
            records.extend(json.loads(line) for line in Path(source).read_text().splitlines() if line.strip())
        data_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    else:
        _prepare_gsm8k_500_jsonl(str(data_path), tokenizer=tokenizer,
                               revision=config.data.dataset_revision)
        records = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(records) < budget["prompt_presentations"]:
        raise ValueError("Dataset is smaller than the requested prompt budget")
    from .reward import parse_number
    ids = []
    for row in records[:budget["prompt_presentations"]]:
        if not isinstance(row.get("prompt"), str) or parse_number(str(row.get("label", ""))) is None:
            raise ValueError("Each prompt must have a valid numeric ground-truth label")
        if len(tokenizer.encode(row["prompt"])) > config.data.max_prompt_length:
            raise ValueError("Prompt exceeds max_prompt_length; filtering would change the update budget")
        ids.append(row.get("example_id") or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    return data_path, model_path, {"dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "example_ids": ids, "model_revision": config.model.revision,
            "model_snapshot": model_path, "tokenizer_revision": config.model.revision,
            "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
            "prompt_format": "provided_jsonl" if config.data.train_data else "native_chat_template",
            "prompt_truncation_count": 0, "thinking_mode": "tokenizer_default"}


def validate_history(rows: list[dict], expected_rounds: int) -> list[dict]:
    observations = []
    seen = set()
    for row in rows:
        if "train/reward" not in row:
            continue
        step, reward = row.get("_step"), row["train/reward"]
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or step in seen:
            raise RuntimeError("Invalid or duplicate reward observation step")
        if reward is None or not math.isfinite(float(reward)):
            raise RuntimeError("Training completed without valid reward observations")
        seen.add(step)
        observations.append({"step": step, "reward": float(reward)})
    observations.sort(key=lambda r: r["step"])
    if len(observations) != expected_rounds:
        raise RuntimeError("Incomplete reward history: observation count differs from rollout budget")
    steps = [r["step"] for r in observations]
    if steps not in [list(range(expected_rounds)), list(range(1, expected_rounds + 1))]:
        raise RuntimeError("Incomplete reward history: noncontiguous rollout steps")
    return observations


def read_rewards(config: OpenRLHFConfig, run_id: str, expected_rounds: int) -> list[dict]:
    import wandb
    api = wandb.Api()
    entity = config.wandb_entity or api.default_entity
    child = api.run(f"{entity}/{config.project_name}/{run_id}")
    if child.id != run_id or child.state != "finished":
        raise RuntimeError("Child W&B identity or completion state is unverified")
    return validate_history(list(child.scan_history(keys=["_step", "train/reward"])), expected_rounds)


def runtime_metadata() -> dict:
    packages = {}
    for name in ("openrlhf", "torch", "transformers", "vllm", "ray", "wandb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Do not report parent CUDA allocator measurements as worker measurements.
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        gpu_model = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10, stderr=subprocess.DEVNULL).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpu_model = None
    return {"repository_commit": commit, "backend_distribution": "openrlhf==" + BACKEND_VERSION,
            "backend_source_commit": None, "host": platform.node(), "platform": platform.platform(), "packages": packages,
            "image_digest": os.environ.get("CONTAINER_IMAGE_DIGEST"),
            "gpu_model": gpu_model, "worker_memory_telemetry": "unavailable",
            "precision": "bf16", "worker_seed_derivation": "backend_defined_unverified"}


def run_openrlhf_training(config: OpenRLHFConfig, output_dir: str | None = None) -> dict:
    effective = effective_config(config)
    validate_backend()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY environment authentication is required for exact-run metrics")
    run_id = uuid.uuid4().hex
    run_dir = Path(output_dir or tempfile.mkdtemp(prefix="openrlhf-")).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {"protocol": PROTOCOL, "run_directory": str(run_dir), "attempt_id": run_id, "wandb_run_id": run_id,
                "continuation_of": None, "resume_policy": "new_attempt_only",
                "data_kind": "empirical", "state": "running", "effective_config": effective,
                "runtime": runtime_metadata(), "reward_trace": None,
                "last10_avg": None, "peak_reward": None, "first5_avg": None}
    atomic_json(run_dir / "attempt.json", manifest)
    start = time.monotonic()
    try:
        data_path, model_path, provenance = prepare_inputs(config, run_dir, effective["budget"])
        cmd = build_command(config, run_dir, data_path, model_path)
        env = os.environ.copy()
        env.update(PYTHONPATH="", WANDB_RUN_ID=run_id, WANDB_RESUME="never",
                   WANDB_PROJECT=config.project_name, WANDB_DIR=str(run_dir))
        cli_preflight(cmd, env, run_dir)
        from .reward import smoke_test
        smoke_test()
        manifest.update(input_provenance=provenance, command=cmd,
                        reward_parser_sha256=hashlib.sha256(Path(__file__).with_name("reward.py").read_bytes()).hexdigest())
        atomic_json(run_dir / "attempt.json", manifest)
        print("[openrlhf] cmd:", shlex.join(cmd))
        proc = subprocess.run(cmd, cwd=run_dir, env=env, check=True, timeout=config.wall_timeout_s)
        # Also enforce the exit code for adapters/test doubles that ignore check=True.
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        manifest["state"] = "incomplete"
        observations = read_rewards(config, run_id, effective["budget"]["rollout_rounds"])
        atomic_json(run_dir / "metrics.json", {"attempt_id": run_id, "data_kind": "empirical",
                                              "source": "wandb_exact_id_train/reward", "observations": observations})
        rewards = [row["reward"] for row in observations]
        manifest.update(state="completed", mode="real", framework="openrlhf", algorithm="grpo",
                        seed=config.seed, group_size=config.algorithm.sample_num, model=config.model_name,
                        observed_counters={"rollout_rounds": len(rewards), "optimizer_updates": None,
                                           "generated_tokens": None},
                        reward_trace=rewards, last10_avg=float(np.mean(rewards[-10:])),
                        first5_avg=float(np.mean(rewards[:5])), peak_reward=max(rewards),
                        duration_s=time.monotonic()-start, subprocess_exit=0)
        atomic_json(run_dir / "result.json", manifest)
        return manifest
    except Exception as exc:
        if manifest["state"] != "incomplete":
            manifest["state"] = "failed"
        # Exception strings may contain SDK authentication details: record type only.
        manifest.update(error_type=type(exc).__name__, duration_s=time.monotonic()-start)
        raise
    finally:
        atomic_json(run_dir / "attempt.json", manifest)


def run(config: OpenRLHFConfig, *, dry_run: bool = False, output_dir: str | None = None) -> dict:
    return run_simulation(config) if dry_run else run_openrlhf_training(config, output_dir)
