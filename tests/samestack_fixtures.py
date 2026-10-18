"""Explicitly synthetic run records; never evidence of an actual experiment."""

import copy

from platform_hybrid.experiments.samestack_contract import canonical_hash
from platform_hybrid.experiments.samestack_dependencies import LOCK, lock_manifest
from platform_hybrid.experiments.samestack_protocol import (
    ANALYSIS_PLAN_ID,
    PROTOCOL_HASH,
    PROTOCOL_ID,
    PROTOCOL_SPEC,
    prompt_schedules,
)


def synthetic_run(arm="grpo_g8", seed=42, smoke=True, environment="colab"):
    n, steps = (32, 2) if smoke else (1319, 30)
    group = {"grpo_g8": 8, "grpo_g2": 2, "ppo": 1, "ppo_8x8": 8}[arm]
    warm_count = (1 if smoke else 4) if arm.startswith("ppo") else 0
    critic = (
        {
            "v_lr": 1e-3,
            "v_iters": 50,
            "warm_batches": warm_count,
            "warm_generations": warm_count * 64,
        }
        if arm.startswith("ppo")
        else None
    )
    generation = {
        "train": dict(
            PROTOCOL_SPEC["generation"],
            do_sample=True,
            eos_token_id=[2],
            pad_token_id=2,
            max_new_tokens=512,
        ),
        "eval": dict(
            PROTOCOL_SPEC["generation"],
            do_sample=False,
            eos_token_id=[2],
            pad_token_id=2,
            max_new_tokens=512,
        ),
    }
    config = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "arm": arm,
        "seed": seed,
        "smoke": smoke,
        "n_steps": steps,
        "n_eval": n,
        "n_gen": 64,
        "group": group,
        "prompts_per_step": 64 // group,
        "k_epochs": 2,
        "lr": 1e-5,
        "max_new": 512,
        "critic": critic,
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "dataset_revision": "c" * 40,
        "clip": 0.2,
        "eval_bs": 64,
        "eps": 1e-6,
        "chunk": 4,
        "generation": generation,
    }
    config.update(
        policy_optimizer={
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "max_grad_norm": 1.0,
        },
        lora={
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        loss_normalization="per-response mean over first-EOS-inclusive tokens",
        reference_kl=None,
        policy_dropout_enabled=False,
        attention_backend="sdpa",
        precision={
            "model": "torch.bfloat16",
            "logprobs": "torch.float32",
            "value_head": "torch.float32",
        },
        advantage_normalization="batch sample-std"
        if arm.startswith("ppo")
        else "group population-std",
    )
    ids, train_ids = (
        [f"synthetic:test:{i}" for i in range(n)],
        [f"synthetic:train:{i}" for i in range(100)],
    )
    source = {
        "status": "observed",
        "commit": "d" * 40,
        "dirty_hash": "e" * 64,
        "dirty": True,
        "source_files": {"SYNTHETIC-NOT-REAL.py": "f" * 64},
    }
    hardware = {"gpu_names": ["SYNTHETIC-NOT-A-GPU"], "python": "3.11.0"}
    hardware.update(
        driver_versions=None,
        cuda_runtime="SYNTHETIC",
        cudnn=None,
        attention_backend="sdpa",
        model_dtype="torch.bfloat16",
        logprob_dtype="torch.float32",
        value_head_dtype="torch.float32",
        tf32_matmul=False,
        deterministic_algorithms=False,
    )
    schedules = prompt_schedules(seed, len(train_ids), steps, warm_count, 64 // group)
    schedules["policy_item_ids"] = [[train_ids[i] for i in row] for row in schedules["policy"]]
    schedules["critic_warmup_item_ids"] = [
        [train_ids[i] for i in row] for row in schedules["critic_warmup"]
    ]
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": PROTOCOL_HASH,
        "protocol_spec": copy.deepcopy(PROTOCOL_SPEC),
        "analysis_plan_id": ANALYSIS_PLAN_ID,
        "environment": environment,
        "dependencies": dict(LOCK),
        "dependency_lock": lock_manifest(),
        "hardware": hardware,
        "effective_config": config,
        "config_hash": canonical_hash(config),
        "environment_hash": canonical_hash(
            {"environment": environment, "dependencies": LOCK, "hardware": hardware}
        ),
        "chat_template_hash": canonical_hash("SYNTHETIC chat template"),
        "test_item_ids": ids,
        "train_item_ids": train_ids,
        "ordered_test_item_hash": canonical_hash(ids),
        "source": source,
        "generation": generation,
        "schedules": schedules,
        "revisions": {
            k: {"revision": config[k + "_revision"]} for k in ("model", "tokenizer", "dataset")
        },
        "timestamps": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
        },
    }
    manifest["run_id"] = canonical_hash(
        {
            "protocol_hash": PROTOCOL_HASH,
            "config_hash": manifest["config_hash"],
            "environment": environment,
            "environment_hash": manifest["environment_hash"],
            "source_commit": source["commit"],
            "source_dirty_hash": source["dirty_hash"],
        }
    )
    record = {
        **config,
        "manifest": manifest,
        "heldout_pre_acc": 0.0,
        "heldout_post_acc": 0.0,
        "heldout_pre_trunc": 0.0,
        "heldout_post_trunc": 0.0,
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "last10_avg": 0.0,
        "pre_correct": [0] * n,
        "post_correct": [0] * n,
        "pre_trunc": [0] * n,
        "post_trunc": [0] * n,
        "step_log": [{"step": i, "mean_reward": 0.0} for i in range(steps)],
        "warm_log": [{"warm_step": i} for i in range(warm_count)],
    }
    return record, {
        "pre": ["SYNTHETIC NaN text"] * n,
        "post": ["SYNTHETIC text"] * n,
        "test_item_ids": ids,
    }
