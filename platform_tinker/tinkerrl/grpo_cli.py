"""CLI entrypoint for the consolidated GRPO experiments.

Usage::

    python -m platform_tinker.tinkerrl.grpo_cli --preset tooluse_synth --steps 200
    python -m platform_tinker.tinkerrl.grpo_cli --preset gsm8k --model Qwen/Qwen3-8B --seed 137

Presets encode the experiment configurations that used to live in separate
``grpo_*.py`` files.  The short ``--preset`` parameter selects one; CLI flags
override individual fields.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .grpo import (
    PAVLOV_DECLARED_DOMAINS,
    PAVLOV_DOMAIN_TAGS,
    PAVLOV_HELDOUT_SUITE_IDS,
    PAVLOV_NON_XLAM_DATASET_REVISION,
    PAVLOV_PRIMARY_EVALUATION_SUITE_IDS,
    PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION,
    PAVLOV_TRAINING_DOMAIN_UNION,
    PAVLOV_TRAINING_SUITE_IDS,
    GRPOConfig,
    ExactMathReward,
    MathReward,
    PavlovNonXLAMReward,
    StrictToolCallReward,
    ToolCallReward,
    make_synthetic_math_dataset,
    make_synthetic_tool_use_dataset,
    make_gsm8k_dataset,
    make_pavlov_non_xlam_dataset,
    make_xlam_dataset,
    run_grpo,
    _strict_json_object,
    _redact_error,
    _CheckpointLockError,
)


# ---------------------------------------------------------------------------
# Presets — config rows replace the old grpo_*.py files
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "tooluse_synth": {
        "name": "tooluse_synth",
        "model": "Qwen/Qwen3-8B",
        "lora_rank": 32,
        "steps": 100,
        "group_size": 4,
        "batch_size": 2,
        "lr": 3e-5,
        "temperature": 0.8,
        "max_prompt_tokens": 1024,
        "max_response_tokens": 128,
        "save_every": 25,
        "num_seeds": 5,
        "seed": 0,
    },
    "tooluse_baseline": {
        "name": "A_baseline",
        "model": "Qwen/Qwen3-8B",
        "lora_rank": 32,
        "steps": 200,
        "group_size": 8,
        "batch_size": 4,
        "lr": 3e-5,
        "temperature": 0.8,
        "max_response_tokens": 192,
        "save_every": 10,
        "num_seeds": 5,
        "seed": 0,
    },
    "tooluse_xlam": {
        "name": "tooluse_xlam",
        "model": "Qwen/Qwen3-8B",
        "lora_rank": 32,
        "steps": 100,
        "group_size": 4,
        "batch_size": 2,
        "lr": 3e-5,
        "temperature": 0.8,
        "max_prompt_tokens": 2048,
        "max_response_tokens": 128,
        "save_every": 25,
        "num_seeds": 1,
        "seed": 42,
        "evaluate_heldout": True,
    },
    "gsm8k": {
        "name": "gsm8k",
        "model": "Qwen/Qwen3-8B",
        "lora_rank": 32,
        "steps": 200,
        "group_size": 4,
        "batch_size": 2,
        "lr": 3e-5,
        "temperature": 0.8,
        "max_prompt_tokens": 1024,
        "max_response_tokens": 512,
        "save_every": None,
        "num_seeds": 1,
        "evaluate_heldout": False,
        "seed": 42,
    },
    "math100": {
        "name": "math100",
        "model": "Qwen/Qwen3-8B",
        "lora_rank": 32,
        "steps": 100,
        "group_size": 4,
        "batch_size": 2,
        "lr": 5e-6,
        "temperature": 0.9,
        "max_prompt_tokens": 1024,
        "max_response_tokens": 256,
        "save_every": 25,
        "num_seeds": 1,
        "seed": 42,
        "evaluate_heldout": True,
    },
    "pavlov_xlam": {
        "name": "pavlov_xlam_qwen36",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "lora_rank": 32,
        "steps": 200,
        "group_size": 4,
        "batch_size": 2,
        "lr": 2e-5,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_prompt_tokens": 1200,
        "max_response_tokens": 128,
        "save_every": 25,
        "num_seeds": 1,
        "seed": 809,
        "evaluate_heldout": True,
        "dataset_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
        "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "wandb_project": "tinker-rl-lab-pavlov",
        "wandb_entity": "arvindcr4-pes-university",
        "wandb_group": "pavlov-tinker-18usd-20260809",
        "hf_public": True,
        "hf_repo_prefix": "pavlov-xlam-qwen36",
        "checkpoint_name_prefix": "pavlov-xlam-qwen36",
        "campaign_status": "authorized",
        "budget_status": "AUTHORIZED_TINKER_ONLY",
        "paid_jobs_may_launch": True,
        "authorized_budget_usd": 18.0,
        "maximum_usd": 18.0,
        "training_suite_ids": PAVLOV_TRAINING_SUITE_IDS,
        "heldout_suite_ids": PAVLOV_HELDOUT_SUITE_IDS,
        "primary_evaluation_suite_ids": PAVLOV_PRIMARY_EVALUATION_SUITE_IDS,
        "domain_tags": PAVLOV_DOMAIN_TAGS,
        "declared_domains": PAVLOV_DECLARED_DOMAINS,
        "training_domain_union": PAVLOV_TRAINING_DOMAIN_UNION,
        "primary_evaluation_domain_union": PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION,
    },
    "pavlov_portfolio": {
        "name": "pavlov_portfolio_api_swegym_qwen36_20260809",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "lora_rank": 32,
        "steps": 40,
        "group_size": 4,
        "batch_size": 2,
        "lr": 1e-5,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_prompt_tokens": 1536,
        "max_response_tokens": 384,
        "save_every": 20,
        "num_seeds": 1,
        "seed": 809,
        "evaluate_heldout": False,
        "dataset_revision": PAVLOV_NON_XLAM_DATASET_REVISION,
        "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "wandb_project": "tinker-rl-lab-pavlov",
        "wandb_entity": "arvindcr4-pes-university",
        "wandb_group": "pavlov-portfolio-20260809",
        "wandb_tags": ("grpo", "tinker", "pavlov", "api-bank", "swe-gym"),
        "hf_public": True,
        "hf_repo_prefix": "pavlov-portfolio-qwen36",
        "checkpoint_name_prefix": "pavlov-portfolio-qwen36",
        "campaign_status": "authorized",
        "budget_status": "AUTHORIZED_TINKER_ONLY",
        "paid_jobs_may_launch": True,
        "authorized_budget_usd": 16.5,
        "maximum_usd": 18.0,
        "training_suite_ids": ("api_bank_rlvr_train", "swe_gym_train"),
        "heldout_suite_ids": (),
        "primary_evaluation_suite_ids": PAVLOV_PRIMARY_EVALUATION_SUITE_IDS,
        "domain_tags": ("code", "finance", "enterprise", "tools", "long_horizon"),
        "declared_domains": ("code", "finance", "enterprise", "tool_use", "long_horizon"),
        "training_domain_union": (
            "code",
            "finance",
            "enterprise",
            "tool_use",
            "long_horizon",
        ),
        "primary_evaluation_domain_union": PAVLOV_PRIMARY_EVALUATION_DOMAIN_UNION,
    },
}

# Heldout reruns the baseline grid and only flips evaluation on.
PRESETS["tooluse_heldout"] = {
    **PRESETS["tooluse_baseline"],
    "name": "grpo_tooluse_qwen3_8b",
    "evaluate_heldout": True,
}

DATASET_FACTORIES = {
    "tooluse_synth": make_synthetic_tool_use_dataset,
    "tooluse_baseline": make_synthetic_tool_use_dataset,
    "tooluse_heldout": make_synthetic_tool_use_dataset,
    "tooluse_xlam": make_xlam_dataset,
    "gsm8k": make_gsm8k_dataset,
    "math100": make_synthetic_math_dataset,
    "pavlov_xlam": make_xlam_dataset,
    "pavlov_portfolio": make_pavlov_non_xlam_dataset,
}

REWARD_MAP = {
    "tooluse_synth": ToolCallReward,
    "tooluse_baseline": ToolCallReward,
    "tooluse_heldout": ToolCallReward,
    "tooluse_xlam": ToolCallReward,
    "gsm8k": ExactMathReward,
    "math100": MathReward,
    "pavlov_xlam": StrictToolCallReward,
    "pavlov_portfolio": PavlovNonXLAMReward,
}


def _dataset_request(args: argparse.Namespace, config: GRPOConfig) -> tuple[Any, Dict[str, Any]]:
    """Validate adapter selection without loading data or contacting a service."""
    dataset_name = args.dataset or args.preset
    factory = DATASET_FACTORIES.get(dataset_name)
    if factory is None:
        raise SystemExit(f"Unknown dataset preset: {dataset_name}")
    inherited_revision = PRESETS.get(args.preset, {}).get("dataset_revision")
    if (
        not getattr(args, "json_config", None)
        and args.dataset
        and args.dataset != args.preset
        and getattr(args, "dataset_revision", None) is None
        and inherited_revision is not None
        and config.dataset_revision == inherited_revision
        and DATASET_FACTORIES.get(args.preset) is not factory
    ):
        raise ValueError(
            "dataset override would reuse another adapter's inherited revision; "
            "pass --dataset-revision explicitly or supply a matching JSON config"
        )
    if dataset_name in {"tooluse_xlam", "pavlov_xlam", "gsm8k"}:
        if config.dataset_revision is not None:
            return factory, {"seed": config.seed, "revision": config.dataset_revision}
        return factory, {"seed": config.seed}
    if dataset_name == "pavlov_portfolio":
        if config.dataset_revision not in (None, PAVLOV_NON_XLAM_DATASET_REVISION):
            raise ValueError("dataset revision must match the pinned portfolio composite")
        return factory, {"seed": config.seed}
    if config.dataset_revision is not None:
        raise ValueError("dataset revision is unsupported for this dataset adapter")
    return factory, {}


def _build_dataset(args: argparse.Namespace, config: GRPOConfig) -> Any:
    factory, kwargs = _dataset_request(args, config)
    return factory(**kwargs)


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override preset fields with any CLI values the user actually passed."""
    for key in (
        "model",
        "model_revision",
        "lora_rank",
        "steps",
        "group_size",
        "batch_size",
        "lr",
        "temperature",
        "top_p",
        "max_prompt_tokens",
        "max_response_tokens",
        "save_every",
        "seed",
        "dataset_revision",
        "num_seeds",
        "name",
        "checkpoint_dir",
        "wandb_project",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
        "hf_owner",
        "hf_repo_prefix",
        "checkpoint_name_prefix",
        "dynamic_sampling_max_resamples",
        "nll_aux_coef",
        "nll_aux_min_reward",
        "gspo_update_epochs",
        "gspo_epsilon_low",
        "gspo_epsilon_high",
        "critic_lr",
        "critic_pretrain_batches",
        "critic_hidden_dim",
        "critic_updates_per_step",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    for attr, key, value in (
        ("evaluate_heldout", "evaluate_heldout", True),
        ("debias_advantages", "debias_advantages", True),
        ("dynamic_sampling", "dynamic_sampling", True),
        ("mask_truncated_responses", "mask_truncated_responses", True),
        ("seed_sampling", "seed_sampling", True),
        ("global_advantage_normalization", "global_advantage_normalization", True),
        ("no_global_norm_exclude_truncated", "global_norm_exclude_truncated", False),
        ("gspo", "gspo_enabled", True),
        ("critic", "critic_enabled", True),
        ("no_resume", "resume", False),
        ("hf_public", "hf_public", True),
    ):
        if getattr(args, attr):
            cfg[key] = value
    if args.nll_aux or args.nll_aux_coef is not None:
        # A coef without the flag still opts in: silently ignoring an
        # explicitly passed coefficient would be a worse surprise.
        cfg["nll_aux_enabled"] = True
    return cfg


def build_config(args: argparse.Namespace) -> GRPOConfig:
    if args.json_config:
        data = _strict_json_object(args.json_config.read_text(encoding="utf-8"))
        if data is None:
            raise ValueError("JSON config must be one object with unique keys and finite numbers")
        config = GRPOConfig(**_apply_overrides(data, args))
        config.validate_tracking()
        config.validate_campaign_gate()
        return config

    if args.preset not in PRESETS:
        raise SystemExit(f"Unknown preset: {args.preset!r}.  Choose from {list(PRESETS)}")
    cfg_dict = dict(PRESETS[args.preset])
    cfg_dict = _apply_overrides(cfg_dict, args)
    config = GRPOConfig(**cfg_dict)
    config.validate_tracking()
    config.validate_campaign_gate()
    return config


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m platform_tinker.tinkerrl.grpo_cli",
        description="Consolidated GRPO experiment runner.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tooluse_synth",
        help="Experiment preset (default: tooluse_synth).",
    )
    parser.add_argument(
        "--json-config",
        type=Path,
        help="GRPO hyperparameters in JSON; explicitly supplied CLI hyperparameters override JSON values. Dataset/reward selection follows CLI flags.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check local config and adapter selection only; print JSON without credentials or network.",
    )
    parser.add_argument("--dataset", help="Dataset preset (overrides --preset default).")
    parser.add_argument("--reward", help="Reward adapter (overrides --preset default).")
    # overrides
    parser.add_argument("--name", help="Experiment name.")
    parser.add_argument("--model")
    parser.add_argument(
        "--model-revision",
        dest="model_revision",
        help="Tokenizer revision request and run metadata; does not pin managed provider weights.",
    )
    parser.add_argument("--lora-rank", dest="lora_rank", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--group-size", dest="group_size", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", dest="top_p", type=float)
    parser.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int)
    parser.add_argument("--max-response-tokens", dest="max_response_tokens", type=int)
    parser.add_argument("--save-every", dest="save_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--num-seeds", dest="num_seeds", type=int)
    parser.add_argument("--evaluate-heldout", dest="evaluate_heldout", action="store_true")
    parser.add_argument("--debias-advantages", dest="debias_advantages", action="store_true")
    parser.add_argument("--dynamic-sampling", dest="dynamic_sampling", action="store_true")
    parser.add_argument("--seed-sampling", dest="seed_sampling", action="store_true")
    parser.add_argument(
        "--dynamic-sampling-max-resamples", dest="dynamic_sampling_max_resamples", type=int
    )
    parser.add_argument(
        "--mask-truncated-responses", dest="mask_truncated_responses", action="store_true"
    )
    parser.add_argument(
        "--global-advantage-normalization",
        dest="global_advantage_normalization",
        action="store_true",
    )
    parser.add_argument(
        "--no-global-norm-exclude-truncated",
        dest="no_global_norm_exclude_truncated",
        action="store_true",
    )
    parser.add_argument("--nll-aux", dest="nll_aux", action="store_true")
    parser.add_argument("--nll-aux-coef", dest="nll_aux_coef", type=float)
    parser.add_argument("--nll-aux-min-reward", dest="nll_aux_min_reward", type=float)
    parser.add_argument("--gspo", dest="gspo", action="store_true")
    parser.add_argument("--gspo-update-epochs", dest="gspo_update_epochs", type=int)
    parser.add_argument("--gspo-epsilon-low", dest="gspo_epsilon_low", type=float)
    parser.add_argument("--gspo-epsilon-high", dest="gspo_epsilon_high", type=float)
    parser.add_argument("--critic", dest="critic", action="store_true")
    parser.add_argument("--critic-lr", dest="critic_lr", type=float)
    parser.add_argument("--critic-pretrain-batches", dest="critic_pretrain_batches", type=int)
    parser.add_argument("--critic-hidden-dim", dest="critic_hidden_dim", type=int)
    parser.add_argument("--critic-updates-per-step", dest="critic_updates_per_step", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--hf-owner")
    parser.add_argument("--hf-repo-prefix")
    parser.add_argument("--checkpoint-name-prefix")
    parser.add_argument("--hf-public", action="store_true")
    parser.add_argument("--help-overrides", action="store_true", help="Show all override flags.")
    return parser.parse_args(argv)


def _selected_reward_name(args: argparse.Namespace) -> str:
    reward_name = args.reward or args.dataset or args.preset
    if reward_name not in REWARD_MAP:
        raise SystemExit(f"Unknown reward: {reward_name!r}.  Choose from {sorted(REWARD_MAP)}")
    return str(reward_name)


def _configuration_check(args: argparse.Namespace) -> Dict[str, Any]:
    reward_name = _selected_reward_name(args)
    config = build_config(args)
    _dataset_request(args, config)
    encoded = json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()
    # Whitelist numeric/boolean knobs: never echo full config, endpoint, owner,
    # private model identifiers, experiment names or filesystem paths.
    effective = {
        key: getattr(config, key)
        for key in (
            "num_seeds",
            "seed",
            "batch_size",
            "group_size",
            "lr",
            "temperature",
            "top_p",
            "max_prompt_tokens",
            "max_response_tokens",
            "resume",
            "evaluate_heldout",
            "gspo_enabled",
            "gspo_update_epochs",
            "nll_aux_enabled",
            "nll_aux_coef",
            "dynamic_sampling",
            "dynamic_sampling_max_resamples",
            "seed_sampling",
            "critic_enabled",
            "critic_pretrain_batches",
            "critic_updates_per_step",
            "hf_public",
        )
    }
    effective.update(
        steps_per_seed=config.steps,
        total_configured_steps=config.steps * config.num_seeds,
        save_every=config.effective_save_every(),
    )
    return {
        "schema_version": 1,
        "status": "config_checks_passed",
        "scope": "Local config and adapter selection only; no dataset, tokenizer, credentials, network, provider or scientific verification.",
        "not_checked": [
            "dataset_contents",
            "tokenizer",
            "credentials",
            "provider_availability",
            "tracking_export",
            "scientific_validity",
        ],
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "dataset": args.dataset or args.preset,
        "reward": reward_name,
        "effective": effective,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.help_overrides:
        # argparse --help covers this; kept for discoverability.
        args = _parse_args(["--help"])
        return 0

    if args.check_config:
        print(json.dumps(_configuration_check(args), sort_keys=True, allow_nan=False))
        return 0

    if "TINKER_API_KEY" not in os.environ:
        print("ERROR: Set TINKER_API_KEY in the environment.", file=sys.stderr)
        return 1

    # The reward follows the dataset actually trained on; an unknown name
    # fails before any dataset load instead of silently scoring with
    # ToolCallReward (fixed 2026-10-03).
    reward_name = _selected_reward_name(args)
    config = build_config(args)
    dataset = _build_dataset(args, config)
    reward = REWARD_MAP[reward_name]()

    print(f"[grpo_cli] preset={args.preset} seeds={config.num_seeds} steps={config.steps}")
    print(f"[grpo_cli] model={config.model} lr={config.lr} group={config.group_size}")
    print()

    results = run_grpo(config, dataset, reward)

    for r in results:
        print()
        print(f"[grpo_cli] Seed {r.seed} done.")
        print(f"  run_id        : {r.run_id}")
        print(f"  sampler       : {r.sampler_path}")
        print(f"  avg_first5    : {r.avg_first5:.3f}")
        print(f"  avg_last10    : {r.avg_last10:.3f}")
        print(f"  peak_reward   : {r.peak_reward:.3f}")
        print(f"  zero_loss     : {r.zero_loss_steps}/{config.steps}")
        print(f"  zero_reward   : {r.zero_reward_steps}/{config.steps}")
        if r.heldout_reward is not None:
            print(f"  heldout_reward: {r.heldout_reward:.3f}")
        print(f"  reward_trace  : {[round(v, 3) for v in r.reward_trace]}")

    return 0


def legacy_main(preset: str, argv: Optional[List[str]] = None) -> int:
    """Preserve old script flags while routing execution through this module."""
    translations = {
        "--rank": "--lora-rank",
        "--group": "--group-size",
        "--batch": "--batch-size",
        "--tag": "--name",
    }
    forwarded = [translations.get(arg, arg) for arg in (sys.argv[1:] if argv is None else argv)]
    return main(["--preset", preset, *forwarded])


def cli_entrypoint(argv: Optional[List[str]] = None) -> int:
    """Keep common shell input/I/O errors readable; library main still raises."""
    try:
        return main(argv)
    except (OSError, TypeError, ValueError, _CheckpointLockError) as exc:
        print(f"ERROR: {_redact_error(exc)}", file=sys.stderr)
        print(
            "Inspect supported options with python -m platform_tinker.tinkerrl.grpo_cli --help",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
