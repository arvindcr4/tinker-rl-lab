"""CLI entrypoint for the consolidated GRPO experiments.

Usage::

    python -m tinkerrl.grpo_cli --preset tooluse_synth --steps 200
    python -m tinkerrl.grpo_cli --preset gsm8k --model Qwen/Qwen3-8B --seed 137

Presets encode the experiment configurations that used to live in separate
``grpo_*.py`` files.  The short ``--preset`` parameter selects one; CLI flags
override individual fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .grpo import (
    GRPOConfig,
    ExactMathReward,
    MathReward,
    ToolCallReward,
    make_synthetic_math_dataset,
    make_synthetic_tool_use_dataset,
    make_gsm8k_dataset,
    make_xlam_dataset,
    run_grpo,
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
    "tooluse_heldout": {
        "name": "grpo_tooluse_qwen3_8b",
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
        "evaluate_heldout": True,
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
        "max_prompt_tokens": 1536,
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
}

DATASET_FACTORIES = {
    "tooluse_synth": make_synthetic_tool_use_dataset,
    "tooluse_baseline": make_synthetic_tool_use_dataset,
    "tooluse_heldout": make_synthetic_tool_use_dataset,
    "tooluse_xlam": make_xlam_dataset,
    "gsm8k": make_gsm8k_dataset,
    "math100": make_synthetic_math_dataset,
}

REWARD_MAP = {
    "tooluse_synth": ToolCallReward,
    "tooluse_baseline": ToolCallReward,
    "tooluse_heldout": ToolCallReward,
    "tooluse_xlam": ToolCallReward,
    "gsm8k": ExactMathReward,
    "math100": MathReward,
}


def _build_dataset(args: argparse.Namespace, config: GRPOConfig) -> Any:
    dataset_name = args.dataset or args.preset
    factory = DATASET_FACTORIES.get(dataset_name)
    if factory is None:
        raise SystemExit(f"Unknown dataset preset: {dataset_name}")
    if dataset_name in {"tooluse_xlam", "gsm8k"}:
        return factory(seed=config.seed)
    return factory()


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override preset fields with any CLI values the user actually passed."""
    mapping = {
        "model": "model",
        "lora_rank": "lora_rank",
        "steps": "steps",
        "group_size": "group_size",
        "batch_size": "batch_size",
        "lr": "lr",
        "temperature": "temperature",
        "top_p": "top_p",
        "max_prompt_tokens": "max_prompt_tokens",
        "max_response_tokens": "max_response_tokens",
        "save_every": "save_every",
        "seed": "seed",
        "num_seeds": "num_seeds",
        "name": "name",
        "checkpoint_dir": "checkpoint_dir",
    }
    for cfg_key, attr in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg[cfg_key] = val
    if args.evaluate_heldout:
        cfg["evaluate_heldout"] = True
    if args.no_resume:
        cfg["resume"] = False
    if args.no_wandb:
        cfg["wandb_project"] = None
    return cfg


def build_config(args: argparse.Namespace) -> GRPOConfig:
    if args.json_config:
        data = json.loads(args.json_config.read_text())
        return GRPOConfig(**data)

    if args.preset not in PRESETS:
        raise SystemExit(f"Unknown preset: {args.preset!r}.  Choose from {list(PRESETS)}")
    cfg_dict = dict(PRESETS[args.preset])
    cfg_dict = _apply_overrides(cfg_dict, args)
    return GRPOConfig(**cfg_dict)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tinkerrl.grpo_cli",
        description="Consolidated GRPO experiment runner.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tooluse_synth",
        help="Experiment preset (default: tooluse_synth).",
    )
    parser.add_argument("--json-config", type=Path, help="Path to a JSON config file.")
    parser.add_argument("--dataset", help="Dataset preset (overrides --preset default).")
    parser.add_argument("--reward", help="Reward adapter (overrides --preset default).")
    # overrides
    parser.add_argument("--name", help="Experiment name.")
    parser.add_argument("--model")
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
    parser.add_argument("--num-seeds", dest="num_seeds", type=int)
    parser.add_argument("--evaluate-heldout", dest="evaluate_heldout", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--help-overrides", action="store_true", help="Show all override flags.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.help_overrides:
        # argparse --help covers this; kept for discoverability.
        args = _parse_args(["--help"])
        return 0

    if "TINKER_API_KEY" not in os.environ:
        print("ERROR: Set TINKER_API_KEY in the environment.", file=sys.stderr)
        return 1

    config = build_config(args)
    dataset = _build_dataset(args, config)
    reward_name = args.reward or args.preset
    reward_cls = REWARD_MAP.get(reward_name, ToolCallReward)
    reward = reward_cls()

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
    forwarded = [translations.get(arg, arg) for arg in (argv or [])]
    return main(["--preset", preset, *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
