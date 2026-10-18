"""Generate a bounded, explicitly routed pilot; never launch training on generation."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import sys
from itertools import product
from pathlib import Path
import yaml


def shell_command(arguments):
    return shlex.join([str(arg) for arg in arguments])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", default="sweep_configs")
    parser.add_argument("--run-script-name", default="run_sweep.sh")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["tinker-atropos"],
        help="Only the implemented Tinker/Atropos route is supported",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=3,
        help="Bounded pilot size; larger campaigns require an explicit limit",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args(argv)
    if args.max_configs <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("max-configs must be positive and seeds unique")
    base = Path(args.base_config).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    cfg = yaml.safe_load(base.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Base config must be a mapping")
    root = Path(__file__).resolve().parents[2]
    training_script = root / "platform_tinker/atropos/train_grpo_unsloth.py"
    if not training_script.is_file():
        raise FileNotFoundError(training_script)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands, entries = [], []
    runner = Path(__file__).with_name("_sweep_attempt.py").resolve()
    for lr, rank, batch, group in list(
        product([1e-5, 3e-5, 1e-4], [8, 16, 32, 64], [64, 128, 256], [8, 16, 32])
    )[: args.max_configs]:
        actual = copy.deepcopy(cfg)
        actual.setdefault("env", {}).update(group_size=group, batch_size=batch)
        actual.setdefault("tinker", {}).update(lora_rank=rank, learning_rate=lr)
        digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        for seed in args.seeds:
            seeded = copy.deepcopy(actual)
            run_name = f"grpo-pilot-{digest[:16]}-s{seed}"
            checkpoint_dir = output / "artifacts" / digest / f"seed-{seed}"
            seeded["tinker"].update(checkpoint_dir=str(checkpoint_dir), wandb_run_name=run_name)
            config_path = output / f"config_{digest[:16]}_s{seed}.yaml"
            config_path.write_text(yaml.safe_dump(seeded, sort_keys=False))
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            manifest = output / "attempts" / f"{config_hash}_{seed}.json"
            command = [
                sys.executable,
                str(runner),
                "--manifest",
                str(manifest),
                "--config-hash",
                config_hash,
                "--seed",
                str(seed),
                "--",
                sys.executable,
                str(training_script),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            ]
            commands.append(shell_command(command))
            entries.append(
                {
                    "config_hash": config_hash,
                    "treatment_hash": digest,
                    "seed": seed,
                    "backend": args.backend,
                    "command": command,
                    "attempt_manifest": str(manifest),
                    "checkpoint_dir": str(checkpoint_dir),
                    "wandb_run_name": run_name,
                }
            )
    plan = {
        "protocol": "grpo-sweep-pilot-v3",
        "backend": args.backend,
        "stage": "pilot",
        "screening": "requires held-out validation before expansion",
        "confirmation_seeds": "must be selected independently after screening",
        "training_started": False,
        "planned_attempts": entries,
    }
    (output / "sweep_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    script = output / args.run_script_name
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + "\n".join(
            "printf '%s\\n' " + shlex.quote("Running: " + command) + "\n" + command
            for command in commands
        )
        + "\n"
    )
    os.chmod(script, 0o700)
    print(
        f"Generated {len(entries)} planned attempts for {args.backend}; no training launched. Script: {script}"
    )
    return plan


if __name__ == "__main__":
    main()
