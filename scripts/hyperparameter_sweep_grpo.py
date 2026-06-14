import os
import argparse
import yaml
from itertools import product

def main():
    parser = argparse.ArgumentParser(description="Generate and run an exhaustive GRPO hyperparameter sweep.")
    parser.add_argument("--base-config", required=True, help="Path to base YAML config (e.g., atropos/configs/gsm8k_qwen_8b.yaml)")
    parser.add_argument("--output-dir", default="sweep_configs", help="Directory to save generated configs and run script")
    parser.add_argument("--run-script-name", default="run_sweep.sh", help="Name of the generated bash script")
    
    args = parser.parse_args()

    # Define the sweep grid
    learning_rates = [1e-5, 3e-5, 1e-4]
    lora_ranks = [8, 16, 32, 64]
    batch_sizes = [64, 128, 256]
    group_sizes = [8, 16, 32]

    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    run_commands = []
    
    # Keep track of generated config count
    count = 0

    print(f"Generating exhaustive sweep configurations based on {args.base_config}")

    for lr, rank, bs, gs in product(learning_rates, lora_ranks, batch_sizes, group_sizes):
        # Constraints: batch_size must be a multiple of group_size
        if bs % gs != 0:
            continue
            
        # Copy base config
        cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
        
        # Ensure env and tinker sections exist
        if "env" not in cfg:
            cfg["env"] = {}
        if "tinker" not in cfg:
            cfg["tinker"] = {}

        cfg["env"]["group_size"] = gs
        cfg["env"]["batch_size"] = bs
        cfg["tinker"]["lora_rank"] = rank
        cfg["tinker"]["learning_rate"] = lr
        
        # Give it a unique run name for wandb
        base_name = cfg["tinker"].get("wandb_run_name", "grpo-sweep")
        run_name = f"{base_name}-lr{lr}-lora{rank}-bs{bs}-gs{gs}"
        cfg["tinker"]["wandb_run_name"] = run_name
        cfg["tinker"]["wandb_group"] = "grpo-exhaustive-sweep"

        config_filename = f"config_lr{lr}_lora{rank}_bs{bs}_gs{gs}.yaml"
        config_filepath = os.path.join(args.output_dir, config_filename)
        
        with open(config_filepath, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        count += 1
        # Add to run script
        # Assuming the generated bash script will be run from the repo root or atropos dir,
        # we provide the relative path to the config file.
        # But to be safe, we can use absolute paths or relative to execution dir.
        cmd = f"python3 atropos/train_grpo_unsloth.py --config {config_filepath} --seed 42"
        run_commands.append(cmd)

    print(f"Generated {count} valid configurations.")

    run_script_path = os.path.join(args.output_dir, args.run_script_name)
    with open(run_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated exhaustive hyperparameter sweep script\n")
        f.write("set -euo pipefail\n\n")
        f.write("echo 'Starting exhaustive hyperparameter sweep...'\n\n")
        for cmd in run_commands:
            f.write(f"echo 'Running: {cmd}'\n")
            f.write(f"{cmd}\n")
            f.write("echo '---------------------------------------------------'\n\n")

    os.chmod(run_script_path, 0o755)
    print(f"Run script written to {run_script_path}")
    print(f"To execute the sweep, run: bash {run_script_path}")

if __name__ == '__main__':
    main()
