#!/usr/bin/env python3
"""
platform_hybrid/experiments/modal_batch_runner.py

Modal GPU runner for real multi-seed TRL experiments.
Replaces synthetic data with actual training runs on A10G GPUs.

Usage:
    modal run platform_hybrid/experiments/modal_batch_runner.py --seeds 3 --steps 20
"""

import modal
import os
import json
from pathlib import Path

app = modal.App("tinker-rl-lab-real")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        "numpy>=1.26.0",
        "scipy>=1.14.0",
    )
)

# Mount repo
repo_mount = modal.Mount.from_local_dir(
    "/Users/arvind/platform_hybrid/paper/tinker-rl-lab",
    remote_path="/root/tinker-rl-lab",
    condition=lambda path: not any(
        x in path for x in [".git", "__pycache__", "wandb", ".pyc", "platform_hybrid/paper/"]
    ),
)

results_vol = modal.Volume.from_name("tinker-rl-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    mounts=[repo_mount],
    volumes={"/results": results_vol},
    timeout=1800,
    retries=1,
)
def run_trl_grpo(seed: int, num_steps: int = 20) -> dict:
    """Run TRL GRPO math experiment with a given seed."""
    import subprocess
    import time
    import sys

    exp_path = "/root/tinker-rl-lab/platform_hybrid/experiments/implementations/trl_grpo_math.py"
    result_dir = f"/results/trl_grpo/seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)

    # Patch the script to limit steps and set seed
    patched = f'''
import sys, os
sys.path.insert(0, "/root/tinker-rl-lab")

# Override config before importing main
os.environ["SEED"] = "{seed}"

# Read original and patch
with open("{exp_path}") as f:
    src = f.read()

# Replace num_train_epochs with max_steps for quick runs
src = src.replace(
        num_train_epochs=1,", "max_steps={num_steps},")

# Reduce dataset size for speed
src = src.replace("num_problems=1000", "num_problems=200")

# Change output dir
src = src.replace('output_dir="./grpo_math_output"', 'output_dir="{result_dir}/grpo_output"')
src = src.replace('output_dir = f"./grpo_math_final_seed{{seed}}"', 'output_dir = "{result_dir}/final_model"')

exec(src)
'''
    script_path = f"/tmp/run_seed_{seed}.py"
    with open(script_path, "w") as f:
        f.write(patched)

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=1200,
            cwd="/root/tinker-rl-lab",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = time.time() - start

        # Parse training log for reward trace
        reward_trace = []
        for line in proc.stdout.split("\n"):
            if "reward" in line.lower() and any(c.isdigit() for c in line):
                # Best-effort parse
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "reward" in p.lower() and i + 1 < len(parts):
                            val = float(parts[i + 1].replace(",", "").replace("=", ""))
                            reward_trace.append(val)
                            break
                except:
                    pass

        result = {
            "experiment": "trl_grpo_math",
            "seed": seed,
            "success": proc.returncode == 0,
            "elapsed_seconds": round(elapsed, 1),
            "returncode": proc.returncode,
            "reward_trace": reward_trace[-20:] if len(reward_trace) > 20 else reward_trace,
            "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
        }

        with open(f"{result_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)

        results_vol.commit()
        return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "experiment": "trl_grpo_math",
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": "timeout",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "experiment": "trl_grpo_math",
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
        }


@app.local_entrypoint()
def main(seeds: int = 3, steps: int = 20):
    """Launch multi-seed TRL experiments on Modal GPUs."""
    seed_list = [42, 123, 456][:seeds]
    print(f"Launching {len(seed_list)} TRL GRPO jobs (steps={steps})")

    jobs = [run_trl_grpo.spawn(s, steps) for s in seed_list]

    results = []
    for job in jobs:
        r = job.get()
        status = "OK" if r.get("success") else "FAIL"
        print(f"  [{status}] seed={r['seed']} ({r['elapsed_seconds']}s)")
        results.append(r)

    succeeded = sum(1 for r in results if r.get("success"))
    print(f"\n{succeeded}/{len(seed_list)} succeeded")
    print("Results saved to Modal volume: tinker-rl-results")
