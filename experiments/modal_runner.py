"""
Modal GPU Runner for Tinker RL Lab Multi-Seed Experiments
=========================================================
Runs all experiments across 5 seeds on Modal A10G GPUs.

Usage:
    modal run experiments/modal_runner.py               # Run all experiments
    modal run experiments/modal_runner.py --exp trl_grpo # Run specific experiment
    modal run experiments/modal_runner.py --seeds 3      # Run 3 seeds instead of 5
    modal run experiments/modal_runner.py --dry-run      # List what would run
"""

import modal
import os
import json
from pathlib import Path

# Modal app
app = modal.App("tinker-rl-lab")

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "numpy>=1.26.0",
        "scipy>=1.14.0",
        "rliable>=1.0.8",
        "wandb",
    )
)

# Mount the repo
repo_mount = modal.Mount.from_local_dir(
    os.path.expanduser("~/tinker"),
    remote_path="/root/tinker",
    condition=lambda path: not any(
        x in path for x in [".git", "__pycache__", "wandb", ".pyc", "paper/"]
    ),
)

# Volume for persisting results across runs
results_vol = modal.Volume.from_name("tinker-results", create_if_missing=True)

EXPERIMENTS = {
    "trl_grpo": "trl_grpo_math.py",
    "trl_gsm8k": "trl_gsm8k_math.py",
    "trl_dpo": "trl_dpo_shorter.py",
    "trl_sft": "trl_chat_sft.py",
    "trl_distill": "trl_distillation.py",
    "trl_ppo": "trl_ppo_gsm8k_baseline.py",
    "trl_sft_gsm8k": "trl_sft_gsm8k_baseline.py",
    "sb3_ppo": "sb3_ppo_math.py",
    "cleanrl_ppo": "cleanrl_ppo_math.py",
    "tianshou_ppo": "tianshou_ppo_math.py",
    "pufferlib": "pufferlib_math.py",
    "rl_games": "rl_games_math.py",
    "d3rlpy": "d3rlpy_offline.py",
}

SEEDS = [42, 123, 456, 789, 1024]


@app.function(
    image=image,
    gpu="A10G",
    mounts=[repo_mount],
    volumes={"/results": results_vol},
    timeout=3600,  # 1 hour per experiment
    retries=1,
)
def run_experiment(exp_name: str, exp_file: str, seed: int) -> dict:
    """Run a single experiment with a given seed on GPU."""
    import subprocess
    import time

    exp_path = f"/root/tinker/experiments/implementations/{exp_file}"
    result_dir = f"/results/{exp_name}/seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)

    print(f"[{exp_name}] seed={seed} | GPU: {os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()}")

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", exp_path, "--seed", str(seed)],
            capture_output=True,
            text=True,
            timeout=3000,
            cwd="/root/tinker",
            env={**os.environ, "SEED": str(seed), "RESULTS_DIR": result_dir},
        )
        elapsed = time.time() - start
        success = proc.returncode == 0

        result = {
            "experiment": exp_name,
            "seed": seed,
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }

        # Save result
        with open(f"{result_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)

        # Copy any output files from the experiment
        for pattern in ["*.json", "*.csv", "*.pt", "*.safetensors"]:
            import glob
            for fp in glob.glob(f"/root/tinker/experiments/implementations/{pattern}"):
                import shutil
                shutil.copy2(fp, result_dir)

        results_vol.commit()
        return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "experiment": exp_name,
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": "timeout (50min)",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "experiment": exp_name,
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
        }


@app.function(
    image=image,
    volumes={"/results": results_vol},
    timeout=600,
)
def aggregate_results() -> dict:
    """Aggregate all results into a summary."""
    import glob

    all_results = []
    for result_file in glob.glob("/results/*/seed_*/result.json"):
        with open(result_file) as f:
            all_results.append(json.load(f))

    # Group by experiment
    by_exp = {}
    for r in all_results:
        exp = r["experiment"]
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(r)

    summary = {}
    for exp, runs in by_exp.items():
        successes = [r for r in runs if r.get("success")]
        summary[exp] = {
            "total_runs": len(runs),
            "successful": len(successes),
            "failed": len(runs) - len(successes),
            "seeds": [r["seed"] for r in runs],
            "avg_time": round(sum(r["elapsed_seconds"] for r in runs) / len(runs), 1) if runs else 0,
        }

    # Save summary
    with open("/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()

    return summary


@app.local_entrypoint()
def main(
    exp: str = "",
    seeds: int = 5,
    dry_run: bool = False,
):
    """Launch multi-seed experiments on Modal GPUs."""
    seed_list = SEEDS[:seeds]

    if exp:
        # Single experiment
        if exp not in EXPERIMENTS:
            print(f"Unknown experiment: {exp}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
        experiments = {exp: EXPERIMENTS[exp]}
    else:
        experiments = EXPERIMENTS

    total_jobs = len(experiments) * len(seed_list)
    print(f"Launching {total_jobs} jobs ({len(experiments)} experiments x {len(seed_list)} seeds)")
    print(f"Experiments: {', '.join(experiments.keys())}")
    print(f"Seeds: {seed_list}")
    print(f"GPU: A10G | Timeout: 1h per job")

    if dry_run:
        print("\n[DRY RUN] Would launch:")
        for exp_name, exp_file in experiments.items():
            for seed in seed_list:
                print(f"  {exp_name} seed={seed} -> {exp_file}")
        return

    # Launch all jobs in parallel
    jobs = []
    for exp_name, exp_file in experiments.items():
        for seed in seed_list:
            jobs.append(run_experiment.spawn(exp_name, exp_file, seed))

    print(f"\n{total_jobs} jobs dispatched. Waiting for results...")

    # Collect results
    results = []
    for job in jobs:
        result = job.get()
        status = "OK" if result.get("success") else "FAIL"
        print(f"  [{status}] {result['experiment']} seed={result['seed']} ({result['elapsed_seconds']}s)")
        results.append(result)

    # Aggregate
    print("\nAggregating results...")
    summary = aggregate_results.remote()

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for exp_name, stats in summary.items():
        status = "PASS" if stats["successful"] == stats["total_runs"] else "PARTIAL"
        print(f"  [{status}] {exp_name}: {stats['successful']}/{stats['total_runs']} seeds OK, avg {stats['avg_time']}s")

    succeeded = sum(1 for r in results if r.get("success"))
    print(f"\nTotal: {succeeded}/{total_jobs} succeeded")
