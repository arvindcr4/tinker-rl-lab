"""
Modal GPU Runner for Tinker RL Lab Multi-Seed Experiments
=========================================================
Runs all experiments across seeds on Modal A10G GPUs (SEEDS pool has 10; released runs use the first 5: 42, 123, 456, 789, 1024).

Usage:
    modal run experiments/modal_runner.py               # Run all experiments
    modal run experiments/modal_runner.py --exp trl_grpo # Run specific experiment
    modal run experiments/modal_runner.py --seeds 3      # Run 3 seeds instead of 5
    modal run experiments/modal_runner.py --dry-run      # List what would run
"""

import modal
import os
import json

app = modal.App("tinker-rl-lab")

# GPU image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML — layer 1
        "torch==2.6.0",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "gymnasium==0.29.1",
    )
    .pip_install(
        # HuggingFace ecosystem — layer 2
        "transformers>=4.46.0,<5.0.0",
        "trl>=1.0.0,<1.2.0",
        "datasets>=3.0.0",
        "accelerate>=1.4.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "safetensors>=0.4.0",
    )
    .pip_install(
        # RL libraries — compatible with gymnasium 0.29.1
        "stable-baselines3==2.3.2",
        "rliable>=1.1.0",
        "matplotlib",
    )
    .run_commands(
        # These have conflicting gymnasium pins but work fine at runtime with 0.29.1
        "pip install --no-deps tianshou==1.1.0 d3rlpy==2.6.0 rl-games==1.6.5",
        # Install their non-gymnasium deps (including gym for d3rlpy, sensai for tianshou)
        "pip install gym h5py structlog colorama dataclasses-json 'deepdiff<8.0.0' tensorboard tensorboardX setproctitle watchdog sensai-utils numba overrides pettingzoo wandb",
        "pip install gymnasium==0.29.1",  # Re-pin after pettingzoo overwrites it
    )
    .add_local_dir(
        os.path.expanduser("~/tinker"),
        remote_path="/root/tinker",
        ignore=[".git", "wandb", "__pycache__", "*.pyc"],
    )
)

# Volume for persisting results
results_vol = modal.Volume.from_name("tinker-results", create_if_missing=True)

# HF token for gated models (Llama etc.)
hf_secret = modal.Secret.from_name("huggingface-secret")

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

SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/results": results_vol},
    secrets=[hf_secret],
    timeout=3600,
    retries=1,
)
def run_experiment(exp_name: str, exp_file: str, seed: int) -> dict:
    """Run a single experiment with a given seed on GPU."""
    import subprocess
    import time
    import glob
    import shutil

    exp_path = f"/root/tinker/experiments/implementations/{exp_file}"
    result_dir = f"/results/{exp_name}/seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)

    gpu_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()
    print(f"[{exp_name}] seed={seed} | GPU: {gpu_name}")

    # HF login if token available
    hf_token = os.environ.get("HF_TOKEN", "")
    run_env = {**os.environ, "SEED": str(seed), "RESULTS_DIR": result_dir}
    if hf_token:
        run_env["HF_TOKEN"] = hf_token
        run_env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    start = time.time()
    try:
        proc = subprocess.run(
            ["python", exp_path, "--seed", str(seed)],
            capture_output=True,
            text=True,
            timeout=3000,
            cwd="/root/tinker",
            env=run_env,
        )
        elapsed = time.time() - start

        result = {
            "experiment": exp_name,
            "seed": seed,
            "success": proc.returncode == 0,
            "elapsed_seconds": round(elapsed, 1),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }

        with open(f"{result_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)

        for pattern in ["*.json", "*.csv", "*.pt", "*.safetensors"]:
            for fp in glob.glob(f"/root/tinker/experiments/implementations/{pattern}"):
                shutil.copy2(fp, result_dir)

        results_vol.commit()
        return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        try:
            results_vol.commit()
        except Exception:
            pass
        return {
            "experiment": exp_name,
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": "timeout (50min)",
        }
    except Exception as e:
        elapsed = time.time() - start
        try:
            results_vol.commit()
        except Exception:
            pass
        return {
            "experiment": exp_name,
            "seed": seed,
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
        }


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def aggregate_results() -> dict:
    """Aggregate all results into a summary."""
    import glob

    all_results = []
    for rf in glob.glob("/results/*/seed_*/result.json"):
        with open(rf) as f:
            all_results.append(json.load(f))

    by_exp = {}
    for r in all_results:
        by_exp.setdefault(r["experiment"], []).append(r)

    summary = {}
    for exp, runs in by_exp.items():
        ok = [r for r in runs if r.get("success")]
        summary[exp] = {
            "total_runs": len(runs),
            "successful": len(ok),
            "failed": len(runs) - len(ok),
            "seeds": [r["seed"] for r in runs],
            "avg_time": round(sum(r["elapsed_seconds"] for r in runs) / max(len(runs), 1), 1),
        }

    with open("/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(exp: str = "", seeds: int = 5, dry_run: bool = False):
    seed_list = SEEDS[:seeds]
    experiments = {exp: EXPERIMENTS[exp]} if exp else EXPERIMENTS

    if exp and exp not in EXPERIMENTS:
        print(f"Unknown: {exp}. Available: {', '.join(EXPERIMENTS.keys())}")
        return

    total = len(experiments) * len(seed_list)
    print(f"Launching {total} jobs ({len(experiments)} exp x {len(seed_list)} seeds) on A10G")

    if dry_run:
        for name, f in experiments.items():
            for s in seed_list:
                print(f"  {name} seed={s} -> {f}")
        return

    jobs = []
    for name, f in experiments.items():
        for s in seed_list:
            jobs.append(run_experiment.spawn(name, f, s))

    print(f"{total} jobs dispatched. Waiting...")

    results = []
    for job in jobs:
        r = job.get()
        tag = "OK" if r.get("success") else "FAIL"
        print(f"  [{tag}] {r['experiment']} seed={r['seed']} ({r['elapsed_seconds']}s)")
        if not r.get("success") and r.get("stderr_tail"):
            print(f"        stderr: {r['stderr_tail'][-300:]}")
        results.append(r)

    summary = aggregate_results.remote()
    print("\n" + "=" * 50)
    for name, s in summary.items():
        tag = "PASS" if s["successful"] == s["total_runs"] else "PARTIAL"
        print(f"  [{tag}] {name}: {s['successful']}/{s['total_runs']} OK, avg {s['avg_time']}s")

    ok = sum(1 for r in results if r.get("success"))
    print(f"\nTotal: {ok}/{total} succeeded")
