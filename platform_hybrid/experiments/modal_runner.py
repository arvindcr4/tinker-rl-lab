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
def run_experiment(exp_name: str, exp_file: str, seed: int, invocation_id: str) -> dict:
    """Each attempt writes a distinct terminal execution receipt, including failure."""
    import subprocess
    import time
    import uuid
    from pathlib import Path
    if not invocation_id or not invocation_id.isalnum():
        raise ValueError("Explicit safe invocation ID required")
    if exp_name not in EXPERIMENTS or EXPERIMENTS[exp_name] != exp_file:
        raise ValueError("Unknown experiment entrypoint")
    attempt_id = uuid.uuid4().hex
    result_dir = Path(f"/results/{invocation_id}/{exp_name}/seed_{seed}/{attempt_id}")
    result_dir.mkdir(parents=True, exist_ok=False)
    result = dict(experiment=exp_name, seed=seed, invocation_id=invocation_id,
        attempt_id=attempt_id, success=False, status="running", artifact_type="execution_receipt",
        measured=False, claim_eligible=False)
    start = time.time()
    try:
        proc = subprocess.run(["python", f"/root/tinker/platform_hybrid/experiments/implementations/{exp_file}", "--seed", str(seed)],
            capture_output=True, text=True, timeout=3000, cwd="/root/tinker",
            env={**os.environ, "SEED": str(seed), "RESULTS_DIR": str(result_dir)})
        result.update(success=proc.returncode == 0, status="completed" if proc.returncode == 0 else "failed",
            returncode=proc.returncode, stdout_tail=proc.stdout[-2000:], stderr_tail=proc.stderr[-2000:])
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error="3000-second deadline exceeded")
    except Exception as exc:
        result.update(status="error", error=str(exc))
    finally:
        result["elapsed_seconds"] = round(time.time()-start, 1)
        (result_dir / "result.json").write_text(json.dumps(result, indent=2))
        results_vol.commit()
    return result


@app.function(image=image, volumes={"/results": results_vol}, timeout=600)
def aggregate_results(invocation_id: str, selected_attempt_ids: list[str]) -> dict:
    """Aggregate only the explicit invocation/attempt manifest, never stale glob hits."""
    from pathlib import Path
    if not invocation_id.isalnum() or len(selected_attempt_ids) != len(set(selected_attempt_ids)):
        raise ValueError("Invalid selected invocation manifest")
    selected = set(selected_attempt_ids)
    rows = []
    for path in Path(f"/results/{invocation_id}").glob("*/seed_*/*/result.json"):
        record = json.loads(path.read_text())
        if record.get("invocation_id") == invocation_id and record.get("attempt_id") in selected:
            rows.append(record)
    if {r["attempt_id"] for r in rows} != selected or len(rows) != len(selected):
        raise ValueError("Missing or duplicate terminal receipts")
    summary = {}
    for record in rows:
        if record["status"] not in ("completed", "failed", "timeout", "error"):
            raise ValueError("Nonterminal receipt")
        group = summary.setdefault(record["experiment"], dict(total_runs=0, successful=0, failed=0, times=[]))
        group["total_runs"] += 1
        group["successful"] += int(record["success"])
        group["failed"] += int(not record["success"])
        group["times"].append(record["elapsed_seconds"])
    for group in summary.values():
        times = group.pop("times")
        group["avg_time"] = sum(times)/len(times)
    (Path("/results") / invocation_id / "summary.json").write_text(json.dumps(summary, indent=2))
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

    import uuid
    invocation_id = uuid.uuid4().hex
    jobs = []
    for name, f in experiments.items():
        for s in seed_list:
            jobs.append(run_experiment.spawn(name, f, s, invocation_id))

    print(f"{total} jobs dispatched. Waiting...")

    results = []
    for job in jobs:
        r = job.get()
        tag = "OK" if r.get("success") else "FAIL"
        print(f"  [{tag}] {r['experiment']} seed={r['seed']} ({r['elapsed_seconds']}s)")
        if not r.get("success") and r.get("stderr_tail"):
            print(f"        stderr: {r['stderr_tail'][-300:]}")
        results.append(r)

    summary = aggregate_results.remote(invocation_id, [r["attempt_id"] for r in results])
    print("\n" + "=" * 50)
    for name, s in summary.items():
        tag = "PASS" if s["successful"] == s["total_runs"] else "PARTIAL"
        print(f"  [{tag}] {name}: {s['successful']}/{s['total_runs']} OK, avg {s['avg_time']}s")

    ok = sum(1 for r in results if r.get("success"))
    print(f"\nTotal: {ok}/{total} succeeded")
