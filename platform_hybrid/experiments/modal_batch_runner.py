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

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.4.0",
    "transformers>=4.45.0",
    "trl>=0.12.0",
    "datasets>=3.0.0",
    "accelerate>=1.0.0",
    "peft>=0.13.0",
    "numpy>=1.26.0",
    "scipy>=1.14.0",
)

# Mount repo
repo_mount = modal.Mount.from_local_dir(
    str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    remote_path="/root/tinker-rl-lab",
    condition=lambda path: (
        not any(
            x in path for x in [".git", "__pycache__", "wandb", ".pyc", "platform_hybrid/paper/"]
        )
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
    """Invoke supported CLI arguments; compile source without generating patches."""
    import subprocess
    import time
    import uuid

    if num_steps <= 0 or type(seed) is not int:
        raise ValueError("Valid seed and positive step budget required")
    exp_path = Path(
        "/root/tinker-rl-lab/platform_hybrid/experiments/implementations/trl_grpo_math.py"
    )
    compile(exp_path.read_text(), str(exp_path), "exec")  # syntax preflight, never executes source
    attempt_id = uuid.uuid4().hex
    result_dir = Path(f"/results/trl_grpo/seed_{seed}/{attempt_id}")
    result_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    result = dict(
        experiment="trl_grpo_math",
        seed=seed,
        attempt_id=attempt_id,
        status="running",
        success=False,
    )
    try:
        proc = subprocess.run(
            [
                "python",
                str(exp_path),
                "--seed",
                str(seed),
                "--max_steps",
                str(num_steps),
                "--num_train_problems",
                "200",
                "--output_dir",
                str(result_dir / "model"),
            ],
            capture_output=True,
            text=True,
            timeout=1200,
            cwd="/root/tinker-rl-lab",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "RESULTS_DIR": str(result_dir)},
        )
        result.update(
            success=proc.returncode == 0,
            status="completed" if proc.returncode == 0 else "failed",
            returncode=proc.returncode,
            stdout_tail=proc.stdout[-1500:],
            stderr_tail=proc.stderr[-1000:],
        )
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error="1200-second deadline exceeded")
    except Exception as exc:
        result.update(status="error", error=str(exc))
    finally:
        result["elapsed_seconds"] = round(time.time() - start, 1)
        # Process success alone is never an empirical evidence attestation.
        result.update(measured=False, claim_eligible=False, artifact_type="execution_receipt")
        (result_dir / "result.json").write_text(json.dumps(result, indent=2))
        results_vol.commit()
    return result


@app.local_entrypoint()
def main(seeds: int = 3, steps: int = 20):
    """Launch multi-seed TRL experiments on Modal GPUs."""
    source = Path(__file__).resolve().parent / "implementations" / "trl_grpo_math.py"
    compile(source.read_text(), str(source), "exec")
    if type(seeds) is not int or not 1 <= seeds <= 3 or steps <= 0:
        raise ValueError("Valid seed count and positive steps required")
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
