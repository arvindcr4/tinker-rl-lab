"""Modal GRPO Campaign — Tinker launcher.

Runs the canonical GSM8K GRPO run via the Tinker SDK on a single Modal H100,
invoking the repo's own ``platform_tinker/tinkerrl/grpo_cli.py`` (the same entry
point the local backend's ``_run_tinker`` uses). Sister script to
``modal_grpo_trl.py`` / ``modal_grpo_verl.py``. Writes result.json into
``/home/user/workspace/elevation_outputs/modal_tinker_grpo.json`` for pickup by
``experiments/results/aggregate_framework_comparison.py``.

The repo is mounted into the image so grpo_cli and the Tinker cookbook code run
verbatim — this is the real Tinker path, not a TRL stand-in.
"""

import json
import os
import sys
from pathlib import Path

import modal

app = modal.App("tinker-rl-tinker-grpo")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
TINKER_KEY = os.environ.get("TINKER_API_KEY", "")

# Repo root: platform_hybrid/experiments/modal/<this file> -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.46",
        "datasets>=3.0.0",
        "accelerate>=1.4.0",
        "peft>=0.13.0",
        "numpy",
        "wandb",
        "huggingface_hub",
        # Tinker SDK + cookbook (the framework grpo_cli drives).
        "tinker>=0.5.0",
        "tinker-cookbook>=0.2.0",
    )
    .env({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
        "WANDB_PROJECT": "tinker-rl-lab-world-class",
    })
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/tinker-rl-lab",
        ignore=[".git", "wandb", "__pycache__", "*.pyc", "checkpoints"],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[
        modal.Secret.from_dict({
            "HF_TOKEN": HF_TOKEN,
            "WANDB_API_KEY": WANDB_KEY,
            "TINKER_API_KEY": TINKER_KEY,
        })
    ],
)
def run_tinker_qwen3_8b():
    """Execute the Tinker GRPO GSM8K run on a single H100 via grpo_cli."""
    import subprocess
    import time

    MODEL = "Qwen/Qwen3-8B"
    REPO = "/root/tinker-rl-lab"

    cmd = [
        sys.executable, "-m", "platform_tinker.tinkerrl.grpo_cli",
        "--preset", "gsm8k",
        "--model", MODEL,
        "--steps", "30",
        "--seed", "211",
    ]
    print("[tinker] cmd:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", "tinker-rl-lab-world-class")
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    duration = time.time() - start
    print(f"[tinker] subprocess exit={proc.returncode} in {duration:.1f}s")

    return {
        "framework": "tinker",
        "mode": "real",
        "model": MODEL,
        "algorithm": "GRPO",
        "seed": 211,
        "task": "gsm8k",
        "platform": "modal-h100",
        "duration_s": duration,
        "subprocess_exit": proc.returncode,
        "command": " ".join(cmd),
    }


@app.local_entrypoint()
def main():
    print("Launching Tinker GRPO (Qwen3-8B, GSM8K, 30 steps) on H100...")
    result = run_tinker_qwen3_8b.remote()
    os.makedirs("/home/user/workspace/elevation_outputs", exist_ok=True)
    out = "/home/user/workspace/elevation_outputs/modal_tinker_grpo.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. Wrote {out}.")
    print(f"exit={result.get('subprocess_exit')}  duration={result.get('duration_s')}s")
