"""Modal GRPO Campaign — SkyRL launcher.

Runs the canonical GSM8K GRPO run via SkyRL on a single Modal H100, invoking the
recipe documented in ``platform_hybrid/skyrl/configs/grpo_gsm8k.yaml``::

    uv run --extra vllm -m skyrl_train.entrypoints.main_base @configs/grpo_gsm8k.yaml

SkyRL is not pip-installable, so the image clones ``NovaSky-AI/SkyRL`` at the
``skyrl_train-v0.4.0`` tag (the same pin the vast.ai provisioner uses) and the
repo — carrying the GSM8K config — is mounted in. This is the real SkyRL path,
not a TRL stand-in. Sister script to ``modal_grpo_trl.py`` / ``modal_grpo_verl.py``.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("tinker-rl-skyrl-grpo")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
TINKER_KEY = os.environ.get("TINKER_API_KEY", "")

# Repo root: platform_hybrid/experiments/modal/<this file> -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
SKYRL_TAG = "skyrl_train-v0.4.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("packaging", "wheel", "ninja", "setuptools", "uv")
    .run_commands(
        # Clone the pinned SkyRL release the config's entrypoint lives in.
        f"git clone --depth 1 --branch {SKYRL_TAG} "
        "https://github.com/NovaSky-AI/SkyRL.git /root/SkyRL"
    )
    .env({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
        "WANDB_PROJECT": "tinker-rl-lab-world-class",
        "SKYRL_CHECKOUT": "/root/SkyRL",
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
def run_skyrl_qwen3_8b():
    """Execute the SkyRL GRPO GSM8K recipe on a single H100."""
    import subprocess
    import time

    REPO = "/root/tinker-rl-lab"
    CONFIG = f"{REPO}/platform_hybrid/skyrl/configs/grpo_gsm8k.yaml"

    # The recipe command documented at the top of grpo_gsm8k.yaml.
    cmd = [
        "uv", "run", "--extra", "vllm",
        "-m", "skyrl_train.entrypoints.main_base",
        f"@{CONFIG}",
    ]
    print("[skyrl] cmd:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd="/root/SkyRL", env=env)
    duration = time.time() - start
    print(f"[skyrl] subprocess exit={proc.returncode} in {duration:.1f}s")

    return {
        "framework": "skyrl",
        "mode": "real",
        "model": "Qwen/Qwen3-8B",
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
    print("Launching SkyRL GRPO (GSM8K recipe) on H100...")
    result = run_skyrl_qwen3_8b.remote()
    os.makedirs("/home/user/workspace/elevation_outputs", exist_ok=True)
    out = "/home/user/workspace/elevation_outputs/modal_skyrl_grpo.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. Wrote {out}.")
    print(f"exit={result.get('subprocess_exit')}  duration={result.get('duration_s')}s")
