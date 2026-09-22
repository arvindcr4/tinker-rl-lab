"""E13 BALROG actor05: pinned Qwen3.6-35B-A3B base served via transformers on 1xH200.

Infrastructure only, not a result. Bounded: $8 all-in intent, 3600 s envelope,
single container, bearer auth, immutable model identity verified at startup.
Episode dispatch is a separate, sealed step — this app only serves generations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

APP_NAME = "e13-balrog-actor05"
HF_REPO = "Qwen/Qwen3.6-35B-A3B"
HF_COMMIT = "995ad96eacd98c81ed38be0c5b274b04031597b0"
# Recorded deviation 2026-09-20: seed809 adapter weights are unrecoverable
# (Tinker run purged, HF mirror holds only .gitattributes). Actor05 serves the
# pinned BASE model. Native-score admissibility vs the v10 base+adapter binding
# requires explicit lead acknowledgment; proxy-grade generations unaffected.
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
BASE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MAX_USD_ALL_IN = "8.00"
WALL_SECONDS = 3600

actor_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.55",
        "accelerate",
        "fastapi",
        "uvicorn[standard]",
        "huggingface_hub",
        "flash-linear-attention",
    )
    .add_local_file(
        Path(__file__).resolve().parent / "hf_actor_server.py",
        "/root/hf_actor_server.py",
        copy=True,
    )
    .run_commands(
        "find /usr/local/lib/python3.12/site-packages -maxdepth 5 -name nvcc -type f 2>/dev/null | head -n 3; echo ---; ls /usr/local/cuda/bin 2>/dev/null | head -n 3; echo probed"
    )
)
core_secret = modal.Secret.from_name("pavlov-e1-e14")
auth_secret = modal.Secret.from_name("e13-actor05-auth")

hf_cache_vol = modal.Volume.from_name("e13-actor05-hfcache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=actor_image,
    gpu="H200",
    cpu=4.0,
    memory=131072,
    timeout=WALL_SECONDS,
    max_containers=1,
    scaledown_window=3600,
    secrets=[core_secret, auth_secret],
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def generate(prompt: str, max_tokens: int = 512) -> str:
    """Modal-native generation entrypoint (bypasses the public HTTP proxy)."""
    import sys
    sys.path.insert(0, "/root")
    from hf_actor_server import generate_text
    return generate_text(prompt, max_tokens)


@app.local_entrypoint()
def smoke():
    out = generate.remote("Reply with exactly: OK", 8)
    print("SMOKE:", repr(out[:200]))


@app.local_entrypoint()
def bench(n: int = 256):
    import time
    t0 = time.time()
    out = generate.remote("Explain in one paragraph why the sky is blue.", n)
    dt = time.time() - t0
    print(f"BENCH secs={dt:.1f} chars={len(out)}")
    print(out[:300])


@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    cmd = [
        "uvicorn", "hf_actor_server:app",
        "--host", "0.0.0.0", "--port", "8000", "--workers", "1",
        "--app-dir", "/root",
    ]
    proc = subprocess.Popen(cmd, cwd="/root")
    try:
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
