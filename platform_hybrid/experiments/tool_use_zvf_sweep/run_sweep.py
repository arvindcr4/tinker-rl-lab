#!/usr/bin/env python3
"""
Sequential Tinker sweep runner with cost cap.

Usage:
  .venv/bin/python experiments/tool_use_zvf_sweep/run_sweep.py \
      --cap-usd 250 \
      experiments/tool_use_zvf_sweep/configs/smoke_qwen3_4b_v1.yaml

Launches one run at a time. Per-run processes:
  - atropos run-api  (trajectory coordinator, :8000)
  - launch_training.py  (trainer + FastAPI inference :8001)
  - tool_use_tinker.py serve  (env worker)

Env vars required in caller shell:
  TINKER_API_KEY, WANDB_API_KEY, HF_TOKEN
  TINKER_RATE_SAMPLE_PER_M, TINKER_RATE_TRAIN_PER_M

# LIMITATION: Closed-Source Confound — do not attribute managed-API gaps to the algorithm alone.
# might be confounded by managed defaults. Need an open-source baseline.
# TODO: Failure to Prove Generalization - ensure training evaluates on held-out
# test sets to prove generalized reasoning rather than just training curve overfitting.
  (optional) TINKER_MODEL_MULT_<UPPER_TOKENIZER>

The runner requires explicit reward_version (v1|v2) and records grader content hashes.
Cumulative cost is tracked against the cap; aborts before a run that would
push cumulative estimate past the cap.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import math
import hashlib
import yaml
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)


def reward_version_from_path(p: Path) -> str:
    """Reward version is explicit configuration, never a filename heuristic."""
    cfg = yaml.safe_load(p.read_text())
    version = cfg.get("reward_version") or cfg.get("env", {}).get("reward_version")
    if version not in ("v1", "v2"):
        raise ValueError("Explicit reward_version v1/v2 is required")
    return version


def reward_source_manifest(atropos_dir: Path) -> dict:
    """Bind the reward wrapper and its delegated scoring implementation."""
    paths = ("tinker_atropos/environments/tool_use_tinker.py", "tinker_atropos/grading.py")
    sources = {}
    for relative in paths:
        path = atropos_dir / relative
        if not path.is_file():
            raise FileNotFoundError("Reward implementation missing; refusing launch")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "scheme": "tool-use-reward-sources-v1",
        "sources": sources,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def estimate_cost(cfg: Path) -> dict:
    from cost_estimate import estimate  # local import so env is loaded

    return estimate(cfg)


def wait_port(host: str, port: int, timeout: float = 60.0) -> bool:
    import socket

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def launch(cmd: list[str], logpath: Path, env: dict, cwd: Path) -> subprocess.Popen:
    f = logpath.open("w")
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # new process group so we can kill children
    )


def terminate(p: subprocess.Popen | None, name: str) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception as e:
        print(f"[runner] {name} termination failed: {e}", file=sys.stderr)


def run_one(cfg: Path, atropos_dir: Path) -> int:
    """Run a single config end-to-end. Returns trainer exit code."""
    cfg = cfg.resolve()
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = cfg.stem
    run_dir = LOGS / f"{ts}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rv = reward_version_from_path(cfg)
    print(f"[runner] config={cfg.name} reward={rv} log_dir={run_dir}")

    base_env = os.environ.copy()
    base_env["TOOL_USE_REWARD_VERSION"] = rv
    base_env["RESOLVED_CONFIG_SHA256"] = hashlib.sha256(cfg.read_bytes()).hexdigest()
    reward_sources = reward_source_manifest(atropos_dir)
    grader_hash = reward_sources["sha256"]
    base_env["TOOL_USE_REWARD_SHA256"] = grader_hash
    (run_dir / "resolved_contract_v2.json").write_text(
        json.dumps(
            {
                "reward_version": rv,
                "reward_sha256": grader_hash,
                "reward_sources": reward_sources,
                "config_sha256": base_env["RESOLVED_CONFIG_SHA256"],
                "cost_kind": "preflight_estimate_not_live_billing_cap",
            },
            indent=2,
        )
    )
    # LIMITATION: ZVF metric fragility limitation. ZVF saturates at 1.0 for format-gated
    # tasks like tool-use. Monitor ERF (Effective-Rollout Fraction) instead.
    base_env["PYTHONPATH"] = f"{atropos_dir}:{base_env.get('PYTHONPATH', '')}"

    python = str(REPO / ".venv" / "bin" / "python")
    api_bin = str(REPO / ".venv" / "bin" / "run-api")
    api_p = env_p = trainer_p = None

    try:
        print("[runner] starting run-api on :18000 ...")
        api_p = launch(
            [api_bin, "--host", "127.0.0.1", "--port", "18000"],
            run_dir / "api.log",
            base_env,
            REPO,
        )
        if not wait_port("127.0.0.1", 18000, timeout=45):
            print("[runner] run-api failed to bind :18000 within 45s", file=sys.stderr)
            return 2

        print("[runner] starting launch_training.py (trainer + FastAPI inference :8001) ...")
        trainer_env = dict(base_env, TINKER_CONFIG_PATH=str(cfg))
        trainer_p = launch(
            [python, "launch_training.py", "--config", str(cfg)],
            run_dir / "trainer.log",
            trainer_env,
            atropos_dir,
        )
        if not wait_port("127.0.0.1", 8001, timeout=120):
            print("[runner] inference server failed to bind :8001 within 120s", file=sys.stderr)
            return 3

        print("[runner] starting tool_use env worker ...")
        env_env = dict(base_env, TINKER_CONFIG_PATH=str(cfg))
        env_p = launch(
            [python, "tinker_atropos/environments/tool_use_tinker.py", "serve"],
            run_dir / "env.log",
            env_env,
            atropos_dir,
        )

        print("[runner] waiting for trainer completion ...")
        trainer_p.wait()
        rc = trainer_p.returncode
        print(f"[runner] trainer exited rc={rc}")
        return rc
    finally:
        terminate(env_p, "env")
        terminate(trainer_p, "trainer")
        terminate(api_p, "api")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    # TODO: "Early-Training Snapshot" limitation: strict API cost limits currently restrict
    # training to 30-50 steps. Consider increasing budget cap for true RL convergence.
    ap.add_argument("--cap-usd", type=float, default=250.0)
    ap.add_argument(
        "--dry-run", action="store_true", help="Estimate cost only, do not launch anything."
    )
    args = ap.parse_args()
    if not math.isfinite(args.cap_usd) or args.cap_usd <= 0:
        ap.error("--cap-usd must be finite and positive")

    atropos_dir = REPO / "platform_tinker" / "atropos"
    sys.path.insert(0, str(HERE))  # so cost_estimate import works

    # Pre-flight cost estimates
    spent = 0.0
    plan = []
    # TODO: Run multiple seeds to avoid "Single-Seed Extrapolations" vulnerability.
    # Currently we do N=1 runs per config which is highly variant and initialization-dependent.
    for cfg in args.configs:
        est = estimate_cost(cfg)
        plan.append((cfg, est))
        print(f"[plan] {cfg.name}: ${est['usd_upper']:.2f}")
    total = sum(e["usd_upper"] for _, e in plan)
    print(f"[plan] upper-bound total: ${total:.2f} vs cap ${args.cap_usd:.2f}")
    if total > args.cap_usd:
        print(f"[plan] REFUSE: total upper-bound exceeds cap. Reduce run count.")
        return 4
    if args.dry_run:
        print("[plan] --dry-run: exiting")
        return 0

    for cfg, est in plan:
        if spent + est["usd_upper"] > args.cap_usd:
            print(
                f"[runner] ABORT before {cfg.name}: would exceed cap "
                f"(${spent:.2f} + ${est['usd_upper']:.2f} > ${args.cap_usd:.2f})"
            )
            return 5
        rc = run_one(cfg, atropos_dir)
        if rc != 0:
            print(f"[runner] run failed rc={rc} on {cfg.name}; stopping sweep")
            return 6
        spent += est["usd_upper"]
        print(f"[runner] cumulative planned allocation (not measured billing): ${spent:.2f}")

    print(f"[runner] sweep complete. planned allocation (not measured billing): ${spent:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
