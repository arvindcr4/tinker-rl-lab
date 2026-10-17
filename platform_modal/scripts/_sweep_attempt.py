"""Execute one user-launched sweep command with content-bound, isolated attempts.

A successful process is not proof of a valid empirical experiment. Success may
only be skipped after rechecking the requested command, YAML bytes, trainer
source and local runtime versions. Old/unverifiable receipts fail closed.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml


RUNTIME_PACKAGES = ("torch", "trl", "unsloth", "transformers", "peft", "datasets",
                    "numpy", "PyYAML", "wandb", "math-verify", "latex2sympy2-extended")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_identity(command, config_hash, seed):
    """Validate the generated Python-script route without importing the trainer."""
    command = list(command)
    if len(command) != 6 or command[2] != "--config" or command[4] != "--seed":
        raise ValueError("Only the generated Python trainer --config PATH --seed N command is supported")
    python, trainer, config = (Path(command[index]) for index in (0, 1, 3))
    if not all(path.is_absolute() and path.is_file() for path in (python, trainer, config)):
        raise ValueError("Interpreter, trainer and YAML paths must be absolute existing files")
    # Resolving symlinks would incorrectly equate different virtualenvs.
    if python.absolute() != Path(sys.executable).absolute():
        raise ValueError("Trainer must use the runner interpreter so runtime versions can be verified")
    if int(command[5]) != seed:
        raise ValueError("Command seed does not match the requested seed")
    actual_hash = sha256(config)
    if actual_hash != config_hash:
        raise ValueError("Actual YAML digest does not match config_hash; regenerate the plan")
    raw = yaml.safe_load(config.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("tinker"), dict):
        raise ValueError("Sweep YAML must contain a tinker mapping")
    if not all(isinstance(raw["tinker"].get(key), str) and raw["tinker"][key]
               for key in ("checkpoint_dir", "wandb_run_name")):
        raise ValueError("Sweep YAML must declare checkpoint_dir and wandb_run_name")
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema": "sweep-command-identity-v3", "command": command, "seed": seed,
            "config_sha256": actual_hash, "trainer_sha256": sha256(trainer),
            "runner_sha256": sha256(__file__), "python_sha256": sha256(python),
            "python_version": sys.version, "package_versions": versions,
            "working_directory": str(Path.cwd().resolve())}


def execute(command, manifest, config_hash, seed):
    # Verify actual inputs BEFORE trusting a previous successful receipt.
    identity = command_identity(command, config_hash, seed)
    command = identity["command"]
    manifest = Path(manifest).resolve()
    previous = json.loads(manifest.read_text()) if manifest.exists() else {"attempts": []}
    if manifest.exists() and previous.get("identity") != identity:
        raise ValueError("Manifest identity is stale or unverified; use a new manifest for changed inputs/source/runtime")
    if previous.get("state") == "process_succeeded":
        return 0
    if previous.get("state") == "running":
        raise RuntimeError("Prior attempt is still marked running; reconcile it before retrying")

    attempt_id = uuid.uuid4().hex
    attempt_dir = manifest.parent / (manifest.stem + "_attempts") / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    raw = yaml.safe_load(Path(command[3]).read_text())
    checkpoint_dir = Path(raw["tinker"]["checkpoint_dir"]).resolve() / attempt_id
    run_name = raw["tinker"]["wandb_run_name"] + "-" + attempt_id
    raw["tinker"].update(checkpoint_dir=str(checkpoint_dir), wandb_run_name=run_name)
    effective_config = attempt_dir / "config.yaml"
    effective_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    executed_command = command.copy()
    executed_command[3] = str(effective_config)
    attempt = {"attempt_id": attempt_id, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": executed_command, "config_sha256": sha256(effective_config),
               "checkpoint_dir": str(checkpoint_dir), "wandb_run_name": run_name,
               "wandb_run_id": attempt_id, "state": "running"}
    result = {"identity": identity, "config_hash": config_hash, "seed": seed, "state": "running",
              "empirical_validation": "not_performed", "attempts": previous["attempts"]+[attempt]}

    def write():
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2)+"\n")
        temp.replace(manifest)
    write()
    try:
        # Input edits during preparation/execution must not produce a reusable success.
        if command_identity(command, config_hash, seed) != identity:
            raise ValueError("Inputs/source/runtime changed during attempt preparation")
        process = subprocess.run(executed_command, check=False,
                                 env={**os.environ, "WANDB_RUN_ID": attempt_id, "WANDB_RESUME": "never"})
        attempt["returncode"] = process.returncode
        if command_identity(command, config_hash, seed) != identity or sha256(effective_config) != attempt["config_sha256"]:
            raise ValueError("Inputs/source/runtime changed during execution; success cannot be reused")
        result["state"] = attempt["state"] = "process_succeeded" if process.returncode == 0 else "failed"
        return process.returncode
    except BaseException:
        result["state"] = attempt["state"] = "interrupted"
        raise
    finally:
        write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Training command required")
    raise SystemExit(execute(command, args.manifest, args.config_hash, args.seed))


if __name__ == "__main__":
    main()
