"""Execute one user-launched sweep command, recording attempts and skipping successes.

A successful process is not proof of a valid empirical experiment. Validation is
separate; this manifest only prevents re-executing the identical successful job.
"""
import argparse
import datetime
import json
import subprocess
import uuid
from pathlib import Path


def execute(command, manifest, config_hash, seed):
    manifest = Path(manifest)
    previous = json.loads(manifest.read_text()) if manifest.exists() else {"attempts": []}
    if previous.get("config_hash", config_hash) != config_hash or previous.get("seed", seed) != seed:
        raise ValueError("Manifest config/seed identity mismatch")
    if previous.get("state") == "process_succeeded":
        return 0
    if previous.get("state") == "running":
        raise RuntimeError("Prior attempt is still marked running; reconcile it before retrying")
    attempt = {"attempt_id": uuid.uuid4().hex, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "command": command, "state": "running"}
    result = {"config_hash": config_hash, "seed": seed, "state": "running",
              "empirical_validation": "not_performed", "attempts": previous["attempts"]+[attempt]}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    def write():
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2)+"\n")
        temp.replace(manifest)
    write()
    try:
        process = subprocess.run(command, check=False)
        attempt["returncode"] = process.returncode
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
