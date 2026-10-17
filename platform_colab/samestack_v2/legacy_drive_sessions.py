# Historical review contract only; not interchangeable with canonical v3 evidence.
"""Stage complete Colab result bundles transactionally; own only new sessions."""

import gzip
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODAL_SCRIPT = ROOT / "platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
RUNNER = ROOT / "platform_colab/samestack_v2/colab_run_arms.py"
STAGE = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab"


def colab(*args, check=True):
    if args and args[0] == "exec":
        raise RuntimeError("Legacy driver is transfer/recovery-only; use drive_sessions.py for new runs")
    return subprocess.run(
        ["colab", *args], check=check, timeout=10920 if args[0] == "exec" else 180
    )


def bundle_complete(stage, name):
    try:
        receipt = json.loads((stage / f"{name}.complete.json").read_text())
        expected = {f"{name}.json", f"{name}_texts.json.gz"}
        return set(receipt["sha256"]) == expected and all(
            hashlib.sha256((stage / filename).read_bytes()).hexdigest() == digest
            for filename, digest in receipt["sha256"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


class TransferIncomplete(RuntimeError):
    """Remote artifacts may exist; preserve the session for transfer recovery."""


def transfer_bundle(session, name, stage):
    """Retryable transfer. A completion marker is the atomic visibility boundary."""
    stage.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=stage, prefix=".transfer-") as tmp:
        temp = pathlib.Path(tmp)
        for suffix in (".json", "_texts.json.gz"):
            proc = colab(
                "download",
                "-s",
                session,
                f"/content/results/samestack_gsm8k_v2s/{name}{suffix}",
                str(temp / f"{name}{suffix}"),
                check=False,
            )
            if proc.returncode:
                if suffix != ".json":
                    raise TransferIncomplete(
                        "Completion transfer failed; preserve the session and retry transfer"
                    )
                return False
        try:
            result = json.loads((temp / f"{name}.json").read_text())
            with gzip.open(temp / f"{name}_texts.json.gz", "rt") as handle:
                texts = json.load(handle)
            arm, seed = name.rsplit("_s", 1)
            n = result["n_eval"]
            if (
                result["arm"] != arm
                or result["seed"] != int(seed)
                or result.get("smoke") is not False
                or type(n) is not int
                or n <= 0
                or any(
                    len(texts.get(k, [])) != n
                    or any(not isinstance(t, str) or not t.strip() for t in texts[k])
                    for k in ("pre", "post")
                )
            ):
                raise ValueError("Incomplete or mismatched staging bundle")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TransferIncomplete(
                "Downloaded bundle is incomplete; preserve remote artifacts"
            ) from exc
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in temp.iterdir()}
        for path in temp.iterdir():
            os.replace(path, stage / path.name)
        marker = stage / f"{name}.complete.json"
        marker.with_suffix(".tmp").write_text(
            json.dumps({"schema": "colab-staging-v2", "sha256": hashes})
        )
        marker.with_suffix(".tmp").replace(marker)
        return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    session, runs = argv[0], argv[1:]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session) or any(
        not re.fullmatch(r"(?:ppo|ppo_8x8|grpo_g8|grpo_g2):[0-9]+", run) for run in runs
    ):
        raise ValueError("Invalid exact session/run identity")
    # No substring lookup: session identifiers are the first field of each CLI row.
    proc = subprocess.run(
        ["colab", "sessions"], capture_output=True, text=True, check=True, timeout=60
    )
    existing = {line.split()[0] for line in proc.stdout.splitlines() if line.split()}
    created = False
    retain_for_recovery = False
    try:
        if session not in existing:
            colab("new", "--gpu", "L4", "-s", session)
            created = True
        colab(
            "upload", "-s", session, str(MODAL_SCRIPT), "/content/modal_samestack_gsm8k_cot_v2.py"
        )
        colab("upload", "-s", session, str(RUNNER), "/content/colab_run_arms.py")
        for run in runs:
            arm, seed = run.split(":")
            name = f"{arm}_s{seed}"
            if bundle_complete(STAGE, name):
                continue
            # Try existing remote artifacts before repeating expensive training.
            if not transfer_bundle(session, name, STAGE):
                retain_for_recovery = True
                colab(
                    "exec",
                    "-s",
                    session,
                    "-f",
                    str(RUNNER),
                    "--timeout",
                    "10800",
                    "--env",
                    f"RUNS={run}",
                    "--env",
                    "SMOKE=0",
                )
                if not transfer_bundle(session, name, STAGE):
                    raise RuntimeError("Result transfer incomplete; bundle not published")
            retain_for_recovery = False
            print(f"STAGED {name}", flush=True)
    except TransferIncomplete:
        retain_for_recovery = True
        raise
    finally:
        if created and not retain_for_recovery:
            colab("stop", "-s", session, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
