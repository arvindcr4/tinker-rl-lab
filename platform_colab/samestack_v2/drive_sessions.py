"""Preserve complete versioned Colab bundles before stopping a session.

Existing JSON does not skip a run. The remote core checks the current immutable
protocol/config/source identity before reusing a completed run. This is finished
artifact reuse, not optimizer/RNG-state recovery of interrupted training.
Downloads enter a temporary per-run directory. Every advertised file and hash,
run schema, texts, adapters, and tokenizer must verify before atomic promotion.
A failed transfer leaves the remote session running for a retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "platform_hybrid/experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from platform_hybrid.experiments.samestack_contract import completion_path, strict_json_load, validate_completion
from platform_hybrid.experiments.samestack_protocol import PROTOCOL_HASH, PROTOCOL_ID

MODAL_SCRIPT = HELPERS / "modal/modal_samestack_gsm8k_cot_v2.py"
RUNNER = ROOT / "platform_colab/samestack_v2/colab_run_arms.py"
STAGE = HELPERS / "results" / (PROTOCOL_ID + "_colab")


def colab(*args, check=True):
    print("$ colab", " ".join(str(arg) for arg in args), flush=True)
    return subprocess.run(["colab", *map(str, args)], check=check)


def _safe_inventory(marker, name):
    if not isinstance(marker, dict) or not isinstance(marker.get("files"), dict) or not marker["files"]:
        raise ValueError("missing remote artifact inventory")
    files = marker["files"]
    for relative, entry in files.items():
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative
                or str(path) != relative or not isinstance(entry, dict)
                or (relative not in (name + ".json", name + "_texts.json.gz")
                    and not relative.startswith("adapters/" + name + "/"))):
            raise ValueError(f"unsafe or unrelated artifact path: {relative}")
    return files


def preserve_run(session, name, stage, *, expected=None, command=colab):
    """Download and atomically promote one bundle; retain partial files on failure."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial-", dir=stage))
    remote = f"/content/results/{PROTOCOL_ID}"
    result_path = temporary / (name + ".json")
    marker_path = completion_path(result_path)
    # Marker is downloaded into the private temporary directory, never published
    # as complete before its contents have independently passed local validation.
    command("download", "-s", session, f"{remote}/{marker_path.name}", str(marker_path))
    inventory = _safe_inventory(strict_json_load(marker_path), name)
    for relative in sorted(inventory):
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        command("download", "-s", session, f"{remote}/{relative}", str(destination))
    record = validate_completion(result_path, expected=expected)
    final = stage / name
    if final.exists():
        # Preserve prior bundles, including incomplete or differently identified
        # ones. Never overwrite old evidence while promoting a retry.
        archived = stage / f".{name}.previous-{os.urandom(6).hex()}"
        os.replace(final, archived)
    os.replace(temporary, final)
    validate_completion(final / result_path.name, expected=expected)
    return record


def drive_session(session, runs, *, stage=STAGE, command=colab, source=None):
    """Stop only after all requested bundles have been independently preserved."""
    from platform_colab.samestack_v2.colab_run_arms import parse_runs
    parsed = parse_runs(",".join(runs))
    if not parsed:
        raise ValueError("at least one run is required")
    # Session creation is explicit to the CLI; this helper never guesses by a
    # substring match against another session's name.
    source = source or capture_source()
    expected = {"environment": "colab", "protocol_id": PROTOCOL_ID, "protocol_hash": PROTOCOL_HASH,
                "source_commit": source["commit"], "source_dirty_hash": source["dirty_hash"]}
    with tempfile.TemporaryDirectory(prefix="samestack-upload-") as directory:
        source_path = Path(directory) / "samestack_source.json"
        source_path.write_text(json.dumps(source, allow_nan=False), encoding="utf-8")
        archive_path = Path(directory) / "samestack_code.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for relative in source["source_files"]:
                local = ROOT / relative
                if not local.resolve().is_relative_to(ROOT.resolve()) or not local.is_file():
                    raise ValueError(f"source manifest path unavailable: {relative}")
                bundle.write(local, relative)
        uploads = [(archive_path, "/content/samestack_code.zip"),
                   (RUNNER, "/content/" + RUNNER.name),
                   (source_path, "/content/samestack_source.json")]
        for local, remote in uploads:
            command("upload", "-s", session, str(local), remote)
        for arm, seed in parsed:
            name = f"{arm}_s{seed}"
            # Always ask the core, which resolves current revisions/config before
            # testing completion. An existing local JSON never bypasses this step.
            command("exec", "-s", session, "-f", str(RUNNER), "--timeout", "10800",
                    "--env", f"RUNS={arm}:{seed}", "--env", "SMOKE=0")
            preserve_run(session, name, stage, expected=expected, command=command)
            print(f"PRESERVED {name}", flush=True)
    for arm, seed in parsed:
        validate_completion(Path(stage) / f"{arm}_s{seed}" / f"{arm}_s{seed}.json", expected=expected)
    command("stop", "-s", session)


def capture_source():
    """Obtain provenance locally before upload; no VM git metadata is invented."""
    from platform_hybrid.experiments.samestack_manifest import capture_source as capture
    return capture(ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--new", action="store_true", help="create the named L4 session first")
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args(argv)
    if args.new:
        colab("new", "--gpu", "L4", "-s", args.session)
    drive_session(args.session, args.runs, stage=args.stage)


if __name__ == "__main__":
    main()
