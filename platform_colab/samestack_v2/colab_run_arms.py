"""Run the versioned same-stack core on Colab; importing has no side effects.

The driver uploads this bootstrap, a hash-checked source archive, and
samestack_source.json. Runtime package versions are checked against one shared
exact lock. Existing JSON alone never skips training; the core validates an
immutable protocol/config/source-bound artifact completion marker.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import zipfile
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

SELF = "/content/colab_run_arms.py"
HERE = Path(globals().get("__file__", SELF)).resolve()
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = "/content/platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py"
RESULTS = "/content/results"
SOURCE = "/content/samestack_source.json"


def ensure_packages(*, install=True):
    """Verify exact versions; optionally reconcile all mismatches, not only missing ones."""
    from platform_hybrid.experiments.samestack_dependencies import REQUIREMENTS, runtime_versions, verify_runtime
    before = runtime_versions()
    try:
        return verify_runtime(before)
    except ValueError:
        if not install:
            raise
    loaded = [name for name in ("torch", "transformers", "peft", "datasets", "numpy", "accelerate") if name in sys.modules]
    if loaded:
        raise RuntimeError("Restart the runtime before replacing already imported packages: " + ", ".join(loaded))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *REQUIREMENTS])
    subprocess.check_call([sys.executable, "-m", "pip", "check"])
    return verify_runtime()


def parse_runs(value):
    from platform_hybrid.experiments.samestack_contract import ARMS, SEEDS
    result = []
    for item in value.split(","):
        try:
            arm, raw_seed = item.split(":")
            seed = int(raw_seed)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid run specification: {item!r}") from exc
        if arm not in ARMS or seed not in SEEDS:
            raise ValueError(f"arm/seed outside prespecified experiment: {item!r}")
        if (arm, seed) in result:
            raise ValueError(f"duplicate requested run: {item!r}")
        result.append((arm, seed))
    return result


def load_core(script=SCRIPT):
    # The entrypoint deliberately exposes a neutral callable. Never impersonate
    # Modal or replace sys.modules['modal'] globally.
    os.environ["SAMESTACK_EXECUTION_ENV"] = "colab"
    os.environ["SAMESTACK_RESULTS_DIR"] = RESULTS
    spec = importlib.util.spec_from_file_location("samestack_colab_core", script)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import same-stack core: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_arm_local", None)):
        raise RuntimeError("runner does not expose the versioned platform-neutral core")
    return module


def prepare_bundle(archive="/content/samestack_code.zip", source_path=SOURCE, destination="/content"):
    """Extract only declared source files after checking their original hashes."""
    def reject(value):
        raise ValueError(f"Non-finite JSON constant in source metadata: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate source metadata key: {key}")
            result[key] = value
        return result
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            reject(value)
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    source = json.loads(Path(source_path).read_text(encoding="utf-8"), parse_constant=reject,
                        object_pairs_hook=unique)
    finite(source)
    if not isinstance(source, dict):
        raise ValueError("source metadata must be an object")
    files = source.get("source_files", {})
    if source.get("status") != "observed" or not files:
        raise ValueError("observed source file hashes are required")
    destination = Path(destination)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError("source archive inventory does not match source manifest")
        checked = []
        for name in names:
            relative = PurePosixPath(name)
            target = destination / name
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or str(relative) != name or not target.resolve().is_relative_to(destination.resolve())):
                raise ValueError("unsafe source archive path")
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise ValueError(f"uploaded source hash mismatch: {name}")
            # A persistent Colab kernel may retain code imported by an earlier
            # exec even after files are replaced. Refuse stale/untracked helpers.
            for module_name, module in list(sys.modules.items()):
                if not module_name.startswith("platform_hybrid.experiments.samestack_"):
                    continue
                origin = getattr(module, "__file__", None)
                if origin and Path(origin).resolve() == target.resolve():
                    if getattr(module, "__samestack_source_hash__", None) != files[name]:
                        raise RuntimeError("Imported same-stack source changed or is untracked; restart the Colab runtime")
            checked.append((target, data))
        for target, data in checked:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Parallel children see the parent-verified bundle. Avoid truncating
            # identical modules while another child imports them.
            if not target.is_file() or target.read_bytes() != data:
                target.write_bytes(data)
    if str(destination) not in sys.path:
        sys.path.insert(0, str(destination))
    return source


def run_parallel(items, n):
    # Tensors peak at ~14 GiB per run, but the caching allocator keeps growing past 20 GiB unless capped,
    # so each child gets a hard 1/n share of the GPU (GPU_MEM_FRACTION) and n is sized to that peak.
    total = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()[0]) / 1024
    n = max(1, min(n, int(total // float(os.environ.get("GPU_GB_PER_RUN", "18")))))
    print(f"parallel runs: {n} ({total:.0f} GiB GPU)", flush=True)
    os.makedirs("/content/logs", exist_ok=True)
    pending, live, failed, ticks = list(items), [], [], 0
    while pending or live:
        while pending and len(live) < n:
            item = pending.pop(0)
            log = open(f"/content/logs/{item.replace(':', '_s')}.log", "w")
            env = {**os.environ, "RUNS": item, "PARALLEL": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "GPU_MEM_FRACTION": f"{0.97 / n:.3f}"}
            try:
                live.append((item, subprocess.Popen([sys.executable, SELF], env=env, stdout=log, stderr=subprocess.STDOUT)))
            finally:
                log.close()
        time.sleep(20)
        ticks += 1
        if ticks % 30 == 0:  # every ~10 min: GPU memory and utilisation, to size PARALLEL
            q = "--query-gpu=memory.used,memory.total,utilization.gpu"
            print("GPU", subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                  f"live={len(live)} pending={len(pending)}", flush=True)
        for item, proc in [x for x in live if x[1].poll() is not None]:
            live.remove((item, proc))
            tail = open(f"/content/logs/{item.replace(':', '_s')}.log").read().strip().splitlines()[-1:]
            print(f"{'FINISHED' if proc.returncode == 0 else 'FAILED'} {item}: {tail}", flush=True)
            if proc.returncode:
                failed.append(item)
    if failed:
        sys.exit(f"failed runs: {failed}")


def main():
    source = prepare_bundle()
    from platform_hybrid.experiments.samestack_contract import validate_completion
    from platform_hybrid.experiments.samestack_protocol import PROTOCOL_ID
    runs = parse_runs(os.environ["RUNS"])
    smoke = os.environ.get("SMOKE") == "1"
    ensure_packages()
    parallel = int(os.environ.get("PARALLEL", "1"))
    if parallel < 1:
        raise ValueError("PARALLEL must be positive")
    if parallel > 1 and len(runs) > 1:
        return run_parallel([f"{arm}:{seed}" for arm, seed in runs], parallel)
    if os.environ.get("GPU_MEM_FRACTION"):
        fraction = float(os.environ["GPU_MEM_FRACTION"])
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("GPU_MEM_FRACTION must be in (0, 1]")
        import torch
        torch.cuda.set_per_process_memory_fraction(fraction)
    module = load_core()
    for name, imported in list(sys.modules.items()):
        if name.startswith("platform_hybrid.experiments.samestack_"):
            origin = getattr(imported, "__file__", None)
            if origin:
                try:
                    relative = str(Path(origin).resolve().relative_to(Path("/content").resolve()))
                except ValueError:
                    continue
                if relative in source["source_files"]:
                    imported.__samestack_source_hash__ = source["source_files"][relative]
    sub = PROTOCOL_ID + ("_smoke" if smoke else "")
    for arm, seed in runs:
        started = time.monotonic()
        record = module.run_arm_local(arm, seed, smoke=smoke, source_provenance=source)
        result_path = Path(RESULTS) / sub / f"{arm}_s{seed}.json"
        validate_completion(result_path, expected={"environment": "colab", "protocol_id": PROTOCOL_ID})
        print(f"PRESERVED {arm} s{seed} post={record['heldout_post_acc']:.4f} "
              f"pre={record['heldout_pre_acc']:.4f} wall={time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
