"""Run same-stack GSM8K-CoT v2 arms on a Colab GPU with the unchanged Modal training code.

The Modal script's `run_arm` holds the whole experiment; this runner replaces the `modal`
import with inert stand-ins so that function runs as plain Python on the Colab VM, then
stamps each result with where it ran (Modal used A10G; Colab runs are disclosed as such).

On a Colab session (after uploading this file and the Modal script to /content):
  colab exec -s NAME -f colab_run_arms.py --timeout 30000 --env RUNS=ppo:4096,ppo_8x8:42 [--env SMOKE=1] [--env PARALLEL=4]
PARALLEL>1 trains that many runs at once on the GPU, one process each (logs in /content/logs/).
Results: /content/results/samestack_gsm8k_v2s[_smoke]/<arm>_s<seed>.json and _texts.json.gz
"""

import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import types

SCRIPT = "/content/modal_samestack_gsm8k_cot_v2.py"
RESULTS = "/content/results"
SELF = "/content/colab_run_arms.py"  # uploaded copy; `colab exec -f` runs code in the kernel without __file__


def install_modal_stub():
    stub = types.ModuleType("modal")

    class _Chain:  # Image builder calls chain and do nothing.
        def __getattr__(self, _):
            return lambda *a, **k: self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, *a, **k):
            return lambda f: f

        def local_entrypoint(self, *a, **k):
            return lambda f: f

    class Volume:
        @staticmethod
        def from_name(*a, **k):
            return types.SimpleNamespace(commit=lambda: None, reload=lambda: None)

    stub.App, stub.Volume = App, Volume
    stub.Image = _Chain()
    stub.Secret = types.SimpleNamespace(from_name=lambda *a, **k: None)
    sys.modules["modal"] = stub


def ensure_packages():
    # Colab ships an old torchao that makes recent peft refuse to import; the experiment does not use it.
    if importlib.util.find_spec("torchao") is not None:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    need = [p for p in ("peft", "datasets", "accelerate") if importlib.util.find_spec(p) is None]
    if need:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])


def environment():
    import torch
    import transformers
    import peft
    import datasets
    return {
        "platform": "google-colab", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "peft": peft.__version__, "datasets": datasets.__version__,
        "runner": "platform_colab/samestack_v2/colab_run_arms.py",
        "note": "Same run_arm code as the Modal script; the original v2 runs used Modal A10G.",
    }


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
            live.append((item, subprocess.Popen([sys.executable, SELF], env=env, stdout=log, stderr=subprocess.STDOUT)))
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
    items = os.environ["RUNS"].split(",")
    ensure_packages()
    if int(os.environ.get("PARALLEL", "1")) > 1 and len(items) > 1:
        return run_parallel(items, int(os.environ["PARALLEL"]))
    if os.environ.get("GPU_MEM_FRACTION"):
        import torch
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRACTION"]))
    install_modal_stub()
    spec = importlib.util.spec_from_file_location("samestack_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS_DIR = RESULTS  # run_arm reads this module global for every output path
    smoke = os.environ.get("SMOKE") == "1"
    sub = "samestack_gsm8k_v2s_smoke" if smoke else "samestack_gsm8k_v2s"
    env = environment()
    print(json.dumps(env), flush=True)
    for item in items:
        arm, seed = item.split(":")
        t0 = time.time()
        r = mod.run_arm(arm, int(seed), smoke=smoke)
        path = f"{RESULTS}/{sub}/{arm}_s{seed}.json"
        r.setdefault("execution_environment", env)
        with open(path, "w") as f:
            json.dump(r, f)
        print(f"DONE {arm} s{seed} post={r['heldout_post_acc']:.4f} pre={r['heldout_pre_acc']:.4f} "
              f"wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
