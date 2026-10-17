"""Run same-stack GSM8K-CoT v2 arms on a Colab GPU with the unchanged Modal training code.

The Modal script's `run_arm` holds the whole experiment; this runner replaces the `modal`
import with inert stand-ins so that function runs as plain Python on the Colab VM, then
stamps each result with where it ran (Modal used A10G; Colab runs are disclosed as such).

On a Colab session (after uploading this file and the Modal script to /content):
  colab exec -s NAME -f colab_run_arms.py --timeout 30000 --env RUNS=ppo:4096,ppo_8x8:42 [--env SMOKE=1]
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


def main():
    ensure_packages()
    install_modal_stub()
    spec = importlib.util.spec_from_file_location("samestack_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS_DIR = RESULTS  # run_arm reads this module global for every output path
    smoke = os.environ.get("SMOKE") == "1"
    sub = "samestack_gsm8k_v2s_smoke" if smoke else "samestack_gsm8k_v2s"
    env = environment()
    print(json.dumps(env), flush=True)
    for item in os.environ["RUNS"].split(","):
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
