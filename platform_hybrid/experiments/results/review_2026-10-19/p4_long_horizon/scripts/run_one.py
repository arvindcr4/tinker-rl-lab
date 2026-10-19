"""Run a *_lenstrat.py Modal driver's run_arm in-process on a Colab GPU (Modal workspace paused).

A stub `modal` module makes the decorators no-ops, so the training/eval code is the
driver's own, unchanged. W&B is disabled; the per-run JSON written by run_arm is the record.
Usage: python run_one.py <driver.py> <max_new> <out.json> <n_steps> <algo> <seed> [n_eval]
"""

import importlib.util
import json
import os
import sys
import types


class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return self


class _App(_Any):
    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, *a, **k):
        return lambda f: f


stub = types.ModuleType("modal")
stub.App = _App
stub.Image = _Any()
stub.Volume = _Any()
stub.Secret = _Any()
sys.modules["modal"] = stub
os.environ.setdefault("WANDB_MODE", "disabled")

driver, max_new, out, n_steps, algo, seed = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
spec = importlib.util.spec_from_file_location("driver", driver)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(out)), "per_run")
mod.N_STEPS = int(sys.argv[4])
assert algo in mod.ALGOS and seed in mod.SEEDS
if len(sys.argv) > 7:
    mod.N_EVAL = int(sys.argv[7])
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
print("START", {"algo": algo, "seed": seed, "n_steps": mod.N_STEPS,
                "max_new": max_new, "n_eval": mod.N_EVAL}, flush=True)
r = mod.run_arm(algo, seed, max_new)
import torch
import importlib.metadata as metadata
result = {"max_new": max_new, "n_steps": mod.N_STEPS, "model": mod.MODEL,
          "backend": "colab", "runs": [r], "smoke": mod.N_EVAL != 200,
          "gpu": torch.cuda.get_device_name(),
          "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
          "versions": {p: metadata.version(p) for p in
                       ("torch", "transformers", "peft", "datasets", "numpy")}}
with open(out, "w") as f:
    json.dump(result, f)
print("DONE", out, result["peak_memory_gib"], flush=True)
