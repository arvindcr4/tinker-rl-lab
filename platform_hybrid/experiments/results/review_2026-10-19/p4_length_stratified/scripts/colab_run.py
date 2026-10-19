"""Run a *_lenstrat.py Modal driver's run_arm in-process on a Colab GPU (Modal workspace paused).

A stub `modal` module makes the decorators no-ops, so the training/eval code is the
driver's own, unchanged. W&B is disabled; the per-run JSON written by run_arm is the record.
Usage: python colab_run.py <driver.py> <max_new> <out.json> [n_steps]
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

driver, max_new, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("driver", driver)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(out)), "per_run")
if len(sys.argv) > 4:
    mod.N_STEPS = int(sys.argv[4])

runs = []
for algo in mod.ALGOS:
    for seed in mod.SEEDS:
        r = mod.run_arm(algo, seed, max_new)
        runs.append(r)
        with open(out, "w") as f:
            json.dump(
                {"max_new": max_new, "n_steps": mod.N_STEPS, "model": mod.MODEL, "backend": "colab", "runs": runs}, f
            )
print("DONE", out, len(runs), flush=True)
