"""Runs inside python:3.11-slim (--network none). Pass = exit 0, empty stderr, within 5 s (upstream run_code.sh).
Timing = median wall time of 3 runs for passing programs."""
import json
import os
import statistics
import subprocess
import time

TIMEOUT = 5
out = {}
for kind in ("canonical", "completion"):
    out[kind] = {}
    d = f"/programs/{kind}"
    for name in sorted(os.listdir(d)):
        runs, ok, err = [], True, ""
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                p = subprocess.run(["python", os.path.join(d, name)], capture_output=True, text=True, timeout=TIMEOUT, cwd="/tmp")
                dt = time.perf_counter() - t0
                if p.returncode != 0 or p.stderr.strip():
                    ok, err = False, (p.stderr or f"rc={p.returncode}")[-500:]
                    break
                runs.append(dt)
            except subprocess.TimeoutExpired:
                ok, err = False, "timeout"
                break
        out[kind][name[:-3]] = {"passed": ok, "runs_s": runs, "median_s": statistics.median(runs) if ok else None, "error": err}
print(json.dumps(out))
