"""Print last training step per job log on each Colab session (local helper)."""

import subprocess, sys

COLAB = "/Users/arvind/.local/bin/colab"
REMOTE = r"""
import glob, json, os, re, subprocess
for st in sorted(glob.glob("/content/pf/**/status.json", recursive=True)):
    d = json.load(open(st))
    for k, v in d.items():
        if k.startswith("_"): continue
        log = os.path.join(os.path.dirname(st), k + ".log")
        last = ""
        if os.path.exists(log):
            lines = [l for l in open(log, errors="ignore").read().splitlines() if re.search(r"\] step|DONE|pre=|Error|acc_", l)]
            last = lines[-1][:150] if lines else ""
        print(f"{k:16s} {v.get('state'):12s} {last}")
print(subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
"""
for s in sys.argv[1:]:
    r = subprocess.run(
        [COLAB, "exec", "-s", s, "--timeout", "120"],
        input=REMOTE,
        capture_output=True,
        text=True,
        timeout=200,
    )
    print("==", s)
    print(r.stdout[-4000:], r.stderr[-500:])
