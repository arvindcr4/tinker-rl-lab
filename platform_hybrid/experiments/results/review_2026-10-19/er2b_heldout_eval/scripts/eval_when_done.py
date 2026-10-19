"""Wait for each E-R2b rerun to finish training, then score its final sampler with tinker_passk.py.

Usage: python eval_when_done.py <python-with-tinker>
"""

import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ["er2b_rerun_g16_s123", "er2b_rerun_g16_s456", "er2b_rerun_g2_s123", "er2b_rerun_g2_s456"]
py = sys.argv[1]
pending = set(ARMS)
while pending:
    for tag in sorted(pending):
        d = json.loads((ROOT / "train" / f"{tag}.json").read_text())
        if d.get("status") not in ("completed", "failed", "error"):
            continue
        pending.discard(tag)
        if d.get("status") != "completed" or not d.get("checkpoint"):
            print(f"{tag}: training ended {d.get('status')}, no eval", flush=True)
            continue
        print(f"{tag}: evaluating {d['checkpoint']}", flush=True)
        with open(ROOT / "logs" / f"{tag}.log", "w") as log:
            rc = subprocess.call(
                [py, str(HERE / "tinker_passk.py"), "--model", d["model"], "--sampler-path", d["checkpoint"],
                 "--tag", f"tinker_passk_{tag}_gsm8k_p200_n32_s42", "--out", str(ROOT / "raw" / f"{tag}.json")],
                stdout=log, stderr=subprocess.STDOUT,
            )
        print(f"{tag}: eval rc={rc}", flush=True)
    if pending:
        time.sleep(120)
print("ALL DONE", flush=True)
