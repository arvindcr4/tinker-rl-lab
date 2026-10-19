"""Colab driver: train the four E-R2b arms sequentially (skipping arms whose adapter has a DONE
marker), then run the held-out eval for base + the four adapters (skipping finished evals).

  nohup python run_all.py /content/out > /content/run_all.log 2>&1 &
"""

import os, subprocess, sys

OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["g16_s123", "g16_s456", "g2_s123", "g2_s456"]

for arm in ARMS:
    if os.path.exists(f"{OUT}/adapters/er2b_trl_{arm}/DONE"):
        print("skip trained", arm, flush=True)
        continue
    print("train", arm, flush=True)
    with open(f"{OUT}/train_{arm}.log", "a") as log:
        rc = subprocess.run([sys.executable, f"{HERE}/er2b_trl_colab.py", arm, OUT], stdout=log, stderr=subprocess.STDOUT).returncode
    print("trained", arm, "rc", rc, flush=True)
    if rc:
        sys.exit(rc)

models = ["base=base"] + [f"er2b_trl_{a}={OUT}/adapters/er2b_trl_{a}" for a in ARMS]
with open(f"{OUT}/eval.log", "a") as log:
    rc = subprocess.run([sys.executable, f"{HERE}/er2b_vllm_passk.py", f"{OUT}/raw", *models], stdout=log, stderr=subprocess.STDOUT).returncode
print("eval rc", rc, flush=True)
print("ALL_DONE" if rc == 0 else "EVAL_FAILED", flush=True)
