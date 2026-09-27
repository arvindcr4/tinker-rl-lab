#!/usr/bin/env python3
"""Runs ON the task VM as root. Replicates native CodeOceanTask download +
CodeOceanBenchmark.__setup_task_environment for codeocean_hard (core-bench
benchmark/benchmark.py), then applies the native Azure-VM run hygiene
(codeocean.com blocked in /etc/hosts). Input: /root/task_in.json with
capsule_id, task_prompt, json_fields_str (str(results[0].keys())), prompt template.
No ground-truth answers ever reach the VM.
"""
import json, os, re, shutil, tarfile, urllib.request, time

t = json.load(open("/root/task_in.json"))
cid = t["capsule_id"]
dl = "/root/dl"
os.makedirs(dl, exist_ok=True)
url = f"https://corebench.cs.princeton.edu/capsules/{cid}.tar.gz"
tar_path = os.path.join(dl, f"{cid}.tar.gz")
for attempt in range(5):
    try:
        urllib.request.urlretrieve(url, tar_path)
        break
    except Exception as e:  # native: 5 attempts, exponential backoff
        if attempt == 4:
            raise
        time.sleep(2 ** attempt)
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=dl)
os.remove(tar_path)
src = os.path.join(dl, cid)

gt_result_files = [f for _, _, files in os.walk(os.path.join(src, "results")) for f in files]
repro = open(os.path.join(src, "REPRODUCING.md")).read()
uses_gpu = "gpu" in repro
m = re.search(r'`(registry\.codeocean\.com/published/[\w-]+:v\d+)`', repro)
registry_link = m.group(1) if m else None

home = "/home/crab"
env = os.path.join(home, "environment")
if os.path.exists(env):
    shutil.rmtree(env)
os.makedirs(env)
cap = os.path.join(env, cid)
shutil.copytree(src, cap)
# codeocean_hard: empty results, drop REPRODUCING.md, environment/, run scripts
shutil.rmtree(os.path.join(cap, "results"))
os.makedirs(os.path.join(cap, "results"))
os.remove(os.path.join(cap, "REPRODUCING.md"))
shutil.rmtree(os.path.join(cap, "environment"))
for rf in ("run.sh", "run"):
    p = os.path.join(cap, "code", rf)
    if os.path.exists(p):
        os.remove(p)
task_str = t["prompt_template"].replace("{task_prompt}", t["task_prompt"])
task_str = task_str.replace("{json_fields}", t["json_fields_str"])
task_str = task_str.replace("{registry_link}", str(registry_link))
open(os.path.join(env, "task.txt"), "w").write(task_str)
shutil.rmtree(dl)  # the un-stripped capsule (answers in results/) must not remain
with open("/etc/hosts", "a") as fh:
    fh.write("127.0.0.1 codeocean.com\n")
os.system(f"chown -R crab:crab {home}")
json.dump({"capsule_id": cid, "uses_gpu": uses_gpu, "registry_link": registry_link,
           "gt_result_files": gt_result_files, "task_txt": task_str},
          open("/root/prep_out.json", "w"))
print("PREP_OK", cid, "uses_gpu", uses_gpu)
