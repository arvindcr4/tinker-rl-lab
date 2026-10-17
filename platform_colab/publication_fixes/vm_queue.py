"""Tiny job queue for a Colab VM: run shell commands with bounded concurrency.

python vm_queue.py jobs.json N   (jobs.json = [{"name": ..., "cmd": ...}, ...])
Writes <logdir>/<name>.log and <logdir>/status.json (name -> state, rc, start, end).
Jobs whose done-marker file (optional "done" key) exists are skipped.
"""
import json, os, subprocess, sys, time

jobs = json.load(open(sys.argv[1]))
N = int(sys.argv[2])
logdir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "logs")
os.makedirs(logdir, exist_ok=True)
status = {j["name"]: {"state": "queued"} for j in jobs}
pending = list(jobs)
running = {}


def dump():
    with open(os.path.join(logdir, "status.json.tmp"), "w") as f:
        json.dump(status, f, indent=1)
    os.replace(os.path.join(logdir, "status.json.tmp"), os.path.join(logdir, "status.json"))


while pending or running:
    while pending and len(running) < N:
        j = pending.pop(0)
        if j.get("done") and os.path.exists(j["done"]):
            status[j["name"]] = {"state": "skipped_done"}
            continue
        env = {**os.environ, "PF_CONCURRENCY": str(N), **j.get("env", {})}
        p = subprocess.Popen(j["cmd"], shell=True, stdout=open(os.path.join(logdir, j["name"] + ".log"), "w"),
                             stderr=subprocess.STDOUT, env=env)
        running[j["name"]] = p
        status[j["name"]] = {"state": "running", "start": time.time()}
        dump()
    for name, p in list(running.items()):
        if p.poll() is not None:
            status[name].update(state="ok" if p.returncode == 0 else "failed", rc=p.returncode, end=time.time())
            del running[name]
            dump()
    time.sleep(10)
status["_all_done"] = time.time()
dump()
