"""E5 small-scale driver: one fresh Modal Sandbox (official Archipelago environment Dockerfile)
per APEX-Agents task; agent + native grader run locally via the patched official example
(code/hf_task/main.py). Agent LLM = local Tinker shim (base Qwen3.6-35B-A3B).

usage: python run_e5.py --archipelago <dir> --shim-port 18742 --per-task-tokens 250000 TASK_ID...
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import modal

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
RAW = LANE / "raw"

ap = argparse.ArgumentParser()
ap.add_argument("--archipelago", required=True)
ap.add_argument("--shim-port", type=int, default=18742)
ap.add_argument("--per-task-tokens", type=int, default=250_000)
ap.add_argument("--cpu", type=float, default=4.0)
ap.add_argument("--memory", type=int, default=8192)
ap.add_argument("tasks", nargs="+")
a = ap.parse_args()
arch = Path(a.archipelago).resolve()


def ledger_used() -> int:
    p = RAW / "shim_ledger.jsonl"
    if not p.exists():
        return 0
    return sum(json.loads(l).get("prompt_tokens", 0) + json.loads(l).get("completion_tokens", 0)
               for l in p.read_text().splitlines())


REV = "92c86856cf1b11f9833a8a076b3a45a63afa3929"
TASKS = {t["task_id"]: t for t in json.loads(
    (LANE.parents[1] / "e5_apex_agents" / "hf_dataset" / "tasks_and_rubrics.json").read_text())}
from huggingface_hub import get_token  # noqa: E402

HF_TOKEN = get_token()  # gated dataset; passed to the sandbox as an ephemeral secret
# Runs INSIDE the sandbox: pull the pinned world zip from HF and populate via the env's own
# /data/populate endpoint (same tar layout as the official populate_subsystems). Avoids
# uploading 0.1-0.7 GB worlds over the local uplink.
POPULATE = r"""
import os, sys, tarfile, zipfile, subprocess, urllib.request, shutil, pathlib
rev, world = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/datasets/mercor/apex-agents/resolve/{rev}/world_files_zipped/{world}.zip"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]})
with urllib.request.urlopen(req) as r, open("/tmp/w.zip", "wb") as f:
    shutil.copyfileobj(r, f, 1 << 20)
zipfile.ZipFile("/tmp/w.zip").extractall("/tmp/w")
for sub in ("filesystem", ".apps_data"):
    d = pathlib.Path("/tmp/w") / sub
    entries = list(d.rglob("*")) if d.exists() else []
    if not entries:
        continue
    tp = f"/tmp/{sub.strip('.')}.tar.gz"
    with tarfile.open(tp, "w:gz") as t:
        t.dereference = True
        for e in entries:
            t.add(e, arcname=str(e.relative_to(d)), recursive=False)
    out = subprocess.run(["curl", "-sS", "-f", "-F", f"archive=@{tp};type=application/gzip",
                          f"http://localhost:8080/data/populate?subsystem={sub}"], capture_output=True, text=True)
    print(sub, len(entries), out.returncode, out.stdout[:300], out.stderr[:300], flush=True)
    if out.returncode:
        sys.exit(1)
"""

app = modal.App.lookup("e5-small-apex-env", create_if_missing=True)
image = modal.Image.from_dockerfile(str(arch / "environment" / "Dockerfile"), context_dir=str(arch))

for task_id in a.tasks:
    used = ledger_used()
    (RAW / "cap.json").write_text(json.dumps({"cap_total_tokens": used + a.per_task_tokens}))
    rec = {"task_id": task_id, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tokens_before": used}
    sb = None
    t0 = time.time()
    try:
        with modal.enable_output():
            sb = modal.Sandbox.create(app=app, image=image, timeout=2 * 3600, cpu=a.cpu,
                                      memory=a.memory, encrypted_ports=[8080],
                                      secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})])
            sb.tunnels(timeout=1800)
        rec["sandbox_id"] = sb.object_id
        rec["sandbox_ready_s"] = round(time.time() - t0, 1)
        for _ in range(120):  # wait for env server inside sandbox
            if sb.exec("curl", "-sf", "http://localhost:8080/health").wait() == 0:
                break
            time.sleep(2)
        t1 = time.time()
        p = sb.exec("python3", "-c", POPULATE, REV, TASKS[task_id]["world_id"], timeout=1800)
        p.wait()
        rec["populate_out"] = p.stdout.read()[-600:] + p.stderr.read()[-600:]
        rec["populate_rc"], rec["populate_s"] = p.returncode, round(time.time() - t1, 1)
        if p.returncode != 0:
            raise RuntimeError("in-sandbox populate failed")
        cred = sb.create_connect_token(port=8080)
        env = dict(os.environ)
        env.update({
            "ENV_URL": cred.url.rstrip("/"), "APEX_ENV_AUTH_TOKEN": cred.token,
            "APEX_SKIP_LOCAL_ENVIRONMENT": "1", "APEX_REMOTE_POPULATED": "1", "EXAMPLE_DIR": str(HERE / "hf_task"),
            "ARCHIPELAGO_DIR": str(arch),
            "E5_SHIM_BASE": f"http://127.0.0.1:{a.shim_port}/v1", "E5_SHIM_KEY": env["SHIM_KEY"],
        })
        log = RAW / "logs" / f"{task_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            r = subprocess.run([sys.executable, str(HERE / "hf_task" / "main.py"), task_id],
                               env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3 * 3600)
        rec["main_rc"] = r.returncode
        # keep raw lean: world snapshots are re-downloadable from the pinned HF revision
        out = RAW / "archipelago_output" / task_id
        for p in list(out.glob("world_*")) + list(out.glob("task_*.tar.gz")) + [out / "final_snapshot.tar.gz"]:
            p.unlink(missing_ok=True)
    except Exception as exc:  # infra error -> recorded, counted as failure downstream
        rec["error"] = repr(exc)[:500]
    finally:
        if sb is not None:
            sb.terminate()
        rec["sandbox_wall_s"] = round(time.time() - t0, 1)
        rec["tokens_after"] = ledger_used()
        rec["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (RAW / "sandbox_log.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
