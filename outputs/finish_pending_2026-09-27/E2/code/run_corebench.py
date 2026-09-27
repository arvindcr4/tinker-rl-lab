#!/usr/bin/env python3
"""E2 CORE-Bench HARD (45 tasks) with the trained actor, GCP per-task VMs, native grader.

Per task: create VM (native shapes: E2as_v5 -> e2-highmem-2; NC4as_T4_v3 -> n1-highmem-4 + T4),
Ubuntu 20.04 (native OS), max-run-duration auto-delete as a cleanup backstop -> native capsule
download + codeocean_hard env setup (prep_task.py) -> agent loop (actor on the Mac, bash tool
executed on the VM over ssh as `crab`, native 8100 s budget) -> fetch report.json exactly as native
__find_report_path -> delete VM -> native eval_result_json / score_results.

Usage: run_corebench.py run [--only cid,...] [--workers 8] | grade | cleanup
"""
from __future__ import annotations

import argparse, base64, json, os, re, shlex, subprocess, sys, threading, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2 = HERE.parent
RAW = E2 / "raw"
REPO = E2.parents[2]
HARNESS = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/native-harness"
DATASET = E2 / "dataset/core_test.json"
CATALOG = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/audit_v11/prepared45_catalog.json"
PROJECT, ZONES = "electric-armor-388216", ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"]
SSH_DIR = Path(os.environ.get("E2_SSH_DIR", "/private/tmp/claude-501/e2_corebench_ssh"))
KEY = SSH_DIR / "id_ed25519"
# role-less SA: needed only so the Ubuntu Pro image can auto-attach (esm repos; without it libc6-dev/gcc are uninstallable)
SA = f"e2-corebench-vm@{PROJECT}.iam.gserviceaccount.com"
MODEL = "pavlov-public-portfolio-bf16"
BASE_URL = os.environ.get("TRAINED_ACTOR_BASE_URL", "https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run")
AGENT_TIMEOUT = 8100          # native run_agent_on_vm timeout
MAX_TURNS = 150
MAX_TOKENS = 3072
CTX_BUDGET_TOKENS = 26000     # server max-model-len 32768 minus output
MODEL_SEM = threading.Semaphore(int(os.environ.get("E2_MODEL_CONC", "4")))  # lane rule: <=4 in-flight requests total across E2 processes
LOCK = threading.Lock()
STOP = threading.Event()
COST_CAP_USD = 17.0           # hard lane cap is $20; stop admitting new tasks above this estimate
# Conservative hourly list prices (us-central1). spot/std.
PRICE = {("e2-highmem-2", "SPOT"): 0.045, ("e2-highmem-2", "STANDARD"): 0.091,
         ("n1-highmem-4", "SPOT"): 0.30, ("n1-highmem-4", "STANDARD"): 0.59}
DISK_HOUR = 80 * 0.10 / 730   # pd-balanced 80 GB
PRO_HOUR = 0.02               # generous Ubuntu Pro premium allowance

SYSTEM = """You are an autonomous research-reproduction agent operating a Linux VM (Ubuntu 20.04, user `crab` with passwordless sudo, internet access{gpu}).
You act only through tools:
- bash(command, timeout): runs a bash script on the VM. The working directory and exported environment variables persist between calls (e.g. after `cd` or `conda activate`). stdin is closed, so use non-interactive flags (apt-get -y, pip -q, conda -y). Output is truncated to the first 3000 and last 5000 characters. timeout is in seconds (default 900, max 7200); background long jobs with nohup and poll them if needed.
- query_image(path, question): ask a vision model a question about an image file on the VM (png/jpg/gif/webp).
- finish(summary): call once report.json has been written.
Your total time budget is {budget} seconds of wall clock. Work efficiently: read the README/code, install what is needed, run the code, read the outputs, write report.json (a JSON object whose keys are exactly the requested questions) in /home/crab/environment, then call finish."""


def log(*a):
    with LOCK:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, timeout=600, input=None, check=False):
    r = subprocess.run(cmd, input=input, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{cmd[:4]} rc={r.returncode}: {r.stderr[-800:]}")
    return r


def ledger(event: dict):
    event["t"] = time.time()
    with LOCK:
        with open(RAW / "vm_ledger.jsonl", "a") as fh:
            fh.write(json.dumps(event) + "\n")


def cost_estimate() -> float:
    # all lane ledgers (aborted runs + manual test VMs) count toward the lane cap
    live, total = {}, 0.0
    for lf in sorted(RAW.glob("**/*vm_ledger*.jsonl")):
        for line in open(lf):
            e = json.loads(line)
            if e["ev"] == "create":
                live[e["name"]] = e
            elif e["ev"] in ("deleted", "gone") and e["name"] in live:
                c = live.pop(e["name"])
                total += (e["t"] - c["t"]) / 3600 * (PRICE[(c["machine"], c["prov"])] + DISK_HOUR + PRO_HOUR)
    now = time.time()
    for c in live.values():
        total += (now - c["t"]) / 3600 * (PRICE[(c["machine"], c["prov"])] + DISK_HOUR + PRO_HOUR)
    return total


# ---------------------------------------------------------------- VM lifecycle
class Preempted(Exception):
    pass


def create_vm(name, gpu, prov):
    machine = "n1-highmem-4" if gpu else "e2-highmem-2"
    pub = (SSH_DIR / "id_ed25519.pub").read_text().strip()
    last = ""
    for zone in ZONES:
        cmd = ["gcloud", "compute", "instances", "create", name, "--project", PROJECT, "--zone", zone,
               "--machine-type", machine, "--image-family", "ubuntu-pro-2004-lts", "--image-project", "ubuntu-os-pro-cloud",
               "--boot-disk-size", "80GB", "--boot-disk-type", "pd-balanced", "--provisioning-model", prov,
               "--instance-termination-action", "DELETE", "--max-run-duration", "14400s",
               "--metadata", f"ssh-keys=crab:{pub}", "--labels", "lane=e2,bench=corebench-hard",
               "--service-account", SA, "--no-scopes", "--format", "json", "--quiet"]
        if gpu:
            cmd += ["--accelerator", "type=nvidia-tesla-t4,count=1", "--maintenance-policy", "TERMINATE"]
        ledger({"ev": "create_intent", "name": name, "zone": zone, "machine": machine, "prov": prov})
        r = sh(cmd, timeout=300)
        if r.returncode == 0:
            ip = json.loads(r.stdout)[0]["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
            ledger({"ev": "create", "name": name, "zone": zone, "machine": machine, "prov": prov, "ip": ip})
            return zone, ip, machine
        last = r.stderr[-600:]
        ledger({"ev": "create_failed", "name": name, "zone": zone, "err": last})
        if not re.search(r"ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|does not have enough resources|not available in zone", last):
            break
    raise RuntimeError("create failed: " + last)


def delete_vm(name, zone):
    for _ in range(3):
        r = sh(["gcloud", "compute", "instances", "delete", name, "--project", PROJECT, "--zone", zone, "--quiet"], timeout=600)
        if r.returncode == 0 or "was not found" in r.stderr:
            ledger({"ev": "deleted", "name": name, "zone": zone})
            return True
        time.sleep(10)
    ledger({"ev": "delete_failed", "name": name, "zone": zone, "err": r.stderr[-400:]})
    return False


def vm_status(name, zone):
    r = sh(["gcloud", "compute", "instances", "describe", name, "--project", PROJECT, "--zone", zone,
            "--format", "value(status)"], timeout=120)
    return r.stdout.strip() if r.returncode == 0 else "GONE"


SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=8", "-o", "BatchMode=yes"]


class VM:
    def __init__(self, name, zone, ip):
        self.name, self.zone, self.ip = name, zone, ip

    def ssh(self, remote, input=None, timeout=900):
        try:
            return sh(["ssh", *SSH_OPTS, f"crab@{self.ip}", remote], input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess([], 124, "", "LOCAL_SSH_TIMEOUT")

    def check_alive(self):
        st = vm_status(self.name, self.zone)
        if st not in ("RUNNING", "STAGING", "PROVISIONING"):
            raise Preempted(f"{self.name} status {st}")

    def wait_ssh(self, limit=600):
        t0 = time.time()
        while time.time() - t0 < limit:
            if self.ssh("true", timeout=40).returncode == 0:
                return
            time.sleep(10)
        self.check_alive()
        raise RuntimeError("ssh never came up")


# ---------------------------------------------------------------- model
def chat(messages, tools=None, max_tokens=MAX_TOKENS):
    import urllib.request, urllib.error
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["TRAINED_ACTOR_API_KEY"]}
    last = None
    for attempt in range(8):  # transport retries only
        try:
            with MODEL_SEM:
                with urllib.request.urlopen(urllib.request.Request(BASE_URL + "/v1/chat/completions",
                                            data=json.dumps(body).encode(), headers=hdr), timeout=1800) as r:
                    return json.loads(r.read())
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")[:600]
            if e.code == 400:
                return {"error": txt, "code": 400}
            last = f"HTTP {e.code}: {txt}"
        except Exception as e:
            last = repr(e)[:300]
        time.sleep(min(120, 10 * 2 ** attempt))
    return {"error": last, "code": -1}


TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a bash script on the VM (cwd and exported env persist).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "seconds, default 900, max 7200"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "query_image", "description": "Ask a question about an image file on the VM.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "question": {"type": "string"}},
                    "required": ["path", "question"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the task after report.json is written.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def est_tokens(msgs):
    return sum(len(json.dumps(m)) for m in msgs) // 3


def fit_context(msgs):
    """Keep system + first user message; drop oldest assistant/tool groups until under budget."""
    head, tail = msgs[:2], msgs[2:]
    while tail and est_tokens(head + tail) > CTX_BUDGET_TOKENS:
        tail = tail[1:]
        while tail and tail[0]["role"] == "tool":
            tail = tail[1:]
    if len(tail) < len(msgs) - 2:
        note = {"role": "user", "content": "[Earlier turns were truncated to fit the context window. Re-inspect files if needed.]"}
        return head + [note] + tail
    return head + tail


RUNNER = """source ~/.agent_env 2>/dev/null
cd "$(cat ~/.agent_cwd 2>/dev/null || echo /home/crab/environment)" 2>/dev/null || cd /home/crab/environment
timeout -k 15 {t} bash -c 'source ~/.agent_env 2>/dev/null; cd "$(cat ~/.agent_cwd 2>/dev/null || echo /home/crab/environment)"; source /tmp/agent_cmd.sh; __rc=$?; pwd > ~/.agent_cwd; export -p > ~/.agent_env 2>/dev/null; exit $__rc' < /dev/null 2>&1
echo "[exit code: $?]"
"""


def truncate(s, head=3000, tail=5000):
    return s if len(s) <= head + tail else s[:head] + f"\n...[{len(s) - head - tail} chars truncated]...\n" + s[-tail:]


def run_bash(vm, command, timeout):
    up = vm.ssh("cat > /tmp/agent_cmd.sh", input=command, timeout=120)
    if up.returncode != 0:
        vm.check_alive()
        return f"[tool error: could not upload command: {up.stderr[-300:]}]"
    r = vm.ssh("bash -s", input=RUNNER.format(t=int(timeout)), timeout=timeout + 120)
    if r.returncode == 255 or r.returncode == 124 and "LOCAL_SSH_TIMEOUT" in r.stderr:
        vm.check_alive()
    return truncate((r.stdout or "") + (r.stderr or ""))


def query_image(vm, path, question):
    r = vm.ssh(f"cd \"$(cat ~/.agent_cwd 2>/dev/null || echo /home/crab/environment)\"; "
               f"s=$(stat -c %s {shlex.quote(path)}) && [ $s -lt 8000000 ] && base64 -w0 {shlex.quote(path)}", timeout=180)
    if r.returncode != 0 or not r.stdout:
        return f"[query_image error: cannot read {path} (missing or >8MB)] {r.stderr[-200:]}"
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}.get(ext)
    if not mime:
        return "[query_image error: unsupported format; convert to png first]"
    resp = chat([{"role": "user", "content": [{"type": "text", "text": question},
                 {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{r.stdout.strip()}"}}]}], max_tokens=1024)
    if "error" in resp:
        return f"[query_image error: {resp['error'][:300]}]"
    return resp["choices"][0]["message"].get("content") or ""


def agent_loop(vm, task_txt, gpu, out: Path):
    t0 = time.time()
    msgs = [{"role": "system", "content": SYSTEM.format(gpu=", one NVIDIA T4 GPU" if gpu else "", budget=AGENT_TIMEOUT)},
            {"role": "user", "content": task_txt + "\n\nYou start in /home/crab/environment."}]
    trace = open(out / "trace.jsonl", "w")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    no_tool, turns, end = 0, 0, "max_turns"
    for turns in range(1, MAX_TURNS + 1):
        remaining = AGENT_TIMEOUT - (time.time() - t0)
        if remaining < 30:
            end = "timeout"
            break
        resp = chat(fit_context(msgs), TOOLS)
        if "error" in resp:
            trace.write(json.dumps({"turn": turns, "model_error": resp}) + "\n"); trace.flush()
            if resp.get("code") == 400 and "context" in resp["error"].lower():
                msgs = msgs[:2] + msgs[2:][len(msgs[2:]) // 2:]
                while len(msgs) > 2 and msgs[2]["role"] == "tool":
                    msgs.pop(2)
                continue
            end = "model_error"
            break
        usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += resp.get("usage", {}).get(k, 0)
        m = resp["choices"][0]["message"]
        am = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            am["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in m["tool_calls"]]
        msgs.append(am)
        trace.write(json.dumps({"turn": turns, "elapsed": round(time.time() - t0, 1), "assistant": am,
                                "finish_reason": resp["choices"][0].get("finish_reason")}) + "\n"); trace.flush()
        if not m.get("tool_calls"):
            no_tool += 1
            if no_tool >= 3:
                end = "no_tool_calls"
                break
            msgs.append({"role": "user", "content": "Continue using the tools. Call finish once report.json is written."})
            continue
        no_tool = 0
        done = False
        for c in am["tool_calls"]:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except Exception:
                args = None
            remaining = AGENT_TIMEOUT - (time.time() - t0)
            if args is None:
                obs = "[tool error: arguments were not valid JSON]"
            elif name == "bash":
                to = max(10, min(int(args.get("timeout") or 900), 7200, int(remaining) - 10))
                obs = run_bash(vm, str(args.get("command", "")), to)
            elif name == "query_image":
                obs = query_image(vm, str(args.get("path", "")), str(args.get("question", "")))
            elif name == "finish":
                obs, done = "finished", True
            else:
                obs = f"[unknown tool {name}]"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
            trace.write(json.dumps({"turn": turns, "tool": name, "args": args, "obs": obs}) + "\n"); trace.flush()
        if done:
            end = "finish"
            break
    trace.close()
    return {"end": end, "turns": turns, "agent_seconds": round(time.time() - t0, 1), **usage}


FETCH = r"""import json, os
env = "/home/crab/environment"
rp = None
for root, _, files in os.walk(env):
    for f in files:
        if f == "report.json":
            rp = os.path.join(root, f); break
    if rp: break
try:
    rep = json.load(open(rp)) if rp else {}
except Exception as e:
    rep = {}
cap = [d for d in os.listdir(env) if d.startswith("capsule-")]
res = [f for _, _, files in os.walk(os.path.join(env, cap[0], "results")) for f in files] if cap else []
raw = open(rp).read()[:200000] if rp else None
print(json.dumps({"report_path": rp, "report": rep, "report_raw": raw, "result_files": res}))
"""


# ---------------------------------------------------------------- per task
def run_task(task, gpu, prompt_template, attempt_prov="SPOT"):
    cid = task["capsule_id"]
    out = RAW / cid
    out.mkdir(parents=True, exist_ok=True)
    name = f"e2cb-{cid.split('-')[1]}-{int(time.time()) % 100000}"
    meta = {"capsule_id": cid, "vm": name, "gpu_catalog": gpu, "provisioning": attempt_prov, "started": time.time()}
    zone = None
    try:
        zone, ip, machine = create_vm(name, gpu, attempt_prov)
        meta.update(zone=zone, machine=machine)
        vm = VM(name, zone, ip)
        vm.wait_ssh()
        r = vm.ssh("cloud-init status --wait >/dev/null 2>&1; for i in 1 2 3 4 5 6; do "
                   "pro status --format json | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)[\"attached\"] else 1)' && break; "
                   "sudo pro auto-attach >/dev/null 2>&1; sleep 10; done; pro status --format json | head -c 300", timeout=900)
        meta["pro_attach"] = r.stdout[-300:]
        if gpu:
            # prebuilt signed modules for the running gcp kernel (ubuntu-drivers left no module; dkms needs gcc)
            r = vm.ssh("sudo DEBIAN_FRONTEND=noninteractive apt-get update -q >/dev/null; "
                       "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q linux-modules-nvidia-580-server-$(uname -r) "
                       "nvidia-headless-no-dkms-580-server nvidia-utils-580-server >/tmp/drv.log 2>&1; "
                       "sudo modprobe nvidia; for i in 1 2 3 4 5 6; do nvidia-smi -L && break; sleep 10; done", timeout=1500)
            meta["gpu_setup"] = (r.stdout + r.stderr)[-600:]
            if "Tesla T4" not in r.stdout:
                vm.ssh("sudo reboot", timeout=30)
                time.sleep(60)
                vm.wait_ssh()
                r = vm.ssh("nvidia-smi -L", timeout=120)
                meta["gpu_setup_after_reboot"] = (r.stdout + r.stderr)[-400:]
        task_in = {"capsule_id": cid, "task_prompt": task["task_prompt"],
                   "json_fields_str": str(task["results"][0].keys()), "prompt_template": prompt_template}
        vm.ssh("sudo tee /root/task_in.json >/dev/null", input=json.dumps(task_in), timeout=60)
        vm.ssh("sudo tee /root/prep_task.py >/dev/null", input=(HERE / "prep_task.py").read_text(), timeout=60)
        r = vm.ssh("sudo python3 /root/prep_task.py && sudo cat /root/prep_out.json", timeout=1800)
        if "PREP_OK" not in r.stdout:
            raise RuntimeError("prep failed: " + (r.stdout + r.stderr)[-800:])
        prep = json.loads(r.stdout.strip().splitlines()[-1])
        meta["prep"] = {k: prep[k] for k in ("uses_gpu", "registry_link", "gt_result_files")}
        (out / "task.txt").write_text(prep["task_txt"])
        meta["agent"] = agent_loop(vm, prep["task_txt"], gpu, out)
        vm.check_alive()
        r = vm.ssh("python3 -", input=FETCH, timeout=300)
        fetched = json.loads(r.stdout.strip().splitlines()[-1])
        (out / "fetched.json").write_text(json.dumps(fetched, indent=1))
        meta["status"] = "completed"
    except Preempted as e:
        meta["status"] = "preempted"; meta["error"] = str(e)
    except Exception as e:
        meta["status"] = "error"; meta["error"] = traceback.format_exc()[-1500:]
        if zone:
            try:
                if vm_status(name, zone) in ("GONE", "TERMINATED", "STOPPING"):
                    meta["status"] = "preempted"
            except Exception:
                pass
    finally:
        if zone:
            delete_vm(name, zone)
        meta["finished"] = time.time()
        (out / f"meta_{attempt_prov}.json").write_text(json.dumps(meta, indent=1))
    return meta


def cmd_run(args):
    import queue
    RAW.mkdir(exist_ok=True)
    tasks = json.load(open(DATASET))
    gpu_flags = {r["capsule_id"]: bool(r["native_uses_gpu"]) for r in json.load(open(CATALOG))["rows"]}
    template = json.load(open(HARNESS / "benchmark/benchmark_prompts.json"))["codeocean_hard"]
    only = set(args.only.split(",")) if args.only else None
    q = queue.Queue()
    # GPU tasks first (longest, scarcest)
    for t in sorted(tasks, key=lambda t: not gpu_flags[t["capsule_id"]]):
        cid = t["capsule_id"]
        if only and cid not in only:
            continue
        if (RAW / cid / "fetched.json").exists() and not args.force:
            continue
        q.put(t)

    def worker():
        while not q.empty() and not STOP.is_set():
            try:
                t = q.get_nowait()
            except Exception:
                return
            if cost_estimate() > COST_CAP_USD:
                log("COST CAP reached; not admitting", t["capsule_id"]); STOP.set(); return
            cid = t["capsule_id"]
            log("start", cid, "gpu" if gpu_flags[cid] else "cpu")
            m = run_task(t, gpu_flags[cid], template, "SPOT")
            if m["status"] == "preempted" and not STOP.is_set():
                log("preempted -> one STANDARD rerun", cid)
                m = run_task(t, gpu_flags[cid], template, "STANDARD")
            log("done", cid, m["status"], m.get("agent", {}).get("end"), f"cost~${cost_estimate():.2f}")

    ths = [threading.Thread(target=worker) for _ in range(args.workers)]
    for th in ths:
        th.start(); time.sleep(20)
    for th in ths:
        th.join()
    log("all workers done; cost estimate", round(cost_estimate(), 2))


def cmd_grade(args):
    sys.path.insert(0, str(HARNESS))
    from benchmark.evaluations import eval_result_json, score_results
    tasks = json.load(open(DATASET))
    results = {"capsule_results": []}
    per = []
    for t in tasks:
        cid = t["capsule_id"]
        f = RAW / cid / "fetched.json"
        fetched = json.load(open(f)) if f.exists() else {"report": {}, "result_files": []}
        metas = sorted((RAW / cid).glob("meta_*.json")) if (RAW / cid).exists() else []
        gt_files = None
        for mp in metas:
            mm = json.load(open(mp))
            gt_files = (mm.get("prep") or {}).get("gt_result_files", gt_files)
        report = fetched.get("report") if isinstance(fetched.get("report"), dict) else {}
        ev = eval_result_json(t["results"], dict(report))
        rp = fetched.get("result_files", [])
        cr = {"field": t["field"], "language": t["language"], "capsule_title": t["capsule_title"], "capsule_id": cid,
              "result_report": report, "result_paths": rp,
              "result_paths_success": (all(g in rp or g == "output" for g in gt_files) if gt_files is not None else None)}
        cr.update(ev)
        results["capsule_results"].append(cr)
        ok = ev["correct_written_answers"] == ev["total_written_questions"] and ev["correct_vision_answers"] == ev["total_vision_questions"]
        per.append({"capsule_id": cid, "correct": bool(ok), "has_report": bool(report), "ran": f.exists(), **ev})
    rf = E2 / "native_results_codeocean_hard.json"
    json.dump(results, open(rf, "w"), indent=4)
    score_results(str(rf), verbose=True)
    json.dump(per, open(E2 / "per_task.json", "w"), indent=1)


def cmd_cleanup(args):
    r = sh(["gcloud", "compute", "instances", "list", "--project", PROJECT, "--filter", "labels.lane=e2",
            "--format", "value(name,zone)"])
    for line in r.stdout.split("\n"):
        if line.strip():
            n, z = line.split()[:2]
            print("deleting", n, z, delete_vm(n, z.rsplit("/", 1)[-1]))
    r = sh(["gcloud", "compute", "disks", "list", "--project", PROJECT, "--filter", "name~^e2cb-", "--format", "value(name,zone)"])
    print("leftover disks:", r.stdout.strip() or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade", "cleanup"])
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "grade": cmd_grade, "cleanup": cmd_cleanup}[a.cmd](a)
