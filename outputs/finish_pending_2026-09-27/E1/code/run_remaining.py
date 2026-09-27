"""E1 finish-0927: the 174 never-attempted SWE-bench Multilingual tasks (300 - 110 graded/errored in
waves 01-09 - 16 wave10), same actor/protocol as wave10.

Per batch of <=16 tasks, one Modal dockerd sandbox (2026-09-12 runtime shape):
  pull instance image (record digest) -> base-commit source context (e1_runtime.collect_sources,
  validated byte-identical on 3 recorded tasks) -> generation via zvf-program/e1_wave10/driver.py
  generate_one (unchanged receipt schema; max_tokens capped to fit 32768 context) -> native
  `/native/.venv/bin/swebench eval` in the same sandbox -> reports/logs fetched -> sandbox terminated.
No task is retried after a completed generation. Transport failures (no response) are retried <=3x;
a task that still has no response, or whose sources/image cannot be obtained, is recorded and scored
as a failure (empty prediction) — never dropped from the denominator.
Resumable: a batch whose report JSON exists locally is skipped.
"""
import argparse
import copy
import json
import os
import sys
import tarfile
import threading
import time
import traceback
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
E1 = REPO / "outputs/finish_pending_2026-09-27/E1"
OUT = E1 / "remaining"
FIN = REPO / "outputs/PES_Phase2_Review_2026-09-12/finish/e1_completion"
SETUP = REPO / "outputs/public_portfolio_2026-09-05/swe_multilingual_setup/prepared"
sys.path.insert(0, str(REPO / "zvf-program/e1_wave10"))
sys.path.insert(0, str(REPO / "zvf-program/flagship"))
sys.path.insert(0, str(Path(__file__).parent))
import driver as D  # noqa: E402
import public_swe_multilingual_native as psml  # noqa: E402
import e1_runtime as R  # noqa: E402

BASE = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/")
KEY = os.environ["TRAINED_ACTOR_API_KEY"]
MAX_MODEL_LEN = 32768
GEN_SEM = threading.Semaphore(3)  # wave10 driver uses 1 concurrent request -> lane total <= 4
LOCK = threading.Lock()
LOG = open(E1 / "logs/remaining.log", "a")


def log(msg):
    with LOCK:
        LOG.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        LOG.flush()


def task_list():
    rows = [json.loads(l) for l in open(SETUP / "native_dataset.jsonl")]
    import glob
    seen = set()
    for r in glob.glob(str(FIN / "**/reports/*.json"), recursive=True):
        seen.update(json.load(open(r)).get("submitted_ids", []))
    w10 = set(D.SEALED_TASK_IDS)
    return [r for r in rows if r["instance_id"] not in seen and r["instance_id"] not in w10]


def prompt_tokens(req):
    body = {"model": req["model"], "messages": req["messages"],
            "chat_template_kwargs": req["chat_template_kwargs"], "add_generation_prompt": True}
    r = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["count"]


def generate(iid, source, identity, args, root):
    done = root / iid / "generation.json"
    if done.exists():  # completed earlier in this lane (e.g. native phase crashed): reuse, never resample
        return json.load(open(done))
    actor = {k: source["actor_task"][k] for k in psml.ACTOR_FIELDS}
    req = psml.build_actor_request(actor, source["files"], identity["served_model_id"],
                                   max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed)
    with GEN_SEM:
        n = prompt_tokens(req)
        if n >= MAX_MODEL_LEN - 1:  # added on 2nd relaunch: prompt alone fills context; no retry loop
            (root / iid / "context_overflow.json").write_text(json.dumps({"prompt_tokens": n, "max_model_len": MAX_MODEL_LEN}))
            log(f"{iid} CONTEXT_OVERFLOW prompt_tokens={n}")
            return {"status": "CONTEXT_OVERFLOW", "patch": "", "error": f"prompt_tokens={n}"}
        a = copy.copy(args)
        a.max_tokens = max(1, min(args.max_tokens, MAX_MODEL_LEN - n))
        (root / iid / "max_tokens_cap.json").write_text(json.dumps({"prompt_tokens": n, "max_tokens": a.max_tokens}))
        err = None
        for attempt in range(3):
            try:
                return D.generate_one(iid, source, identity, a, root, log)
            except RuntimeError as exc:
                err = str(exc)
                if (root / iid / "generation_http_response.json").exists():
                    raise  # a response was received: never replay
                log(f"{iid} transport/provider error attempt {attempt}: {err[:300]}")
                time.sleep(30 * (attempt + 1))
        return {"status": "PROVIDER_ERROR", "patch": "", "error": err}


def run_batch(k, rows, identity, args):
    tag = f"e1multilingual0927r{k:02d}"
    bdir = OUT / tag
    report = bdir / f"reports/{identity['served_model_id']}.{tag}.json"
    if report.exists():
        log(f"{tag} already done")
        return
    bdir.mkdir(parents=True, exist_ok=True)
    root = bdir / "attempts"
    f = R.pro_flagship()
    sb = R.open_sandbox(timeout=10800, log=log)
    status = {"tag": tag, "sandbox": sb.object_id, "started_at": time.time(), "tasks": {}}
    try:
        preds, runtime_rows = [], []
        for row in rows:
            iid = row["instance_id"]
            st = status["tasks"][iid] = {}
            patch = ""
            try:
                digest, inspect = R.pull(sb, row["image"])
                (root / iid).mkdir(parents=True, exist_ok=True)
                (root / iid / "image_inspect.json").write_text(json.dumps(inspect, indent=1))
                st["image"] = digest
                sc = root / iid / "source_context.json"
                if sc.exists():
                    source = json.load(open(sc))
                else:
                    src = R.collect_sources(sb, row, f)
                    source = {"actor_task": ACTOR_ROWS[iid], "base_commit": row["base_commit"],
                              "files": src["files"], "repo": row["repo"]}
                    sc.write_text(json.dumps(source, sort_keys=True))
                    (root / iid / "source_receipt.json").write_text(json.dumps({**src["receipt"], "image": digest}))
                rec = generate(iid, source, identity, args, root)
                st["generation"] = rec["status"]
                st["finish_reason"] = rec.get("finish_reason")
                patch = rec["patch"]
                runtime_rows.append({**row, "image": digest})
            except Exception as exc:  # recorded failure, stays in denominator
                st["error"] = f"{type(exc).__name__}: {exc}"[:1000]
                log(f"{tag} {iid} FAILED_PRE_EVAL {st['error'][:200]}")
                runtime_rows.append(row)
            preds.append({"instance_id": iid, "model_name_or_path": identity["served_model_id"],
                          "model_patch": patch})
            log(f"{tag} {iid} {st.get('generation', 'ERR')}")
        (bdir / f"{tag}_status.json").write_text(json.dumps(status, indent=2))
        data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                       for r in runtime_rows).encode()
        pj = "".join(json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                     for p in preds).encode()
        (bdir / f"{tag}_native_dataset.jsonl").write_bytes(data)
        (bdir / f"{tag}_predictions.jsonl").write_bytes(pj)
        receipt = R.run_native(sb, tag, data, pj, bdir, log=log)
        with tarfile.open(receipt["receipts_tgz"]) as t:
            t.extractall(bdir, filter="data")
    except Exception:
        log(f"{tag} BATCH_ERROR {traceback.format_exc()[-1500:]}")
        raise
    finally:
        sb.terminate()
        status["finished_at"] = time.time()
        (bdir / f"{tag}_status.json").write_text(json.dumps(status, indent=2))
        log(f"{tag} sandbox terminated")


ACTOR_ROWS = {json.loads(l)["instance_id"]: json.loads(l) for l in open(SETUP / "actor_tasks.jsonl")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="all")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--size", type=int, default=16)
    o = ap.parse_args()
    tasks = task_list()
    assert len(tasks) == 174, len(tasks)
    ids = sorted(t["instance_id"] for t in tasks)
    D.TASK_INVENTORY_SHA256 = psml.object_hash(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps({"task_inventory_sha256": D.TASK_INVENTORY_SHA256,
                                                    "count": len(ids), "ids": ids}, indent=1))
    identity = psml.read(FIN / "model_identity.json")
    psml.validate_identity(identity)
    args = types.SimpleNamespace(endpoint=BASE + "/v1/chat/completions", api_key=KEY, max_tokens=8192,
                                 temperature=0.0, seed=809, request_timeout=900.0,
                                 wandb_run_id=json.load(open(E1 / "wandb_run.json"))["id"],
                                 deployment=E1 / "actor_deployment_0927.json")
    batches = [tasks[i:i + o.size] for i in range(0, len(tasks), o.size)]
    sel = range(len(batches)) if o.batches == "all" else [int(x) for x in o.batches.split(",")]
    log(f"START {len(batches)} batches, selected {list(sel)}")
    with ThreadPoolExecutor(o.parallel) as ex:
        futs = [ex.submit(run_batch, k, batches[k], identity, args) for k in sel]
        for fu in futs:
            try:
                fu.result()
            except Exception as exc:
                log(f"batch failed: {exc}")
    log("DONE")
