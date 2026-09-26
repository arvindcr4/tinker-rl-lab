"""E1 small-scale base-model arm: SWE-bench Pro, agentless single-shot diff.

Stages (run in order):
  select  : any python   -> raw/selection.json, raw/dataset_subset.jsonl, raw/image_manifest_subset.json
  sources : needs modal  -> raw/tasks/<id>/source_context.json (base-commit retrieval in the native image)
  sample  : tinker python-> raw/tasks/<id>/generation.json, raw/candidates.json
  eval    : shell out to the pinned evaluator wrapper (Modal sandboxes, block_network)
  score   : any python   -> raw/score.json

Localization + retrieval + prompt + diff extraction reuse the functions of
zvf-program/flagship/modal_e1_swe_bench_pro_full.py verbatim (same flow as the original lane);
only the actor changes (base weights, no adapter, temperature 0).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "outputs/e1_e14_small_scale_2026-09-26/E1"
RAW = LANE / "raw"
SRC_RUN = ROOT / "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818"
SEED = 20260926
N = 10
MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_TOKENS = 8192
SYSTEM = "You are a deterministic source-code patch generator. Reply with the requested patch only."

sys.path.insert(0, str(ROOT / "zvf-program/flagship"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def flagship(stub_modal: bool):
    if stub_modal:
        m = mock.MagicMock()
        exc = types.ModuleType("modal.exception")
        exc.SandboxFilesystemNotFoundError = type("SandboxFilesystemNotFoundError", (Exception,), {})
        sys.modules["modal"] = m
        sys.modules["modal.exception"] = exc
    import modal_e1_swe_bench_pro_full as f
    return f


def selection():
    return json.loads((RAW / "selection.json").read_text())


def select():
    rows = [json.loads(l) for l in (SRC_RUN / "dataset_test_731.jsonl").read_text().splitlines() if l]
    assert len(rows) == 731
    idx = sorted(random.Random(SEED).sample(range(len(rows)), N))
    ids = [rows[i]["instance_id"] for i in idx]
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "selection.json").write_text(json.dumps(
        {"seed": SEED, "method": "sorted(random.Random(seed).sample(range(731), 10)) over dataset_test_731.jsonl order",
         "indices": idx, "instance_ids": ids}, indent=2))
    (RAW / "dataset_subset.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in idx))
    man = json.loads((SRC_RUN / "image_manifest.json").read_text())
    imgs = [x for x in man["images"] if x["instance_id"] in set(ids)]
    assert len(imgs) == N
    (RAW / "image_manifest_subset.json").write_text(json.dumps({**man, "count": N, "images": imgs}, indent=2))
    print(json.dumps(ids, indent=1))


def sources():
    f = flagship(stub_modal=False)
    import concurrent.futures as cf
    rows = [json.loads(l) for l in (RAW / "dataset_subset.jsonl").read_text().splitlines()]
    imgs = {x["instance_id"]: x["immutable_uri"] for x in json.loads((RAW / "image_manifest_subset.json").read_text())["images"]}
    todo = []
    for r in rows:
        p = RAW / "tasks" / r["instance_id"] / "source_context.json"
        if not p.exists():
            todo.append((f._sanitize_task(r), p))
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(f._snapshot_task_sources, t, imgs[t["instance_id"]]): (t, p) for t, p in todo}
        for fu in cf.as_completed(futs):
            t, p = futs[fu]
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.write_text(json.dumps(fu.result(), indent=1))
                print("ok", t["instance_id"])
            except Exception as e:  # recorded; task counts as failure later
                p.with_name("source_error.json").write_text(json.dumps({"error": repr(e)[:2000]}))
                print("ERR", t["instance_id"], repr(e)[:300])


def sample():
    f = flagship(stub_modal=True)
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    sc = tinker.ServiceClient().create_sampling_client(base_model=MODEL)
    rows = [json.loads(l) for l in (RAW / "dataset_subset.jsonl").read_text().splitlines()]
    cands = []
    for r in rows:
        iid = r["instance_id"]
        d = RAW / "tasks" / iid
        out = d / "generation.json"
        if out.exists():
            g = json.loads(out.read_text())
        else:
            src = d / "source_context.json"
            if not src.exists():
                g = {"instance_id": iid, "status": "SOURCE_CONTEXT_FAILED", "patch": "", "prompt_tokens": 0, "response_tokens": 0}
            else:
                task = f._sanitize_task(r)
                prompt = f._build_prompt(task, json.loads(src.read_text())["files"])
                text = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tok.encode(text, add_special_tokens=False)
                t0 = time.time()
                try:
                    res = sc.sample(T.ModelInput.from_ints(ids), num_samples=1,
                                    sampling_params=T.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)).result()
                    toks = list(res.sequences[0].tokens)
                    resp = tok.decode(toks, skip_special_tokens=True)
                    patch, reason = f._extract_diff(resp)
                    g = {"instance_id": iid, "status": "GENERATED" if patch else "GENERATION_FAILED",
                         "patch": patch, "patch_validation_reason": reason, "response_text": resp,
                         "prompt_tokens": len(ids), "response_tokens": len(toks),
                         "prompt_sha256": f._sha256_text(prompt), "seconds": round(time.time() - t0, 1),
                         "finished_at": now()}
                except Exception as e:
                    g = {"instance_id": iid, "status": "SAMPLING_ERROR", "patch": "", "error": repr(e)[:2000],
                         "prompt_tokens": len(ids), "response_tokens": 0}
            d.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(g, indent=1))
        print(iid, g["status"], g.get("prompt_tokens"), g.get("response_tokens"), g.get("patch_validation_reason", ""))
        # empty patches are still submitted: the native evaluator scores them as unresolved
        cands.append({"instance_id": iid, "patch": g.get("patch") or "", "prefix": "base_qwen36_e1_small"})
    (RAW / "candidates.json").write_text(json.dumps(cands, indent=1))


def evaluate():
    ev = ROOT / "outputs/e1_swe_bench_pro/evaluator"
    cands = json.loads((RAW / "candidates.json").read_text())
    # evaluator cannot apply an empty patch meaningfully; only non-empty go to Modal, empties are scored False
    nonempty = [c for c in cands if c["patch"]]
    (RAW / "candidates_nonempty.json").write_text(json.dumps(nonempty, indent=1))
    cmd = ["uv", "run", "--no-project", "--with", "modal==1.5.4", "--with", "pandas==3.0.5", "--with", "tqdm==4.70.0",
           "python", str(ROOT / "zvf-program/flagship/e1_swe_bench_pro_full_eval.py"),
           "--image_manifest_path", str(RAW / "image_manifest_subset.json"),
           "--raw_sample_path", str(RAW / "dataset_subset.jsonl"),
           "--patch_path", str(RAW / "candidates_nonempty.json"),
           "--output_dir", str(RAW / "evaluation"), "--dockerhub_username", "jefzda",
           "--scripts_dir", str(ev / "run_scripts"), "--num_workers", "5", "--block_network"]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ev, check=True)


def score():
    ids = selection()["instance_ids"]
    er_path = RAW / "evaluation/eval_results.json"
    er = json.loads(er_path.read_text()) if er_path.exists() else {}
    per, ptok, stok = {}, 0, 0
    for iid in ids:
        g = json.loads((RAW / "tasks" / iid / "generation.json").read_text())
        ptok += g.get("prompt_tokens") or 0
        stok += g.get("response_tokens") or 0
        per[iid] = {"generation_status": g["status"], "evaluated": iid in er, "resolved": bool(er.get(iid, False))}
    num = sum(v["resolved"] for v in per.values())
    out = {"numerator": num, "denominator": len(ids), "value": num / len(ids), "per_instance": per,
           "tinker_prompt_tokens": ptok, "tinker_sample_tokens": stok,
           "n_valid_patches": sum(v["generation_status"] == "GENERATED" for v in per.values()),
           "n_errors": sum(v["generation_status"] not in ("GENERATED", "GENERATION_FAILED") for v in per.values())}
    (RAW / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"select": select, "sources": sources, "sample": sample, "eval": evaluate, "score": score}[sys.argv[1]]()
