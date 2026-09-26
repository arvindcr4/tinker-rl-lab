"""E1 paired vLLM arm. Same 10 instances, same source contexts (E1/raw/tasks/*/source_context.json), same prompt,
same diff extraction, same native evaluator. Only endpoint + model id change.

  python run_e1_vllm.py sample <trained|base>   (tinker python: needs transformers; modal is stubbed)
  python run_e1_vllm.py eval   <arm>            (shells out to the pinned evaluator via uv + modal)
  python run_e1_vllm.py apply  <arm>            (uv + modal: git apply --check diagnostic)
  python run_e1_vllm.py score  <arm>
  python run_e1_vllm.py paired
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e1 as base  # noqa: E402

LANE = base.LANE
SRC = LANE / "raw"


def out(arm):
    return LANE / f"vllm_{arm}" / "raw"


def sample(arm):
    import vllm_client as vc
    f = base.flagship(stub_modal=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base.MODEL)
    rows = [json.loads(l) for l in (SRC / "dataset_subset.jsonl").read_text().splitlines()]
    o = out(arm)
    o.mkdir(parents=True, exist_ok=True)
    model, ready = vc.wait_ready(arm)
    t0 = time.time()

    def one(r):
        iid = r["instance_id"]
        d = o / "tasks" / iid
        p = d / "generation.json"
        if p.exists():
            return json.loads(p.read_text())
        task = f._sanitize_task(r)
        prompt = f._build_prompt(task, json.loads((SRC / "tasks" / iid / "source_context.json").read_text())["files"])
        text = tok.apply_chat_template([{"role": "system", "content": base.SYSTEM}, {"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        mt = min(base.MAX_TOKENS, vc.MAX_MODEL_LEN - len(ids))
        s = time.time()
        try:
            res = vc.complete(arm, ids, mt)
            patch, reason = f._extract_diff(res["text"])
            g = {"instance_id": iid, "status": "GENERATED" if patch else "GENERATION_FAILED", "patch": patch,
                 "patch_validation_reason": reason, "response_text": res["text"], "prompt_tokens": res["prompt_tokens"],
                 "response_tokens": res["completion_tokens"], "finish_reason": res["finish_reason"],
                 "max_tokens_used": mt, "prompt_sha256": f._sha256_text(prompt), "seconds": round(time.time() - s, 1)}
        except Exception as e:
            g = {"instance_id": iid, "status": "SAMPLING_ERROR", "patch": "", "error": repr(e)[:2000],
                 "prompt_tokens": len(ids), "response_tokens": 0, "max_tokens_used": mt}
        d.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(g, indent=1))
        print(arm, iid[9:50], g["status"], g["prompt_tokens"], g["response_tokens"], mt, flush=True)
        return g

    with cf.ThreadPoolExecutor(4) as ex:  # <=4 in flight per endpoint
        gens = list(ex.map(one, rows))
    (o / "candidates.json").write_text(json.dumps(
        [{"instance_id": g["instance_id"], "patch": g.get("patch") or "", "prefix": f"vllm_{arm}_e1_small"} for g in gens], indent=1))
    (o / "timing.json").write_text(json.dumps({"model_id": model, "ready_wait_s": round(ready, 1),
                                               "sampling_wall_s": round(time.time() - t0, 1)}, indent=1))


def evaluate(arm):
    o = out(arm)
    ev = base.ROOT / "outputs/e1_swe_bench_pro/evaluator"
    nonempty = [c for c in json.loads((o / "candidates.json").read_text()) if c["patch"]]
    (o / "candidates_nonempty.json").write_text(json.dumps(nonempty, indent=1))
    t0 = time.time()
    if nonempty:
        cmd = ["uv", "run", "--no-project", "--with", "modal==1.5.4", "--with", "pandas==3.0.5", "--with", "tqdm==4.70.0",
               "python", str(base.ROOT / "zvf-program/flagship/e1_swe_bench_pro_full_eval.py"),
               "--image_manifest_path", str(SRC / "image_manifest_subset.json"),
               "--raw_sample_path", str(SRC / "dataset_subset.jsonl"),
               "--patch_path", str(o / "candidates_nonempty.json"),
               "--output_dir", str(o / "evaluation"), "--dockerhub_username", "jefzda",
               "--scripts_dir", str(ev / "run_scripts"), "--num_workers", "5", "--block_network"]
        subprocess.run(cmd, cwd=ev, check=True)
    (o / "eval_timing.json").write_text(json.dumps({"eval_wall_s": round(time.time() - t0, 1)}))


def apply_check(arm):
    import modal
    o = out(arm)
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in (SRC / "dataset_subset.jsonl").read_text().splitlines()}
    imgs = {x["instance_id"]: x["immutable_uri"] for x in json.loads((SRC / "image_manifest_subset.json").read_text())["images"]}
    cands = [c for c in json.loads((o / "candidates.json").read_text()) if c["patch"]]
    app = modal.App.lookup("pavlov-e1-small-apply-check", create_if_missing=True)

    def check(c):
        iid = c["instance_id"]
        sb = modal.Sandbox.create(image=modal.Image.from_registry(imgs[iid]), app=app, timeout=900, block_network=True)
        try:
            sb.filesystem.write_text(c["patch"], "/tmp/p.diff")
            p = sb.exec("bash", "-lc", f"cd /app && git reset -q --hard {rows[iid]['base_commit']} && git apply --check -v /tmp/p.diff")
            p.wait()
            return iid, {"applies": p.returncode == 0, "log": (p.stdout.read() + p.stderr.read())[-1500:]}
        finally:
            sb.terminate()

    with cf.ThreadPoolExecutor(5) as ex:
        res = dict(ex.map(check, cands))
    (o / "apply_check.json").write_text(json.dumps(res, indent=1))
    print(arm, "applies", sum(v["applies"] for v in res.values()), "/", len(res))


def score(arm):
    o = out(arm)
    ids = base.selection()["instance_ids"]
    er_p = o / "evaluation/eval_results.json"
    er = json.loads(er_p.read_text()) if er_p.exists() else {}
    ac_p = o / "apply_check.json"
    ac = json.loads(ac_p.read_text()) if ac_p.exists() else {}
    per, pt, st = {}, 0, 0
    for iid in ids:
        g = json.loads((o / "tasks" / iid / "generation.json").read_text())
        pt += g.get("prompt_tokens") or 0
        st += g.get("response_tokens") or 0
        per[iid] = {"generation_status": g["status"], "max_tokens_used": g.get("max_tokens_used"),
                    "finish_reason": g.get("finish_reason"), "applies": ac.get(iid, {}).get("applies", False),
                    "evaluated": iid in er, "resolved": bool(er.get(iid, False))}
    k = sum(v["resolved"] for v in per.values())
    s = {"numerator": k, "denominator": len(ids), "value": k / len(ids), "wilson95": base_wilson(k, len(ids)),
         "n_valid_patches": sum(v["generation_status"] == "GENERATED" for v in per.values()),
         "n_apply": sum(v["applies"] for v in per.values()),
         "n_errors": sum(v["generation_status"] == "SAMPLING_ERROR" for v in per.values()),
         "prompt_tokens": pt, "sample_tokens": st, "per_instance": per}
    (o / "score.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k2: v for k2, v in s.items() if k2 != "per_instance"}))


def base_wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired():
    t = json.loads((out("trained") / "score.json").read_text())
    b = json.loads((out("base") / "score.json").read_text())
    items, bb, cc = [], 0, 0
    for iid in base.selection()["instance_ids"]:
        x, y = t["per_instance"][iid], b["per_instance"][iid]
        bb += x["resolved"] and not y["resolved"]
        cc += y["resolved"] and not x["resolved"]
        items.append({"instance_id": iid, "trained_resolved": x["resolved"], "base_resolved": y["resolved"],
                      "trained_applies": x["applies"], "base_applies": y["applies"],
                      "trained_generation": x["generation_status"], "base_generation": y["generation_status"]})
    ab = sum(i["trained_applies"] and not i["base_applies"] for i in items)
    ac = sum(i["base_applies"] and not i["trained_applies"] for i in items)
    res = {"lane": "E1", "metric": "resolved rate (native SWE-bench Pro evaluator)", "n_items": len(items),
           "trained_value": t["value"], "base_value": b["value"], "difference": t["value"] - b["value"],
           "discordant_b_trained_only": bb, "discordant_c_base_only": cc, "mcnemar_exact_p": mcnemar_exact(bb, cc),
           "diagnostic_git_apply_check": {"trained": t["n_apply"], "base": b["n_apply"], "b": ab, "c": ac,
                                          "mcnemar_exact_p": mcnemar_exact(ab, ac)},
           "items": items}
    (LANE / "paired.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    fn = {"sample": sample, "eval": evaluate, "apply": apply_check, "score": score, "paired": paired}[sys.argv[1]]
    fn(*sys.argv[2:])
