"""E11 small-scale: VerilogEval (both framings), 25 problems per framing by seed 20260926, base Qwen3.6 on Tinker.
Reuses the retained native driver's prompt loading, extraction, sample layout, configure and sv-iv-test harness."""
import json, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
E11DIR = REPO / "outputs/e11_verilog_eval"
sys.path.insert(0, str(E11DIR))
import e11_model_run as E11
import e11_paid_run_driver as DRV
import os
if os.environ.get('ARM'):  # paired vLLM arm: same items/prompts/caps, different endpoint
    import vc as tk
else:
    import tk
WORKERS = 4 if os.environ.get('ARM') else 16

OUT = REPO / "outputs/e1_e14_small_scale_2026-09-26/E11"
RAW = (OUT / f"vllm_{os.environ['ARM']}" / "raw") if os.environ.get("ARM") else OUT / "raw"; RAW.mkdir(parents=True, exist_ok=True)
PER_DS, SEED, MAX_TOKENS = 25, 20260926, 4096

t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
env = DRV.harness_env()
# sv-iv-analyze/sv-generate use `#!/usr/bin/env python`; no `python` on this host, so shim it.
env["PATH"] = str(OUT / "code/bin") + ":" + env["PATH"]
rng = random.Random(SEED)
results, all_recs = {}, []
for ds in DRV.DATASETS:
    prompts = DRV.load_prompts(ds)  # canonical sorted order
    sel = sorted(rng.sample(prompts, PER_DS))

    def gen(item):
        pid, text = item
        p = RAW / "samples" / ds / f"{pid}.json"; p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return json.loads(p.read_text())
        rec = {"dataset": ds, "problem_id": pid}
        try:
            o = tk.sample([{"role": "user", "content": text}], MAX_TOKENS)
            rec.update(response=o["text"], prompt_tokens=o["prompt_tokens"], completion_tokens=o["completion_tokens"],
                       stop_reason=str(o["stop_reason"]), error=None)
        except Exception as e:
            rec.update(response="", prompt_tokens=0, completion_tokens=0, error=f"{type(e).__name__}: {e}")
        rec["module"] = E11.extract_module(rec["response"])
        p.write_text(json.dumps(rec, indent=1))
        return rec

    with ThreadPoolExecutor(WORKERS) as ex:
        recs = list(ex.map(gen, sel))
    all_recs += recs
    build = RAW / "build" / ds
    if build.exists():
        shutil.rmtree(build)
    cfg = DRV.configure_build(ds, build, env)
    assert cfg["exit_code"] == 0, cfg
    # Restrict the configured problem list to the sampled subset so make never tries to generate unsampled problems.
    (build / "problems.mk").write_text("problems = \\\n" + "".join(f"  {r['problem_id']} \\\n" for r in recs) + "\n")
    for r in recs:
        E11.write_sample(build, r["problem_id"], r["module"], prompt_tokens=r["prompt_tokens"],
                         resp_tokens=r["completion_tokens"], cost_usd=0.0)
    h = DRV.run_harness(build, env, [r["problem_id"] for r in recs])
    for pid, ok in h["direct_results"].items():
        results[f"verilog_eval/{ds}/{pid}"] = ok
    (RAW / f"harness_{ds}.json").write_text(json.dumps({k: v for k, v in h.items()}, indent=1))

score = E11.score_pass_at_1(results)
summary = {"started_utc": t0, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "per_item": results, "passes": sum(results.values()), "n": len(results), "score": score,
           "extraction_failures": sum(1 for r in all_recs if r["module"] is None),
           "errors": sum(1 for r in all_recs if r["error"]),
           "prefill": sum(r["prompt_tokens"] for r in all_recs), "sample": sum(r["completion_tokens"] for r in all_recs)}
(RAW / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_item"}, indent=1))
