"""E2 paired vLLM arm. Same 30 EffiBench problems, same prompt, same program build + Docker grader as run_e2.py.
Only endpoint + model id change.

  python run_e2_vllm.py sample <trained|base>  (tinker python for transformers tokenizer)
  python run_e2_vllm.py grade  <arm>           (build + docker exec + score)
  python run_e2_vllm.py paired
"""
from __future__ import annotations

import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "E1/code"))
import run_e2 as r2  # noqa: E402

LANE = r2.LANE
SRC = LANE / "raw"


def out(arm):
    return LANE / f"vllm_{arm}" / "raw"


def sample(arm):
    import concurrent.futures as cf
    import vllm_client as vc
    from transformers import AutoTokenizer

    d = r2.data()
    sel = json.loads((SRC / "selection.json").read_text())["positions"]
    header = (r2.CODE / "upstream_prompt.txt").read_text()
    tok = AutoTokenizer.from_pretrained(r2.MODEL)
    o = out(arm)
    o.mkdir(parents=True, exist_ok=True)
    model, ready = vc.wait_ready(arm)
    t0 = time.time()

    def one(i):
        e = d[i]
        prompt = (f"{header}\n# Task description:\n```python\n{e['markdown_description']}\n```\n"
                  f"# Test case:\n```python\n{e['small_test_cases']}\n```")
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False)
        try:
            res = vc.complete(arm, ids, r2.MAX_TOKENS)
            return {"problem_idx": e["problem_idx"], "completion": res["text"], "prompt_tokens": res["prompt_tokens"],
                    "response_tokens": res["completion_tokens"], "finish_reason": res["finish_reason"], "error": None}
        except Exception as ex:
            return {"problem_idx": e["problem_idx"], "completion": "", "prompt_tokens": len(ids), "response_tokens": 0,
                    "error": repr(ex)[:1000]}

    with cf.ThreadPoolExecutor(4) as ex:  # <=4 in flight per endpoint
        gens = list(ex.map(one, sel))
    with (o / "generations.jsonl").open("w") as fh:
        for g in gens:
            fh.write(json.dumps(g) + "\n")
    (o / "timing.json").write_text(json.dumps({"model_id": model, "ready_wait_s": round(ready, 1),
                                               "sampling_wall_s": round(time.time() - t0, 1)}, indent=1))
    print(arm, "errors", sum(bool(g["error"]) for g in gens), "tokens", sum(g["response_tokens"] or 0 for g in gens))


def grade(arm):
    o = out(arm)
    t0 = time.time()
    d = {e["problem_idx"]: e for e in r2.data()}
    gens = [json.loads(l) for l in (o / "generations.jsonl").read_text().splitlines()]
    for kind in ("canonical", "completion"):
        (o / "programs" / kind).mkdir(parents=True, exist_ok=True)
    for g in gens:
        e = d[g["problem_idx"]]
        (o / "programs/canonical" / f"{g['problem_idx']}.py").write_text(r2._program(e["canonical_solution"], e["test_case"]))
        p = r2._program(g["completion"], e["test_case"])
        if p is not None:
            (o / "programs/completion" / f"{g['problem_idx']}.py").write_text(p)
    cmd = ["docker", "--context", "colima", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "2g",
           "-v", f"{o / 'programs'}:/programs:ro", "-v", f"{r2.CODE / 'timer.py'}:/timer.py:ro",
           "python:3.11-slim", "python", "/timer.py"]
    (o / "exec_results.json").write_text(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    # score (same rule as run_e2.score)
    ex = json.loads((o / "exec_results.json").read_text())
    per, nets = {}, []
    for g in gens:
        k = str(g["problem_idx"])
        can, com = ex["canonical"].get(k, {}), ex["completion"].get(k)
        rec = {"passed": bool(com and com["passed"]), "canonical_passed": bool(can.get("passed")),
               "extractable": com is not None, "gen_error": g["error"], "finish_reason": g.get("finish_reason")}
        if rec["passed"] and can.get("passed"):
            rec["net"] = com["median_s"] / can["median_s"]
            nets.append(rec["net"])
        per[k] = rec
    n, k = len(per), sum(r["passed"] for r in per.values())
    s = {"numerator": k, "denominator": n, "value": k / n, "wilson95": r2.wilson(k, n),
         "net_geomean_over_passed": math.exp(statistics.mean(math.log(x) for x in nets)) if nets else None,
         "n_faster_than_canonical": sum(x < 1 for x in nets),
         "prompt_tokens": sum(g["prompt_tokens"] for g in gens), "sample_tokens": sum(g["response_tokens"] or 0 for g in gens),
         "n_errors": sum(bool(g["error"]) for g in gens), "grade_wall_s": round(time.time() - t0, 1), "per_problem": per}
    (o / "score.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({x: v for x, v in s.items() if x != "per_problem"}))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired():
    t = json.loads((out("trained") / "score.json").read_text())["per_problem"]
    b = json.loads((out("base") / "score.json").read_text())["per_problem"]
    keys = [str(x) for x in json.loads((SRC / "selection.json").read_text())["problem_idx"]]
    items = [{"problem_idx": int(k), "trained_passed": t[k]["passed"], "base_passed": b[k]["passed"],
              "trained_net": t[k].get("net"), "base_net": b[k].get("net")} for k in keys]
    bb = sum(i["trained_passed"] and not i["base_passed"] for i in items)
    cc = sum(i["base_passed"] and not i["trained_passed"] for i in items)
    tv, bv = sum(i["trained_passed"] for i in items) / len(items), sum(i["base_passed"] for i in items) / len(items)
    # secondary continuous: log NET on items passed in both arms, bootstrap CI of paired mean difference
    diffs = [math.log(i["trained_net"]) - math.log(i["base_net"]) for i in items if i["trained_net"] and i["base_net"]]
    ci = None
    if diffs:
        rng = random.Random(20260926)
        boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        ci = [boots[249], boots[9749]]
    res = {"lane": "E2", "benchmark": "EffiBench substitute", "metric": "pass@1", "n_items": len(items),
           "trained_value": tv, "base_value": bv, "difference": tv - bv,
           "discordant_b_trained_only": bb, "discordant_c_base_only": cc, "mcnemar_exact_p": mcnemar_exact(bb, cc),
           "secondary_log_net_paired": {"n_both_passed": len(diffs),
                                        "mean_diff_log_net_trained_minus_base": statistics.mean(diffs) if diffs else None,
                                        "bootstrap95": ci, "resamples": 10000, "seed": 20260926,
                                        "note": "wall-clock timing noise; negative = trained faster"},
           "items": items}
    (LANE / "paired.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({x: v for x, v in res.items() if x != "items"}, indent=1))


if __name__ == "__main__":
    {"sample": sample, "grade": grade, "paired": paired}[sys.argv[1]](*sys.argv[2:])
