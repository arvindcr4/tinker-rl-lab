#!/usr/bin/env python3
"""Build E3/E7/E12 result.json purely from files under each lane's raw/. Usage: finalize.py E3|E7|E12 '<json overrides>'"""
import glob, json, math, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def shim_tokens(lane):
    p = f"{BASE}/{lane}/raw/shim_calls.jsonl"
    return sum(r["prompt_tokens"] + r["completion_tokens"] for r in map(json.loads, open(p)))


def harbor_items(lane, jobs, selected):
    """Latest trial per task across the given jobs; env-build errors before any model call are superseded by a
    later job's trial of the same task (E7 env-fix rerun). Anything else without a reward = failure."""
    per = {}
    for job in jobs:
        for f in sorted(glob.glob(f"{BASE}/{lane}/raw/jobs/{job}/*/result.json")):
            d = json.load(open(f))
            task = d["task_name"]
            rew = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            exc = (d.get("exception_info") or {}).get("exception_type")
            per[task] = {"job": job, "trial": os.path.basename(os.path.dirname(f)), "reward": rew, "exception": exc}
    missing = [t for t in selected if t not in per]
    for t in missing:
        per[t] = {"job": None, "trial": None, "reward": None, "exception": "not_run"}
    return {t: per[t] for t in selected}


def main():
    lane = sys.argv[1]; ov = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    raw = f"{BASE}/{lane}/raw"
    if lane in ("E3", "E7"):
        sel = json.load(open(f"{raw}/selection.json"))["selected"]
        items = harbor_items(lane, ov.pop("jobs"), sel)
        json.dump(items, open(f"{raw}/per_item.json", "w"), indent=1)
        num = sum(1 for v in items.values() if v["reward"] == 1.0)
        den = len(sel); errs = sum(1 for v in items.values() if v["reward"] is None)
        toks = shim_tokens(lane)
    else:
        s = json.load(open(f"{raw}/scores_v2.json"))
        sel = [f"appbench-{p['task']}:{p['app']}" for p in s["per_task"]]
        num, den = s["passed"], s["total"]
        errs = sum(p["unparsed_as_fail"] for p in s["per_task"])
        toks = sum(r["prompt_tokens"] + r["completion_tokens"] for r in map(json.loads, open(f"{raw}/calls.jsonl")))
    res = {"lane": lane, "actor": "Qwen/Qwen3.6-35B-A3B base (no adapter), Tinker", "thinking": False,
           "temperature": 0, "n_selected": len(sel), "n_scored": len(sel) - (errs if lane != "E12" else 0),
           "n_errors": errs, "item_ids": sel, "value": round(num / den, 4), "numerator": num, "denominator": den,
           "wilson95": wilson(num, den), "raw_dir": f"{lane}/raw/"}
    res.update(ov)
    res["cost"]["tinker_tokens"] = toks
    order = ["lane", "original_benchmark", "benchmark_run", "scope", "substitute_gap", "actor", "thinking",
             "temperature", "max_tokens", "n_selected", "n_scored", "n_errors", "item_ids", "metric", "value",
             "numerator", "denominator", "wilson95", "grader", "compute_route", "cost", "started_utc",
             "finished_utc", "commands", "raw_dir", "caveats"]
    res = {k: res[k] for k in order if k in res} | {k: v for k, v in res.items() if k not in order}
    json.dump(res, open(f"{BASE}/{lane}/result.json", "w"), indent=1)
    print(lane, num, den, res["value"], res["wilson95"], "tokens", toks, "errors", errs)


if __name__ == "__main__":
    main()
