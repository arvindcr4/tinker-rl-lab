"""Build E6/E9 vllm_{trained,base}/result.json and E<n>/paired.json from raw files.
Usage: paired_vllm.py <lane E6|E9> <root dir of small-scale outputs>"""
import json, math, os, random, sys
from math import comb

lane, root = sys.argv[1], sys.argv[2]
L = os.path.join(root, lane)
ARMS = {"vllm_trained": ("https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "pavlov-public-portfolio-bf16"),
        "vllm_base": ("https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "qwen36-base-bf16")}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(diffs, seed=20260926, B=10000):
    rng = random.Random(seed); n = len(diffs)
    ms = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B) - 1], 4)]


def usage(arm):
    u = [json.loads(x) for x in open(os.path.join(L, arm, "raw", "shim_usage.jsonl"))]
    return sum(x["prompt_tokens"] + x["sample_tokens"] for x in u), sum(1 for x in u if x.get("error"))


base_res = json.load(open(os.path.join(L, "result.json")))
outcomes, results = {}, {}
for arm, (url, mid) in ARMS.items():
    raw = os.path.join(L, arm, "raw")
    started = open(os.path.join(raw, "started_utc.txt")).read().strip()
    finished = open(os.path.join(raw, "finished_utc.txt")).read().strip()
    toks, shim_errs = usage(arm)
    res = dict(base_res)
    res.update({"actor": f"{mid} @ {url} (Modal vLLM, raw /v1/completions, same chat-templated token ids)", "compute_route": "Modal vLLM H200",
                "started_utc": started, "finished_utc": finished, "raw_dir": f"{lane}/{arm}/raw/"})
    if lane == "E6":
        eps = [json.loads(l) for l in open(os.path.join(raw, "episodes.jsonl"))]
        o = {f"{e['task']}#seed{e['episode_seed']}": int(e["success"]) for e in eps}
        k, n = sum(o.values()), len(o)
        res.update({"n_selected": n, "n_scored": n, "n_errors": sum(1 for e in eps if e["error"]), "item_ids": list(o),
                    "value": round(k / n, 4), "numerator": k, "denominator": n, "wilson95": wilson(k, n),
                    "per_task": {t: {"success": sum(v for i, v in o.items() if i.startswith(t + "#")), "n": sum(1 for i in o if i.startswith(t + "#"))}
                                 for t in json.load(open(os.path.join(raw, "selection.json")))["tasks"]},
                    "cost": {"tinker_tokens": 0, "vllm_tokens": toks, "colab_units": 0.0, "modal_usd": None}})
    else:
        g = json.load(open(os.path.join(raw, "grades.json")))
        o = {x["competition_id"]: {"any_medal": int(bool(x["any_medal"])), "above_median": int(bool(x["above_median"])),
                                   "valid": int(bool(x["valid_submission"])), "score": x["score"]} for x in g}
        n = len(o); med = sum(v["any_medal"] for v in o.values()); am = sum(v["above_median"] for v in o.values())
        res.update({"n_selected": n, "n_scored": n, "item_ids": list(o), "value": round(med / n, 4), "numerator": med, "denominator": n,
                    "wilson95": wilson(med, n),
                    "secondary_metrics": {"above_median": {"value": round(am / n, 4), "numerator": am, "denominator": n, "wilson95": wilson(am, n)},
                                          "valid_submission": {"numerator": sum(v["valid"] for v in o.values()), "denominator": n}},
                    "per_competition": [{k: x[k] for k in ("competition_id", "score", "any_medal", "gold_medal", "silver_medal", "bronze_medal",
                                                            "above_median", "median_threshold", "is_lower_better", "valid_submission")} for x in g],
                    "compute_route": "Modal vLLM H200 (actor) + Colab CPU (solution execution) + local mlebench grade_csv",
                    "cost": {"tinker_tokens": 0, "vllm_tokens": toks, "colab_units": 0.02, "modal_usd": None}})
    from datetime import datetime
    res["cost"]["wall_time_s"] = int((datetime.fromisoformat(finished.replace("Z", "+00:00")) - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds())
    res["caveats"] = [c for c in base_res["caveats"] if "tinker_tokens" not in c and "aborted" not in c] + [
        "Paired vLLM arm: identical items, prompts (chat template, enable_thinking=False), temperature 0, max_tokens, harness and grader as the Tinker-base run; only endpoint/model id changed.",
        f"shim request errors after 3 retries: {shim_errs}", "modal_usd not attributed per lane (shared GPU bill reconciled by lead)."]
    json.dump(res, open(os.path.join(L, arm, "result.json"), "w"), indent=1)
    outcomes[arm], results[arm] = o, res

tr, ba = outcomes["vllm_trained"], outcomes["vllm_base"]
items = [i for i in tr if i in ba]
paired = {"lane": lane, "n_items": len(items), "metric": results["vllm_trained"]["metric"],
          "trained_value": results["vllm_trained"]["value"], "base_value": results["vllm_base"]["value"],
          "difference": round(results["vllm_trained"]["value"] - results["vllm_base"]["value"], 4)}


def binary(key=None):
    t = [tr[i] if key is None else tr[i][key] for i in items]; b_ = [ba[i] if key is None else ba[i][key] for i in items]
    b = sum(1 for x, y in zip(t, b_) if x == 1 and y == 0); c = sum(1 for x, y in zip(t, b_) if x == 0 and y == 1)
    return {"b_trained_only": b, "c_base_only": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}


if lane == "E6":
    paired.update(binary())
    paired["per_item"] = [{"item": i, "trained": tr[i], "base": ba[i]} for i in items]
else:
    paired.update(binary("any_medal"))
    paired["above_median"] = {"trained": results["vllm_trained"]["secondary_metrics"]["above_median"]["value"],
                              "base": results["vllm_base"]["secondary_metrics"]["above_median"]["value"], **binary("above_median")}
    paired["per_item"] = [{"item": i, "trained": tr[i], "base": ba[i]} for i in items]
paired["arms"] = {a: {"endpoint": ARMS[a][0], "model": ARMS[a][1], "started_utc": results[a]["started_utc"],
                      "finished_utc": results[a]["finished_utc"], "vllm_tokens": results[a]["cost"]["vllm_tokens"], "wall_time_s": results[a]["cost"]["wall_time_s"]} for a in ARMS}
paired["claim_boundary"] = "Within-vLLM paired difference only; never differenced against the Tinker-base arm; no pooling across lanes."
json.dump(paired, open(os.path.join(L, "paired.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in paired.items() if k not in ("per_item",)}, indent=1))
