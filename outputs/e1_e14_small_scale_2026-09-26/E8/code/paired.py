"""Paired vLLM arm: write E<n>/vllm_{trained,base}/result.json and E<n>/paired.json from raw files only.
Usage: python3 paired.py E8 E10 E11 E14"""
import glob, json, math, random, sys
from datetime import datetime
from pathlib import Path

BASE = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab/outputs/e1_e14_small_scale_2026-09-26")
EP = {"trained": ("https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "pavlov-public-portfolio-bf16"),
      "base": ("https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "qwen36-base-bf16")}
SEED = 20260926


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def boot_ci(diffs, reps=10000):
    rng = random.Random(SEED); n = len(diffs); ms = []
    for _ in range(reps):
        ms.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    ms.sort()
    return [round(ms[int(0.025 * reps)], 4), round(ms[int(0.975 * reps) - 1], 4)]


def wall(lane, arm):
    """Sum of [resume, end] segments (runs were interrupted once by a host disk-full event)."""
    lines = (BASE / lane / f"vllm_{arm}/raw/segments.txt").read_text().split("\n")
    tot, start = 0.0, None
    for l in lines:
        if not l.strip():
            continue
        tag, t = l.split(" ", 1)
        try:
            ts = datetime.strptime(t.strip(), "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            ts = None
        if tag.startswith("#"):
            continue
        if tag == "resume":
            start = ts
        elif tag == "end" and start and ts:
            tot += (ts - start).total_seconds(); start = None
    return round(tot)


def arm_result(lane, arm, tinker, k, n, per_item, tokens, extra, n_err):
    url, model = EP[arm]
    r = dict(tinker)
    r.update({"actor": f"vLLM endpoint {url} model {model} ({'Qwen3.6-35B-A3B + seed809 stepfinal LoRA, merged BF16' if arm == 'trained' else 'Qwen3.6-35B-A3B base BF16, same volume/engine'})",
              "compute_route": "Modal vLLM H200", "n_scored": 60 if lane == "E10" else n, "n_errors": n_err, "numerator": k, "denominator": n,
              "value": round(k / n, 4), "wilson95": wilson(k, n),
              "cost": {"tinker_tokens": extra.pop("tinker_judge_tokens", 0), "vllm_tokens": tokens, "colab_units": 0.0,
                       "modal_usd": extra.pop("judge_modal_usd", 0.0), "wall_seconds": wall(lane, arm),
                       "note": "actor GPU bill is shared and reconciled by the lead"},
              "started_utc": (BASE / lane / f"vllm_{arm}/raw/started_utc.txt").read_text().strip(),
              "finished_utc": (BASE / lane / f"vllm_{arm}/raw/finished_utc.txt").read_text().strip(),
              "raw_dir": f"{lane}/vllm_{arm}/raw/", "arm": f"paired-vllm-{arm}"})
    r.update(extra)
    r["caveats"] = [f"Paired vLLM arm ({arm}); compare only with the other vLLM arm, never with the Tinker-base result.json."] + extra.pop("arm_caveats", [])
    r.pop("arm_caveats", None)
    for key in ("per_category_correct_of_10", "native_metrics", "pass_at_1_both_denominators", "extraction_failures",
                "truncated_at_max_tokens", "correct_among_truncated", "correct_among_finished", "harm_score", "benign_score", "per_split"):
        if key in tinker and key not in extra:
            r.pop(key, None)
    r["commands"] = [f"ARM={arm} ACTOR_PORT=... JUDGE_PORT=... bash E8/code/run_arm.sh", f"python3 E8/code/paired.py {lane}"]
    (BASE / lane / f"vllm_{arm}").mkdir(exist_ok=True)
    (BASE / lane / f"vllm_{arm}/result.json").write_text(json.dumps(r, indent=1) + "\n")
    return r


def paired_binary(lane, items, outcomes, metric, arms_res, note=None):
    t, b_ = outcomes["trained"], outcomes["base"]
    b = sum(1 for i in items if t[i] and not b_[i]); c = sum(1 for i in items if b_[i] and not t[i])
    out = {"lane": lane, "metric": metric, "n_items": len(items),
           "trained_value": arms_res["trained"]["value"], "base_value": arms_res["base"]["value"],
           "difference": round(arms_res["trained"]["value"] - arms_res["base"]["value"], 4),
           "discordant_b_trained_only": b, "discordant_c_base_only": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4),
           "per_item": {i: {"trained": bool(t[i]), "base": bool(b_[i])} for i in items},
           "cost": {a: arms_res[a]["cost"] for a in arms_res}}
    if note:
        out["note"] = note
    return out


def load_tinker(lane):
    return json.loads((BASE / lane / "result.json").read_text())


def e8():
    tin = load_tinker("E8"); res, outc = {}, {}
    for arm in ("trained", "base"):
        g = [json.loads(l) for l in (BASE / f"E8/vllm_{arm}/raw/graded.jsonl").read_text().splitlines()]
        s = json.loads((BASE / f"E8/vllm_{arm}/raw/summary.json").read_text())
        assert s["item_ids"] == tin["item_ids"]
        outc[arm] = {x["task_id"]: x["correct"] for x in g}
        res[arm] = arm_result("E8", arm, tin, sum(outc[arm].values()), len(g), outc[arm],
                              s["tokens"]["prefill"] + s["tokens"]["sample"],
                              {"native_metrics": s["native_metrics"], "per_category_correct_of_10": s["per_category_correct_of_10"],
                               "arm_caveats": ["Text items: /v1/completions with the identical chat-templated prompt string; FigQA/TableQA image items: /v1/chat/completions (enable_thinking=false) with the same image bytes Tinker received."]},
                              s["errors"])
    return paired_binary("E8", tin["item_ids"], outc, tin["metric"], res)


def e11():
    tin = load_tinker("E11"); res, outc = {}, {}
    for arm in ("trained", "base"):
        s = json.loads((BASE / f"E11/vllm_{arm}/raw/summary.json").read_text())
        assert sorted(s["per_item"]) == tin["item_ids"]
        outc[arm] = s["per_item"]
        res[arm] = arm_result("E11", arm, tin, s["passes"], s["n"], outc[arm], s["prefill"] + s["sample"],
                              {"pass_at_1_both_denominators": s["score"], "extraction_failures": s["extraction_failures"],
                               "arm_caveats": ["/v1/completions with the identical chat-templated prompt string; native iverilog-12 sv-iv-test harness."]},
                              s["errors"])
    return paired_binary("E11", tin["item_ids"], outc, tin["metric"], res)


def e14():
    tin = load_tinker("E14"); res, outc = {}, {}
    for arm in ("trained", "base"):
        a = json.loads((BASE / f"E14/vllm_{arm}/raw/actor_summary.json").read_text())
        s = json.loads((BASE / f"E14/vllm_{arm}/raw/score_summary.json").read_text())
        g = [json.loads(l) for l in (BASE / f"E14/vllm_{arm}/raw/graded.jsonl").read_text().splitlines()]
        assert [f"row{i:05d}" for i in a["row_indices"]] == tin["item_ids"]
        outc[arm] = {f"row{x['row_index']:05d}": x["correct"] for x in g}
        res[arm] = arm_result("E14", arm, tin, s["correct"], s["n"], outc[arm], a["prefill"] + a["sample"],
                              {"truncated_at_max_tokens": s["truncated"], "correct_among_truncated": s["correct_among_truncated"],
                               "judge_modal_usd": s.get("judge_modal_usd", 0.0),
                               "arm_caveats": ["/v1/completions with the identical chat-templated prompt string; same Omni-Judge (Modal A10G) as the Tinker run."]},
                              a["errors"])
    return paired_binary("E14", tin["item_ids"], outc, tin["metric"], res)


def e10():
    tin = load_tinker("E10"); res, per = {}, {}
    for arm in ("trained", "base"):
        sc = {}; errs = 0
        for k in ("harmful", "benign"):
            d = json.loads(Path(glob.glob(str(BASE / f"E10/vllm_{arm}/raw/inspect_logs/{k}/*.json"))[0]).read_text())
            for x in d["samples"]:
                sc[(k, x["id"])] = x["scores"]["combined_scorer"]["value"]; errs += bool(x.get("error"))
        per[arm] = sc
        shim = [json.loads(l) for l in (BASE / f"E10/vllm_{arm}/raw/shim_log.jsonl").read_text().splitlines()]
        judge = [json.loads(l) for l in (BASE / f"E10/vllm_{arm}/raw/judge_shim_log.jsonl").read_text().splitlines()]
        tok = lambda L: sum(x["response"]["usage"]["total_tokens"] for x in L if "usage" in x["response"])
        H = [v for (k, _), v in sc.items() if k == "harmful"]; Bn = [v for (k, _), v in sc.items() if k == "benign"]
        refusals = int(sum(v["refusal"] for v in H))
        res[arm] = arm_result("E10", arm, tin, refusals, len(H), sc, tok(shim),
                              {"tinker_judge_tokens": tok(judge),
                               "harm_score": {"value": round(sum(v["score"] for v in H) / len(H), 4), "full_score_count": sum(v["score"] == 1.0 for v in H)},
                               "benign_score": {"value": round(sum(v["score"] for v in Bn) / len(Bn), 4), "full_score_count": sum(v["score"] == 1.0 for v in Bn),
                                                "benign_refusals": f'{int(sum(v["refusal"] for v in Bn))}/{len(Bn)}'},
                               "arm_caveats": ["Actor via the same local shim (code/shim.py) whose backend is /v1/completions with the identical rendered prompt (tools via Qwen template); judges = Tinker base via a separate shim, as in the Tinker run."]},
                              errs)
    sel = tin["item_ids"]
    items_h = [("harmful", i) for i in sel["harmful"]]; items_b = [("benign", i) for i in sel["benign"]]
    ref = {a: {k: per[a][k]["refusal"] == 1.0 for k in items_h} for a in per}
    b = sum(ref["trained"][i] and not ref["base"][i] for i in items_h); c = sum(ref["base"][i] and not ref["trained"][i] for i in items_h)
    dh = [per["trained"][i]["score"] - per["base"][i]["score"] for i in items_h]
    db = [per["trained"][i]["score"] - per["base"][i]["score"] for i in items_b]
    return {"lane": "E10", "n_items": {"harmful": len(items_h), "benign": len(items_b)},
            "refusal_rate": {"trained_value": res["trained"]["value"], "base_value": res["base"]["value"],
                             "difference": round(res["trained"]["value"] - res["base"]["value"], 4),
                             "discordant_b_trained_only": b, "discordant_c_base_only": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)},
            "harm_score": {"trained_value": res["trained"]["harm_score"]["value"], "base_value": res["base"]["harm_score"]["value"],
                           "paired_mean_difference": round(sum(dh) / len(dh), 4), "bootstrap95": boot_ci(dh)},
            "benign_score": {"trained_value": res["trained"]["benign_score"]["value"], "base_value": res["base"]["benign_score"]["value"],
                             "paired_mean_difference": round(sum(db) / len(db), 4), "bootstrap95": boot_ci(db)},
            "per_item": {f"{k}/{i}": {"trained": per["trained"][(k, i)], "base": per["base"][(k, i)]} for k, i in items_h + items_b},
            "cost": {a: res[a]["cost"] for a in res}}


if __name__ == "__main__":
    for lane in sys.argv[1:]:
        p = globals()[lane.lower()]()
        (BASE / lane / "paired.json").write_text(json.dumps(p, indent=1) + "\n")
        print(lane, json.dumps({k: v for k, v in p.items() if k not in ("per_item", "cost")}))
