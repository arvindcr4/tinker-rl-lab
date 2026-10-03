#!/usr/bin/env python3
"""Paired vLLM arm: build E<n>/vllm_{trained,base}/result.json and E<n>/paired.json from raw files only.
Usage: paired_finalize.py E3|E7|E12 '<json: {"common": {...}, "vllm_trained": {...}, "vllm_base": {...}}>'"""
import datetime as dt, glob, json, math, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = {"vllm_trained": ("https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "pavlov-public-portfolio-bf16"),
        "vllm_base": ("https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "qwen36-base-bf16")}
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))  # harbor writes naive local (IST) timestamps


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def utc(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=IST).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ledger(p):
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    return sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows), len(rows)


def harbor_arm(lane, arm):
    raw = f"{BASE}/{lane}/{arm}/raw"
    sel = json.load(open(f"{BASE}/{lane}/raw/selection.json"))["selected"]
    per, starts, ends = {}, [], []
    for jf in glob.glob(f"{raw}/jobs/*/result.json"):
        j = json.load(open(jf)); starts.append(j["started_at"]); ends.append(j["finished_at"])
    for f in glob.glob(f"{raw}/jobs/*/*/result.json"):
        d = json.load(open(f))
        rew = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        per[d["task_name"]] = {"trial": os.path.basename(os.path.dirname(f)), "reward": rew,
                               "exception": (d.get("exception_info") or {}).get("exception_type")}
    items = {t: per.get(t, {"trial": None, "reward": None, "exception": "not_run"}) for t in sel}
    json.dump(items, open(f"{raw}/per_item.json", "w"), indent=1)
    outcome = {t: int(v["reward"] == 1.0) for t, v in items.items()}
    s, e = utc(min(starts)), utc(max(ends))
    wall = (dt.datetime.fromisoformat(max(ends)) - dt.datetime.fromisoformat(min(starts))).total_seconds()
    tok, calls = ledger(f"{raw}/shim_calls.jsonl")
    return sel, outcome, sum(v["reward"] is None for v in items.values()), s, e, wall, {"vllm_tokens": tok, "vllm_calls": calls}


def e12_arm(arm):
    raw = f"{BASE}/E12/{arm}/raw"
    sc = json.load(open(f"{raw}/scores_v2.json"))
    outcome = {f"t{p['task']}.{i}": int(x is True) for p in sc["per_task"] for i, x in sorted(p["verdicts"].items(), key=lambda kv: int(kv[0]))}
    wall = json.load(open(f"{raw}/scores_v2_wall.json"))["wall_s_this_invocation"]
    tok, calls = ledger(f"{raw}/shim_calls.jsonl"); jt, jc = ledger(f"{raw}/judge_calls.jsonl")
    s = open(f"{raw}/started_utc.txt").read().strip()
    e = (dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ") + dt.timedelta(seconds=wall)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [f"appbench-{p['task']}:{p['app']}" for p in sc["per_task"]], outcome, 0, s, e, wall, \
        {"vllm_tokens": tok, "vllm_calls": calls, "judge_tinker_tokens": jt}


def main():
    lane = sys.argv[1]; ov = json.loads(sys.argv[2])
    arms = {}
    for arm in ARMS:
        ids, outc, errs, s, e, wall, tok = e12_arm(arm) if lane == "E12" else harbor_arm(lane, arm)
        k, n = sum(outc.values()), len(outc)
        url, mid = ARMS[arm]
        res = {"lane": lane, **ov["common"], "actor": f"{mid} @ {url} (vLLM 0.28.0, BF16, one H200)",
               "n_selected": len(ids), "n_scored": len(ids) - errs, "n_errors": errs, "item_ids": ids,
               "value": round(k / n, 4), "numerator": k, "denominator": n, "wilson95": wilson(k, n),
               "compute_route": "Modal vLLM H200" + ov["common"].get("compute_route_suffix", ""),
               "cost": {"tinker_tokens": tok.get("judge_tinker_tokens", 0), "vllm_tokens": tok["vllm_tokens"],
                        "vllm_calls": tok["vllm_calls"], "colab_units": 0.0, "modal_usd": None,
                        "modal_usd_note": "shared GPU bill reconciled by lead"},
               "wall_time_s": round(wall), "started_utc": s, "finished_utc": e, "raw_dir": f"{lane}/{arm}/raw/"}
        res.pop("compute_route_suffix", None)
        res.update(ov.get(arm, {}))
        json.dump(res, open(f"{BASE}/{lane}/{arm}/result.json", "w"), indent=1)
        arms[arm] = (outc, res)
    t, b = arms["vllm_trained"][0], arms["vllm_base"][0]
    assert t.keys() == b.keys()
    nb = sum(t[i] == 1 and b[i] == 0 for i in t); nc = sum(t[i] == 0 and b[i] == 1 for i in t)
    tv, bv = arms["vllm_trained"][1]["value"], arms["vllm_base"][1]["value"]
    paired = {"lane": lane, "n_items": len(t), "trained_value": tv, "base_value": bv,
              "difference_trained_minus_base": round(tv - bv, 4),
              "discordant_b_trained_only": nb, "discordant_c_base_only": nc,
              "mcnemar_exact_p": round(mcnemar_exact(nb, nc), 4),
              "per_item": {i: {"trained": t[i], "base": b[i]} for i in t},
              "wall_time_s": {a: arms[a][1]["wall_time_s"] for a in arms},
              "tokens": {a: arms[a][1]["cost"] for a in arms}, **ov.get("paired", {})}
    json.dump(paired, open(f"{BASE}/{lane}/paired.json", "w"), indent=1)
    print(lane, "trained", tv, "base", bv, "diff", paired["difference_trained_minus_base"], "b/c", nb, nc,
          "p", paired["mcnemar_exact_p"], "wall", paired["wall_time_s"])


if __name__ == "__main__":
    main()
