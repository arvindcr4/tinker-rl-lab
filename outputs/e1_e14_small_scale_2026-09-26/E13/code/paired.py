#!/usr/bin/env python3
"""E13 paired vLLM arm: trained vs base on the same 26 BALROG episodes (same engine).
Per-item outcome = native progression in [0,1] (continuous) -> paired bootstrap 95% CI (10k, seed 20260926)
on (a) mean per-episode difference and (b) the difference of the 5-env equal-weighted progression % (headline,
resampling episodes within each env). Also McNemar exact on full-success (progression==1) as a secondary."""
import json, math, random
from datetime import datetime, timezone
from pathlib import Path

E = Path(__file__).resolve().parents[1]
arms = {a: json.loads((E / f"vllm_{a}/raw/per_episode.json").read_text()) for a in ("trained", "base")}
res = {a: json.loads((E / f"vllm_{a}/result.json").read_text()) for a in arms}
ids = [r["id"] for r in arms["base"]]
assert ids == [r["id"] for r in arms["trained"]]
T = {r["id"]: r["progression"] for r in arms["trained"]}
B = {r["id"]: r["progression"] for r in arms["base"]}
env_of = {i: i.split("/")[0] for i in ids}
envs = list(dict.fromkeys(env_of.values()))


def overall(sel_ids, P):
    by = {}
    for i in sel_ids:
        by.setdefault(env_of[i], []).append(P[i])
    return sum(100 * sum(v) / len(v) for v in by.values()) / len(by)


d_items = [T[i] - B[i] for i in ids]
diff_overall = overall(ids, T) - overall(ids, B)
rng = random.Random(20260926)
by_env = {e: [i for i in ids if env_of[i] == e] for e in envs}
bo, bm = [], []
for _ in range(10000):
    s = [i for e in envs for i in (rng.choice(by_env[e]) for _ in by_env[e])]  # stratified by env
    bo.append(overall(s, T) - overall(s, B))
    bm.append(sum(T[i] - B[i] for i in s) / len(s))
q = lambda v: [round(sorted(v)[int(0.025 * len(v))], 3), round(sorted(v)[int(0.975 * len(v)) - 1], 3)]

b = sum(1 for i in ids if T[i] >= 1 and B[i] < 1)  # trained-only success
c = sum(1 for i in ids if B[i] >= 1 and T[i] < 1)
n = b + c
p_mc = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0

out = {
    "lane": "E13", "benchmark_run": res["base"]["benchmark_run"], "engine": "Modal vLLM H200 (both arms)",
    "trained": {"endpoint": "https://arvindcr4--pavlov-trained-actor-e1e14-serve.modal.run", "model": "pavlov-public-portfolio-bf16"},
    "base": {"endpoint": "https://arvindcr4--pavlov-trained-actor-e1e14-basectl-serve.modal.run", "model": "qwen36-base-bf16"},
    "n_items": len(ids), "item_type": "continuous (native BALROG progression in [0,1])",
    "metric": res["base"]["metric"],
    "trained_value": res["trained"]["value"], "base_value": res["base"]["value"],
    "difference": round(diff_overall, 3),
    "difference_bootstrap95": q(bo),
    "per_env": {e: {"trained": res["trained"]["per_env_progression_pct"][e], "base": res["base"]["per_env_progression_pct"][e],
                    "n": len(by_env[e])} for e in envs},
    "mean_item_difference": round(sum(d_items) / len(d_items), 4), "mean_item_difference_bootstrap95": q(bm),
    "bootstrap": {"resamples": 10000, "seed": 20260926, "scheme": "episodes resampled with replacement within each env (paired)"},
    "secondary_full_success_mcnemar": {"trained_only_b": b, "base_only_c": c, "exact_p": round(p_mc, 4)},
    "n_items_identical_progression": sum(1 for i in ids if T[i] == B[i]),
    "cost": {a: {"wall_time_s": res[a]["wall_time_s"], "vllm_tokens": res[a]["cost"]["vllm_tokens"],
                 "calls": res[a]["sampler_speed"]["calls"]} for a in arms},
    "items": [{"id": i, "trained": T[i], "base": B[i]} for i in ids],
    "caveats": ["Same engine for both arms; never differenced against the Tinker-base arm (E13/result.json).",
                "Small n (26 episodes, 2-10 per env); the CI is the claim, nothing stronger.",
                "Arms ran concurrently on separate H200 containers, each shim capped at 4 in-flight requests."],
    "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
(E / "paired.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("trained_value", "base_value", "difference", "difference_bootstrap95", "per_env",
                                      "mean_item_difference_bootstrap95", "secondary_full_success_mcnemar", "cost")}, indent=1))
