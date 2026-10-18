"""Recompute PES IEEE paper numbers from released artifacts (seed = unit).

Run from repo root:  python3 -I tools/review_ledger/PES_IEEE_Attribution_recompute.py
Writes platform_hybrid/experiments/results/review_2026-10-18/PES_IEEE_Attribution/recompute.json
"""
import glob
import json
import math
import os
import statistics as st

from scipy import stats

R = "platform_hybrid/experiments/results"
OUT = f"{R}/review_2026-10-18/PES_IEEE_Attribution/recompute.json"
out = {}

# 1. N2 Phase-1 four-method tensors (Qwen3.5-4B GSM8K G=8, 16 prompts/step, 40 steps, seed 0)
n2 = {}
esc_all_one = esc_all_zero = 0
for f in sorted(glob.glob(f"{R}/n2_reward_tensor_resume/*_s0_tensors.jsonl")):
    m = os.path.basename(f).split("_")[0]
    zv, a1, a0, ng = [], 0, 0, 0
    for line in open(f):
        rec = json.loads(line)
        step_z = 0
        for g in rec["rewards"]:
            ng += 1
            if max(g) == min(g):
                step_z += 1
                if g[0] == 1.0:
                    a1 += 1
                else:
                    a0 += 1
        zv.append(step_z / len(rec["rewards"]))
    n2[m] = {"steps": len(zv), "groups": ng, "mean_step_zvf": round(st.mean(zv), 4),
             "degenerate": a1 + a0, "all_correct": a1, "all_wrong": a0,
             "all_correct_frac_of_groups": round(a1 / ng, 4)}
    esc_all_one += a1
    esc_all_zero += a0
out["n2"] = n2
out["n2_pooled"] = {"degenerate": esc_all_one + esc_all_zero, "all_correct": esc_all_one,
                    "share_all_correct": round(esc_all_one / (esc_all_one + esc_all_zero), 4)}

# 2. Group-size sweep (Qwen2.5-0.5B, arithmetic_correctness, 3 seeds x 40 steps)
sw = json.load(open(f"{R}/groupsize_zvf_sweep.json"))["runs"]
gs = {}
for G in (2, 4, 8, 16):
    z = [r["mean_zvf"] for r in sw if r["group_size"] == G]
    gu = [1 - v for v in z]
    gs[G] = {"n_seeds": len(z), "task": sorted({r["task"] for r in sw if r["group_size"] == G}),
             "mean_zvf": round(st.mean(z), 4), "gu": round(st.mean(gu), 4), "gu_sd": round(st.stdev(gu), 4),
             "heldout": [r["heldout_acc"] for r in sw if r["group_size"] == G]}
out["groupsize"] = gs

# 3. Same-stack PPO vs GRPO (Qwen2.5-0.5B, 40 steps, 5 seeds)
ss = f"{R}/modal_registry/modal_artifacts/tinkerrl-results/samestack"
d = []
per = {}
for s in (42, 123, 456, 789, 1024):
    g = json.load(open(f"{ss}/grpo_s{s}.json"))["heldout_acc"]
    p = json.load(open(f"{ss}/ppo_s{s}.json"))["heldout_acc"]
    per[s] = {"grpo": g, "ppo": p}
    d.append(g - p)
t = stats.ttest_1samp(d, 0.0)
sd = st.stdev(d)
tc = stats.t.ppf(0.975, len(d) - 1)
out["samestack"] = {"per_seed": per, "mean_delta_grpo_minus_ppo": round(st.mean(d), 4),
                    "t": round(float(t.statistic), 3), "df": len(d) - 1, "p": round(float(t.pvalue), 4),
                    "ci95": [round(st.mean(d) - tc * sd / math.sqrt(len(d)), 4),
                             round(st.mean(d) + tc * sd / math.sqrt(len(d)), 4)]}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
