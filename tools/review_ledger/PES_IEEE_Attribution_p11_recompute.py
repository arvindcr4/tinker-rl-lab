"""Recompute the eight-seed same-stack audit (W&B tinker-rl-lab e1-confirmatory-*) used in PES Sec. IV-D.

Seed is the unit: paired differences vs GRPO by seed, two-sided paired t (df=7),
Benjamini-Hochberg over the four contrasts, and the 80%-power two-sided alpha=0.05 MDE
from the noncentral t for each contrast.
Usage: python3 -I tools/review_ledger/PES_IEEE_Attribution_p11_recompute.py <wandb_runs.json> <out.json>
"""

import json
import math
import re
import statistics as st
import sys

from scipy import optimize, stats

runs = [
    r
    for r in json.load(open(sys.argv[1]))
    if r.get("project") == "tinker-rl-lab"
    and (r.get("name") or "").startswith("e1-confirmatory-")
    and r.get("state") == "finished"
]
acc: dict[str, dict[int, float]] = {}
cfgs = {}
for r in runs:
    match = re.match(r"e1-confirmatory-(\w+)-s(\d+)-", r["name"])
    if match is None:
        raise ValueError(f"invalid confirmatory run name: {r['name']!r}")
    arm, seed = match.group(1), int(match.group(2))
    s = r["summary"]
    key = (
        next(k for k in ("heldout_score", "heldout_acc") if k in s)
        if any(k in s for k in ("heldout_score", "heldout_acc"))
        else None
    )
    if key is None:
        raise SystemExit(f"no held-out key in {r['name']}: {sorted(s)}")
    acc.setdefault(arm, {})[seed] = s[key]
    cfgs[arm] = {
        k: r["config"].get(k)
        for k in (
            "model",
            "model_name",
            "max_steps",
            "steps",
            "num_generations",
            "group_size",
            "task",
            "dataset",
        )
    }


def mde(n, sd, alpha=0.05, power=0.8):
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    f = lambda d: (
        (
            1
            - stats.nct.cdf(tc, df, d * math.sqrt(n) / sd)
            + stats.nct.cdf(-tc, df, d * math.sqrt(n) / sd)
        )
        - power
    )
    return optimize.brentq(f, 1e-9, 10 * sd)


out = {
    "held_out_key_cfg": cfgs,
    "arm_means": {
        a: {
            "n": len(v),
            "mean": round(st.mean(v.values()), 4),
            "sd": round(st.stdev(v.values()), 4),
        }
        for a, v in acc.items()
    },
}
seeds = sorted(acc["grpo"])
con, ps = {}, []
for a in ("dapo", "gspo", "drgrpo", "aero"):
    d = [acc[a][s] - acc["grpo"][s] for s in seeds]
    t = stats.ttest_1samp(d, 0.0)
    sd = st.stdev(d)
    tc = stats.t.ppf(0.975, len(d) - 1)
    con[a] = {
        "n": len(d),
        "df": len(d) - 1,
        "mean_delta": round(st.mean(d), 4),
        "sd_d": round(sd, 5),
        "ci95": [
            round(st.mean(d) - tc * sd / math.sqrt(len(d)), 4),
            round(st.mean(d) + tc * sd / math.sqrt(len(d)), 4),
        ],
        "t": round(float(t.statistic), 3),
        "p": round(float(t.pvalue), 4),
        "mde80": round(mde(len(d), sd), 5),
    }
    ps.append((a, float(t.pvalue)))
ps.sort(key=lambda x: x[1])
m = len(ps)
adj, prev = {}, 1.0
for i in range(m - 1, -1, -1):
    prev = min(prev, ps[i][1] * m / (i + 1))
    adj[ps[i][0]] = round(prev, 4)
for a in con:
    con[a]["p_bh"] = adj[a]
out["contrasts_vs_grpo"] = con
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
