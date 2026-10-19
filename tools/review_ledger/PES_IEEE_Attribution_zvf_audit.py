"""Recompute PES paper claims from the W&B `zvf-audit` project export (368 runs).

Input: W&B run export JSON (list of runs with config/summary), path as argv[1].
Output: JSON summary to argv[2].
Claims checked: (a) 368-run U-shape "ZVF 0.95-0.97 at reward extremes, 0.25-0.29 at
mid-range rewards 0.35-0.50"; (b) model x G grid "Qwen3-32B: 0.33 -> 0.18 -> 0.08 at G=4/8/16".
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict

runs = [r for r in json.load(open(sys.argv[1])) if r.get("project") == "zvf-audit"]
out = {"n_runs": len(runs), "states": Counter(r.get("state") for r in runs)}
tags = Counter(t for r in runs for t in r.get("tags", []))
out["tags"] = tags.most_common(20)
out["models"] = Counter(r["config"].get("model") for r in runs)
out["groups"] = Counter(r["config"].get("group") for r in runs)
out["steps"] = Counter(r["config"].get("steps") for r in runs)

# (a) U-shape: bin runs by mean reward proxy. Use summary keys available.
bins = defaultdict(list)
for r in runs:
    s = r.get("summary", {})
    z = s.get("mean_zvf")
    rew = s.get("reward")
    if z is None or rew is None:
        continue
    b = min(int(rew * 10), 9) / 10
    bins[b].append(z)
out["zvf_by_final_reward_bin"] = {
    f"{b:.1f}": {"n": len(v), "mean_zvf": round(st.mean(v), 3)} for b, v in sorted(bins.items())
}

# (b) model x G grid of mean ZVF
grid = defaultdict(list)
for r in runs:
    z = r.get("summary", {}).get("mean_zvf")
    if z is None:
        continue
    grid[(r["config"].get("model"), r["config"].get("group"))].append(z)
out["grid_mean_zvf"] = {
    f"{m}|G={g}": {"n": len(v), "mean": round(st.mean(v), 3)}
    for (m, g), v in sorted(grid.items(), key=lambda x: (str(x[0][0]), x[0][1] or 0))
}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
