"""Build E<n>/paired.json from vllm_trained/result.json and vllm_base/result.json (per-item outcomes).
usage: python paired.py <lane_dir> <binary|continuous>"""
import json
import random
import sys
from math import comb
from pathlib import Path

lane, kind = Path(sys.argv[1]), sys.argv[2]
tr = json.loads((lane / "vllm_trained" / "result.json").read_text())
ba = json.loads((lane / "vllm_base" / "result.json").read_text())
key = "per_item_pass" if kind == "binary" else "per_item_reward"
items = list(tr[key])
assert items == list(ba[key]), "arms must score identical item ids"
t = [float(tr[key][i]) for i in items]
b = [float(ba[key][i]) for i in items]
n = len(items)
out = {"lane": lane.name, "n_items": n, "item_ids": items,
       "per_item": {i: {"trained": x, "base": y} for i, x, y in zip(items, t, b)},
       "trained_value": round(sum(t) / n, 4), "base_value": round(sum(b) / n, 4),
       "difference": round((sum(t) - sum(b)) / n, 4)}
if kind == "binary":
    bb = sum(1 for x, y in zip(t, b) if x == 1 and y == 0)  # trained pass, base fail
    cc = sum(1 for x, y in zip(t, b) if x == 0 and y == 1)
    m = bb + cc
    p = min(1.0, 2 * sum(comb(m, k) for k in range(0, min(bb, cc) + 1)) / 2 ** m) if m else 1.0
    out.update({"discordant_b_trained_only": bb, "discordant_c_base_only": cc, "mcnemar_exact_p": round(p, 4)})
else:
    d = [x - y for x, y in zip(t, b)]
    rng = random.Random(20260926)
    boots = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(10_000))
    out.update({"paired_mean_difference": round(sum(d) / n, 4),
                "bootstrap95_ci": [round(boots[249], 4), round(boots[9749], 4)],
                "bootstrap": "10k resamples of per-item differences, seed 20260926, percentile"})
(lane / "paired.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
