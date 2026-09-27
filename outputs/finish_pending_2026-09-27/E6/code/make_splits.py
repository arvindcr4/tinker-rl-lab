"""E6: partition the 812 WebArena tasks into site-disjoint worker queues, each in seeded-random order.

Workers never share a writable site, so concurrent tasks cannot mutate each other's state.
Map and Wikipedia are read-only, so they ride along with any worker. Cross-site reddit tasks
(gitlab+reddit, shopping+reddit) run in a final serial phase after the other queues finish.
If the run is cut short, the completed prefix of every queue is a seeded random subset of that stratum.
"""
import json, random, sys
from collections import Counter

SEED = 20260927
raw = json.load(open(sys.argv[1]))
assert [t["task_id"] for t in raw] == list(range(812))
q = {"w1_shopping": [], "w2_shopping_admin": [], "w3_gitlab": [], "w4_reddit_map": [], "w5_cross_reddit": []}
for t in raw:
    s = set(t["sites"])
    if "reddit" in s and len(s) > 1:
        q["w5_cross_reddit"].append(t["task_id"])
    elif s == {"shopping"}:
        q["w1_shopping"].append(t["task_id"])
    elif "shopping_admin" in s:
        q["w2_shopping_admin"].append(t["task_id"])
    elif "gitlab" in s:
        q["w3_gitlab"].append(t["task_id"])
    else:  # reddit, map, wikipedia+map
        q["w4_reddit_map"].append(t["task_id"])
rng = random.Random(SEED)
for k in q:
    q[k].sort()
    rng.shuffle(q[k])
assert sorted(sum(q.values(), [])) == list(range(812))
out = sys.argv[2]
for k, v in q.items():
    open(f"{out}/{k}.txt", "w").write("\n".join(map(str, v)) + "\n")
json.dump({"seed": SEED, "rng": "python random.Random(seed).shuffle over each sorted queue, queues in listed order",
           "sizes": {k: len(v) for k, v in q.items()},
           "sites": {k: Counter("+".join(raw[i]["sites"]) for i in v) for k, v in q.items()}},
          open(f"{out}/splits.json", "w"), indent=1)
print({k: len(v) for k, v in q.items()})
