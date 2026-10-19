"""P9 Deep Review (2026-10-18): recompute the 505-task conditional-utility audit.

Inputs (measured, Tinker Qwen3-8B GSM8K, 8 rollouts/task at T=1.0, 200 problems x 3 seeds):
  platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json
  platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv  (505 filtered tasks)
Outputs: platform_hybrid/experiments/results/review_2026-10-18/P9_dnb/u505.json

The 505 (seed, problem) tasks share 200 problems, so the bootstrap resamples
problems (clusters), not tasks.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "platform_hybrid/experiments/results"
OUT = RES / "review_2026-10-18/P9_dnb"

raw = {}
for s in (42, 123, 456):
    j = json.loads((RES / f"tinker_gsm8k_zvf_s{s}.json").read_text())
    for p in j["per_problem"]:
        raw[(str(s), str(p["problem_id"]))] = float(np.mean(p["rewards"]))

tasks = {}
for line in (RES / "zvf_iter46_per_prompt_isog.tsv").read_text().splitlines():
    if line.startswith(("#", "source")) or not line.strip():
        continue
    f = line.split("\t")
    tasks[(f[1], f[2])] = float(f[3])
assert max(abs(raw[k] - v) for k, v in tasks.items()) == 0.0

keys = list(tasks)
P = np.array([tasks[k] for k in keys])
prob = np.array([k[1] for k in keys])
P_all = np.array(list(raw.values()))


def Y(p, G):
    return 1 - p**G - (1 - p) ** G


def U(p, G):
    return Y(p, G).mean() / np.sqrt(G)


res: dict[str, object] = {
    "n_tasks_filtered": int(len(P)),
    "n_tasks_all": int(len(P_all)),
    "n_problems": int(len(set(prob))),
    "mean_px_filtered": round(float(P.mean()), 4),
    "U4": round(float(U(P, 4)), 5),
    "U5": round(float(U(P, 5)), 5),
    "U4_minus_U5": round(float(U(P, 4) - U(P, 5)), 5),
    "U4_all600": round(float(U(P_all, 4)), 5),
    "U5_all600": round(float(U(P_all, 5)), 5),
    "argmax_G_1_32": int(max(range(1, 33), key=lambda G: U(P, G))),
}
rng = np.random.default_rng(42)
uniq = np.array(sorted(set(prob)))
idx_by_prob = {u: np.where(prob == u)[0] for u in uniq}
bs = []
for _ in range(10000):
    pick = rng.choice(uniq, len(uniq), replace=True)
    q = P[np.concatenate([idx_by_prob[u] for u in pick])]
    bs.append(U(q, 4) - U(q, 5))
res["cluster_boot_ci95_U4_minus_U5"] = [round(float(x), 5) for x in np.percentile(bs, [2.5, 97.5])]
bins: dict[str, dict[str, float]] = {}
res["bins_Y32_minus_Y4"] = bins
for lo, hi in ((0, 0.5), (0.5, 0.75), (0.75, 1.01)):
    q = P[(P >= lo) & (P < hi)]
    bins[f"[{lo},{hi})"] = {"n": int(len(q)), "delta": round(float((Y(q, 32) - Y(q, 4)).mean()), 3)}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "u505.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
