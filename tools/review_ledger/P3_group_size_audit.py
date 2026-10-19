"""Re-verify the P3 505-task conditional-utility audit numbers against the checked-in cohort file.

Input:  platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv (columns seed, problem_id, p_x)
        platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json (to confirm p_x = k/8 provenance)
Output: platform_hybrid/experiments/results/review_2026-10-18/P3_group_size/utility_audit.json
Usage:  python tools/review_ledger/P3_group_size_audit.py
"""

import csv
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "platform_hybrid/experiments/results"
OUT = RES / "review_2026-10-18/P3_group_size"

rows = [
    r
    for r in csv.DictReader(
        (l for l in open(RES / "zvf_iter46_per_prompt_isog.tsv") if not l.startswith("#")),
        delimiter="\t",
    )
]
px = {}
for r in rows:
    px[(r["seed"], r["problem_id"])] = float(r["p_x"])
p = np.array(list(px.values()))

# provenance: p_x equals k/8 of the mixed groups in the three GSM8K JSONs
mixed = {}
for s in (42, 123, 456):
    d = json.load(open(RES / f"tinker_gsm8k_zvf_s{s}.json"))
    for i, pp in enumerate(d["per_problem"]):
        k = int(sum(pp["rewards"]))
        if 0 < k < 8:
            mixed[(str(s), str(pp.get("problem_id", i)))] = k / 8
match = all(abs(mixed.get(key, -1) - v) < 1e-4 for key, v in px.items())


def Y(G, q=p):
    return float(np.mean(1 - q**G - (1 - q) ** G))


U = {G: Y(G) / math.sqrt(G) for G in range(2, 33)}
rng = np.random.default_rng(0)
boot = []
for _ in range(4000):
    q = rng.choice(p, size=len(p), replace=True)
    boot.append(Y(4, q) / 2 - Y(5, q) / math.sqrt(5))
strata = {}
for name, m in (("lt0.5", p < 0.5), ("0.5-0.75", (p >= 0.5) & (p < 0.75)), ("ge0.75", p >= 0.75)):
    strata[name] = dict(n=int(m.sum()), Y32_minus_Y4=Y(32, p[m]) - Y(4, p[m]))
out = dict(
    n_tasks=len(p),
    mean_px=float(p.mean()),
    px_equals_k_over_8_of_mixed_groups=match,
    n_mixed_groups=len(mixed),
    U4=U[4],
    U5=U[5],
    argmax_sqrtG=max(U, key=U.__getitem__),
    diff_4_5=U[4] - U[5],
    boot_ci_4_minus_5=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    argmax_linear=max(range(2, 33), key=lambda G: Y(G) / G),
    strata=strata,
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "utility_audit.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
