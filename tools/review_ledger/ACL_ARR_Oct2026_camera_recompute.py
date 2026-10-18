"""Recompute the ACL ARR camera-ready exhibit statistics from released artifacts.

E-4: platform_hybrid/experiments/results/e1_confirmatory_wandb_export.tsv
     (W&B e1-confirmatory-*, Qwen3-8B, GSM8K test n=500, 8 seeds/arm).
     Unit = seed. Paired differences vs GRPO; paired t (df=7); paired seed
     bootstrap 95% CI (10,000 resamples, RNG seed 20260714 as preregistered);
     BH over the four contrasts; exact 80%-power MDE (noncentral t, two-sided
     alpha .05) from each contrast's paired SD; TOST at +/-0.010.
E-2: platform_hybrid/experiments/results/samestack_ppo_grpo.json (5 seeds).
E-7: platform_hybrid/experiments/tinker-runs/results/h2h2_*_Qwen3.5-4B_*_S15.json
     (last-10 training reward per seed; one-way eta^2 over 12 cells).
"""

import csv, glob, json, math, pathlib, statistics as st
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/ACL_ARR_Oct2026_camera"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# ---- E-4
rows = [
    r
    for r in csv.DictReader(
        (
            l
            for l in open(
                ROOT / "platform_hybrid/experiments/results/e1_confirmatory_wandb_export.tsv"
            )
            if not l.startswith("#")
        ),
        delimiter="\t",
    )
]
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[int(r["seed"])] = float(r["heldout_score"])
seeds = sorted(by["grpo"])
e4 = {}
rng = np.random.default_rng(20260714)
pvals = {}
for arm in ["dapo", "gspo", "drgrpo", "aero"]:
    d = np.array([by[arm][s] - by["grpo"][s] for s in seeds])
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), n - 1)
    idx = rng.integers(0, n, size=(10000, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tcrit = stats.t.ppf(0.975, n - 1)

    # exact MDE: smallest delta with power 0.8 under noncentral t
    def power(delta):
        nc = delta / se
        return stats.nct.sf(tcrit, n - 1, nc) + stats.nct.cdf(-tcrit, n - 1, nc)

    lo_d, hi_d = 0.0, 1.0
    for _ in range(100):
        mid = (lo_d + hi_d) / 2
        lo_d, hi_d = (mid, hi_d) if power(mid) < 0.8 else (lo_d, mid)
    t90 = stats.t.ppf(0.95, n - 1)
    ci90 = (m - t90 * se, m + t90 * se)
    tost_p = max(stats.t.sf((m + 0.01) / se, n - 1), stats.t.cdf((m - 0.01) / se, n - 1))
    e4[arm] = dict(
        n_seeds=n,
        df=n - 1,
        mean_diff=round(m, 5),
        sd_diff=round(sd, 5),
        t=round(t, 3),
        p_raw=round(p, 4),
        boot95=[round(lo, 5), round(hi, 5)],
        t95=[round(m - tcrit * se, 5), round(m + tcrit * se, 5)],
        t90=[round(ci90[0], 5), round(ci90[1], 5)],
        tost_p=round(tost_p, 4),
        mde80=round(hi_d, 5),
    )
    pvals[arm] = p
order = sorted(pvals, key=pvals.get)
m4 = len(order)
adj = {}
prev = 1.0
for i in range(m4 - 1, -1, -1):
    a = order[i]
    prev = min(prev, pvals[a] * m4 / (i + 1))
    adj[a] = round(prev, 4)
for a in e4:
    e4[a]["p_bh"] = adj[a]
e4["_arm_means"] = {a: round(st.mean(v.values()), 5) for a, v in by.items()}
res["E4"] = e4

# ---- E-2
d2 = json.load(open(ROOT / "platform_hybrid/experiments/results/samestack_ppo_grpo.json"))
res["E2_stored"] = (
    {k: v for k, v in d2.items() if not isinstance(v, (list, dict))}
    if isinstance(d2, dict)
    else None
)

# ---- E-7
cells = {}
for f in glob.glob(
    str(ROOT / "platform_hybrid/experiments/tinker-runs/results/h2h2_*_Qwen3.5-4B_s*_S15.json")
):
    j = json.load(open(f))
    arm = pathlib.Path(f).name.split("_")[1]
    log = j.get("step_log") or j.get("steps") or j.get("history")
    r = [s.get("reward", s.get("mean_reward")) for s in log]
    cells.setdefault(arm, []).append(st.mean(r[-10:]))
allv = [v for vs in cells.values() for v in vs]
gm = st.mean(allv)
ssb = sum(len(v) * (st.mean(v) - gm) ** 2 for v in cells.values())
sst = sum((x - gm) ** 2 for x in allv)
res["E7"] = {
    "means": {a: round(st.mean(v), 4) for a, v in cells.items()},
    "n_per_arm": {a: len(v) for a, v in cells.items()},
    "within_sd": {a: round(st.stdev(v), 4) for a, v in cells.items()},
    "eta2_algorithm": round(ssb / sst, 4),
    "steps_per_run": len(r),
}
json.dump(res, open(OUT / "recompute.json", "w"), indent=1)
print(json.dumps(res, indent=1))
