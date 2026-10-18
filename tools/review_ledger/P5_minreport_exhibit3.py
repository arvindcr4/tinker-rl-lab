"""Recompute every number in P5 Exhibit 3 (12-cell head-to-head) from raw run JSON.

Input : platform_hybrid/experiments/tinker-runs/results/week_h2h/h2h2_*.json
        (Tinker; Qwen3.5-4B; GSM8K; G=8; batch 2; 15 steps; seeds 42/123/456)
Output: platform_hybrid/experiments/results/review_2026-10-18/P5_minreport/exhibit3_stats.json

Unit of analysis = seed run (n=3 per arm). Unpaired: pooled SD + TOST (df=4 per pair).
Seed-blocked: two-way additive ANOVA arm + seed (df_arm=3, df_seed=2, df_res=6);
paired contrasts use the block-model residual MSE (df=6).
"""

import glob
import itertools
import json
import os
import statistics as st
from pathlib import Path
from typing import TypedDict


class ArmStatistics(TypedDict):
    last10_by_seed: list[float]
    mean_last10: float
    mean_zvf: float


class ExhibitStatistics(TypedDict, total=False):
    arms: dict[str, ArmStatistics]
    spread: float
    pooled_within_arm_sd: float
    tost_unpaired_df4: dict[str, dict[str, float]]
    tost_unpaired_pooled_df8_min_margin: dict[str, float]
    block_anova: dict[str, float | list[int]]
    paired_contrasts_df6: dict[str, dict[str, float | list[float]]]
    block_all_pairs_equivalence_margin: float


from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "platform_hybrid/experiments/tinker-runs/results/week_h2h"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P5_minreport"

ARMS = ["dapo", "drgrpo", "grpo", "gspo"]
SEEDS = [42, 123, 456]


def load():
    y, z = {}, {}
    for p in sorted(glob.glob(str(SRC / "h2h2_*_Qwen3.5-4B_*.json"))):
        d = json.load(open(p))
        arm = os.path.basename(p).split("_")[1]
        y[(arm, d["seed"])] = d["last10_avg"]
        z[(arm, d["seed"])] = st.mean(s["zvf"] for s in d["step_log"])
        assert d["steps"] == 15 and d["group_size"] == 8 and d["batch"] == 2
    assert len(y) == 12
    return y, z


def tost_unpaired(a, b, margin, alpha=0.05):
    n1, n2 = len(a), len(b)
    sp = ((n1 - 1) * st.variance(a) + (n2 - 1) * st.variance(b)) / (n1 + n2 - 2)
    se = (sp * (1 / n1 + 1 / n2)) ** 0.5
    df = n1 + n2 - 2
    d = st.mean(a) - st.mean(b)
    p1 = 1 - stats.t.cdf((d + margin) / se, df)
    p2 = stats.t.cdf((d - margin) / se, df)
    return max(p1, p2)


def min_margin(fn, lo=0.0, hi=1.0):
    for _ in range(60):
        mid = (lo + hi) / 2
        if fn(mid) < 0.05:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    y, z = load()
    res: ExhibitStatistics = {"arms": {}}
    for a in ARMS:
        res["arms"][a] = {
            "last10_by_seed": [y[(a, s)] for s in SEEDS],
            "mean_last10": st.mean(y[(a, s)] for s in SEEDS),
            "mean_zvf": st.mean(z[(a, s)] for s in SEEDS),
        }
    means = [res["arms"][a]["mean_last10"] for a in ARMS]
    res["spread"] = max(means) - min(means)

    # pooled within-arm SD (df = 8)
    ssw = sum((y[(a, s)] - res["arms"][a]["mean_last10"]) ** 2 for a in ARMS for s in SEEDS)
    res["pooled_within_arm_sd"] = (ssw / (12 - 4)) ** 0.5

    # unpaired TOST, min margin for each pair
    tost = {}
    for a, b in itertools.combinations(ARMS, 2):
        xa = [y[(a, s)] for s in SEEDS]
        xb = [y[(b, s)] for s in SEEDS]
        tost[f"{a}-{b}"] = {
            "p_at_0.05": tost_unpaired(xa, xb, 0.05),
            "p_at_0.10": tost_unpaired(xa, xb, 0.10),
            "p_at_0.15": tost_unpaired(xa, xb, 0.15),
            "min_margin": min_margin(lambda m: tost_unpaired(xa, xb, m)),
        }
    res["tost_unpaired_df4"] = tost

    # unpaired TOST with SD pooled over all four arms (df = 8)
    se8 = res["pooled_within_arm_sd"] * (2 / 3) ** 0.5
    t8 = stats.t.ppf(0.95, 8)
    res["tost_unpaired_pooled_df8_min_margin"] = {
        f"{a}-{b}": abs(res["arms"][a]["mean_last10"] - res["arms"][b]["mean_last10"]) + t8 * se8
        for a, b in itertools.combinations(ARMS, 2)
    }

    # two-way additive ANOVA (arm + seed)
    g = st.mean(y.values())
    sst = sum((v - g) ** 2 for v in y.values())
    ss_arm = 3 * sum((res["arms"][a]["mean_last10"] - g) ** 2 for a in ARMS)
    seed_means = {s: st.mean(y[(a, s)] for a in ARMS) for s in SEEDS}
    ss_seed = 4 * sum((seed_means[s] - g) ** 2 for s in SEEDS)
    ss_res = sst - ss_arm - ss_seed
    ms_res = ss_res / 6
    f_arm = (ss_arm / 3) / ms_res
    res["block_anova"] = {
        "frac_ss_seed": ss_seed / sst,
        "frac_ss_arm": ss_arm / sst,
        "frac_ss_res": ss_res / sst,
        "F_arm": f_arm,
        "df": [3, 6],
        "p_arm": 1 - stats.f.cdf(f_arm, 3, 6),
        "residual_sd": ms_res**0.5,
    }
    # seed-paired contrasts, block-model SE, 90% CI (= TOST alpha 0.05), and 95% CI
    se = (2 * ms_res / 3) ** 0.5
    t90 = stats.t.ppf(0.95, 6)
    t95 = stats.t.ppf(0.975, 6)
    pc: dict[str, dict[str, float | list[float]]] = {}
    worst = 0
    for a, b in itertools.combinations(ARMS, 2):
        d = res["arms"][a]["mean_last10"] - res["arms"][b]["mean_last10"]
        pc[f"{a}-{b}"] = {
            "diff": d,
            "ci90": [d - t90 * se, d + t90 * se],
            "ci95": [d - t95 * se, d + t95 * se],
        }
        worst = max(worst, abs(d) + t90 * se)
    res["paired_contrasts_df6"] = pc
    res["block_all_pairs_equivalence_margin"] = worst
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "exhibit3_stats.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
