"""Recompute the GRPO evidence numbers in paper_P12_signal_starvation.tex (Section 6).

Inputs (read-only, checked-in):
  platform_hybrid/experiments/results/zvf_iter46_per_prompt_isog.tsv
      505 (seed, problem) keys x 5 Y_target rows; p_x = k/8 from one G=8 group
      (Qwen3-8B, GSM8K, Tinker, seeds 42/123/456), restricted to 0.05 < p_x < 0.95.
  platform_hybrid/experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl
      raw per-prompt G=8 binary reward tensors (Qwen3.5-4B, Tinker, seed 0, 40 steps x 16 prompts).

Output: platform_hybrid/experiments/results/review_2026-10-18/P12_signal_starvation/recompute.json
"""

import csv
import json
import math
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "platform_hybrid/experiments/results"
OUT = RES / "review_2026-10-18/P12_signal_starvation/recompute.json"


def load_cohort():
    rows = [
        r
        for r in csv.DictReader(
            (l for l in open(RES / "zvf_iter46_per_prompt_isog.tsv") if not l.startswith("#")),
            delimiter="\t",
        )
    ]
    keys: dict[tuple[str, str], float] = {}
    for r in rows:
        k = (r["seed"], r["problem_id"])
        p = float(r["p_x"])
        assert keys.setdefault(k, p) == p
    ytargets = sorted({r["Y_target"] for r in rows})
    return rows, keys, ytargets


def zvf_plugin(p, G):
    return np.mean(p**G + (1 - p) ** G)


def zvf_unbiased(k, n, G):
    # Unbiased estimate of p^G + (1-p)^G from k successes in n draws (G <= n).
    c = comb(n, G)
    return np.mean([(comb(int(kk), G) + comb(n - int(kk), G)) / c for kk in k])


def main():
    rows, keys, ytargets = load_cohort()
    seeds = np.array([k[0] for k in keys])
    probs = np.array([k[1] for k in keys])
    p = np.array(list(keys.values()))
    k8 = np.rint(p * 8).astype(int)
    out = {
        "n_rows": len(rows),
        "n_keys": len(keys),
        "Y_targets": ytargets,
        "n_problems": len(set(probs)),
        "seeds": sorted(set(seeds)),
        "k_values": sorted(set(k8.tolist())),
        "mean_p": p.mean(),
        "var_p": p.var(),
    }
    U = {G: (1 - zvf_plugin(p, G)) / math.sqrt(G) for G in range(2, 17)}
    lin = {G: (1 - zvf_plugin(p, G)) / G for G in range(2, 17)}
    out["U_sqrt"] = U
    out["U_sqrt_argmax"] = max(U, key=U.__getitem__)
    out["U_linear"] = lin
    out["U_linear_argmax"] = max(lin, key=lin.__getitem__)
    out["ZVF5"] = zvf_plugin(p, 5)
    # unbiased (hypergeometric) estimator from k of n=8, G<=8
    Uu = {G: (1 - zvf_unbiased(k8, 8, G)) / math.sqrt(G) for G in range(2, 9)}
    out["U_sqrt_unbiased_G2to8"] = Uu
    out["U_sqrt_unbiased_argmax"] = max(Uu, key=Uu.__getitem__)
    out["ZVF5_unbiased"] = zvf_unbiased(k8, 8, 5)
    # bootstraps of U(4)-U(5)
    rng = np.random.default_rng(20261018)
    d = lambda pp: (1 - zvf_plugin(pp, 4)) / 2 - (1 - zvf_plugin(pp, 5)) / math.sqrt(5)
    out["U4_minus_U5"] = d(p)
    B = 10000
    iid = [d(p[rng.integers(0, len(p), len(p))]) for _ in range(B)]
    uprob = sorted(set(probs))
    idx = {q: np.where(probs == q)[0] for q in uprob}
    clu = []
    for _ in range(B):
        pick = rng.choice(uprob, len(uprob))
        clu.append(d(p[np.concatenate([idx[q] for q in pick])]))
    out["boot_iid_95"] = [float(np.percentile(iid, 2.5)), float(np.percentile(iid, 97.5))]
    out["boot_problem_clustered_95"] = [
        float(np.percentile(clu, 2.5)),
        float(np.percentile(clu, 97.5)),
    ]
    out["boot_problem_clustered_frac_positive"] = float(np.mean(np.array(clu) > 0))
    # polarized construction as in analyze_breakthroughs.polarization_audit
    mu = p.mean()
    hi, lo = min(1.0, mu + 0.35), mu - 0.35
    w = (mu - lo) / (hi - lo)
    nh = int(round(1000 * w))
    pol = np.array([hi] * nh + [lo] * (1000 - nh))
    out["polarized"] = {
        "hi": hi,
        "lo": lo,
        "n_hi": nh,
        "n_lo": 1000 - nh,
        "mean": pol.mean(),
        "var": pol.var(),
        "ZVF5": zvf_plugin(pol, 5),
    }
    # boundary fires from raw tensors
    fires = {}
    for m in ("grpo", "aero", "gift", "areal"):
        ac = aw = obs = 0
        steps = set()
        for line in open(RES / f"n2_reward_tensor_resume/{m}_s0_tensors.jsonl"):
            rec = json.loads(line)
            steps.add(rec["step"])
            for g in rec["rewards"]:
                obs += 1
                s = sum(g)
                ac += s == len(g)
                aw += s == 0
        fires[m] = {"obs": obs, "steps": len(steps), "all_correct": ac, "all_wrong": aw}
    tot_ac = sum(v["all_correct"] for v in fires.values())
    tot_aw = sum(v["all_wrong"] for v in fires.values())
    out["boundary"] = {
        "per_method": fires,
        "obs": sum(v["obs"] for v in fires.values()),
        "all_correct": tot_ac,
        "all_wrong": tot_aw,
        "share_all_correct": tot_ac / (tot_ac + tot_aw),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
