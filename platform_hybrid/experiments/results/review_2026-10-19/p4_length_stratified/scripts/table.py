"""Summarise correctness-stratified rollout lengths from the *_lenstrat runs (DeepReview #245).

Usage: python table.py raw/arith_cap10.json raw/cot_cap200.json raw/cot_cap512.json > summary.md
Per run: rollout-weighted mean length of correct / incorrect rollouts over steps 0-4 and the
last 5 steps, the Spearman trend over steps of each, and the fraction of rollouts at the cap.
"""

import json
import sys

import numpy as np
from scipy.stats import spearmanr


def pooled(steps, key):
    L = np.concatenate([np.array(s["lengths"])[np.array(s["rewards"]) == key] for s in steps])
    return float(L.mean()) if L.size else float("nan")


def trend(steps, key):
    xs, ys = [], []
    for s in steps:
        m = s["len_correct_mean" if key == 1 else "len_incorrect_mean"]
        if m is not None:
            xs.append(s["step"])
            ys.append(m)
    return float(spearmanr(xs, ys).statistic) if len(xs) > 2 else float("nan")


print("| file | algo | seed | cap | held-out | len correct first5→last5 | len incorrect first5→last5 | gap (inc−cor) first5→last5 | ρ(step,len) cor / inc | at cap first5→last5 | correct share first5→last5 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
agg = {}
for path in sys.argv[1:]:
    d = json.load(open(path))
    for r in sorted(d["runs"], key=lambda r: (r["algo"], r["seed"])):
        sl = r["step_log"]
        f5, l5 = sl[:5], sl[-5:]
        c0, c1 = pooled(f5, 1), pooled(l5, 1)
        i0, i1 = pooled(f5, 0), pooled(l5, 0)
        cap0 = np.mean([s["frac_at_cap"] for s in f5])
        cap1 = np.mean([s["frac_at_cap"] for s in l5])
        sh0 = np.mean([s["n_correct"] / (s["n_correct"] + s["n_incorrect"]) for s in f5])
        sh1 = np.mean([s["n_correct"] / (s["n_correct"] + s["n_incorrect"]) for s in l5])
        ho = r.get("heldout_post_acc", r.get("heldout_acc"))
        pre = r.get("heldout_pre_acc")
        hos = f"{pre:.3f}→{ho:.3f}" if pre is not None else f"{ho:.3f}"
        print(
            f"| {path.split('/')[-1]} | {r['algo']} | {r['seed']} | {d['max_new']} | {hos} | {c0:.1f}→{c1:.1f} | {i0:.1f}→{i1:.1f} | "
            f"{i0 - c0:+.1f}→{i1 - c1:+.1f} | {trend(sl, 1):+.2f} / {trend(sl, 0):+.2f} | {cap0:.2f}→{cap1:.2f} | {sh0:.2f}→{sh1:.2f} |"
        )
        a = agg.setdefault((path.split("/")[-1], r["algo"]), [])
        a.append((i0 - c0, i1 - c1, trend(sl, 0), trend(sl, 1), cap1))
print()
print("| file | algo | n | gap first5 | gap last5 | Δgap | ρ(step,len_inc) | ρ(step,len_cor) | at cap last5 |")
print("|---|---|---|---|---|---|---|---|---|")
for (f, al), v in agg.items():
    v = np.array(v, dtype=float)
    m = np.nanmean(v, axis=0)
    print(f"| {f} | {al} | {len(v)} | {m[0]:+.1f} | {m[1]:+.1f} | {m[1] - m[0]:+.1f} | {m[2]:+.2f} | {m[3]:+.2f} | {m[4]:.2f} |")
