"""Summarize the Colab iso-compute group-size sweep (p3_isocompute_groupsize.py).

Reads platform_hybrid/experiments/results/p3_isocompute/*.json and writes
p3_isocompute_summary.tsv plus a LaTeX table body on stdout.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "platform_hybrid/experiments/results"
runs = [json.load(open(p)) for p in sorted((RES / "p3_isocompute").glob("*.json"))]


def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


rows = []
for w in (1, 0):
    for g in sorted({r["group_size"] for r in runs}):
        rs = [r for r in runs if r["group_size"] == g and int(r["whiten"]) == w]
        if not rs:
            continue
        post = np.array([r["heldout_post"] for r in rs])
        gain = np.array([r["heldout_post"] - r["heldout_pre"] for r in rs])
        zvf = np.array([r["mean_zvf"] for r in rs])
        cf = np.array([np.mean([s["zvf_closed_form_hetero"] for s in r["step_log"]]) for r in rs])
        homog = np.array([np.mean([s["mean_reward"] ** g + (1 - s["mean_reward"]) ** g for s in r["step_log"]]) for r in rs])
        rows.append({"whiten": w, "G": g, "n": len(rs), "prompts_per_step": rs[0]["prompts_per_step"],
                     "heldout_post": post.mean(), "heldout_post_ci95": ci95(post),
                     "heldout_gain": gain.mean(), "heldout_gain_ci95": ci95(gain),
                     "last10": np.mean([r["last10_avg"] for r in rs]),
                     "zvf": zvf.mean(), "zvf_hetero_closed_form": cf.mean(), "zvf_homog_closed_form": homog.mean(),
                     "seeds": ",".join(str(r["seed"]) for r in rs)})

with open(RES / "p3_isocompute_summary.tsv", "w") as fh:
    fh.write("\t".join(rows[0]) + "\n")
    for r in rows:
        fh.write("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")

for r in rows:
    print(f"{'yes' if r['whiten'] else 'no'} & {r['G']} & {r['prompts_per_step']} & {r['n']} & "
          f"${r['heldout_post']:.3f} \\pm {r['heldout_post_ci95']:.3f}$ & ${r['heldout_gain']:+.3f}$ & "
          f"{r['last10']:.3f} & {r['zvf']:.3f} & {r['zvf_hetero_closed_form']:.3f} & {r['zvf_homog_closed_form']:.3f} \\\\")

# Pairwise held-out contrasts among whitened arms (Welch; seeds are the unit).
w1 = {r["G"]: np.array([x["heldout_post"] for x in runs if x["group_size"] == r["G"] and int(x["whiten"]) == 1])
      for r in rows if r["whiten"] == 1}
gs = sorted(w1)
best = max(gs, key=lambda g: w1[g].mean())
print(f"\nbest whitened G by mean held-out: {best}")
for g in gs:
    if g != best and w1[g].size > 1:
        t = stats.ttest_ind(w1[best], w1[g], equal_var=False)
        print(f"G={best} vs G={g}: diff={w1[best].mean() - w1[g].mean():+.3f} Welch p={t.pvalue:.3g}")
# Whitening effect at each G (unpaired; seeds 42/123/456 shared).
for g in gs:
    a = np.array([x["heldout_post"] for x in runs if x["group_size"] == g and int(x["whiten"]) == 1 and x["seed"] in (42, 123, 456)])
    b = np.array([x["heldout_post"] for x in runs if x["group_size"] == g and int(x["whiten"]) == 0])
    if a.size > 1 and b.size > 1:
        t = stats.ttest_rel(a, b) if a.size == b.size else stats.ttest_ind(a, b, equal_var=False)
        print(f"G={g} whiten-unwhiten (seeds 42/123/456): {a.mean() - b.mean():+.3f} p={t.pvalue:.3g}")
