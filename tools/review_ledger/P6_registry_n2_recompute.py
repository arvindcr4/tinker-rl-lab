#!/usr/bin/env python3
"""Recompute the P6 N2 same-stack table (Deep Review 2026-10-18).

N2 is ONE seed (seed 0) per method, 40 steps. Steps within a run are not
independent replicates, so this script reports descriptive point deltas only
(variant - grpo); no CI or significance test is computed (protocol rule 4).

Inputs : platform_hybrid/experiments/results/n2_reward_tensor_resume/{n2_metrics.tsv,*_s0_tensors.jsonl}
Outputs: platform_hybrid/experiments/results/review_2026-10-18/P6_registry/n2_descriptive_deltas.tsv
"""
import csv, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
N2 = ROOT / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P6_registry"
METHODS = ("grpo", "aero", "gift", "areal")
METRICS = ("reward_mean", "zvf", "mean_len", "cv_len", "loss")

rows = list(csv.DictReader((N2 / "n2_metrics.tsv").open(), delimiter="\t"))
by = {}
for r in rows:
    assert r["seed"] == "0"
    by.setdefault(r["method"], {})[int(r["step"])] = {m: float(r[m]) for m in METRICS}
for m in METHODS:
    assert sorted(by[m]) == list(range(40)), m

def wmean(m, metric, steps):
    return statistics.fmean(by[m][s][metric] for s in steps)

windows = {"last10": range(30, 40), "full40": range(40)}
out = []
for wname, steps in windows.items():
    for metric in METRICS:
        g = wmean("grpo", metric, steps)
        for v in METHODS[1:]:
            x = wmean(v, metric, steps)
            out.append(dict(window=wname, metric=metric, variant=v, grpo_mean=g, variant_mean=x, delta=x - g))

# per-prompt pooled reward: each prompt's mean reward over all steps it appears in, then mean over prompts
def prompt_means(method, steps):
    acc = {}
    for line in (N2 / f"{method}_s0_tensors.jsonl").open():
        rec = json.loads(line)
        if rec["step"] not in steps:
            continue
        for pid, g in zip(rec["prompt_indices"], rec["rewards"]):
            acc.setdefault(pid, []).append(statistics.fmean(g))
    return {p: statistics.fmean(v) for p, v in acc.items()}

for wname, steps in windows.items():
  pg = prompt_means("grpo", steps)
  for v in METHODS[1:]:
    pv = prompt_means(v, steps)
    assert set(pv) == set(pg)
    d = statistics.fmean(pv[p] - pg[p] for p in pg)
    out.append(dict(window="per_prompt_" + wname, metric="reward_mean", variant=v,
                    grpo_mean=statistics.fmean(pg.values()), variant_mean=statistics.fmean(pv.values()), delta=d))

# truncation share of N2 rollouts (completion length == 512 cap)
trunc = {}
for m in METHODS:
    n = k = 0
    for line in (N2 / f"{m}_s0_tensors.jsonl").open():
        for g in json.loads(line)["lengths"]:
            n += len(g); k += sum(L >= 512 for L in g)
    trunc[m] = k / n

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "n2_descriptive_deltas.tsv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t")
    w.writeheader(); w.writerows(out)
(OUT / "n2_truncation_share.json").write_text(json.dumps(dict(cap=512, share_at_cap=trunc, n_prompts=16), indent=1))
for r in out:
    print(f"{r['window']:22s} {r['metric']:12s} {r['variant']:6s} grpo={r['grpo_mean']:.4f} var={r['variant_mean']:.4f} d={r['delta']:+.4f}")
print("truncation share at 512:", trunc)
