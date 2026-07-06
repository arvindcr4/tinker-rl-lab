"""P2 reproducibility: computes collapse_baseline_analysis.json from the n2 tensors.
ZVF (per-group zero-variance), easy/hard collapse, naive cross-prompt |adv| vs difficulty-isocline |adv|."""
import json, glob, numpy as np, os
files = sorted(glob.glob("experiments/results/n2_reward_tensor_resume/*_tensors.jsonl"))
summary = {}
for f in files:
    method = f.split('/')[-1].split('_')[0]
    rows = [json.loads(l) for l in open(f)]
    zvf, f1, f0, naive_rec, iso_rec = [], [], [], [], []
    for r in rows:
        rew = np.array(r["rewards"], float)
        if rew.ndim != 2: continue
        pmean = rew.mean(1); collapsed = rew.var(1) == 0
        zvf.append(collapsed.mean()); f1.append((pmean==1).mean()); f0.append((pmean==0).mean())
        if collapsed.sum():
            g = pmean.mean(); naive_rec.append(np.abs(pmean[collapsed]-g).mean())
            iso = []
            for i in np.where(collapsed)[0]:
                nb = np.abs(pmean-pmean[i]) <= 0.15; nb[i] = False
                iso.append(abs(pmean[i] - (pmean[nb].mean() if nb.sum() else pmean[i])))
            iso_rec.append(np.mean(iso))
    summary[method] = {"mean_ZVF": round(float(np.mean(zvf)),3),
        "frac_all_correct": round(float(np.mean(f1)),3), "frac_all_wrong": round(float(np.mean(f0)),3),
        "naive_baseline_absadv_on_collapsed": round(float(np.mean(naive_rec)),3),
        "isocline_baseline_absadv_on_collapsed": round(float(np.mean(iso_rec)),3)}
os.makedirs("experiments/results/p2_openings", exist_ok=True)
json.dump(summary, open("experiments/results/p2_openings/collapse_baseline_analysis.json","w"), indent=2)
print(json.dumps(summary, indent=2))
