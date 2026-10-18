"""Regenerate RAIT2027 Fig. 1 (gradient utilisation vs G) from measured runs.

Input : platform_hybrid/experiments/results/groupsize_zvf_sweep.json
        (Qwen2.5-0.5B, synthetic a+b arithmetic, 3 seeds per G, 40 steps).
Output: outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/figures/fig_gradient_utilisation.pdf
        platform_hybrid/experiments/results/review_2026-10-18/RAIT2027_camera/fig_gu_fit.json
Replaces a hand-written TikZ figure with hard-coded values whose legend
quoted the ZVF fit R^2 against a GU curve. Fit: log ZVF = log a + b log G
(least squares on the four seed-averaged means); GU = 1 - a G^b.
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(ROOT / "platform_hybrid/experiments/results/groupsize_zvf_sweep.json"))
Gs = sorted(int(g) for g in d["summary"])
zvf = np.array([d["summary"][str(g)]["mean_zvf"] for g in Gs])
per_seed = {g: [1 - r["mean_zvf"] for r in d["runs"] if r["group_size"] == g] for g in Gs}
G = np.array(Gs, float)
gu = 1 - zvf
b, la = np.polyfit(np.log(G), np.log(zvf), 1)
a = float(np.exp(la))
pred = a * G ** b
r2 = 1 - ((np.log(zvf) - np.log(pred)) ** 2).sum() / ((np.log(zvf) - np.log(zvf).mean()) ** 2).sum()
rel = [(gu[i + 1] / gu[i] - 1) * 100 for i in range(len(Gs) - 1)]

out = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/RAIT2027_camera"
out.mkdir(parents=True, exist_ok=True)
json.dump({"a": round(a, 4), "b": round(float(b), 4), "r2_log_zvf": round(float(r2), 5),
           "gu_mean": dict(zip(Gs, np.round(gu, 4).tolist())),
           "rel_gain_pct_per_doubling": np.round(rel, 1).tolist()},
          open(out / "fig_gu_fit.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
for g in Gs:
    ax[0].scatter([g] * len(per_seed[g]), per_seed[g], s=10, color="0.6", zorder=2)
ax[0].plot(G, gu, "o", color="C0", label="seed mean (n=3)", zorder=3)
gg = np.geomspace(2, 16, 100)
ax[0].plot(gg, 1 - a * gg ** b, "-", color="C1",
           label=rf"$1-{a:.3f}\,G^{{{b:.3f}}}$")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(Gs); ax[0].set_xticklabels(Gs)
ax[0].set_xlabel("Group size $G$"); ax[0].set_ylabel(r"GU $=1-$ZVF")
ax[0].set_title("(a) Utilisation rises with $G$", fontsize=9); ax[0].legend(fontsize=7)
labels = [f"{Gs[i]}$\\to${Gs[i+1]}" for i in range(len(Gs) - 1)]
ax[1].bar(labels, rel, color="C0")
for i, v in enumerate(rel):
    ax[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
ax[1].set_ylabel("Relative GU gain (%)"); ax[1].set_title("(b) Gain per doubling declines", fontsize=9)
fig.tight_layout()
dst = ROOT / "outputs/PES_Conference_Submissions_2026-09-25/RAIT2027/figures/fig_gradient_utilisation.pdf"
fig.savefig(dst)
print(a, b, r2, rel)
