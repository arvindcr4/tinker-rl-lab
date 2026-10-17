"""Re-analysis of the P2 'Bernoulli ZVF model is sign-wrong' result (delta = +0.1224 vs -0.07).

Shows that
  (1) the GSM8K per-group plug-in delta  h_G(k/G) - 1[k in {0,G}]  is >= 0 for EVERY group by
      construction (k in {0,G}: both terms are 1; otherwise the indicator is 0 and h_G > 0), and is
      strongly positive even when rollouts are exactly i.i.d. Bernoulli -- so its sign carries no
      information about 'anti-herding';
  (2) the arithmetic per-step delta  h_G(pbar_step) - ZVF_step  is <= 0 in expectation whenever
      prompt difficulty varies within the step (Jensen: h_G is convex, so mean_i h_G(p_i) >= h_G(pbar));
  (3) the GSM8K pooled ZVF (0.158) is ~3x the homogeneous-binomial value h_8(pbar) and is reproduced by a
      beta-binomial (prompt-difficulty dispersion) fit -- i.e. ZVF = E_i[h_G(p_i)] and the excess over
      h_G(pbar) measures dispersion of per-prompt success rates.

Run: python analysis_publication_fixes/p2_delta_artifact.py
Out: experiments/results/publication_fixes/zvf_confound/p2_delta_artifact.json (+ stdout .txt)
"""
import csv
import json
import math
import pathlib
import random
import statistics as st

from scipy import optimize, stats
from scipy.special import betaln, comb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/results"
OUT = RES / "publication_fixes/zvf_confound"
G = 8


def h(p, g=G):
    return p ** g + (1 - p) ** g


out = {}
# ---------- (1) GSM8K plug-in delta ----------
ks = []
for s in (42, 123, 456):
    d = json.load(open(RES / f"tinker_gsm8k_zvf_s{s}.json"))
    ks += [sum(int(x) for x in pp["rewards"]) for pp in d["per_problem"]]
n = len(ks)
dl = [h(k / G) - (k in (0, G)) for k in ks]
pbar = sum(ks) / (G * n)
zvf = sum(k in (0, G) for k in ks) / n
print(f"GSM8K: n={n} groups, ZVF_obs={zvf:.4f}, pbar={pbar:.4f}")
print(f"(1) plug-in delta: mean={st.mean(dl):+.4f}  min={min(dl):.4f}  frac>0={sum(x > 0 for x in dl) / n:.3f}  "
      f"frac<0={sum(x < 0 for x in dl) / n:.3f}")
# expected plug-in delta under the exact i.i.d. Bernoulli null, with per-prompt p drawn from the fitted beta
# and, separately, with every prompt at pbar
null_hom = sum(stats.binom.pmf(k, G, pbar) * (h(k / G) - (k in (0, G))) for k in range(G + 1))


def nll(t):
    a, b = math.exp(t[0]), math.exp(t[1])
    return -sum(math.log(comb(G, k)) + betaln(k + a, G - k + b) - betaln(a, b) for k in ks)


r = optimize.minimize(nll, [0, 0], method="Nelder-Mead")
a, b = map(math.exp, r.x)
bb = [comb(G, k) * math.exp(betaln(k + a, G - k + b) - betaln(a, b)) for k in range(G + 1)]
null_bb = sum(bb[k] * (h(k / G) - (k in (0, G))) for k in range(G + 1))
print(f"    E[plug-in delta] under exact iid Bernoulli(pbar): {null_hom:+.4f}")
print(f"    E[plug-in delta] under iid Bernoulli(p_i), p_i ~ Beta({a:.2f},{b:.2f}): {null_bb:+.4f}")
out["gsm8k_plugin_delta"] = dict(n=n, mean=st.mean(dl), min=min(dl), frac_pos=sum(x > 0 for x in dl) / n,
                                 null_iid_homogeneous=null_hom, null_iid_betabinomial=null_bb)

# ---------- (3) dispersion ----------
hist = [ks.count(k) for k in range(G + 1)]
zb = h(pbar)
zbb = bb[0] + bb[G]
icc = 1 / (a + b + 1)
binom = [n * stats.binom.pmf(k, G, pbar) for k in range(G + 1)]
chi = sum((o - e) ** 2 / e for o, e in zip(hist, binom))
print(f"(3) ZVF_obs={zvf:.4f}  homogeneous h_8(pbar)={zb:.4f} (ratio {zvf / zb:.2f})  "
      f"beta-binomial (ICC={icc:.3f}) predicts {zbb:.4f}")
print(f"    k-histogram obs={hist}; chi2 vs Binomial(8,pbar)={chi:.0f} (df=7)")
out["gsm8k_dispersion"] = dict(zvf_obs=zvf, pbar=pbar, h_pbar=zb, ratio=zvf / zb, beta_a=a, beta_b=b, icc=icc,
                               zvf_betabinomial=zbb, k_hist=hist, chi2_vs_binomial=chi)

# ---------- (2) arithmetic per-step delta = Jensen ----------
rows = [r for r in csv.DictReader((l for l in open(RES / "zvf_contrastive_yield.tsv") if not l.startswith("#")),
                                  delimiter="\t") if r["source"] == "groupsize_zvf_sweep"]
dd = [float(r["delta_div"]) for r in rows]
rng = random.Random(0)
bs = sorted(st.mean(rng.choices(dd, k=len(dd))) for _ in range(5000))
print(f"(2) arithmetic per-step delta (h_G(pbar_step) - ZVF_step): n={len(dd)} mean={st.mean(dd):+.4f} "
      f"95%CI [{bs[125]:+.4f},{bs[4875]:+.4f}]")
# Jensen illustration: same per-step mean, heterogeneous p_i, exact iid Bernoulli rollouts -> delta < 0
for Gx in (2, 4, 8, 16):
    sub = [r for r in rows if int(r["G"]) == Gx]
    if not sub:
        continue
    pb = st.mean(float(r["p_x"]) for r in sub)
    # two-point difficulty mixture with the same mean (half the prompts at pb +/- 0.3, clipped)
    lo, hi = max(pb - 0.3, 0), min(pb + 0.3, 1)
    jensen = h(pb, Gx) - 0.5 * (h(lo, Gx) + h(hi, Gx))
    print(f"    G={Gx:2d}: n={len(sub)} obs mean delta={st.mean(float(r['delta_div']) for r in sub):+.4f}; "
          f"iid-Bernoulli expected delta for a +/-0.3 difficulty spread at pbar={pb:.2f}: {jensen:+.4f}")
out["arith_step_delta"] = dict(n=len(dd), mean=st.mean(dd), ci=[bs[125], bs[4875]])
json.dump(out, open(OUT / "p2_delta_artifact.json", "w"), indent=1)
