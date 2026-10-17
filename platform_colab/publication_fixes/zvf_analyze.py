"""Claim B analysis: observed ZVF vs binomial-identity predictions.

  python3 -I zvf_analyze.py zvf_raw.json OUT_JSON
Predictions from the 32 reference samples per prompt (k_i correct):
  hyper : mean_i [C(k,G)+C(32-k,G)]/C(32,G)   (U-statistic: unbiased for E[p^G+(1-p)^G] per prompt)
  plugin: mean_i [p_i^G + (1-p_i)^G], p_i = k_i/32
  h_G(pbar): pbar^G + (1-pbar)^G with pbar pooled (homogeneous-difficulty prediction)
  betabin: Beta(a,b) MLE fitted to k_i; E[p^G+(1-p)^G] = B(a+G,b)/B(a,b) + B(a,b+G)/B(a,b)
Observed: fraction of fresh groups (3 per prompt per G) with all-equal rewards.
CIs: prompt-level cluster bootstrap (10,000), paired for observed - predicted.
"""
import json, sys
import numpy as np
from scipy.special import comb, betaln
from scipy.optimize import minimize

D = json.load(open(sys.argv[1]))
ref = np.array(D["ref_rewards"])  # prompts x 32
N, M = ref.shape
k = ref.sum(1)
p = k / M
GS = sorted({int(key.split("_")[0][1:]) for key in D["groups"]})


def betabin_fit(k, n):
    def nll(t):
        a, b = np.exp(t)
        return -np.sum(betaln(k + a, n - k + b) - betaln(a, b))  # + const
    r = minimize(nll, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return np.exp(r.x)


def preds(idx, G):
    kk, pp = k[idx], p[idx]
    pbar = pp.mean()
    a, b = betabin_fit(kk, M)
    return {"hyper": float(np.mean((comb(kk, G) + comb(M - kk, G)) / comb(M, G))),
            "plugin": float(np.mean(pp ** G + (1 - pp) ** G)),
            "h_G_pbar": float(pbar ** G + (1 - pbar) ** G),
            "betabin": float(np.exp(betaln(a + G, b) - betaln(a, b)) + np.exp(betaln(a, b + G) - betaln(a, b)))}


def observed_per_prompt(G):
    reps = [np.array(D["groups"][f"G{G}_rep{j}"]["rewards"]) for j in range(3)]
    return np.mean([(r.min(1) == r.max(1)).astype(float) for r in reps], axis=0)  # per prompt, mean over 3 groups


a, b = betabin_fit(k, M)
out = {"model": D["model"], "n_prompts": N, "n_ref": M, "env": D["env"],
       "pass_rate_mean": float(p.mean()), "frac_prompts_all_correct_ref": float((k == M).mean()),
       "frac_prompts_all_wrong_ref": float((k == 0).mean()), "ref_trunc_frac": float(np.mean(D["ref_trunc"])),
       "betabin": {"alpha": float(a), "beta": float(b), "icc": float(1 / (a + b + 1))}, "by_G": {}}
rng = np.random.default_rng(0)
B = 10000
boot_idx = rng.integers(0, N, (B, N))
for G in GS:
    obs_pp = observed_per_prompt(G)
    P = preds(np.arange(N), G)
    hyper_pp = (comb(k, G) + comb(M - k, G)) / comb(M, G)
    obs_b = obs_pp[boot_idx].mean(1)
    hyp_b = hyper_pp[boot_idx].mean(1)
    plug_pp = p ** G + (1 - p) ** G
    diff_b = obs_b - hyp_b
    # h_G(pbar) and beta-binomial need a refit per bootstrap draw; 2,000 draws suffice.
    hpb, bb = [], []
    for idx in boot_idx[:2000]:
        pr = preds(idx, G); hpb.append(obs_pp[idx].mean() - pr["h_G_pbar"]); bb.append(obs_pp[idx].mean() - pr["betabin"])
    trunc = np.mean([np.mean(D["groups"][f"G{G}_rep{j}"]["trunc"]) for j in range(3)])
    out["by_G"][f"G{G}"] = {
        "observed": float(obs_pp.mean()), "observed_ci95": list(np.percentile(obs_b, [2.5, 97.5])),
        "n_groups": int(3 * N), "group_trunc_frac": float(trunc), "pred": P,
        "pred_hyper_ci95": list(np.percentile(hyp_b, [2.5, 97.5])),
        "obs_minus_hyper": float(obs_pp.mean() - P["hyper"]), "obs_minus_hyper_ci95": list(np.percentile(diff_b, [2.5, 97.5])),
        "obs_minus_plugin": float(obs_pp.mean() - P["plugin"]),
        "obs_minus_plugin_ci95": list(np.percentile(obs_b - plug_pp[boot_idx].mean(1), [2.5, 97.5])),
        "obs_minus_h_G_pbar": float(obs_pp.mean() - P["h_G_pbar"]), "obs_minus_h_G_pbar_ci95": list(np.percentile(hpb, [2.5, 97.5])),
        "obs_minus_betabin": float(obs_pp.mean() - P["betabin"]), "obs_minus_betabin_ci95": list(np.percentile(bb, [2.5, 97.5])),
        "prompt_level_corr_obs_vs_hyper": float(np.corrcoef(obs_pp, hyper_pp)[0, 1]),
    }
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({g: {k2: v[k2] for k2 in ("observed", "observed_ci95", "pred", "obs_minus_hyper_ci95", "obs_minus_h_G_pbar_ci95", "obs_minus_betabin_ci95")}
                  for g, v in out["by_G"].items()}, indent=1), out["betabin"], out["pass_rate_mean"])
