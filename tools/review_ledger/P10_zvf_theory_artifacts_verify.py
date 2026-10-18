#!/usr/bin/env python3
"""Verify P10 (ZVF theory) closed forms numerically and re-read every Section 7 number
from its checked-in artifact. Writes a JSON summary to
platform_hybrid/experiments/results/review_2026-10-18/P10_zvf_theory/verify.json.

Algebra checks (no data): T2 discrete quantile, T3 identities J(2)=J(3)>J(G>=4),
Beta closed form, continuous-relaxation critical point in (2,3), E[Varhat]=p(1-p).
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import integrate, special, stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "zvf-program/experiments-next/results"
TR = ROOT / "platform_hybrid/experiments/tinker-runs/results"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P10_zvf_theory"
out = {}

# ---------------- algebra ----------------
ps = np.linspace(1e-4, 1 - 1e-4, 20001)
f = lambda G, p: (1 - p**G - (1 - p) ** G) / G
out["T3_identity_maxabs_G2_vs_G3"] = float(np.max(np.abs(f(2, ps) - f(3, ps))))
out["T3_identity_maxabs_G2_vs_p1mp"] = float(np.max(np.abs(f(2, ps) - ps * (1 - ps))))
out["T3_strict_G4to64_all_p"] = bool(all(np.all(f(G, ps) < ps * (1 - ps)) for G in range(4, 65)))
# continuous relaxation: derivative of f at G=2 positive, at G=3 negative, for all p
dG = lambda G, p: (
    -(1 - p**G - (1 - p) ** G) / G**2 - (p**G * np.log(p) + (1 - p) ** G * np.log(1 - p)) / G
)
out["T3_dfdG_at2_min"] = float(np.min(dG(2, ps)))
out["T3_dfdG_at3_max"] = float(np.max(dG(3, ps)))
# is f_p unimodal on [2, 64] for every p? (sign changes of derivative)
Gs = np.linspace(2, 64, 3101)
sign_changes = max(
    int(np.sum(np.diff(np.sign(dG(Gs, p))) != 0)) for p in np.linspace(0.001, 0.999, 999)
)
out["T3_max_sign_changes_dfdG_on_[2,64]"] = sign_changes
g_half = Gs[np.argmax(f(Gs, 0.5))]
from scipy.optimize import minimize_scalar

out["T3_cont_argmax_p0.5"] = float(
    minimize_scalar(lambda G: -f(G, 0.5), bounds=(2, 3), method="bounded").x
)
# E[Varhat | p] = p(1-p) for unbiased sample variance of K Bernoulli
ev = []
for K in (2, 3, 4, 8, 16):
    for p in (0.1, 0.37, 0.8):
        s = np.arange(K + 1)
        pm = stats.binom.pmf(s, K, p)
        ev.append(abs(np.sum(pm * s * (K - s) / (K * (K - 1))) - p * (1 - p)))
out["E_varhat_eq_p1mp_maxerr"] = float(max(ev))
# Beta closed form vs quadrature
errs = []
for a, b in ((0.5, 0.5), (2, 5), (1, 1), (0.3, 3)):
    for G in (2, 3, 4, 8, 16):
        B = special.beta
        cf = (B(a + 1, b + 1) - B(a + G + 1, b + 1) - B(a + 1, b + G + 1)) / (G * B(a, b))
        q = integrate.quad(lambda p: f(G, p) * p * (1 - p) * stats.beta.pdf(p, a, b), 0, 1)[0]
        errs.append(abs(cf - q))
out["beta_closed_form_maxerr"] = float(max(errs))
# T2: exact discrete quantile vs continuous bound at the E-T2 hardest stratum
z, d, K = 0.8862, 0.1, 8
n_cont = math.log(1 / d) / -math.log(z)
out["T2_hardest"] = {
    "n_cont": n_cont,
    "N_cont": K * n_cont,
    "N_disc": K * math.ceil(n_cont),
    "P_fail_at_floor_groups": z ** math.floor(n_cont),
    "P_fail_at_ceil": z ** math.ceil(n_cont),
}
# Clopper-Pearson one-sided lower bound at X=m stays < 1
out["CP_lower_X_eq_m_m32_alpha05"] = float(stats.beta.ppf(0.05, 32, 1))

# ---------------- artifacts ----------------
t1 = json.loads((RES / "t1_ci_coverage_qwen3-8b_train_n512_r32_s42.json").read_text())
sh = [r for r in t1["rows"] if not r["correlated"]]
co = [r for r in t1["rows"] if r["correlated"]]
rng = lambda xs: [min(xs), max(xs)]
out["E-T1"] = {
    "grid_G": sorted({r["G"] for r in t1["rows"]}),
    "grid_M": sorted({r["M"] for r in t1["rows"]}),
    "batches_per_cell": t1["batches_per_cell"],
    "shuffled_wald_m>=64": rng([r["wald_coverage"] for r in sh if r["M"] >= 64]),
    "shuffled_wilson_all": rng([r["wilson_coverage"] for r in sh]),
    "sorted_wald_all": rng([r["wald_coverage"] for r in co]),
    "sorted_wald_m>=64": rng([r["wald_coverage"] for r in co if r["M"] >= 64]),
    "sorted_wilson_all": rng([r["wilson_coverage"] for r in co]),
}
cf = json.loads((RES / "t1_correlated_fix_qwen3-8b_train_n512_r32_s42_G8_M32.json").read_text())
out["E-T1_fix"] = {
    k: cf[k]
    for k in (
        "G",
        "M",
        "B",
        "coverage_correlated_vs_local_truth",
        "coverage_correlated_vs_global_truth",
        "coverage_stratified_vs_global_truth",
    )
}
t2 = json.loads((RES / "t2_floor_qwen3-8b_train_n512_r32_s42_G8.json").read_text())
out["E-T2"] = {
    "meta": {k: v for k, v in t2.items() if k != "rows"},
    "rows": [
        {
            "stratum": r["stratum"],
            "n": r["n_prompts"],
            "p": r["mean_p_hat"],
            "zvf": r["zvf_G"],
            "trials": r["trials"],
            "bounds": [
                (
                    b["delta"],
                    b["reliability_budget_rollouts"],
                    b["observed_q_at_1_minus_delta"],
                    b["fit_ratio_obs_over_model"],
                )
                for b in r["bounds"]
            ],
        }
        for r in t2["rows"]
    ],
}
t3 = json.loads((RES / "t3_gstar_v2_qwen3-8b_train_n512_r32_s42.json").read_text())
out["E-T3a"] = [
    (r["stratum"], r["empirical_J_argmax"], r["t3_gstar_prediction"]) for r in t3["rows"]
]
bd = json.loads((RES / "bandit_realloc_qwen3-8b_train_n512_r32_s42_G8.json").read_text())["results"]
ts, orc, uni, hard = (
    bd[k]["informative_groups_per_1k_rollouts"]
    for k in ("thompson-gu", "static-oracle", "uniform", "static-hardest")
)
out["E-B"] = {
    "thompson": ts,
    "oracle": orc,
    "ratio": ts / orc,
    "vs_uniform": ts / uni - 1,
    "vs_hardest": ts / hard,
}


def series(path):
    d = json.loads(path.read_text())
    return d


out["E-R2b_files"] = sorted(p.name for p in TR.glob("er2b_*.json"))
out["E-P4_files"] = sorted(p.name for p in TR.glob("p4uncap_*.json"))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "verify.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
