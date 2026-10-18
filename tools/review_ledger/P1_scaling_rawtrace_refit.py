"""P1_scaling Deep Review (2026-10-18): raw-trace recomputation.

Replaces analyses that previously ran on traces *synthesised* from six summary
statistics (scaling_law_iter33/37/41/45/49/53/65 and the Pass@1 table of
berkeley/sweagent_passk_aci.py) with fits on the released per-step reward
traces in platform_hybrid/experiments/tinker-runs/results/*_gsm8k_*.json.

Conventions
  * Every saturation-type fit constrains R_max to [0, 1] (rewards are means of
    binary correctness) and lambda to [1e-4, 10].
  * Unit of analysis for cross-scale statistics = anchor (one single-seed run
    per checkpoint); df = n - 2 for OLS slopes.  Permutation p-values are exact
    (all n! orderings) for n <= 8, else 20,000 Monte Carlo shuffles.
  * Gaussian log-likelihood for AICc; k counts every free mean parameter plus
    the residual variance.

Outputs: platform_hybrid/experiments/results/review_2026-10-18/P1_scaling/
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "platform_hybrid/experiments/tinker-runs/results"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P1_scaling"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261018

# name: (file, N_total_B, N_active_B, arch)
ANCHORS = {
    "Qwen3.5-4B": ("scale_gsm8k_qwen3.5-4b.json", 4.0, 4.0, "dense"),
    "Qwen3-8B": ("scale_gsm8k_qwen3-8b.json", 8.0, 8.0, "dense"),
    "Llama-3.1-8B-Instruct": ("scale_gsm8k_llama-8b-inst.json", 8.0, 8.0, "dense"),
    "Qwen3-32B": ("scale_gsm8k_qwen3-32b.json", 32.0, 32.0, "dense"),
    "Qwen3.5-27B": ("scale_gsm8k_qwen3.5-27b.json", 27.0, 27.0, "dense"),
    "gpt-oss-20B": ("arch_gsm8k_gpt-oss-20b.json", 20.0, 3.6, "moe"),
    "Qwen3-30B-MoE": ("moe_gsm8k_qwen3-30b-moe.json", 30.0, 3.0, "moe"),
    "Qwen3-30B-MoE-Inst": ("moe_gsm8k_qwen3-30b-inst.json", 30.0, 3.0, "moe"),
    "DeepSeek-V3.1": ("frontier_gsm8k_deepseek-v3.1.json", 685.0, 37.0, "moe"),
    "Nemotron-120B": ("frontier_gsm8k_nemotron-120b.json", 120.0, 12.0, "moe"),
    "Qwen3-235B-MoE": ("frontier_gsm8k_qwen3-235b.json", 235.0, 22.0, "moe"),
    "Kimi-K2-Thinking": ("arch_gsm8k_kimi-k2.json", 1000.0, 32.0, "moe"),
}
CORE5 = ["Qwen3.5-4B", "Qwen3-8B", "Llama-3.1-8B-Instruct", "DeepSeek-V3.1", "Nemotron-120B"]
LAM_LO, LAM_HI = 1e-4, 10.0


def load():
    rows = {}
    for name, (fn, nt, na, arch) in ANCHORS.items():
        d = json.loads((TRACES / fn).read_text())
        y = np.asarray(d["reward_trace"], float)
        src = "console-partial" if str(d.get("run_id", "")).startswith("partial") else "script-json"
        rows[name] = dict(
            file=fn,
            N_tot=nt,
            N_act=na,
            arch=arch,
            y=y,
            src=src,
            group=d.get("group"),
            lr=d.get("lr"),
            rank=d.get("rank"),
            steps_cfg=d.get("steps"),
            steps_completed=d.get("steps_completed"),
        )
    return rows


def sat2(t, rmax, lam):
    return rmax * (1 - np.exp(-lam * t))


def fit2(t, y):
    best = None
    for p0 in ([0.8, 1.0], [max(y.mean(), 0.05), 5.0], [0.5, 0.1]):
        try:
            p, _ = curve_fit(sat2, t, y, p0=p0, bounds=([0, LAM_LO], [1.0, LAM_HI]), maxfev=20000)
        except Exception:
            continue
        rss = float(np.sum((y - sat2(t, *p)) ** 2))
        if best is None or rss < best[1]:
            best = (p, rss)
    return best


def sat3(t, c, u, lam):
    rmax = c + (1 - c) * u  # enforces c <= R_max <= 1
    return c + (rmax - c) * (1 - np.exp(-lam * t))


def fit3(t, y):
    best = None
    for c0 in (0.0, 0.25, 0.5, 0.75, float(y[:5].mean())):
        for lam0 in (0.05, 0.5, 3.0):
            for u0 in (0.1, 0.5, 0.9):
                r = least_squares(
                    lambda p: sat3(t, *p) - y,
                    [min(max(c0, 0), 1), u0, lam0],
                    bounds=([0, 0, LAM_LO], [1, 1, LAM_HI]),
                )
                rss = float(np.sum(r.fun**2))
                if best is None or rss < best[1]:
                    best = (r.x, rss)
    return best


def aicc(rss, n, k_mean):
    k = k_mean + 1  # + residual variance
    rss = max(rss, 1e-12)
    aic = n * math.log(rss / n) + 2 * k
    return aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")


def piecewise(t, y, tpk):
    """Eq. (5): saturation up to empirical peak step, linear decay after, continuous."""

    def f(p):
        rmax, lam, g = p
        base = rmax * (1 - np.exp(-lam * np.minimum(t, tpk)))
        return np.where(t <= tpk, base, rmax * (1 - np.exp(-lam * tpk)) - g * (t - tpk))

    best = None
    for p0 in ([0.8, 1.0, 0.0], [0.5, 5.0, 0.01], [0.9, 0.2, -0.01]):
        r = least_squares(lambda p: f(p) - y, p0, bounds=([0, LAM_LO, -0.2], [1.0, LAM_HI, 0.2]))
        rss = float(np.sum(r.fun**2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    return best


def hinge3(t, y):
    """Continuous 3-segment OLS, >=2 points per segment (same design as iter65)."""
    n = len(t)
    best = None
    for i in range(2, n - 3):
        for j in range(i + 2, n - 1):
            X = np.column_stack([np.ones(n), t, np.maximum(0, t - t[i]), np.maximum(0, t - t[j])])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ b) ** 2))
            if best is None or sse < best[0]:
                best = (sse, i, j, b)
    sse, i, j, b = best
    m1, m2, m3 = b[1], b[1] + b[2], b[1] + b[2] + b[3]
    return dict(i=i, j=j, m1=m1, m2=m2, m3=m3, c1=(i + 1) / n, c2=(j + 1) / n)


def ols(x, y):
    r = stats.linregress(x, y)
    return r.slope, r.stderr, r.intercept, r.rvalue**2, r.pvalue


def perm_slope_p(x, y, rng, mc=20000):
    x, y = np.asarray(x), np.asarray(y)
    obs = abs(np.polyfit(x, y, 1)[0])
    n = len(x)
    if n <= 8:
        perms = itertools.permutations(range(n))
        cnt = tot = 0
        for p in perms:
            tot += 1
            cnt += abs(np.polyfit(x, y[list(p)], 1)[0]) >= obs - 1e-12
        return cnt / tot, "exact"
    cnt = sum(abs(np.polyfit(x, rng.permutation(y), 1)[0]) >= obs - 1e-12 for _ in range(mc))
    return (cnt + 1) / (mc + 1), f"MC{mc}"


def perm_group_p(vals, labels, one_sided=True):
    """Exact two-sample permutation on mean difference (moe - dense)."""
    vals = np.asarray(vals)
    labels = np.asarray(labels)
    n, k = len(vals), int(labels.sum())
    obs = vals[labels == 1].mean() - vals[labels == 0].mean()
    cnt = tot = 0
    for comb in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(comb)] = True
        d = vals[m].mean() - vals[~m].mean()
        tot += 1
        cnt += (d >= obs - 1e-12) if one_sided else (abs(d) >= abs(obs) - 1e-12)
    return obs, cnt / tot, tot


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (
                        (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}")
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in r.items()
                }
            )


def main():
    rng = np.random.default_rng(SEED)
    A = load()
    summary = {}

    # ---- (1) per-anchor descriptive + bounded 2p fit
    per = []
    for name, a in A.items():
        y = a["y"]
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        k3 = max(1, n // 3)
        row = dict(
            model=name,
            arch=a["arch"],
            N_tot=a["N_tot"],
            N_act=a["N_act"],
            src=a["src"],
            T=n,
            group=a["group"],
            lr=a["lr"],
            R1=float(y[0]),
            mean=float(y.mean()),
            peak=float(y.max()),
            peak_step=int(np.argmax(y)) + 1,
            zero_frac=float(np.mean(y == 0)),
            early=float(y[:k3].mean()),
            late=float(y[-k3:].mean()),
            last5=float(y[-5:].mean()) if n >= 5 else float("nan"),
        )
        if n >= 6:
            p, rss = fit2(t, y)
            row.update(
                Rmax2p=float(p[0]),
                lam2p=float(p[1]),
                lam_at_bound=bool(p[1] > 0.99 * LAM_HI),
                rss2p=rss,
            )
        else:
            row.update(Rmax2p=float("nan"), lam2p=float("nan"), lam_at_bound="", rss2p=float("nan"))
        per.append(row)
    write("rawtrace_anchor_table.tsv", per)
    P = {r["model"]: r for r in per}

    # ---- (2) core-5 offset (3p) joint bounded fit vs 2p; piecewise Eq. (5)
    rows3, rowspw = [], []
    for name in CORE5:
        y = A[name]["y"]
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        p2, rss2 = fit2(t, y)
        p3, rss3 = fit3(t, y)
        c, u, lam = p3
        rmax3 = c + (1 - c) * u
        a2, a3 = aicc(rss2, n, 2), aicc(rss3, n, 3)
        rows3.append(
            dict(
                model=name,
                n=n,
                Rmax2p=p2[0],
                lam2p=p2[1],
                rss2p=rss2,
                AICc2p=a2,
                c3p=c,
                Rmax3p=rmax3,
                lam3p=lam,
                t80_3p=(-math.log(0.2) / lam),
                rss3p=rss3,
                AICc3p=a3,
                dAICc=a3 - a2,
                nested_ok=bool(rss3 <= rss2 + 1e-9),
            )
        )
        tpk = float(np.argmax(y) + 1)
        ppw, rsspw = piecewise(t, y, tpk)
        # k: Rmax, lambda, gamma + t_peak (selected from data) -> 4 mean params
        rowspw.append(
            dict(
                model=name,
                n=n,
                Rmax_sat=p2[0],
                Rmax_pw=ppw[0],
                lam_pw=ppw[1],
                t_peak=tpk,
                gamma=ppw[2],
                rss_pw=rsspw,
                dAICc_pw_vs_sat_k4=aicc(rsspw, n, 4) - a2,
                dAICc_pw_vs_sat_k3=aicc(rsspw, n, 3) - a2,
                F_p=float(stats.f.sf(max(rss2 - rsspw, 0) / (rsspw / (n - 3)), 1, n - 3)),
            )
        )
    write("core5_offset_fit_bounded.tsv", rows3)
    xs5 = np.log10([A[m]["N_tot"] for m in CORE5])
    s3 = []
    for yname, yv in (
        ("log10_t80_3p", np.log10([q["t80_3p"] for q in rows3])),
        ("Rmax3p", np.array([q["Rmax3p"] for q in rows3])),
    ):
        s, se, ic, r2, p = ols(xs5, yv)
        s3.append(
            dict(y=yname, n=5, df=3, slope=s, se=se, p_t=p, spearman=stats.spearmanr(xs5, yv)[0])
        )
    write("core5_offset_fit_crossscale.tsv", s3)
    write("core5_piecewise_bounded.tsv", rowspw)

    # ---- (3) Pass@1-style step bootstrap on the RAW traces (replaces synthetic draws)
    pk = []
    for name in CORE5:
        y = A[name]["y"]
        bs = rng.choice(y, size=(20000, len(y)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pk.append(
            dict(
                model=name,
                n=len(y),
                Rmax2p=P[name]["Rmax2p"],
                p_hat=float(y.mean()),
                lo95=float(lo),
                hi95=float(hi),
                width=float(hi - lo),
            )
        )
    write("core5_step_bootstrap.tsv", pk)
    pairs = []
    for a, b in itertools.combinations(pk, 2):
        overlap = not (a["hi95"] < b["lo95"] or b["hi95"] < a["lo95"])
        pairs.append(
            dict(a=a["model"], b=b["model"], gap=abs(a["p_hat"] - b["p_hat"]), overlap=overlap)
        )
    write("core5_step_bootstrap_pairs.tsv", pairs)

    # ---- (4) cross-scale regressions (anchor = unit)
    reg = []
    long7 = [m for m in P if P[m]["src"] == "script-json"]
    pools = {
        "core5": CORE5,
        "long7": long7,
        "all12": list(P),
        "long7_minus_Nemotron": [m for m in long7 if m != "Nemotron-120B"],
        "all12_minus_Nemotron": [m for m in P if m != "Nemotron-120B"],
    }
    for pool, ms in pools.items():
        for xname in ("N_tot", "N_act"):
            for yname in ("mean", "Rmax2p", "R1", "late"):
                xs = np.log10([P[m][xname] for m in ms])
                ys = np.array([P[m][yname] for m in ms], float)
                ok = np.isfinite(ys)
                if ok.sum() < 4:
                    continue
                s, se, ic, r2, p = ols(xs[ok], ys[ok])
                pp, how = perm_slope_p(xs[ok], ys[ok], rng)
                rho, _ = stats.spearmanr(xs[ok], ys[ok])
                tcrit = stats.t.ppf(0.975, ok.sum() - 2)
                reg.append(
                    dict(
                        pool=pool,
                        x=f"log10_{xname}",
                        y=yname,
                        n=int(ok.sum()),
                        df=int(ok.sum() - 2),
                        slope=s,
                        se=se,
                        ci_lo=s - tcrit * se,
                        ci_hi=s + tcrit * se,
                        r2=r2,
                        p_t=p,
                        p_perm=pp,
                        perm=how,
                        spearman=rho,
                    )
                )
    # compute proxy: C = N * T * 512 tokens (billions of parameter-tokens), total and active
    for pool in ("long7", "all12"):
        ms = pools[pool]
        for xname in ("N_tot", "N_act"):
            xs = np.log10([P[m][xname] * P[m]["T"] * 512 for m in ms])
            ys = np.array([P[m]["mean"] for m in ms])
            s, se, ic, r2, p = ols(xs, ys)
            pp, how = perm_slope_p(xs, ys, rng)
            tcrit = stats.t.ppf(0.975, len(ms) - 2)
            reg.append(
                dict(
                    pool=pool,
                    x=f"log10_C_{xname}",
                    y="mean",
                    n=len(ms),
                    df=len(ms) - 2,
                    slope=s,
                    se=se,
                    ci_lo=s - tcrit * se,
                    ci_hi=s + tcrit * se,
                    r2=r2,
                    p_t=p,
                    p_perm=pp,
                    perm=how,
                    spearman=stats.spearmanr(xs, ys)[0],
                )
            )
    write("cross_scale_regressions.tsv", reg)

    # ---- (5) MoE vs dense on mean reward (exact permutation, one-sided moe>dense)
    grp = []
    for pool, ms in (
        ("all12", list(P)),
        ("all12_minus_Nemotron", pools["all12_minus_Nemotron"]),
        ("long7", long7),
        ("long7_minus_Nemotron", pools["long7_minus_Nemotron"]),
    ):
        vals = [P[m]["mean"] for m in ms]
        lab = [1 if P[m]["arch"] == "moe" else 0 for m in ms]
        obs, p1, tot = perm_group_p(vals, lab, True)
        _, p2, _ = perm_group_p(vals, lab, False)
        grp.append(
            dict(
                pool=pool,
                n_moe=sum(lab),
                n_dense=len(lab) - sum(lab),
                gap_moe_minus_dense=obs,
                p_one_sided=p1,
                p_two_sided=p2,
                partitions=tot,
            )
        )
    write("moe_vs_dense_permutation.tsv", grp)

    # ---- (6) three-phase checks on raw long traces
    tp = []
    for m in long7:
        y = A[m]["y"]
        t = np.arange(1, len(y) + 1, dtype=float)
        h = hinge3(t, y)
        tmpl = (abs(h["m1"]) < 0.020) and (h["m2"] >= 0.040) and (abs(h["m3"]) < 0.015)
        r = P[m]
        tp.append(
            dict(
                model=m,
                T=len(y),
                peak_step=r["peak_step"],
                peak_frac=(r["peak_step"] - 1) / (len(y) - 1),
                late_minus_early=r["late"] - r["early"],
                m1=h["m1"],
                m2=h["m2"],
                m3=h["m3"],
                c1=h["c1"],
                c2=h["c2"],
                template=tmpl,
                collapse=(
                    r["peak"] >= 0.4 and r["last5"] < 0.4 * r["peak"] and r["zero_frac"] >= 0.30
                ),
            )
        )
    write("long7_three_phase.tsv", tp)

    # ---- (7) exogenous pre-update indicator R(1) >= 0.5 vs log10 N (iter133 AICc convention:
    #      k = number of mean parameters, Gaussian RSS; response = bounded 2p R_max where T>=6,
    #      else trace mean, exactly as scaling_law_iter133.py builds its R_max proxy)
    def aicc_k(rss, n, k):
        return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

    ex = []
    for pool, ms in (("core5", CORE5), ("long7", long7), ("all12", list(P))):
        yv = np.array([P[m]["Rmax2p"] if np.isfinite(P[m]["Rmax2p"]) else P[m]["mean"] for m in ms])
        xs = np.log10([P[m]["N_tot"] for m in ms])
        z = np.array([1.0 if P[m]["R1"] >= 0.5 else 0.0 for m in ms])
        n = len(ms)

        def rss_of(X):
            b, *_ = np.linalg.lstsq(X, yv, rcond=None)
            return float(np.sum((yv - X @ b) ** 2))

        one = np.ones(n)
        r_int = rss_of(one[:, None])
        r_n = rss_of(np.column_stack([one, xs]))
        r_z = rss_of(np.column_stack([one, z]))
        r_nz = rss_of(np.column_stack([one, xs, z]))
        ex.append(
            dict(
                pool=pool,
                n=n,
                n_R1_high=int(z.sum()),
                high=";".join(m for m, zz in zip(ms, z) if zz),
                AICc_intercept=aicc_k(r_int, n, 1),
                AICc_log10N=aicc_k(r_n, n, 2),
                AICc_R1ind=aicc_k(r_z, n, 2),
                AICc_log10N_plus_R1ind=aicc_k(r_nz, n, 3),
                R2_log10N=1 - r_n / r_int,
                R2_R1ind=1 - r_z / r_int,
            )
        )
    write("exogenous_R1_indicator_aicc.tsv", ex)

    # ---- (8) iter61 ZVF strata (step-level ZVF = P(R=0)+P(R=1)) on the long traces
    st = []
    for m in long7:
        y = A[m]["y"]
        z = float(np.mean(y == 0) + np.mean(y == 1))
        st.append(
            dict(
                model=m,
                step_zvf=z,
                stratum="HIGH" if z >= 0.1 else "LOW",
                lam2p=P[m]["lam2p"],
                lam_at_bound=P[m]["lam_at_bound"],
                rmse2p=math.sqrt(P[m]["rss2p"] / P[m]["T"]),
            )
        )
    write("long7_zvf_strata.tsv", st)

    summary["n_lam_at_bound_long7"] = sum(1 for m in long7 if P[m]["lam_at_bound"] is True)
    summary["template_pass_long7"] = int(sum(bool(r["template"]) for r in tp))
    summary["collapse_long7"] = [r["model"] for r in tp if r["collapse"]]
    summary["p1_peakfrac_gt_0.10"] = sum(r["peak_frac"] > 0.10 for r in tp)
    summary["p2_late_gt_early"] = sum(r["late_minus_early"] > 0 for r in tp)
    summary["unique_zero_gt_0.5_mean_lt_0.2_all12"] = [
        m for m in P if P[m]["zero_frac"] > 0.5 and P[m]["mean"] < 0.2
    ]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
