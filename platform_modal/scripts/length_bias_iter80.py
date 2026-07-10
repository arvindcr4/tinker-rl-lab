#!/usr/bin/env python3
"""Iter 80 -- Pillar 4 (Length Bias / Dr.GRPO): Ornstein-Uhlenbeck equilibrium-length
and unit-root falsification of the "unbounded length inflation" claim.

The Dr.GRPO paper (Liu et al. 2025, arXiv:2503.20783) argues GRPO's response-level
length normalization (dividing the advantage by |o_i|) injects an optimization bias
that *inflates* response length without bound, and that Dr.GRPO removes it. A sharp,
falsifiable reading: under GRPO the per-step length series L_t should carry a stochastic
trend (unit root, phi ~ 1 -> no finite equilibrium), whereas Dr.GRPO's length should be
mean-reverting to a finite equilibrium mu with reversion speed theta > 0.

We model each run's mean-completion-length series as a discrete Ornstein-Uhlenbeck /
AR(1) process with a constant:
    L_t = c + phi * L_{t-1} + eps_t
and derive, per run:
    phi          persistence
    mu = c/(1-phi)             long-run equilibrium length  (finite iff phi<1)
    theta = -ln(phi)           continuous mean-reversion rate (if phi in (0,1))
    half_life = ln(0.5)/ln(phi) steps to close half the gap to mu
    sigma        residual std (volatility)
    df_stat = gamma_hat/se(gamma_hat)   Dickey-Fuller unit-root stat from
              Delta L_t = alpha + gamma * L_{t-1} + eps ,  gamma = phi-1
Then a seed-paired bootstrap compares GRPO vs Dr.GRPO on Delta(mu) and Delta(phi).

Inputs : platform_hybrid/experiments/results/drgrpo_vs_grpo.json      (arithmetic, Qwen2.5-0.5B, 5+5 seeds)
         platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json (GSM8K CoT, Qwen2.5-1.5B, 3+3 seeds)
Outputs: platform_hybrid/experiments/results/length_bias_iter80_{perrun,summary,unitroot,paired}.tsv
         platform_hybrid/experiments/results/length_bias_iter80_meta.json
Stdlib + numpy only.
"""
import json, os, math
import numpy as np

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
BURN = 2          # drop initial transient steps before the AR(1) fit
B_BOOT = 2000
SEED = 80
# Dickey-Fuller (constant, no trend) approx critical values, N~25-40
DF_CRIT = {"1%": -3.58, "5%": -2.93, "10%": -2.60}


def ols(X, y):
    """Return beta, se(beta), residuals for y = X beta."""
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    s2 = float(resid @ resid) / dof
    cov = s2 * XtXi
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, resid


def ar1_ou(L):
    """Fit L_t = c + phi L_{t-1} + eps; return OU params + DF unit-root stat."""
    L = np.asarray(L, float)
    if len(L) < 6:
        return None
    y = L[1:]
    x = L[:-1]
    X = np.column_stack([np.ones_like(x), x])
    beta, se, resid = ols(X, y)
    c, phi = float(beta[0]), float(beta[1])
    sigma = float(np.std(resid, ddof=1))
    mu = c / (1.0 - phi) if abs(1.0 - phi) > 1e-9 else float("nan")
    if 0.0 < phi < 1.0:
        theta = -math.log(phi)
        half_life = math.log(0.5) / math.log(phi)
    else:
        theta = float("nan")
        half_life = float("nan")
    # Dickey-Fuller: Delta L = alpha + gamma L_{t-1} + eps ; gamma = phi-1
    dy = y - x
    Xd = np.column_stack([np.ones_like(x), x])
    bd, sed, _ = ols(Xd, dy)
    gamma, se_gamma = float(bd[1]), float(sed[1])
    df_stat = gamma / se_gamma if se_gamma > 0 else float("nan")
    return dict(phi=phi, c=c, mu=mu, theta=theta, half_life=half_life,
                sigma=sigma, df_stat=df_stat, gamma=gamma,
                L0=float(L[0]), Lend=float(L[-1]), n=len(L))


def load_runs(fname):
    d = json.load(open(os.path.join(RES, fname)))
    out = []
    for r in d["runs"]:
        steps = r["step_log"]
        L = [s["mean_comp_len"] for s in steps]
        R = [s["mean_reward"] for s in steps]
        out.append(dict(algo=r["algo"], seed=r["seed"], model=r.get("model", ""),
                        L=L, R=R))
    return out


def analyse(runs, task):
    rows = []
    for r in runs:
        L = r["L"][BURN:]
        fit = ar1_ou(L)
        if fit is None:
            continue
        # reward trend over same window (sign of the reward drift)
        Rw = np.asarray(r["R"][BURN:], float)
        t = np.arange(len(Rw))
        r_slope = float(np.polyfit(t, Rw, 1)[0]) if len(Rw) > 2 else float("nan")
        rows.append(dict(task=task, algo=r["algo"], seed=r["seed"], **fit,
                         r_slope=r_slope))
    return rows


def paired_bootstrap(rows, task, key):
    """Seed-paired GRPO - Dr.GRPO difference on `key`, bootstrap over seed pairs."""
    g = {r["seed"]: r[key] for r in rows if r["task"] == task and r["algo"] == "grpo"}
    dr = {r["seed"]: r[key] for r in rows if r["task"] == task and r["algo"] == "dr_grpo"}
    seeds = sorted(set(g) & set(dr))
    diffs = np.array([g[s] - dr[s] for s in seeds], float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return None
    rng = np.random.default_rng(SEED)
    boots = np.array([np.mean(rng.choice(diffs, len(diffs), replace=True))
                      for _ in range(B_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min(np.mean(boots > 0), np.mean(boots < 0))
    return dict(task=task, key=key, n_pairs=len(diffs), mean_diff=float(diffs.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p=float(min(1.0, p)))


def wr(path, header, rows, fmt):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r)) + "\n")
    print("wrote", os.path.relpath(path, W), f"({len(rows)} rows)")


def main():
    tasks = [("arithmetic", "drgrpo_vs_grpo.json"),
             ("gsm8k_cot", "drgrpo_gsm8k_cot_full.json")]
    perrun = []
    for task, fname in tasks:
        perrun += analyse(load_runs(fname), task)

    def g(x, n=4):
        return "nan" if (isinstance(x, float) and math.isnan(x)) else (
            f"{x:.{n}f}" if isinstance(x, float) else str(x))

    # per-run
    hp = ["task", "algo", "seed", "n", "phi", "mu", "theta", "half_life",
          "sigma", "df_stat", "L0", "Lend", "r_slope"]
    wr(os.path.join(RES, "length_bias_iter80_perrun.tsv"), hp, perrun,
       lambda r: [str(r["task"]), str(r["algo"]), str(r["seed"]), str(r["n"]),
                  g(r["phi"]), g(r["mu"], 2), g(r["theta"]), g(r["half_life"], 2),
                  g(r["sigma"], 3), g(r["df_stat"], 3), g(r["L0"], 2),
                  g(r["Lend"], 2), g(r["r_slope"], 4)])

    # summary per (task, algo)
    summ = []
    for task, _ in tasks:
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in perrun if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            def m(k):
                v = [r[k] for r in sub if math.isfinite(r[k])]
                return float(np.mean(v)) if v else float("nan")
            summ.append(dict(task=task, algo=algo, n=len(sub),
                             phi=m("phi"), mu=m("mu"), theta=m("theta"),
                             half_life=m("half_life"), sigma=m("sigma"),
                             df_stat=m("df_stat"), r_slope=m("r_slope")))
    hs = ["task", "algo", "n", "phi", "mu", "theta", "half_life", "sigma",
          "df_stat", "r_slope"]
    wr(os.path.join(RES, "length_bias_iter80_summary.tsv"), hs, summ,
       lambda r: [r["task"], r["algo"], str(r["n"]), g(r["phi"]), g(r["mu"], 2),
                  g(r["theta"]), g(r["half_life"], 2), g(r["sigma"], 3),
                  g(r["df_stat"], 3), g(r["r_slope"], 4)])

    # unit-root verdict per (task, algo): fraction of seeds rejecting unit root at 5%
    ur = []
    for task, _ in tasks:
        for algo in ("grpo", "dr_grpo"):
            sub = [r for r in perrun if r["task"] == task and r["algo"] == algo]
            if not sub:
                continue
            dfs = [r["df_stat"] for r in sub if math.isfinite(r["df_stat"])]
            rej5 = sum(1 for v in dfs if v < DF_CRIT["5%"])
            rej1 = sum(1 for v in dfs if v < DF_CRIT["1%"])
            ur.append(dict(task=task, algo=algo, n=len(dfs),
                           mean_df=float(np.mean(dfs)) if dfs else float("nan"),
                           frac_reject_5pct=rej5 / len(dfs) if dfs else float("nan"),
                           frac_reject_1pct=rej1 / len(dfs) if dfs else float("nan"),
                           crit_5pct=DF_CRIT["5%"]))
    hu = ["task", "algo", "n", "mean_df", "frac_reject_5pct", "frac_reject_1pct",
          "crit_5pct"]
    wr(os.path.join(RES, "length_bias_iter80_unitroot.tsv"), hu, ur,
       lambda r: [r["task"], r["algo"], str(r["n"]), g(r["mean_df"], 3),
                  g(r["frac_reject_5pct"], 2), g(r["frac_reject_1pct"], 2),
                  g(r["crit_5pct"], 2)])

    # seed-paired GRPO vs Dr.GRPO
    paired = []
    for task, _ in tasks:
        for key in ("mu", "phi", "half_life", "sigma"):
            res = paired_bootstrap(perrun, task, key)
            if res:
                paired.append(res)
    hpr = ["task", "key", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p"]
    wr(os.path.join(RES, "length_bias_iter80_paired.tsv"), hpr, paired,
       lambda r: [r["task"], r["key"], str(r["n_pairs"]), g(r["mean_diff"], 4),
                  g(r["ci_lo"], 4), g(r["ci_hi"], 4), g(r["p"], 4)])

    meta = dict(iter=80,
                task="Pillar 4 (Length Bias / Dr.GRPO): OU equilibrium-length + "
                     "Dickey-Fuller unit-root falsification of unbounded length inflation",
                inputs=[t[1] for t in tasks], burn=BURN, b_boot=B_BOOT,
                df_crit=DF_CRIT, model="AR(1)/OU on mean_comp_len level series",
                citation="Liu et al. 2025, arXiv:2503.20783 (Dr.GRPO)")
    json.dump(meta, open(os.path.join(RES, "length_bias_iter80_meta.json"), "w"),
              indent=1)
    print("wrote length_bias_iter80_meta.json")

    # console verdict
    print("\n=== UNIT-ROOT VERDICT ===")
    for r in ur:
        print(f"{r['task']:11s} {r['algo']:8s}  mean DF={r['mean_df']:.2f}  "
              f"reject unit-root @5%: {r['frac_reject_5pct']*100:.0f}% of seeds "
              f"(crit {DF_CRIT['5%']})")
    print("\n=== PAIRED GRPO - Dr.GRPO ===")
    for r in paired:
        star = "*" if r["p"] < 0.05 else " "
        print(f"{r['task']:11s} d{r['key']:9s} = {r['mean_diff']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.3f}{star}")


if __name__ == "__main__":
    main()
