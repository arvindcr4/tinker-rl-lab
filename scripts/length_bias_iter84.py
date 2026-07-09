#!/usr/bin/env python3
"""Iter 84 -- Pillar 4 (Length Bias / Dr.GRPO): frequency-domain + long-memory decomposition.

Iter 80 settled the boundedness question (length is mean-reverting, not unit-root).
Iter 84 attacks the orthogonal question: *how* does length co-vary with reward in
the time-frequency domain?  Three new measurements per (algorithm, seed, task):

  (A) Hurst exponent via Detrended Fluctuation Analysis (DFA)
      Long-memory exponent H in [0,1].  H>0.5 -> persistent, H<0.5 -> anti-persistent.
      Comparable to but more robust on short series than the AR(1) phi of iter 80.

  (B) Magnitude-squared spectral coherence |C_xy(f)|^2 between L_t and R_t
      Welch-style periodogram coherence at 4 frequency bins (very-low < 0.05,
      low 0.05-0.15, mid 0.15-0.30, high 0.30+ cycles/step).  Tests *at which
      frequencies* length and reward share variance.

  (C) Granger-style F-test from a VAR(2) of (L_t, R_t)
      Reports F_LR (does L_t-1, L_t-2 predict R_t?) and F_RL (reverse).  Each
      series is standardised to N(0,1) before fitting so the F's are comparable.

Sharp falsifiable reading of Dr.GRPO (Liu et al. 2025, arXiv:2503.20783): GRPO's
length normaliser should *couple* length to reward more tightly than Dr.GRPO's
removal does, so GRPO should show (i) higher coherence at the low-frequency band
where Dr.GRPO's mean correction is most active, (ii) a stronger L->R Granger
F, and (iii) higher Hurst exponent (longer memory in L_t).

Inputs : drgrpo_vs_grpo.json (arithmetic, Qwen2.5-0.5B, 5 seeds x 2 algos)
         drgrpo_gsm8k_cot_full.json (GSM8K CoT, Qwen2.5-1.5B, 3 seeds x 2 algos)
Outputs: length_bias_iter84_{perrun,coherence,granger,hurst}.tsv
         length_bias_iter84_meta.json
Stdlib + numpy + scipy (signal, stats).
"""
import json, os, math
import numpy as np
from scipy import signal as sps
from scipy import stats as scs

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
B_BOOT = 2000
RNG = np.random.default_rng(84)

# --------------------------------------------------------------------- DFA ----
def _dfa_one(x, scales):
    """Detrended fluctuation analysis: F(s) ~ s^H.  Returns H."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    n = len(y)
    fs = []
    ss = []
    for s in scales:
        if s < 4 or s > n // 2:
            continue
        nw = n // s
        rms = []
        for k in range(nw):
            seg = y[k * s:(k + 1) * s]
            t = np.arange(s)
            a, b = np.polyfit(t, seg, 1)
            trend = a * t + b
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fs.append(np.sqrt(np.mean(rms)))
        ss.append(s)
    if len(fs) < 4:
        return float("nan")
    lf = np.log(np.array(fs))
    ls = np.log(np.array(ss))
    H, _ = np.polyfit(ls, lf, 1)
    return float(H)


def hurst_dfa(L, scales=None):
    if scales is None:
        scales = np.unique(np.logspace(np.log10(4), np.log10(max(5, len(L) // 2)), 12).astype(int))
    return _dfa_one(L, scales)


# ----------------------------------------------------- spectral coherence ----
def ms_coherence_bands(L, R, fs_bands):
    """Magnitude-squared coherence averaged in each frequency band."""
    L = np.asarray(L, float); R = np.asarray(R, float)
    n = len(L)
    if n < 8:
        return [float("nan")] * len(fs_bands)
    nperseg = max(4, min(8, n // 2))
    noverlap = nperseg // 2
    f, cxy = sps.coherence(L, R, fs=1.0, nperseg=nperseg, noverlap=noverlap)
    out = []
    for lo, hi in fs_bands:
        mask = (f >= lo) & (f < hi)
        out.append(float(cxy[mask].mean()) if mask.any() else float("nan"))
    return out


# --------------------------------------------------- Granger F from VAR(2) ----
def granger_f(L, R, lag=2):
    """F-stat for H0: L_t does NOT Granger-cause R_t, plus reverse.
    Standardise both to N(0,1) so the F's are scale-free and comparable.
    Returns (F_lr, F_rl, p_lr, p_rl, r2_lr, r2_rl).
    """
    L = (np.asarray(L, float) - np.asarray(L, float).mean()) / (np.asarray(L, float).std() + 1e-9)
    R = (np.asarray(R, float) - np.asarray(R, float).mean()) / (np.asarray(R, float).std() + 1e-9)
    n = len(L); T = n - lag
    if T <= lag + 2:
        return (float("nan"),) * 6
    YR = R[lag:]
    YL = L[lag:]
    Xrest = np.column_stack([np.ones(T), R[lag - 1:n - 1], R[lag - 2:n - 2]])  # restricted
    Xfull = np.column_stack([Xrest, L[lag - 1:n - 1], L[lag - 2:n - 2]])  # + L lags
    def _ols(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        return resid, X.shape[1]
    rr, k_r = _ols(Xrest, YR)
    fr, k_f = _ols(Xfull, YR)
    ss_r = float(rr @ rr); ss_f = float(fr @ fr)
    df_num = k_f - k_r
    df_den = T - k_f
    if ss_f <= 0 or df_den <= 0 or df_num <= 0:
        return (float("nan"),) * 6
    F_lr = ((ss_r - ss_f) / df_num) / (ss_f / df_den)
    p_lr = float(1 - scs.f.cdf(F_lr, df_num, df_den))
    # reverse: predict L from R lags
    rl, _ = _ols(Xrest, YL)
    fl, _ = _ols(Xfull, YL)
    ss_rl = float(rl @ rl); ss_fl = float(fl @ fl)
    F_rl = ((ss_rl - ss_fl) / df_num) / (ss_fl / df_den)
    p_rl = float(1 - scs.f.cdf(F_rl, df_num, df_den))
    r2_lr = 1 - ss_f / ss_r if ss_r > 0 else float("nan")
    r2_rl = 1 - ss_fl / ss_rl if ss_rl > 0 else float("nan")
    return F_lr, F_rl, p_lr, p_rl, r2_lr, r2_rl


# ---------------------------------------------------------- load + analyse ----
def load_runs():
    out = []
    d1 = json.load(open(os.path.join(RES, "drgrpo_vs_grpo.json")))
    for r in d1["runs"]:
        sl = r["step_log"]
        L = np.array([s["mean_comp_len"] for s in sl])
        R = np.array([s["mean_reward"] for s in sl])
        out.append(dict(task="arithmetic", algo=r["algo"], seed=r["seed"], L=L, R=R,
                        last10=r.get("last10_avg", float("nan"))))
    d2 = json.load(open(os.path.join(RES, "drgrpo_gsm8k_cot_full.json")))
    for r in d2["runs"]:
        sl = r["step_log"]
        L = np.array([s["mean_comp_len"] for s in sl])
        R = np.array([s["mean_reward"] for s in sl])
        out.append(dict(task="gsm8k_cot", algo=r["algo"], seed=r["seed"], L=L, R=R,
                        last10=r.get("last10_avg", float("nan"))))
    return out


def analyse():
    runs = load_runs()
    bands = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50)]
    band_names = ["vlow", "low", "mid", "high"]
    perrun = []
    coh = []  # long-form coherence
    grg = []  # granger long-form
    hurst_summary = []
    for run in runs:
        L, R = run["L"], run["R"]
        H_L = hurst_dfa(L)
        H_R = hurst_dfa(R)
        cohs = ms_coherence_bands(L, R, bands)
        F_lr, F_rl, p_lr, p_rl, r2_lr, r2_rl = granger_f(L, R, lag=2)
        # speed & level stats for context
        rho, p_rho = scs.spearmanr(L, R)
        perrun.append(dict(
            task=run["task"], algo=run["algo"], seed=run["seed"], n=int(len(L)),
            hurst_L=H_L, hurst_R=H_R,
            coh_vlow=cohs[0], coh_low=cohs[1], coh_mid=cohs[2], coh_high=cohs[3],
            F_lr=F_lr, F_rl=F_rl, p_lr=p_lr, p_rl=p_rl,
            r2_lr=r2_lr, r2_rl=r2_rl,
            spearman_rho=float(rho), spearman_p=float(p_rho),
            L_mean=float(L.mean()), L_std=float(L.std()),
            R_mean=float(R.mean()), R_std=float(R.std()),
            last10_acc=run["last10"],
        ))
        for i, nm in enumerate(band_names):
            coh.append(dict(task=run["task"], algo=run["algo"], seed=run["seed"],
                            band=nm, lo=bands[i][0], hi=bands[i][1], coh=cohs[i]))
        grg.append(dict(task=run["task"],algo=run["algo"], seed=run["seed"],
                        direction="L_to_R", F=F_lr, p=p_lr, r2=r2_lr))
        grg.append(dict(task=run["task"], algo=run["algo"], seed=run["seed"],
                        direction="R_to_L", F=F_rl, p=p_rl, r2=r2_rl))
    # aggregate per (task, algo) means
    keys_h = ["hurst_L", "hurst_R", "coh_vlow", "coh_low", "coh_mid", "coh_high",
              "F_lr", "F_rl", "spearman_rho", "L_mean", "L_std", "R_mean", "R_std"]
    for task in sorted({r["task"] for r in perrun}):
        for algo in sorted({r["algo"] for r in perrun}):
            sel = [r for r in perrun if r["task"] == task and r["algo"] == algo]
            if not sel:
                continue
            row = dict(task=task, algo=algo, n_seeds=len(sel))
            for k in keys_h:
                vals = [x[k] for x in sel if np.isfinite(x[k])]
                row[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[k + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            hurst_summary.append(row)
    # seed-paired bootstrap: Dr.GRPO - GRPO per task on each measurement
    paired = []
    for task in sorted({r["task"] for r in perrun}):
        seeds_g = {r["seed"]: r for r in perrun if r["task"] == task and r["algo"] == "grpo"}
        seeds_d = {r["seed"]: r for r in perrun if r["task"] == task and r["algo"] == "dr_grpo"}
        common = sorted(set(seeds_g) & set(seeds_d))
        if not common:
            continue
        for k in ["hurst_L", "hurst_R", "coh_vlow", "coh_low", "coh_mid", "coh_high",
                  "F_lr", "F_rl", "spearman_rho"]:
            diffs = np.array([seeds_d[s][k] - seeds_g[s][k] for s in common])
            diffs = diffs[np.isfinite(diffs)]
            if len(diffs) < 2:
                continue
            obs = float(diffs.mean())
            # bootstrap CI
            n = len(diffs)
            boots = np.empty(B_BOOT)
            for b in range(B_BOOT):
                idx = RNG.integers(0, n, n)
                boots[b] = diffs[idx].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            # two-sided p against H0: mean=0
            p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
            paired.append(dict(task=task, key=k, n_pairs=int(n), mean_diff=obs,
                               ci_lo=float(lo), ci_hi=float(hi), p=p))
    return perrun, coh, grg, hurst_summary, paired


def write_tsv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(_fmt(r.get(h)) for h in header) + "\n")


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if v is None:
        return ""
    return str(v)


def main():
    perrun, coh, grg, summary, paired = analyse()
    base = os.path.join(RES, "length_bias_iter84_")
    write_tsv(base + "perrun.tsv", perrun,
              ["task", "algo", "seed", "n", "hurst_L", "hurst_R",
               "coh_vlow", "coh_low", "coh_mid", "coh_high",
               "F_lr", "F_rl", "p_lr", "p_rl", "r2_lr", "r2_rl",
               "spearman_rho", "spearman_p", "L_mean", "L_std", "R_mean", "R_std",
               "last10_acc"])
    write_tsv(base + "coherence.tsv", coh,
              ["task", "algo", "seed", "band", "lo", "hi", "coh"])
    write_tsv(base + "granger.tsv", grg,
              ["task", "algo", "seed", "direction", "F", "p", "r2"])
    write_tsv(base + "summary.tsv", summary,
              ["task", "algo", "n_seeds"] +
              [k + "_mean" for k in
               ["hurst_L", "hurst_R", "coh_vlow", "coh_low", "coh_mid", "coh_high",
                "F_lr", "F_rl", "spearman_rho", "L_mean", "L_std", "R_mean", "R_std"]] +
              [k + "_se" for k in
               ["hurst_L", "hurst_R", "coh_vlow", "coh_low", "coh_mid", "coh_high",
                "F_lr", "F_rl", "spearman_rho", "L_mean", "L_std", "R_mean", "R_std"]])
    write_tsv(base + "paired.tsv", paired,
              ["task", "key", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p"])
    meta = dict(iteration=84, pillar="P4-LengthBias",
                method="DFA-Hurst + magnitude-squared spectral coherence + VAR(2) Granger F",
                inputs=["drgrpo_vs_grpo.json", "drgrpo_gsm8k_cot_full.json"],
                n_runs=len(perrun),
                n_per_task={t: sum(1 for r in perrun if r["task"] == t) for t in
                            sorted({r["task"] for r in perrun})},
                band_definitions={"vlow": "0.00-0.05 cyc/step", "low": "0.05-0.15",
                                  "mid": "0.15-0.30", "high": "0.30-0.50"},
                rng_seed=84)
    with open(base + "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[iter84] wrote", len(perrun), "perrun,", len(summary), "summary,", len(paired), "paired")


if __name__ == "__main__":
    main()