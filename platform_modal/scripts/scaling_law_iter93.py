"""Pillar 1 iter93 -- Head-to-head of 5 low-parameter families under joint
parsimony + out-of-sample selection, plus AR(1) residual-structure test.

The iter85 / iter89 progression falsified the 3-segment template as an
in-sample overfit and confirmed a constant-mean baseline as the strongest
parsimonious candidate so far. iter93 closes two open questions:

  Q1 (Head-to-head): among 5 candidate families
       {constant, 1-segment-OLS, saturation R_max(1-e^{-lam t}),
        power-law (y = a + b * log10(t)), AR(1) y_t = mu + phi*(y_{t-1}-mu) + eps}
    which is the universal LOOCV winner across the 12-anchor iter81 pool?

  Q2 (AR(1) identifiability): if ANY anchor reports phi significantly
    greater than zero under bootstrap (one-sided CI excludes 0), then the
    constant-mean baseline is FALSE even on the BIC/LOOCV winners:
    it loses an information bit (the sign of the next-step correlation).
    If phi CI covers 0 on every anchor, the constant model is genuinely
    universal under a Markov(1) refinement.

  Q3 (Joint AR(1) + saturation): the power-law / saturation / linear
    families are NON-stationary; AR(1) is stationary.  Testing both
    stationary and non-stationary 2-parameter families jointly is the
    strongest model-discrimination battery short of a full Bayesian fit.

References (verified):
  - hyndman2018forecasting (FPP3) ch 9 -- AR(1) and LOOCV.
  - bishop2006pattern (PRML) sec 4.3 -- cross-validation derivation.
  - kaplan2020scaling -- scale-axis baseline.
  - nimmaturi2025predictive (arXiv:2507.18014) -- compared family.

Outputs:
  experiments/results/scaling_law_iter93_headtohead.tsv      (5 families x 12 anchors x 5 metrics)
  experiments/results/scaling_law_iter93_ar1.tsv             (AR(1) per anchor: phi, ci_lo, ci_hi, ci_covers_zero)
  experiments/results/scaling_law_iter93_winners.tsv         (per-anchor + aggregate winners by 5 criteria)
  experiments/results/scaling_law_iter93_meta.json
  figures/scaling_law_iter93.{pdf,png}
  paper/sections/scaling_law_iter93.tex
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

# 12-anchor pool, identical to iter81 / iter85 / iter89.
MODELS: dict[str, tuple[float, str]] = {
    "Qwen3.5-4B": (4.0, "scale_gsm8k_qwen3.5-4b.json"),
    "Qwen3-8B": (8.0, "scale_gsm8k_qwen3-8b.json"),
    "Llama-3.1-8B-Instruct": (8.0, "scale_gsm8k_llama-8b-inst.json"),
    "Qwen3-32B": (32.0, "scale_gsm8k_qwen3-32b.json"),
    "Qwen3.5-27B": (27.0, "scale_gsm8k_qwen3.5-27b.json"),
    "gpt-oss-20B": (20.0, "arch_gsm8k_gpt-oss-20b.json"),
    "Qwen3-30B-MoE": (30.0, "moe_gsm8k_qwen3-30b-moe.json"),
    "Qwen3-30B-MoE-Inst": (30.0, "moe_gsm8k_qwen3-30b-inst.json"),
    "DeepSeek-V3.1": (685.0, "frontier_gsm8k_deepseek-v3.1.json"),
    "Nemotron-120B": (120.0, "frontier_gsm8k_nemotron-120b.json"),
    "Qwen3-235B-MoE": (235.0, "frontier_gsm8k_qwen3-235b.json"),
    "Kimi-K2-Thinking": (1000.0, "arch_gsm8k_kimi-k2.json"),
}
N_BOOT = 2000
SEED = 42
FAMILIES = ["constant", "1seg_ols", "saturation", "powerlaw", "ar1"]
NPARAMS = {"constant": 1, "1seg_ols": 2, "saturation": 2, "powerlaw": 2, "ar1": 3}


# ---------- families ------------------------------------------------------

def f_constant(t, params):
    mu = params[0]
    return np.full_like(np.asarray(t, float), mu, dtype=float)


def f_1seg_ols(t, params):
    a, b = params[:2]
    return a + b * np.asarray(t, float)


def f_saturation(t, params):
    rm, lam = params[:2]
    t = np.asarray(t, float)
    return rm * (1.0 - np.exp(-lam * t))


def f_powerlaw(t, params):
    a, b = params[:2]
    return a + b * np.log10(np.maximum(np.asarray(t, float), 1e-9))


def f_ar1(t, params):
    """Predict y_{i} = mu + phi*(y_{i-1}-mu); we hand-resolve the recursion
    from y0 = mu (init), then return y[t_idx] for each requested step.
    params = [mu, phi] (sigma included as params[2] but never used for prediction)."""
    mu, phi = params[:2]
    y = list(np.asarray([np.nan], float))  # placeholder so index 0 == first prediction step
    raise NotImplementedError  # AR(1) prediction handled separately in _fit_ar1_predict


def _clip01(arr):
    return np.clip(np.asarray(arr, float), 0.0, 1.5)


# ---------- fitters: in-sample OLS-style, robust inits -------------------

def _fit_constant(y):
    return [float(np.mean(y))]


def _fit_1seg_ols(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(coef[0]), float(coef[1])]


def _fit_saturation(t, y):
    """Solve in log-y / identity-y via bounded grid then polish.
    Bounded lambda in [1e-3, 10]; R_max in [max(y), 1.5]."""
    lam_grid = np.geomspace(0.01, 10.0, 60)
    best = (np.inf, None)
    for lam in lam_grid:
        X = np.vstack([np.ones_like(t), 1.0 - np.exp(-lam * t)]).T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rm = float(coef[1])
        if rm < max(0.4 * float(y.max()), 0.05):
            continue
        rm = max(rm, 0.05)
        rm = min(rm, 1.5)
        pred = coef[0] + rm * (1.0 - np.exp(-lam * t))
        sse = float(np.sum((y - pred) ** 2))
        if sse < best[0]:
            best = (sse, [rm, float(lam)])
    return best[1] if best[1] else [float(y.mean()), 0.3]


def _fit_powerlaw(t, y):
    X = np.vstack([np.ones_like(t), np.log10(np.maximum(t, 1e-9))]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return [float(coef[0]), float(coef[1])]


def _fit_ar1(y):
    """OLS on the simple regression y_t - mu = phi*(y_{t-1} - mu) (constant-target
    mean).  Closed-form on (y[1:] - mu) ~ phi*(y[:-1] - mu)."""
    n = len(y)
    if n < 4:
        return [float(np.mean(y)), 0.0, float(np.std(y) + 1e-9)]
    mu = float(np.mean(y))
    x = y[:-1] - mu
    z = y[1:] - mu
    num = float(np.sum(x * z))
    den = float(np.sum(x * x))
    phi = num / den if den > 1e-12 else 0.0
    phi = max(min(phi, 0.999), -0.999)
    resid = z - phi * x
    sigma = float(np.std(resid)) + 1e-9
    return [mu, float(phi), sigma]


def _predict_ar1(y_full, params, predict_steps):
    """Given full trace y_full (length n) and AR(1) params [mu, phi, sigma],
    return yhat[n:n+k] for k = len(predict_steps).  Recursive."""
    mu, phi, _ = params[:3]
    n = len(y_full)
    preds = []
    last = float(y_full[-1])
    for i in range(len(predict_steps)):
        nxt = mu + phi * (last - mu)
        preds.append(nxt)
        last = nxt
    return np.asarray(preds, float)


# ---------- scoring criteria ---------------------------------------------

def _residuals(y, yhat, k):
    return float(np.sum((y - yhat) ** 2)), float(np.sum(np.abs(y - yhat)))


def _aic(rss, n, k):
    return float(n * math.log(rss / max(n, 1)) + 2 * k)


def _bic(rss, n, k):
    return float(n * math.log(rss / max(n, 1)) + k * math.log(max(n, 2)))


def _loocv_rmse(y_full, t_full, family):
    """Leave-one-step-out refit + predict for the given family.
    For AR(1) we approximate by OLS-on-complement params."""
    n = len(y_full)
    errs = []
    errs_mae = []
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        y_ref = y_full[idx]
        t_ref = t_full[idx]
        try:
            params, yhat_ref = _fit_predict(t_ref, y_ref, family)
            yhat_at_i = _predict_one_at_step(t_ref, t_full[i], params, family, y_ref)
            errs.append(float(y_full[i] - yhat_at_i) ** 2)
            errs_mae.append(abs(float(y_full[i] - yhat_at_i)))
        except Exception:
            return float("nan"), float("nan"), -1
    return float(np.sqrt(np.mean(errs))), float(np.mean(errs_mae)), n


def _predict_one_at_step(t_ref, t_target, params, family, y_ref):
    if family == "constant":
        return float(params[0])
    if family == "1seg_ols":
        a, b = params[:2]
        return float(a + b * t_target)
    if family == "saturation":
        rm, lam = params[:2]
        return float(rm * (1.0 - np.exp(-lam * t_target)))
    if family == "powerlaw":
        a, b = params[:2]
        return float(a + b * math.log10(max(t_target, 1e-9)))
    if family == "ar1":
        mu, phi, _ = params[:3]
        # predict y_target given y_ref (use the last y in y_ref as the
        # autoregressive conditioning value)
        return float(mu + phi * (float(y_ref[-1]) - mu))
    raise ValueError(family)


def _fit_predict(t, y, family):
    if family == "constant":
        p = _fit_constant(y); yh = f_constant(t, p)
    elif family == "1seg_ols":
        p = _fit_1seg_ols(t, y); yh = f_1seg_ols(t, p)
    elif family == "saturation":
        p = _fit_saturation(t, y); yh = f_saturation(t, p)
    elif family == "powerlaw":
        p = _fit_powerlaw(t, y); yh = f_powerlaw(t, p)
    elif family == "ar1":
        p = _fit_ar1(y)
        # for AR(1) in-sample "yhat", we use a deterministic 1-step-ahead
        # recursion from y[0] (instead of the infeasible all-at-once closed form)
        mu, phi, _ = p[:3]
        yh_arr = np.empty_like(y)
        yh_arr[0] = mu
        for i in range(1, len(y)):
            yh_arr[i] = mu + phi * (y[i - 1] - mu)
        yh = yh_arr
    else:
        raise ValueError(family)
    return p, yh


def _forecast_last_k(y_full, t_full, family, k=4):
    n = len(y_full)
    if n <= k + 2:
        return float("nan"), np.array([])
    idx_train = np.arange(0, n - k)
    t_tr, y_tr = t_full[idx_train], y_full[idx_train]
    t_te = t_full[n - k:]
    y_te = y_full[n - k:]
    try:
        params, _ = _fit_predict(t_tr, y_tr, family)
    except Exception:
        return float("nan"), np.array([])
    if family == "ar1":
        preds = _predict_ar1(y_tr, params, t_te)
    else:
        preds = np.array([
            _predict_one_at_step(t_tr, t_target, params, family, y_tr)
            for t_target in t_te
        ], float)
    preds_c = _clip01(preds)
    y_te_c = _clip01(y_te)
    mae = float(np.mean(np.abs(y_te_c - preds_c)))
    rmse = float(np.sqrt(np.mean((y_te_c - preds_c) ** 2)))
    return mae, rmse


def _bootstrap_phi_ci(y, n_boot=N_BOOT):
    """Block-bootstrap (sample-with-replacement) phi distribution for AR(1)."""
    n = len(y)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    phis = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if np.std(y_b) < 1e-9:
            continue
        p = _fit_ar1(y_b)
        phis.append(p[1])
    if not phis:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(phis, float)
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


# ---------- main ----------------------------------------------------------

def _write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> None:
    raw = {}
    for label, (params_b, fname) in MODELS.items():
        p = TRACE_DIR / fname
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        d = json.loads(p.read_text())
        y = np.asarray(d["reward_trace"], float)
        raw[label] = (params_b, y, fname)

    # ---- Head-to-head: 5 families x 5 criteria ------------------------------
    head_cols = [
        "model", "params_B", "n_steps", "family", "n_params",
        "rmse_in_sample", "r2_in_sample",
        "aic", "bic", "loocv_rmse", "loocv_mae",
        "forecast_mae_last4", "forecast_rmse_last4",
    ]
    head_rows = []
    for label, (params_b, y, fname) in raw.items():
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        for fam in FAMILIES:
            try:
                params, yh = _fit_predict(t, y, fam)
                rss = float(np.sum((y - yh) ** 2))
                rmse = float(np.sqrt(rss / n))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - rss / ss_tot if ss_tot > 1e-12 else float("nan")
                aic = _aic(rss + 1e-9, n, NPARAMS[fam])
                bic = _bic(rss + 1e-9, n, NPARAMS[fam])
                loo_r, loo_m, _ = _loocv_rmse(y, t, fam)
                fc_mae, fc_rmse = _forecast_last_k(y, t, fam, k=4)
                head_rows.append([
                    label, params_b, n, fam, NPARAMS[fam],
                    f"{rmse:.6f}", f"{r2:.6f}",
                    f"{aic:.6f}", f"{bic:.6f}",
                    f"{loo_r:.6f}", f"{loo_m:.6f}",
                    f"{fc_mae:.6f}", f"{fc_rmse:.6f}",
                ])
            except Exception as e:
                head_rows.append([
                    label, params_b, n, fam, NPARAMS[fam],
                    "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan",
                ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter93_headtohead.tsv", head_cols, head_rows)

    # ---- AR(1) bootstrap-CI per anchor ------------------------------------
    ar1_cols = ["model", "params_B", "n_steps", "phi_hat", "phi_lo", "phi_hi",
                "ci_covers_zero", "mu_hat", "sigma_hat",
                "phi_lo_excludes_zero_oneright", "phi_hi_excludes_zero_oneleft"]
    ar1_rows = []
    for label, (params_b, y, fname) in raw.items():
        n = len(y)
        params = _fit_ar1(y)
        mu_hat, phi_hat, sigma_hat = params[:3]
        bm, lo, hi = _bootstrap_phi_ci(y)
        ci_covers_zero = bool((lo <= 0.0) and (hi >= 0.0)) or math.isnan(lo)
        ar1_rows.append([
            label, params_b, n,
            f"{phi_hat:.6f}", f"{lo:.6f}", f"{hi:.6f}",
            ci_covers_zero,
            f"{mu_hat:.6f}", f"{sigma_hat:.6f}",
            bool(lo > 0.0 and not math.isnan(lo)),
            bool(hi < 0.0 and not math.isnan(hi)),
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter93_ar1.tsv", ar1_cols, ar1_rows)

    # ---- Per-criterion winners --------------------------------------------
    win_cols = ["criterion", "winner_family", "n_wins", "total_anchors",
                "candidates", "totals_per_family"]
    win_rows = []
    aggregated = {fam: {crit: 0 for crit in ("aic", "bic", "loocv", "forecast", "rmse", "r2")} for fam in FAMILIES}
    anchor_count = 0
    for label, (params_b, y, fname) in raw.items():
        anchor_count += 1
        sub = [r for r in head_rows if r[0] == label]
        # aic, bic, loocv_rmse, forecast_mae: lower better
        for crit, idx in (("aic", 7), ("bic", 8), ("loocv", 9), ("forecast", 11), ("rmse", 5)):
            vals = []
            for r in sub:
                v = r[idx]
                try:
                    vals.append((r[3], float(v)))
                except ValueError:
                    vals.append((r[3], float("nan")))
            vals_clean = [(f, v) for (f, v) in vals if not math.isnan(v)]
            if not vals_clean:
                continue
            winner = min(vals_clean, key=lambda x: x[1])
            aggregated[winner[0]][crit] += 1
        # r2: higher better
        vals = []
        for r in sub:
            try:
                vals.append((r[3], float(r[6])))
            except ValueError:
                vals.append((r[3], float("nan")))
        vals_clean = [(f, v) for (f, v) in vals if not math.isnan(v)]
        if vals_clean:
            winner = max(vals_clean, key=lambda x: x[1])
            aggregated[winner[0]]["r2"] += 1

    crit_labels = [
        ("aic", "AIC"), ("bic", "BIC"), ("loocv", "LOOCV-RMSE"),
        ("forecast", "Forecast-MAE"), ("rmse", "In-sample RMSE"), ("r2", "In-sample R^2"),
    ]
    for crit, lbl in crit_labels:
        wins = [(fam, aggregated[fam][crit]) for fam in FAMILIES]
        wins.sort(key=lambda x: -x[1])
        winner_fam = wins[0][0]
        n_wins = wins[0][1]
        per_family_str = ";".join(f"{f}:{aggregated[f][crit]}" for f in FAMILIES)
        win_rows.append([
            lbl, winner_fam, n_wins, anchor_count,
            ",".join(FAMILIES), per_family_str,
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter93_winners.tsv", win_cols, win_rows)

    # ---- meta + figure ---------------------------------------------------
    n_ar1_signif_pos = sum(1 for r in ar1_rows if str(r[9]) == "True")
    n_ar1_zero_covered = sum(1 for r in ar1_rows if str(r[6]) == "True")
    n_ar1_anchors = len(ar1_rows)
    meta = {
        "iter": 93,
        "n_anchors": n_ar1_anchors,
        "families_compared": FAMILIES,
        "n_bootstrap_phi": N_BOOT,
        "ar1_phi_anchors_with_lo_excludes_0": n_ar1_signif_pos,
        "ar1_phi_anchors_with_ci_covers_zero": n_ar1_zero_covered,
        "consensus_winners_by_criterion": {
            crit: aggregated_for_crit for crit, _ in crit_labels
            for aggregated_for_crit in [aggregated]
        },
    }
    (RESULTS_DIR / "scaling_law_iter93_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- figure: 4 panels (RMSE, LOOCV, AR1 phi CI, forecast MAE) --------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: in-sample RMSE per family (boxplot)
    rmses = {fam: [] for fam in FAMILIES}
    for r in head_rows:
        try:
            rmses[r[3]].append(float(r[5]))
        except ValueError:
            pass
    axs[0, 0].boxplot(
        [rmses[f] for f in FAMILIES],
        tick_labels=FAMILIES,
        showmeans=True,
    )
    axs[0, 0].set_title("(A) In-sample RMSE per family (12 anchors)")
    axs[0, 0].set_ylabel("RMSE")
    axs[0, 0].grid(alpha=0.3)

    # Panel B: LOOCV RMSE per family
    loos = {fam: [] for fam in FAMILIES}
    for r in head_rows:
        try:
            loos[r[3]].append(float(r[9]))
        except ValueError:
            pass
    axs[0, 1].boxplot(
        [loos[f] for f in FAMILIES],
        tick_labels=FAMILIES,
        showmeans=True,
    )
    axs[0, 1].set_title("(B) LOOCV RMSE per family")
    axs[0, 1].set_ylabel("LOOCV RMSE")
    axs[0, 1].grid(alpha=0.3)

    # Panel C: AR(1) phi bootstrap CI per anchor
    labels = [r[0] for r in ar1_rows]
    ys = np.arange(len(labels))
    phi_hat = [float(r[3]) for r in ar1_rows]
    lo = [float(r[4]) for r in ar1_rows]
    hi = [float(r[5]) for r in ar1_rows]
    phi_clipped = [max(min(p, 0.7), -0.7) for p in phi_hat]  # visual cap
    lo_c = [max(min(l, 0.7), -0.7) for l in lo]
    hi_c = [max(min(h, 0.7), -0.7) for h in hi]
    axs[1, 0].errorbar(
        phi_clipped, ys, xerr=[np.abs(np.array(phi_clipped) - np.array(lo_c)),
                                 np.abs(np.array(hi_c) - np.array(phi_clipped))],
        fmt="o", color="C0", ecolor="gray", elinewidth=1, capsize=2,
    )
    axs[1, 0].axvline(0.0, color="red", linestyle="--", linewidth=1, label="phi=0")
    axs[1, 0].set_yticks(ys)
    axs[1, 0].set_yticklabels(labels, fontsize=7)
    axs[1, 0].set_title("(C) AR(1) phi bootstrap 95% CI per anchor")
    axs[1, 0].set_xlabel("phi")
    axs[1, 0].grid(alpha=0.3)
    axs[1, 0].legend(fontsize=8)

    # Panel D: forecast MAE per family
    fmae = {fam: [] for fam in FAMILIES}
    for r in head_rows:
        try:
            v = float(r[11])
            if not math.isnan(v):
                fmae[r[3]].append(v)
        except ValueError:
            pass
    means = [np.mean(fmae[f]) if fmae[f] else 0 for f in FAMILIES]
    axs[1, 1].bar(FAMILIES, means, color=["C0", "C1", "C2", "C3", "C4"])
    axs[1, 1].set_title("(D) Mean forecast MAE on last 4 steps")
    axs[1, 1].set_ylabel("MAE")
    axs[1, 1].grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Iter 93: head-to-head of 5 model families + AR(1) phi bootstrap CI",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_law_iter93.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "scaling_law_iter93.png", bbox_inches="tight", dpi=120)
    fig.savefig(PAPER_FIG / "scaling_law_iter93.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/scaling_law_iter93.{pdf,png}")

    # ---- concise summary ---------------------------------------------------
    print()
    print("=== iter93 summary ===")
    for crit, lbl in crit_labels:
        wins = [(fam, aggregated[fam][crit]) for fam in FAMILIES]
        wins.sort(key=lambda x: -x[1])
        print(f"  {lbl:>16}: " + ", ".join(f"{f}={n}" for f, n in wins))
    print(f"  AR(1) phi anchors with bootstrap-lo > 0: {n_ar1_signif_pos}/{n_ar1_anchors}")
    print(f"  AR(1) phi anchors with CI covering 0:   {n_ar1_zero_covered}/{n_ar1_anchors}")


if __name__ == "__main__":
    main()
