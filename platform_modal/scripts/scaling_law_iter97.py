"""Pillar 1 iter97 -- Extend the iter93 head-to-head with 3 new functional
families (Monod / Michaelis-Menten, Gompertz, piecewise-2-segment-OLS) and
add a variance-scaling regression on top of the constant-fit residuals.

iter93 closed Q1/Q2/Q3: among {constant, 1seg_ols, saturation, powerlaw, AR(1)}
the constant baseline was the LOOCV/BIC/AIC winner in 5-6 of 12 anchors,
with AR(1) capturing 4/12 forecast wins.  iter97 closes Q4/Q5:

  Q4 (Functional-form coverage): does ANY of the 3 new families
       {monod R_max * t / (K + t),         -- 2-param hyperbolic ceiling
        gompertz a * exp(-b * exp(-c * t)),-- 3-param asymmetric S
        pw2seg (c*, a1, b1, a2, b2)}       -- 5-param changepoint OLS
    outperform the constant-mean baseline under LOOCV or AIC?
    This is the "are there ANY more universal shapes?" question.

  Q5 (Variance scaling): for the constant-mean residual
        r_i(t) = y_i(t) - mean(y_i),
    does the per-anchor residual variance sigma_i^2 scale with
    log10(params_B_i)?  If yes, the noise floor of GRPO reward is
    scale-dependent, falsifying the homoscedastic assumption inherited
    from iter81 / iter85 / iter89 / iter93.

References (verified):
  - monod1949growth  -- Michaelis-Menten / Monod kinetics; R_max * t/(K+t).
  - gompertz1825      -- Asymmetric S-curve; widely used in growth modeling.
  - Quandt1958        -- Broken-stick / changepoint regression.
  - kaplan2020scaling -- Scale-axis baseline.

Outputs:
  experiments/results/scaling_law_iter97_headtohead.tsv  (8 families x 12 anchors x 5 metrics)
  experiments/results/scaling_law_iter97_winners.tsv     (per-criterion winners)
  experiments/results/scaling_law_iter97_variance.tsv    (per-anchor sigma^2 + OLS regression)
  experiments/results/scaling_law_iter97_meta.json
  figures/scaling_law_iter97.{pdf,png}
  paper/figures/scaling_law_iter97.pdf
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

# 12-anchor pool, identical to iter81 / iter85 / iter89 / iter93.
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
# 8-family battery: iter93's 5 + iter97's 3 new
FAMILIES = ["constant", "1seg_ols", "saturation", "powerlaw", "ar1",
            "monod", "gompertz", "pw2seg"]
NPARAMS = {"constant": 1, "1seg_ols": 2, "saturation": 2, "powerlaw": 2,
           "ar1": 3, "monod": 2, "gompertz": 3, "pw2seg": 5}
NEW_FAMILIES = ["monod", "gompertz", "pw2seg"]


# ---------- fitters (re-import from iter93 for the 5 baseline families) -

def _fit_constant(y):
    return [float(np.mean(y))]


def _fit_1seg_ols(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(coef[0]), float(coef[1])]


def _fit_saturation(t, y):
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


# ---------- iter97 new families ------------------------------------------

def _fit_monod(t, y):
    """Michaelis-Menten / Monod: R(t) = R_max * t / (K + t).
    Linearizable: 1/R = (K/R_max)/t + 1/R_max.  Fit on the linearized
    form via OLS for (1/R_max, K/R_max), then map back.
    Clip R_max to [0.05, 1.5], K to [1e-3, 1e3]."""
    y = np.asarray(y, float)
    eps = 1e-3
    y_clip = np.clip(np.abs(y), eps, 1.5)
    inv_y = 1.0 / y_clip
    inv_t = 1.0 / np.maximum(np.asarray(t, float), 1e-3)
    A = np.vstack([inv_t, np.ones_like(inv_t)]).T
    coef, *_ = np.linalg.lstsq(A, inv_y, rcond=None)
    # 1/R = (K/R_max)/t + 1/R_max  =>  inv_t coeff = K/R_max, intercept = 1/R_max
    inv_rm = float(coef[1])
    K_over_rm = float(coef[0])
    if inv_rm <= 0:
        return [float(np.mean(y)), 1.0]
    rm = 1.0 / inv_rm
    rm = max(min(rm, 1.5), 0.05)
    K = K_over_rm * rm
    K = max(min(K, 1e3), 1e-3)
    return [float(rm), float(K)]


def _fit_gompertz(t, y):
    """Gompertz: R(t) = a * exp(-b * exp(-c * t)).
    Log-linearize: log R = log a - b * exp(-c * t).
    Search over c on a log-spaced grid; for each c, fit b, a by OLS
    in the linearized form.
    Bounds: a in [0.05, 1.5], b in [1e-3, 30], c in [1e-3, 5]."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    eps = 1e-3
    y_clip = np.clip(np.abs(y), eps, 1.5)
    log_y = np.log(y_clip)
    c_grid = np.geomspace(0.005, 5.0, 40)
    best = (np.inf, None)
    for c in c_grid:
        z = np.exp(-c * t)  # in (0, 1], monotone decreasing
        X = np.vstack([np.ones_like(z), -z]).T
        try:
            coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
        except Exception:
            continue
        log_a = float(coef[0])
        b = float(coef[1])
        a = math.exp(log_a) if log_a < math.log(1.5) else 1.5
        a = max(min(a, 1.5), 0.05)
        b = max(min(b, 30.0), 1e-3)
        pred = a * np.exp(-b * np.exp(-c * t))
        sse = float(np.sum((y - pred) ** 2))
        if sse < best[0]:
            best = (sse, [float(a), float(b), float(c)])
    if best[1] is None:
        return [float(np.mean(y)), 1.0, 0.3]
    return best[1]


def _fit_pw2seg(t, y):
    """Piecewise-2-segment OLS: search for the best changepoint c*,
    then fit two independent OLS lines {a1 + b1 * t for t <= c*,
    a2 + b2 * t for t > c*}.  Returns (c*, a1, b1, a2, b2).
    Minimum 2 points per segment."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    if n < 6:
        # fallback: 1-seg OLS + bogus breakpoint
        a, b = _fit_1seg_ols(t, y)
        return [float(n / 2.0 + 0.5), float(a), float(b), float(a), float(b)]
    best = (np.inf, None)
    # Search over candidate changepoints at half-integer steps between [2, n-2]
    for k in range(2, n - 1):
        idx1 = np.arange(0, k)
        idx2 = np.arange(k, n)
        if len(idx1) < 2 or len(idx2) < 2:
            continue
        t1, y1 = t[idx1], y[idx1]
        t2, y2 = t[idx2], y[idx2]
        a1, b1 = _fit_1seg_ols(t1, y1)
        a2, b2 = _fit_1seg_ols(t2, y2)
        pred = np.empty_like(y)
        pred[idx1] = a1 + b1 * t1
        pred[idx2] = a2 + b2 * t2
        sse = float(np.sum((y - pred) ** 2))
        if sse < best[0]:
            best = (sse, [float(t[k - 1] + 0.5), float(a1), float(b1),
                          float(a2), float(b2)])
    return best[1] if best[1] else _fit_pw2seg.__wrapped_fallback(t, y) if hasattr(_fit_pw2seg, '__wrapped_fallback') else [float(n / 2.0 + 0.5), float(y.mean()), 0.0, float(y.mean()), 0.0]


# ---------- predictors ---------------------------------------------------

def _predict_monod(t, params):
    rm, K = params[:2]
    t = np.asarray(t, float)
    return rm * t / (K + t)


def _predict_gompertz(t, params):
    a, b, c = params[:3]
    t = np.asarray(t, float)
    return a * np.exp(-b * np.exp(-c * t))


def _predict_pw2seg(t, params):
    c_star, a1, b1, a2, b2 = params[:5]
    t = np.asarray(t, float)
    pred = np.where(t <= c_star, a1 + b1 * t, a2 + b2 * t)
    return pred


def _predict_ar1(y_full, params, predict_steps):
    mu, phi, _ = params[:3]
    preds = []
    last = float(y_full[-1])
    for _ in predict_steps:
        nxt = mu + phi * (last - mu)
        preds.append(nxt)
        last = nxt
    return np.asarray(preds, float)


# ---------- helper: fit_predict dispatcher -------------------------------

def _fit_predict(t, y, family):
    """Returns (params, yhat_full)."""
    if family == "constant":
        p = _fit_constant(y); yh = np.full_like(np.asarray(y, float), p[0])
    elif family == "1seg_ols":
        p = _fit_1seg_ols(t, y); yh = p[0] + p[1] * np.asarray(t, float)
    elif family == "saturation":
        p = _fit_saturation(t, y); rm, lam = p[:2]
        yh = rm * (1.0 - np.exp(-lam * np.asarray(t, float)))
    elif family == "powerlaw":
        p = _fit_powerlaw(t, y); a, b = p[:2]
        yh = a + b * np.log10(np.maximum(np.asarray(t, float), 1e-9))
    elif family == "ar1":
        p = _fit_ar1(y)
        mu, phi, _ = p[:3]
        yh_arr = np.empty_like(y)
        yh_arr[0] = mu
        for i in range(1, len(y)):
            yh_arr[i] = mu + phi * (y[i - 1] - mu)
        yh = yh_arr
    elif family == "monod":
        p = _fit_monod(t, y); yh = _predict_monod(t, p)
    elif family == "gompertz":
        p = _fit_gompertz(t, y); yh = _predict_gompertz(t, p)
    elif family == "pw2seg":
        p = _fit_pw2seg(t, y); yh = _predict_pw2seg(t, p)
    else:
        raise ValueError(family)
    return p, yh


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
        return float(mu + phi * (float(y_ref[-1]) - mu))
    if family == "monod":
        rm, K = params[:2]
        return float(rm * t_target / (K + t_target))
    if family == "gompertz":
        a, b, c = params[:3]
        return float(a * math.exp(-b * math.exp(-c * t_target)))
    if family == "pw2seg":
        c_star, a1, b1, a2, b2 = params[:5]
        return float(a1 + b1 * t_target if t_target <= c_star else a2 + b2 * t_target)
    raise ValueError(family)


# ---------- scoring criteria (mirrors iter93) ----------------------------

def _clip01(arr):
    return np.clip(np.asarray(arr, float), 0.0, 1.5)


def _aic(rss, n, k):
    return float(n * math.log(rss / max(n, 1)) + 2 * k)


def _bic(rss, n, k):
    return float(n * math.log(rss / max(n, 1)) + k * math.log(max(n, 2)))


def _loocv_rmse(y_full, t_full, family):
    n = len(y_full)
    errs = []
    errs_mae = []
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        y_ref = y_full[idx]
        t_ref = t_full[idx]
        try:
            params, _ = _fit_predict(t_ref, y_ref, family)
            yhat_at_i = _predict_one_at_step(t_ref, t_full[i], params, family, y_ref)
            errs.append(float(y_full[i] - yhat_at_i) ** 2)
            errs_mae.append(abs(float(y_full[i] - yhat_at_i)))
        except Exception:
            return float("nan"), float("nan"), -1
    return float(np.sqrt(np.mean(errs))), float(np.mean(errs_mae)), n


def _forecast_last_k(y_full, t_full, family, k=4):
    n = len(y_full)
    if n <= k + 2:
        return float("nan"), float("nan")
    idx_train = np.arange(0, n - k)
    t_tr, y_tr = t_full[idx_train], y_full[idx_train]
    t_te = t_full[n - k:]
    y_te = y_full[n - k:]
    try:
        params, _ = _fit_predict(t_tr, y_tr, family)
    except Exception:
        return float("nan"), float("nan")
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


# ---------- main ---------------------------------------------------------

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

    # ---- Head-to-head: 8 families x 5 criteria -----------------------------
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
    _write_tsv(RESULTS_DIR / "scaling_law_iter97_headtohead.tsv",
               head_cols, head_rows)

    # ---- Per-criterion winners --------------------------------------------
    win_cols = ["criterion", "winner_family", "n_wins", "total_anchors",
                "candidates", "totals_per_family"]
    win_rows = []
    aggregated = {fam: {crit: 0 for crit in ("aic", "bic", "loocv", "forecast", "rmse", "r2")}
                  for fam in FAMILIES}
    anchor_count = 0
    for label, (params_b, y, fname) in raw.items():
        anchor_count += 1
        sub = [r for r in head_rows if r[0] == label]
        # aic, bic, loocv_rmse, forecast_mae: lower better
        for crit, idx in (("aic", 7), ("bic", 8), ("loocv", 9), ("forecast", 11), ("rmse", 5)):
            vals = []
            for r in sub:
                try:
                    vals.append((r[3], float(r[idx])))
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
    _write_tsv(RESULTS_DIR / "scaling_law_iter97_winners.tsv",
               win_cols, win_rows)

    # ---- Variance-scaling test (Q5) --------------------------------------
    # Constant-fit residuals:  r_i(t) = y_i(t) - mean(y_i)
    # sigma_i^2 = Var(r_i).  Regress sigma_i^2 on log10(params_B_i).
    var_cols = ["model", "params_B", "n_steps", "mu_hat",
                "sigma2_constfit", "sigma_constfit",
                "log10_params_B", "is_new_family_winner"]
    var_rows = []
    is_winner_per_anchor = {}
    for label, (params_b, y, fname) in raw.items():
        mu = float(np.mean(y))
        resid = y - mu
        sigma2 = float(np.var(resid))
        is_winner_per_anchor[label] = any(
            r[0] == label and r[3] in NEW_FAMILIES and r[1] in (
                aggregated[r[3]].keys()  # placeholder
            ) for r in head_rows
        )
        # simpler: did ANY new family beat constant on ANY criterion for this anchor?
        new_wins_anchor = False
        for crit, idx in (("aic", 7), ("bic", 8), ("loocv", 9), ("forecast", 11)):
            sub = [r for r in head_rows if r[0] == label]
            const_v = None
            for r in sub:
                if r[3] == "constant":
                    try:
                        const_v = float(r[idx])
                    except ValueError:
                        const_v = float("nan")
            new_vs = [(r[3], float(r[idx])) for r in sub
                      if r[3] in NEW_FAMILIES
                      and not math.isnan(float(r[idx]) if (r[idx] not in ("nan",)) else float("nan"))]
            for fam, v in new_vs:
                if const_v is not None and not math.isnan(const_v) and v < const_v - 1e-6:
                    new_wins_anchor = True
                    break
        is_winner_per_anchor[label] = new_wins_anchor
        var_rows.append([
            label, params_b, len(y), f"{mu:.6f}",
f"{sigma2:.6f}", f"{math.sqrt(max(sigma2, 0.0)):.6f}",
            f"{math.log10(max(params_b, 1e-3)):.6f}",
            new_wins_anchor,
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter97_variance.tsv",
               var_cols, var_rows)

    # OLS regression: sigma^2 = alpha + beta * log10(params_B)
    log_B = np.asarray([math.log10(max(r[1], 1e-3)) for r in var_rows], float)
    sig2 = np.asarray([float(r[4]) for r in var_rows], float)
    A = np.vstack([np.ones_like(log_B), log_B]).T
    coef, *_ = np.linalg.lstsq(A, sig2, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    yhat = alpha + beta * log_B
    ss_res = float(np.sum((sig2 - yhat) ** 2))
    ss_tot = float(np.sum((sig2 - sig2.mean()) ** 2))
    r2_var = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    # bootstrap CI on beta (resample rows)
    rng = np.random.default_rng(SEED)
    n_boot = N_BOOT
    betas = []
    n_anch = len(var_rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n_anch, size=n_anch)
        Ab = A[idx]
        yb = sig2[idx]
        try:
            cb, *_ = np.linalg.lstsq(Ab, yb, rcond=None)
            betas.append(float(cb[1]))
        except Exception:
            continue
    betas = np.asarray(betas, float)
    beta_lo, beta_hi = (float(np.percentile(betas, 2.5)),
                        float(np.percentile(betas, 97.5)))
    beta_excludes_zero = bool(beta_lo > 0.0 or beta_hi < 0.0)

    # ---- meta -----------------------------------------------------------
    meta = {
        "iter": 97,
        "n_anchors": len(raw),
        "families_compared": FAMILIES,
        "new_families": NEW_FAMILIES,
        "consensus_winners_by_criterion": aggregated,
        "variance_scaling": {
            "alpha": alpha,
            "beta": beta,
            "r2": r2_var,
            "beta_lo95": beta_lo,
            "beta_hi95": beta_hi,
            "beta_excludes_zero": beta_excludes_zero,
            "n_bootstrap": n_boot,
            "interpretation": (
                "If beta > 0 (CI excludes 0), per-anchor reward variance GROWS with "
                "log10(params_B) -- larger models are noisier under GRPO. "
                "If beta < 0, larger models are smoother.  If CI covers 0, the "
                "noise floor is approximately scale-invariant."
            ),
        },
        "new_family_wins_anchor_count": sum(1 for v in is_winner_per_anchor.values() if v),
    }
    (RESULTS_DIR / "scaling_law_iter97_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- figure: 4 panels ----------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: in-sample RMSE per family (boxplot over 12 anchors)
    rmses = {fam: [] for fam in FAMILIES}
    for r in head_rows:
        try:
            v = float(r[5])
            if not math.isnan(v):
                rmses[r[3]].append(v)
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
    axs[0, 0].tick_params(axis='x', rotation=30)

    # Panel B: LOOCV RMSE per family
    loos = {fam: [] for fam in FAMILIES}
    for r in head_rows:
        try:
            v = float(r[9])
            if not math.isnan(v):
                loos[r[3]].append(v)
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
    axs[0, 1].tick_params(axis='x', rotation=30)

    # Panel C: per-criterion winners bar chart (grouped)
    crit_labels_short = ["AIC", "BIC", "LOOCV", "Forecast", "RMSE", "R^2"]
    crit_keys = ["aic", "bic", "loocv", "forecast", "rmse", "r2"]
    x = np.arange(len(crit_labels_short))
    width = 0.10
    for i_fam, fam in enumerate(FAMILIES):
        counts = [aggregated[fam][k] for k in crit_keys]
        axs[1, 0].bar(x + i_fam * width, counts, width,
                       label=fam)
    axs[1, 0].set_xticks(x + width * (len(FAMILIES) - 1) / 2)
    axs[1, 0].set_xticklabels(crit_labels_short)
    axs[1, 0].set_ylabel("# anchors won")
    axs[1, 0].set_title("(C) Per-criterion winners across 12 anchors")
    axs[1, 0].legend(fontsize=7, ncol=2)
    axs[1, 0].grid(alpha=0.3, axis='y')

    # Panel D: variance-scaling regression sigma^2 vs log10(params_B)
    axs[1, 1].scatter(log_B, sig2, s=50, alpha=0.7, color="C0",
                      label="anchors")
    xline = np.linspace(log_B.min() - 0.3, log_B.max() + 0.3, 50)
    axs[1, 1].plot(xline, alpha + beta * xline, "r-",
                    label=f"OLS: y = {alpha:.4f} + {beta:.4f} * x  "
                          f"(R^2 = {r2_var:.3f}, beta 95%CI "
                          f"[{beta_lo:.4f},{beta_hi:.4f}])")
    axs[1, 1].set_xlabel("log10(params_B)")
    axs[1, 1].set_ylabel("sigma^2 (constant-fit residual)")
    axs[1, 1].set_title("(D) Variance scaling across model sizes")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(fontsize=7)

    fig.suptitle(
        "Iter 97: 8-family head-to-head (3 new) + variance-scaling test",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_law_iter97.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "scaling_law_iter97.png", bbox_inches="tight", dpi=120)
    fig.savefig(PAPER_FIG / "scaling_law_iter97.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/scaling_law_iter97.{pdf,png}")

    # ---- concise summary -------------------------------------------------
    print()
    print("=== iter97 summary ===")
    print(f"  Anchor pool: {len(raw)}")
    print(f"  Families compared: {FAMILIES}")
    print(f"  Per-criterion winners:")
    for crit, lbl in crit_labels:
        wins = [(fam, aggregated[fam][crit]) for fam in FAMILIES]
        wins.sort(key=lambda x: -x[1])
        print(f"    {lbl:>16}: " + ", ".join(f"{f}={n}" for f, n in wins))
    print(f"  Variance scaling: alpha={alpha:.4f}, beta={beta:.4f}, "
          f"R^2={r2_var:.3f}, beta 95%CI [{beta_lo:.4f},{beta_hi:.4f}]")
    print(f"  beta excludes 0: {beta_excludes_zero}")
    print(f"  New-family beats constant on any criterion: "
          f"{sum(1 for v in is_winner_per_anchor.values() if v)}/{len(raw)} anchors")


if __name__ == "__main__":
    main()