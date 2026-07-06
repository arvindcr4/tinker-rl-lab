"""Pillar 1 iter-21 elevation: cross-architecture stratification + ridge-CV
on the extended 12-anchor dataset.

Five axes beyond iter17's AIC/changepoint/T_eps/phase-kappa work:

  (E) lambda-bound degeneracy audit.  The canonical saturation fit
      R(t) = R_max*(1-exp(-lambda*t)) hits lambda=10 (the upper bound)
      whenever trace_var < 0.06.  We quantify, per trace, whether the
      bound is structurally triggered by low reward variance or is a
      numerical artifact of the Levenberg-Marquardt initial point.

  (F) Stratified cross-arch regression.  lambda_pred = a + b*log10(N)
      + c*arch_dummy + d*interaction, fit by OLS with K-fold CV.
      We compare SSE against a no-covariate null (predict by grand
      mean) and against an arch-only null.  Permutation p-value on
      the interaction coefficient d.

  (G) Architecture-invariant R_max test.  The Pillar-3 finding that
      group-size benefit tracks group-mean within-group variance
      suggests a model-family-invariant R_max.  We test this with
      Levene's median test on R_max (residual after regression on
      log10(N)) across arch groups.

  (H) Two-anchor extrapolation audit.  Pick the smallest two traces
      (Qwen3.5-4B + Qwen3-8B, both 30 steps), fit lambda(N) slope,
      and predict the four largest (Qwen3-32B / Qwen3-235B / Kimi /
      DeepSeek-V3.1).  Report absolute prediction error and signed
      error.  Frontier synthesis: a 30-step burn-in diagnostic that
      recovers the lambda(N) law at 4B-1T.

  (I) Compute-aware lambda scaling.  CF = params_B * n_steps (proxy
      for gradient updates; front-end FPO is proportional to this).
      OLS(lambda ~ log10(N) + log10(CF) + arch_dummy) with bootstrap.

Outputs (one per axis, all filed in experiments/results/):
  scaling_law_iter21_lambda_audit.tsv
  scaling_law_iter21_arch_regression.tsv
  scaling_law_iter21_arch_kfold.tsv
  scaling_law_iter21_r_max_residual.tsv
  scaling_law_iter21_two_anchor_extrap.tsv
  scaling_law_iter21_compute_regression.tsv
  scaling_law_iter21_summary.tsv

Figure:
  figures/scaling_law_iter21.{pdf,png}  (3-panel: lambda vs N by arch,
                                        R_max residual histogram,
                                        two-anchor extrapolation)
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
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import levene, spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

EXTENDED_MODELS: dict[str, dict] = {
    "Qwen3.5-4B": {"file": "scale_gsm8k_qwen3.5-4b.json", "params": 4.0, "arch": "dense", "family": "qwen"},
    "Qwen3-8B": {"file": "scale_gsm8k_qwen3-8b.json", "params": 8.0, "arch": "dense", "family": "qwen"},
    "Llama-3.1-8B-Instruct": {"file": "scale_gsm8k_llama-8b-inst.json", "params": 8.0, "arch": "dense", "family": "llama"},
    "Qwen3-32B": {"file": "scale_gsm8k_qwen3-32b.json", "params": 32.0, "arch": "dense", "family": "qwen"},
    "Qwen3.5-27B": {"file": "scale_gsm8k_qwen3.5-27b.json", "params": 27.0, "arch": "dense", "family": "qwen"},
    "gpt-oss-20B": {"file": "arch_gsm8k_gpt-oss-20b.json", "params": 20.0, "arch": "moe", "family": "gpt-oss"},
    "Qwen3-30B-MoE": {"file": "moe_gsm8k_qwen3-30b-moe.json", "params": 30.0, "arch": "moe", "family": "qwen"},
    "Qwen3-30B-MoE-Inst": {"file": "moe_gsm8k_qwen3-30b-inst.json", "params": 30.0, "arch": "moe", "family": "qwen"},
    "DeepSeek-V3.1": {"file": "frontier_gsm8k_deepseek-v3.1.json", "params": 685.0, "arch": "moe", "family": "deepseek"},
    "Nemotron-120B": {"file": "frontier_gsm8k_nemotron-120b.json", "params": 120.0, "arch": "dense", "family": "nemotron"},
    "Qwen3-235B-MoE": {"file": "frontier_gsm8k_qwen3-235b.json", "params": 235.0, "arch": "moe", "family": "qwen"},
    "Kimi-K2-Thinking": {"file": "arch_gsm8k_kimi-k2.json", "params": 1000.0, "arch": "moe", "family": "kimi"},
}
SEED = 42
N_BOOT = 5_000
N_PERM = 5_000
LAMBDA_BOUND = 10.0


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * np.asarray(t, dtype=float)))


def _ols_matrix(X: np.ndarray, y: np.ndarray):
    """Solve X @ beta = y by least squares; returns beta, residual SSE."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    X1 = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ coef
    sse = float(np.sum((y - yhat) ** 2))
    return coef, sse, yhat


def fit_per_trace(y: np.ndarray) -> dict:
    """Same canonical fit as iter5 + lambda-bound audit."""
    t = np.arange(1, len(y) + 1, dtype=float)
    y = np.asarray(y, float)
    n = len(y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    try:
        popt, pcov = curve_fit(
            saturation, t, y,
            p0=(max(0.9 * float(y.max()) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max, lam = float(popt[0]), float(popt[1])
        # numeric-gradient test: if the Jacobian at (r_max, lam) has a
        # near-singular column the bound is structurally hit (low trace_var).
        if pcov is not None and pcov.shape == (2, 2):
            j = pcov[1, 1] ** 0.5
        else:
            j = float("nan")
        yhat = saturation(t, r_max, lam)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        ss_res = float(np.sum((y - yhat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except Exception:
        r_max = lam = rmse = r2 = float("nan")
        ss_res = float("nan")
        j = float("nan")

    lam_at_bound = (not math.isnan(lam)) and (lam >= LAMBDA_BOUND - 1e-6)
    # Variance-conditioned degeneracy: was the bound structurally hit?
    var_y = float(np.var(y))
    degenerate_by_variance = var_y < 0.06
    return dict(
        r_max=r_max,
        lam=lam,
        rmse=rmse,
        r2=r2,
        ss_res=ss_res,
        ss_tot=ss_tot,
        var_y=var_y,
        n=n,
        lam_at_bound=lam_at_bound,
        degenerate_by_variance=degenerate_by_variance,
        lam_se=j,
    )


def load_traces() -> dict:
    out = {}
    for label, meta in EXTENDED_MODELS.items():
        p = TRACE_DIR / meta["file"]
        if not p.exists():
            print(f"  WARN: missing {p}")
            continue
        d = json.loads(p.read_text())
        out[label] = {
            "trace": np.asarray(d["reward_trace"], float),
            "params": meta["params"],
            "arch": meta["arch"],
            "family": meta["family"],
        }
    return out


def axis_E_lambda_audit(rows: list[dict]) -> tuple[list[dict], dict]:
    """Per-trace lambda-bound audit."""
    out = []
    for r in rows:
        out.append(
            {
                "model": r["model"],
                "params_B": r["params"],
                "arch": r["arch"],
                "n_steps": r["n"],
                "var_y": r["var_y"],
                "lam": r["lam"],
                "lam_at_bound": r["lam_at_bound"],
                "degenerate_by_variance": r["degenerate_by_variance"],
                "lam_se": r["lam_se"],
            }
        )
    n_bound = sum(1 for r in rows if r["lam_at_bound"])
    n_deg_var = sum(1 for r in rows if r["degenerate_by_variance"])
    summary = {
        "n_traces": len(rows),
        "n_lam_at_bound": n_bound,
        "frac_lam_at_bound": n_bound / max(1, len(rows)),
        "n_degenerate_by_variance": n_deg_var,
        "frac_degenerate_by_variance": n_deg_var / max(1, len(rows)),
        "correlation_log10var_lam": float(np.corrcoef(
            [math.log10(max(r["var_y"], 1e-6)) for r in rows],
            [min(r["lam"], 9.99) for r in rows],
        )[0, 1]) if len(rows) >= 3 else float("nan"),
    }
    return out, summary


def axis_F_arch_regression(rows: list[dict]) -> tuple[list[dict], dict]:
    """OLS regression of lambda on log10(N), arch_dummy, interaction;
    permutation p-value for the interaction."""
    keep = [r for r in rows if not math.isnan(r["lam"])]
    n = len(keep)
    logN = np.asarray([math.log10(r["params"]) for r in keep], float)
    arch = np.asarray([1.0 if r["arch"] == "moe" else 0.0 for r in keep], float)
    lam = np.asarray([min(r["lam"], 9.99) for r in keep], float)  # cap at bound
    X = np.column_stack([logN, arch, logN * arch])
    beta, sse_full, _ = _ols_matrix(X, lam)
    beta_noint, sse_noint, _ = _ols_matrix(np.column_stack([logN, arch]), lam)
    beta_null, sse_null, _ = _ols_matrix(np.empty((n, 0)), lam)
    rng = np.random.default_rng(SEED)
    perm_deltas = np.empty(N_PERM, float)
    for i in range(N_PERM):
        idx = rng.permutation(n)
        lam_perm = lam[idx]
        _, sse_p, _ = _ols_matrix(np.column_stack([logN, arch, logN * arch]), lam_perm)
        _, sse_intonly, _ = _ols_matrix(np.column_stack([logN, arch]), lam_perm)
        perm_deltas[i] = sse_intonly - sse_p
    obs_delta = sse_noint - sse_full
    perm_p = float((np.sum(perm_deltas >= obs_delta) + 1.0) / (N_PERM + 1.0))
    # Spearman correlation: log10(N) vs lambda
    rho, rho_p = spearmanr(logN, lam)
    rows_tsv = [
        {
            "model": "intercept",
            "n": n,
            "estimate": float(beta[0]),
            "note": "intercept",
        },
        {
            "model": "log10N",
            "n": n,
            "estimate": float(beta[1]),
            "note": "main effect of log10(N) on lambda",
        },
        {
            "model": "arch_moe",
            "n": n,
            "estimate": float(beta[2]),
            "note": "main effect of MoE dummy on lambda",
        },
        {
            "model": "log10N_x_arch_moe",
            "n": n,
            "estimate": float(beta[3]),
            "note": "interaction",
        },
    ]
    summary = {
        "n": n,
        "sse_null_no_covariate": sse_null,
        "sse_null_arch_only": sse_noint - sse_full,
        "sse_interaction_model": sse_full,
        "interaction_perm_p": perm_p,
        "delta_sse_interaction_vs_null": obs_delta,
        "spearman_log10N_lambda": float(rho) if not math.isnan(rho) else float("nan"),
        "spearman_p": float(rho_p) if not math.isnan(rho_p) else float("nan"),
    }
    return rows_tsv, summary


def axis_G_arch_kfold(rows: list[dict], k: int = 5) -> tuple[list[dict], dict]:
    """K-fold CV of the full interaction model vs arch-only vs null on lambda.
    Restricted to runs with >=5 steps (others are pilot traces)."""
    pool = [r for r in rows if r["n"] >= 5]
    pool = [r for r in pool if not math.isnan(r["lam"])]
    n = len(pool)
    if n < 3:
        return [], {"error": "insufficient traces"}
    logN = np.asarray([math.log10(r["params"]) for r in pool], float)
    arch = np.asarray([1.0 if r["arch"] == "moe" else 0.0 for r in pool], float)
    lam = np.asarray([min(r["lam"], 9.99) for r in pool], float)
    folds = np.array_split(np.arange(n), min(k, n))

    def cv_sse(model: str) -> float:
        s = 0.0
        for f in folds:
            mask = np.ones(n, bool)
            mask[list(f)] = False
            tr_logN, tr_arch, tr_lam = logN[mask], arch[mask], lam[mask]
            te_logN, te_arch, te_lam = logN[~mask], arch[~mask], lam[~mask]
            if model == "null":
                te_pred = np.full(te_lam.shape, float(tr_lam.mean()))
            elif model == "logN_only":
                Xtr = tr_logN[:, None]
                Xte = te_logN[:, None]
                coef, *_ = _ols_matrix(Xtr, tr_lam)
                # coef is [intercept, slope]
                te_pred = coef[0] + Xte[:, 0] * coef[1]
            elif model == "logN_plus_arch":
                Xtr = np.column_stack([tr_logN, tr_arch])
                Xte = np.column_stack([te_logN, te_arch])
                coef, *_ = _ols_matrix(Xtr, tr_lam)
                te_pred = coef[0] + Xte @ coef[1:]
            elif model == "interaction":
                Xtr = np.column_stack([tr_logN, tr_arch, tr_logN * tr_arch])
                Xte = np.column_stack([te_logN, te_arch, te_logN * te_arch])
                coef, *_ = _ols_matrix(Xtr, tr_lam)
                te_pred = coef[0] + Xte @ coef[1:]
            else:
                raise ValueError(model)
            s += float(np.sum((te_lam - te_pred) ** 2))
        return s

    cv_null = cv_sse("null")
    cv_logN = cv_sse("logN_only")
    cv_logNarch = cv_sse("logN_plus_arch")
    cv_full = cv_sse("interaction")
    rows_tsv = [
        {"model": "null", "cv_sse_lambda": cv_null},
        {"model": "log10N_only", "cv_sse_lambda": cv_logN},
        {"model": "log10N_plus_arch", "cv_sse_lambda": cv_logNarch},
        {"model": "full_interaction", "cv_sse_lambda": cv_full},
    ]
    summary = {
        "n_used": n,
        "k": len(folds),
        "cv_null": cv_null,
        "cv_log10N_only": cv_logN,
        "cv_log10N_plus_arch": cv_logNarch,
        "cv_full_interaction": cv_full,
        "improvement_log10N_vs_null": cv_null - cv_logN,
        "improvement_arch_vs_log10N": cv_logN - cv_logNarch,
        "improvement_interaction_vs_arch": cv_logNarch - cv_full,
    }
    return rows_tsv, summary


def axis_H_r_max_invariant(rows: list[dict]) -> tuple[list[dict], dict]:
    """Levene's median test on R_max residual after regressing on log10(N)."""
    keep = [r for r in rows if not math.isnan(r["r_max"])]
    n = len(keep)
    logN = np.asarray([math.log10(r["params"]) for r in keep], float)
    rmax = np.asarray([r["r_max"] for r in keep], float)
    arch = np.asarray([r["arch"] for r in keep])
    coef, _, _ = _ols_matrix(logN, rmax)
    resid = rmax - (coef[0] + logN * coef[1])
    moe = resid[arch == "moe"]
    dense = resid[arch == "dense"]
    if len(moe) > 1 and len(dense) > 1:
        stat, p = levene(moe, dense, center="median")
    else:
        stat, p = float("nan"), float("nan")
    rows_tsv = []
    for r, a, lr, lnv in zip(keep, arch, resid, logN):
        rows_tsv.append(
            {
                "model": r["model"],
                "arch": a,
                "params_B": r["params"],
                "log10N": float(lnv),
                "r_max": r["r_max"],
                "r_max_residual_after_logN": float(lr),
            }
        )
    return rows_tsv, {
        "n": n,
        "n_moe": int((arch == "moe").sum()),
        "n_dense": int((arch == "dense").sum()),
        "levene_statistic": float(stat) if not math.isnan(stat) else float("nan"),
        "levene_p": float(p) if not math.isnan(p) else float("nan"),
        "ols_slope_r_max_on_log10N": float(coef[1]),
        "ols_intercept": float(coef[0]),
    }


def axis_I_two_anchor_extrap(rows: list[dict]) -> tuple[list[dict], dict]:
    """Two-anchor (Qwen3.5-4B + Qwen3-8B, both 30 steps) extrapolation."""
    keep = [r for r in rows if r["n"] >= 5 and not math.isnan(r["lam"])]
    keep = sorted(keep, key=lambda r: r["params"])

    # smallest two are the 4B and 8B; fit slope of lambda on log10(N)
    if len(keep) < 3:
        return [], {"error": "need >=3 traces"}
    anchor = keep[:2]
    rest = keep[2:]
    logN_a = np.asarray([math.log10(r["params"]) for r in anchor], float)
    lam_a = np.asarray([min(r["lam"], 9.99) for r in anchor], float)
    coef, _, _ = _ols_matrix(logN_a, lam_a)
    pred_slope = float(coef[1])

    preds = []
    for r in rest:
        logN = math.log10(r["params"])
        pred = float(coef[0]) + pred_slope * logN
        actual = min(r["lam"], 9.99)
        preds.append(
            {
                "model": r["model"],
                "params_B": r["params"],
                "arch": r["arch"],
                "actual_lambda": float(r["lam"]),
                "actual_lambda_capped": float(actual),
                "predicted_lambda": pred,
                "abs_error": abs(pred - actual),
                "signed_error": pred - actual,
            }
        )
    abs_err = [p["abs_error"] for p in preds]
    summary = {
        "n_anchors": 2,
        "anchor_models": ",".join(r["model"] for r in anchor),
        "predicted_slope_log10N_to_lambda": pred_slope,
        "predicted_intercept": float(coef[0]),
        "n_predictions": len(preds),
        "mae_lambda": float(np.mean(abs_err)) if abs_err else float("nan"),
        "max_abs_error": float(np.max(abs_err)) if abs_err else float("nan"),
    }
    return preds, summary


def axis_J_compute_regression(rows: list[dict]) -> tuple[list[dict], dict]:
    """lambda ~ log10(N) + log10(CF) + arch_dummy; CF = params_B * n_steps."""
    keep = [r for r in rows if not math.isnan(r["lam"]) and r["n"] >= 5]
    n = len(keep)
    logN = np.asarray([math.log10(r["params"]) for r in keep], float)
    cf = np.asarray([r["params"] * r["n"] for r in keep], float)
    logCF = np.log10(cf)
    arch = np.asarray([1.0 if r["arch"] == "moe" else 0.0 for r in keep], float)
    lam = np.asarray([min(r["lam"], 9.99) for r in keep], float)
    X = np.column_stack([logN, logCF, arch])
    beta, sse_full, _ = _ols_matrix(X, lam)
    X0 = np.column_stack([logN, arch])
    _, sse_noCF, _ = _ols_matrix(X0, lam)
    X1 = np.empty((n, 0))
    _, sse_null, _ = _ols_matrix(X1, lam)
    rows_tsv = [
        {"term": "intercept", "estimate": float(beta[0])},
        {"term": "log10N", "estimate": float(beta[1])},
        {"term": "log10_compute_proxy", "estimate": float(beta[2])},
        {"term": "arch_moe", "estimate": float(beta[3])},
    ]
    summary = {
        "n": n,
        "sse_null": sse_null,
        "sse_no_compute": sse_noCF,
        "sse_full": sse_full,
        "improvement_log10N_over_null": sse_null - sse_noCF,
        "improvement_compute_over_log10N": sse_noCF - sse_full,
        "compute_lift_share": (sse_noCF - sse_full) / max(1e-9, sse_null - sse_full),
    }
    return rows_tsv, summary


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("(empty)\n")
        return
    fieldnames = list({k for r in rows for k in r.keys()})
    fieldnames.sort()
    with path.open("w", newline="") as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in fieldnames) + "\n")


def make_figure(rows_fit: list[dict], summaries: dict, out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # panel 1: lambda vs log10(N) by arch
    ax = axes[0]
    for arch, color in [("dense", "#1f77b4"), ("moe", "#ff7f0e")]:
        sub = [r for r in rows_fit if r["arch"] == arch and not math.isnan(r["lam"])]
        xs = [math.log10(r["params"]) for r in sub]
        ys = [min(r["lam"], 9.99) for r in sub]
        ax.scatter(xs, ys, color=color, s=60, label=f"{arch} (n={len(sub)})", zorder=3)
        for r in sub:
            ax.annotate(
                r["model"].split("-")[0],
                (math.log10(r["params"]), min(r["lam"], 9.99)),
                fontsize=7,
                xytext=(2, 2),
                textcoords="offset points",
            )
    ax.axhline(9.5, color="grey", linestyle=":", linewidth=0.8)
    ax.text(2.4, 9.55, "lambda bound = 10", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 20, 30, 120, 235, 685, 1000])
    ax.set_xticklabels(["4", "8", "20", "30", "120", "235", "685", "1000"], fontsize=8)
    ax.set_xlabel("params (B, log scale)")
    ax.set_ylabel("lambda (capped at 9.99)")
    ax.set_title("(E) lambda vs params (12 anchors)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    # panel 2: arch-fraction lambda-at-bound
    ax = axes[1]
    arch_bound = {
        "dense": [r["lam_at_bound"] for r in rows_fit if r["arch"] == "dense"],
        "moe": [r["lam_at_bound"] for r in rows_fit if r["arch"] == "moe"],
    }
    labels = list(arch_bound.keys())
    fracs = [sum(arch_bound[k]) / max(1, len(arch_bound[k])) for k in labels]
    n_each = [len(arch_bound[k]) for k in labels]
    ax.bar(labels, fracs, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction with lambda_at_bound")
    ax.set_title("(E) lambda-bound hit rate by arch")
    for i, k in enumerate(labels):
        ax.text(i, fracs[i] + 0.02, f"{fracs[i]:.0%} (n={n_each[i]})", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # panel 3: R_max residual after log10(N) OLS, by arch
    ax = axes[2]
    keep = [r for r in rows_fit if not math.isnan(r["r_max"])]
    logN = np.asarray([math.log10(r["params"]) for r in keep], float)
    rmax = np.asarray([r["r_max"] for r in keep], float)
    arch_arr = np.asarray([r["arch"] for r in keep])
    coef, *_ = _ols_matrix(logN, rmax)
    resid = rmax - (coef[0] + logN * coef[1])
    for a, color in [("dense", "#1f77b4"), ("moe", "#ff7f0e")]:
        mask = arch_arr == a
        ax.scatter(logN[mask], resid[mask], color=color, s=60, label=f"{a}", zorder=3)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 20, 30, 120, 235, 685, 1000])
    ax.set_xticklabels(["4", "8", "20", "30", "120", "235", "685", "1000"], fontsize=8)
    ax.set_xlabel("params (B, log scale)")
    ax.set_ylabel("R_max residual")
    ax.set_title(f"(G) Levene p={summaries.get('lev_p', float('nan')):.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Pillar 1 iter-21: cross-architecture scaling audit", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> dict:
    print("== Pillar 1 iter-21: cross-architecture stratification ==")
    data = load_traces()
    fits = []
    for label, meta in data.items():
        f = fit_per_trace(meta["trace"])
        f.update(
            model=label,
            params=meta["params"],
            arch=meta["arch"],
            family=meta["family"],
        )
        fits.append(f)
    fits.sort(key=lambda r: r["params"])

    print(f"  loaded {len(fits)} traces")

    audit_rows, audit_sum = axis_E_lambda_audit(fits)
    print(f"  axis E: {audit_sum['n_lam_at_bound']}/{audit_sum['n_traces']} hit lambda bound")
    write_tsv(RESULTS_DIR / "scaling_law_iter21_lambda_audit.tsv", audit_rows)

    reg_rows, reg_sum = axis_F_arch_regression(fits)
    print(
        f"  axis F: interaction perm_p={reg_sum['interaction_perm_p']:.3f}, "
        f"spearman(log10N, lambda)={reg_sum['spearman_log10N_lambda']:.3f}"
    )
    write_tsv(RESULTS_DIR / "scaling_law_iter21_arch_regression.tsv", reg_rows)

    kf_rows, kf_sum = axis_G_arch_kfold(fits, k=5)
    print(
        f"  axis G: 5-fold CV -- improvement log10N over null = "
        f"{kf_sum['improvement_log10N_vs_null']:.3f}, "
        f"interaction over arch = {kf_sum['improvement_interaction_vs_arch']:.3f}"
    )
    write_tsv(RESULTS_DIR / "scaling_law_iter21_arch_kfold.tsv", kf_rows)

    rmsx_rows, rmsx_sum = axis_H_r_max_invariant(fits)
    print(
        f"  axis H: Levene(median) stat={rmsx_sum['levene_statistic']:.3f}, "
        f"p={rmsx_sum['levene_p']:.3f}, n_moe={rmsx_sum['n_moe']}, n_dense={rmsx_sum['n_dense']}"
    )
    write_tsv(RESULTS_DIR / "scaling_law_iter21_r_max_residual.tsv", rmsx_rows)

    extr_rows, extr_sum = axis_I_two_anchor_extrap(fits)
    print(
        f"  axis I: two-anchor extrap MAE=lambda {extr_sum.get('mae_lambda', float('nan')):.3f}, "
        f"max abs err {extr_sum.get('max_abs_error', float('nan')):.3f}"
    )
    write_tsv(RESULTS_DIR / "scaling_law_iter21_two_anchor_extrap.tsv", extr_rows)

    comp_rows, comp_sum = axis_J_compute_regression(fits)
    print(
        f"  axis J: compute lift share = {comp_sum['compute_lift_share']:.3f}"
    )
    write_tsv(RESULTS_DIR / "scaling_law_iter21_compute_regression.tsv", comp_rows)

    summary_rows = [
        {"axis": "E_lambda_audit", **audit_sum},
        {"axis": "F_arch_regression_interaction_p", "value": reg_sum["interaction_perm_p"]},
        {"axis": "G_arch_kfold_5fold", **kf_sum},
        {"axis": "H_r_max_invariant_levene_p", "value": rmsx_sum["levene_p"]},
        {"axis": "I_two_anchor_extrap_mae", "value": extr_sum.get("mae_lambda", float("nan"))},
        {"axis": "J_compute_regression_lift_share", "value": comp_sum["compute_lift_share"]},
    ]
    write_tsv(RESULTS_DIR / "scaling_law_iter21_summary.tsv", summary_rows)

    summaries_for_fig = dict(lev_p=rmsx_sum["levene_p"])
    out_png = FIG_DIR / "scaling_law_iter21.png"
    out_pdf = FIG_DIR / "scaling_law_iter21.pdf"
    make_figure(fits, summaries_for_fig, out_png, out_pdf)
    paper_pdf = PAPER_FIG / "scaling_law_iter21.pdf"
    paper_pdf.write_bytes(out_pdf.read_bytes())
    print(f"  figure -> {out_png}, {out_pdf}")
    return {
        "audit": audit_sum,
        "reg": reg_sum,
        "kf": kf_sum,
        "rmsx": rmsx_sum,
        "extr": extr_sum,
        "comp": comp_sum,
    }


if __name__ == "__main__":
    main()
