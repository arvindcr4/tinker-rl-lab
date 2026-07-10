"""Pillar 1 iter89 -- Holdout validation + bootstrap stability of the
3-phase template (Nimmaturi et al. 2025, arXiv:2507.18014).

iter85 (3-phase AIC conformity) and iter73 (saturation AIC) both rely on
in-sample fit quality.  Iter89 closes the over-fitting / generalisation
question from two complementary angles:

  Q1 (LOOCV) -- leave-one-step-out prediction error per anchor for
        {constant-mean, 1-segment OLS, saturation R_max*(1-exp(-lambda t)),
         3-segment piecewise-linear (iter85)}.  Lower CV-RMSE = better
         generalisation.  Reports CV-RMSE and the per-anchor winner.

  Q2 (bootstrap change-point stability) -- resample the trace 1000 times
        WITH replacement (treating sequential step-rewards as an iid-ish
        residual pool around an unknown level), refit the 3-segment
        template, collect (cp1, cp2) and segment slopes.  Reports
        empirical SD and the canonical-order satisfaction rate under
        bootstrap noise.

  Q3 (BIC vs AIC for 3-phase) -- Burnham-style model selection; BIC
        penalises free parameters more strongly than AIC.  Reports
        delta_bic_3v1 and delta_bic_3v2 per anchor; verifies whether
        the iter85 AIC verdict (6/9 winners for 3-phase) survives the
        tighter BIC penalty.

  Q4 (k-fold forecast; k=4 over the LAST 4 steps) -- for n>=8 anchors,
        fit on steps [1..n-4] using each candidate family, then
        forecast y_{n-3..n}.  Mean absolute forecast error (MAFE) is
        reported; the model with the lowest median MAFE wins.

Outputs:
  experiments/results/scaling_law_iter89_loocv.tsv
  experiments/results/scaling_law_iter89_bootstrap.tsv
  experiments/results/scaling_law_iter89_bic.tsv
  experiments/results/scaling_law_iter89_kfold_forecast.tsv
  experiments/results/scaling_law_iter89_meta.json
  figures/scaling_law_iter89.{pdf,png}
  paper/sections/scaling_law_iter89.tex

Citations (verified):
  - nimmaturi2025predictive (arXiv:2507.18014) -- 3-phase template.
  - burnham2002model -- AIC / BIC model-selection theory.
  - kaplan2020scaling -- Chinchilla step axis baseline.
  - bishop2006pattern (PRML sec 1.3 / 4.3) -- LOOCV model selection.
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

# 12-anchor pool (kept identical to iter81/85 for comparability).
MODELS: dict[str, tuple[str, float, str]] = {
    "Qwen3.5-4B": ("qwen3.5-4b", 4.0, "scale_gsm8k_qwen3.5-4b.json"),
    "Qwen3-8B": ("qwen3-8b", 8.0, "scale_gsm8k_qwen3-8b.json"),
    "Llama-3.1-8B-Instruct": ("llama-8b-inst", 8.0, "scale_gsm8k_llama-8b-inst.json"),
    "Qwen3-32B": ("qwen3-32b", 32.0, "scale_gsm8k_qwen3-32b.json"),
    "Qwen3.5-27B": ("qwen3.5-27b", 27.0, "scale_gsm8k_qwen3.5-27b.json"),
    "gpt-oss-20B": ("gpt-oss-20b", 20.0, "arch_gsm8k_gpt-oss-20b.json"),
    "Qwen3-30B-MoE": ("qwen3-30b-moe", 30.0, "moe_gsm8k_qwen3-30b-moe.json"),
    "Qwen3-30B-MoE-Inst": ("qwen3-30b-moe-inst", 30.0, "moe_gsm8k_qwen3-30b-inst.json"),
    "DeepSeek-V3.1": ("deepseek-v3.1", 685.0, "frontier_gsm8k_deepseek-v3.1.json"),
    "Nemotron-120B": ("nemotron-120b", 120.0, "frontier_gsm8k_nemotron-120b.json"),
    "Qwen3-235B-MoE": ("qwen3-235b-moe", 235.0, "frontier_gsm8k_qwen3-235b.json"),
    "Kimi-K2-Thinking": ("kimi-k2", 1000.0, "arch_gsm8k_kimi-k2.json"),
}

RNG = np.random.default_rng(42)
N_BOOT = 1000
N_LOOCV_MIN = 8  # anchors with n>=8 enter LOOCV


# ---------------------------------------------------------------------------
# candidate-family fits
# ---------------------------------------------------------------------------
def fit_constant(y: np.ndarray) -> dict:
    """1-parameter constant-mean baseline."""
    n = len(y)
    mu = float(np.mean(y))
    rss = float(np.sum((y - mu) ** 2))
    return {"name": "constant", "k": 1, "rss": rss, "pred": np.full(n, mu),
            "params": {"mu": mu}}


def fit_1seg(y: np.ndarray) -> dict:
    """1-segment OLS: slope + intercept (k=2)."""
    n = len(y)
    x = np.arange(n, dtype=float)
    sl, ic = np.polyfit(x, y, 1)
    pred = sl * x + ic
    rss = float(np.sum((y - pred) ** 2))
    return {"name": "1seg_ols", "k": 3, "rss": rss,
            "pred": pred, "params": {"slope": float(sl), "intercept": float(ic)}}


def fit_saturation(y: np.ndarray) -> dict:
    """R_max*(1 - exp(-lambda*t)) fit (k=2).  Uses closed-form regression
    in the linearised domain then a small grid refinement for lambda.
    """
    n = len(y)
    if n < 3:
        mu = float(np.mean(y))
        return {"name": "saturation", "k": 2, "rss": float(np.sum((y - mu) ** 2)),
                "pred": np.full(n, mu), "params": {"R_max": mu, "lam": 0.01}}

    # Linearised regression: y ~ 1 - (1 - y/R_max)*exp(-lam t).  Without
    # a fixed R_max this is awkward; use a coarse R_max grid and for each
    # R_max do a linear regression on log(1 - y/R_max).
    x = np.arange(n, dtype=float)
    best = (math.inf, 1.0, 0.01)
    for r_max in np.linspace(0.5, 1.5, 21):
        # protect against |y| >= r_max
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = (1 - y / r_max) > 0
        if mask.sum() < 4:
            continue
        ratio = 1 - y[mask] / r_max
        # OLS for log(ratio) = -lam * t + c
        xt = x[mask]
        z = np.log(np.clip(ratio, 1e-6, None))
        if not np.all(np.isfinite(z)):
            continue
        sl, ic = np.polyfit(xt, z, 1)
        lam = max(1e-3, -sl)
        pred = r_max * (1 - np.exp(-lam * x))
        rss = float(np.sum((y - pred) ** 2))
        if rss < best[0]:
            best = (rss, r_max, lam)
    rss, r_max, lam = best
    pred = r_max * (1 - np.exp(-lam * x))
    return {"name": "saturation", "k": 2, "rss": float(np.sum((y - pred) ** 2)),
            "pred": pred, "params": {"R_max": float(r_max), "lam": float(lam)}}


def fit_3seg(y: np.ndarray) -> dict:
    """Iter85 greedy 3-segment piecewise-linear fit (k=5 free: 3 slopes + 2 CPs)."""
    from scaling_law_iter85 import fit_piecewise_linear

    n = len(y)
    sse, cps, sl, ic, lens = fit_piecewise_linear(y, 3)
    if len(sl) != 3:
        return fit_1seg(y) | {"name": "3seg_failed"}
    # Reconstruct prediction
    pred = np.zeros(n)
    segments = [(0, cps[0]), (cps[0], cps[1]), (cps[1], n)] if len(cps) == 2 else []
    if not segments:
        return fit_1seg(y) | {"name": "3seg_failed"}
    for (a, b), s, i in zip(segments, sl, ic):
        x = np.arange(a, b, dtype=float)
        pred[a:b] = s * x + i
    return {"name": "3seg", "k": 8,  # 3*(slope+intercept) + 2 CPs + sigma
            "rss": float(sse), "pred": pred,
            "params": {"cp1": int(cps[0]), "cp2": int(cps[1]),
                       "slope1": float(sl[0]), "slope2": float(sl[1]), "slope3": float(sl[2])}}


FAMILIES = ["constant", "1seg_ols", "saturation", "3seg"]


def fit_family(name: str, y: np.ndarray) -> dict:
    if name == "constant":
        return fit_constant(y)
    if name == "1seg_ols":
        return fit_1seg(y)
    if name == "saturation":
        return fit_saturation(y)
    if name == "3seg":
        return fit_3seg(y)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# information criteria
# ---------------------------------------------------------------------------
def aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    if n <= 0 or rss <= 0:
        return float("nan"), float("nan")
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return float(aic), float(bic)


# ---------------------------------------------------------------------------
# LOOCV (Bishop PRML 4.3 / Efron 1983)
# ---------------------------------------------------------------------------
def loocv(y: np.ndarray, family: str) -> dict:
    """Leave-one-out CV: leave each step out in turn, refit, predict.
    Returns CV-RMSE and CV-MAE.
    """
    n = len(y)
    sq, ab = 0.0, 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        fit = fit_family(family, y[mask])
        try:
            p = float(fit["pred"][i]) if i == 0 else _predict_at(fit, i)
            # if the family doesn't expose a closed-form predictor, fall
            # back to using the in-sample position from the held-in subset
            if not math.isfinite(p):
                p = float(np.mean(y[mask]))
        except Exception:
            p = float(np.mean(y[mask]))
        err = float(y[i]) - p
        sq += err * err
        ab += abs(err)
    return {"cv_rmse": math.sqrt(sq / n), "cv_mae": ab / n}


def _predict_at(fit: dict, idx: int) -> float:
    """Evaluate a fitted family at held-out step idx."""
    n = len(fit["pred"]) + 1  # the dropped step was inserted somewhere
    name = fit["name"]
    if name == "constant":
        return fit["params"]["mu"]
    if name == "1seg_ols":
        return fit["params"]["slope"] * idx + fit["params"]["intercept"]
    if name == "saturation":
        return fit["params"]["R_max"] * (1 - math.exp(-fit["params"]["lam"] * idx))
    if name == "3seg":
        cp1 = fit["params"]["cp1"]
        cp2 = fit["params"]["cp2"]
        # NB: cps were fit on n-1 points; an idx that landed *before* the new
        # gap shouldn't appear (iter85 greedy fit) but for safety clamp.
        if idx < cp1:
            return fit["params"]["slope1"] * idx + 0.0
        if idx < cp2:
            return fit["params"]["slope2"] * idx + 0.0
        return fit["params"]["slope3"] * idx + 0.0
    return float("nan")


# ---------------------------------------------------------------------------
# k-fold forecast on last k steps
# ---------------------------------------------------------------------------
def kfold_forecast(y: np.ndarray, k_hold: int = 4) -> dict:
    """Fit each family on the first (n-k_hold) steps; predict y[k_hold-4..n-1].

    Returns dict[family] = {'forecast': np.ndarray, 'mae': float, 'rmse': float}.
    """
    n = len(y)
    train = y[: n - k_hold]
    test = y[n - k_hold:]
    out = {}
    for fam in FAMILIES:
        fit = fit_family(fam, train)
        idxs = np.arange(n - k_hold, n)
        if fam == "constant":
            preds = np.full(k_hold, fit["params"]["mu"])
        elif fam == "1seg_ols":
            s = fit["params"]["slope"]
            i = fit["params"]["intercept"]
            preds = s * idxs + i
        elif fam == "saturation":
            preds = fit["params"]["R_max"] * (1 - np.exp(-fit["params"]["lam"] * idxs))
        elif fam == "3seg":
            cp1 = fit["params"]["cp1"]
            cp2 = fit["params"]["cp2"]
            preds = np.zeros(k_hold)
            for j, idx in enumerate(idxs):
                if idx < cp1:
                    preds[j] = fit["params"]["slope1"] * idx
                elif idx < cp2:
                    preds[j] = fit["params"]["slope2"] * idx
                else:
                    preds[j] = fit["params"]["slope3"] * idx
        else:
            preds = np.full(k_hold, float(np.mean(train)))
        # clip to reward bounds [0,1] (GRPO is sparse-binary reward trace)
        preds = np.clip(preds, 0.0, 1.0)
        err = test - preds
        out[fam] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "preds": [round(float(p), 4) for p in preds],
            "test": [round(float(t), 4) for t in test],
        }
    return out


# ---------------------------------------------------------------------------
# bootstrap stability of change-points
# ---------------------------------------------------------------------------
def bootstrap_3seg(y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Resample trace with replacement; refit 3-segment;
    collect CP1/CP2 distribution and slope distributions.
    """
    from scaling_law_iter85 import fit_piecewise_linear

    n = len(y)
    cp1s, cp2s = [], []
    sl1s, sl2s, sl3s = [], [], []
    canon_orders = 0
    spurt_largest_count = 0
    failed = 0
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot = y[idx]
        try:
            _, cps, sls, _, _ = fit_piecewise_linear(boot, 3)
            if len(cps) != 2 or len(sls) != 3:
                failed += 1
                continue
            cp1s.append(int(cps[0]))
            cp2s.append(int(cps[1]))
            sl1s.append(float(sls[0]))
            sl2s.append(float(sls[1]))
            sl3s.append(float(sls[2]))
            abs3 = [abs(s) for s in sls]
            spurt_index = int(np.argmax(abs3))
            creep_index = int(np.argmin(abs3))
            level_index = next(i for i in range(3) if i not in (creep_index, spurt_index))
            if creep_index < spurt_index < level_index:
                canon_orders += 1
            if abs(sls[1]) >= abs(sls[0]) and abs(sls[1]) >= abs(sls[2]):
                spurt_largest_count += 1
        except Exception:
            failed += 1
    n_eff = len(cp1s)
    out = {
        "n_boot_total": n_boot,
        "n_boot_succeeded": int(n_eff),
        "n_boot_failed": int(failed),
        "cp1_mean": float(np.mean(cp1s)) if cp1s else float("nan"),
        "cp1_sd": float(np.std(cp1s)) if cp1s else float("nan"),
        "cp2_mean": float(np.mean(cp2s)) if cp2s else float("nan"),
        "cp2_sd": float(np.std(cp2s)) if cp2s else float("nan"),
        "cp1_cv": float(np.std(cp1s) / max(np.mean(cp1s), 1e-6)) if cp1s else float("nan"),
        "cp2_cv": float(np.std(cp2s) / max(np.mean(cp2s), 1e-6)) if cp2s else float("nan"),
        "slope1_mean": float(np.mean(sl1s)) if sl1s else float("nan"),
        "slope2_mean": float(np.mean(sl2s)) if sl2s else float("nan"),
        "slope3_mean": float(np.mean(sl3s)) if sl3s else float("nan"),
        "slope1_sd": float(np.std(sl1s)) if sl1s else float("nan"),
        "slope2_sd": float(np.std(sl2s)) if sl2s else float("nan"),
        "slope3_sd": float(np.std(sl3s)) if sl3s else float("nan"),
        "canon_order_rate": float(canon_orders / max(n_eff, 1)),
        "spurt_largest_rate": float(spurt_largest_count / max(n_eff, 1)),
    }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    loocv_rows = []
    bs_rows = []
    bic_rows = []
    kfold_rows = []
    anchor_data = {}

    for model_name, (short, params_b, fname) in MODELS.items():
        fpath = TRACE_DIR / fname
        if not fpath.exists():
            print(f"WARN: missing {model_name} -> {fpath}")
            continue
        d = json.load(open(fpath))
        rt = np.array(d["reward_trace"], dtype=float)
        n = len(rt)
        if n < N_LOOCV_MIN:
            # skip LOOCV for very short traces but still report BIC/BS
            print(f"NOTE: skipping LOOCV for {model_name} (n={n}<{N_LOOCV_MIN})")

        # In-sample fits for AIC/BIC reference
        fits: dict[str, dict] = {}
        for fam in FAMILIES:
            try:
                fits[fam] = fit_family(fam, rt)
            except Exception as e:
                fits[fam] = {"name": fam, "k": 0, "rss": float("nan"),
                             "pred": np.zeros(n), "params": {}, "err": str(e)}

        # AIC/BIC table
        aics, bics = {}, {}
        for fam in FAMILIES:
            f = fits[fam]
            a, b = aic_bic(f["rss"], n, f["k"])
            aics[fam] = a
            bics[fam] = b
        best_aic = min(aics.values())
        best_bic = min(bics.values())
        bic_rows.append({
            "model": model_name, "model_short": short, "params_B": params_b,
            "n_steps": n,
            "n_params_const": fits["constant"]["k"],
            "n_params_1seg": fits["1seg_ols"]["k"],
            "n_params_sat": fits["saturation"]["k"],
            "n_params_3seg": fits["3seg"]["k"],
            "aic_const": round(aics["constant"], 3),
            "aic_1seg": round(aics["1seg_ols"], 3),
            "aic_sat": round(aics["saturation"], 3),
            "aic_3seg": round(aics["3seg"], 3),
            "bic_const": round(bics["constant"], 3),
            "bic_1seg": round(bics["1seg_ols"], 3),
            "bic_sat": round(bics["saturation"], 3),
            "bic_3seg": round(bics["3seg"], 3),
            "best_aic_family": min(aics, key=aics.get),
            "best_bic_family": min(bics, key=bics.get),
            "delta_aic_3v1": round(aics["3seg"] - aics["1seg_ols"], 3),
            "delta_aic_3v0": round(aics["3seg"] - aics["constant"], 3),
            "delta_bic_3v1": round(bics["3seg"] - bics["1seg_ols"], 3),
            "delta_bic_3v0": round(bics["3seg"] - bics["constant"], 3),
            "delta_bic_3vsat": round(bics["3seg"] - bics["saturation"], 3),
            "best_aic_aic": round(best_aic, 3),
            "best_bic_aic": round(best_bic, 3),
        })

        # LOOCV (only if n is large enough)
        if n >= N_LOOCV_MIN:
            for fam in FAMILIES:
                res = loocv(rt, fam)
                loocv_rows.append({
                    "model": model_name, "model_short": short, "params_B": params_b,
                    "n_steps": n, "family": fam,
                    "cv_rmse": round(res["cv_rmse"], 4),
                    "cv_mae": round(res["cv_mae"], 4),
                })

        # Bootstrap stability of change-points (3-segment)
        bs = bootstrap_3seg(rt)
        bs_rows.append({
            "model": model_name, "model_short": short, "params_B": params_b,
            "n_steps": n,
            "cp1_mean": round(bs["cp1_mean"], 2),
            "cp1_sd": round(bs["cp1_sd"], 2),
            "cp1_cv": round(bs["cp1_cv"], 3),
            "cp2_mean": round(bs["cp2_mean"], 2),
            "cp2_sd": round(bs["cp2_sd"], 2),
            "cp2_cv": round(bs["cp2_cv"], 3),
            "slope1_mean": round(bs["slope1_mean"], 4),
            "slope2_mean": round(bs["slope2_mean"], 4),
            "slope3_mean": round(bs["slope3_mean"], 4),
            "slope1_sd": round(bs["slope1_sd"], 4),
            "slope2_sd": round(bs["slope2_sd"], 4),
            "slope3_sd": round(bs["slope3_sd"], 4),
            "canon_order_rate": round(bs["canon_order_rate"], 3),
            "spurt_largest_rate": round(bs["spurt_largest_rate"], 3),
            "n_boot_succeeded": bs["n_boot_succeeded"],
        })

        # k-fold forecast (last 4 steps)
        if n >= 12:
            kfold = kfold_forecast(rt, k_hold=4)
            for fam, r in kfold.items():
                kfold_rows.append({
                    "model": model_name, "model_short": short, "params_B": params_b,
                    "n_steps": n, "family": fam,
                    "forecast_mae": round(r["mae"], 4),
                    "forecast_rmse": round(r["rmse"], 4),
                    "test_steps": ",".join(str(t) for t in r["test"]),
                    "pred_steps": ",".join(str(p) for p in r["preds"]),
                })

        anchor_data[model_name] = {
            "rt": rt, "fits": {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                                | ({"last_pred": float(v["pred"][-1])} if "pred" in v else {})
                                for k, v in fits.items()},
        }

    # ----- write outputs -----
    keys = bic_rows[0].keys() if bic_rows else []
    with open(RESULTS_DIR / "scaling_law_iter89_bic.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in bic_rows:
            w.writerow(r)

    keys = loocv_rows[0].keys() if loocv_rows else ["model", "note"]
    with open(RESULTS_DIR / "scaling_law_iter89_loocv.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in loocv_rows:
            w.writerow(r)
        if not loocv_rows:
            f.write("model\tnote\nEMPTY\tno anchor with n>=8\n")

    keys = bs_rows[0].keys() if bs_rows else []
    with open(RESULTS_DIR / "scaling_law_iter89_bootstrap.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in bs_rows:
            w.writerow(r)

    keys = kfold_rows[0].keys() if kfold_rows else ["model", "note"]
    with open(RESULTS_DIR / "scaling_law_iter89_kfold_forecast.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in kfold_rows:
            w.writerow(r)
        if not kfold_rows:
            f.write("model\tnote\nEMPTY\tno anchor with n>=12\n")

    # ----- summary meta -----
    n_anchors = len(bic_rows)
    n_aic_3seg = sum(1 for r in bic_rows if r["best_aic_family"] == "3seg")
    n_bic_3seg = sum(1 for r in bic_rows if r["best_bic_family"] == "3seg")
    n_aic_const = sum(1 for r in bic_rows if r["best_aic_family"] == "constant")
    n_bic_const = sum(1 for r in bic_rows if r["best_bic_family"] == "constant")
    n_aic_1seg = sum(1 for r in bic_rows if r["best_aic_family"] == "1seg_ols")
    n_bic_1seg = sum(1 for r in bic_rows if r["best_bic_family"] == "1seg_ols")
    n_aic_sat = sum(1 for r in bic_rows if r["best_aic_family"] == "saturation")
    n_bic_sat = sum(1 for r in bic_rows if r["best_bic_family"] == "saturation")

    # LOOCV winners: per anchor -> min CV-RMSE family
    loo_winners: dict[str, str] = {}
    for r in loocv_rows:
        anchor = r["model"]
        if anchor not in loo_winners or r["cv_rmse"] < next(
                x["cv_rmse"] for x in loocv_rows
                if x["model"] == anchor and x["family"] == loo_winners[anchor]):
            loo_winners[anchor] = r["family"]
    # actually we need to find min properly across rows for the anchor
    by_anchor_loocv: dict[str, dict[str, float]] = {}
    for r in loocv_rows:
        by_anchor_loocv.setdefault(r["model"], {})[r["family"]] = r["cv_rmse"]
    loo_winners = {a: min(d, key=d.get) for a, d in by_anchor_loocv.items()}
    n_loo_3seg = sum(1 for v in loo_winners.values() if v == "3seg")
    n_loo_const = sum(1 for v in loo_winners.values() if v == "constant")
    n_loo_1seg = sum(1 for v in loo_winners.values() if v == "1seg_ols")
    n_loo_sat = sum(1 for v in loo_winners.values() if v == "saturation")

    # kfold winners
    by_anchor_kf: dict[str, dict[str, float]] = {}
    for r in kfold_rows:
        by_anchor_kf.setdefault(r["model"], {})[r["family"]] = r["forecast_mae"]
    kf_winners = {a: min(d, key=d.get) for a, d in by_anchor_kf.items()}
    n_kf_3seg = sum(1 for v in kf_winners.values() if v == "3seg")
    n_kf_const = sum(1 for v in kf_winners.values() if v == "constant")
    n_kf_1seg = sum(1 for v in kf_winners.values() if v == "1seg_ols")
    n_kf_sat = sum(1 for v in kf_winners.values() if v == "saturation")

    # Bootstrap stability summary
    canon_rates = [r["canon_order_rate"] for r in bs_rows if r["n_steps"] >= 10]
    spurt_rates = [r["spurt_largest_rate"] for r in bs_rows if r["n_steps"] >= 10]
    cp2_cvs = [r["cp2_cv"] for r in bs_rows if r["n_steps"] >= 10]

    meta = {
        "n_anchors": n_anchors,
        "n_loocv_anchors": len(by_anchor_loocv),
        "n_kfold_anchors": len(by_anchor_kf),
        "AIC_winner_counts": {
            "constant": n_aic_const, "1seg_ols": n_aic_1seg,
            "saturation": n_aic_sat, "3seg": n_aic_3seg,
        },
"BIC_winner_counts": {
            "constant": n_bic_const, "1seg_ols": n_bic_1seg,
            "saturation": n_bic_sat, "3seg": n_bic_3seg,
        },
        "LOOCV_winner_counts": {
            "constant": n_loo_const, "1seg_ols": n_loo_1seg,
            "saturation": n_loo_sat, "3seg": n_loo_3seg,
        },
        "kfold_forecast_winner_counts": {
            "constant": n_kf_const, "1seg_ols": n_kf_1seg,
            "saturation": n_kf_sat, "3seg": n_kf_3seg,
        },
        "n_bic_3seg_winners": n_bic_3seg,
        "n_aic_3seg_winners": n_aic_3seg,
        "n_loo_3seg_winners": n_loo_3seg,
        "n_kfold_3seg_winners": n_kf_3seg,
        "bootstrap_canon_order_rate_mean": float(np.mean(canon_rates)) if canon_rates else None,
        "bootstrap_spurt_largest_rate_mean": float(np.mean(spurt_rates)) if spurt_rates else None,
        "bootstrap_cp2_cv_mean": float(np.mean(cp2_cvs)) if cp2_cvs else None,
        "kfold_winners_per_anchor": kf_winners,
        "loocv_winners_per_anchor": loo_winners,
        "bic_winners_per_anchor": {r["model"]: r["best_bic_family"] for r in bic_rows},
        "aic_winners_per_anchor": {r["model"]: r["best_aic_family"] for r in bic_rows},
    }
    with open(RESULTS_DIR / "scaling_law_iter89_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ----- figure (4-panel) -----
    # Panel A: LOOCV per-anchor RMSE bar (grouped by family)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    if loocv_rows:
        anchors = sorted(by_anchor_loocv.keys())
        x = np.arange(len(anchors))
        w_b = 0.2
        for fi, fam in enumerate(FAMILIES):
            vals = [by_anchor_loocv[a].get(fam, 0) for a in anchors]
            ax.bar(x + (fi - 1.5) * w_b, vals, width=w_b, label=fam)
        ax.set_xticks(x)
        ax.set_xticklabels([a[:10] for a in anchors], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("LOOCV RMSE")
        ax.set_title("(A) Leave-one-step-out RMSE per anchor", fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no LOOCV (no anchor n>=8)", ha="center")

    # Panel B: AIC vs BIC winner counts (bar)
    ax = axes[0, 1]
    labels = ["constant", "1seg_ols", "saturation", "3seg"]
    aic_cnts = [meta["AIC_winner_counts"][l] for l in labels]
    bic_cnts = [meta["BIC_winner_counts"][l] for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, aic_cnts, width=0.35, label="AIC", color="#4477AA")
    ax.bar(x + 0.18, bic_cnts, width=0.35, label="BIC", color="#EE6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("# anchors (winner)")
    ax.set_title("(B) Information-criterion winners across 12 anchors", fontsize=10)
    ax.legend()

    # Panel C: bootstrap CP2 SD vs n
    ax = axes[1, 0]
    bs_long = [r for r in bs_rows if r["n_steps"] >= 10]
    if bs_long:
        ns = [r["n_steps"] for r in bs_long]
        cp2_sd = [r["cp2_sd"] for r in bs_long]
        spurt = [r["spurt_largest_rate"] for r in bs_long]
        sizes = [120 * r["params_B"] / 1000 + 30 for r in bs_long]
        ax.scatter(ns, cp2_sd, s=sizes, alpha=0.7, color="#228833")
        ax.set_xlabel("trace length n (steps)")
        ax.set_ylabel("CP2 SD (steps)")
        ax.set_title("(C) Bootstrap SD of 3-seg changepoint #2", fontsize=10)
        for r in bs_long:
            ax.annotate(r["model_short"][:8], (r["n_steps"], r["cp2_sd"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

    # Panel D: k-fold forecast MAE winners
    ax = axes[1, 1]
    if kfold_rows:
        anchors = sorted(by_anchor_kf.keys())
        x = np.arange(len(anchors))
        w_b = 0.2
        for fi, fam in enumerate(FAMILIES):
            vals = [by_anchor_kf[a].get(fam, 0) for a in anchors]
            ax.bar(x + (fi - 1.5) * w_b, vals, width=w_b, label=fam)
        ax.set_xticks(x)
        ax.set_xticklabels([a[:10] for a in anchors], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("forecast MAE on last 4 steps")
        ax.set_title("(D) Forecast MAE -- fit on first (n-4), predict last 4", fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper left")

    fig.suptitle(f"Iter89 -- LOOCV / BIC / bootstrap / k-fold on iter81 anchor pool (12 anchors; N_BOOT={N_BOOT})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "scaling_law_iter89.png", dpi=120)
    fig.savefig(FIG_DIR / "scaling_law_iter89.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_iter89.png", dpi=120)
    fig.savefig(PAPER_FIG / "scaling_law_iter89.pdf")

    print("\n===== iter89 summary =====")
    print(f"Anchors parsed:                    {n_anchors}")
    print(f"AIC winner counts:                 {meta['AIC_winner_counts']}")
    print(f"BIC winner counts:                 {meta['BIC_winner_counts']}")
    print(f"LOOCV winner counts:               {meta['LOOCV_winner_counts']}")
    print(f"k-fold forecast winner counts:     {meta['kfold_forecast_winner_counts']}")
    print(f"BIC 3-phase winners:               {n_bic_3seg}/{n_anchors}")
    print(f"AIC 3-phase winners:               {n_aic_3seg}/{n_anchors}")
    print(f"LOOCV 3-phase winners:             {n_loo_3seg}/{len(by_anchor_loocv)}")
    print(f"k-fold 3-phase winners:            {n_kf_3seg}/{len(by_anchor_kf)}")
    if canon_rates:
        print(f"Bootstrap canon-order rate (mean): {meta['bootstrap_canon_order_rate_mean']:.3f}")
        print(f"Bootstrap spurt-largest rate (mean): {meta['bootstrap_spurt_largest_rate_mean']:.3f}")
        print(f"Bootstrap CP2 CV  (mean):          {meta['bootstrap_cp2_cv_mean']:.3f}")
    print(f"LOOCV winners: {loo_winners}")
    print(f"k-fold winners: {kf_winners}")


if __name__ == "__main__":
    main()
