"""Exploratory recorded-group ZVF and saturation optimizer diagnostics.
Extreme step means are a distinct descriptive statistic, never group ZVF.
Cross-cohort summaries do not identify causal effects or matched protocols.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"
PAPER_FIG = CANDIDATE / "paper_figures"

# 12-anchor pool: identical to iter13 / iter49.
MODELS: dict[str, dict] = {
    "Qwen3.5-4B": {"file": "scale_gsm8k_qwen3.5-4b.json", "params": 4.0, "arch": "dense"},
    "Qwen3-8B": {"file": "scale_gsm8k_qwen3-8b.json", "params": 8.0, "arch": "dense"},
    "Llama-3.1-8B-Instruct": {
        "file": "scale_gsm8k_llama-8b-inst.json",
        "params": 8.0,
        "arch": "dense",
    },
    "Qwen3-32B": {"file": "scale_gsm8k_qwen3-32b.json", "params": 32.0, "arch": "dense"},
    "Qwen3.5-27B": {"file": "scale_gsm8k_qwen3.5-27b.json", "params": 27.0, "arch": "dense"},
    "gpt-oss-20B": {"file": "arch_gsm8k_gpt-oss-20b.json", "params": 20.0, "arch": "moe"},
    "Qwen3-30B-MoE": {"file": "moe_gsm8k_qwen3-30b-moe.json", "params": 30.0, "arch": "moe"},
    "Qwen3-30B-MoE-Inst": {"file": "moe_gsm8k_qwen3-30b-inst.json", "params": 30.0, "arch": "moe"},
    "DeepSeek-V3.1": {"file": "frontier_gsm8k_deepseek-v3.1.json", "params": 685.0, "arch": "moe"},
    "Nemotron-120B": {
        "file": "frontier_gsm8k_nemotron-120b.json",
        "params": 120.0,
        "arch": "dense",
    },
    "Qwen3-235B-MoE": {"file": "frontier_gsm8k_qwen3-235b.json", "params": 235.0, "arch": "moe"},
    "Kimi-K2-Thinking": {"file": "arch_gsm8k_kimi-k2.json", "params": 1000.0, "arch": "moe"},
}

# Threshold for recorded group-ZVF measurements only.
ZVF_THRESHOLD = 0.10

LAM_BOUND = 10.0


def _load_trace(fname: str) -> list[float]:
    try:
        from ._review27_rank import load_reward_trace
    except ImportError:
        from _review27_rank import load_reward_trace
    return load_reward_trace(TRACE_DIR / fname)


def _load_step_log(fname: str) -> list[dict]:
    """Return per-step records with zvf/gu/reward fields, when present."""
    fp = TRACE_DIR / fname
    if not fp.exists():
        return []
    obj = json.loads(fp.read_text())
    log = obj.get("step_log") or []
    return [s for s in log if isinstance(s, dict)]


def _sat(t, rmax, lam):
    return rmax * (1.0 - np.exp(-lam * np.asarray(t, dtype=float)))


def _fit_sat(t, y, lam_bound=LAM_BOUND):
    """Fit canonical saturation R(t) = R_max * (1 - exp(-lambda * t))."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if (
        t.ndim != 1
        or y.shape != t.shape
        or len(t) < 3
        or not np.isfinite(t).all()
        or not np.isfinite(y).all()
        or np.ptp(y) < 1e-6
    ):
        return (math.nan, math.nan, math.nan, math.nan)
    try:
        p0 = (float(np.mean(y)), 1.0)
        popt, _ = curve_fit(
            lambda tt, rm, lm: _sat(tt, rm, lm),
            t,
            y,
            p0=p0,
            bounds=([0.0, 1e-4], [1.5, lam_bound]),
            maxfev=20000,
        )
        rmax, lam = float(popt[0]), float(popt[1])
        pred = _sat(t, rmax, lam)
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - float(np.sum(resid**2)) / max(1e-12, ss_tot)
        return rmax, lam, rmse, r2
    except Exception:
        return (math.nan, math.nan, math.nan, math.nan)


def _zvf_proxy(rewards: list[float]) -> dict[str, float]:
    """Fraction of extreme step means; not an estimator of group ZVF."""
    if not rewards:
        return dict(
            p_zero=math.nan,
            p_one=math.nan,
            fraction_extreme_step_means=math.nan,
            p_extreme=math.nan,
            p_mid=math.nan,
            mean_r=math.nan,
            var_r=math.nan,
        )
    arr = np.asarray(rewards, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError("finite reward probabilities required")
    p_zero = float(np.mean(arr <= 0.0 + 1e-9))
    p_one = float(np.mean(arr >= 1.0 - 1e-9))
    p_extreme = p_zero + p_one
    p_mid = float(np.mean((arr > 0.0 + 1e-9) & (arr < 1.0 - 1e-9)))
    return dict(
        p_zero=p_zero,
        p_one=p_one,
        fraction_extreme_step_means=p_extreme,
        p_extreme=p_extreme,
        p_mid=p_mid,
        mean_r=float(np.mean(arr)),
        var_r=float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
    )


def _ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym))) / den
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = float(np.sum(resid**2)) / max(1, len(x) - 2)
    se_b = float(math.sqrt(s2 / den)) if den > 0 else float("nan")
    return a, b, se_b, float(np.sum(resid**2))


# ----------------------------------------------------------------------
# (1) Per-trace ZVF proxy
# ----------------------------------------------------------------------
def artefact_zvf_proxy() -> Path:
    rows = []
    for label, meta in MODELS.items():
        rew = _load_trace(meta["file"])
        log = _load_step_log(meta["file"])
        z = _zvf_proxy(rew)
        # Per-step empirical ZVF from the trace's step_log (when present):
        # this is the direct analogue of Pillar 2's rollout-level ZVF,
        # measured at the per-step aggregation scale.
        zvf_per_step = [float(s["zvf"]) for s in log if s.get("zvf") is not None]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in zvf_per_step):
            raise ValueError("recorded group ZVF must be finite and in [0, 1]")
        gu_per_step = [float(s.get("gu", 0.0)) for s in log if "gu" in s]
        z.update(
            zvf_per_step_mean=(float(np.mean(zvf_per_step)) if zvf_per_step else float("nan")),
            zvf_per_step_max=(float(np.max(zvf_per_step)) if zvf_per_step else float("nan")),
            gu_per_step_mean=(float(np.mean(gu_per_step)) if gu_per_step else float("nan")),
            has_step_zvf=bool(zvf_per_step),
            n_recorded_zvf_steps=len(zvf_per_step),
        )
        rows.append(
            {
                "model": label,
                "params_B": meta["params"],
                "arch": meta["arch"],
                "n_steps": len(rew),
                **z,
            }
        )
    out = RESULTS_DIR / "scaling_law_iter61_zvf_proxy.tsv"
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[iter61] wrote {out}  ({len(rows)} rows)")
    return out


# ----------------------------------------------------------------------
# (2) Stratified saturation fit
# ----------------------------------------------------------------------
def artefact_stratum_fit(zvf_path: Path) -> Path:
    proxy = list(csv.DictReader(zvf_path.open(), delimiter="\t"))
    high, low = [], []
    for row in proxy:
        if not math.isfinite(float(row["zvf_per_step_mean"])):
            continue
        stratum = "HIGH" if float(row["zvf_per_step_mean"]) >= ZVF_THRESHOLD else "LOW"
        (high if stratum == "HIGH" else low).append(row["model"])

    rows = []
    for stratum_name, stratum_models in [
        ("HIGH_zvf", high),
        ("LOW_zvf", low),
        ("ALL", [r["model"] for r in proxy]),
    ]:
        lam_at_bound = 0
        lam_below = 0
        rmses = []
        lam_list = []
        for label in stratum_models:
            meta = MODELS[label]
            rew = _load_trace(meta["file"])
            if len(rew) < 3:
                continue
            t = np.arange(1, len(rew) + 1, dtype=float)
            rmax, lam, rmse, r2 = _fit_sat(t, rew)
            if not all(math.isfinite(v) for v in (rmax, lam, rmse)):
                continue
            lam_at_bound += int(abs(lam - LAM_BOUND) < 1e-3)
            lam_below += int(abs(lam - LAM_BOUND) >= 1e-3)
            rmses.append(rmse)
            lam_list.append(lam)
        rows.append(
            {
                "stratum": stratum_name,
                "n_anchors": len(lam_list),
                "n_anchors_attempted": len(stratum_models),
                "n_lam_at_bound": lam_at_bound,
                "n_lam_below_bound": lam_below,
                "frac_at_bound": (lam_at_bound / len(lam_list) if lam_list else math.nan),
                "mean_rmse": float(np.mean(rmses)) if rmses else math.nan,
                "mean_lambda": float(np.mean(lam_list)) if lam_list else math.nan,
                "median_lambda": float(np.median(lam_list)) if lam_list else math.nan,
                "models": "|".join(stratum_models),
            }
        )
    out = RESULTS_DIR / "scaling_law_iter61_stratum_fit.tsv"
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[iter61] wrote {out}  (HIGH={len(high)}, LOW={len(low)})")
    return out


# ----------------------------------------------------------------------
# (3) Jackknife stability of cross-scale slope
# ----------------------------------------------------------------------
def artefact_jackknife() -> Path:
    """Leave-one-out refit of the iter-49 two-parameter joint fit
    R_max = a*log10(P) + b*log10(C) + c, where C = P * T."""
    rows = []
    all_data = []
    for label, meta in MODELS.items():
        rew = _load_trace(meta["file"])
        if len(rew) < 3:
            continue
        t = np.arange(1, len(rew) + 1, dtype=float)
        rmax, lam, rmse, r2 = _fit_sat(t, rew)
        if not all(math.isfinite(v) for v in (rmax, lam, rmse)):
            continue
        # Fitted saturation target; no quantile substitution for an asymptote.
        rmax_robust = rmax
        all_data.append(
            {
                "model": label,
                "params": meta["params"],
                "arch": meta["arch"],
                "n_steps": len(rew),
                "rmax_robust": rmax_robust,
                "log10P": math.log10(meta["params"]),
                "log10C": math.log10(meta["params"] * len(rew)),
            }
        )

    if len(all_data) < 4:
        raise ValueError("jackknife requires at least four successful fits")
    # full fit
    X_full = np.array([[d["log10P"], d["log10C"], 1.0] for d in all_data])
    Y_full = np.array([d["rmax_robust"] for d in all_data])
    beta_full, *_ = np.linalg.lstsq(X_full, Y_full, rcond=None)
    a_full, b_full, c_full = beta_full

    for i, held in enumerate(all_data):
        train = [d for j, d in enumerate(all_data) if j != i]
        X = np.array([[d["log10P"], d["log10C"], 1.0] for d in train])
        Y = np.array([d["rmax_robust"] for d in train])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        a, b, c = beta
        pred_held = a * held["log10P"] + b * held["log10C"] + c
        rows.append(
            {
                "held_out": held["model"],
                "params_B": held["params"],
                "arch": held["arch"],
                "rmax_fitted": held["rmax_robust"],
                "rmax_predicted": float(pred_held),
                "residual": float(held["rmax_robust"] - pred_held),
                "abs_residual": float(abs(held["rmax_robust"] - pred_held)),
                "alpha_logP_loo": float(a),
                "alpha_logC_loo": float(b),
                "intercept_loo": float(c),
                "delta_alpha_logP": float(a - a_full),
                "delta_alpha_logC": float(b - b_full),
            }
        )
    out = RESULTS_DIR / "scaling_law_iter61_jackknife.tsv"
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[iter61] wrote {out}  (full alpha_logP={a_full:.3f}, alpha_logC={b_full:.3f})")
    return out


# ----------------------------------------------------------------------
# (4) Cross-pillar ZVF prediction
# ----------------------------------------------------------------------
def artefact_cross_pillar(zvf_path: Path) -> Path:
    """Export separate-cohort descriptive means without a compatibility claim."""
    source = REPO / "experiments/results/tinker_gsm8k_zvf_summary.json"
    summary = json.loads(source.read_text()) if source.exists() else {}
    proxy = list(csv.DictReader(zvf_path.open(), delimiter="\t"))
    finite = [
        float(r["zvf_per_step_mean"]) for r in proxy if math.isfinite(float(r["zvf_per_step_mean"]))
    ]
    pzero = [float(r["p_zero"]) for r in proxy if math.isfinite(float(r["p_zero"]))]
    rows = []
    for source_name, metric, value, n, unit in (
        (
            "Pillar2_tinker_G8",
            "mean_zvf_per_group",
            summary.get("mean_zvf", math.nan),
            summary.get("n_problems_total", 0),
            "group_rollout",
        ),
        (
            "Pillar1_iter61_step_zvf",
            "mean_recorded_group_zvf_per_step",
            np.mean(finite) if finite else math.nan,
            len(finite),
            "anchor_mean",
        ),
        (
            "Pillar1_iter61_step_zvf",
            "median_recorded_group_zvf_per_step",
            np.median(finite) if finite else math.nan,
            len(finite),
            "anchor_mean",
        ),
        (
            "Pillar1_iter61_step_zvf",
            "mean_p_zero_per_step",
            np.mean(pzero) if pzero else math.nan,
            len(pzero),
            "anchor_mean",
        ),
        (
            "Pillar1_iter61_step_zvf",
            "median_p_zero_per_step",
            np.median(pzero) if pzero else math.nan,
            len(pzero),
            "anchor_mean",
        ),
    ):
        rows.append(
            dict(
                source=source_name,
                metric=metric,
                value=value,
                n=n,
                scale=unit,
                comparison_status="unavailable_protocol_match",
            )
        )
    out = RESULTS_DIR / "scaling_law_iter61_cross_pillar.tsv"
    with out.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return out


# ----------------------------------------------------------------------
# (5) Predictions table
# ----------------------------------------------------------------------
def artefact_predictions(stratum_path: Path, jack_path: Path, cross_path: Path) -> Path:
    stratum = list(csv.DictReader(stratum_path.open(), delimiter="\t"))
    jack = list(csv.DictReader(jack_path.open(), delimiter="\t"))
    cross = list(csv.DictReader(cross_path.open(), delimiter="\t"))

    high = next(r for r in stratum if r["stratum"] == "HIGH_zvf")
    low = next(r for r in stratum if r["stratum"] == "LOW_zvf")

    abs_residuals = sorted(
        [(r["held_out"], float(r["abs_residual"])) for r in jack], key=lambda x: -x[1]
    )

    predictions = [
        (
            "P1_high_stratum_more_bound",
            "HIGH-ZVF stratum has higher bound-degenerate rate than LOW-ZVF",
            float(high["frac_at_bound"]) > float(low["frac_at_bound"]),
            f"{float(high['frac_at_bound']):.2f} > {float(low['frac_at_bound']):.2f}",
            "HIGH > LOW",
        ),
        (
            "P2_nemotron_in_high_floor",
            "Nemotron-120B has recorded mean group ZVF at or above the threshold",
            "Nemotron-120B" in high["models"].split("|"),
            "Nemotron-120B" in high["models"].split("|"),
            True,
        ),
        (
            "P3_nemotron_largest_jackknife_residual",
            "Nemotron-120B has the largest |LOO residual| in the cross-scale fit",
            (abs_residuals[0][0] if abs_residuals else "unavailable") == "Nemotron-120B",
            (abs_residuals[0][0] if abs_residuals else "unavailable"),
            "Nemotron-120B",
        ),
        (
            "P4_nemotron_alpha_logP_shift",
            "Removing Nemotron-120B shifts alpha_logP (signed, in same direction as the full-fit residual)",
            float(next((r["residual"] for r in jack if r["held_out"] == "Nemotron-120B"), 0.0))
            * float(
                next((r["delta_alpha_logP"] for r in jack if r["held_out"] == "Nemotron-120B"), 0.0)
            )
            > 0,
            next(
                (float(r["delta_alpha_logP"]) for r in jack if r["held_out"] == "Nemotron-120B"),
                float("nan"),
            ),
            "sign(residual)*sign(shift)>0",
        ),
        (
            "P5_step_zvf_above_rollout_zvf",
            "Cross-cohort ZVF comparison requires matching model, task, and group protocol",
            float(
                next(r["value"] for r in cross if r["metric"] == "mean_recorded_group_zvf_per_step")
            )
            > float(next(r["value"] for r in cross if r["metric"] == "mean_zvf_per_group")),
            float(
                next(r["value"] for r in cross if r["metric"] == "mean_recorded_group_zvf_per_step")
            ),
            "> rollout-level",
        ),
    ]
    rows = []
    for pid, claim, passed, observed, expected in predictions:
        rows.append(
            {
                "prediction_id": pid,
                "claim": claim,
                "observed": observed,
                "expected": expected,
                "pass_fail": (
                    "UNAVAILABLE"
                    if pid.startswith("P5_")
                    or (
                        pid.startswith(("P2_", "P3_", "P4_"))
                        and not any(r["held_out"] == "Nemotron-120B" for r in jack)
                    )
                    or (
                        pid.startswith("P1_")
                        and not all(math.isfinite(float(r["frac_at_bound"])) for r in (high, low))
                    )
                    else "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )
    out = RESULTS_DIR / "scaling_law_iter61_predictions.tsv"
    with out.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(
        f"[iter61] wrote {out}  ({sum(1 for r in rows if r['pass_fail'] == 'PASS')}/{len(rows)} PASS)"
    )
    return out


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_all(zvf_path: Path, stratum_path: Path, jack_path: Path, cross_path: Path) -> Path:
    proxy = list(csv.DictReader(zvf_path.open(), delimiter="\t"))
    stratum = list(csv.DictReader(stratum_path.open(), delimiter="\t"))
    jack = list(csv.DictReader(jack_path.open(), delimiter="\t"))
    cross = list(csv.DictReader(cross_path.open(), delimiter="\t"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) ZVF proxy vs lambda-bound status
    ax = axes[0, 0]
    colors = []
    for r in proxy:
        rew = _load_trace(MODELS[r["model"]]["file"])
        t = np.arange(1, len(rew) + 1, dtype=float)
        _, lam, _, _ = _fit_sat(t, rew)
        colors.append("tab:red" if abs(lam - LAM_BOUND) < 1e-3 else "tab:green")
    xs = [float(r["zvf_per_step_mean"]) for r in proxy]
    ys = [float(r["mean_r"]) for r in proxy]
    ax.scatter(xs, ys, c=colors, s=80, edgecolor="black")
    for r, x, y in zip(proxy, xs, ys, strict=False):
        ax.annotate(
            r["model"].replace("Instruct", "-Inst"),
            (x, y),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axvline(ZVF_THRESHOLD, color="grey", ls="--", lw=1, label=f"ZVF threshold = {ZVF_THRESHOLD}")
    ax.set_xlabel("Recorded mean group ZVF")
    ax.set_ylabel("Mean reward")
    ax.set_title("(A) Recorded group ZVF vs mean reward")
    ax.legend(loc="lower left", fontsize=8)

    # (B) Stratum fit: fraction at bound
    ax = axes[0, 1]
    s_labels = [r["stratum"] for r in stratum]
    s_frac = [float(r["frac_at_bound"]) for r in stratum]
    s_n = [int(r["n_anchors"]) for r in stratum]
    bars = ax.bar(s_labels, s_frac, color=["tab:red", "tab:green", "tab:blue"])
    for b, n, f in zip(bars, s_n, s_frac, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.02, f"n={n}, {f:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of anchors with λ at upper bound")
    ax.set_title("(B) Saturation-bound degeneracy by ZVF stratum")

    # (C) Jackknife LOO residual per held-out anchor
    ax = axes[1, 0]
    labels = [r["held_out"].replace("Instruct", "-Inst") for r in jack]
    abs_res = [float(r["abs_residual"]) for r in jack]
    sorted_idx = sorted(range(len(labels)), key=lambda i: -abs_res[i])
    labels_sorted = [labels[i] for i in sorted_idx]
    abs_res_sorted = [abs_res[i] for i in sorted_idx]
    colors_sorted = [
        "tab:red" if lbl.startswith("Nemotron") else "tab:blue" for lbl in labels_sorted
    ]
    ax.barh(range(len(labels_sorted)), abs_res_sorted, color=colors_sorted)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xlabel("|LOO residual|  Fitted R_max − LOO predicted R_max")
    ax.set_title("(C) Jackknife LOO residual per held-out anchor")

    # (D) Cross-pillar comparison: rollout ZVF vs step ZVF
    ax = axes[1, 1]
    p2_rows = [r for r in cross if r["source"] == "Pillar2_tinker_G8"]
    p1_rows = [r for r in cross if r["source"] == "Pillar1_iter61_step_zvf"]
    p2_labels = [r["metric"] for r in p2_rows]
    p2_vals = [float(r["value"]) for r in p2_rows]
    p1_labels = [r["metric"] for r in p1_rows]
    p1_vals = [float(r["value"]) for r in p1_rows]
    short_p2 = ["zvf"]
    short_p1 = ["step_zvf", "step_med_zvf", "p_zero", "med_p_zero"]
    width = 0.35
    x_p2 = np.arange(len(p2_vals))
    x_p1 = np.arange(len(p1_vals)) + len(p2_vals) + 1
    ax.bar(x_p2, p2_vals, width, color="tab:purple", label="Pillar 2 (rollout)")
    ax.bar(x_p1, p1_vals, width, color="tab:orange", label="Pillar 1 (step)")
    ax.set_xticks(list(x_p2) + list(x_p1))
    ax.set_xticklabels(short_p2 + short_p1, rotation=30, fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("(D) Cross-pillar ZVF: rollout-level vs step-level")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Iter 61 — Recorded ZVF and saturation optimizer diagnostics (Pillar 1 × Pillar 2)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = FIG_DIR / "scaling_law_iter61.pdf"
    out_png = FIG_DIR / "scaling_law_iter61.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    # mirror to paper/figures
    (PAPER_FIG / "scaling_law_iter61.pdf").write_bytes(out_pdf.read_bytes())
    print(f"[iter61] wrote {out_pdf}")
    return out_pdf


# ----------------------------------------------------------------------
# LaTeX section
# ----------------------------------------------------------------------
def write_section(
    zvf_path: Path,
    stratum_path: Path,
    jack_path: Path,
    cross_path: Path,
    pred_path: Path,
    fig_path: Path,
) -> Path:
    """Render only supplied observations, with no hard-coded causal verdict."""
    strata = list(csv.DictReader(stratum_path.open(), delimiter="\t"))
    preds = list(csv.DictReader(pred_path.open(), delimiter="\t"))
    lines = [
        r"\subsection{Exploratory saturation diagnostics (review27 v3)}",
        "Extreme step-mean fractions are not group ZVF. Only recorded group-level measurements define ZVF strata. Missing traces and failed fits are excluded with counts; these associations do not establish a causal mechanism.",
        r"\begin{tabular}{lrr}",
        r"Stratum & Attempted & Fitted \\",
    ]
    for row in strata:
        label = row["stratum"].replace("_", r"\_")
        lines.append(
            f"{label} & {row.get('n_anchors_attempted', row['n_anchors'])} & {row['n_anchors']} "
            + r"\\"
        )
    lines.append(r"\end{tabular}")
    for row in preds:
        label = row["prediction_id"].replace("_", r"\_")
        lines.append(f"Descriptive check {label}: {row['pass_fail']}.")
    out = PAPER_SEC / "scaling_law_iter61_review27_v3.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(output_dir=None, source_root=None):
    global REPO, CANDIDATE, FIG_DIR, PAPER_SEC
    global RESULTS_DIR, TRACE_DIR, PAPER_FIG
    if source_root is not None:
        REPO = Path(source_root)
    if output_dir is not None:
        CANDIDATE = Path(output_dir)
    RESULTS_DIR = CANDIDATE
    TRACE_DIR = REPO / "experiments/tinker-runs/results"
    PAPER_FIG = CANDIDATE / "paper_figures"
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    FIG_DIR, PAPER_SEC = CANDIDATE / "figures", CANDIDATE / "sections"
    for directory in (CANDIDATE, FIG_DIR, PAPER_SEC):
        directory.mkdir(parents=True, exist_ok=True)

    zvf_path = artefact_zvf_proxy()
    stratum_path = artefact_stratum_fit(zvf_path)
    jack_path = artefact_jackknife()
    cross_path = artefact_cross_pillar(zvf_path)
    pred_path = artefact_predictions(stratum_path, jack_path, cross_path)
    fig_path = plot_all(zvf_path, stratum_path, jack_path, cross_path)
    write_section(zvf_path, stratum_path, jack_path, cross_path, pred_path, fig_path)
    print("[iter61] all artefacts landed.")

    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(
        __file__,
        CANDIDATE,
        [TRACE_DIR / cfg["file"] for cfg in MODELS.values()]
        + [REPO / "experiments/results/tinker_gsm8k_zvf_summary.json"],
        seed=None,
    )


if __name__ == "__main__":
    main()
