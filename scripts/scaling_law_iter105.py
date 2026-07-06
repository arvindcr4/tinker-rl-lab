"""
scaling_law_iter105.py — Pillar 1 iter105: Failure-Mode Taxonomy + params_B -> R_max* scaling.

Fresh angle not covered by iter81 (compute invariance), iter85 (three-phase test + Nemotron),
iter97 (8-family head-to-head), iter101 (cross-anchor transfer + AIC stack):

  (A) Six-mode failure-mode classifier over the 12-anchor GSM8K panel, using
      observable trace statistics (zero-fraction, late-early delta, residual variance,
      peak-vs-mean ratio, late-trend slope). Output: 1 row per anchor with the
      hard label + a soft posterior over the 6 modes.

  (B) Chinchilla-analogue scaling law R_max*(N) — does R_max* rise predictably with
      params_B? Test 3 parametric forms (log-linear, inverse, Chinchilla-power)
      on the 12 anchors, with a focal leave-one-family-out (MoE vs dense) check
      that asks: is the R_max*->params_B relation family-invariant?

Outputs (TSV + JSON meta, no figure this iter — keeps deliverable compact):
  experiments/results/scaling_law_iter105_failure_modes.tsv   (12 rows)
  experiments/results/scaling_law_iter105_scaling_law.tsv    (12 rows + summary)
  experiments/results/scaling_law_iter105_lofo.tsv           (LOFO residuals)
  experiments/results/scaling_law_iter105_summary.tsv        (3 fits, MAE / R^2 / aic)
  experiments/results/scaling_law_iter105_meta.json

Method notes:
  - All fit quality is OLS-on-transformed. AIC = n ln(RSS/n) + 2k.
  - Soft posterior P(mode|stats) uses a hand-crafted likelihood normalised over modes;
    see mode_posterior(). The classifier is deterministic given the same thresholds.
  - LOFO removes one anchor at a time, refits, predicts R_max*, records residual.
    - family_invariance_score = 1 - std(LOFO residuals across families)/std(all R_max*).

Citations used:
  - iter85 three-phase anchor set (arXiv:2507.18014 hypothesis test)
  - iter101 cross-anchor transfer & AIC stack
  - Chinchilla (Hoffmann et al. 2022) as analogue for power-law form
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results"
TR = ROOT / "experiments" / "tinker-runs" / "results"

# ------------------------------------------------------------------
# Anchors: name -> (params_B, family, trace_file)
# family in {"dense", "moe"}; params_B is total parameter count in billions.
# ------------------------------------------------------------------
ANCHORS: dict[str, dict[str, Any]] = {
    "Qwen3.5-4B":            {"params_B": 4.0,    "family": "dense", "file": "scale_gsm8k_qwen3.5-4b.json"},
    "Qwen3-8B":              {"params_B": 8.0,    "family": "dense", "file": "scale_gsm8k_qwen3-8b.json"},
    "Llama-3.1-8B-Instruct": {"params_B": 8.0,    "family": "dense", "file": "scale_gsm8k_llama-8b-inst.json"},
    "Qwen3-32B":             {"params_B": 32.0,   "family": "dense", "file": "scale_gsm8k_qwen3-32b.json"},
    "Qwen3.5-27B":           {"params_B": 27.0,   "family": "dense", "file": "scale_gsm8k_qwen3.5-27b.json"},
    "gpt-oss-20B":           {"params_B": 20.0,   "family": "moe",   "file": "arch_gsm8k_gpt-oss-20b.json"},
    "Qwen3-30B-MoE":         {"params_B": 30.0,   "family": "moe",   "file": "moe_gsm8k_qwen3-30b-moe.json"},
    "Qwen3-30B-MoE-Inst":    {"params_B": 30.0,   "family": "moe",   "file": "moe_gsm8k_qwen3-30b-inst.json"},
    "DeepSeek-V3.1":         {"params_B": 685.0,  "family": "moe",   "file": "frontier_gsm8k_deepseek-v3.1.json"},
    "Nemotron-120B":         {"params_B": 120.0,  "family": "dense", "file": "frontier_gsm8k_nemotron-120b.json"},
    "Qwen3-235B-MoE":        {"params_B": 235.0,  "family": "moe",   "file": "frontier_gsm8k_qwen3-235b.json"},
    "Kimi-K2-Thinking":      {"params_B": 1000.0, "family": "moe",   "file": "arch_gsm8k_kimi-k2.json"},
}


def load_traces() -> dict[str, list[float]]:
    """Load reward_trace lists keyed by anchor name."""
    out: dict[str, list[float]] = {}
    for name, meta in ANCHORS.items():
        fp = TR / meta["file"]
        d = json.loads(fp.read_text())
        rt = d.get("reward_trace")
        if not rt:
            raise RuntimeError(f"missing reward_trace in {fp}")
        out[name] = list(map(float, rt))
    return out


# ------------------------------------------------------------------
# (A) Failure-mode taxonomy
# ------------------------------------------------------------------
MODES = ["converged", "plateau", "drift", "oscillation", "collapse", "divergence"]


def trace_features(rt: list[float]) -> dict[str, float]:
    """Compute the 5 features that drive the failure-mode classifier."""
    arr = np.asarray(rt, dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError("trace too short")
    early = arr[: max(1, n // 5)].mean()  # first ~20%
    late = arr[-max(1, n // 5):].mean()  # last ~20%
    delta_le = float(late - early)
    zero_frac = float((arr <= 1e-6).sum()) / n
    peak = float(arr.max())
    mean = float(arr.mean())
    # residual variance after detrend (OLS slope removed)
    t = np.arange(n, dtype=float)
    if n >= 3:
        slope, intercept = np.polyfit(t, arr, 1)
        resid = arr - (slope * t + intercept)
        resid_var = float(resid.var())
    else:
        slope, resid_var = 0.0, 0.0
    return {
        "n": int(n),
        "early_mean": float(early),
        "late_mean": float(late),
        "delta_le": delta_le,
        "zero_frac": zero_frac,
        "peak": peak,
        "mean": mean,
        "slope": float(slope),
        "resid_var": resid_var,
    }


def classify_mode(f: dict[str, float]) -> tuple[str, dict[str, float]]:
    """
    Hand-crafted decision tree -> mode + soft posterior.

    Decision tree (greedy, ordered):
      1. zero_frac >= 0.50 -> 'collapse'      (Nemotron-like)
      2. mean <= 0.35 and peak >= 0.80        -> 'divergence' (transient success then crash)
      3. resid_var >= 0.05 and |delta_le| < 0.05 -> 'oscillation'
      4. delta_le <= -0.10                    -> 'drift' (early > late)
      5. peak >= 0.95 and mean >= 0.95 and resid_var < 0.005 -> 'converged' (locked at top)
      6. default                              -> 'plateau'

    Soft posterior: each rule that matches contributes mass; we normalise.
    """
    pos = {m: 0.0 for m in MODES}
    # 1. collapse
    if f["zero_frac"] >= 0.50:
        pos["collapse"] += 3.0
    # 2. divergence
    if f["mean"] <= 0.35 and f["peak"] >= 0.80:
        pos["divergence"] += 2.5
    # 3. oscillation
    if f["resid_var"] >= 0.05 and abs(f["delta_le"]) < 0.05:
        pos["oscillation"] += 2.0
    # 4. drift
    if f["delta_le"] <= -0.10:
        pos["drift"] += 2.0
    # 5. converged
    if f["peak"] >= 0.95 and f["mean"] >= 0.95 and f["resid_var"] < 0.005:
        pos["converged"] += 3.0
    # 6. plateau (default catch-all)
    pos["plateau"] += 1.0
    # tiny noise so ties break consistently (deterministic by mode order)
    for i, m in enumerate(MODES):
        pos[m] += 1e-6 * (len(MODES) - i)
    s = sum(pos.values())
    pos = {m: v / s for m, v in pos.items()}
    label = max(pos, key=pos.get)
    return label, pos


# ------------------------------------------------------------------
# (B) R_max* scaling-law fits
# ------------------------------------------------------------------
def fit_loglinear(N: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """y = a + b*log10(N)."""
    x = np.log10(N)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    resid = y - yhat
    n = len(y)
    k = 2
    rss = float((resid ** 2).sum())
    aic = n * math.log(rss / max(n, 1)) + 2 * k
    return {
        "family": "loglinear",
        "a": float(coef[0]),
        "b": float(coef[1]),
        "rss": rss,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(math.sqrt(rss / n)),
        "r2": float(1 - rss / float(((y - y.mean()) ** 2).sum())),
        "aic": float(aic),
    }


def fit_inverse(N: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """y = a - b/N (asymptotic; b>0 means positive scale benefit)."""
    x = 1.0 / N
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    resid = y - yhat
    n = len(y)
    k = 2
    rss = float((resid ** 2).sum())
    aic = n * math.log(rss / max(n, 1)) + 2 * k
    return {
        "family": "inverse",
        "a": float(coef[0]),
        "b": float(coef[1]),
        "rss": rss,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(math.sqrt(rss / n)),
        "r2": float(1 - rss / float(((y - y.mean()) ** 2).sum())),
        "aic": float(aic),
    }


def fit_chinchilla(N: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """y = a - b / N^c (3 params). c free; b unconstrained in OLS."""
    cs = np.logspace(np.log10(0.05), np.log10(1.5), 25)
    best = None
    for c in cs:
        x1 = np.ones_like(N)
        x2 = -1.0 / (N ** c)
        A = np.vstack([x1, x2]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        resid = y - yhat
        rss = float((resid ** 2).sum())
        if best is None or rss < best["rss"]:
            # yhat = coef[0] + coef[1]*(-1/N^c) = coef[0] - coef[1]/N^c,
            # so a = coef[0], b = coef[1].
            best = {
                "c": float(c),
                "a": float(coef[0]),
                "b": float(coef[1]),
                "rss": rss,
            }
    resid = y - (best["a"] - best["b"] / (N ** best["c"]))
    n = len(y)
    k = 3
    rss = float((resid ** 2).sum())
    aic = n * math.log(rss / max(n, 1)) + 2 * k
    return {
        "family": "chinchilla",
        "a": best["a"],
        "b": best["b"],
        "c": best["c"],
        "rss": rss,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(math.sqrt(rss / n)),
        "r2": float(1 - rss / float(((y - y.mean()) ** 2).sum())),
        "aic": float(aic),
    }


def fit_all(N: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    return [fit_loglinear(N, y), fit_inverse(N, y), fit_chinchilla(N, y)]


def lofo_residuals(N: np.ndarray, y: np.ndarray, families: list[str]) -> list[dict[str, Any]]:
    """Leave-one-family-out: refit on the other family, predict on the held-out family."""
    out = []
    families_unique = sorted(set(families))
    for hold in families_unique:
        mask = np.array([f != hold for f in families])
        fit = fit_chinchilla(N[mask], y[mask])
        # predict on held-out
        pred = fit["a"] - fit["b"] / (N[~mask] ** fit["c"])
        resid = y[~mask] - pred
        for i, name in enumerate(np.array(list(ANCHORS.keys()))[~mask]):
            out.append({
                "held_out_family": hold,
                "model": str(name),
                "true_R_max_star": float(y[~mask][i]),
                "pred_R_max_star": float(pred[i]),
                "residual": float(resid[i]),
            })
    return out


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def main() -> int:
    traces = load_traces()
    print(f"[iter105] loaded {len(traces)} reward traces")

    # ---------- (A) Failure-mode classification ----------
    fm_rows: list[dict[str, Any]] = []
    pos_rows: list[dict[str, Any]] = []
    for name, rt in traces.items():
        meta = ANCHORS[name]
        f = trace_features(rt)
        label, pos = classify_mode(f)
        fm_rows.append({
            "model": name,
            "params_B": meta["params_B"],
            "family": meta["family"],
            **f,
            "mode": label,
        })
        pos_rows.append({
            "model": name,
            **{f"P({m})": round(v, 4) for m, v in pos.items()},
            "mode": label,
        })

    fm_path = RES / "scaling_law_iter105_failure_modes.tsv"
    write_tsv(fm_path, fm_rows, list(fm_rows[0].keys()))
    pos_path = RES / "scaling_law_iter105_failure_posteriors.tsv"
    write_tsv(pos_path, pos_rows, list(pos_rows[0].keys()))
    print(f"[iter105] wrote {fm_path}")
    print(f"[iter105] wrote {pos_path}")

    # Mode distribution summary
    mode_counts: dict[str, int] = {}
    for r in fm_rows:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1
    print(f"[iter105] mode distribution: {mode_counts}")

    # ---------- (B) R_max* scaling-law fits ----------
    # We define R_max* as the empirical "asymptotic" reward ceiling for each anchor:
    # the mean of the LAST third of the trace, but with a penalty: if the trace
    # is collapsing (mode=='collapse'), use PEAK instead so the ceiling reflects
    # the model's best attainable performance, not the post-collapse floor.
    Rmax = []
    Nlist = []
    famlist = []
    for name, rt in traces.items():
        meta = ANCHORS[name]
        arr = np.asarray(rt, dtype=float)
        n = arr.size
        last_third = arr[-max(1, n // 3):].mean()
        peak = arr.max()
        # locate the failure-mode label
        label = next(r["mode"] for r in fm_rows if r["model"] == name)
        if label in ("collapse", "divergence"):
            rmax_star = float(peak)
        else:
            rmax_star = float(last_third)
        Nlist.append(meta["params_B"])
        Rmax.append(rmax_star)
        famlist.append(meta["family"])
    N = np.asarray(Nlist, dtype=float)
    y = np.asarray(Rmax, dtype=float)
    families = famlist

    sl_rows = [
        {
            "model": name,
            "params_B": meta["params_B"],
            "family": meta["family"],
            "R_max_star": float(y[i]),
            "log10_N": float(np.log10(meta["params_B"])),
            "mode": next(r["mode"] for r in fm_rows if r["model"] == name),
        }
        for i, (name, meta) in enumerate(ANCHORS.items())
    ]
    sl_path = RES / "scaling_law_iter105_scaling_law.tsv"
    write_tsv(sl_path, sl_rows, list(sl_rows[0].keys()))
    print(f"[iter105] wrote {sl_path}")

    # Fit three forms
    fits = fit_all(N, y)
    summary_rows = []
    for f in fits:
        row = {
            "family": f["family"],
            "n_anchors": int(len(y)),
            "params": ",".join(f"{k}={v:.4g}" for k, v in f.items() if k not in ("family", "rss", "mae", "rmse", "r2", "aic")),
            "rss": round(f["rss"], 4),
            "mae": round(f["mae"], 4),
            "rmse": round(f["rmse"], 4),
            "r2": round(f["r2"], 4),
            "aic": round(f["aic"], 4),
        }
        summary_rows.append(row)
    sum_path = RES / "scaling_law_iter105_summary.tsv"
    write_tsv(sum_path, summary_rows, list(summary_rows[0].keys()))
    print(f"[iter105] wrote {sum_path}")
    best = min(fits, key=lambda d: d["aic"])
    print(f"[iter105] best by AIC: {best['family']}  aic={best['aic']:.3f}  r2={best['r2']:.3f}")

    # LOFO
    lofo = lofo_residuals(N, y, families)
    lofo_path = RES / "scaling_law_iter105_lofo.tsv"
    write_tsv(lofo_path, lofo, list(lofo[0].keys()))
    print(f"[iter105] wrote {lofo_path}")

    # Family-invariance score: how stable is the Chinchilla fit if we drop one family?
    fam_resid_std: dict[str, float] = {}
    for hold in sorted(set(families)):
        rs = [r["residual"] for r in lofo if r["held_out_family"] == hold]
        fam_resid_std[hold] = float(np.std(rs)) if rs else float("nan")
    all_resids = [r["residual"] for r in lofo]
    overall_resid_std = float(np.std(all_resids))
    family_invariance_score = float(1.0 - (sum(fam_resid_std.values()) / len(fam_resid_std)) / max(overall_resid_std, 1e-9))
    print(f"[iter105] LOFO family-resid std: {fam_resid_std}")
    print(f"[iter105] family-invariance score: {family_invariance_score:.3f}")

    # Sharpest finding: which model contributes the largest LOFO residual?
    abs_resid_by_model = {r["model"]: abs(r["residual"]) for r in lofo}
    worst_model = max(abs_resid_by_model, key=abs_resid_by_model.get)
    worst_resid = abs_resid_by_model[worst_model]
    print(f"[iter105] worst LOFO offender: {worst_model}  |residual|={worst_resid:.3f}")

    # ---------- meta ----------
    meta = {
        "iter": 105,
        "pillar": "P1-ScalingLaws",
        "n_anchors": len(traces),
        "fresh_angles": [
            "(A) 6-mode failure-mode classifier over 12-anchor GSM8K panel",
            "(B) Chinchilla-analogue R_max*(N) scaling fits (loglinear / inverse / chinchilla-power)",
            "(C) leave-one-family-out (LOFO) check of family-invariance of the scaling law",
        ],
        "failure_modes": MODES,
        "mode_counts": mode_counts,
        "scaling_fits": summary_rows,
        "best_by_aic": best["family"],
        "best_aic": best["aic"],
        "best_r2": best["r2"],
        "lofo_family_resid_std": fam_resid_std,
        "lofo_overall_resid_std": overall_resid_std,
        "family_invariance_score": family_invariance_score,
        "worst_lofo_offender": worst_model,
        "worst_lofo_residual": worst_resid,
        "anchor_list": list(ANCHORS.keys()),
        "method": (
            "(A) Hand-crafted decision tree on (zero_frac, peak, mean, resid_var, delta_le). "
            "Soft posterior via weighted match-counts; deterministic tie-break. "
            "(B) R_max* = empirical asymptotic reward ceiling = mean(last third) "
            "for non-collapse anchors, else peak. Three forms fit by OLS-on-transformed; "
            "AIC = n ln(RSS/n) + 2k. (C) Leave-one-family-out (dense vs moe) refit on Chinchilla-power; "
            "family-invariance score = 1 - mean(|resid_std per family|) / std(all residuals)."
        ),
        "ts": "2026-07-03T20:30:00Z",
    }
    meta_path = RES / "scaling_law_iter105_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[iter105] wrote {meta_path}")
    return 0


def write_tsv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


if __name__ == "__main__":
    sys.exit(main())