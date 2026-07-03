"""Iter 60 — Pillar 4 (Length Bias / Dr.GRPO): Length-Elasticity of Reward.

Novel angle vs iter28/32/36/40/44/48/52/56: instead of measuring average
productivity R/L (iter56) or absolute length drift, iter60 measures the
MARGINAL reward produced per unit additional length -- the elasticity
dR/dL evaluated locally on the (L_t, R_t) trajectory. This is the
analogue of the price-elasticity-of-demand question applied to a learning
trajectory: how much extra reward does the policy buy with each extra
token?

Three deliverables:

(A) Per-step elasticity epsilon_i = DeltaR_i / DeltaL_i (with sign convention:
    epsilon>0 means the policy's marginal token PRODUCES reward, epsilon<0
    means the marginal token DESTROYS reward). Smoothing via centred window
    of width k=3 to suppress single-step R noise.

(B) Convexity test: fit R(L) = a * (L - L*)^2 + b + noise per (algo, seed).
    The optimum length L* is the length that MAXIMIZES reward; the curvature
    a captures how steeply the policy loses reward away from L*. Dr.GRPO
    should have a HIGHER L* (longer "natural length") and SIMILAR a
    (comparable sharpness around L*).

(C) Iso-reward length band: for each (algo, seed), find the range of L
    over which the policy achieves reward >= R_max - delta (delta=0.02).
    Dr.GRPO's band should be WIDER, indicating the policy wastes a larger
    range of lengths that don't improve reward.

Reads:
  experiments/results/drgrpo_vs_grpo.json      (arithmetic_easy: 5 seeds)
  experiments/results/drgrpo_gsm8k_cot_full.json (gsm8k_cot: 3 seeds)

Outputs (5 TSVs):
  experiments/results/length_bias_iter60_elasticity.tsv
  experiments/results/length_bias_iter60_curvature.tsv
  experiments/results/length_bias_iter60_iso_band.tsv
  experiments/results/length_bias_iter60_grpo_vs_drgrpo.tsv
  experiments/results/length_bias_iter60_summary.tsv
"""
from __future__ import annotations
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
DRGRPO_JSON = RES / "drgrpo_vs_grpo.json"
GSM_JSON = RES / "drgrpo_gsm8k_cot_full.json"

RNG_SEED = 60
N_BOOT = 4000
SMOOTH_K = 3  # centred window width for elasticity smoothing
ISO_DELTA = 0.02  # reward band tolerance


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx, my - (sxy / sxx) * mx


def load_runs(path: Path, task: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        ts = [int(s["step"]) for s in sl]
        rs = [float(s["mean_reward"]) for s in sl]
        ls = [float(s["mean_comp_len"]) for s in sl]
        zs = [float(s.get("zvf", float("nan"))) for s in sl]
        out.append({"task": task, "algo": r["algo"], "seed": r["seed"],
                    "t": ts, "R": rs, "L": ls, "zvf": zs})
    return out


def centred_smooth(xs: list[float], k: int = SMOOTH_K) -> list[float]:
    """Centred moving-average smoothing with window k (must be odd)."""
    n = len(xs)
    half = k // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def compute_elasticity(R: list[float], L: list[float]) -> list[float]:
    """Per-step marginal elasticity epsilon_i = DeltaR_i / DeltaL_i,
    where Delta is the centred difference (or one-sided at endpoints)."""
    n = len(R)
    eps = [0.0] * n
    for i in range(n):
        if i == 0:
            dR = R[1] - R[0]
            dL = L[1] - L[0]
        elif i == n - 1:
            dR = R[-1] - R[-2]
            dL = L[-1] - L[-2]
        else:
            dR = (R[i + 1] - R[i - 1]) / 2.0
            dL = (L[i + 1] - L[i - 1]) / 2.0
        if abs(dL) < 1e-6:
            # avoid div-by-zero: use a large finite value with same sign as dR
            eps[i] = float("inf") if dR > 0 else (
                float("-inf") if dR < 0 else 0.0)
        else:
            eps[i] = dR / dL
    return eps


def fit_quadratic_R_of_L(R: list[float], L: list[float]) -> dict:
    """Fit R = a*(L - L*)^2 + b by closed-form least squares on
    {L -> R}. Returns coefficients and R^2 fit quality."""
    n = len(R)
    # Parametrise R = alpha * L^2 + beta * L + gamma
    # OLS: [L^2, L, 1] -> R
    A = [[ls * ls, ls, 1.0] for ls in L]
# Solve via normal equations A^T A x = A^T b
    AtA = [[0.0] * 3 for _ in range(3)]
    Atb = [0.0, 0.0, 0.0]
    for i in range(n):
        for r in range(3):
            for c in range(3):
                AtA[r][c] += A[i][r] * A[i][c]
            Atb[r] += A[i][r] * R[i]
    # Invert 3x3 via cofactors (small matrix)
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det3(AtA)
    if abs(d) < 1e-12:
        return {"a": 0.0, "L_star": float("nan"), "b": 0.0,
                "R_max": 0.0, "R2": 0.0, "n_pts": n}
    cof = [[0.0] * 3 for _ in range(3)]
    cof[0][0] = AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]
    cof[0][1] = -(AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0])
    cof[0][2] = AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]
    cof[1][0] = -(AtA[0][1] * AtA[2][2] - AtA[0][2] * AtA[2][1])
    cof[1][1] = AtA[0][0] * AtA[2][2] - AtA[0][2] * AtA[2][0]
    cof[1][2] = -(AtA[0][0] * AtA[2][1] - AtA[0][1] * AtA[2][0])
    cof[2][0] = AtA[0][1] * AtA[1][2] - AtA[0][2] * AtA[1][1]
    cof[2][1] = -(AtA[0][0] * AtA[1][2] - AtA[0][2] * AtA[1][0])
    cof[2][2] = AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0]
    inv = [[cof[c][r] / d for c in range(3)] for r in range(3)]
    x = [sum(inv[r][c] * Atb[c] for c in range(3)) for r in range(3)]
    alpha, beta, gamma = x
    # R = a*(L-L*)^2 + b ==> alpha=a, -2 a L*=beta => L*=-beta/(2 a)
    a = alpha
    if abs(a) < 1e-12:
        L_star = float("nan")
        R_max = max(R)
    else:
        L_star = -beta / (2.0 * a)
        R_max = gamma - beta * beta / (4.0 * a)  # max of the parabola
    # R^2 of the quadratic fit
    R_pred = [alpha * ls * ls + beta * ls + gamma for ls in L]
    ss_res = sum((R[i] - R_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((R[i] - sum(R) / n) ** 2 for i in range(n))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": a, "L_star": L_star, "b": gamma,
            "R_max": R_max, "R2": R2, "n_pts": n}


def iso_reward_band(R: list[float], L: list[float],
                    delta: float = ISO_DELTA) -> dict:
    """Range of L values where R >= R_max - delta."""
    R_max = max(R)
    threshold = R_max - delta
    in_band = [(R[i] >= threshold, L[i]) for i in range(len(R))]
    if not any(b for b, _ in in_band):
        return {"L_min": float("nan"), "L_max": float("nan"),
                "width": 0.0, "n_pts": 0, "R_max": R_max,
                "delta": delta}
    Ls = [L[i] for i in range(len(R)) if R[i] >= threshold]
    return {"L_min": min(Ls), "L_max": max(Ls),
            "width": max(Ls) - min(Ls), "n_pts": len(Ls),
            "R_max": R_max, "delta": delta}


def paired_bootstrap(g: list[float], d: list[float], n_boot: int = N_BOOT,
                     rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(RNG_SEED)
    diffs = [di - gi for gi, di in zip(g, d)]
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "sd_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_le0": 1.0, "n_pairs": 0}
    mean_diff = sum(diffs) / n
    var = sum((x - mean_diff) ** 2 for x in diffs) / max(1, n - 1)
    sd_diff = math.sqrt(var)
    idx = list(range(n))
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.choice(idx)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {
        "mean_diff": round(mean_diff, 8),
        "sd_diff": round(sd_diff, 8),
        "ci_lo": round(boots[int(0.025 * n_boot)], 8),
        "ci_hi": round(boots[int(0.975 * n_boot)], 8),
        "p_le0": round((sum(1 for b in boots if b <= 0) + 1) / (n_boot + 1), 4),
        "n_pairs": n,
    }


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(x)
    except Exception:
        return False


def main() -> None:
    random.seed(RNG_SEED)
    rng = random.Random(RNG_SEED)

    rows = []
    rows.extend(load_runs(DRGRPO_JSON, "arithmetic_easy"))
    rows.extend(load_runs(GSM_JSON, "gsm8k_cot"))

    # ---- A. Per-run elasticity
    elast_rows = []
    for r in rows:
        R = r["R"]; L = r["L"]; t = r["t"]
        R_smooth = centred_smooth(R, SMOOTH_K)
        L_smooth = centred_smooth(L, SMOOTH_K)
        eps = compute_elasticity(R_smooth, L_smooth)
        finite_eps = [e for e in eps if _is_finite(e)]
        mean_eps = sum(finite_eps) / len(finite_eps) if finite_eps else 0.0
        median_eps = statistics.median(finite_eps) if finite_eps else 0.0
        # fraction of positive elasticities (marginal token produces reward)
        pos_frac = sum(1 for e in finite_eps if e > 0) / len(finite_eps) \
            if finite_eps else 0.0
        # fraction negative (marginal token destroys reward)
        neg_frac = sum(1 for e in finite_eps if e < 0) / len(finite_eps) \
            if finite_eps else 0.0
        elast_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "mean_eps": round(mean_eps, 8),
            "median_eps": round(median_eps, 8),
            "pos_frac": round(pos_frac, 6),
            "neg_frac": round(neg_frac, 6),
            "n_steps": len(R),
            "L_first": round(L[0], 4),
            "L_last": round(L[-1], 4),
            "R_first": round(R[0], 4),
            "R_last": round(R[-1], 4),
        })
    write_tsv(RES / "length_bias_iter60_elasticity.tsv", elast_rows,
              fieldnames=list(elast_rows[0].keys()))

    # ---- B. Quadratic R(L) fit per (algo, seed)
    curv_rows = []
    for r in rows:
        R = r["R"]; L = r["L"]
        fit = fit_quadratic_R_of_L(R, L)
        curv_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "a": round(fit["a"], 10),
            "L_star": round(fit["L_star"], 6) if _is_finite(fit["L_star"])
            else "nan",
            "b": round(fit["b"], 6),
            "R_max_fit": round(fit["R_max"], 6),
            "R2": round(fit["R2"], 6),
            "n_pts": fit["n_pts"],
            "L_first": round(L[0], 4),
            "L_last": round(L[-1], 4),
        })
    write_tsv(RES / "length_bias_iter60_curvature.tsv", curv_rows,
              fieldnames=list(curv_rows[0].keys()))

    # ---- C. Iso-reward length band
    band_rows = []
    for r in rows:
        R = r["R"]; L = r["L"]
        b = iso_reward_band(R, L, delta=ISO_DELTA)
        band_rows.append({
            "task": r["task"], "algo": r["algo"], "seed": r["seed"],
            "L_min": round(b["L_min"], 4) if _is_finite(b["L_min"]) else "nan",
            "L_max": round(b["L_max"], 4) if _is_finite(b["L_max"]) else "nan",
            "width": round(b["width"], 4) if _is_finite(b["width"]) else "nan",
            "n_pts": b["n_pts"],
            "R_max": round(b["R_max"], 4),
            "delta": b["delta"],
        })
    write_tsv(RES / "length_bias_iter60_iso_band.tsv", band_rows,
              fieldnames=list(band_rows[0].keys()))

    # ---- Paired bootstrap across (A), (B), (C)
    paired = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        for metric in ("mean_eps", "median_eps", "pos_frac", "neg_frac"):
            sub = elast_rows
            grpo = {r["seed"]: r for r in sub if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in sub if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [grpo[s][metric] for s in common]
            dv = [drgrpo[s][metric] for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            interp = "inconclusive"
            if metric == "pos_frac" and boot["mean_diff"] < 0 and boot["ci_hi"] < 0:
                interp = "Dr.GRPO fewer positive-elasticity steps"
            elif metric == "neg_frac" and boot["mean_diff"] > 0 and boot["ci_lo"] > 0:
                interp = "Dr.GRPO more negative-elasticity steps"
            elif metric == "mean_eps" and boot["mean_diff"] < 0 and boot["ci_hi"] < 0:
                interp = "Dr.GRPO marginal-token productivity lower"
            paired.append({
                "task": task, "metric": metric, "kind": "elasticity",
                "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 8),
                "mean_drgrpo": round(sum(dv) / len(dv), 8),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
                "interpretation": interp,
            })
        for metric in ("L_star", "a", "R_max_fit", "R2"):
            sub = curv_rows
            grpo = {r["seed"]: r for r in sub if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in sub if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [float(grpo[s][metric]) if grpo[s][metric] != "nan" else 0.0
                  for s in common]
            dv = [float(drgrpo[s][metric]) if drgrpo[s][metric] != "nan" else 0.0
                  for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            interp = "inconclusive"
            if metric == "L_star" and boot["mean_diff"] > 0 and boot["ci_lo"] > 0:
                interp = "Dr.GRPO optimal length L* larger"
            elif metric == "R2" and boot["mean_diff"] < 0 and boot["ci_hi"] < 0:
                interp = "Dr.GRPO R(L) less quadratic"
            paired.append({
                "task": task, "metric": metric, "kind": "curvature",
                "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 8),
                "mean_drgrpo": round(sum(dv) / len(dv), 8),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
                "interpretation": interp,
            })
        for metric in ("width", "n_pts"):
            sub = band_rows
            grpo = {r["seed"]: r for r in sub if r["task"] == task
                    and r["algo"] == "grpo"}
            drgrpo = {r["seed"]: r for r in sub if r["task"] == task
                      and r["algo"] in ("dr_grpo", "drgrpo")}
            common = sorted(set(grpo) & set(drgrpo))
            if not common:
                continue
            gv = [float(grpo[s][metric]) if grpo[s][metric] != "nan" else 0.0
                  for s in common]
            dv = [float(drgrpo[s][metric]) if drgrpo[s][metric] != "nan" else 0.0
                  for s in common]
            boot = paired_bootstrap(gv, dv, rng=rng)
            interp = "inconclusive"
            if metric == "width" and boot["mean_diff"] > 0 and boot["ci_lo"] > 0:
                interp = "Dr.GRPO iso-reward length band wider"
            paired.append({
                "task": task, "metric": metric, "kind": "iso_band",
                "n_pairs": boot["n_pairs"],
                "mean_grpo": round(sum(gv) / len(gv), 6),
                "mean_drgrpo": round(sum(dv) / len(dv), 6),
                "mean_diff": boot["mean_diff"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "p_le0": boot["p_le0"],
                "interpretation": interp,
            })
    write_tsv(RES / "length_bias_iter60_grpo_vs_drgrpo.tsv", paired,
              fieldnames=list(paired[0].keys()))

    # ---- Summary rollup -- the strongest cross-task findings
    summary_rows = []
    for p in paired:
        summary_rows.append({
            "task": p["task"], "kind": p["kind"], "metric": p["metric"],
            "n_pairs": p["n_pairs"],
            "mean_grpo": p["mean_grpo"],
            "mean_drgrpo": p["mean_drgrpo"],
            "mean_diff": p["mean_diff"],
            "ci_lo": p["ci_lo"],
            "ci_hi": p["ci_hi"],
            "p_le0": p["p_le0"],
            "interpretation": p["interpretation"],
        })
    write_tsv(RES / "length_bias_iter60_summary.tsv", summary_rows,
              fieldnames=list(summary_rows[0].keys()))

    print("=== iter60 length elasticity summary ===")
    for r in summary_rows:
        print(r)
    print("\n=== per-run elasticity ===")
    for r in elast_rows:
        print(r)
    print("\n=== per-run curvature fit ===")
    for r in curv_rows:
        print(r)
    print("\n=== per-run iso-reward length band ===")
    for r in band_rows:
        print(r)


if __name__ == "__main__":
    main()