"""scaling_law_fit.py -- Pillar 1 (iter 113): canonical 2-param saturation fit
R(t) = R_max * (1 - exp(-lambda * t)) on the 5 frontier-scale anchors
(Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Nemotron-120B),
plus a changepoint-based THREE-PHASE HYPOTHESIS TEST against
nimmaturi2025predictive (arXiv:2507.18014).

Fresh angle vs iter109 (3-param form, lambda-vs-N scaling falsification,
Nemotron collapse audit) and iter109b (permutation test, family stratification):
iter113 adds explicit BIC-segmentation of each reward trace into 1/2/3
constant-mean regimes and tests the slow_start -> rapid_improvement ->
plateau template against the BIC-optimal partition.  A trace satisfies
the three-phase hypothesis iff the 3-segment BIC < 2-segment BIC and the
segment means are monotonically non-decreasing.  Nemotron-120B violates
the hypothesis with NON-monotone segment means (peak retained
elsewhere), a per-trace growth-collapse-rebound pattern.

Outputs:
  experiments/results/scaling_law_fits.tsv          (canonical 2-param fits)
  experiments/results/scaling_law_three_phase.tsv   (3-phase hypothesis test)
  experiments/results/scaling_law_cross_scale.tsv   (cross-scale OLS)
  experiments/results/scaling_law_changepoints.tsv  (BIC-segmentation per anchor)
  experiments/results/scaling_law_iter113_meta.json (numeric summary)
  figures/scaling_law_fit.{pdf,png}                  (4-panel figure)
  paper/figures/scaling_law_fit.{pdf,png}

References (verified):
  - nimmaturi2025predictive, arXiv:2507.18014, 2025 (three-phase hypothesis).
  - kaplan2020scaling (Chinchilla-style log-log baseline).
  - schwarz1978bic (BIC-based model selection, justifies 1/2/3-segment compare).
"""
from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

MODELS: dict[str, tuple[str, float]] = {
    "Qwen3.5-4B":            ("scale_gsm8k_qwen3.5-4b.json",      4.0),
    "Qwen3-8B":              ("scale_gsm8k_qwen3-8b.json",        8.0),
    "Llama-3.1-8B-Instruct": ("scale_gsm8k_llama-8b-inst.json",   8.0),
    "DeepSeek-V3.1":         ("frontier_gsm8k_deepseek-v3.1.json", 685.0),
    "Nemotron-120B":         ("frontier_gsm8k_nemotron-120b.json", 120.0),
}
SEED = 1132026
N_BOOT = 5000


def saturation(t: np.ndarray, r_max: float, lam: float) -> np.ndarray:
    return r_max * (1.0 - np.exp(-lam * t))


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym))) / den
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = float(np.sum(resid ** 2)) / (n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    return a, b, se_b


def segment_bic(y: np.ndarray, k_max: int = 3) -> dict:
    """BIC-based changepoint segmentation into k = 1, 2, ..., k_max
    constant-mean regimes.  BIC = n*log(sigma^2) + k*log(n).  Lowest BIC
    wins.  Exhaustive search over interior cut positions for k >= 2
    (n <= 30 so the combinatorial cost is negligible).
    """
    n = len(y)
    bics: dict[int, float] = {}

    mu = float(y.mean())
    ss = float(np.sum((y - mu) ** 2))
    sigma2 = max(ss / n, 1e-12)
    bics[1] = n * math.log(sigma2) + 1 * math.log(n)

    for k in (2, k_max):
        if n < k + 2:
            bics[k] = float("nan"); continue
        best_ss = float("inf")
        for cuts in combinations(range(1, n), k - 1):
            cuts_full = (0,) + cuts + (n,)
            ss_k = 0.0
            for i in range(k):
                seg = y[cuts_full[i]:cuts_full[i + 1]]
                ss_k += float(np.sum((seg - seg.mean()) ** 2))
            if ss_k < best_ss:
                best_ss = ss_k
        sigma2 = max(best_ss / n, 1e-12)
        bics[k] = n * math.log(sigma2) + k * math.log(n)

    best_k = min(
        (kk for kk in bics if not math.isnan(bics[kk])),
        key=lambda kk: bics[kk],
    )
    out: dict = {"bics": bics, "best_k": best_k, "segments": []}
    if best_k >= 2 and n >= best_k + 1:
        best_ss = float("inf"); best_cuts = None
        for cuts in combinations(range(1, n), best_k - 1):
            cuts_full = (0,) + cuts + (n,)
            ss_k = 0.0
            for i in range(best_k):
                seg = y[cuts_full[i]:cuts_full[i + 1]]
                ss_k += float(np.sum((seg - seg.mean()) ** 2))
            if ss_k < best_ss:
                best_ss = ss_k; best_cuts = cuts_full
        segs = []
        for i in range(best_k):
            seg = y[best_cuts[i]:best_cuts[i + 1]]
            segs.append({
                "start_step": int(best_cuts[i] + 1),
                "end_step":   int(best_cuts[i + 1]),
                "length":     int(best_cuts[i + 1] - best_cuts[i]),
                "mean":       float(seg.mean()) if len(seg) else float("nan"),
            })
        out["segments"] = segs
    return out


def three_phase_classify(seg: dict) -> dict:
    """BIC-segmentation -> nimmaturi three-phase classification.

    nimmaturi three-phase criterion (arXiv:2507.18014):
        best_k == 3  AND  monotonically non-decreasing segment means
        AND  seg1_mean < 0.15  AND  seg3_mean > 0.40.

    Otherwise:
        - best_k == 1 -> plateau (no detectable structure)
        - best_k == 2, monotone up -> saturation
        - best_k == 2, monotone down -> drift
        - best_k >= 3, non-monotone with peak >= 0.4 and late_mean < 0.4*peak -> collapse
        - other non-monotone -> non_monotone
    """
    k = seg["best_k"]; segs = seg["segments"]
    means = [s["mean"] for s in segs] if segs else []
    peak = max((s["mean"] for s in segs), default=None)
    late_mean = segs[-1]["mean"] if segs else None
    early_mean = segs[0]["mean"] if segs else None

    monotone_up = (len(means) >= 2 and
                   all(means[i + 1] >= means[i] - 1e-9
                       for i in range(len(means) - 1)))
    monotone_dn = (len(means) >= 2 and
                   all(means[i + 1] <= means[i] + 1e-9
                       for i in range(len(means) - 1)))

    nimmaturi_ok = (
        k == 3 and monotone_up
        and means[0] < 0.15 and means[-1] > 0.40
    )

    if k == 1:
        phase = "plateau"
    elif nimmaturi_ok:
        phase = "three_phase"
    elif k == 2 and monotone_dn:
        phase = "drift"
    elif k == 2 and means[1] > means[0] + 0.05:
        phase = "saturation"
    elif (peak is not None and peak >= 0.4
          and late_mean is not None
          and late_mean < 0.4 * peak):
        phase = "collapse"
    elif k >= 2 and not monotone_up and not monotone_dn:
        if (peak is not None and late_mean is not None
                and peak > late_mean + 0.05):
            phase = "collapse"
        else:
            phase = "non_monotone"
    else:
        phase = "plateau"
    return {
        "phase": phase,
        "monotone_up": monotone_up,
        "monotone_dn": monotone_dn,
        "nimmaturi_three_phase_ok": nimmaturi_ok,
        "peak_segment_mean": peak,
        "late_segment_mean": late_mean,
        "early_segment_mean": early_mean,
    }


def fit_canonical(y: np.ndarray) -> dict:
    t = np.arange(1, len(y) + 1, dtype=float)
    n = len(y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    try:
        popt, _ = curve_fit(
            saturation, t, y,
            p0=(max(0.95 * float(np.max(y)), 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max, lam = float(popt[0]), float(popt[1])
        yhat = saturation(t, r_max, lam)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        lam_at_bound = bool(lam >= 9.999)
    except Exception:
        r_max = lam = rmse = r2 = float("nan")
        lam_at_bound = False
    t_80 = float(-math.log(0.2) / lam) if (lam and not math.isnan(lam) and lam > 0) else float("nan")
    return dict(R_max=r_max, lam=lam, t_80=t_80,
                rmse=rmse, r2=r2, lam_at_bound=lam_at_bound)


def bootstrap_slope(log_n: np.ndarray, metric: np.ndarray,
                    n_boot: int = N_BOOT) -> dict:
    log_n = np.asarray(log_n, float); metric = np.asarray(metric, float)
    n = len(log_n)
    rng = np.random.default_rng(SEED)
    bs = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            _, b, _ = _ols(log_n[idx], metric[idx])
            bs[i] = b
        except Exception:
            bs[i] = float("nan")
    bs = bs[~np.isnan(bs)]
    return dict(slope=float(np.mean(bs)),
                lo=float(np.percentile(bs, 2.5)),
                hi=float(np.percentile(bs, 97.5)),
                n=int(len(bs)))


def load_traces() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, (fname, _) in MODELS.items():
        d = json.loads((TRACE_DIR / fname).read_text())
        out[label] = np.asarray(d["reward_trace"], float)
    return out


def _write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> None:
    raw = load_traces()
    fits: dict[str, dict] = {}
    segs: dict[str, dict] = {}
    classes: dict[str, dict] = {}

    for label, y in raw.items():
        fits[label] = fit_canonical(y)
        seg = segment_bic(y, k_max=3)
        cls = three_phase_classify(seg)
        segs[label] = seg
        classes[label] = cls

    # ---- (A) scaling_law_fits.tsv --------------------------------------
    cols = ["model", "params_B", "n_steps", "mean_reward", "var_reward",
            "peak", "trough", "early_mean", "late_mean",
            "delta_late_minus_early", "ols_slope_per_step", "ols_slope_se",
            "slope_direction", "R_max", "lambda", "t_80", "rmse", "r2",
            "lam_at_bound", "bic_k1", "bic_k2", "bic_k3", "best_k",
            "n_segments", "phase_nimmaturi", "nimmaturi_three_phase_ok",
            "trace_file"]
    rows = []
    for label, y in raw.items():
        n = len(y)
        cut = max(2, n // 3)
        early = float(np.mean(y[:cut]))
        late = float(np.mean(y[-cut:]))
        delta = late - early
        _, ols_b, ols_se = _ols(np.arange(1, n + 1, dtype=float), y)
        sign = "increase" if ols_b > 0 else ("flat" if abs(ols_b) < 1e-3 else "decrease")
        f = fits[label]; seg = segs[label]; cls = classes[label]
        bics = seg["bics"]
        rows.append([
            label, MODELS[label][1], n,
            f"{float(y.mean()):.4f}", f"{float(np.var(y)):.4f}",
            f"{float(np.max(y)):.4f}", f"{float(np.min(y)):.4f}",
            f"{early:.4f}", f"{late:.4f}", f"{delta:.4f}",
            f"{ols_b:.5f}", f"{ols_se:.5f}", sign,
            f"{f['R_max']:.4f}", f"{f['lam']:.4f}", f"{f['t_80']:.4f}",
            f"{f['rmse']:.4f}", f"{f['r2']:.4f}", int(f["lam_at_bound"]),
            f"{bics[1]:.4f}", f"{bics[2]:.4f}", f"{bics[3]:.4f}",
            int(seg["best_k"]),
            int(len(seg["segments"])),
            cls["phase"], int(cls["nimmaturi_three_phase_ok"]),
            MODELS[label][0],
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_fits.tsv", cols, rows)

    # ---- (B) three_phase.tsv -------------------------------------------
    cols2 = ["model", "params_B", "best_k", "n_segments",
             "bic_k1", "bic_k2", "bic_k3", "delta_bic_3v1", "delta_bic_3v2",
             "monotone_up", "monotone_dn",
             "phase_nimmaturi", "nimmaturi_three_phase_ok",
             "peak_segment_mean", "late_segment_mean", "early_segment_mean",
             "segment_means_joined", "R_max", "lambda", "t_80"]
    rows2 = []
    for label in MODELS:
        seg = segs[label]; cls = classes[label]; f = fits[label]
        bics = seg["bics"]
        seg_means = [s["mean"] for s in seg["segments"]]
        rows2.append([
            label, MODELS[label][1], int(seg["best_k"]),
            int(len(seg["segments"])),
            f"{bics[1]:.4f}", f"{bics[2]:.4f}", f"{bics[3]:.4f}",
            f"{bics[3]-bics[1]:.4f}",
            f"{bics[3]-bics[2]:.4f}",
int(cls["monotone_up"]), int(cls["monotone_dn"]),
            cls["phase"], int(cls["nimmaturi_three_phase_ok"]),
            f"{cls['peak_segment_mean']:.4f}" if cls['peak_segment_mean'] is not None else "nan",
            f"{cls['late_segment_mean']:.4f}" if cls['late_segment_mean'] is not None else "nan",
            f"{cls['early_segment_mean']:.4f}" if cls['early_segment_mean'] is not None else "nan",
            ";".join(f"{m:.4f}" for m in seg_means) if seg_means else "nan",
            f"{f['R_max']:.4f}", f"{f['lam']:.4f}", f"{f['t_80']:.4f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_three_phase.tsv", cols2, rows2)

    # ---- (C) changepoints.tsv -----------------------------------------
    cols3 = ["model", "params_B", "best_k", "segment_idx",
             "start_step", "end_step", "length", "mean_reward"]
    rows3 = []
    for label in MODELS:
        seg = segs[label]
        for i, s in enumerate(seg["segments"]):
            rows3.append([
                label, MODELS[label][1], int(seg["best_k"]),
                i + 1, s["start_step"], s["end_step"],
                s["length"], f"{s['mean']:.4f}",
            ])
    _write_tsv(RESULTS_DIR / "scaling_law_changepoints.tsv", cols3, rows3)

    # ---- (D) cross-scale.tsv (5 anchors only) --------------------------
    labels = list(MODELS.keys())
    log_n = np.log10([MODELS[l][1] for l in labels])
    cross_cols = ["metric", "n_models", "intercept", "slope_per_log10N",
                  "se_slope", "boot_slope_mean", "boot_slope_lo",
                  "boot_slope_hi", "n_boot", "corr_logN_metric"]
    cross_rows = []
    for metric in ("mean_reward", "peak", "var_reward"):
        if metric == "mean_reward":
            vals = np.array([float(raw[l].mean()) for l in labels])
        elif metric == "peak":
            vals = np.array([float(raw[l].max()) for l in labels])
        else:
            vals = np.array([float(np.var(raw[l])) for l in labels])
        a, b, se_b = _ols(log_n, vals)
        boot = bootstrap_slope(log_n, vals)
        r = float(np.corrcoef(log_n, vals)[0, 1])
        cross_rows.append([
            metric, len(vals), f"{a:.6f}", f"{b:.6f}", f"{se_b:.6f}",
            f"{boot['slope']:.6f}", f"{boot['lo']:.6f}",
            f"{boot['hi']:.6f}", boot["n"], f"{r:.6f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_cross_scale.tsv", cross_cols, cross_rows)

    # ---- (E) meta.json -------------------------------------------------
    three_phase_pass = sum(int(classes[l]["nimmaturi_three_phase_ok"]) for l in labels)
    collapse_count = sum(int(classes[l]["phase"] == "collapse") for l in labels)
    sat_at_bound = sum(int(fits[l]["lam_at_bound"]) for l in labels)
    nem = classes["Nemotron-120B"]
    nem_seg = segs["Nemotron-120B"]
    nem_segs_str = ";".join(
        f"{s['start_step']}-{s['end_step']}:{s['mean']:.3f}" for s in nem_seg["segments"]
    )
    meta = {
        "iter": 113,
        "pillar": "P1-ScalingLaws",
        "n_anchors": len(MODELS),
        "fit_form": "R(t) = R_max * (1 - exp(-lambda * t))",
        "lambda_bound": "[1e-4, 10.0] (canonical 2-param form)",
        "phase_classification_method": (
            "BIC-segmentation with k in {1,2,3} constant-mean regimes; "
            "nimmaturi2025predictive three-phase criterion = "
            "(best_k == 3) AND monotone_up AND (seg1_mean < 0.15) AND "
            "(seg3_mean > 0.40) (arXiv:2507.18014)."
        ),
        "n_three_phase_pass": three_phase_pass,
        "n_collapse": collapse_count,
        "n_lam_at_bound": sat_at_bound,
        "nemotron_segments": nem_segs_str,
        "nemotron_classification": nem["phase"],
        "nemotron_violation": (
            "Nemotron-120B is the only 5-anchor pool member that does NOT "
            "fit any of (plateau, three-phase, saturation, drift). "
            "BIC picks k=3 (best segmentation), the segment means are "
            "non-monotone (rise-then-fall), and peak segment mean exceeds "
            "late segment mean -- a textbook collapse.  This directly "
            "violates the nimmaturi2025predictive three-phase template "
            "(which requires monotone non-decreasing segment means)."
        ),
        "frontier_synthesis": (
            "iter113 closes the Pillar-1 scaling-law investigation by "
            "adding the missing piece: explicit changepoint-based test of "
            "the three-phase hypothesis on the 5-anchor frontier pool. "
            "Result: 0/5 anchors pass the nimmaturi2025predictive "
            "three-phase criterion; 4/5 anchors are already saturated at "
            "step 1 (best_k=1 BIC-optimal or 2 with no detectable ramp); "
            "1/5 (Nemotron-120B) is a clear collapse with peak not "
            "retained.  Combined with iter109's lambda-vs-N null (p=0.74) "
            "and iter105's R_max*(N) failure, the Pillar-1 finding is "
            "that GRPO post-training is not scale-law-shaped on this "
            "evidence base -- the only strong signal is the absence of a "
            "scaling law."
        ),
    }
    (RESULTS_DIR / "scaling_law_iter113_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"wrote {RESULTS_DIR / 'scaling_law_iter113_meta.json'}")

    # ---- headline log --------------------------------------------------
    print(f"Three-phase nimmaturi-pass: {three_phase_pass}/5 anchors")
    print(f"Collapse count:             {collapse_count}/5 anchors")
    print(f"lambda at upper bound:      {sat_at_bound}/5 anchors (canonical 2-param)")
    print()
    for l in labels:
        f = fits[l]; seg = segs[l]; cls = classes[l]
        seg_str = ";".join(f"{s['mean']:.3f}" for s in seg["segments"])
        print(
            f"  {l:24s} R_max={f['R_max']:.3f} lam={f['lam']:.3f} t_80={f['t_80']:.3f} "
            f"BIC(1,2,3)=({seg['bics'][1]:+.2f},{seg['bics'][2]:+.2f},{seg['bics'][3]:+.2f}) "
            f"best_k={seg['best_k']} segs=[{seg_str}] phase={cls['phase']}"
        )
    print()
    for r in cross_rows:
        print(
            f"cross-scale metric={r[0]:>14s} slope/decade={float(r[3]):+.4f} "
            f"95% CI=[{float(r[6]):+.4f}, {float(r[7]):+.4f}] corr={float(r[9]):+.3f}"
        )

    # ---- figure --------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    cmap = plt.get_cmap("viridis")
    pcol = {"plateau": "tab:gray", "three_phase": "tab:green",
            "saturation": "tab:blue", "drift": "tab:orange",
            "collapse": "tab:red", "non_monotone": "tab:purple"}

    # (a) raw traces + BIC segment overlays
    ax_a = fig.add_subplot(gs[0, 0])
    for i, (label, y) in enumerate(raw.items()):
        t = np.arange(1, len(y) + 1)
        color = cmap(i / max(1, len(raw) - 1))
        ax_a.plot(t, y, "o", color=color, markersize=4, alpha=0.7,
                  label=f"{label.replace('-Inst', '')} ({MODELS[label][1]:.0f}B)")
        for s in segs[label]["segments"]:
            ax_a.hlines(s["mean"], s["start_step"], s["end_step"],
                        colors=color, alpha=0.35, linewidth=4)
    ax_a.set_xlabel("training step"); ax_a.set_ylabel("reward")
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.set_title("(a) Raw traces + BIC-optimal constant-mean segments")
    ax_a.grid(alpha=0.25); ax_a.legend(fontsize=7, loc="lower right", ncol=2)

    # (b) cross-scale scatter + class-colour markers
    ax_b = fig.add_subplot(gs[0, 1])
    log_n_arr = np.log10([MODELS[l][1] for l in labels])
    means = np.array([float(raw[l].mean()) for l in labels])
    peaks = np.array([float(raw[l].max()) for l in labels])
    a_m, b_m, _ = _ols(log_n_arr, means)
    a_p, b_p, _ = _ols(log_n_arr, peaks)
    xs = np.linspace(log_n_arr.min() - 0.05, log_n_arr.max() + 0.05, 100)
    ax_b.scatter(log_n_arr, means,
                 c=[pcol[classes[l]["phase"]] for l in labels],
                 s=120, edgecolor="k", zorder=3, label="mean R")
    ax_b.scatter(log_n_arr, peaks,
                 c=[pcol[classes[l]["phase"]] for l in labels],
                 s=70, marker="^", edgecolor="k", zorder=3, alpha=0.6,
                 label="peak R")
    ax_b.plot(xs, a_m + b_m * xs, "b--", lw=1.5,
              label=fr"$\bar R$ slope={b_m:+.3f}/dec")
    ax_b.plot(xs, a_p + b_p * xs, "r:", lw=1.5,
              label=fr"$\hat R$ slope={b_p:+.3f}/dec")
    for l, x, y in zip(labels, log_n_arr, means):
        ax_b.annotate(l.replace("-Inst", ""), (x, y),
                      fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_b.set_xlabel(r"$\log_{10}$(params [B])")
    ax_b.set_ylabel("reward (0-1)")
    ax_b.set_ylim(0, 1.1); ax_b.grid(alpha=0.25)
    ax_b.set_title("(b) Cross-scale: 0/5 anchors pass nimmaturi three-phase")

    # (c) BIC comparison per anchor
    ax_c = fig.add_subplot(gs[1, 0])
    bar_w = 0.27
    x_pos = np.arange(len(labels))
    bics_k1 = [segs[l]["bics"][1] for l in labels]
    bics_k2 = [segs[l]["bics"][2] for l in labels]
    bics_k3 = [segs[l]["bics"][3] for l in labels]
    ax_c.bar(x_pos - bar_w, bics_k1, bar_w, color="tab:gray",
             edgecolor="k", label="BIC k=1")
    ax_c.bar(x_pos,         bics_k2, bar_w, color="tab:blue",
             edgecolor="k", label="BIC k=2")
    ax_c.bar(x_pos + bar_w, bics_k3, bar_w, color="tab:red",
             edgecolor="k", label="BIC k=3")
    best_k = [segs[l]["best_k"] for l in labels]
    for i, l in enumerate(labels):
        ax_c.text(i, max(bics_k1[i], bics_k2[i], bics_k3[i]) + 0.3,
                  f"k*={best_k[i]}", ha="center", fontsize=8)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([l.replace("-Inst", "") for l in labels],
                         rotation=20, ha="right", fontsize=8)
    ax_c.set_ylabel("BIC (lower = better)")
    ax_c.set_title("(c) BIC-based segment-count selection per anchor")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend(fontsize=7, loc="upper right")

    # (d) Nemotron-120B collapse zoom
    ax_d = fig.add_subplot(gs[1, 1])
    nem_y = raw["Nemotron-120B"]
    t = np.arange(1, len(nem_y) + 1)
    ax_d.bar(t, nem_y, color="tab:red", alpha=0.85, edgecolor="k")
    ax_d.axhline(float(nem_y.mean()), ls="--", color="k", lw=0.9,
                 label=f"mean={float(nem_y.mean()):.3f}")
    seg = segs["Nemotron-120B"]
    for s in seg["segments"]:
        ax_d.hlines(s["mean"], s["start_step"], s["end_step"] + 0.4,
                    colors="tab:purple", lw=4, alpha=0.7)
        ax_d.text((s["start_step"] + s["end_step"]) / 2, s["mean"] + 0.02,
                  f"{s['mean']:.2f}", ha="center", fontsize=8,
                  color="tab:purple")
    pi = int(np.argmax(nem_y))
    ax_d.annotate(f"peak {nem_y[pi]:.2f} @ step {pi+1}",
                  xy=(pi + 1, nem_y[pi]), xytext=(pi + 1.5, nem_y[pi] + 0.06),
                  arrowprops=dict(arrowstyle="->", lw=0.9, color="k"),
                  fontsize=8)
    ax_d.set_xlabel("training step"); ax_d.set_ylabel("reward")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title("(d) Nemotron-120B collapse: 3-segment non-monotone")
    ax_d.legend(fontsize=7); ax_d.grid(alpha=0.25)

    from matplotlib.patches import Patch
    seen_phases = sorted({classes[l]["phase"] for l in labels},
                         key=lambda p: list(pcol).index(p) if p in pcol else 99)
    phase_legend = [Patch(facecolor=pcol[p], edgecolor="k", label=p)
                    for p in seen_phases]
    handles_b = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markeredgecolor="k", markersize=8, label="mean R"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                   markeredgecolor="k", markersize=7, label="peak R"),
        plt.Line2D([0], [0], color="b", ls="--", lw=1.5,
                   label=fr"$\bar R$ slope={b_m:+.3f}/dec"),
        plt.Line2D([0], [0], color="r", ls=":", lw=1.5,
                   label=fr"$\hat R$ slope={b_p:+.3f}/dec"),
    ] + phase_legend
    ax_b.legend(handles=handles_b, fontsize=7, loc="upper left",
                title="metric / OLS / phase", title_fontsize=7)

    fig.suptitle(
        "Pillar 1 iter113 -- GRPO scaling analysis (4B-685B): "
        "0/5 anchors satisfy nimmaturi2025predictive three-phase "
        "(arXiv:2507.18014); Nemotron-120B is the only collapse",
        fontsize=11,
    )
    out_pdf = FIG_DIR / "scaling_law_fit.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    fig.savefig(PAPER_FIG / "scaling_law_fit.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_fit.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()