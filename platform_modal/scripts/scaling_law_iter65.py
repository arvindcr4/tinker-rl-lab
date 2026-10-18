"""Exploratory continuous slow-growth / rapid-growth / plateau diagnostics.

Fits minimize original-space residuals with continuous hinge segments.
Signed shape checks are distinct from fit quality. Residual-block boundary
intervals are conditional on the fitted trend and are not coverage validation.
Candidate outputs go to review27_v3 or an explicitly supplied output directory.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
try:
    from ._review27_rank import rank_correlation
except ImportError:
    from _review27_rank import rank_correlation  # noqa: E402

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"
PAPER_FIG = CANDIDATE / "paper_figures"

MODELS: dict[str, dict] = {
    "Qwen3.5-4B":            {"file": "scale_gsm8k_qwen3.5-4b.json",     "params":   4.0, "arch": "dense"},
    "Qwen3-8B":              {"file": "scale_gsm8k_qwen3-8b.json",       "params":   8.0, "arch": "dense"},
    "Llama-3.1-8B-Instruct": {"file": "scale_gsm8k_llama-8b-inst.json",  "params":   8.0, "arch": "dense"},
    "Qwen3-32B":             {"file": "scale_gsm8k_qwen3-32b.json",      "params":  32.0, "arch": "dense"},
    "Qwen3.5-27B":           {"file": "scale_gsm8k_qwen3.5-27b.json",    "params":  27.0, "arch": "dense"},
    "gpt-oss-20B":           {"file": "arch_gsm8k_gpt-oss-20b.json",     "params":  20.0, "arch": "moe"},
    "Qwen3-30B-MoE":         {"file": "moe_gsm8k_qwen3-30b-moe.json",    "params":  30.0, "arch": "moe"},
    "Qwen3-30B-MoE-Inst":    {"file": "moe_gsm8k_qwen3-30b-inst.json",   "params":  30.0, "arch": "moe"},
    "DeepSeek-V3.1":         {"file": "frontier_gsm8k_deepseek-v3.1.json","params": 685.0, "arch": "moe"},
    "Nemotron-120B":         {"file": "frontier_gsm8k_nemotron-120b.json","params": 120.0, "arch": "dense"},
    "Qwen3-235B-MoE":        {"file": "frontier_gsm8k_qwen3-235b.json",  "params": 235.0, "arch": "moe"},
    "Kimi-K2-Thinking":      {"file": "arch_gsm8k_kimi-k2.json",         "params":1000.0, "arch": "moe"},
}

SEED = 20260702
N_BOOT = 2000

# Canonical three-phase template (target slopes per phase, in
# reward units per step).  The hypothesis is that the slow-start
# phase has slope m1, the rapid-improvement phase has slope m2 > m1
# (typically 5x-10x), and the plateau phase has slope m3 near 0
# (|m3| < eps).  See nimmaturi2025predictive (arXiv:2507.18014).
SLOW_SLOPE_MAX = 0.020      # |slope| < 0.020/step => slow
FAST_SLOPE_MIN = 0.040      # |slope| >= 0.040/step => fast
PLATEAU_SLOPE_MAX = 0.015   # |slope| < 0.015/step => plateau
PLATEAU_VAR_FRAC = 0.50     # plateau variance < 0.5 * mean variance


def _load_trace(fname: str) -> list[float]:
    try:
        from ._review27_rank import load_reward_trace
    except ImportError:
        from _review27_rank import load_reward_trace
    return load_reward_trace(TRACE_DIR / fname)


def _fit_3piece(t: np.ndarray, y: np.ndarray) -> tuple:
    """Exhaustive continuous three-segment OLS with two points per segment."""
    t,y=np.asarray(t,float),np.asarray(y,float)
    missing=(None,None,math.nan,*( (math.nan,)*3 for _ in range(5)))
    if t.ndim!=1 or t.shape!=y.shape or len(t)<6 or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0):
        return missing
    best=None
    for i in range(2,len(t)-3):
        for j in range(i+2,len(t)-1):
            k1,k2=t[i],t[j]
            design=np.column_stack([np.ones(len(t)),t,np.maximum(0,t-k1),np.maximum(0,t-k2)])
            beta,*_=np.linalg.lstsq(design,y,rcond=None)
            predicted=design@beta
            sse=float(np.sum((y-predicted)**2))
            if best is None or sse<best[0]:best=(sse,i,j,beta,predicted)
    if best is None:return missing
    sse,i,j,beta,predicted=best
    slopes=(float(beta[1]),float(beta[1]+beta[2]),float(beta[1]+beta[2]+beta[3]))
    intercepts=(float(beta[0]),float(beta[0]-beta[2]*t[i]),float(beta[0]-beta[2]*t[i]-beta[3]*t[j]))
    slices=(slice(None,i),slice(i,j),slice(j,None))
    residuals=tuple(float(np.sqrt(np.mean((y[k]-predicted[k])**2))) for k in slices)
    means=tuple(float(y[k].mean()) for k in slices)
    stds=tuple(float(y[k].std()) for k in slices)
    return i,j,sse,slopes,intercepts,residuals,means,stds




def _classify(slopes: tuple, residuals: tuple, n: int, i: int, j: int,
              seg_means: tuple, seg_stds: tuple) -> str:
    """Signed slow/rise/plateau; drift is late decline without a low floor."""
    if i is None or j is None or not np.isfinite(slopes).all():
        return "insufficient_data"
    m1, m2, m3 = slopes
    mu1, mu2, mu3 = seg_means
    sd_means = float(np.std([mu1, mu2, mu3]))
    if sd_means < 1e-6:
        return "constant"
    if 0 <= m1 < SLOW_SLOPE_MAX and m2 >= FAST_SLOPE_MIN and abs(m3) < PLATEAU_SLOPE_MAX and mu3 > mu1 and mu3 >= mu2 and min(i,j-i,n-j) >= 2:
        return "three-phase"
    if mu3 < 0.2 and max(mu1, mu2) - mu3 > 0.2:
        return "collapse"
    if m3 < -PLATEAU_SLOPE_MAX:
        return "drift"
    # A hill is a rise-then-decline, not slow-fast-plateau confirmation.
    if mu2 > mu1 and mu2 > mu3:
        return "rise-then-decline"
    # Valley: P2 is the trough
    if mu1 > mu2 and mu2 < mu3:
        # Distinguish collapse (low floor) from valley-recovery (high floor)
        if mu3 < 0.2 and mu1 - mu3 > 0.2:
            return "collapse"
        return "valley"
    # All-rising
    if mu1 < mu2 < mu3:
        return "monotonic-rising"
    # All-declining
    if mu1 > mu2 > mu3:
        return "monotonic-decline"
    # Mixed shape: e.g. mu1 ~ mu2 < mu3
    if abs(mu1 - mu2) < 0.05 and mu3 > mu1:
        return "monotonic-rising"
    if abs(mu2 - mu3) < 0.05 and mu1 < mu2:
        return "monotonic-saturation"
    if abs(mu1 - mu3) < 0.05 and mu1 < mu2:
        return "rise-then-plateau"
    return "anomalous"


def _phase_conformity(slopes: tuple, seg_means: tuple, seg_stds: tuple) -> float:
    """Half a point for each of six signed slow/rise/plateau conditions."""
    if not all(np.isfinite(v).all() for v in (slopes, seg_means, seg_stds)):
        return math.nan
    m1, m2, m3 = slopes
    mu1, mu2, mu3 = seg_means
    conditions = (0 <= m1 < SLOW_SLOPE_MAX,
                  m2 >= FAST_SLOPE_MIN,
                  abs(m3) < PLATEAU_SLOPE_MAX,
                  mu2 > mu1,
                  mu3 > mu1 and mu3 >= mu2,
                  seg_stds[2] <= seg_stds[1])
    return 0.5 * sum(conditions)


def _bootstrap_boundaries(y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Residual circular-block bootstrap around the fitted continuous trend.

    Exploratory conditional boundary intervals, not a preregistered coverage claim.
    """
    n = len(y)
    if n < 6 or not np.isfinite(y).all():
        return {"cp1_lo": -1, "cp1_med": -1, "cp1_hi": -1,
                "cp2_lo": -1, "cp2_med": -1, "cp2_hi": -1, "n_boot_success": 0}
    tt = np.arange(1, n + 1, dtype=float)
    i, j, _, slopes, intercepts, *_ = _fit_3piece(tt, y)
    if i is None:
        return {"cp1_lo": -1, "cp1_med": -1, "cp1_hi": -1,
                "cp2_lo": -1, "cp2_med": -1, "cp2_hi": -1, "n_boot_success": 0}
    fitted = np.concatenate([slopes[k]*tt[a:b]+intercepts[k]
                             for k,(a,b) in enumerate(((0,i),(i,j),(j,n)))])
    residual = np.asarray(y) - fitted
    residual -= residual.mean()
    rng = np.random.default_rng(SEED)
    block = max(2, n // 6)
    cps1, cps2 = [], []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n))
            take = min(block, n - len(idx))
            idx.extend(((start + k) % n) for k in range(take))
        idx = np.array(idx[:n], dtype=int)
        yy = fitted + residual[idx]
        tt = np.arange(1, n + 1, dtype=float)
        i, j, _, _, _, _, _, _ = _fit_3piece(tt, yy)
        if i is not None and j is not None:
            cps1.append(int(i))
            cps2.append(int(j))
    if not cps1:
        return {"cp1_lo": -1, "cp1_med": -1, "cp1_hi": -1,
                "cp2_lo": -1, "cp2_med": -1, "cp2_hi": -1, "n_boot_success": 0}
    return {
        "cp1_lo": int(np.percentile(cps1, 2.5)),
        "cp1_med": int(np.percentile(cps1, 50.0)),
        "cp1_hi": int(np.percentile(cps1, 97.5)),
        "cp2_lo": int(np.percentile(cps2, 2.5)),
        "cp2_med": int(np.percentile(cps2, 50.0)),
        "cp2_hi": int(np.percentile(cps2, 97.5)),
        "n_boot_success": len(cps1),
    }


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

    # ---- main per-trace analysis
    phase_rows = []
    conformity_rows = []
    boundary_rows = []
    for m, cfg in MODELS.items():
        rt = _load_trace(cfg["file"])
        if not rt:
            continue
        y = np.array(rt, dtype=float)
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        i, j, sse, slopes, intercepts, residuals, seg_means, seg_stds = _fit_3piece(t, y)
        cls = _classify(slopes, residuals, n, i, j, seg_means, seg_stds)
        pci = _phase_conformity(slopes, seg_means, seg_stds)
        # n_phase_well: how many of the three template constraints
        # (slow, fast, plateau) the trace satisfies.
        m1, m2, m3 = slopes
        n_slow = int(0 <= m1 < SLOW_SLOPE_MAX)
        n_fast = int(m2 >= FAST_SLOPE_MIN)
        n_plateau = int(abs(m3) < PLATEAU_SLOPE_MAX)
        # template score: number of 6 template constraints satisfied
        template_score = int(pci / 0.5) if pci > 0 else 0
        phase_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n_steps": n, "mean_reward": float(y.mean()),
            "var_reward": float(y.var()), "peak": float(y.max()),
            "trough": float(y.min()), "cp1": i if i is not None else -1,
            "cp2": j if j is not None else -1, "cp1_frac": (i / n) if i else -1,
            "cp2_frac": (j / n) if j else -1,
            "slope_p1": slopes[0], "slope_p2": slopes[1], "slope_p3": slopes[2],
            "rmse_p1": residuals[0], "rmse_p2": residuals[1], "rmse_p3": residuals[2],
            "inter_p1": intercepts[0], "inter_p2": intercepts[1], "inter_p3": intercepts[2],
            "sse_3piece": sse, "phase_class": cls, "pci": pci,
            "seg_mean_p1": seg_means[0], "seg_mean_p2": seg_means[1], "seg_mean_p3": seg_means[2],
            "seg_std_p1": seg_stds[0], "seg_std_p2": seg_stds[1], "seg_std_p3": seg_stds[2],
            "n_slow": n_slow, "n_fast": n_fast, "n_plateau": n_plateau,
            "n_shape_conditions_ok": template_score,
            "n_slope_conditions_ok": n_slow + n_fast + n_plateau,
        })
        conformity_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "pci": pci, "phase_class": cls,
            "n_slow_ok": n_slow, "n_fast_ok": n_fast, "n_plateau_ok": n_plateau,
            "n_slope_conditions_ok": n_slow + n_fast + n_plateau,
            "n_shape_conditions_ok": template_score,
        })
        # bootstrap phase boundaries
        boot = _bootstrap_boundaries(y, n_boot=N_BOOT)
        boundary_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n_steps": n, "cp1_point": i if i is not None else -1,
            "cp2_point": j if j is not None else -1,
            "cp1_frac": (i / n) if i else -1,
            "cp2_frac": (j / n) if j else -1,
            **boot,
        })

    if not phase_rows:
        raise ValueError("no observed reward traces available")
    # ---- write per-trace outputs
    phase_path = RESULTS_DIR / "scaling_law_iter65_phase_pieces_v2.tsv"
    with phase_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(phase_rows)
    conformity_path = RESULTS_DIR / "scaling_law_iter65_conformity_v2.tsv"
    with conformity_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conformity_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(conformity_rows)
    boundary_path = RESULTS_DIR / "scaling_law_iter65_boundaries_v2.tsv"
    with boundary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(boundary_rows)

    attempted_count = len(phase_rows)
    phase_rows = [r for r in phase_rows if math.isfinite(r["pci"])]
    if not phase_rows:
        raise ValueError("no eligible six-step phase fits available")
    eligible_models = {r["model"] for r in phase_rows}
    boundary_rows = [r for r in boundary_rows if r["model"] in eligible_models]
    # ---- architecture-level phase composition
    arch_phase_rows = []
    for arch in ("dense", "moe"):
        sub = [r for r in phase_rows if r["arch"] == arch]
        if not sub:
            continue
        n = len(sub)
        cls_counts: dict[str, int] = {}
        for r in sub:
            cls_counts[r["phase_class"]] = cls_counts.get(r["phase_class"], 0) + 1
        pci_mean = float(np.mean([r["pci"] for r in sub]))
        pci_med = float(np.median([r["pci"] for r in sub]))
        three_phase = cls_counts.get("three-phase", 0)
        collapse = cls_counts.get("collapse", 0)
        drift = cls_counts.get("drift", 0)
        n_template_full = sum(1 for r in sub if r["n_slope_conditions_ok"] == 3)
        arch_phase_rows.append({
            "arch": arch, "n_anchors": n,
            "n_three_phase": three_phase, "frac_three_phase": three_phase / n,
            "n_collapse": collapse, "frac_collapse": collapse / n,
            "n_drift": drift, "frac_drift": drift / n,
            "n_template_full": n_template_full, "frac_template_full": n_template_full / n,
            "pci_mean": pci_mean, "pci_median": pci_med,
            "class_counts": ";".join(f"{k}:{v}" for k, v in sorted(cls_counts.items())),
        })
    arch_path = RESULTS_DIR / "scaling_law_iter65_arch_phase_v2.tsv"
    with arch_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arch_phase_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(arch_phase_rows)

    # ---- exploratory predictions
    n_total = len(phase_rows)
    n_three_phase = sum(1 for r in phase_rows if r["phase_class"] == "three-phase")
    n_collapse = sum(1 for r in phase_rows if r["phase_class"] == "collapse")
    n_drift = sum(1 for r in phase_rows if r["phase_class"] == "drift")
    n_template_full = sum(1 for r in phase_rows if r["n_slope_conditions_ok"] == 3)
    n_template_partial = sum(1 for r in phase_rows if r["n_slope_conditions_ok"] >= 2)
    pci_values = [r["pci"] for r in phase_rows]
    pci_mean = float(np.mean(pci_values))
    # Spearman correlation between params and pci
    params = np.array([r["params_B"] for r in phase_rows])
    pci = np.array(pci_values)
    finite = np.isfinite(params) & np.isfinite(pci)
    sp = rank_correlation(params[finite], pci[finite])
    # PCI of Nemotron-120B vs others
    nem = next((r for r in phase_rows if r["model"] == "Nemotron-120B"), None)
    others = [r["pci"] for r in phase_rows if r["model"] != "Nemotron-120B"]
    others_med = float(np.median(others)) if others else math.nan
    nem_pci = float(nem["pci"]) if nem else math.nan
    # Per-arch three-phase rate
    dense_3p = sum(1 for r in phase_rows if r["arch"] == "dense" and r["phase_class"] == "three-phase")
    dense_n = sum(1 for r in phase_rows if r["arch"] == "dense")
    moe_3p = sum(1 for r in phase_rows if r["arch"] == "moe" and r["phase_class"] == "three-phase")
    moe_n = sum(1 for r in phase_rows if r["arch"] == "moe")
    # Exploratory predictions
    pred_rows = [
        {
            "prediction_id": "P1_three_phase_majority",
            "claim": "At least half of eligible anchors match the canonical three-phase pattern (slow, fast, plateau).",
            "observed": f"{n_three_phase}/{n_total} = {n_three_phase/n_total:.3f}",
            "expected": ">= 0.50",
            "pass_fail": "PASS" if n_three_phase/n_total >= 0.5 else "FAIL",
        },
        {
            "prediction_id": "P2_nemotron_classified_collapse",
            "claim": "Nemotron-120B is the unique 'collapse' class anchor (peak-then-decay).",
            "observed": f"nemotron_class={nem['phase_class'] if nem else 'NA'}, n_collapse={n_collapse}",
            "expected": "nemotron_class == collapse and n_collapse == 1",
            "pass_fail": "PASS" if (nem and nem["phase_class"] == "collapse" and n_collapse == 1) else "FAIL",
        },
        {
            "prediction_id": "P3_pci_below_3_for_most",
            "claim": "Mean PCI across the pool is below 1.5 (most traces violate at least one phase template).",
            "observed": f"pci_mean={pci_mean:.3f}",
            "expected": "< 1.5",
            "pass_fail": "PASS" if pci_mean < 1.5 else "FAIL",
        },
        {
            "prediction_id": "P4_nemotron_lowest_pci",
            "claim": "Nemotron-120B has the lowest PCI in the eligible pool.",
            "observed": f"nemotron_pci={nem_pci:.3f}, others_median={others_med:.3f}, all_pci={[round(x,2) for x in pci_values]}",
            "expected": "nemotron_pci <= min(others)",
            "pass_fail": "PASS" if (nem and others and nem_pci <= min(others)) else "FAIL",
        },
        {
            "prediction_id": "P5_template_full_at_most_3",
            "claim": "At most three eligible anchors satisfy ALL three phase-template constraints (slow, fast, plateau).",
            "observed": f"{n_template_full}/{n_total}",
            "expected": "<= 3",
            "pass_fail": "PASS" if n_template_full <= 3 else "FAIL",
        },
        {
            "prediction_id": "P6_dense_higher_three_phase_rate",
            "claim": "Dense models have a higher three-phase rate than MoE.",
            "observed": f"dense_3p={dense_3p}/{dense_n}, moe_3p={moe_3p}/{moe_n}",
            "expected": "dense_3p_rate > moe_3p_rate",
            "pass_fail": "PASS" if (dense_n and moe_n and dense_3p / dense_n > moe_3p / moe_n) else "FAIL",
        },
        {
            "prediction_id": "P7_pci_logparams_correlation",
            "claim": "Spearman(params_B, PCI) is non-negative (larger models => more phase-conformity).",
            "observed": f"spearman_log={sp:.3f}",
            "expected": ">= 0",
            "pass_fail": "UNAVAILABLE" if not math.isfinite(sp) else "PASS" if sp >= 0 else "FAIL",
        },
        {
            "prediction_id": "P8_drift_count_3plus",
            "claim": "At least three eligible anchors are 'drift' (late slope negative, no early collapse).",
            "observed": f"n_drift={n_drift}",
            "expected": ">= 3",
            "pass_fail": "PASS" if n_drift >= 3 else "FAIL",
        },
    ]
    for result in pred_rows:
        if ((result["prediction_id"].startswith(("P2_", "P4_")) and nem is None)
                or (result["prediction_id"].startswith("P6_") and not (dense_n and moe_n))):
            result["pass_fail"] = "UNAVAILABLE"
    pred_path = RESULTS_DIR / "scaling_law_iter65_predictions_v2.tsv"
    with pred_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(pred_rows)

    # ---- 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    color_map = {
        "three-phase": "#2ca02c", "monotonic-saturation": "#1f77b4",
        "monotonic": "#aec7e8", "drift": "#ff7f0e", "collapse": "#d62728",
        "constant": "#7f7f7f", "anomalous": "#bcbd22",
    }
    # Panel A: per-anchor slopes (P1, P2, P3) with PCI score
    ax = axes[0, 0]
    labels = [r["model"] for r in phase_rows]
    p1 = [r["slope_p1"] for r in phase_rows]
    p2 = [r["slope_p2"] for r in phase_rows]
    p3 = [r["slope_p3"] for r in phase_rows]
    x = np.arange(len(labels))
    wbar = 0.27
    ax.bar(x - wbar, p1, wbar, color="#9467bd", label="phase 1 slope")
    ax.bar(x,         p2, wbar, color="#2ca02c", label="phase 2 slope")
    ax.bar(x + wbar, p3, wbar, color="#d62728", label="phase 3 slope")
    ax.axhline(SLOW_SLOPE_MAX, color="grey", linestyle=":", label=f"slow bound {SLOW_SLOPE_MAX}")
    ax.axhline(FAST_SLOPE_MIN, color="grey", linestyle="--", label=f"fast bound {FAST_SLOPE_MIN}")
    ax.axhline(-PLATEAU_SLOPE_MAX, color="black", linestyle=":", linewidth=0.6)
    ax.axhline(PLATEAU_SLOPE_MAX, color="black", linestyle=":", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("OLS slope (reward / step)")
    ax.set_title("(A) per-phase OLS slopes across eligible anchors")
    ax.legend(fontsize=7, loc="upper left")
    # Panel B: phase-conformity index (PCI) per anchor
    ax = axes[0, 1]
    pcis = [r["pci"] for r in phase_rows]
    colors = [color_map.get(r["phase_class"], "#000000") for r in phase_rows]
    ax.bar(x, pcis, color=colors)
    ax.axhline(1.5, color="black", linestyle=":", label="PCI=1.5 (descriptive threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("phase-conformity index (PCI)")
    ax.set_title("(B) per-anchor PCI (colour = phase class)")
    # add legend for colours
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=k) for k, c in color_map.items()]
    ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=2)
    # Panel C: phase boundary 95% CI (cp1, cp2)
    ax = axes[1, 0]
    for k, r in enumerate(boundary_rows):
        if r["n_steps"] < 6:
            # short traces: skip the bootstrap CIs
            c1m = r["cp1_frac"]; c2m = r["cp2_frac"]
            if c1m >= 0 and c2m >= 0:
                ax.scatter(c1m, k, marker="o", color="#1f77b4", s=30)
                ax.scatter(c2m, k, marker="s", color="#d62728", s=30)
            continue
        c1m = r["cp1_frac"]; c2m = r["cp2_frac"]
        c1lo = max(0, r["cp1_lo"] / r["n_steps"]); c1hi = min(1, r["cp1_hi"] / r["n_steps"])
        c2lo = max(0, r["cp2_lo"] / r["n_steps"]); c2hi = min(1, r["cp2_hi"] / r["n_steps"])
        if c1m < 0 or c2m < 0:
            continue
        ax.errorbar(c1m, k, xerr=[[max(0, c1m - c1lo)], [max(0, c1hi - c1m)]], fmt="o", color="#1f77b4", capsize=3)
        ax.errorbar(c2m, k, xerr=[[max(0, c2m - c2lo)], [max(0, c2hi - c2m)]], fmt="s", color="#d62728", capsize=3)
    ax.axvline(0.10, color="black", linestyle=":", label="10% in (P1 boundary)")
    ax.axvline(0.50, color="black", linestyle="--", label="50% in (mid-trace)")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("phase boundary as fraction of trace length")
    ax.set_title("(C) bootstrap CI of phase boundaries\n(blue = cp1, red = cp2)")
    ax.legend(fontsize=7, loc="lower right")
    # Panel D: per-arch PCI
    ax = axes[1, 1]
    dense_pci = [r["pci"] for r in phase_rows if r["arch"] == "dense"]
    moe_pci   = [r["pci"] for r in phase_rows if r["arch"] == "moe"]
    bp = ax.boxplot([dense_pci, moe_pci], tick_labels=["dense", "moe"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.scatter([1] * len(dense_pci), dense_pci, color="#1f77b4", zorder=3, s=30)
    ax.scatter([2] * len(moe_pci),   moe_pci,   color="#ff7f0e", zorder=3, s=30)
    ax.set_ylabel("phase-conformity index (PCI)")
    ax.set_title(f"(D) per-architecture PCI\n(dense n={len(dense_pci)}, moe n={len(moe_pci)})")
    plt.tight_layout()
    out_pdf = FIG_DIR / "scaling_law_iter65_v2.pdf"
    out_png = FIG_DIR / "scaling_law_iter65_v2.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close(fig)
    # mirror to paper/figures/
    (PAPER_FIG / "scaling_law_iter65_v2.pdf").write_bytes(out_pdf.read_bytes())
    (PAPER_FIG / "scaling_law_iter65_v2.png").write_bytes(out_png.read_bytes())

    # ---- write LaTeX section
    cls_counts = {}
    for r in phase_rows:
        cls_counts[r["phase_class"]] = cls_counts.get(r["phase_class"], 0) + 1
    class_str = "; ".join(f"{k}: {v}" for k, v in sorted(cls_counts.items()))
    n_three_phase = cls_counts.get("three-phase", 0)
    n_collapse = cls_counts.get("collapse", 0)
    n_drift = cls_counts.get("drift", 0)
    n_monosat = cls_counts.get("monotonic-saturation", 0)
    n_mono = cls_counts.get("monotonic", 0)
    n_const = cls_counts.get("constant", 0)
    n_anom = cls_counts.get("anomalous", 0)
    nem_row = next((r for r in phase_rows if r["model"] == "Nemotron-120B"), None)
    pred_str = "; ".join(f"{p['prediction_id']}={p['pass_fail']}" for p in pred_rows)
    # top 5 by |residual| in piece 3 (plateau drift)
    sorted_by_plateau = sorted(phase_rows, key=lambda r: -abs(r["slope_p3"]))[:6]
    # CSV-friendly strings for the LaTeX
    top_table = "\n".join(
        f"      {r['model']} & {r['params_B']:.1f} & {r['arch']} & "
        f"{r['slope_p1']:+.4f} & {r['slope_p2']:+.4f} & {r['slope_p3']:+.4f} & "
        f"{r['cp1_frac']:.2f} & {r['cp2_frac']:.2f} & {r['phase_class']} \\\\"
        for r in sorted_by_plateau
    )
    # arch summary table
    arch_table = "\n".join(
        f"      {r['arch']} & {r['n_anchors']} & {r['n_three_phase']}/{r['n_anchors']} & "
        f"{r['n_collapse']}/{r['n_anchors']} & {r['n_drift']}/{r['n_anchors']} & "
        f"{r['n_template_full']}/{r['n_anchors']} & {r['pci_mean']:.2f} \\\\"
        for r in arch_phase_rows
    )
    sec = r"\subsection{Exploratory continuous three-phase fits}" + "\n"
    sec += "The least-squares hinge model is continuous at both changepoints. PCI counts six signed growth and plateau conditions, worth half a point each. Slope-condition counts are separate. Residual-block boundary intervals are conditional exploratory summaries; short nonstationary traces do not establish generalization.\n"
    sec += f"Attempted traces: {attempted_count}; eligible fits: {n_total}. Observed class counts: {class_str}.\n"
    sec += r"\begin{tabular}{lrlrrrrrl}" + "\n" + top_table + "\n" + r"\end{tabular}" + "\n"
    for result in pred_rows:
        label = result["prediction_id"].replace("_", r"\_")
        sec += f"Descriptive check {label}: {result['pass_fail']}.\n"
    (PAPER_SEC / "scaling_law_iter65_v2.tex").write_text(sec)

    # ---- console summary
    print(f"phase_class distribution: {class_str}")
    print(f"three-phase: {n_three_phase}/{n_total}; collapse: {n_collapse}/{n_total}; drift: {n_drift}/{n_total}")
    print(f"n_template_full: {n_template_full}/{n_total}; pci_mean: {pci_mean:.3f}")
    print(f"spearman(log params, PCI) = {sp:.3f}")
    print("predictions:")
    for p in pred_rows:
        print(f"  {p['prediction_id']}: {p['pass_fail']} -- {p['observed']}")

    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(__file__, CANDIDATE, [TRACE_DIR / cfg["file"] for cfg in MODELS.values()])


if __name__ == "__main__":
    main()
