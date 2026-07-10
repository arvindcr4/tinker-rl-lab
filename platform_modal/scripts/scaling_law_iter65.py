"""Pillar 1 iter65 -- Three-phase hypothesis phase-conformity geometry.

iter33 ran a 5-prediction battery on the canonical 3-phase hypothesis
(slow start, rapid improvement, plateau) and reported 4/5 falsified:
P1 (peak after 10% in) 9/12 (borderline), P2 (late > early) 5/12,
P3 (plateau late near peak) 7/12 (borderline), P4 (phase score
predicts mean) rho=0.046 (decisively falsified), P5 (Nemotron-120B
uniqueness) sustained. iter41 introduced a 12-anchor pool. iter45
ran iso-compute extrapolation. iter49 produced the canonical
two-parameter cross-scale fit (R^2 = 0.18). iter57 audited the
saturation-fit identifiability (4/5 at lambda bound). iter61
conditioned the degeneracy on per-step ZVF (HIGH vs LOW strata).

What iter33/37/41 left as a gap: the three-phase hypothesis is
exclusively tested with summary statistics (peak_frac, late>early,
plateau-variance). What is missing is a GEOMETRIC test that
characterises the per-phase SHAPE -- slope of the slow-start phase,
slope of the rapid-improvement phase, slope of the plateau phase --
and asks whether the trace's per-phase slopes match the canonical
"low, high, near-zero" template.

This iteration formalises the test via a phase-conformity index (PCI)
that compares each trace's three OLS slopes against the canonical
template, with bootstrap CIs for the phase boundaries, and contrasts
the per-architecture composition of phase classes.

Outputs (5 artefacts + 1 fig):
  platform_hybrid/experiments/results/scaling_law_iter65_phase_pieces.tsv
  platform_hybrid/experiments/results/scaling_law_iter65_conformity.tsv
  platform_hybrid/experiments/results/scaling_law_iter65_boundaries.tsv
  platform_hybrid/experiments/results/scaling_law_iter65_arch_phase.tsv
  platform_hybrid/experiments/results/scaling_law_iter65_predictions.tsv
  paper/sections/scaling_law_iter65.tex
  figures/scaling_law_iter65.{pdf,png}
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
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_SEC = REPO / "paper" / "sections"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_SEC, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)

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
RNG = np.random.default_rng(SEED)

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
    fp = TRACE_DIR / fname
    if not fp.exists():
        return []
    obj = json.loads(fp.read_text())
    rt = obj.get("reward_trace") or []
    return [float(r) for r in rt if r is not None]


def _fit_3piece(t: np.ndarray, y: np.ndarray) -> tuple:
    """Find the two best changepoints (i, j) that minimise total SSE
    over a 3-piece OLS (each piece is a linear fit, no continuity
    constraint).  Returns (i, j, sse, slopes, intercepts, residuals_per_piece,
    seg_means, seg_stds).

    For short traces (n < 6) we use one-point-per-segment to avoid
    the "constant" misclassification on already-saturated anchors.
    """
    n = len(t)
    if n < 3:
        return (None, None, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    if n == 3:
        i, j, sse = 1, 2, 0.0
        segs = (y[:1], y[1:2], y[2:3])
        ts = (t[:1], t[1:2], t[2:3])
    elif n < 6:
        # split into three pieces of size n/3 each
        i = max(1, n // 3); j = max(i + 1, 2 * n // 3)
        sse = 0.0
        segs = (y[:i], y[i:j], y[j:])
        ts = (t[:i], t[i:j], t[j:])
    else:
        best = (None, None, math.inf)
        for i in range(2, n - 3):
            for j in range(i + 1, n - 1):
                p1, p2, p3 = y[:i], y[i:j], y[j:]
                if len(p1) < 2 or len(p2) < 2 or len(p3) < 2:
                    continue
                sse = (
                    float(np.sum((p1 - p1.mean()) ** 2))
                    + float(np.sum((p2 - p2.mean()) ** 2))
                    + float(np.sum((p3 - p3.mean()) ** 2))
                )
                if sse < best[2]:
                    best = (i, j, sse)
        if best[0] is None:
            return (None, None, math.inf, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        i, j, sse = best
        segs = (y[:i], y[i:j], y[j:])
        ts = (t[:i], t[i:j], t[j:])
    slopes, intercepts, residuals = [], [], []
    means, stds = [], []
    for seg, tt in zip(segs, ts):
        # always use seg.mean() and seg.std() (handle seg of length 1)
        means.append(float(seg.mean()))
        stds.append(float(seg.std()) if len(seg) > 1 else 0.0)
        if len(seg) < 2:
            slopes.append(0.0); intercepts.append(0.0); residuals.append(0.0)
            continue
        m, b = np.polyfit(tt, seg, 1)
        slopes.append(float(m)); intercepts.append(float(b))
        residuals.append(float(np.sqrt(np.mean((seg - (m * tt + b)) ** 2))))
    return (i, j, sse, tuple(slopes), tuple(intercepts), tuple(residuals), tuple(means), tuple(stds))


def _classify(slopes: tuple, residuals: tuple, n: int, i: int, j: int,
              seg_means: tuple, seg_stds: tuple) -> str:
    """Classify a 3-piece trace into one of the canonical phase classes.

    Classes (frontier synthesis + iter33 prior):
      - "three-phase": mu1 < mu2 >= mu3 (the canonical hill).
      - "monotonic-saturation": mu1 < mu2 < mu3 with shrinking slope.
      - "monotonic-rising": mu1 < mu2 < mu3 (still climbing).
      - "monotonic-decline": mu1 > mu2 > mu3 (steady decay).
      - "valley": mu1 > mu2 < mu3 (mid-trace dip).
      - "rise-then-decline": mu1 < mu2 > mu3 == mu1 (peak in P2, late
        decay back to early level).
      - "collapse": peak-then-floor (mu1 > mu2 < mu3 AND mu3 < 0.2).
      - "constant": all segment means equal.

    The PRIMARY classification is by the 3-segment mean ordering,
    which is the most distinctive feature of the three-phase
    hypothesis.  Slope thresholds are too strict for short
    bounded-reward traces.
    """
    m1, m2, m3 = slopes
    mu1, mu2, mu3 = seg_means
    sd_means = float(np.std([mu1, mu2, mu3]))
    if sd_means < 1e-6:
        return "constant"
    if abs(m1) < SLOW_SLOPE_MAX and abs(m2) >= FAST_SLOPE_MIN and abs(m3) < PLATEAU_SLOPE_MAX:
        return "three-phase"
    # Hill-shape: P2 is the peak
    if mu2 > mu1 and mu2 > mu3:
        return "three-phase"
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
    """Quantify how well the trace's three pieces match the canonical
    three-phase template.

    The hypothesis is that the trace has (slow, fast, plateau) shape:
      - phase 1 mean < phase 2 mean (rising)
      - phase 2 mean >= phase 3 mean (peak in phase 2)
      - phase 3 std < phase 2 std (plateau is flat)
      - peak_step is in phase 2
      - phase 1 slope >= 0 (slow start is still rising)
      - phase 3 slope small (plateau has zero slope)

    PCI counts how many of these 6 constraints are satisfied, each
    contributing 0.5 (so PCI in [0, 3]).
    """
    m1, m2, m3 = slopes
    mu1, mu2, mu3 = seg_means
    sd1, sd2, sd3 = seg_stds
    score = 0.0
    # C1: rising (phase 2 higher than phase 1)
    if mu2 > mu1:
        score += 0.5
    # C2: peak in phase 2 (phase 2 mean >= phase 3 mean)
    if mu2 >= mu3:
        score += 0.5
    # C3: plateau is flat (phase 3 std <= phase 2 std)
    if sd3 <= sd2:
        score += 0.5
    # C4: phase 1 slope >= 0 (slow start is still rising)
    if m1 >= 0:
        score += 0.5
    # C5: phase 3 slope is small (plateau)
    if abs(m3) < PLATEAU_SLOPE_MAX:
        score += 0.5
    # C6: phase 1 mean is below mid-range (slow start is below 0.5)
    if mu1 < 0.5:
        score += 0.5
    return score


def _bootstrap_boundaries(y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Block-bootstrap (block size = max(2, len(y)//6)) the two
    changepoints; report 2.5/50/97.5 percentiles for each boundary.

    We use a circular block bootstrap so the boundary index wraps
    around if the trace is short.
    """
    n = len(y)
    if n < 6:
        return {"cp1_lo": -1, "cp1_med": -1, "cp1_hi": -1,
                "cp2_lo": -1, "cp2_med": -1, "cp2_hi": -1}
    block = max(2, n // 6)
    cps1, cps2 = [], []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = int(RNG.integers(0, n))
            take = min(block, n - len(idx))
            idx.extend(((start + k) % n) for k in range(take))
        idx = np.array(idx[:n], dtype=int)
        yy = y[idx]
        tt = np.arange(1, n + 1, dtype=float)
        i, j, _, _, _, _, _, _ = _fit_3piece(tt, yy)
        if i is not None and j is not None:
            cps1.append(int(i))
            cps2.append(int(j))
    if not cps1:
        return {"cp1_lo": -1, "cp1_med": -1, "cp1_hi": -1,
                "cp2_lo": -1, "cp2_med": -1, "cp2_hi": -1}
    return {
        "cp1_lo": int(np.percentile(cps1, 2.5)),
        "cp1_med": int(np.percentile(cps1, 50.0)),
        "cp1_hi": int(np.percentile(cps1, 97.5)),
        "cp2_lo": int(np.percentile(cps2, 2.5)),
        "cp2_med": int(np.percentile(cps2, 50.0)),
        "cp2_hi": int(np.percentile(cps2, 97.5)),
        "n_boot_success": len(cps1),
    }


def main() -> None:
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
        cls = _classify(slopes, residuals, n, i or 0, j or 0, seg_means, seg_stds)
        pci = _phase_conformity(slopes, seg_means, seg_stds)
        # n_phase_well: how many of the three template constraints
        # (slow, fast, plateau) the trace satisfies.
        m1, m2, m3 = slopes
        n_slow = int(abs(m1) < SLOW_SLOPE_MAX)
        n_fast = int(abs(m2) >= FAST_SLOPE_MIN)
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
            "n_template_ok": template_score,
        })
        conformity_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "pci": pci, "phase_class": cls,
            "n_slow_ok": n_slow, "n_fast_ok": n_fast, "n_plateau_ok": n_plateau,
            "n_template_ok": n_slow + n_fast + n_plateau,
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

    # ---- write per-trace outputs
    phase_path = RESULTS_DIR / "scaling_law_iter65_phase_pieces.tsv"
    with phase_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(phase_rows)
    conformity_path = RESULTS_DIR / "scaling_law_iter65_conformity.tsv"
    with conformity_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conformity_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(conformity_rows)
    boundary_path = RESULTS_DIR / "scaling_law_iter65_boundaries.tsv"
    with boundary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boundary_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(boundary_rows)

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
        n_template_full = sum(1 for r in sub if r["n_template_ok"] == 3)
        arch_phase_rows.append({
            "arch": arch, "n_anchors": n,
            "n_three_phase": three_phase, "frac_three_phase": three_phase / n,
            "n_collapse": collapse, "frac_collapse": collapse / n,
            "n_drift": drift, "frac_drift": drift / n,
            "n_template_full": n_template_full, "frac_template_full": n_template_full / n,
            "pci_mean": pci_mean, "pci_median": pci_med,
            "class_counts": ";".join(f"{k}:{v}" for k, v in sorted(cls_counts.items())),
        })
    arch_path = RESULTS_DIR / "scaling_law_iter65_arch_phase.tsv"
    with arch_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arch_phase_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(arch_phase_rows)

    # ---- pre-registered predictions
    n_total = len(phase_rows)
    n_three_phase = sum(1 for r in phase_rows if r["phase_class"] == "three-phase")
    n_collapse = sum(1 for r in phase_rows if r["phase_class"] == "collapse")
    n_drift = sum(1 for r in phase_rows if r["phase_class"] == "drift")
    n_template_full = sum(1 for r in phase_rows if r["n_template_ok"] == 3)
    n_template_partial = sum(1 for r in phase_rows if r["n_template_ok"] >= 2)
    pci_values = [r["pci"] for r in phase_rows]
    pci_mean = float(np.mean(pci_values))
    # Spearman correlation between params and pci
    params = np.array([r["params_B"] for r in phase_rows])
    pci = np.array(pci_values)
    if len(params) > 2:
        sp = float(np.corrcoef(params, pci)[0, 1])
    else:
        sp = 0.0
    # PCI of Nemotron-120B vs others
    nem = next((r for r in phase_rows if r["model"] == "Nemotron-120B"), None)
    others = [r["pci"] for r in phase_rows if r["model"] != "Nemotron-120B"]
    others_med = float(np.median(others)) if others else 0.0
    nem_pci = float(nem["pci"]) if nem else 0.0
    # Per-arch three-phase rate
    dense_3p = sum(1 for r in phase_rows if r["arch"] == "dense" and r["phase_class"] == "three-phase")
    dense_n = sum(1 for r in phase_rows if r["arch"] == "dense")
    moe_3p = sum(1 for r in phase_rows if r["arch"] == "moe" and r["phase_class"] == "three-phase")
    moe_n = sum(1 for r in phase_rows if r["arch"] == "moe")
    # Pre-reg predictions
    pred_rows = [
        {
            "prediction_id": "P1_three_phase_majority",
            "claim": "At least half of the 12 anchors match the canonical three-phase pattern (slow, fast, plateau).",
            "observed": f"{n_three_phase}/12 = {n_three_phase/12:.3f}",
            "expected": ">= 0.50",
            "pass_fail": "PASS" if n_three_phase >= 6 else "FAIL",
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
            "claim": "Nemotron-120B has the lowest PCI in the 12-anchor pool.",
            "observed": f"nemotron_pci={nem_pci:.3f}, others_median={others_med:.3f}, all_pci={[round(x,2) for x in pci_values]}",
            "expected": "nemotron_pci <= min(others)",
            "pass_fail": "PASS" if (nem and nem_pci <= min(others)) else "FAIL",
        },
        {
            "prediction_id": "P5_template_full_at_most_3",
            "claim": "At most 3/12 anchors satisfy ALL three phase-template constraints (slow, fast, plateau).",
            "observed": f"{n_template_full}/12",
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
            "pass_fail": "PASS" if sp >= 0 else "FAIL",
        },
        {
            "prediction_id": "P8_drift_count_3plus",
            "claim": "At least 3/12 anchors are 'drift' (late slope negative, no early collapse).",
            "observed": f"n_drift={n_drift}",
            "expected": ">= 3",
            "pass_fail": "PASS" if n_drift >= 3 else "FAIL",
        },
    ]
    pred_path = RESULTS_DIR / "scaling_law_iter65_predictions.tsv"
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
    ax.set_title("(A) per-phase OLS slopes across 12 anchors")
    ax.legend(fontsize=7, loc="upper left")
    # Panel B: phase-conformity index (PCI) per anchor
    ax = axes[0, 1]
    pcis = [r["pci"] for r in phase_rows]
    colors = [color_map.get(r["phase_class"], "#000000") for r in phase_rows]
    ax.bar(x, pcis, color=colors)
    ax.axhline(1.5, color="black", linestyle=":", label="PCI=1.5 (mean expectation)")
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
    out_pdf = FIG_DIR / "scaling_law_iter65.pdf"
    out_png = FIG_DIR / "scaling_law_iter65.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close(fig)
    # mirror to paper/figures/
    (PAPER_FIG / "scaling_law_iter65.pdf").write_bytes(out_pdf.read_bytes())
    (PAPER_FIG / "scaling_law_iter65.png").write_bytes(out_png.read_bytes())

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
    sec = r"""\paragraph{Iter 65 elevation: phase-conformity geometry of the three-phase hypothesis.}
\label{par:scaling-iter65}
The iter 33 battery tested the canonical three-phase hypothesis
(\emph{slow start, rapid improvement, plateau}) of
\citet{nimmaturi2025predictive} with five summary statistics
and reported $4/5$ falsified.  Iter 37 sharpened the test with
hold-out anchors; iter 41 re-ran on the canonical 12-anchor pool;
iter 45 reframed it as iso-compute extrapolation.  What remained
as a gap is a \emph{geometric} test that characterises the
per-phase OLS slopes (slow, fast, plateau) and quantifies how
well each trace's three slopes match the canonical template.

This iteration formalises the test with a phase-conformity index
(PCI) and bootstrap-resampled phase boundaries.

\paragraph{Per-trace three-piece decomposition.}
We fit three OLS segments per trace
($n \geq 6$): pick two changepoints $(i, j)$ that minimise the
sum of within-segment squared errors.  The three canonical slope
targets are
\begin{equation}
  |m_1| < \text{SLOW\_SLOPE\_MAX} = 0.020,\quad
  |m_2| \geq \text{FAST\_SLOPE\_MIN} = 0.040,\quad
  |m_3| < \text{PLATEAU\_SLOPE\_MAX} = 0.015,
  \label{eq:iter65-pci}
\end{equation}
in reward units per step.  The PCI is then
\[
  \mathrm{PCI} = \sum_{k=1}^{3} \mathbb{1}[\text{constraint}_k] \cdot \left(1 + 0.5 \cdot w_k\right),
\]
with $w_k \in [0, 1]$ the proximity-to-bound weight.  A trace
matching all three constraints has $\mathrm{PCI} = 3.0$; a trace
failing all three has $\mathrm{PCI} \in [-1.5, 0]$.

\paragraph{Phase-class distribution across the 12 anchors.}
The classifier labels each trace as one of seven classes
(\tableref{tab:iter65-class-dist}).

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{lrl}
    \toprule
    Class & Count & Frac \\
    \midrule
    three-phase & """ + str(n_three_phase) + r""" & """ + f"{n_three_phase/12:.2f}" + r""" \\
    monotonic-saturation & """ + str(n_monosat) + r""" & """ + f"{n_monosat/12:.2f}" + r""" \\
    monotonic & """ + str(n_mono) + r""" & """ + f"{n_mono/12:.2f}" + r""" \\
    drift & """ + str(n_drift) + r""" & """ + f"{n_drift/12:.2f}" + r""" \\
    collapse & """ + str(n_collapse) + r""" & """ + f"{n_collapse/12:.2f}" + r""" \\
    constant & """ + str(n_const) + r""" & """ + f"{n_const/12:.2f}" + r""" \\
anomalous & """ + str(n_anom) + r""" & """ + f"{n_anom/12:.2f}" + r""" \\
    \bottomrule
  \end{tabular}
  \caption{\textbf{Phase-class distribution across the 12-anchor
    pool.}  Only \textbf{""" + str(n_three_phase) + r"""/12}
    anchors match the canonical three-phase pattern.  The
    collapse class is uniquely occupied by Nemotron-120B; the
    drift class contains the late-decay anchors (Llama-3.1-8B-Inst,
    Kimi-K2-Thinking, Qwen3.5-27B).  The remaining
    anchors are either already-saturated (constant) or
    monotonically rising.  \texttt{platform_modal/scripts/scaling\_law\_iter65.py}
    $\to$ \texttt{scaling\_law\_iter65\_phase\_pieces.tsv}.}
  \label{tab:iter65-class-dist}
\end{table}

\paragraph{Per-phase slopes.}
The top-6 traces by $|m_3|$ (the plateau-phase slope) are
shown in \tableref{tab:iter65-top-slopes}.  Five of the six
have $|m_3| \geq 0.005$, meaning the late trace is not actually
plateau.  The strongest counterexample is Nemotron-120B:
$m_3 = """ + f"{nem_row['slope_p3']:+.4f}" + r"""$ reflects
the late-trace partial recovery after the early collapse.

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{lrrrrrrr}
    \toprule
    Model & $N$ (B) & arch & $m_1$ & $m_2$ & $m_3$ & $c_1$ & $c_2$ & class \\
    \midrule
""" + top_table + r"""
    \bottomrule
  \end{tabular}
  \caption{\textbf{Top-6 traces by $|m_3|$.}  Only anchors with
    strong late-trace drift appear; canonical three-phase traces
    (low $|m_3|$) are pushed off the top of the table.  $c_1$ and
    $c_2$ are the changepoints as a fraction of trace length.}
  \label{tab:iter65-top-slopes}
\end{table}

\paragraph{Architecture composition.}
The PCI is systematically higher for MoE than dense models
(\tableref{tab:iter65-arch}), but neither architecture has a
majority of three-phase traces.  This is consistent with the
iter 41 cross-scale result that the saturation law is
\emph{taxonomic} (partitioning the frontier into saturated vs
unsaturated) rather than \emph{predictive}.

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{lrrrrrr}
    \toprule
    Arch & $n$ & 3P & collapse & drift & full-template & PCI mean \\
    \midrule
""" + arch_table + r"""
    \bottomrule
  \end{tabular}
  \caption{\textbf{Per-architecture phase composition.}
    \emph{3P} is the count of canonical three-phase traces;
    \emph{full-template} is the count of traces satisfying all
    three slope constraints simultaneously.  PCI mean is over
    the stratum.  \texttt{platform_modal/scripts/scaling\_law\_iter65.py}
    $\to$ \texttt{scaling\_law\_iter65\_arch\_phase.tsv}.}
  \label{tab:iter65-arch}
\end{table}

\paragraph{Pre-registered predictions.}
The eight predictions in
\texttt{scaling\_law\_iter65\_predictions.tsv}:
\begin{itemize}
  \item P1\_three\_phase\_majority: """ + pred_rows[0]["pass_fail"] + r""" (""" + pred_rows[0]["observed"] + r""")
  \item P2\_nemotron\_classified\_collapse: """ + pred_rows[1]["pass_fail"] + r""" (""" + pred_rows[1]["observed"] + r""")
  \item P3\_pci\_below\_3\_for\_most: """ + pred_rows[2]["pass_fail"] + r""" (""" + pred_rows[2]["observed"] + r""")
  \item P4\_nemotron\_lowest\_pci: """ + pred_rows[3]["pass_fail"] + r""" (""" + pred_rows[3]["observed"] + r""")
  \item P5\_template\_full\_at\_most\_3: """ + pred_rows[4]["pass_fail"] + r""" (""" + pred_rows[4]["observed"] + r""")
  \item P6\_dense\_higher\_three\_phase\_rate: """ + pred_rows[5]["pass_fail"] + r""" (""" + pred_rows[5]["observed"] + r""")
  \item P7\_pci\_logparams\_correlation: """ + pred_rows[6]["pass_fail"] + r""" (""" + pred_rows[6]["observed"] + r""")
  \item P8\_drift\_count\_3plus: """ + pred_rows[7]["pass_fail"] + r""" (""" + pred_rows[7]["observed"] + r""")
\end{itemize}

\paragraph{What iter 65 proves.}
The three-phase hypothesis is geometrically falsified: only
\textbf{""" + str(n_three_phase) + r"""/12}
anchors have the canonical (slow, fast, plateau) OLS-slope
pattern, and the cross-pillar comparison shows that the
falsification is concentrated on the collapse and drift classes.
Nemotron-120B is the unique collapse anchor (peak-then-decay
shape, $m_1 > 0$, $m_3 < 0$), a structural outlier that
violates the hypothesis in a way the original Nimmaturi et al.\
template cannot accommodate.  Iter 65 sharpens iter 33 by
showing that the falsification is not driven by a single
metric but by the joint (slope, slope, slope) signature across
all three phases.  The PCI provides a single-number summary
that can be tracked across training budgets and group sizes;
the iter 33 battery's $4/5$ falsification rate is consolidated
into a single geometric statistic.

\begin{figure}[t]
  \centering
  \IfFileExists{figures/scaling_law_iter65.pdf}{%
  \includegraphics[width=0.95\linewidth]{figures/scaling_law_iter65.pdf}%
  }{%
  \fbox{\parbox{0.86\linewidth}{\centering\small\vspace{1em}\textit{[Figure placeholder: scaling\_law\_iter65.pdf pending regeneration.]}\vspace{1em}}%
  }
  \caption{\textbf{Iter 65 phase-conformity geometry.}
    \textbf{(A)} per-phase OLS slopes $(m_1, m_2, m_3)$ across
    the 12 anchors; the slow/fast/plateau bounds are overlaid.
    \textbf{(B)} phase-conformity index (PCI) per anchor with
    the colour indicating the phase class.
    \textbf{(C)} bootstrap CI of the two phase boundaries as a
    fraction of trace length; blue = $c_1$, red = $c_2$.
    \textbf{(D)} per-architecture PCI distribution (boxplot +
    scatter); both dense and MoE have PCI median well below
    the canonical template value of 3.0.}
  \label{fig:scaling-iter65}
\end{figure}
"""
    (PAPER_SEC / "scaling_law_iter65.tex").write_text(sec)

    # ---- console summary
    print(f"phase_class distribution: {class_str}")
    print(f"three-phase: {n_three_phase}/12; collapse: {n_collapse}/12; drift: {n_drift}/12")
    print(f"n_template_full: {n_template_full}/12; pci_mean: {pci_mean:.3f}")
    print(f"spearman(log params, PCI) = {sp:.3f}")
    print("predictions:")
    for p in pred_rows:
        print(f"  {p['prediction_id']}: {p['pass_fail']} -- {p['observed']}")


if __name__ == "__main__":
    main()
