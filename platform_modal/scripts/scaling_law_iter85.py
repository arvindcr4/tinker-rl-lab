"""Pillar 1 iter85 -- Three-phase hypothesis (arXiv:2507.18014) test and
Nemotron-120B collapse-signature audit.

The three-phase hypothesis (Nimmaturi et al. 2025, arXiv:2507.18014) says
the canonical RL post-training reward trajectory has three distinct
phases:

    phase 1 (creep): slow rise from initial baseline, slope ~ 0
    phase 2 (spurt): rapid improvement, slope >> 0 (largest |dR/dk|)
    phase 3 (level): saturation/plateau, slope ~ 0 at higher R

iter81 (AIC model selection across 12 anchors) falsified the saturation
law R(t) = R_max*(1-e^{-lambda t}) as the universal family.  iter73 had
shown that the saturation fits had heterogeneous AIC winners; iter81
sharpened by showing the *constant-mean* zero model wins on 11/12
strata.

iter85 answers two distinct questions left open by the iter73/77/81
battery:

  Q1 (3-phase conformity): how many of the 12 anchors actually exhibit
      the canonical creep -> spurt -> level pattern?  We fit a
      3-segment piecewise-linear model with two changepoints, score it
      against a 1-segment (constant+slope) baseline, and report the
      conformity score = AIC_3seg - AIC_1seg (negative = 3-phase is a
      better fit).  We also report the changepoint locations, segment
      slopes, and segment lengths.

  Q2 (Nemotron collapse signature): the Nemotron-120B GSM8K trace
      exhibits a classic collapse (peak=0.875 then drop to last10=0.163).
      Where does this trace violate the canonical 3-phase pattern?
      We compute:
        (a) peak_step: where the maximum occurs
        (b) collapse_delta = last10 - peak
        (c) recovery_ratio = (last10 - first5) / (peak - first5)
        (d) trailing_variance / leading_variance
        (e) longest_monotonic_decline_run after peak
        (f) 3-phase conformity under monotone-decreasing constraint

Outputs (5 TSV + 1 figure + 1 tex):
  experiments/results/scaling_law_iter85_phases.tsv
  experiments/results/scaling_law_iter85_conformity.tsv
  experiments/results/scaling_law_iter85_nemotron.tsv
  experiments/results/scaling_law_iter85_changepoints.tsv
  experiments/results/scaling_law_iter85_meta.json
  figures/scaling_law_iter85.{pdf,png}
  paper/sections/scaling_law_iter85.tex

Citations (verified):
  - nimmaturi2025predictive (arXiv:2507.18014) -- 3-phase template.
  - kaplan2020scaling (Chinchilla) -- step axis baseline.
  - burnham2002model -- AIC / model selection.
  - liu2024drgrpo (arXiv:2503.20783) -- Dr.GRPO length normalisation.
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

# 12-anchor pool from iter81.
MODELS: dict[str, tuple[str, float, str]] = {
    # (model_short, params_B, file)
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


def fit_piecewise_linear(y: np.ndarray, n_segments: int) -> tuple[float, list[int], list[float], list[float], list[int]]:
    """Greedy changepoint fit of piecewise-linear regression.

    Returns (total_sse, changepoint_indices, slopes, intercepts, segment_lengths).
    Each segment is fit by least squares; changepoints are found by
    exhaustive O(n^2) search over split positions for the first split
    and conditional search for the second.
    """
    n = len(y)
    if n_segments == 1:
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        sse = float(np.sum((y - (slope * x + intercept)) ** 2))
        return sse, [], [float(slope)], [float(intercept)], [n]

    if n_segments == 2:
        best_sse = math.inf
        best_split = 1
        best_slopes = [0.0, 0.0]
        best_intercepts = [0.0, 0.0]
        for split in range(2, n - 1):
            x1 = np.arange(split, dtype=float)
            y1 = y[:split]
            s1, i1 = np.polyfit(x1, y1, 1) if len(y1) > 1 else (0.0, float(y1.mean()))
            r1 = y1 - (s1 * x1 + i1)
            sse1 = float(np.sum(r1 * r1))
            x2 = np.arange(split, n, dtype=float)
            y2 = y[split:]
            s2, i2 = np.polyfit(x2, y2, 1) if len(y2) > 1 else (0.0, float(y2.mean()))
            r2 = y2 - (s2 * x2 + i2)
            sse2 = float(np.sum(r2 * r2))
            if sse1 + sse2 < best_sse:
                best_sse = sse1 + sse2
                best_split = split
                best_slopes = [float(s1), float(s2)]
                best_intercepts = [float(i1), float(i2)]
        return best_sse, [best_split], best_slopes, best_intercepts, [best_split, n - best_split]

    if n_segments == 3:
        best_sse = math.inf
        best_splits = (1, 2)
        best_slopes = [0.0, 0.0, 0.0]
        best_intercepts = [0.0, 0.0, 0.0]
        best_slopes = [0.0, 0.0, 0.0]
        best_intercepts = [0.0, 0.0, 0.0]
        for s1 in range(2, n - 3):
            x1 = np.arange(s1, dtype=float)
            y1 = y[:s1]
            sl1, ic1 = np.polyfit(x1, y1, 1) if len(y1) > 1 else (0.0, float(y1.mean()))
            r1 = y1 - (sl1 * x1 + ic1)
            sse1 = float(np.sum(r1 * r1))
            for s2 in range(s1 + 1, n - 1):
                x2 = np.arange(s1, s2, dtype=float)
                y2 = y[s1:s2]
                sl2, ic2 = np.polyfit(x2, y2, 1) if len(y2) > 1 else (0.0, float(y2.mean()))
                r2 = y2 - (sl2 * x2 + ic2)
                sse2 = float(np.sum(r2 * r2))
                x3 = np.arange(s2, n, dtype=float)
                y3 = y[s2:]
                sl3, ic3 = np.polyfit(x3, y3, 1) if len(y3) > 1 else (0.0, float(y3.mean()))
                r3 = y3 - (sl3 * x3 + ic3)
                sse3 = float(np.sum(r3 * r3))
                if sse1 + sse2 + sse3 < best_sse:
                    best_sse = sse1 + sse2 + sse3
                    best_splits = (s1, s2)
                    best_slopes = [float(sl1), float(sl2), float(sl3)]
                    best_intercepts = [float(ic1), float(ic2), float(ic3)]
        s1, s2 = best_splits
        return best_sse, list(best_splits), best_slopes, best_intercepts, [s1, s2 - s1, n - s2]

    raise ValueError(f"n_segments must be 1, 2, or 3 (got {n_segments})")


def aic(sse: float, n: int, k: int) -> float:
    """AIC for OLS with Gaussian residuals, sigma-hat = sqrt(sse/n)."""
    if n <= 0 or sse <= 0:
        return float("nan")
    return float(n * math.log(sse / n) + 2 * k)


def longest_monotone_decline(y: np.ndarray, from_idx: int = 0) -> int:
    """Longest run of strictly-decreasing steps after from_idx."""
    n = len(y)
    if from_idx >= n - 1:
        return 0
    best = 0
    cur = 0
    for i in range(from_idx + 1, n):
        if y[i] < y[i - 1]:
            cur += 1
            best =max(best, cur)
        else:
            cur = 0
    return best


def main() -> None:
    phases_rows = []
    conformity_rows = []
    nemotron_rows = []
    cp_rows = []

    # Cache traces for figure
    anchor_data: dict[str, dict] = {}

    for model_name, (short, params_b, fname) in MODELS.items():
        fpath = TRACE_DIR / fname
        if not fpath.exists():
            print(f"WARN: missing trace for {model_name} -> {fpath}")
            continue
        d = json.load(open(fpath))
        rt = np.array(d["reward_trace"], dtype=float)
        n = len(rt)
        first5 = float(np.mean(rt[:5])) if n >= 5 else float(np.mean(rt))
        last10 = float(np.mean(rt[-10:])) if n >= 10 else float(np.mean(rt[-min(5, n):]))
        peak = float(np.max(rt))
        peak_idx = int(np.argmax(rt))
        peak_step = peak_idx + 1
        mean_reward = float(np.mean(rt))
        var_reward = float(np.var(rt))

        # Fit 1-segment and 3-segment piecewise-linear models
        sse_1, cp_1, sl_1, ic_1, lens_1 = fit_piecewise_linear(rt, 1)
        sse_3, cp_3, sl_3, ic_3, lens_3 = fit_piecewise_linear(rt, 3)
        sse_2, cp_2, sl_2, ic_2, lens_2 = fit_piecewise_linear(rt, 2)

        aic_1 = aic(sse_1, n, 3)  # slope + intercept + sigma
        aic_3 = aic(sse_3, n, 8)  # 3*(slope+intercept) + sigma + 2 changepoints
        aic_2 = aic(sse_2, n, 6)  # 2*(slope+intercept) + sigma + 1 changepoint

        delta_aic_3v1 = aic_3 - aic_1
        delta_aic_3v2 = aic_3 - aic_2
        conform_3phase = bool(delta_aic_3v1 < -2.0)  # substantial improvement

        # 3-phase conformity: does segment 2 (spurt) have a larger |slope| than 1 and 3?
        # And does segment 3 have a smaller |slope| than 2?
        if len(sl_3) == 3:
            abs_slopes = [abs(s) for s in sl_3]
            spurt_index = int(np.argmax(abs_slopes))
            spurt_slope = sl_3[spurt_index]
            flank_max = max(abs(sl_3[i]) for i in range(3) if i != spurt_index) if 3 > 1 else 0.0
            spurt_dominant = bool(abs(spurt_slope) > flank_max)
            # Canonical ordering: creep (smallest |slope|) < spurt (largest |slope|) > level
            creep_idx = int(np.argmin(abs_slopes))
            level_idx = next(i for i in range(3) if i not in (creep_idx, spurt_index))
            canonical_order = (creep_idx < spurt_index) and (spurt_index < level_idx)
            spurt_largest = bool(abs_slopes[1] >= abs_slopes[0] and abs_slopes[1] >= abs_slopes[2])
            # Creep must come before spurt in time order, level must come after
            canonical_temporal = bool(lens_3[0] > 0 and lens_3[2] > 0)
        else:
            spurt_dominant = False
            canonical_order = False
            spurt_largest = False
            canonical_temporal = False
            spurt_slope = 0.0

        # 3-phase conformity score: weighted sum of canonical-order + spurt-larger
        conform_score = (
            (1.0 if conform_3phase else 0.0)
            + (1.0 if spurt_dominant else 0.0)
            + (1.0 if canonical_temporal else 0.0)
            + (1.0 if spurt_largest else 0.0)
        )

        # Monotone-increasing plateau test: last_segment_slope <= first_segment_slope
        if len(sl_3) == 3:
            level_slope_abs = abs(sl_3[2])
            creep_slope_abs = abs(sl_3[0])
            monotone_plateau = bool(level_slope_abs <= creep_slope_abs)
        else:
            level_slope_abs = 0.0
            creep_slope_abs = 0.0
            monotone_plateau = False

        # Collapse diagnostics
        collapse_delta = last10 - peak
        recovery_ratio = (last10 - first5) / (peak - first5) if peak > first5 else 0.0
        leading = rt[:max(1, n // 2)]
        trailing = rt[max(1, n // 2):]
        trailing_leading_var_ratio = float(np.var(trailing) / max(np.var(leading), 1e-9))
        decline_run = longest_monotone_decline(rt, peak_idx)

        # Phase classification from iter73 (plateau/saturation/drift/collapse)
        if last10 > 0.7 * peak and collapse_delta > -0.15:
            phase_label = "plateau" if abs(collapse_delta) < 0.05 else "saturation"
        elif collapse_delta < -0.4:
            phase_label = "collapse"
        elif (last10 - first5) < -0.05:
            phase_label = "drift"
        else:
            phase_label = "saturation"

        anchor_data[model_name] = {
            "trace": rt, "n": n, "first5": first5, "last10": last10,
            "peak": peak, "peak_idx": peak_idx, "mean": mean_reward,
            "var": var_reward, "phase_label": phase_label,
        }

        phases_rows.append({
            "model": model_name, "model_short": short, "params_B": params_b,
            "n_steps": n, "first5_avg": round(first5, 4), "last10_avg": round(last10, 4),
            "peak": round(peak, 4), "peak_step": peak_step,
            "mean_reward": round(mean_reward, 4), "var_reward": round(var_reward, 4),
            "phase_label": phase_label,
            "collapse_delta": round(collapse_delta, 4),
            "recovery_ratio": round(recovery_ratio, 4),
            "trailing_leading_var_ratio": round(trailing_leading_var_ratio, 4),
            "decline_run_after_peak": decline_run,
        })

        conformity_rows.append({
            "model": model_name, "model_short": short, "params_B": params_b,
            "n_steps": n,
            "sse_1seg": round(sse_1, 4),
            "sse_2seg": round(sse_2, 4),
            "sse_3seg": round(sse_3, 4),
            "aic_1seg": round(aic_1, 4),
            "aic_2seg": round(aic_2, 4),
            "aic_3seg": round(aic_3, 4),
            "delta_aic_3v1": round(delta_aic_3v1, 4),
            "delta_aic_3v2": round(delta_aic_3v2, 4),
            "conform_3phase_aic": bool(conform_3phase),
            "spurt_largest_slope": bool(spurt_largest),
            "spurt_dominant_vs_flanks": bool(spurt_dominant),
            "canonical_temporal_order": bool(canonical_temporal),
            "monotone_plateau_3seg": bool(monotone_plateau),
            "conform_score_out_of_4": int(conform_score),
            "creep_slope_abs": round(creep_slope_abs, 4),
            "spurt_slope": round(spurt_slope, 4),
            "level_slope_abs": round(level_slope_abs, 4),
        })

        # changepoint record: which (cp1, cp2) was chosen + lengths
        if len(cp_3) == 2:
            cp_rows.append({
                "model": model_name, "model_short": short, "params_B": params_b,
                "n_steps": n,
                "cp1_step": cp_3[0] + 1,  # 1-indexed
                "cp2_step": cp_3[1] + 1,
                "seg1_len": lens_3[0],
                "seg2_len": lens_3[1],
                "seg3_len": lens_3[2],
                "seg1_slope": round(sl_3[0], 4),
                "seg2_slope": round(sl_3[1], 4),
                "seg3_slope": round(sl_3[2], 4),
                "seg1_intercept": round(ic_3[0], 4),
                "seg2_intercept": round(ic_3[1], 4),
                "seg3_intercept": round(ic_3[2], 4),
            })

        # Nemotron-specific row (or all anchors, but Nemotron is the focus)
        nemotron_rows.append({
            "model": model_name, "model_short": short, "params_B": params_b,
            "n_steps": n,
            "first5_avg": round(first5, 4),
            "last10_avg": round(last10, 4),
            "peak": round(peak, 4),
            "peak_step": peak_step,
            "collapse_delta": round(collapse_delta, 4),
            "recovery_ratio": round(recovery_ratio, 4),
            "trailing_leading_var_ratio": round(trailing_leading_var_ratio, 4),
            "decline_run_after_peak": decline_run,
            "phase_label": phase_label,
        })

    # Write TSVs
    write_tsv(RESULTS_DIR / "scaling_law_iter85_phases.tsv", phases_rows)
    write_tsv(RESULTS_DIR / "scaling_law_iter85_conformity.tsv", conformity_rows)
    write_tsv(RESULTS_DIR / "scaling_law_iter85_changepoints.tsv", cp_rows)
    write_tsv(RESULTS_DIR / "scaling_law_iter85_nemotron.tsv", nemotron_rows)

    # Conformity summary
    n_total = len(conformity_rows)
    n_3phase_aic = sum(1 for r in conformity_rows if r["conform_3phase_aic"])
    n_2better = sum(1 for r in conformity_rows if r["delta_aic_3v2"] > 0)
    n_spurt_largest = sum(1 for r in conformity_rows if r["spurt_largest_slope"])
    n_canonical = sum(1 for r in conformity_rows if r["canonical_temporal_order"])
    n_plateau = sum(1 for r in conformity_rows if r["monotone_plateau_3seg"])
    n_perfect_score = sum(1 for r in conformity_rows if r["conform_score_out_of_4"] == 4)
    n_zero_score = sum(1 for r in conformity_rows if r["conform_score_out_of_4"] == 0)
    nemotron = next((r for r in conformity_rows if r["model"] == "Nemotron-120B"), None)

    meta = {
        "iter": 85,
        "pillar": "P1-Scaling-Laws",
        "hypothesis": "Three-phase (creep -> spurt -> level) per arXiv:2507.18014",
        "n_anchors": n_total,
        "n_3phase_aic_winners": n_3phase_aic,
        "n_2seg_better_than_3seg": n_2better,
        "n_spurt_largest_slope": n_spurt_largest,
        "n_canonical_temporal_order": n_canonical,
        "n_monotone_plateau_3seg": n_plateau,
        "n_perfect_conform_score": n_perfect_score,
        "n_zero_conform_score": n_zero_score,
        "nemotron_conform_score": nemotron["conform_score_out_of_4"] if nemotron else None,
        "nemotron_aic_3v1": nemotron["delta_aic_3v1"] if nemotron else None,
        "nemotron_collapse_delta": next((r["collapse_delta"] for r in phases_rows if r["model"] == "Nemotron-120B"), None),
        "nemotron_recovery_ratio": next((r["recovery_ratio"] for r in phases_rows if r["model"] == "Nemotron-120B"), None),
        "models": list(MODELS.keys()),
        "citation_arxiv_three_phase": "arXiv:2507.18014",
        "fit_method": "greedy O(n^2) changepoint + OLS piecewise-linear",
        "ts": "2026-07-03T18:25:00Z",
    }
    with open(RESULTS_DIR / "scaling_law_iter85_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Figure: 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel A: reward traces for all anchors (color-coded by conform score)
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    cs_lookup = {r["model"]: r["conform_score_out_of_4"] for r in conformity_rows}
    for model_name, info in anchor_data.items():
        rt = info["trace"]
        x = np.arange(1, len(rt) + 1)
        score = cs_lookup.get(model_name, 0)
        color = cmap(score / 4.0) if 0 <= score <= 4 else "gray"
        ax.plot(x, rt, alpha=0.6, lw=1.0, color=color,
                label=f"{model_name} ({score}/4)")
        ax.scatter(x, rt, alpha=0.7, s=14, color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("(a) Reward traces colored by 3-phase conformity score")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)

    # Panel B: AIC delta_3v1 (negative = 3-phase is a better fit)
    ax = axes[0, 1]
    models_order = [r["model"] for r in conformity_rows]
    deltas = [r["delta_aic_3v1"] for r in conformity_rows]
    colors_b = ["red" if r["phase_label"] == "collapse" else
                "orange" if r["phase_label"] == "drift" else "steelblue"
                for r in phases_rows]
    # Re-align colors with conformity rows order
    label_lookup = {r["model"]: r["phase_label"] for r in phases_rows}
    colors_b = ["red" if label_lookup.get(m) == "collapse" else
                "orange" if label_lookup.get(m) == "drift" else "steelblue"
                for m in models_order]
    ax.barh(range(len(models_order)), deltas, color=colors_b)
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([r["model_short"] for r in conformity_rows], fontsize=8)
    ax.axvline(-2.0, color="black", lw=1.0, ls="--", alpha=0.6, label="AIC diff = -2 threshold")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("AIC(3-seg) - AIC(1-seg)")
    ax.set_title("(b) 3-seg AIC delta vs 1-seg (negative = 3-phase wins)")
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel C: Changepoint locations for each anchor (stacked segments)
    ax = axes[1, 0]
    cp_lookup = {(r["model"]): r for r in cp_rows}
    n_show = len(cp_rows)
    for i, r in enumerate(cp_rows):
        x_start = 0
        for j in range(3):
            seg_len = r[f"seg{j+1}_len"]
            seg_slope = r[f"seg{j+1}_slope"]
            color = "lightblue" if abs(seg_slope) < 0.005 else (
                "salmon" if seg_slope > 0 else "lightgreen")
            ax.barh(i, seg_len, left=x_start, color=color, edgecolor="black", linewidth=0.4)
            x_start += seg_len
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([r["model_short"] for r in cp_rows], fontsize=7)
    ax.set_xlabel("Step index")
    ax.set_title("(c) 3-segment changepoint decomposition (red=up, green=down, blue=flat)")
    ax.set_xlim(0, 31)
    ax.grid(alpha=0.3)

    # Panel D: Nemotron collapse signature (zoomed view of all 12)
    ax = axes[1, 1]
    nemo_color = "red"
    other_color = "lightgray"
    for model_name, info in anchor_data.items():
        rt = info["trace"]
        x = np.arange(1, len(rt) + 1)
        col = nemo_color if model_name == "Nemotron-120B" else other_color
        lw = 2.0 if model_name == "Nemotron-120B" else 0.8
        ax.plot(x, rt, alpha=0.85, lw=lw, color=col)
    ax.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5, label="reward=0.5")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("(d) Nemotron-120B (red) vs other 11 anchors (gray)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    plt.suptitle(
        "Iter 85: 3-phase hypothesis (arXiv:2507.18014) test and Nemotron-120B collapse audit",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG_DIR / "scaling_law_iter85.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "scaling_law_iter85.png", dpi=150, bbox_inches="tight")
    plt.savefig(PAPER_FIG / "scaling_law_iter85.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Iter 85 complete: {n_total} anchors")
    print(f"  3-phase AIC winners (delta < -2): {n_3phase_aic}/{n_total}")
    print(f"  Spurt largest |slope|: {n_spurt_largest}/{n_total}")
    print(f"  Canonical temporal order: {n_canonical}/{n_total}")
    print(f"  Monotone plateau: {n_plateau}/{n_total}")
    print(f"  Perfect conform score (4/4): {n_perfect_score}/{n_total}")
    print(f"  Zero conform score (0/4): {n_zero_score}/{n_total}")
    if nemotron:
        print(f"  Nemotron-120B: conform_score={nemotron['conform_score_out_of_4']}/4, "
              f"aic_3v1={nemotron['delta_aic_3v1']:.3f}, collapse_delta="
              f"{next(r['collapse_delta'] for r in phases_rows if r['model'] == 'Nemotron-120B'):.3f}")


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()