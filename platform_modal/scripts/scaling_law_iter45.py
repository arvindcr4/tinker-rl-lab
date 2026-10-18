"""Exploratory associations between fitted training rewards and a compute proxy.

Only explicitly identified observed traces may supply saturation-fit targets.
The proxy P_B * n_steps * 512 uses an assumed response length, not measured
FLOPs. Architecture strata do not establish matched training protocols.
OLS fits raw fitted R_max against log10(proxy), so its coefficient is a
semi-log reward slope, not a power-law exponent. These descriptive comparisons
cannot identify an optimal model/data allocation or a causal compute effect.
Outputs are written only by main into a separate review candidate directory.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from ._analysis_common import saturation as model_saturation
except ImportError:  # direct script invocation
    from _analysis_common import saturation as model_saturation

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
DATA = REPO / "experiments" / "results" / "scaling_law_extended_frontier.tsv"
RESULTS = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"

L_BAR = 512.0  # assumed tokens/response; no measured token accounting
B_BOOT = 400


def fit_saturation(
    n_steps,
    r_first,
    r_final,
    r_mean,
    r_peak,
    r_var,
    early_mean,
    late_mean,
    zero_frac,
    *,
    observed_trace=None,
):
    """Fit actual ordered rewards; legacy summary arguments are compatibility-only.

    Summary arguments cannot supply a target or affect the observed-trace fit.
    """
    if observed_trace is None:
        return float("nan"), float("nan"), False
    y = np.asarray(observed_trace, dtype=float)
    if y.ndim != 1 or len(y) < 4 or not np.all(np.isfinite(y)):
        return float("nan"), float("nan"), False
    t = np.arange(1, len(y) + 1, dtype=float)
    try:
        popt, _ = curve_fit(
            model_saturation,
            t,
            y,
            p0=[min(1.9, max(0.05, float(y.mean()))), 0.3],
            bounds=([0.0, 1e-3], [2.0, 5.0]),
            maxfev=8000,
        )
        r_max_fit = float(popt[0])
        lam_fit = float(popt[1])
        return r_max_fit, lam_fit, True
    except Exception:
        return float("nan"), float("nan"), False


def ols(x, y):
    """Plain OLS slope/intercept + Spearman rank corr (no scipy.stats needed)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    xb, yb = x.mean(), y.mean()
    num = float(np.sum((x - xb) * (y - yb)))
    den = float(np.sum((x - xb) ** 2))
    slope = num / den if den > 1e-12 else float("nan")
    intercept = yb - slope * xb
    try:
        from ._review27_rank import rank_correlation
    except ImportError:
        from _review27_rank import rank_correlation
    spear = rank_correlation(x, y)
    return slope, intercept, spear


def spearman_pvalue(x, y, b=2000):
    try:
        from ._review27_rank import spearman
    except ImportError:
        from _review27_rank import spearman
    return spearman(x, y, b=b)[1]


def main(output_dir=None, source_root=None) -> None:
    global REPO, CANDIDATE, DATA, RESULTS, FIG_DIR, PAPER_SEC
    if source_root is not None:
        REPO = Path(source_root)
    CANDIDATE = (
        Path(output_dir) if output_dir is not None else REPO / "experiments/results/review27_v3"
    )
    DATA = REPO / "experiments/results/scaling_law_extended_frontier.tsv"
    RESULTS, FIG_DIR, PAPER_SEC = CANDIDATE, CANDIDATE / "figures", CANDIDATE / "sections"
    for directory in (RESULTS, FIG_DIR, PAPER_SEC):
        directory.mkdir(parents=True, exist_ok=True)
    with open(DATA) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            if k in {"run_id", "trace_file", "trace_sha256", "model", "arch", "family"}:
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass

    # ---- (1) per-anchor compute proxy + R_max ----
    comp_rows = []
    try:
        from ._review27_rank import observed_trace
    except ImportError:
        from _review27_rank import observed_trace
    for r in rows:
        if not math.isfinite(float(r["params_B"])) or float(r["params_B"]) <= 0:
            raise ValueError("positive finite parameter count required")
        trace, trace_identity = observed_trace(r, REPO)
        rmax, lam, ok = fit_saturation(
            r["n_steps"],
            r["r_first"],
            r["r_final"],
            r["r_mean"],
            r["r_peak"],
            r["r_var"],
            r["early_mean"],
            r["late_mean"],
            r["zero_frac"],
            observed_trace=trace,
        )
        # Compute proxy in arbitrary units (params_B * n_steps * L_bar).
        # It omits rollout counts, measured lengths, and active-parameter accounting.
        c_proxy = float(r["params_B"]) * float(r["n_steps"]) * L_BAR
        log_c = math.log10(c_proxy) if c_proxy > 0 else float("nan")
        log_p = math.log10(float(r["params_B"])) if r["params_B"] > 0 else float("nan")
        comp_rows.append(
            {
                "model": r["model"],
                "params_B": float(r["params_B"]),
                "arch": r["arch"],
                "family": r["family"],
                "n_steps": int(r["n_steps"]),
                "C_proxy": round(c_proxy, 1),
                "log10_C": round(log_c, 4),
                "log10_P": round(log_p, 4),
                "R_max": round(rmax, 4) if ok else float("nan"),
                "lambda": round(lam, 4) if ok else float("nan"),
                "r_mean": r["r_mean"],
                "zero_frac": r["zero_frac"],
                "fit_ok": ok,
                "status": "unavailable",
                "reason": "",
                "trace_file": "",
                "trace_sha256": "",
                "run_id": "",
                **trace_identity,
                "evidence_kind": "measured_trace_fit" if ok else "unavailable",
            }
        )

    if not any(r["fit_ok"] for r in comp_rows):
        raise ValueError("No measured reward traces; summary reconstruction is not permitted")

    with open(RESULTS / "scaling_law_iter45_compute_proxy_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(comp_rows)

    # ---- (2) architecture-stratified semi-log associations ----
    fit_rows = []
    by_arch = {"all": comp_rows}
    for arch in ("dense", "moe"):
        by_arch[arch] = [r for r in comp_rows if r["arch"] == arch]
    for label, recs in by_arch.items():
        recs_ok = [r for r in recs if r["fit_ok"] and r["R_max"] > 0]
        if len(recs_ok) < 3:
            fit_rows.append(
                {
                    "stack": label,
                    "n_used": len(recs_ok),
                    "semi_log_reward_slope": float("nan"),
                    "intercept": float("nan"),
                    "spear_logC_Rmax": float("nan"),
                    "p_perm": float("nan"),
                    "spear_logP_Rmax": float("nan"),
                    "p_perm_logP": float("nan"),
                    "median_Rmax": float("nan"),
                    "note": "insufficient",
                }
            )
            continue
        log_c = np.array([r["log10_C"] for r in recs_ok])
        log_p = np.array([r["log10_P"] for r in recs_ok])
        rmax = np.array([r["R_max"] for r in recs_ok])
        slope_c, int_c, spear_c = ols(log_c, rmax)
        _, _, spear_p = ols(log_p, rmax)
        p_c = spearman_pvalue(log_c, rmax, b=B_BOOT)
        p_p = spearman_pvalue(log_p, rmax, b=B_BOOT)
        fit_rows.append(
            {
                "stack": label,
                "n_used": len(recs_ok),
                "semi_log_reward_slope": round(float(slope_c), 4),
                "intercept": round(float(int_c), 4),
                "spear_logC_Rmax": round(float(spear_c), 4),
                "p_perm": round(float(p_c), 4),
                "spear_logP_Rmax": round(float(spear_p), 4),
                "p_perm_logP": round(float(p_p), 4),
                "median_Rmax": round(float(np.median(rmax)), 4),
                "note": "OLS R_max = intercept + semi_log_reward_slope * log10(C_proxy); descriptive",
            }
        )

    with open(RESULTS / "scaling_law_iter45_scaling_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fit_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(fit_rows)

    # ---- (3) iso-compute bands: pair anchors with similar C, compare R_max ----
    iso_rows = []
    recs_ok = [r for r in comp_rows if r["fit_ok"] and r["R_max"] > 0]
    # Greedy pair: for each anchor, find the nearest-C partner of a different params_B.
    used = set()
    sorted_by_c = sorted(recs_ok, key=lambda r: r["log10_C"])
    for i, r1 in enumerate(sorted_by_c):
        if r1["model"] in used:
            continue
        best = None
        best_gap = float("inf")
        for j, r2 in enumerate(sorted_by_c):
            if j == i or r2["model"] in used or r2["params_B"] == r1["params_B"]:
                continue
            gap = abs(r2["log10_C"] - r1["log10_C"])
            if gap < best_gap:
                best_gap = gap
                best = r2
        if best is None or best_gap > 0.20:  # require |delta log10 C| < 0.20
            continue
        delta_rmax = r1["R_max"] - best["R_max"]
        rel = delta_rmax / max(abs(best["R_max"]), 1e-9)
        iso_rows.append(
            {
                "model_a": r1["model"],
                "params_B_a": r1["params_B"],
                "arch_a": r1["arch"],
                "logC_a": round(r1["log10_C"], 4),
                "Rmax_a": round(r1["R_max"], 4),
                "model_b": best["model"],
                "params_B_b": best["params_B"],
                "arch_b": best["arch"],
                "logC_b": round(best["log10_C"], 4),
                "Rmax_b": round(best["R_max"], 4),
                "delta_logC": round(r1["log10_C"] - best["log10_C"], 4),
                "delta_Rmax": round(delta_rmax, 4),
                "rel_delta_Rmax": round(rel, 4),
            }
        )
        used.add(r1["model"])
        used.add(best["model"])

    if iso_rows:
        with open(RESULTS / "scaling_law_iter45_isocompute_v2.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(iso_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(iso_rows)
    else:
        # always write something
        with open(RESULTS / "scaling_law_iter45_isocompute_v2.tsv", "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "model_a",
                    "params_B_a",
                    "arch_a",
                    "logC_a",
                    "Rmax_a",
                    "model_b",
                    "params_B_b",
                    "arch_b",
                    "logC_b",
                    "Rmax_b",
                    "delta_logC",
                    "delta_Rmax",
                    "rel_delta_Rmax",
                ],
                delimiter="\t",
            )
            w.writeheader()

    # ---- (4) predictions P1-P3 ----
    pred_rows = []
    n_pairs = len(iso_rows)
    max_rel = (
        float(np.max([abs(r["rel_delta_Rmax"]) for r in iso_rows])) if iso_rows else float("nan")
    )
    p1_pass = max_rel == max_rel and max_rel > 0.05
    pred_rows.append(
        {
            "prediction": "P1_isocompute_NOT_invariant",
            "value": round(max_rel, 4) if max_rel == max_rel else float("nan"),
            "pass": bool(p1_pass) if iso_rows else None,
            "note": f"max|delta_Rmax|/Rmax across {n_pairs} iso-compute pairs",
        }
    )

    # P2: positive semi-log association in any architecture stratum
    pos = [
        r
        for r in fit_rows
        if r["semi_log_reward_slope"] == r["semi_log_reward_slope"]
        and r["semi_log_reward_slope"] > 0
    ]
    fits_dense_row = next((r for r in fit_rows if r["stack"] == "dense"), None)
    fits_moe_row = next((r for r in fit_rows if r["stack"] == "moe"), None)
    fits_all_row = next((r for r in fit_rows if r["stack"] == "all"), None)
    pred_rows.append(
        {
            "prediction": "P2_positive_descriptive_slope",
            "value": len(pos),
            "pass": bool(len(pos) >= 1)
            if any(math.isfinite(r["semi_log_reward_slope"]) for r in fit_rows)
            else None,
            "note": (
                f"slope_dense={fits_dense_row['semi_log_reward_slope']:.3f}, "
                f"slope_moe={fits_moe_row['semi_log_reward_slope']:.3f}, "
                f"slope_all={fits_all_row['semi_log_reward_slope']:.3f}"
                if fits_all_row
                else "semi-log fit unavailable"
            ),
        }
    )

    # No scientific grade is available for compute allocation from this proxy.
    pred_rows.append(
        {
            "prediction": "P3_allocation_not_identified",
            "value": float("nan"),
            "pass": None,
            "note": "UNAVAILABLE: proxy accounting and an additive association do not identify an allocation optimum",
        }
    )
    for result in pred_rows:
        result["analysis_kind"] = "exploratory_descriptive"
        result["status"] = (
            "unavailable"
            if result["pass"] is None
            else "criterion_met"
            if result["pass"]
            else "criterion_not_met"
        )

    if pred_rows:
        with open(RESULTS / "scaling_law_iter45_predictions_v2.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(pred_rows)
    else:
        with open(RESULTS / "scaling_law_iter45_predictions_v2.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["prediction", "pass", "note"], delimiter="\t")
            w.writeheader()

    # ---- (5) summary ----
    n_anchors = len(comp_rows)
    n_ok = sum(1 for r in comp_rows if r["fit_ok"])
    n_total_iso = len(iso_rows)
    n_preds_pass = sum(1 for r in pred_rows if r["pass"])
    fits_dense = next((r for r in fit_rows if r["stack"] == "dense"), None)
    fits_moe = next((r for r in fit_rows if r["stack"] == "moe"), None)
    summary = {
        "n_anchors": n_anchors,
        "n_fits_ok": n_ok,
        "assumed_L_bar_tokens": L_BAR,
        "analysis_kind": "exploratory_descriptive",
        "allocation_status": "not_identified",
        "n_iso_compute_pairs": n_total_iso,
        "n_exploratory_criteria": len(pred_rows),
        "n_criteria_available": sum(r["pass"] is not None for r in pred_rows),
        "n_criteria_met": n_preds_pass,
        "semi_log_reward_slope_dense": fits_dense["semi_log_reward_slope"]
        if fits_dense
        else float("nan"),
        "semi_log_reward_slope_moe": fits_moe["semi_log_reward_slope"]
        if fits_moe
        else float("nan"),
        "spear_logC_Rmax_dense": fits_dense["spear_logC_Rmax"] if fits_dense else float("nan"),
        "spear_logC_Rmax_moe": fits_moe["spear_logC_Rmax"] if fits_moe else float("nan"),
        "max_abs_rel_delta_Rmax_iso": (
            float(max(abs(r["rel_delta_Rmax"]) for r in iso_rows)) if iso_rows else float("nan")
        ),
        "median_abs_rel_delta_Rmax_iso": (
            float(np.median([abs(r["rel_delta_Rmax"]) for r in iso_rows]))
            if iso_rows
            else float("nan")
        ),
    }
    with open(RESULTS / "scaling_law_iter45_summary_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()), delimiter="\t")
        w.writeheader()
        w.writerow(summary)

    # ---- (6) figure: 4-panel ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    arch_color = {"dense": "tab:blue", "moe": "tab:red"}
    for i, r in enumerate(comp_rows):
        if not r["fit_ok"]:
            continue
        ax.scatter(
            r["log10_C"],
            r["R_max"],
            c=arch_color[r["arch"]],
            s=70,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.7,
        )
        ax.annotate(
            r["model"][:10],
            (r["log10_C"], r["R_max"]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    # OLS lines
    xs = np.linspace(
        min(r["log10_C"] for r in comp_rows if r["fit_ok"]),
        max(r["log10_C"] for r in comp_rows if r["fit_ok"]),
        50,
    )
    for arch, color, label in (("dense", "tab:blue", "dense OLS"), ("moe", "tab:red", "MoE OLS")):
        recs = [r for r in comp_rows if r["arch"] == arch and r["fit_ok"]]
        if len(recs) >= 3:
            log_c = np.array([r["log10_C"] for r in recs])
            rmax = np.array([r["R_max"] for r in recs])
            slope, intercept, _ = ols(log_c, rmax)
            if slope == slope:
                ax.plot(
                    xs,
                    intercept + slope * xs,
                    "--",
                    color=color,
                    alpha=0.6,
                    label=f"{label} slope={slope:.3f}",
                )
    ax.set_xlabel(r"$\log_{10}(C_\mathrm{proxy})$  (params_B * n_steps * 512)")
    ax.set_ylabel(r"Fitted saturation $R_{\max}$")
    ax.set_title("Fitted training reward vs assumed compute proxy")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if iso_rows:
        params_a = [r["params_B_a"] for r in iso_rows]
        params_b = [r["params_B_b"] for r in iso_rows]
        delta = [r["rel_delta_Rmax"] for r in iso_rows]
        ax2.bar(range(len(iso_rows)), delta, color="tab:purple", alpha=0.8)
        ax2.set_xticks(range(len(iso_rows)))
        ax2.set_xticklabels(
            [f"({int(p_a)}B vs {int(p_b)}B)" for p_a, p_b in zip(params_a, params_b, strict=False)],
            rotation=45,
            fontsize=7,
        )
        ax2.axhline(0.05, color="k", linestyle="--", alpha=0.5, label="5% threshold")
        ax2.axhline(-0.05, color="k", linestyle="--", alpha=0.5)
        ax2.set_ylabel(r"$\Delta R_{\max} / R_{\max}^b$")
        ax2.set_title("Nearby-proxy pairs: relative fitted reward gap")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_law_iter45_v2.pdf")
    fig.savefig(FIG_DIR / "scaling_law_iter45_v2.png", dpi=150)
    plt.close(fig)

    # ---- (7) data-derived report; no retrospective confirmation claims ----
    def tex_escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [
        r"\subsection{Descriptive compute-proxy associations (iter45)}",
        r"\label{sec:scaling-law-iter45}",
        r"Only identified observed reward traces supply the fitted training-reward targets. "
        r"The proxy is $C=P_{B}n_{\mathrm{steps}}512$, with assumed response length. "
        r"It omits actual response counts, measured token lengths, and active-parameter accounting. "
        r"Architecture strata need not share a matched training protocol.",
        r"\paragraph{Semi-log associations.} The fitted equation is "
        r"$\widehat R_{\max}=b+s\log_{10}C$. The coefficient $s$ is a reward slope per "
        r"decade of the proxy; it is not a power-law exponent or a causal effect.",
        r"\begin{tabular}{lrrr}",
        r"stratum & $n$ & slope $s$ & Spearman $\rho$ \\",
    ]
    for result in fit_rows:
        lines.append(
            f"{tex_escape(result['stack'])} & {result['n_used']} & "
            f"{result['semi_log_reward_slope']:.3f} & {result['spear_logC_Rmax']:.3f}" + " \\\\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            f"There are {n_ok} successful observed-trace fits and {n_total_iso} nearby-proxy pairs. "
            f"Maximum absolute relative fitted-reward gap is {summary['max_abs_rel_delta_Rmax_iso']:.3f}; "
            f"median is {summary['median_abs_rel_delta_Rmax_iso']:.3f}.",
            r"\paragraph{Exploratory criteria.}",
        ]
    )
    for result in pred_rows:
        lines.append(tex_escape(f"{result['prediction']}: {result['status']}. {result['note']}"))
    lines.append(
        r"These criteria are descriptive and have no timestamped prospective protocol. "
        r"Nearby proxy values are not matched measured-compute budgets. An allocation optimum "
        r"is unavailable: it requires measured resource accounting, feasible allocations, matched "
        r"protocols, and a validated trade-off model. No scientific grade is assigned to P3."
    )
    sec_path = PAPER_SEC / "scaling_law_iter45_v2.tex"
    sec_path.write_text("\n".join(lines) + "\n")
    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(
        __file__, RESULTS, [DATA] + [REPO / r["trace_file"] for r in comp_rows if r["trace_file"]]
    )
    print(
        f"iter45 done. fits_ok={n_ok}/{n_anchors} "
        f"iso_pairs={n_total_iso} preds_pass={n_preds_pass}/{len(pred_rows)}"
    )


if __name__ == "__main__":
    main()
