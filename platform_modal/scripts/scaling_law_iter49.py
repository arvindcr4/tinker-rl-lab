"""Exploratory two-predictor associations and leave-one-anchor-out residuals.

The model is R_max = a*log10(P_B) + b*log10(C_proxy) + c, fitted to
identified observed-trace saturation fits. Its coefficients are semi-log
reward slopes. The additive compute term is common to all candidates at a
fixed budget, so candidate ordering cannot change with that budget. Candidate
scores are descriptive extrapolations, not a feasible allocation optimum.
No prospective confirmation or matched-protocol causal inference is claimed.
"""

from __future__ import annotations

import csv
import math
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
SRC = CANDIDATE / "scaling_law_iter45_compute_proxy_v2.tsv"
PHASE_SRC = CANDIDATE / "scaling_law_iter33_phase_score.tsv"
RESULTS = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"


def ols_3var(x1, x2, y):
    """OLS y = a*x1 + b*x2 + c. Returns a, b, c, R^2, fitted array, residuals."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    y = np.asarray(y, float)
    if x1.ndim != 1 or x1.shape != x2.shape or x1.shape != y.shape:
        raise ValueError("aligned one-dimensional OLS inputs required")
    if len(y) < 3 or not all(np.isfinite(v).all() for v in (x1, x2, y)):
        raise ValueError("at least three finite OLS observations required")
    X = np.column_stack([x1, x2, np.ones_like(x1)])
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError("two-predictor design is rank deficient")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ymean_safe(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), float(coef[2]), r2, yhat, y - yhat


def ymean_safe(y):
    return float(np.mean(y))


def spearman_pvalue(x, y, b=2000):
    try:
        from ._review27_rank import spearman
    except ImportError:
        from _review27_rank import spearman
    return spearman(x, y, b=b)


def load_iter45():
    """Fail closed on unverified, synthetic, duplicate, or unidentified targets."""
    try:
        from ._review27_rank import observed_trace
    except ImportError:
        from _review27_rank import observed_trace
    with open(SRC) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("requires measured-trace fits with provenance")
    identities = set()
    source_hashes = set()
    numeric = {"params_B", "n_steps", "C_proxy", "log10_C", "log10_P", "R_max", "lambda"}
    for row in rows:
        if row.get("evidence_kind") != "measured_trace_fit":
            raise ValueError(
                "reconstructed/legacy/synthetic targets rejected; measured-trace fits required"
            )
        for key in ("run_id", "trace_file", "trace_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or value.strip().lower() in {"", "nan", "none", "null"}:
                raise ValueError("trace source hashes, source paths, and run identities required")
        if not re.fullmatch(r"[0-9a-f]{64}", row["trace_sha256"]):
            raise ValueError("invalid trace source hash")
        if row["run_id"] in identities or row["trace_sha256"] in source_hashes:
            raise ValueError("duplicate run identity or source trace")
        for key in numeric:
            try:
                row[key] = float(row[key])
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"finite {key} required") from exc
            if not math.isfinite(row[key]):
                raise ValueError(f"finite {key} required")
        if row.get("fit_ok") not in (True, "True", "true", "1") or row["R_max"] <= 0:
            raise ValueError("successful positive measured-trace fit required")
        if row["params_B"] <= 0 or row["n_steps"] < 4 or not row["n_steps"].is_integer():
            raise ValueError("positive parameters and valid trace length required")
        if row["C_proxy"] <= 0 or not math.isclose(
            row["log10_P"], math.log10(row["params_B"]), abs_tol=1e-4
        ):
            raise ValueError("inconsistent parameter or compute coordinates")
        if not math.isclose(row["log10_C"], math.log10(row["C_proxy"]), abs_tol=1e-4):
            raise ValueError("inconsistent compute coordinates")
        trace, identity = observed_trace(row, REPO)
        source = json.loads((REPO / identity["trace_file"]).read_text())
        for record in (row, source):
            if any(
                str(record.get(key, "")).lower() in {"true", "1", "yes"}
                for key in ("synthetic", "is_synthetic", "reconstructed")
            ):
                raise ValueError("synthetic/reconstructed traces are not measured evidence")
            kind = str(record.get("evidence_kind", "")).lower()
            if "synthetic" in kind or "reconstruct" in kind:
                raise ValueError("synthetic/reconstructed traces are not measured evidence")
        if trace is None:
            raise ValueError("observed trace is unavailable")
        identities.add(row["run_id"])
        source_hashes.add(row["trace_sha256"])
    return rows


def budget_scores(log_params, log_budget, a, b, intercept):
    """Descriptive additive scores; budget changes shift every candidate equally."""
    return a * np.asarray(log_params, float) + b * float(log_budget) + intercept


def load_phase():
    """Attach deterministic phase classifier to each anchor (iter33 score)."""
    if not PHASE_SRC.exists():
        return {}
    with open(PHASE_SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        if "model" in r and "phase_classifier" in r:
            out[r["model"]] = r["phase_classifier"]
    return out


def main(output_dir=None, source_root=None) -> None:
    global REPO, CANDIDATE, RESULTS, FIG_DIR, PAPER_SEC, SRC, PHASE_SRC
    if source_root is not None:
        REPO = Path(source_root)
    CANDIDATE = (
        Path(output_dir) if output_dir is not None else REPO / "experiments/results/review27_v3"
    )
    RESULTS, FIG_DIR, PAPER_SEC = CANDIDATE, CANDIDATE / "figures", CANDIDATE / "sections"
    SRC = CANDIDATE / "scaling_law_iter45_compute_proxy_v2.tsv"
    PHASE_SRC = REPO / "experiments/results/scaling_law_iter33_phase_score.tsv"
    for directory in (RESULTS, FIG_DIR, PAPER_SEC):
        directory.mkdir(parents=True, exist_ok=True)
    rows = load_iter45()
    phase_of = load_phase()

    # Filter to good fits with R_max > 0
    recs = [r for r in rows if r.get("fit_ok") in (True, "True", "true", 1) and r["R_max"] > 0]
    if len(recs) < 4:
        raise ValueError(
            "at least four independent observed-trace fits required for leave-one-out analysis"
        )
    log_p = np.array([r["log10_P"] for r in recs])
    log_c = np.array([r["log10_C"] for r in recs])
    rmax = np.array([r["R_max"] for r in recs])

    # ---- (1) Two-parameter OLS fit (full sample) ----
    a_full, b_full, c_full, r2_full, _, resid_full = ols_3var(log_p, log_c, rmax)

    fit_rows = []
    fit_rows.append(
        {
            "stack": "all",
            "n_used": int(len(recs)),
            "semi_log_reward_slope_logP": round(a_full, 4),
            "semi_log_reward_slope_logC": round(b_full, 4),
            "intercept": round(c_full, 4),
            "r_squared": round(r2_full, 4),
            "rmse": round(float(np.sqrt(np.mean(resid_full**2))), 4),
            "median_Rmax": round(float(np.median(rmax)), 4),
            "note": "OLS R_max = a*log10(P) + b*log10(C) + c on full sample",
        }
    )
    # Stack-conditional fits
    for arch in ("dense", "moe"):
        ix = np.array([i for i, r in enumerate(recs) if r["arch"] == arch])
        if len(ix) < 3:
            continue
        if np.linalg.matrix_rank(np.column_stack([log_p[ix], log_c[ix], np.ones(len(ix))])) < 3:
            continue
        a_x, b_x, c_x, r2_x, _, res_x = ols_3var(log_p[ix], log_c[ix], rmax[ix])
        fit_rows.append(
            {
                "stack": arch,
                "n_used": int(len(ix)),
                "semi_log_reward_slope_logP": round(a_x, 4),
                "semi_log_reward_slope_logC": round(b_x, 4),
                "intercept": round(c_x, 4),
                "r_squared": round(r2_x, 4),
                "rmse": round(float(np.sqrt(np.mean(res_x**2))), 4),
                "median_Rmax": round(float(np.median(rmax[ix])), 4),
                "note": "OLS R_max = a*log10(P) + b*log10(C) + c on stack-conditional sample",
            }
        )

    fields1 = list(fit_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_two_param_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields1, delimiter="\t")
        w.writeheader()
        w.writerows(fit_rows)

    # ---- (2) Leave-one-out predictive residuals ----
    n = len(recs)
    loo_rows = []
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        a_i, b_i, c_i, _, _, _ = ols_3var(log_p[mask], log_c[mask], rmax[mask])
        yhat_i = a_i * log_p[i] + b_i * log_c[i] + c_i
        loo_rows.append(
            {
                "model": recs[i]["model"],
                "params_B": recs[i]["params_B"],
                "arch": recs[i]["arch"],
                "phase": phase_of.get(recs[i]["model"], "unknown"),
                "R_max_actual": round(float(rmax[i]), 4),
                "R_max_predicted": round(float(yhat_i), 4),
                "residual": round(float(rmax[i] - yhat_i), 4),
                "abs_residual": round(float(abs(rmax[i] - yhat_i)), 4),
                "rel_residual": round(float((rmax[i] - yhat_i) / max(1e-9, yhat_i)), 4),
                "log10_P": round(float(log_p[i]), 4),
                "log10_C": round(float(log_c[i]), 4),
            }
        )

    fields2 = list(loo_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_loo_residuals_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields2, delimiter="\t")
        w.writeheader()
        w.writerows(loo_rows)

    # The downstream iter53 candidate reader uses the unversioned name.
    with open(RESULTS / "scaling_law_iter49_loo_residuals.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields2, delimiter="\t")
        writer.writeheader()
        writer.writerows(loo_rows)

    # ---- (3) Aggregated residual by phase ----
    by_phase = {}
    for r in loo_rows:
        by_phase.setdefault(r["phase"], []).append(r["abs_residual"])
    phase_rows = []
    for ph, arr in sorted(by_phase.items()):
        arr = np.array(arr, float)
        phase_rows.append(
            {
                "phase": ph,
                "n_anchors": int(len(arr)),
                "mean_abs_residual": round(float(np.mean(arr)), 4),
                "median_abs_residual": round(float(np.median(arr)), 4),
                "max_abs_residual": round(float(np.max(arr)), 4),
                "empirical_p90": round(float(np.percentile(arr, 90)), 4),
                "note": "LOO |residual| from two-param joint fit",
            }
        )
    fields3 = list(phase_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_phase_residual_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields3, delimiter="\t")
        w.writeheader()
        w.writerows(phase_rows)

    # ---- (4) Descriptive score surface; no allocation winner is inferred ----
    grid_log_c = np.linspace(log_c.min(), log_c.max(), 50)
    score_rows = []
    for lc in grid_log_c:
        values = budget_scores(log_p, lc, a_full, b_full, c_full)
        for record, value in zip(recs, values, strict=False):
            score_rows.append(
                {
                    "log10_C_proxy_target": float(lc),
                    "model": record["model"],
                    "params_B": record["params_B"],
                    "descriptive_R_max_score": float(value),
                    "allocation_status": "not_identified",
                    "note": "additive budget term leaves candidate ordering unchanged; feasibility untested",
                }
            )
    with open(RESULTS / "scaling_law_iter49_budget_scores_v2.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(score_rows)

    # ---- (5) Exploratory descriptive criteria ----
    loo_abs = np.array([r["abs_residual"] for r in loo_rows])
    rmse_loo = float(np.sqrt(np.mean(loo_abs**2)))
    # P1: LOO RMSE < 0.30 R_max units.
    p1_pass = bool(rmse_loo < 0.30)
    # P2: named-anchor residual ordering, without inferring trajectory phase.
    collapse_row = next((r for r in loo_rows if r["model"] == "Nemotron-120B"), None)
    if collapse_row is not None:
        max_idx = int(np.argmax(loo_abs))
        max_model = loo_rows[max_idx]["model"]
        p2_pass = bool(max_model == "Nemotron-120B")
        collapse_abs = float(collapse_row["abs_residual"])
    else:
        p2_pass = None
        max_model = loo_rows[int(np.argmax(loo_abs))]["model"]
        collapse_abs = float("nan")
    pred_rows = [
        {
            "prediction": "P1_two_param_LOO_RMSE_lt_0p30",
            "value": round(rmse_loo, 4),
            "pass": p1_pass,
            "note": "LOO RMSE = sqrt(mean((R_max - pred)**2))",
        },
        {
            "prediction": "P2_nemotron_largest_abs_residual",
            "value": round(collapse_abs, 4) if not math.isnan(collapse_abs) else float("nan"),
            "pass": p2_pass,
            "note": f"max |resid| is on {max_model}",
        },
        {
            "prediction": "P3_allocation_not_identified",
            "value": float("nan"),
            "pass": None,
            "note": "UNAVAILABLE: the additive compute term cannot identify a budget-dependent allocation",
        },
    ]
    for result in pred_rows:
        result["analysis_kind"] = "exploratory_descriptive"
        result["status"] = (
            "unavailable"
            if result["pass"] is None
            else "criterion_met"
            if result["pass"]
            else "criterion_not_met"
        )
    fields5 = list(pred_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_predictions_v2.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields5, delimiter="\t")
        w.writeheader()
        w.writerows(pred_rows)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    ax = axes[0]
    colors = {
        "plateau": "#3a86ff",
        "saturation": "#06d6a0",
        "drift": "#ffbe0b",
        "collapse": "#ef476f",
        "converged": "#3a86ff",
        "unknown": "#999999",
    }
    for r in loo_rows:
        ax.scatter(
            r["R_max_predicted"],
            r["R_max_actual"],
            c=colors.get(r["phase"], "#999"),
            alpha=0.85,
            s=85,
            edgecolor="black",
            linewidth=0.4,
        )
        if r["model"] == "Nemotron-120B":
            ax.annotate(
                r["model"],
                (r["R_max_predicted"], r["R_max_actual"]),
                xytext=(6, -14),
                textcoords="offset points",
                fontsize=8,
                color="#a00",
                weight="bold",
            )
    lo_line = np.linspace(0, max(rmax) * 1.05, 100)
    ax.plot(lo_line, lo_line, "--", color="black", alpha=0.5, label="y=x")
    ax.set_xlabel("LOO predicted R_max (two-param OLS)")
    ax.set_ylabel("R_max fitted to measured traces")
    ax.set_title(f"Full-sample R^2={r2_full:.2f}; LOO RMSE={rmse_loo:.2f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, record in enumerate(recs):
        values = [budget_scores(log_p, lc, a_full, b_full, c_full)[i] for lc in grid_log_c]
        ax.plot(grid_log_c, values, label=record["model"], alpha=0.75)
    ax.set_xlabel("log10(C_proxy) target")
    ax.set_ylabel("Descriptive fitted-reward score")
    ax.set_title("Additive scores: ordering is budget-invariant")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "scaling_law_iter49_v2.png", dpi=160)
    plt.savefig(FIG_DIR / "scaling_law_iter49_v2.pdf")
    plt.close()

    # ---- Data-derived interpretation ----
    report = (
        r"\subsection{Descriptive two-predictor associations (iter49)}" + "\n"
        r"The fitted training-reward equation is $\widehat R_{\max}=a\log_{10}P_B+b\log_{10}C+c$. "
        f"The semi-log reward slopes are a={a_full:.4f}, b={b_full:.4f}; "
        f"full-sample R-squared={r2_full:.4f}, leave-one-anchor-out RMSE={rmse_loo:.4f}. "
        r"The targets are saturation parameters fitted to identified observed training traces, "
        r"not directly measured asymptotic or held-out accuracies. At any fixed proxy budget, "
        r"the compute term is the same for every candidate; budget changes cannot change their ordering. "
        r"Allocation is not identified. Scores do not establish feasibility or a resource optimum. "
        r"The criteria are exploratory; no prospective protocol is supplied."
    )
    (PAPER_SEC / "scaling_law_iter49_v2.tex").write_text(report + "\n")
    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(
        __file__, RESULTS, [SRC, PHASE_SRC] + [REPO / r["trace_file"] for r in recs], seed=None
    )

    # ---- summary ----
    n_pass = sum(1 for r in pred_rows if r["pass"])
    print(
        f"iter49 done: a_logP={a_full:.3f}, b_logC={b_full:.3f}, c={c_full:.3f}, "
        f"R^2={r2_full:.3f}, LOO RMSE={rmse_loo:.3f}, descriptive criteria met={n_pass}; allocation unavailable."
    )


if __name__ == "__main__":
    main()
