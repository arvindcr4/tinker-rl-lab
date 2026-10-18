"""Exploratory optimizer-bound diagnostics and observed reward summaries.

An optimizer-bound flag is a numerical diagnostic, not an identifiability or
phase classification. The summary max(trace q75, second-half mean) describes
observed training rewards; it is not an asymptotic or held-out accuracy estimate.
Cross-anchor log-log fits use base 10 on both axes and strictly positive
summaries, with no additive reward offset. Criteria are retrospective summaries.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from ._analysis_common import saturation, write_rows_tsv as _write_tsv
except ImportError:  # direct script invocation
    from _analysis_common import saturation, write_rows_tsv as _write_tsv

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"
PAPER_FIG = CANDIDATE / "paper_figures"

# Pool A: 5 frontier anchors (the canonical Pillar-1 set)
POOL_A: dict[str, tuple[str, float]] = {
    "Qwen3.5-4B": ("scale_gsm8k_qwen3.5-4b.json", 4.0),
    "Qwen3-8B": ("scale_gsm8k_qwen3-8b.json", 8.0),
    "Llama-3.1-8B-Instruct": ("scale_gsm8k_llama-8b-inst.json", 8.0),
    "DeepSeek-V3.1": ("frontier_gsm8k_deepseek-v3.1.json", 685.0),
    "Nemotron-120B": ("frontier_gsm8k_nemotron-120b.json", 120.0),
}

# Pool B: 4 partial / extended-scale anchors (added in iter57)
POOL_B: dict[str, tuple[str, float]] = {
    "Qwen3-32B": ("scale_gsm8k_qwen3-32b.json", 32.0),
    "Qwen3.5-27B": ("scale_gsm8k_qwen3.5-27b.json", 27.0),
    "Qwen3-30B-MoE": ("moe_gsm8k_qwen3-30b-moe.json", 30.0),
    "Qwen3-30B-MoE-Inst": ("moe_gsm8k_qwen3-30b-inst.json", 30.0),
}

SEED = 20260702
N_BOOT = 5000


def _ols(x, y):
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
    s2 = float(np.sum(resid**2)) / (n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    return a, b, se_b


def fit_canonical(y: np.ndarray) -> dict:
    """Per-trace canonical saturation R(t) = R_max (1 - e^{-lam t}); may pin
    to the lambda upper bound when the trace is essentially flat."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("at least three finite observations required")
    t = np.arange(1, len(y) + 1, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    try:
        popt, _ = curve_fit(
            saturation,
            t,
            y,
            p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
    except Exception as exc:
        raise ValueError("canonical saturation fit failed") from exc
    r_max, lam = float(popt[0]), float(popt[1])
    yhat = saturation(t, r_max, lam)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {
        "R_max": r_max,
        "lam": lam,
        "rmse": rmse,
        "r2": r2,
        "lam_at_bound": lam >= 9.999,
        "t_80": -math.log(0.2) / lam if lam > 1e-4 else float("nan"),
    }


def reward_summary(y: np.ndarray) -> dict:
    """Observed max(q75, second-half mean), with no asymptotic interpretation."""
    y = np.asarray(y, float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("a nonempty finite observed reward vector is required")
    tail = y[len(y) // 2 :]
    q75, late_mean = float(np.percentile(y, 75)), float(tail.mean())
    threshold = late_mean - float(np.std(tail))
    above = np.flatnonzero(y >= threshold)
    return {
        "reward_summary": max(q75, late_mean),
        "q75": q75,
        "late_mean": late_mean,
        "rise_step": int(above[0]) + 1 if len(above) else -1,
    }


def robust_rmax(y: np.ndarray) -> dict:
    """Compatibility alias; r_max_robust is an observed summary, not an asymptote."""
    result = reward_summary(y)
    return {**result, "r_max_robust": result["reward_summary"]}


def load_pool(pool: dict[str, tuple[str, float]]) -> dict[str, dict]:
    out = {}
    for label, (fname, params_b) in pool.items():
        path = TRACE_DIR / fname
        if not path.exists():
            print(f"[unavailable] {label}: trace file missing")
            continue
        raw = path.read_bytes()
        record = json.loads(raw)
        if any(
            str(record.get(key, "")).lower() in {"true", "yes", "1"}
            for key in ("synthetic", "is_synthetic", "reconstructed")
        ):
            raise ValueError(f"{label}: synthetic trace is not observed evidence")
        if any(
            term in str(record.get("evidence_kind", "")).lower()
            for term in ("synthetic", "reconstruct")
        ):
            raise ValueError(f"{label}: synthetic trace is not observed evidence")
        trace = np.asarray(record.get("reward_trace"), float)
        reward_summary(trace)  # validates finite, nonempty, one-dimensional data
        if not math.isfinite(params_b) or params_b <= 0:
            raise ValueError(f"{label}: positive finite parameter count required")
        out[label] = {
            "trace": trace,
            "params_b": params_b,
            "n_steps": len(trace),
            "meta": record,
            "mean_reward": float(trace.mean()),
            "var_reward": float(trace.var()),
            "peak": float(trace.max()),
            "trough": float(trace.min()),
            "trace_file": fname,
            "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "run_id": record.get("run_id", ""),
        }
    return out


def bootstrap_alpha(log_n: np.ndarray, log_rmax: np.ndarray, n_boot=None) -> dict:
    """IID anchor bootstrap for the descriptive log10/log10 slope, not a forecast."""
    x, y = np.asarray(log_n, float), np.asarray(log_rmax, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("aligned finite log-coordinate vectors required")
    n_boot = N_BOOT if n_boot is None else n_boot
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("positive integer bootstrap count required")
    result = {
        "alpha": math.nan,
        "log_c": math.nan,
        "alpha_lo": math.nan,
        "alpha_hi": math.nan,
        "r2": math.nan,
        "n": len(x),
        "n_boot_attempted": n_boot,
        "n_boot_success": 0,
        "status": "unavailable",
    }
    if len(x) < 3 or np.ptp(x) == 0:
        return result
    a, b, _ = _ols(x, y)
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rng = np.random.default_rng(SEED)
    slopes = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(x), len(x))
        if np.ptp(x[ix]) == 0:
            continue
        slope = _ols(x[ix], y[ix])[1]
        if math.isfinite(slope):
            slopes.append(slope)
    result.update(
        alpha=b,
        log_c=a,
        r2=1 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        n_boot_success=len(slopes),
        status="ok" if slopes else "ci_unavailable",
    )
    if slopes:
        result.update(
            alpha_lo=float(np.percentile(slopes, 2.5)), alpha_hi=float(np.percentile(slopes, 97.5))
        )
    return result


def log_summary_fit(records):
    """Fit only strictly positive summaries, retaining the exclusion identities."""
    included = [r for r in records if r["reward_summary"] > 0]
    excluded = [r["model"] for r in records if r["reward_summary"] <= 0]
    x = np.log10([r["params_B"] for r in included])
    y = np.log10([r["reward_summary"] for r in included])
    fit = bootstrap_alpha(x, y)
    return {**fit, "included": included, "excluded_nonpositive": excluded, "x": x, "y": y}


def descriptive_criteria(records, fit_a, fit_ab=None):
    """Use the same explicit predicates for tabular and narrative output."""
    fitted = [r for r in records if r["fit_status"] == "ok"]
    unbound = [r["model"] for r in fitted if not r["lam_at_bound"]]
    gaps = [r["gap_canonical_minus_summary"] for r in fitted]
    n_ge = sum(gap <= 0 for gap in gaps)
    nemotron_present = any(r["model"] == "Nemotron-120B" for r in fitted)
    ci_finite = math.isfinite(fit_a["alpha_lo"]) and math.isfinite(fit_a["alpha_hi"])
    rows = []

    def add(identifier, claim, value, outcome, note):
        rows.append(
            {
                "prediction_id": identifier,
                "claim": claim,
                "observed_value": value,
                "pass_fail": "UNAVAILABLE" if outcome is None else "PASS" if outcome else "FAIL",
                "analysis_kind": "exploratory_descriptive",
                "notes": note,
            }
        )

    add(
        "P1_nemotron_only_unbound",
        "Nemotron-120B is the only non-bound observed anchor",
        "|".join(unbound),
        unbound == ["Nemotron-120B"] if nemotron_present and len(fitted) == len(records) else None,
        "Optimizer-bound status does not classify phase or establish parameter identifiability",
    )
    add(
        "P2_summary_ge_canonical",
        "Observed reward summary >= canonical fitted R_max on at least 3 anchors",
        n_ge,
        n_ge >= 3 if len(fitted) >= 3 else None,
        f"canonical-minus-summary gaps: {gaps}; a larger summary is not evidence of estimator superiority",
    )
    add(
        "P3_alpha_CI_includes_zero",
        "Anchor-bootstrap slope interval includes zero",
        f"[{fit_a['alpha_lo']:.4f}, {fit_a['alpha_hi']:.4f}]",
        fit_a["alpha_lo"] <= 0 <= fit_a["alpha_hi"] if ci_finite else None,
        "Inclusion of zero does not establish equivalence or scale invariance",
    )
    if fit_ab is not None:
        width_a = fit_a["alpha_hi"] - fit_a["alpha_lo"]
        width_ab = fit_ab["alpha_hi"] - fit_ab["alpha_lo"]
        available = math.isfinite(width_a) and math.isfinite(width_ab)
        add(
            "P4_extended_pool_narrows_CI",
            "Extended-pool slope interval is narrower",
            width_ab,
            width_ab < width_a if available else None,
            f"base width={width_a:.4f}, extended width={width_ab:.4f}; heterogeneous anchors are exploratory",
        )
    return rows


def _write_records(path, records):
    if not records:
        raise ValueError("cannot write an unspecified empty result schema")
    columns = list(records[0])
    _write_tsv(path, columns, [[row[column] for column in columns] for row in records])


def _tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_section(records, fits, criteria, path):
    """All numerical narrative comes from named result records."""
    fitted = [r for r in records if r["fit_status"] == "ok"]
    n_bound = sum(r["lam_at_bound"] for r in fitted)
    lines = [
        r"\subsection{Optimizer-bound diagnostics and observed reward summaries (iter57)}",
        r"\label{sec:scaling-law-iter57}",
        f"{n_bound}/{len(fitted)} successful canonical fits touch the lambda upper bound; "
        f"{len(records) - len(fitted)} fits are unavailable. "
        r"This is an optimizer diagnostic only. Neither bound status establishes parameter identifiability "
        r"or a trajectory phase. Fit adequacy and parameter uncertainty require separate assessment.",
        r"\paragraph{Observed reward summary.} We report $S=\max(q_{0.75},\bar R_{\mathrm{second-half}})$. "
        r"The second-half statistic is an ordinary mean, not a trimmed mean. This finite-trace training-reward "
        r"summary has no asymptotic or held-out accuracy interpretation and does not replace the fitted parameter.",
    ]
    if fitted:
        largest = max(fitted, key=lambda r: abs(r["gap_canonical_minus_summary"]))
        gap_mean = float(np.mean([r["gap_canonical_minus_summary"] for r in fitted]))
        n_ge = sum(r["gap_canonical_minus_summary"] <= 0 for r in fitted)
        lines.append(
            f"Canonical-minus-summary gap averages {gap_mean:+.4f}; the largest absolute gap is "
            f"{_tex_escape(largest['model'])} ({largest['gap_canonical_minus_summary']:+.4f}). "
            f"The summary is at least the canonical fit on {n_ge}/{len(fitted)} observed anchors."
        )
    for name, fit in fits.items():
        lines.append(
            f"{_tex_escape(name)} uses {fit['n']} positive summaries: log10/log10 slope "
            f"{fit['alpha']:.4f}, interval [{fit['alpha_lo']:.4f}, {fit['alpha_hi']:.4f}], "
            f"R-squared {fit['r2']:.4f}, successful bootstrap draws {fit['n_boot_success']}/{fit['n_boot_attempted']}. "
            f"Excluded nonpositive summaries: {_tex_escape(', '.join(fit['excluded_nonpositive']) or 'none')}."
        )
    lines.append(
        r"The log-log association uses base 10 on both axes, without a reward offset. "
        r"Anchor resampling is descriptive and assumes independent anchors; it does not quantify trace-estimation "
        r"uncertainty. An interval containing zero does not establish equivalence or scale invariance. "
        r"No prospective protocol is supplied for the exploratory criteria below."
    )
    lines.extend(
        _tex_escape(
            f"{c['prediction_id']}: {c['pass_fail']}; {c['claim']}; observed={c['observed_value']}"
        )
        for c in criteria
    )
    path.write_text("\n\n".join(lines) + "\n")


def main(output_dir=None, source_root=None) -> None:
    global REPO, CANDIDATE, TRACE_DIR, RESULTS_DIR, FIG_DIR, PAPER_SEC, PAPER_FIG
    if source_root is not None:
        REPO = Path(source_root)
    CANDIDATE = (
        Path(output_dir) if output_dir is not None else REPO / "experiments/results/review27_v3"
    )
    TRACE_DIR = REPO / "experiments/tinker-runs/results"
    RESULTS_DIR, FIG_DIR = CANDIDATE, CANDIDATE / "figures"
    PAPER_SEC, PAPER_FIG = CANDIDATE / "sections", CANDIDATE / "paper_figures"
    pool_a, pool_b = load_pool(POOL_A), load_pool(POOL_B)
    if not pool_a:
        raise ValueError("no observed base-pool reward traces available")
    for directory in (RESULTS_DIR, FIG_DIR, PAPER_SEC, PAPER_FIG):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for label, info in {**pool_a, **pool_b}.items():
        result = {
            "model": label,
            "params_B": info["params_b"],
            "n_steps": info["n_steps"],
            "pool": "A" if label in pool_a else "B",
            "mean_reward": info["mean_reward"],
            "var_reward": info["var_reward"],
            "peak": info["peak"],
            "trough": info["trough"],
            **reward_summary(info["trace"]),
            "fit_status": "unavailable",
            "fit_reason": "",
            "R_max_canonical": math.nan,
            "lambda_canonical": math.nan,
            "t_80_canonical": math.nan,
            "rmse_canonical": math.nan,
            "r2_canonical": math.nan,
            "lam_at_bound": None,
            "optimizer_status": "unavailable",
            "gap_canonical_minus_summary": math.nan,
            "partial_flag": info["meta"].get("partial", False),
            "trace_file": info["trace_file"],
            "trace_sha256": info["trace_sha256"],
            "run_id": info["run_id"],
        }
        try:
            cf = fit_canonical(info["trace"])
        except ValueError as exc:
            result["fit_reason"] = str(exc)
        else:
            result.update(
                fit_status="ok",
                R_max_canonical=cf["R_max"],
                lambda_canonical=cf["lam"],
                t_80_canonical=cf["t_80"],
                rmse_canonical=cf["rmse"],
                r2_canonical=cf["r2"],
                lam_at_bound=cf["lam_at_bound"],
                optimizer_status="at_upper_bound"
                if cf["lam_at_bound"]
                else "interior_to_upper_bound",
                gap_canonical_minus_summary=cf["R_max"] - result["reward_summary"],
            )
        records.append(result)
    records_a = [r for r in records if r["pool"] == "A"]
    diagnostic_keys = [
        "model",
        "params_B",
        "n_steps",
        "fit_status",
        "fit_reason",
        "R_max_canonical",
        "lambda_canonical",
        "t_80_canonical",
        "rmse_canonical",
        "r2_canonical",
        "lam_at_bound",
        "optimizer_status",
        "trace_file",
        "trace_sha256",
        "run_id",
    ]
    _write_records(
        RESULTS_DIR / "scaling_law_iter57_lam_bound.tsv",
        [{key: r[key] for key in diagnostic_keys} for r in records_a],
    )
    summary_keys = [
        "model",
        "params_B",
        "n_steps",
        "fit_status",
        "R_max_canonical",
        "lam_at_bound",
        "reward_summary",
        "q75",
        "late_mean",
        "rise_step",
        "gap_canonical_minus_summary",
        "trace_file",
        "trace_sha256",
        "run_id",
    ]
    _write_records(
        RESULTS_DIR / "scaling_law_iter57_robust_rmax.tsv",
        [{key: r[key] for key in summary_keys} for r in records_a],
    )
    _write_records(RESULTS_DIR / "scaling_law_iter57_extended_pool.tsv", records)
    fits = {"pool_A": log_summary_fit(records_a)}
    if pool_b:
        fits["pool_AB"] = log_summary_fit(records)
    fit_rows = []
    for name, fit in fits.items():
        fit_rows.append(
            {
                "pool": name,
                **{
                    key: fit[key]
                    for key in (
                        "n",
                        "alpha",
                        "alpha_lo",
                        "alpha_hi",
                        "log_c",
                        "r2",
                        "n_boot_attempted",
                        "n_boot_success",
                        "status",
                    )
                },
                "anchors": "|".join(r["model"] for r in fit["included"]),
                "excluded_nonpositive": "|".join(fit["excluded_nonpositive"]),
                "estimand": "log10 observed reward summary vs log10 parameters",
            }
        )
    _write_records(RESULTS_DIR / "scaling_law_iter57_powerlaw.tsv", fit_rows)
    criteria = descriptive_criteria(records_a, fits["pool_A"], fits.get("pool_AB"))
    _write_records(RESULTS_DIR / "scaling_law_iter57_predictions.tsv", criteria)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for record in records_a:
        info = pool_a[record["model"]]
        axes[0, 0].plot(
            np.arange(1, len(info["trace"]) + 1), info["trace"], ".-", label=record["model"]
        )
    axes[0, 0].set(
        xlabel="observed step",
        ylabel="training reward",
        title=f"Base pool: {len(records_a)} observed traces",
    )
    axes[0, 0].legend(fontsize=6)
    indices = np.arange(len(records_a))
    axes[0, 1].bar(
        indices - 0.18,
        [r["R_max_canonical"] for r in records_a],
        0.36,
        label="canonical fitted R_max",
    )
    axes[0, 1].bar(
        indices + 0.18,
        [r["reward_summary"] for r in records_a],
        0.36,
        label="observed reward summary",
    )
    axes[0, 1].set_xticks(indices, [r["model"] for r in records_a], rotation=20, fontsize=6)
    axes[0, 1].set(
        ylabel="training reward / fitted parameter", title="Canonical fit and finite-trace summary"
    )
    axes[0, 1].legend(fontsize=7)
    for name, fit in fits.items():
        axes[1, 0].scatter(fit["x"], fit["y"], label=f"{name} ({fit['n']})", alpha=0.65)
        if math.isfinite(fit["alpha"]):
            xs = np.linspace(min(fit["x"]), max(fit["x"]), 100)
            axes[1, 0].plot(xs, fit["log_c"] + fit["alpha"] * xs, "--")
    axes[1, 0].set(
        xlabel="log10(parameters [B])",
        ylabel="log10(observed reward summary)",
        title="Descriptive log-log association",
    )
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].axis("off")
    text = "Exploratory criteria\n" + "\n".join(
        f"{c['prediction_id']}: {c['pass_fail']}" for c in criteria
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", fontsize=8)
    fig.tight_layout()
    for directory in (FIG_DIR, PAPER_FIG):
        fig.savefig(directory / "scaling_law_iter57.pdf")
        fig.savefig(directory / "scaling_law_iter57.png", dpi=150)
    plt.close(fig)
    write_section(records_a, fits, criteria, PAPER_SEC / "scaling_law_iter57.tex")
    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(
        __file__,
        RESULTS_DIR,
        [TRACE_DIR / filename for filename, _ in {**POOL_A, **POOL_B}.values()],
        seed=SEED,
    )
    print(
        f"iter57: {len(records_a)} base traces; {sum(r['fit_status'] == 'ok' for r in records_a)} successful fits; exploratory output only"
    )


if __name__ == "__main__":
    main()
