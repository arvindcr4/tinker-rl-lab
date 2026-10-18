#!/usr/bin/env python3
"""review24-v2: model-dependent ceiling-ratio sensitivity, not paired inference.

Fits Michaelis-Menten curves to summary accuracy observations. Normal draws
assume independence across budgets and groups because joint seed observations
are unavailable. Percentiles quantify that assumption's sensitivity only.
No result establishes a universal asymptote or refutes another task's finding.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from _paths import RESULTS_DIR


def sat(T, a, c):
    return a * np.asarray(T) / (np.asarray(T) + c)


def fit_ceiling(G, draws=None, *, rows, rng=None):
    """Fixed result schema; failures remain explicit, never dropped observations."""
    result = {
        "status": "failed",
        "reason": "",
        "a": float("nan"),
        "c": float("nan"),
        "r2": float("nan"),
    }
    points = sorted([r for r in rows if r["G"] == G], key=lambda r: r["T"])
    if len(points) < 3:
        return {**result, "reason": "need at least three distinct observed budgets"}
    T = np.asarray([r["T"] for r in points], float)
    y = np.asarray([r["acc"] for r in points], float)
    lo = np.asarray([r["lo"] for r in points], float)
    hi = np.asarray([r["hi"] for r in points], float)
    if (
        len(set(T)) != len(T)
        or np.any(T <= 0)
        or not all(np.isfinite(v).all() for v in [T, y, lo, hi])
        or np.any(lo > hi)
    ):
        return {**result, "reason": "invalid/duplicate observations"}
    if draws is not None:
        if rng is None:
            raise ValueError("explicit random stream required")
        y = rng.normal(y, (hi - lo) / (2 * 1.96))
    scale = float(T.max())
    x = T / scale
    try:
        params, _ = curve_fit(
            sat,
            x,
            y,
            p0=[min(0.99, max(0.05, float(y.max()))), 0.25],
            bounds=([0.0, 1e-9], [1.0, 1e6]),
            maxfev=20000,
        )
        if not np.isfinite(params).all():
            raise ValueError("nonfinite fitted parameters")
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {**result, "reason": str(exc)}
    predicted = sat(x, *params)
    rss = float(np.sum((y - predicted) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    return {
        "status": "ok",
        "reason": "",
        "a": float(params[0]),
        "c": float(params[1] * scale),
        "r2": 1 - rss / tss if tss > 0 else float("nan"),
    }


def analyse(rows, B=4000, seed=95, min_valid_fraction=0.95):
    if B < 2:
        raise ValueError("need at least two sensitivity draws")
    groups = sorted({r["G"] for r in rows})
    rng = np.random.default_rng(seed)
    results = {}
    draws = {}
    for G in groups:
        base = fit_ceiling(G, rows=rows)
        if base["status"] != "ok":
            raise ValueError(f"G={G} point fit failed: {base['reason']}")
        fits = [fit_ceiling(G, draws=i, rows=rows, rng=rng) for i in range(B)]
        values = np.asarray([r["a"] for r in fits])
        valid = np.isfinite(values)
        failures = {}
        for fit in fits:
            if fit["status"] != "ok":
                failures[fit["reason"]] = failures.get(fit["reason"], 0) + 1
        if valid.mean() < min_valid_fraction:
            raise ValueError(
                f"G={G}: {sum(valid)}/{B} successful fits below {min_valid_fraction}; {failures}"
            )
        draws[G] = values
        results[G] = {
            **base,
            "G": G,
            "attempted": B,
            "successful": int(valid.sum()),
            "failure_reasons": failures,
            "sensitivity_low": float(np.quantile(values[valid], 0.025)),
            "sensitivity_high": float(np.quantile(values[valid], 0.975)),
        }
    if 4 not in draws or 32 not in draws:
        raise ValueError("requires both G4 and G32")
    valid = np.isfinite(draws[4]) & np.isfinite(draws[32]) & (draws[32] > 0)
    if valid.mean() < min_valid_fraction:
        raise ValueError("insufficient jointly valid independent sensitivity draws")
    ratio = draws[4][valid] / draws[32][valid]
    return {
        "analysis_version": "review24-v2",
        "sampling": "independent_normal_summary_sensitivity",
        "seed": seed,
        "model": "a*T/(T+c), 0<=a<=1",
        "ceilings": list(results.values()),
        "ratio": {
            "estimate": results[4]["a"] / results[32]["a"],
            "low": float(np.quantile(ratio, 0.025)),
            "high": float(np.quantile(ratio, 0.975)),
            "attempted": B,
            "successful": int(valid.sum()),
            "interpretation": "model_dependent_extrapolation",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=RESULTS_DIR / "group_size_token_normalized.tsv"
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "corrected-review24-v2")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=95)
    args = parser.parse_args()
    with args.input.open() as stream:
        rows = [
            {
                "G": int(r["G"]),
                "T": float(r["budget_tokens"]),
                "acc": float(r["heldout_acc_mean"]),
                "lo": float(r["heldout_acc_ci_low"]),
                "hi": float(r["heldout_acc_ci_high"]),
            }
            for r in csv.DictReader(stream, delimiter="\t")
        ]
    result = analyse(rows, args.draws, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "group_size_iter95_sensitivity.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print(
        f"Wrote independent-summary sensitivity to {destination}; no paired or universal inference"
    )


if __name__ == "__main__":
    main()
