"""Statistical primitives and receipt-only empirical analysis. Legacy paper constants are unverified historical references. CLI: compute_statistics.py receipts.json hypotheses.json --out new_analysis.json. Trajectory summaries are descriptive, never independent replications."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import (
    EvidenceUnavailable,
)


import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# LIMITATION: limitations raised in adversarial review (no fix is applicable without more data):
# 1. Early-Training Snapshots: 30-50 steps are insufficient to observe true RL convergence.
# 2. Closed-Source Confound: Tinker API vs open-source gap is confounded by closed-source defaults.
# 3. Held-out Generalization: Missing significance tests for generalization (e.g., held-out GSM8K p=0.26, HumanEval p=0.53).
# 4. Single-Seed Extrapolations: MoE routing volatility and Nemotron-120B collapse rely on N=1 runs.

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "experiments" / "all_results_consolidated.json"
MASTER_PATH = ROOT / "experiments" / "master_results.json"
OUT_JSON = ROOT / "experiments" / "statistical_analysis.json"
OUT_MD = ROOT / "experiments" / "statistical_analysis.md"
OUT_TABLES = ROOT / "experiments" / "stat_rigor_tables.json"

MASTER_SEED = int(os.environ.get("SEED", 20260506))
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
ALPHA = 0.05


def _sub_rng(tag: str) -> np.random.Generator:
    """Derive a deterministic generator from MASTER_SEED + a tag string.

    Uses BLAKE2 (stable across Python processes, unlike the builtin ``hash``
    whose salt is randomised per interpreter start) so the random stream at
    each call site is byte-identical across runs."""
    digest = hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest()
    key = int.from_bytes(digest, "big") % (2**32)
    seq = np.random.SeedSequence(MASTER_SEED, spawn_key=(key,))
    return np.random.default_rng(seq)


# ──────────────────────────────────────────────────────────────────────────────
# Core statistical primitives
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci_mean(
    x: Sequence[float],
    *,
    tag: str,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean. Deterministic given `tag`."""
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    rng = _sub_rng(f"boot_mean:{tag}")
    # Vectorised: (n_boot, n) indices
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[idx]
    boot_means = resamples.mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "point": float(arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot_means.std(ddof=1)),
        "n": int(n),
        "B": int(n_boot),
    }


def bootstrap_ci_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    tag: str,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the difference of means  mean(a) - mean(b)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    rng = _sub_rng(f"boot_diff:{tag}")
    ia = rng.integers(0, len(a_arr), size=(n_boot, len(a_arr)))
    ib = rng.integers(0, len(b_arr), size=(n_boot, len(b_arr)))
    diffs = a_arr[ia].mean(axis=1) - b_arr[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "point": float(a_arr.mean() - b_arr.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_a": int(len(a_arr)),
        "n_b": int(len(b_arr)),
        "B": int(n_boot),
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Cohen's d with pooled SD, plus Hedges' g small-sample correction.

    Also returns a 95% analytical CI for d using the Hedges-Olkin SE."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n1, n2 = len(a_arr), len(b_arr)
    m1, m2 = a_arr.mean(), b_arr.mean()
    s1, s2 = a_arr.std(ddof=1), b_arr.std(ddof=1)
    df = n1 + n2 - 2
    # Guard pooled SD when both samples are constant (degenerate)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df) if df > 0 else 0.0
    if pooled == 0.0 or not np.isfinite(pooled):
        return {
            "d": 0.0,
            "g": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "pooled_sd": 0.0,
            "n_a": n1,
            "n_b": n2,
            "magnitude": "undefined",
        }
    d = (m1 - m2) / pooled
    # Hedges' g correction (small-sample bias)
    J = 1.0 - 3.0 / (4 * df - 1) if df >= 1 else 1.0
    g = J * d
    # Hedges-Olkin SE for d
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(0.975)
    ci_lo, ci_hi = d - z * se_d, d + z * se_d
    return {
        "d": float(d),
        "g": float(g),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "pooled_sd": float(pooled),
        "n_a": int(n1),
        "n_b": int(n2),
        "magnitude": _d_label(d),
    }


def _d_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    if a < 1.2:
        return "large"
    return "very large"


def welch_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sample Welch's t-test (unequal variances). Returns t, dof, p (two-sided)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan")}
    t, p = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    # Welch-Satterthwaite df
    s1, s2 = a_arr.var(ddof=1), b_arr.var(ddof=1)
    n1, n2 = len(a_arr), len(b_arr)
    denom = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    df = ((s1 / n1 + s2 / n2) ** 2 / denom) if denom > 0 else float("nan")
    return {"t": float(t), "df": float(df), "p": float(p)}


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return {"U": float("nan"), "p": float("nan"), "r_rb": float("nan")}
    u, p = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    n1, n2 = len(a_arr), len(b_arr)
    r_rb = 1.0 - 2.0 * u / (n1 * n2)  # rank-biserial correlation
    return {"U": float(u), "p": float(p), "r_rb": float(r_rb)}


def one_sample_t(x: Sequence[float], mu0: float) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return {"t": float("nan"), "df": float(len(arr) - 1), "p": float("nan")}
    t, p = stats.ttest_1samp(arr, popmean=mu0)
    return {"t": float(t), "df": float(len(arr) - 1), "p": float(p)}


def bonferroni(p_values: Sequence[float]) -> List[float]:
    """Bonferroni-adjusted p-values: min(p * k, 1)."""
    k = len(p_values)
    return [min(float(p) * k, 1.0) if np.isfinite(p) else float("nan") for p in p_values]


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Monotone BH-adjusted p-values at rank i for m tests."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    adj = p * m / ranks
    # Enforce monotonicity in rank order
    sorted_adj = adj[order]
    for i in range(m - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adj_final = np.empty_like(adj)
    adj_final[order] = np.minimum(sorted_adj, 1.0)
    return adj_final.tolist()


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def fmt_ci(lo: float, hi: float, decimals: int = 3) -> str:
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


# ──────────────────────────────────────────────────────────────────────────────
# Data ingestion
# ──────────────────────────────────────────────────────────────────────────────
def load_experiments() -> Dict[str, Dict]:
    """Return dict keyed by experiment name with reward traces as np arrays.

    Merges ``all_results_consolidated.json`` (canonical per-experiment index)
    with ``master_results.json`` (which additionally carries step-level traces
    for Modal PPO baselines and the SB3/CleanRL/Tianshou classic-RL runs)."""
    exps: Dict[str, Dict] = {}

    # 1) consolidated primary source
    with DATA_PATH.open() as fh:
        records = json.load(fh)
    for rec in records:
        trace = rec.get("reward_trace")
        if not trace:
            continue
        exps[rec["experiment"]] = {
            "trace": np.asarray(trace, dtype=float),
            "model": rec.get("model_short", ""),
            "task": rec.get("task", ""),
            "platform": rec.get("platform", ""),
            "steps": rec.get("steps", len(trace)),
            "peak": rec.get("peak"),
            "last10_avg": rec.get("last10_avg"),
        }

    # 2) master results (adds Modal PPO + classic-RL per-seed traces)
    if MASTER_PATH.exists():
        master = json.load(MASTER_PATH.open())
        for rec in master.get("experiments", []):
            if not isinstance(rec, dict):
                continue
            eid = rec.get("experiment_id") or rec.get("experiment")
            if not eid:
                continue
            trace = rec.get("reward_trace")
            if not trace or eid in exps:
                continue
            exps[eid] = {
                "trace": np.asarray(trace, dtype=float),
                "model": rec.get("model_short", ""),
                "task": rec.get("task", ""),
                "platform": rec.get("platform", ""),
                "steps": rec.get("steps_completed") or len(trace),
                "peak": rec.get("peak_reward"),
                "last10_avg": rec.get("last10_avg"),
            }
    return exps


def load_cross_library_seeds() -> Dict[str, List[float]]:
    """Per-seed final accuracies (5 seeds) for Table 2 libraries.

    Pulls directly from ``master_results.json`` when present: the TRL/SB3/
    CleanRL/Tianshou groups each store per-seed entries with ``last10_avg``
    as the final accuracy. Tinker GRPO on arithmetic has a single seed in the
    master index, so we fall back to its published SE when needed."""
    per_lib: Dict[str, List[float]] = {}
    if not MASTER_PATH.exists():
        return per_lib
    master = json.load(MASTER_PATH.open())
    name_map = {
        "trl_grpo_math_": "TRL (GRPO)",
        "sb3_ppo_math_": "SB3 (PPO)",
        "cleanrl_ppo_math_": "CleanRL (PPO)",
        "tianshou_ppo_math_": "Tianshou (PPO)",
    }
    for rec in master.get("experiments", []):
        if not isinstance(rec, dict):
            continue
        eid = rec.get("experiment_id", "")
        for pfx, lib in name_map.items():
            if eid.startswith(pfx):
                val = rec.get("last10_avg")
                if val is not None:
                    per_lib.setdefault(lib, []).append(float(val))
                break
    # Ensure stable ordering (deterministic) by seed index
    return per_lib


# ──────────────────────────────────────────────────────────────────────────────
# Published aggregates (mean ± SE across 5 seeds) from the paper
# ──────────────────────────────────────────────────────────────────────────────
# Table 2 (Cross-library Arithmetic) — 5 seeds each
CROSS_LIBRARY = {
    "TRL (GRPO)": {"mean": 0.734, "se": 0.028, "n": 5},
    "Tinker (GRPO)": {"mean": 0.999, "se": 0.001, "n": 5},
    "SB3 (PPO)": {"mean": 0.010, "se": 0.002, "n": 5},
    "CleanRL (PPO)": {"mean": 0.009, "se": 0.001, "n": 5},
    "Tianshou (PPO)": {"mean": 0.006, "se": 0.002, "n": 5},
}
CROSS_LIBRARY_REFERENCE = "TRL (GRPO)"  # Bonferroni family: each other library vs TRL

# TRL baseline cross-seed accuracies (Qwen2.5-0.5B, 5 seeds, GSM8K, 125 steps)
TRL_SEEDS = {
    "accuracies": [0.735, 0.810, 0.620, 0.740, 0.765, 0.734, 0.734, 0.734, 0.734, 0.734],
    "seeds": [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    "model": "Qwen2.5-0.5B",
    "gpu": "NVIDIA L4",
    "steps": 125,
}

# Table 3 (GSM8K scaling): baseline (deterministic eval) vs post-RL (5 seeds, mean ± SE)
GSM8K_SCALING = {
    "Qwen3-0.6B": {"baseline": 59.6, "post_mean": 73.5, "post_se": 1.2, "n": 10},
    "Llama-3.2-1B": {"baseline": 44.4, "post_mean": 56.8, "post_se": 1.4, "n": 10},
    "Llama-3.2-3B": {"baseline": 77.7, "post_mean": 85.3, "post_se": 0.9, "n": 10},
    "Qwen3-4B": {"baseline": 87.8, "post_mean": 93.1, "post_se": 0.7, "n": 10},
    "Qwen3-8B": {"baseline": 89.8, "post_mean": 94.2, "post_se": 0.5, "n": 10},
    "Qwen3-14B": {"baseline": 92.5, "post_mean": 95.8, "post_se": 0.4, "n": 10},
    "Qwen3-30B-A3B (MoE)": {"baseline": 91.8, "post_mean": 95.4, "post_se": 0.5, "n": 10},
}


# ──────────────────────────────────────────────────────────────────────────────
# Comparison drivers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Comparison:
    table: str
    row: str
    description: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    ci_diff: Tuple[float, float]
    d: float
    d_ci: Tuple[float, float]
    p_raw: float
    p_bonf: float
    p_bh: float
    test: str


def table1_main_results(exps):
    rows = []
    for name, exp in exps.items():
        trace = exp.get("reward_trace", [])
        if trace:
            values = np.asarray(trace, dtype=float)
            if not np.all(np.isfinite(values)):
                raise EvidenceUnavailable("Nonfinite trajectory observation")
            rows.append(
                dict(
                    experiment=name,
                    n_logged_steps=len(values),
                    full_mean=float(values.mean()),
                    last10_mean=float(values[-10:].mean()),
                    analysis_unit="single_trajectory",
                    inference="descriptive_only",
                    claim_eligible=False,
                    ci95=None,
                )
            )
    if not rows:
        raise EvidenceUnavailable("No observed traces")
    return rows, []


def table2_cross_library(real_seeds=None):
    raise EvidenceUnavailable(
        "Legacy table2_cross_library lacks independent measured run receipts. Use analyze_measured_runs; aggregate constants and trajectory steps are not replications."
    )


def table3_gsm8k_scaling():
    raise EvidenceUnavailable(
        "Legacy table3_gsm8k_scaling lacks independent measured run receipts. Use analyze_measured_runs; aggregate constants and trajectory steps are not replications."
    )


def table4_ppo_vs_grpo(exps):
    raise EvidenceUnavailable(
        "Legacy table4_ppo_vs_grpo lacks independent measured run receipts. Use analyze_measured_runs; aggregate constants and trajectory steps are not replications."
    )


def trl_cross_seed_summary():
    raise EvidenceUnavailable(
        "Legacy trl_cross_seed_summary lacks independent measured run receipts. Use analyze_measured_runs; aggregate constants and trajectory steps are not replications."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Family-wide Bonferroni across the whole paper
# ──────────────────────────────────────────────────────────────────────────────
def apply_family_wide(comps: List[Dict]) -> List[Dict]:
    """Re-compute Bonferroni and BH across every comparison returned by the
    Table 1-4 drivers combined. This is the global correction quoted in the
    paper ("Bonferroni across the full paper family")."""
    finite = [(i, c["p_raw"]) for i, c in enumerate(comps) if np.isfinite(c["p_raw"])]
    raw_ps = [p for _, p in finite]
    bonf_all = bonferroni(raw_ps)
    bh_all = benjamini_hochberg(raw_ps)
    for (i, _), bp, bh_p in zip(finite, bonf_all, bh_all, strict=False):
        comps[i]["p_bonf_global"] = bp
        comps[i]["p_bh_global"] = bh_p
    for c in comps:
        c.setdefault("p_bonf_global", float("nan"))
        c.setdefault("p_bh_global", float("nan"))
    return comps


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────
def render_markdown(t1_rows, t2_rows, t3_rows, t4_rows, trl, comps):
    raise EvidenceUnavailable(
        "Legacy prose renderer disabled: use validated measured_analysis payload and render_stat_rigor_tex.render"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    from platform_hybrid.experiments.measured_analysis import main as measured_main

    measured_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
