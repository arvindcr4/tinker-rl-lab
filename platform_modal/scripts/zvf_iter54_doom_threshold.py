#!/usr/bin/env python3
"""Iter 54 Pillar 2: ZVF "Doom Threshold" (ZVF_crit) and cross-library
contrastive-yield gap (Y_gap).

Headline claim
--------------
There exists an empirical ZVF value ZVF_crit above which no observed run in
any library recovers. This is the actionable, paper-grade early-stopping
rule derived from cross-library training data: once mean-ZVF clears
ZVF_crit on a held-out validation rollout batch, abort the run.

We derive ZVF_crit by sweeping thresholds over the per-(library, model,
seed) rows in zvf_summary.tsv and finding the smallest threshold that
perfectly separates "converged" rows from "non-converged" rows.

Why this matters
----------------
Existing ZVF work (iters 22, 30, 34, 38, 42, 46, 50) treated ZVF as a
*signal* (correlated with reward) and as a *diagnostic* (anti-herding
diversity bonus). Iter 54 promotes ZVF to a *gating rule*: a single
threshold that, on cross-library evidence, prevents wasted compute by
predicting terminal collapse before heldout_acc has dropped.

We also derive the cross-library contrastive-yield gap:

    Y_gap = Y_obs - Y_iid = (1 - zvf_obs) - (1 - zvf_iid)
          = zvf_iid - zvf_obs = delta_div

so a positive Y_gap is the structural diversity bonus the sampler
provides beyond independent-rollout Bernoulli collision.

Inputs
------
- platform_hybrid/experiments/results/zvf_summary.tsv     (per-run pooled)
- platform_hybrid/experiments/results/groupsize_zvf_sweep.json
- platform_hybrid/experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json
- platform_hybrid/experiments/results/variance_mitigation.tsv   (per-step)
- platform_hybrid/experiments/results/tool_code_reward_diagnostics.tsv

Outputs
-------
- platform_hybrid/experiments/results/zvf_iter54_doom_threshold.tsv
  Rows: threshold_candidate, n_below_converged, n_below_nonconv,
        precision_converged_below, recall_converged_below, J_stat
  Plus a "BEST" row that records ZVF_crit.
- platform_hybrid/experiments/results/zvf_iter54_yield_gap.tsv
  Per-(library, model, group_size) contrastive-yield gap with bootstrap CI.
- platform_hybrid/experiments/results/zvf_iter54_doom_summary.tsv
  One-row summary suitable for inclusion in the paper's headline table.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a TSV. Comment lines starting with '#' are skipped.

    The first non-comment line is treated as the header. Returns the
    header (with leading '#' stripped) and one dict per data row.
    """
    with path.open() as f:
        rdr = csv.reader(f, delimiter="\t")
        rows = [r for r in rdr]
    rows = [r for r in rows if r and not (len(r) > 0 and r[0].startswith("#"))]
    if not rows:
        return [], []
    header = [c.strip() for c in rows[0]]
    out = []
    for r in rows[1:]:
        if not r or all(c == "" for c in r):
            continue
        out.append({h: (v if i < len(r) else "") for i, (h, v) in enumerate(zip(header, r))})
    return header, out


def _f(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Per-(library, model, seed) rows with a converged/non-converged label
# ---------------------------------------------------------------------------


def load_rows() -> List[Dict[str, Any]]:
    """Load per-run pooled rows from zvf_summary.tsv and enrich each
    with the per-step max ZVF observed during training (from the
    per-step time-series files we have on disk).

    Failure label is mapped to a binary "non_converged" indicator:
        converged    -> False
        drift        -> True
        collapse     -> True
        plateau      -> True   (plateau is non-converged for doom-threshold
                               purposes — model did not improve.)

    IMPORTANT: the "max_zvf" column in the source TSV mixes two
    statistics: for tinker_gsm8k_zvf and drgrpo_vs_grpo it is the
    per-problem maximum (some problems are 0/8 or 8/8, hence max=1.0
    by construction); for variance_mitigation and groupsize_zvf_sweep
    it is the per-step time-series maximum. These are different
    signals. We tag rows as `ts_max_zvf = True` only when the
    per-step time-series was used, so downstream sweeps can restrict
    to that subset.
    """
    _, rows = _read_tsv(RESULTS / "zvf_summary.tsv")
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not r.get("mean_zvf"):
            continue
        mean_zvf = _f(r.get("mean_zvf", ""))
        if math.isnan(mean_zvf):
            continue
        peak = _f(r.get("peak", ""))
        last10 = _f(r.get("last10_avg", ""))
        label = r.get("failure_label", "").strip()
        n_seeds = _f(r.get("n_seeds", "1"), default=1.0)
        non_converged = label in {"collapse", "drift", "plateau"}
        exp = r.get("experiment", "").strip()
        # Per-step max ZVF, where the per-step TSV is available.
        max_zvf = float("nan")
        ts_max_zvf = False
        if exp == "variance_mitigation":
            seed = r.get("seed", "").strip()
            if seed:
                mz = _per_step_max_zvf_variance_mitigation(seed)
                if mz is not None and not math.isnan(mz):
                    max_zvf = mz
                    ts_max_zvf = True
        elif exp == "groupsize_zvf_sweep":
            gs_val = r.get("group_size", "").strip()
            if gs_val:
                mz = _per_step_max_zvf_groupsize(int(gs_val))
                if mz is not None and not math.isnan(mz):
                    max_zvf = mz
                    ts_max_zvf = True
        if math.isnan(max_zvf):
            max_zvf = mean_zvf  # fallback
        out.append(
            {
                "library": exp,
                "model": r.get("model", ""),
                "task": r.get("task", ""),
                "group_size": r.get("group_size", ""),
                "n_seeds": n_seeds,
                "mean_zvf": mean_zvf,
                "max_zvf": max_zvf,
                "ts_max_zvf": ts_max_zvf,
                "peak": peak,
                "last10_avg": last10,
                "failure_label": label,
                "non_converged": non_converged,
            }
        )
    return out


def _per_step_max_zvf_variance_mitigation(seed: str) -> Optional[float]:
    path = RESULTS / "variance_mitigation.tsv"
    if not path.exists():
        return None
    _, rows = _read_tsv(path)
    zvfs = [_f(r.get("zvf", "")) for r in rows if r.get("seed", "").strip() == seed]
    zvfs = [z for z in zvfs if not math.isnan(z)]
    return max(zvfs) if zvfs else None


def _per_step_max_zvf_groupsize(G: int) -> Optional[float]:
    """Return the max per-step ZVF across the groupsize sweep runs for G.

    The source file stores per-step zvf in run["step_log"][i]["zvf"].
    """
    path = RESULTS / "groupsize_zvf_sweep.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    runs = d.get("runs", [])
    zvfs: List[float] = []
    for r in runs:
        if r.get("group_size") != G:
            continue
        for s in r.get("step_log", []):
            if "zvf" in s:
                try:
                    zvfs.append(float(s["zvf"]))
                except (TypeError, ValueError):
                    continue
    return max(zvfs) if zvfs else None


# ---------------------------------------------------------------------------
# Doom threshold sweep
# ---------------------------------------------------------------------------


def _sweep_thresholds(
    rows: List[Dict[str, Any]],
    column: str = "mean_zvf",
) -> List[Dict[str, Any]]:
    """Sweep candidate ZVF thresholds in [0.05, 0.99] step 0.01.

    For each threshold T, "below T" = converged_predicted. A threshold
    qualifies as a perfect separator when every row below T is converged
    and every row above T is non_converged.

    `column` selects the ZVF statistic to threshold on; defaults to
    "mean_zvf" (the cross-run mean). Pass "max_zvf" to threshold on
    the per-run maximum ZVF seen in any single training step.
    """
    if not rows:
        return []
    valid = [r for r in rows if column in r and not math.isnan(r[column])]
    if not valid:
        return []
    candidates = [round(0.05 + 0.01 * i, 2) for i in range(int((0.99 - 0.05) / 0.01) + 1)]
    out: List[Dict[str, Any]] = []
    for t in candidates:
        below_conv = sum(1 for r in valid if r[column] < t and not r["non_converged"])
        below_nonconv = sum(1 for r in valid if r[column] < t and r["non_converged"])
        above_conv = sum(1 for r in valid if r[column] >= t and not r["non_converged"])
        above_nonconv = sum(1 for r in valid if r[column] >= t and r["non_converged"])
        total_below = below_conv + below_nonconv
        total_above = above_conv + above_nonconv
        if total_below == 0 or total_above == 0:
            continue
        precision_below = below_conv / total_below
        recall_below = below_conv / max(1, below_conv + above_conv)
        specificity = above_nonconv / max(1, above_nonconv + below_nonconv)
        j_stat = recall_below + specificity - 1.0
        out.append(
            {
                "threshold": t,
                "n_below_total": total_below,
                "n_below_converged": below_conv,
                "n_below_nonconv": below_nonconv,
                "n_above_total": total_above,
                "n_above_converged": above_conv,
                "n_above_nonconv": above_nonconv,
                "precision_below_converged": precision_below,
                "recall_below_converged": recall_below,
                "youden_J": j_stat,
                "perfect_separator": below_nonconv == 0 and above_conv == 0,
            }
        )
    return out


def _select_doom_threshold(sweep: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the threshold with the highest Youden's J.

    If multiple perfect separators exist, return the smallest such
    threshold (most conservative rule). Otherwise return the argmax
    J-stat threshold as the "best fuzzy separator".
    """
    if not sweep:
        return None
    perfects = [s for s in sweep if s["perfect_separator"]]
    if perfects:
        perfects.sort(key=lambda s: s["threshold"])
        best = perfects[0]
        best["separator_type"] = "perfect"
        return best
    by_j = max(sweep, key=lambda s: s["youden_J"])
    by_j = dict(by_j)
    by_j["separator_type"] = "fuzzy"
    return by_j


def _bootstrap_doom_ci(
    rows: List[Dict[str, Any]],
    B: int = 2000,
    seed: int = 54,
    column: str = "mean_zvf",
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """Bootstrap the doom threshold.

    Resample rows with replacement, fit the best threshold on the
    resample, and report the percentile CI of the chosen thresholds.

    Returns (lo, hi, point_estimate_dict_or_None).
    """
    import random

    random.seed(seed)
    thresholds_seen: List[float] = []
    n = len(rows)
    if n < 4:
        return (float("nan"), float("nan"), None)
    for _ in range(B):
        sample = [rows[random.randrange(n)] for _ in range(n)]
        sweep = _sweep_thresholds(sample, column=column)
        best = _select_doom_threshold(sweep)
        if best is not None:
            thresholds_seen.append(best["threshold"])
    if not thresholds_seen:
        return (float("nan"), float("nan"), None)
    thresholds_seen.sort()
    lo = thresholds_seen[int(0.025 * len(thresholds_seen))]
    hi = thresholds_seen[int(0.975 * len(thresholds_seen)) - 1]
    median = thresholds_seen[len(thresholds_seen) // 2]
    return (lo, hi, {"threshold": median})


# ---------------------------------------------------------------------------
# Contrastive-yield gap (Y_gap) per (library, model, G)
# ---------------------------------------------------------------------------


def _load_per_problem_yield_gap() -> List[Dict[str, Any]]:
    """Load per-problem rollouts and compute Y_gap = zvf_iid - zvf_obs.

    Sources: tinker_gsm8k_zvf_s{42,123,456}.json and the
    groupsize_zvf_sweep.json (the per-step zvf is averaged over the
    batch; we approximate per-problem structure for the G-sweep by
    treating the trajectory mean_zvf against the per-G theory mean).
    """
    out: List[Dict[str, Any]] = []
    # Tinker GSM8K: real per-problem rollouts.
    for seed in (42, 123, 456):
        path = RESULTS / f"tinker_gsm8k_zvf_s{seed}.json"
        d = json.loads(path.read_text())
        for p in d["per_problem"]:
            rewards = p["rewards"]
            G = len(rewards)
            k = sum(rewards)
            zvf_obs = 1.0 if (k == 0 or k == G) else 0.0
            # p_x is the mean reward in the group (the empirical estimate
            # of the marginal success probability). For Bernoulli collision
            # the i.i.d. baseline is p**G + (1-p)**G.
            p_x = k / G
            zvf_iid = p_x ** G + (1.0 - p_x) ** G
            out.append(
                {
                    "source": "tinker_gsm8k",
                    "library": "tinker_gsm8k",
                    "model": d.get("model", "Qwen/Qwen3-8B"),
                    "seed": seed,
                    "G": G,
                    "p_x": p_x,
                    "zvf_obs": zvf_obs,
                    "zvf_iid": zvf_iid,
                    "Y_gap": zvf_iid - zvf_obs,
                }
            )

    # groupsize_zvf_sweep: aggregate rows. We use the per-(G) summary
    # from the TSV (which carries mean_reward_train) to compute the
    # i.i.d. Bernoulli-collision baseline.
    _, gs_rows = _read_tsv(RESULTS / "groupsize_zvf_sweep.tsv")
    for r in gs_rows:
        G_int = int(r.get("G", "0"))
        if G_int <= 0:
            continue
        p_x = _f(r.get("mean_reward_train", "0.5"))
        if math.isnan(p_x):
            p_x = 0.5
        p_x = min(max(p_x, 1e-3), 1 - 1e-3)
        zvf_obs = _f(r.get("mean_zvf", ""))
        if math.isnan(zvf_obs):
            continue
        zvf_iid = p_x ** G_int + (1.0 - p_x) ** G_int
        out.append(
            {
                "source": "groupsize_zvf_sweep_agg",
                "library": f"groupsize_G{G_int}",
                "model": "Qwen/Qwen2.5-0.5B",
                "seed": -1,
                "G": G_int,
                "p_x": p_x,
                "zvf_obs": zvf_obs,
                "zvf_iid": zvf_iid,
                "Y_gap": zvf_iid - zvf_obs,
            }
        )
    return out


def _aggregate_yield_gap(
    rows: List[Dict[str, Any]], B: int = 2000, seed: int = 54
) -> List[Dict[str, Any]]:
    """Per-(source, library, G) aggregation with bootstrap CI on Y_gap."""
    import random

    random.seed(seed)
    by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        k = (r["source"], r["library"], r["G"])
        by_key.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for (src, lib, G), group in sorted(by_key.items()):
        n = len(group)
        gaps = [g["Y_gap"] for g in group]
        if not gaps:
            continue
        mean_gap = statistics.fmean(gaps)
        std_gap = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
        # bootstrap CI on the mean
        means = []
        for _ in range(B):
            sample = [gaps[random.randrange(n)] for _ in range(n)]
            means.append(statistics.fmean(sample))
        means.sort()
        lo = means[int(0.025 * len(means))]
        hi = means[int(0.975 * len(means)) - 1]
        median_zvf_iid = statistics.median([g["zvf_iid"] for g in group])
        median_zvf_obs = statistics.median([g["zvf_obs"] for g in group])
        out.append(
            {
                "source": src,
                "library": lib,
                "G": G,
                "n": n,
                "mean_Y_gap": mean_gap,
                "median_Y_gap": statistics.median(gaps),
                "std_Y_gap": std_gap,
                "ci_lo": lo,
                "ci_hi": hi,
                "median_zvf_iid": median_zvf_iid,
                "median_zvf_obs": median_zvf_obs,
                "verdict": "anti-herd" if lo > 0 else ("herd" if hi < 0 else "neutral"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# " + " | ".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def write_doom_sweep(sweep: List[Dict[str, Any]], path: Path) -> None:
    header = (
        "threshold",
        "n_below_total",
        "n_below_converged",
        "n_below_nonconv",
        "n_above_total",
        "n_above_converged",
        "n_above_nonconv",
        "precision_below_converged",
        "recall_below_converged",
        "youden_J",
        "perfect_separator",
    )
    rows: List[Sequence[Any]] = []
    for s in sweep:
        rows.append(
            [
                s["threshold"],
                s["n_below_total"],
                s["n_below_converged"],
                s["n_below_nonconv"],
                s["n_above_total"],
                s["n_above_converged"],
                s["n_above_nonconv"],
                f"{s['precision_below_converged']:.4f}",
                f"{s['recall_below_converged']:.4f}",
                f"{s['youden_J']:.4f}",
                s["perfect_separator"],
            ]
        )
    _write_tsv(path, header, rows)


def write_doom_summary(
    rows: List[Dict[str, Any]],
    sweep: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    bootstrap: Tuple[float, float, Optional[Dict[str, Any]]],
    path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Single-row summary suitable for the paper headline table."""
    n = len(rows)
    n_conv = sum(1 for r in rows if not r["non_converged"])
    n_nonconv = sum(1 for r in rows if r["non_converged"])
    zvf_crit = best["threshold"] if best else float("nan")
    lo, hi, _ = bootstrap
    if best:
        conf = (
            f"[{lo:.3f},{hi:.3f}]" if not (math.isnan(lo) or math.isnan(hi)) else "n/a"
        )
    else:
        conf = "n/a (no perfect separator)"
    zvf_crit_str = f"{zvf_crit:.3f}" if best else "n/a"
    separator_type = best.get("separator_type", "?") if best else "n/a"
    header = (
        "n_total",
        "n_converged",
        "n_non_converged",
        "zvf_crit",
        "zvf_crit_bootstrap_ci",
        "separator_type",
        "min_converged_zvf",
        "max_nonconverged_zvf",
        "n_perfect_separators",
        "max_zvf_crit",
        "max_zvf_n",
        "note",
    )
    if rows:
        max_nonconv = max((r["mean_zvf"] for r in rows if r["non_converged"]), default=float("nan"))
        min_conv = min((r["mean_zvf"] for r in rows if not r["non_converged"]), default=float("nan"))
    else:
        max_nonconv = float("nan")
        min_conv = float("nan")
    n_perfect = sum(1 for s in sweep if s["perfect_separator"])
    max_zvf_crit = "n/a"
    max_zvf_n = "0"
    if extra and "max_zvf" in extra:
        m = extra["max_zvf"]
        if m.get("best_threshold") is not None:
            max_zvf_crit = f"{m['best_threshold']:.3f}"
        max_zvf_n = str(m.get("n_with_max", 0))
    note = (
        f"ZVF_crit={zvf_crit_str}  CI={conf}  type={separator_type}  "
        f"max_zvf_crit={max_zvf_crit} (n={max_zvf_n}); "
        f"n_perfect_separators={n_perfect}"
    )
    _write_tsv(
        path,
        header,
        [
            [
                n,
                n_conv,
                n_nonconv,
                zvf_crit,
                conf,
                separator_type,
                f"{min_conv:.4f}" if not math.isnan(min_conv) else "n/a",
                f"{max_nonconv:.4f}" if not math.isnan(max_nonconv) else "n/a",
                n_perfect,
                max_zvf_crit,
                max_zvf_n,
                note,
            ]
        ],
    )


def write_yield_gap(agg: List[Dict[str, Any]], path: Path) -> None:
    header = (
        "source",
        "library",
        "G",
        "n",
        "mean_Y_gap",
        "median_Y_gap",
        "std_Y_gap",
        "ci_lo",
        "ci_hi",
        "median_zvf_iid",
        "median_zvf_obs",
        "verdict",
    )
    rows: List[Sequence[Any]] = []
    for a in agg:
        rows.append(
            [
                a["source"],
                a["library"],
                a["G"],
                a["n"],
                f"{a['mean_Y_gap']:.4f}",
                f"{a['median_Y_gap']:.4f}",
                f"{a['std_Y_gap']:.4f}",
                f"{a['ci_lo']:.4f}",
                f"{a['ci_hi']:.4f}",
                f"{a['median_zvf_iid']:.4f}",
                f"{a['median_zvf_obs']:.4f}",
                a["verdict"],
            ]
        )
    _write_tsv(path, header, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    rows = load_rows()
    if not rows:
        print("[iter54] no rows loaded from zvf_summary.tsv", file=sys.stderr)
        return 1

    # Threshold on mean-ZVF (the cross-run pooled statistic).
    sweep_mean = _sweep_thresholds(rows, column="mean_zvf")
    best_mean = _select_doom_threshold(sweep_mean)
    boot_mean = _bootstrap_doom_ci(rows, column="mean_zvf")

    # Threshold on max-ZVF (per-step time-series max). Restrict to
    # rows where the time-series max-ZVF is meaningful, not the
    # per-problem max-ZVF (which is 1.0 by construction when any
    # problem is all-correct or all-wrong).
    rows_with_max = [r for r in rows if r.get("ts_max_zvf", False)]
    sweep_max = _sweep_thresholds(rows_with_max, column="max_zvf")
    best_max = _select_doom_threshold(sweep_max)
    boot_max = _bootstrap_doom_ci(rows_with_max, column="max_zvf")

    yield_rows = _load_per_problem_yield_gap()
    yield_agg = _aggregate_yield_gap(yield_rows)

    out_sweep = RESULTS / "zvf_iter54_doom_threshold.tsv"
    out_summary = RESULTS / "zvf_iter54_doom_summary.tsv"
    out_yield = RESULTS / "zvf_iter54_yield_gap.tsv"
    out_sweep_max = RESULTS / "zvf_iter54_doom_threshold_maxzvf.tsv"

    write_doom_sweep(sweep_mean, out_sweep)
    write_doom_sweep(sweep_max, out_sweep_max)
    write_doom_summary(
        rows,
        sweep_mean,
        best_mean,
        boot_mean,
        out_summary,
        extra={
            "max_zvf": {
                "n_with_max": len(rows_with_max),
                "best_threshold": best_max["threshold"] if best_max else None,
                "boot_ci": boot_max,
            }
        },
    )
    write_yield_gap(yield_agg, out_yield)

    print(f"[iter54] loaded {len(rows)} runs from zvf_summary.tsv")
    print(
        f"[iter54] mean-ZVF sweep: {len(sweep_mean)} rows -> {out_sweep.name}"
    )
    print(
        f"[iter54] max-ZVF sweep: {len(sweep_max)} rows -> {out_sweep_max.name}"
    )
    if best_mean:
        lo, hi, _ = boot_mean
        ci_str = (
            f"[{lo:.3f}, {hi:.3f}]"
            if not (math.isnan(lo) or math.isnan(hi))
            else "n/a"
        )
        kind = best_mean.get("separator_type", "?")
        print(
            f"[iter54] ZVF_crit (mean) = {best_mean['threshold']:.3f}  "
            f"CI = {ci_str}  separator={kind}"
        )
    if best_max:
        lo, hi, _ = boot_max
        ci_str = (
            f"[{lo:.3f}, {hi:.3f}]"
            if not (math.isnan(lo) or math.isnan(hi))
            else "n/a"
        )
        kind = best_max.get("separator_type", "?")
        print(
            f"[iter54] ZVF_crit (max)  = {best_max['threshold']:.3f}  "
            f"CI = {ci_str}  separator={kind}"
        )
    print(
        f"[iter54] cross-library Y_gap: {len(yield_agg)} rows -> {out_yield.name}"
    )
    anti = sum(1 for a in yield_agg if a["verdict"] == "anti-herd")
    herd = sum(1 for a in yield_agg if a["verdict"] == "herd")
    neut = sum(1 for a in yield_agg if a["verdict"] == "neutral")
    print(
        f"[iter54] Y_gap verdicts: anti-herd={anti}  herd={herd}  neutral={neut}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
