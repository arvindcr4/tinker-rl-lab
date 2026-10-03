"""Shared analysis primitives for the top-level platform_modal iteration scripts.

Each function here was a body-identical copy in two or more scripts
(length_bias_iter*, scaling_law_*, group_size_iter*, zvf_*, berkeley/*).
Callers import them under their original local names, so call sites are
unchanged. Functions take everything they need as parameters (paths, RNGs,
bootstrap counts); none reads a caller's module globals. RNG seeding stays
at the call sites.

Heavy optional dependencies (scipy, matplotlib) are imported lazily.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# TSV I/O
# ---------------------------------------------------------------------------


def read_tsv_lines(path):
    """Split non-empty lines on tabs (berkeley/*: ``_read_tsv``)."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            rows.append(ln.split("\t"))
    return rows


def write_dict_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """length_bias_iter48/52/56/60: ``write_tsv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_rows_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    """scaling_law_*: ``_write_tsv`` (header row + list rows, prints the path)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def write_header_tsv(path, header, rows):
    """group_size_iter51/55/59: ``write_tsv`` (str() of each cell, no quoting)."""
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")


def write_dicts_tsv(path: Path, dicts: list[dict]) -> None:
    """group_size_iter71/75: ``write_tsv`` (columns from the first row)."""
    if not dicts:
        return
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()), delimiter="\t")
        w.writeheader()
        for r in dicts:
            w.writerow(r)


def write_union_tsv(rows: list[dict], path: Path):
    """group_size_iter79/83: ``write_tsv`` (columns = ordered union of row keys)."""
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_commented_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    header_comment: str,
    cols: list[str] | None = None,
) -> None:
    """zvf_diagnostic_iter106 / zvf_discrimination_iter110: ``_write_tsv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None and rows:
        cols = list(rows[0].keys())
    elif cols is None:
        cols = []
    with path.open("w") as fh:
        fh.write(header_comment)
        if not header_comment.endswith("\n"):
            fh.write("\n")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def write_hash_commented_tsv(path: Path, rows, header_comment: str | None = None) -> None:
    """zvf_iter98/102: ``_write_tsv`` ("# "-prefixed comment lines, "(empty)" if no rows)."""
    with path.open("w") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        if not rows:
            f.write("(empty)\n")
            return
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_step_log(path: str) -> list[dict[str, Any]]:
    """length_bias_iter96/104/108: runs with >=5 steps as {algo, seed, n, L, R}."""
    with open(path) as fh:
        d = json.load(fh)
    runs_raw = d["runs"]
    out = []
    for r in runs_raw:
        step_log = r.get("step_log") or []
        if len(step_log) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in step_log], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in step_log], dtype=np.float64)
        out.append({"algo": r["algo"], "seed": r["seed"], "n": int(len(step_log)), "L": L, "R": R})
    return out


def load_step_log_task(path: str, task_label: str) -> list[dict[str, Any]]:
    """length_bias_iter120/124: as ``load_step_log`` plus a task label and int seed."""
    with open(path) as fh:
        d = json.load(fh)
    out = []
    for r in d["runs"]:
        sl = r.get("step_log") or []
        if len(sl) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in sl], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in sl], dtype=np.float64)
        out.append(
            {
                "task": task_label,
                "algo": r["algo"],
                "seed": int(r["seed"]),
                "n": int(len(sl)),
                "L": L,
                "R": R,
            }
        )
    return out


def load_iter108_perrun(path: str) -> list[dict[str, Any]]:
    """length_bias_iter116/120/124: parse length_bias_iter108_perrun_progress.tsv."""
    out = []
    with open(path) as fh:
        hdr = fh.readline().rstrip().split("\t")
        for line in fh:
            f = line.rstrip().split("\t")
            row = dict(zip(hdr, f))
            row["window"] = int(row["window"])
            row["seed"] = int(row["seed"])
            row["n_total"] = int(row["n_total"])
            row["n_in_window"] = int(row["n_in_window"])
            for k in ("phi_L", "phi_R", "bwd", "fwd", "bwd_signed", "fwd_signed"):
                row[k] = float(row[k])
            out.append(row)
    return out


def load_token_norm(res: Path):
    """group_size_iter51/55/59: rows of group_size_token_normalized.tsv as {T, G, acc, ci_lo, ci_hi, gu}."""
    out = []
    with open(res / "group_size_token_normalized.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "T": int(row["budget_tokens"]),
                    "G": int(row["G"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "ci_lo": float(row["heldout_acc_ci_low"]),
                    "ci_hi": float(row["heldout_acc_ci_high"]),
                    "gu": float(row["gu_estimate"]),
                }
            )
    return out


def load_zvf_sweep(res: Path):
    """group_size_iter55/59: rows of groupsize_zvf_sweep.tsv."""
    out = []
    with open(res / "groupsize_zvf_sweep.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "G": int(row["G"]),
                    "n_seeds": int(row["n_seeds"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "acc_se": float(row["heldout_acc_se"]),
                    "last10": float(row["last10_mean"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_th": float(row["zvf_theory_at_mean_p"]),
                    "mean_reward_train": float(row["mean_reward_train"]),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_bootstrap(g: list[float], d: list[float], n_boot: int, rng: random.Random) -> dict:
    """length_bias_iter56/60: percentile bootstrap of mean(d - g) over pairs."""
    diffs = [di - gi for gi, di in zip(g, d)]
    n = len(diffs)
    if n == 0:
        return {
            "mean_diff": 0.0,
            "sd_diff": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "p_le0": 1.0,
            "n_pairs": 0,
        }
    mean_diff = sum(diffs) / n
    var = sum((x - mean_diff) ** 2 for x in diffs) / max(1, n - 1)
    sd_diff = math.sqrt(var)
    idx = list(range(n))
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.choice(idx)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {
        "mean_diff": round(mean_diff, 8),
        "sd_diff": round(sd_diff, 8),
        "ci_lo": round(boots[int(0.025 * n_boot)], 8),
        "ci_hi": round(boots[int(0.975 * n_boot)], 8),
        "p_le0": round((sum(1 for b in boots if b <= 0) + 1) / (n_boot + 1), 4),
        "n_pairs": n,
    }


def spearman(x, y) -> tuple[float, float]:
    """length_bias_iter116/120/124: scipy Spearman (rho, p)."""
    from scipy import stats

    sp = stats.spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue)


def permutation_null(x, y, B: int, seed: int) -> dict[str, float]:
    """length_bias_iter116/120/124: two-sided permutation null for Spearman rho."""
    rng = np.random.default_rng(seed)
    obs, _ = spearman(x, y)
    abs_obs = abs(obs)
    n = len(x)
    y_arr = np.array(y, dtype=np.float64)
    count = 0
    boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.permutation(n)
        r_b, _ = spearman(x, y_arr[idx])
        boot[b] = r_b
        if abs(r_b) >= abs_obs:
            count += 1
    p_perm = (count + 1) / (B + 1)
    return {
        "obs_rho": float(obs),
        "abs_obs": float(abs_obs),
        "p_perm": float(p_perm),
        "null_mean": float(boot.mean()),
        "null_std": float(boot.std()),
        "null_q025": float(np.quantile(boot, 0.025)),
        "null_q500": float(np.quantile(boot, 0.5)),
        "null_q975": float(np.quantile(boot, 0.975)),
        "n": int(n),
        "B": int(B),
    }


def window_mean(x: np.ndarray, n_w: int) -> np.ndarray:
    """Return window-mean of x over n_w equal-length windows."""
    n = len(x)
    edges = [int(np.floor(n * w / n_w)) for w in range(n_w + 1)]
    for w in range(1, n_w + 1):
        edges[w] = max(edges[w], edges[w - 1] + 4)
        edges[w] = min(edges[w], n)
    out = np.zeros(n_w, dtype=np.float64)
    for w in range(n_w):
        out[w] = float(x[edges[w] : edges[w + 1]].mean())
    return out


def build_long(perrun108: list[dict], step_runs: list[dict], n_w: int) -> list[dict[str, Any]]:
    """length_bias_iter120/124: join iter108 per-window rows with window-mean L/R."""
    by_step: dict[tuple, dict[int, dict]] = {}
    for r in step_runs:
        by_step.setdefault((r["task"], r["algo"]), {})[r["seed"]] = r

    out: list[dict[str, Any]] = []
    for r108 in perrun108:
        task = r108["task"]
        algo = r108["algo"]
        seed = r108["seed"]
        w = r108["window"]
        run = by_step.get((task, algo), {}).get(seed)
        if run is None:
            continue
        L_w = window_mean(run["L"], n_w=n_w)
        R_w = window_mean(run["R"], n_w=n_w)
        out.append(
            {
                "task": task,
                "algo": algo,
                "seed": seed,
                "window": w,
                "bwd": float(r108["bwd"]),
                "fwd": float(r108["fwd"]),
                "bwd_signed": float(r108["bwd_signed"]),
                "phi_L": float(r108["phi_L"]),
                "phi_R": float(r108["phi_R"]),
                "L_w": float(L_w[w]),
                "R_w": float(R_w[w]),
                "n_in_window": int(r108["n_in_window"]),
                "n_total": int(r108["n_total"]),
            }
        )
    return out


def ccf_at_lags(e_a: np.ndarray, e_b: np.ndarray, K: int) -> np.ndarray:
    """CCF for lags -K..+K; length 2K+1 (length_bias_iter96/104/108)."""
    n = min(len(e_a), len(e_b))
    a = e_a[:n] - e_a[:n].mean()
    b = e_b[:n] - e_b[:n].mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum()) + 1e-300
    out = np.zeros(2 * K + 1, dtype=np.float64)
    for i, k in enumerate(range(-K, K + 1)):
        if k >= 0:
            x = a[: n - k]
            y = b[k:n]
        else:
            x = a[-k:n]
            y = b[: n + k]
        if len(x) < 3:
            out[i] = 0.0
            continue
        out[i] = float(np.dot(x, y) / denom)
    return out


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Plain OLS. Returns (intercept, slope, se_slope) (scaling_law_fit/121/137)."""
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


def fit_saturation(t, y):
    """scaling_law_iter93/97/101: ``_fit_saturation`` grid search over lambda."""
    lam_grid = np.geomspace(0.01, 10.0, 60)
    best = (np.inf, None)
    for lam in lam_grid:
        X = np.vstack([np.ones_like(t), 1.0 - np.exp(-lam * t)]).T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rm = float(coef[1])
        if rm < max(0.4 * float(y.max()), 0.05):
            continue
        rm = max(rm, 0.05)
        rm = min(rm, 1.5)
        pred = coef[0] + rm * (1.0 - np.exp(-lam * t))
        sse = float(np.sum((y - pred) ** 2))
        if sse < best[0]:
            best = (sse, [rm, float(lam)])
    return best[1] if best[1] else [float(y.mean()), 0.3]


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC of (score -> positive label) via argsort ranks (zvf_iter98/102: ``_auc_rank``).

    Ties are ranked by argsort order, not averaged.
    """
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci_xy(x: np.ndarray, y: np.ndarray, fn, B: int = 2000, seed: int = 0):
    """zvf_iter98/102: ``_bootstrap_ci`` -> (2.5%, 50%, 97.5%) of fn over paired resamples."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = fn(x[idx], y[idx])
        except Exception:
            boots[b] = float("nan")
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 50)),
        float(np.percentile(boots, 97.5)),
    )


def auroc_argsort_ranks(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC, O(n log n) (zvf_diagnostic_iter134: ``auroc``).

    Ties are ranked by argsort order, not averaged.
    """
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(
    y_true: np.ndarray, score: np.ndarray, rng: np.random.Generator, B: int = 2000
) -> tuple[float, float]:
    """zvf_diagnostic_iter134: 95% percentile CI of ``auroc_argsort_ranks`` over resamples."""
    idx = np.arange(len(y_true))
    aucs = []
    for _ in range(B):
        b = rng.choice(idx, size=len(idx), replace=True)
        a = auroc_argsort_ranks(y_true[b], score[b])
        if not math.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def rankdata_avg(vs) -> list[float]:
    """1-based ranks with ties averaged (zvf_iter118_diagnostic: ``_rankdata``)."""
    n = len(vs)
    order = sorted(range(n), key=lambda i: vs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and vs[order[j]] == vs[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def maybe_matplotlib():
    """Return pyplot with the Agg backend, or None if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None
