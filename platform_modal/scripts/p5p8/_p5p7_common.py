"""Helpers shared verbatim by the p5/p7 analysis scripts in this directory.

Each function is the body its callers used to copy; constants they used to read
from module globals (paths, METRICS, G sizes, thresholds) are parameters here.
Scripts import it as ``from _p5p7_common import ...``; running a script puts this
directory on ``sys.path``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
from collections import defaultdict
from statistics import fmean


def _midranks(vs, n):
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(n), key=lambda i: vs[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman_v1(xs, ys):
    """Spearman rho (midrank ties), rank means as ``sum / n``; nan below n=3, 0.0 if constant."""
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = _midranks(xs, n), _midranks(ys, n)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman_v2(xs, ys):
    """As ``spearman_v1`` but rank means via ``statistics.mean`` (can differ in the last ULP)."""
    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = _midranks(xs, n), _midranks(ys, n)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def axis_variance_fraction(rows, axis_key, value_key):
    """SS_axis / SS_total nested ANOVA-style decomposition.

    Returns (eta2, ss_axis, ss_within, n_groups, grand_mean).
    """
    grand = []
    by_axis = defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand or len(by_axis) < 2:
        return float("nan"), 0.0, 0.0, len(by_axis), float("nan")
    grand_mean = fmean(grand)
    ss_total = sum((x - grand_mean) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - grand_mean) ** 2 for vs in by_axis.values())
    ss_within = ss_total - ss_axis
    eta2 = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    return eta2, ss_axis, ss_within, len(by_axis), grand_mean


def load_rows(path, metrics):
    """Load the N2 metrics TSV into dicts; int step/group_size/seed, float ``metrics``."""
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            d = dict(zip(header, parts))
            for col in ("step", "group_size", "seed"):
                if col in d:
                    d[col] = int(d[col])
            for col in metrics + ["frac_all_zero", "frac_all_one", "lag1_autocorr"]:
                if col in d and d[col] not in ("nan", "", "None"):
                    try:
                        d[col] = float(d[col])
                    except ValueError:
                        d[col] = float("nan")
            rows.append(d)
    return rows


def load_tensors_a(n2_dir, methods, seed):
    """dict[method] -> list of step records from ``{method}_s{seed}_tensors.jsonl``."""
    by_method = {}
    for m in methods:
        path = n2_dir / f"{m}_s{seed}_tensors.jsonl"
        steps = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                steps.append(json.loads(line))
        by_method[m] = steps
    return by_method


def load_tensors_b(data_dir, methods):
    """dict[method] -> step records from every ``*_tensors.jsonl`` in ``data_dir``, step-sorted."""
    out = {m: [] for m in methods}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_tensors.jsonl"))):
        method = os.path.basename(path).split("_")[0]
        if method not in methods:
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out[method].append(json.loads(line))
    for m in methods:
        out[m].sort(key=lambda r: r["step"])
    return out


def load_tensors_c(n2_dir, methods):
    """dict[(method, step)] -> list[list[float]] rewards from ``{method}_s0_tensors.jsonl``."""
    out = {}
    for m in methods:
        path = n2_dir / f"{m}_s0_tensors.jsonl"
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out[(m, d["step"])] = d["rewards"]
    return out


def load_tensors_d(tensor_dir, method):
    """List of step records from ``{method}_s0_tensors.jsonl``."""
    fp = tensor_dir / f"{method}_s0_tensors.jsonl"
    out = []
    with fp.open() as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def c3_hybrid(z, tau, delta, *, g_base, g_esc, g_des):
    """Per-step hybrid controller: shrink above tau+delta, escalate in [tau, tau+delta)."""
    out = []
    for zt in z:
        if zt >= tau + delta:
            out.append(g_des)  # post-escalation easy: shrink to fast
        elif zt >= tau:
            out.append(g_esc)  # boundary: escalate to slow
        else:
            out.append(g_base)
    return out


def bernoulli_z(p_hat: float, G: int) -> float:
    """Closed-form Bernoulli zero-variance fraction: p^G + (1-p)^G."""
    if p_hat <= 0.0:
        return 1.0
    if p_hat >= 1.0:
        return 1.0
    return p_hat**G + (1.0 - p_hat) ** G


def is_boundary(p_hat: float) -> bool:
    """Boundary prompt (k=0 or k=G): no contrast possible at any G."""
    return p_hat <= 0.0 or p_hat >= 1.0


def c_unified_c4(p_hat, z_obs, *, tau_degen, g_base):
    """Iter-119 C4 unified controller (regime-gated composition), per prompt."""
    if z_obs < 0.50:
        # FAST regime: drop G (Dualformer)
        return 2 if is_boundary(p_hat) else 4
    if z_obs >= tau_degen:
        # DEGENERATE regime: escalate via Bernoulli inversion, cap G=32
        if is_boundary(p_hat):
            return g_base
        target_z = max(0.5, 0.5 * z_obs)
        best_g = g_base
        for g in [16, 32]:
            if bernoulli_z(p_hat, g) < target_z:
                best_g = g
                break
        return best_g
    return g_base
