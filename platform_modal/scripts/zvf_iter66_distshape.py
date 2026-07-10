#!/usr/bin/env python3
"""Pillar 2 Iter 66 — Cross-library ZVF distribution shape.

Goal: lift ZVF beyond its single-scalar (mean / median / stratification)
characterisation. For every variance-mitigation library we ask two
distribution-level questions that no prior ZVF artifact in this paper
answers:

    Q1. Are two libraries with identical mean ZVF actually behaving the
        same way on the rollout trace?  (CDF overlay + Kolmogorov-
        Smirnov distance between empirical CDFs of per-step ZVF.)

    Q2. How strong is the *anti-herding* signature for each library,
        relative to the i.i.d. Binomial(G, p_bar) prediction?
        (Anti-herding = observed ZVF below the i.i.d. lower bound.)

Both diagnostics rest on the per-step ZVF already logged in
platform_hybrid/experiments/results/variance_mitigation.tsv, and they are deliberately
orthogonal to the prior scalar and stratification work (iter58 signed
decomposition, iter62 quintile stratification). They answer reviewer
questions about library *identity* rather than library *rank*: are two
methods with similar mean ZVF actually drawing the same per-step ZVF
distribution, or are they hiding different time-series behaviour behind
a similar mean?

Outputs (8 artifacts):

    platform_hybrid/experiments/results/zvf_iter66_cdf_overlay.tsv
        Long-form CDF grid (50 zvf-bins x 9 methods) for the overlay
        figure.  Columns: zvf, method, ecdf.

    platform_hybrid/experiments/results/zvf_iter66_ks_matrix.tsv
        Pairwise Kolmogorov-Smirnov distance between per-step ZVF
        samples (9x9 methods).  Columns: method_a, method_b, ks_D,
        ks_pvalue, n_a, n_b.

    platform_hybrid/experiments/results/zvf_iter66_dendrogram.tsv
        Hierarchical clustering linkage matrix (agglomerative,
        average-linkage on KS distance) flattened to 8 merge rows.
        Columns: merge_a, merge_b, height, n_in_cluster_a, n_in_cluster_b.

    platform_hybrid/experiments/results/zvf_iter66_cluster_assign.tsv
        Flat cluster assignment at k=3 (cut of the dendrogram),
        one row per library.  Columns: method, cluster, mean_zvf,
        mean_acc, ks_to_grpo.

    platform_hybrid/experiments/results/zvf_iter66_anti_herding.tsv
        Per-library anti-herding signature: observed_mean_zvf,
        iid_predicted_zvf (Binomial(G, p_bar)), iid_lower_bound
        (Beta(1,1) prior predictive 5th percentile), delta_obs_minus_pred,
        delta_obs_minus_lower, signature_label
        (anti_herd / neutral / over_herd).  9 rows.

    platform_hybrid/experiments/results/zvf_iter66_summary.tsv
        9-row summary combining everything for paper Table.

    figures/zvf_iter66_distshape.pdf
    figures/zvf_iter66_distshape.png
        Two-panel figure: left = CDF overlay of per-step ZVF (one curve
        per library); right = KS distance heatmap with cluster boundary
        overlay.

Why these outputs:

The CDF overlay is a direct visual on whether two libraries' ZVF
behaviour is genuinely similar beyond the mean.  Two libraries with
identical mean ZVF but different CDF shape (e.g. one heavily bimodal,
the other unimodal) will look the same in scalar ZVF but very
different here.  The KS distance matrix is a numerical signature of
that.  The hierarchical clustering tells us which library families
are actually different from each other in their ZVF time-series.

The anti-herding table operationalises the (frontier synthesis)
framing of ZVF as "signal availability, not difficulty": an
observed-mean ZVF that is *below* the i.i.d. binomial lower bound
demonstrates autoregressive anti-herding (rho<0 sampling diversity),
which is the *opposite* of the naive difficulty story.  A library
whose observed ZVF matches the i.i.d. prediction is a "naive"
library; one that under-shoots it is "anti-herding"; one that
over-shoots it is "over-herding" (sampler collapses more than
i.i.d.).  These are structural properties of the library's sampling
behaviour, not of the difficulty distribution.

Pure stdlib + matplotlib.  No external stats deps.  The KS test is a
direct translation of scipy.stats.ks_2samp using only basic numpy-style
operations on the merged CDF.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "figures")
VM_TSV = os.path.join(RES, "variance_mitigation.tsv")

random.seed(66)


# ---------------------------------------------------------------------------
# Load variance_mitigation per-step ZVF rows.
# ---------------------------------------------------------------------------
def load_per_step():
    """Return dict method -> list of (zvf, heldout_acc, reward_mean).

    Filters out the step-0 (initialisation) rows so the CDF reflects the
    training trajectory and not the random-init precondition.
    """
    out = defaultdict(list)
    with open(VM_TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["step"] == "0":
                continue
            out[r["method"]].append(
                (
                    float(r["zvf"]),
                    float(r["heldout_acc"]),
                    float(r["reward_mean"]),
                )
            )
    return dict(out)


# ---------------------------------------------------------------------------
# Empirical CDF.
# ---------------------------------------------------------------------------
def ecdf(samples, grid):
    s = sorted(samples)
    n = len(s)
    out = []
    j = 0
    for x in grid:
        while j < n and s[j] <= x:
            j += 1
        out.append(j / n if n else 0.0)
    return out


# ---------------------------------------------------------------------------
# Two-sample Kolmogorov-Smirnov statistic + p-value via Smirnov's
# closed-form asymptotic complement (matches scipy.stats.ks_2samp).
# ---------------------------------------------------------------------------
def ks_2samp(a, b):
    a = sorted(a)
    b = sorted(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    # Walk both sorted lists and compute max |F_a(x) - F_b(x)|.
    i, j = 0, 0
    D = 0.0
    while i < n and j < m:
        if a[i] <= b[j]:
            x = a[i]
            while i < n and a[i] == x:
                i += 1
            while j < m and b[j] < x:
                j += 1
        else:
            x = b[j]
            while j < m and b[j] == x:
                j += 1
            while i < n and a[i] < x:
                i += 1
        d = abs(i / n - j / m)
        if d > D:
            D = d
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * D
    # Smirnov complement Q_KS(lam) = sum_{j=1..inf} (-1)^(j-1) exp(-2 j^2 lam^2)
    # (Smirnov 1948 asymptotic).  Truncate when term < 1e-12.
    p = 0.0
    if lam < 1e-12:
        p = 1.0
    else:
        term = math.exp(-2.0 * lam * lam)
        p = term
        sign = -1.0
        j = 2
        while True:
            t = math.exp(-2.0 * j * j * lam * lam)
            if t < 1e-12:
                break
            p += sign * t
            sign = -sign
            j += 1
        p = min(max(p, 0.0), 1.0)
    return D, p


# ---------------------------------------------------------------------------
# Hierarchical (agglomerative, average-linkage) clustering from a distance
# matrix.  Pure-Python; distances are KS distances.
# ---------------------------------------------------------------------------
def hclust_average(dist, labels):
    """Average-linkage agglomerative clustering on a square distance matrix.

    Returns a linkage table in chronological order: each merge row gives
    (left_id, right_id, height_D, size_left, size_right) where left/right
    are cluster IDs (0..n-1 for original leaves, n, n+1, ... for merged
    internal nodes).
    """
    n = len(labels)
    contents = [frozenset({i}) for i in range(n)]

    def cluster_dist(c1, c2):
        tot = 0.0
        cnt = 0
        for a in c1:
            for b in c2:
                tot += dist[a][b]
                cnt += 1
        return tot / cnt if cnt else 0.0

    d = {}
    for i in range(n):
        for j in range(i + 1, n):
            d[(contents[i], contents[j])] = cluster_dist(
                contents[i], contents[j]
            )

    linkage = []
    next_id = n
    while len(contents) > 1:
        best = None
        best_h = math.inf
        for k, v in d.items():
            if v < best_h:
                best_h = v
                best = k
        if best is None:
            break
        c1, c2 = best
        merged = c1 | c2
        # Identify the IDs of c1 and c2 from contents list.
        id_c1 = contents.index(c1)
        id_c2 = contents.index(c2)
        linkage.append((id_c1, id_c2, best_h, len(c1), len(c2)))
        # Drop distances touching c1 or c2.
        to_drop = [k for k in d if k[0] in (c1, c2) or k[1] in (c1, c2)]
        for k in to_drop:
            d.pop(k, None)
        # Compute distances from merged to every remaining cluster.
        for c in contents:
            if c in (c1, c2):
                continue
            key = (
                (merged, c) if hash(merged) < hash(c) else (c, merged)
            )
            d[key] = cluster_dist(merged, c)
        # Replace c1, c2 with merged; the new cluster will be appended
        # at position len(contents) before we pop the merged-out ones.
        contents = [c for c in contents if c not in (c1, c2)] + [merged]
        next_id += 1
    return linkage


def cut_tree(linkage, n_leaves, k):
    """Cut an agglomerative tree into k clusters.

    linkage rows are (left_id, right_id, height, size_left, size_right)
    in chronological order. Internal merged nodes get sequential IDs
    n_leaves, n_leaves+1, ..., n_leaves+len(linkage)-1.

    Returns dict original_index -> cluster_id (0..k-1).
    """
    if k >= n_leaves:
        return {i: i for i in range(n_leaves)}
    # Build contents map keyed by cluster id.
    contents = {i: {i} for i in range(n_leaves)}
    next_id = n_leaves
    for (a, b, h, na, nb) in linkage:
        contents[next_id] = contents[a] | contents[b]
        next_id += 1
    root = next_id - 1

    def collect_active(cid, undone, out):
        """Walk the tree; collect leaves reached via non-undone nodes."""
        if cid < n_leaves:
            out.append(cid)
            return
        if cid in undone:
            mi = cid - n_leaves
            a, b, h, na, nb = linkage[mi]
            collect_active(a, undone, out)
            collect_active(b, undone, out)
        else:
            # Active internal node: report it (the leaves inside it
            # are one cluster).
            out.append(cid)

    undone = set()
    while True:
        active = []
        collect_active(root, undone, active)
        if len(active) >= k:
            break
        # Find the active *internal* node with the largest size (= most
        # leaves inside it). If none are internal, we cannot split
        # further and stop.
        candidates = [
            cid for cid in active
            if cid >= n_leaves and cid not in undone
        ]
        if not candidates:
            break
        # Pick the active internal node with the most leaves inside it
        # (tie-break by highest merge height).
        candidates.sort(
            key=lambda c: (len(contents[c]), -linkage[c - n_leaves][2]),
            reverse=True,
        )
        target = candidates[0]
        undone.add(target)

    # Map each leaf to its active cluster id (using the active cluster
    # id as the cluster label, then re-map to 0..k-1).
    seen = {}
    assign = {}
    for cid in active:
        for orig in contents[cid]:
            if cid not in seen:
                seen[cid] = len(seen)
            assign[orig] = seen[cid]
    return assign

    # Map each leaf to its active cluster id.
    assign = {}
    seen = {}
    for cid in active:
        for orig in contents[cid]:
            assign[orig] = cid
    for orig, cid in assign.items():
        if cid not in seen:
            seen[cid] = len(seen)
        assign[orig] = seen[cid]
    return assign


# ---------------------------------------------------------------------------
# Anti-herding quantification.
# ---------------------------------------------------------------------------
def iid_predicted_zvf(accuracy, G=8):
    """Probability of an all-same outcome under iid Binomial(G, p).

    ZVF_iid = p^G + (1-p)^G.
    """
    return accuracy ** G + (1.0 - accuracy) ** G


def beta_lower_bound(accuracy, G=8, n_eff=200, q=0.05):
    """5th percentile of the Beta-Binomial predictive over p.

    Uses a Beta(1, 1) prior on p (uninformative) and a posterior
    Beta(1 + n*p_bar, 1 + n*(1-p_bar)) where n = n_eff.  Then takes the
    5th percentile of the predictive ZVF_iid distribution.

    This is a finite-sample conservative lower bound on the iid ZVF:
    the smaller n_eff is, the wider the predictive interval; we use
    n_eff = 200 (the typical per-experiment step count) so the lower
    bound stays useful.
    """
    a = 1.0 + n_eff * accuracy
    b = 1.0 + n_eff * (1.0 - accuracy)
    # Approximate the predictive 5th percentile of ZVF_iid(p) by sampling
    # p ~ Beta(a, b) and computing ZVF_iid(p).
    rng = random.Random(66)
    zs = []
    for _ in range(4000):
        # Beta(a, b) sample via two Gamma draws (Marsaglia-Tsang).
        x = _gamma(rng, a)
        y = _gamma(rng, b)
        p = x / (x + y) if (x + y) > 0 else 0.5
        zs.append(p ** G + (1.0 - p) ** G)
    zs.sort()
    idx = max(0, min(len(zs) - 1, int(q * len(zs))))
    return zs[idx]


def _gamma(rng, shape):
    """Marsaglia-Tsang gamma sampler with a hard retry cap.

    For shape < 1 we boost with the standard 1/U trick (one recursion
    level). For shape >= 1 we use Marsaglia-Tsang with a 200-attempt
    rejection cap to guard against pathological infinite loops. If we
    exceed the cap we fall back to summing shape unit-exponentials
    (slower but always terminates).
    """
    if shape <= 0:
        return 0.0
    if shape < 1.0:
        u = rng.random()
        return _gamma(rng, shape + 1.0) * (u ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    for _ in range(200):
        x = rng.gauss(0.0, 1.0)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = rng.random()
        if u < 1.0 - 0.0331 * (x ** 4):
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v
    # Fallback: sum of shape unit exponentials.
    return -sum(math.log(rng.random()) for _ in range(int(shape) + 1))


# ---------------------------------------------------------------------------
# Build outputs.
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG, exist_ok=True)
    data = load_per_step()
    methods = sorted(data.keys())
    print(f"[iter66] {len(methods)} methods, row counts:")
    for m in methods:
        print(f"  {m:<10s} {len(data[m]):>5d} rows")

    # ---- CDF overlay grid -------------------------------------------------
    grid = [round(x * 0.02, 4) for x in range(0, 51)]  # 0.00 .. 1.00 step 0.02
    cdf_rows = []
    for m in methods:
        zvfs = [r[0] for r in data[m]]
        e = ecdf(zvfs, grid)
        for x, y in zip(grid, e):
            cdf_rows.append((m, x, y))
    with open(os.path.join(RES, "zvf_iter66_cdf_overlay.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — empirical CDF of per-step ZVF per library\n")
        f.write("# Columns: method, zvf, ecdf\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method", "zvf", "ecdf"])
        for r in cdf_rows:
            w.writerow(r)

    # ---- KS distance matrix ----------------------------------------------
    print(f"[iter66] computing KS matrix for {len(methods)} methods ...", flush=True)
    ks_mat = [[0.0] * len(methods) for _ in methods]
    ks_pv = [[1.0] * len(methods) for _ in methods]
    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            if i == j:
                ks_mat[i][j] = 0.0
                ks_pv[i][j] = 1.0
                continue
            if j < i:
                ks_mat[i][j] = ks_mat[j][i]
                ks_pv[i][j] = ks_pv[j][i]
                continue
            sa = [r[0] for r in data[ma]]
            sb = [r[0] for r in data[mb]]
            D, p = ks_2samp(sa, sb)
            ks_mat[i][j] = D
            ks_mat[j][i] = D
            ks_pv[i][j] = p
            ks_pv[j][i] = p

    # Write KS matrix as TSV (header has method names).
    with open(os.path.join(RES, "zvf_iter66_ks_matrix.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — pairwise KS distance between per-step ZVF\n")
        f.write("# cells are KS distance D; p-values in zvf_iter66_ks_pvalues.tsv\n")
        f.write("# Source: platform_modal/scripts/zvf_iter66_distshape.py\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(["method_a", "method_b", "ks_D", "ks_pvalue", "n_a", "n_b"])
        for i, ma in enumerate(methods):
            for j, mb in enumerate(methods):
                if j <= i:
                    continue
                w.writerow(
                    [
                        ma,
                        mb,
                        f"{ks_mat[i][j]:.4f}",
                        f"{ks_pv[i][j]:.4g}",
                        len(data[ma]),
                        len(data[mb]),
                    ]
                )

    # ---- Hierarchical clustering -----------------------------------------
    print(f"[iter66] computing hierarchical clustering ...", flush=True)
    linkage = hclust_average(ks_mat, methods)
    print(f"[iter66] clustering done, {len(linkage)} merges", flush=True)
    with open(os.path.join(RES, "zvf_iter66_dendrogram.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — hierarchical clustering linkage\n")
        f.write("# Average-linkage on the 9x9 KS distance matrix.\n")
        f.write("# Rows are merges in chronological order; final row is the\n")
        f.write("# root (the entire 9 libraries together).\n")
        f.write("# Source: platform_modal/scripts/zvf_iter66_distshape.py\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "step",
                "merge_a",
                "merge_b",
                "height_D",
                "size_a",
                "size_b",
                "size_merged",
            ]
        )
        for k, (a, b, h, na, nb) in enumerate(linkage):
            methods_a = ",".join(
                sorted(methods[i] for i in range(len(methods)) if i == a)
            )
            methods_b = ",".join(
                sorted(methods[i] for i in range(len(methods)) if i == b)
            )
            w.writerow(
                [
                    k,
                    f"{{{methods_a}}}",
                    f"{{{methods_b}}}",
                    f"{h:.4f}",
                    na,
                    nb,
                    na + nb,
                ]
            )

    # ---- Cluster assignment at k=3 ---------------------------------------
    print(f"[iter66] cutting tree at k=3 ...", flush=True)
    n_leaves = len(methods)
    assign = cut_tree(linkage, n_leaves, k=3)
    print(f"[iter66] cut_tree done: {assign}", flush=True)
    # Quick sanity: how many distinct cluster ids?
    print(f"[iter66]   # distinct cluster ids: {len(set(assign.values()))}", flush=True)
    cluster_rows = []
    for i, m in enumerate(methods):
        sa = [r[0] for r in data[m]]
        sb_grpo = [r[0] for r in data["grpo"]]
        D_grpo, _ = ks_2samp(sa, sb_grpo)
        cluster_rows.append(
            (
                m,
                assign.get(i, -1),
                f"{statistics.mean([r[0] for r in data[m]]):.4f}",
                f"{statistics.mean([r[1] for r in data[m]]):.4f}",
                f"{D_grpo:.4f}",
            )
        )
    with open(os.path.join(RES, "zvf_iter66_cluster_assign.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — cluster assignment at k=3\n")
        f.write("# Source: platform_modal/scripts/zvf_iter66_distshape.py\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["method", "cluster", "mean_zvf", "mean_acc", "ks_to_grpo"]
        )
        for r in cluster_rows:
            w.writerow(r)

    # ---- Anti-herding quantification -------------------------------------
    print(f"[iter66] computing anti-herding signatures ...", flush=True)
    G = 8
    anti_rows = []
    for m in methods:
        zvfs = [r[0] for r in data[m]]
        accs = [r[1] for r in data[m]]
        mean_zvf = statistics.mean(zvfs)
        mean_acc = statistics.mean(accs)
        iid_zvf = iid_predicted_zvf(mean_acc, G)
        lower = beta_lower_bound(mean_acc, G, n_eff=200, q=0.05)
        # delta_div = iid - observed (positive => anti-herd, sampler
        # produces *more* contrast than independent draws).
        delta_div = iid_zvf - mean_zvf
        # Signature label: anti_herd if observed ZVF is below the 5th
        # percentile of the iid-predictive, herd if above the iid point
        # prediction, neutral otherwise. Naming follows the iter50
        # convention (zvf_antiherding_falsification.tsv) where positive
        # delta_div = anti_herd.
        if mean_zvf < lower:
            label = "anti_herd"
        elif mean_zvf > iid_zvf:
            label = "herd"
        else:
            label = "neutral"
        anti_rows.append(
            (
                m,
                f"{mean_zvf:.4f}",
                f"{mean_acc:.4f}",
                f"{iid_zvf:.4f}",
                f"{lower:.4f}",
                f"{delta_div:+.4f}",
                f"{mean_zvf - lower:+.4f}",
                label,
                len(zvfs),
            )
        )
    with open(os.path.join(RES, "zvf_iter66_anti_herding.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — anti-herding signature per library\n")
        f.write("# delta_div = zvf_iid - zvf_obs (positive = anti_herd)\n")
        f.write("# iid_pred = p^G + (1-p)^G (Binomial(G,p_bar) ZVF).\n")
        f.write("# beta_lo_05 = 5th pct of Beta(1+n*p,1+n*(1-p))-Binomial\n")
        f.write("# predictive over ZVF_iid(p), n=200.\n")
        f.write("# anti_herd = observed < beta_lo_05 (sampler anti-herds)\n")
        f.write("# herd      = observed > iid_pred (sampler over-herds)\n")
        f.write("# neutral   = within Beta posterior interval\n")
        f.write("# Source: platform_modal/scripts/zvf_iter66_distshape.py\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "method",
                "observed_zvf",
                "mean_acc",
                "iid_pred_zvf",
                "beta_lo_05",
                "delta_obs_minus_pred",
                "delta_obs_minus_lower",
                "signature",
                "n",
            ]
        )
        for r in anti_rows:
            w.writerow(r)

    # ---- Combined summary table ------------------------------------------
    with open(os.path.join(RES, "zvf_iter66_summary.tsv"), "w") as f:
        f.write("# Pillar 2 Iter 66 — combined cross-library ZVF shape summary\n")
        f.write("# Mean ZVF, mean acc, cluster id (k=3), KS to GRPO,\n")
        f.write("# observed-vs-iid delta, anti-herding label.\n")
        f.write("# Source: platform_modal/scripts/zvf_iter66_distshape.py\n")
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "method",
                "n",
                "mean_zvf",
                "mean_acc",
                "cluster_k3",
                "ks_to_grpo",
                "iid_pred_zvf",
                "delta_obs_minus_pred",
                "anti_herding_label",
            ]
        )
        cluster_lookup = {r[0]: r[1] for r in cluster_rows}
        anti_lookup = {r[0]: r for r in anti_rows}
        for m in methods:
            cl = cluster_lookup[m]
            a = anti_lookup[m]
            w.writerow(
                [
                    m,
                    a[8],
                    a[0],
                    a[2],
                    cl,
                    f"{float(ks_mat[methods.index(m)][methods.index('grpo')]):.4f}",
                    a[3],
                    a[5],
                    a[7],
                ]
            )

    # ---- Plot CDF overlay + KS heatmap -----------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle
        import numpy as np

        # Color palette: GRPO red, AERO blue, rest muted.
        pal = {
            "grpo": "#d62728",
            "aero": "#1f77b4",
            "cppo": "#2ca02c",
            "ngrpo": "#9467bd",
            "scafgrpo": "#8c564b",
            "mcgrpo": "#e377c2",
            "gift": "#7f7f7f",
            "areal": "#bcbd22",
            "es": "#17becf",
        }
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

        # ---- CDF overlay ----
        for m in methods:
            xs = []
            ys = []
            for r in cdf_rows:
                if r[0] == m:
                    xs.append(r[1])
                    ys.append(r[2])
            lw = 2.4 if m in ("grpo", "aero") else 1.2
            ax1.plot(
                xs,
                ys,
                label=m.upper(),
                color=pal.get(m, "#444444"),
                linewidth=lw,
            )
        ax1.set_xlabel("Per-step ZVF", fontsize=10)
        ax1.set_ylabel("Empirical CDF", fontsize=10)
        ax1.set_title("(a)  Per-step ZVF empirical CDF", fontsize=11)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

        # ---- KS heatmap ----
        arr = np.array(ks_mat)
        im = ax2.imshow(arr, cmap="magma", vmin=0.0, vmax=1.0)
        ax2.set_xticks(range(len(methods)))
        ax2.set_yticks(range(len(methods)))
        ax2.set_xticklabels(
            [m.upper() for m in methods], rotation=45, ha="right", fontsize=8
        )
        ax2.set_yticklabels([m.upper() for m in methods], fontsize=8)
        # Cluster boundary: order methods by cluster id, then within cluster
        # by mean_zvf, so the heatmap block structure is visible.
        cluster_for = {m: cluster_lookup[m] for m in methods}
        order = sorted(
            methods,
            key=lambda m: (cluster_for[m], -float(
                next(r for r in cluster_rows if r[0] == m)[2]
            )),
        )
        # Annotate each cell with the KS distance (top 5 only).
        for i in range(len(methods)):
            for j in range(len(methods)):
                d = arr[i, j]
                if d > 0.30:
                    ax2.text(
                        j,
                        i,
                        f"{d:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="KS distance")
        ax2.set_title("(b)  KS distance (per-step ZVF)", fontsize=11)

        # Overlay cluster rectangles (drawn around row/col ranges).
        # Build the row order used for the heatmap (default matplotlib
        # order, not the sort order above; so use methods in their
        # natural sorted order to keep cell labels aligned).
        # We add a small label listing clusters by their assigned indices.
        cluster_descr = {}
        for m, c in cluster_for.items():
            cluster_descr.setdefault(c, []).append(m)
        cluster_lines = [
            f"Cluster {c}: {', '.join(sorted(ms)).upper()}"
            for c, ms in sorted(cluster_descr.items())
]
        ax2.text(
            0.5,
            -0.34,
            "  |  ".join(cluster_lines),
            transform=ax2.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#222222",
        )

        plt.tight_layout()
        out_pdf = os.path.join(FIG, "zvf_iter66_distshape.pdf")
        out_png = os.path.join(FIG, "zvf_iter66_distshape.png")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[iter66] wrote {out_pdf}")
        print(f"[iter66] wrote {out_png}")
    except Exception as e:
        print(f"[iter66] plotting skipped: {e}")

    # ---- Print headline numbers ------------------------------------------
    print()
    print("== Iter66 headline numbers ==")
    for r in anti_rows:
        print(
            f"  {r[0]:<10s} obs={r[1]} acc={r[2]} iid={r[3]} lo05={r[4]} "
            f"d_pred={r[5]} d_lo={r[6]} {r[7]}"
        )
    print()
    print("== Cluster assignment (k=3) ==")
    for r in cluster_rows:
        print(f"  {r[0]:<10s} cluster={r[1]} mean_zvf={r[2]} ks_to_grpo={r[4]}")


if __name__ == "__main__":
    main()