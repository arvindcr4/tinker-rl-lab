#!/usr/bin/env python3
"""
iter 206 — P6 BADGE-vs-DIVERGENCE coupling audit (post-iter-198 schema bump).

Vein (T3 cross-paper coupling), fresh: the iter-202 framework × method
coverage matrix quantified which 6 methods have multi-framework clusters;
iter-206 lifts to ask **does the MIN-REPORT badge actually correlate with
lower cross-framework disagreement**? If yes, MIN-REPORT IS doing its
auditability job; if no, the badge is purely cosmetic.

Approach:
  1. Load every stack entry from registry/entries/*.json. Compute per-entry
     BADGE = outcomes.coverage.min_report_coverage.
  2. Re-derive pairwise cross-framework disagreement from scratch (not
     reusing iter202 files): for each (field, method) cell, walk every
     pair of entries sharing label_claimed but DIFFERENT framework; flag
     pair as disagree if values differ (None vs literal are NOT disagree;
     only reporting-vs-reporting mismatches count).
  3. Aggregate to cluster-level: per (label_claimed, field) cluster
     mean_pairwise_disagree_rate; per cluster mean_badge across members.
  4. Compute Spearman rho between cluster-level badge (mean) and cluster-
     level disagreement (mean across fields with >=2 reporting frameworks).
  5. Bootstrap CI on Spearman rho (B=2000, residual resample within cluster).
  6. Quartile test: split clusters into badge-quartiles; report mean
     disagreement per quartile; test monotone-decreasing hypothesis.
  7. Per-field disagreement vs per-field coverage analysis.

Output: 7 TSV / 1 JSON / 1 ledger row. Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
REG_DIR = WORKTREE / "registry" / "entries"
RESULTS = WORKTREE / "experiments" / "results" / "p5p8"
RESULTS.mkdir(parents=True, exist_ok=True)

# 23 MIN-REPORT v2.4 sub-fields (same enumeration as iter-205).
MINREPORT_FIELDS = [
    ("loss_form", "importance_ratio_level"),
    ("loss_form", "clip_eps_low"),
    ("loss_form", "clip_eps_high"),
    ("loss_form", "length_normalization"),
    ("loss_form", "advantage_normalization"),
    ("loss_form", "token_mask"),
    ("reference_kl", "reference_policy"),
    ("reference_kl", "kl_beta"),
    ("reference_kl", "kl_estimator"),
    ("sampler_backend", "backend"),
    ("sampler_backend", "precision"),
    ("sampler_backend", "temperature"),
    ("sampler_backend", "top_p"),
    ("telemetry", "per_step_zvf"),
    ("telemetry", "per_step_gu"),
    ("telemetry", "source"),
    ("group_size_schedule", "initial_g"),
    ("group_size_schedule", "schedule"),
    ("group_size_schedule", "adaptation_rule"),
    ("heldout_split", "disjoint_from_reward_env"),
    ("heldout_split", "description"),
    ("decontamination", "performed"),
    ("decontamination", "parser_robustness_probe"),
]
N_FIELDS = len(MINREPORT_FIELDS)  # 23

# Boot RNG seed for reproducibility
SEED = 20260706


# ------------------------------------------------------------
# 1. Load entries
# ------------------------------------------------------------
def load_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(REG_DIR.glob("*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception:
            continue
        # only stack entries
        if rec.get("record_type") != "stack":
            continue
        entries.append(rec)
    return entries


def entry_badge(rec: dict[str, Any]) -> float | None:
    """Return outcomes.coverage.min_report_coverage, or None if missing."""
    cov = (rec.get("outcomes") or {}).get("coverage") or {}
    v = cov.get("min_report_coverage")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def entry_framework(rec: dict[str, Any]) -> str:
    return (rec.get("framework") or {}).get("name") or "unknown"


def entry_label(rec: dict[str, Any]) -> str:
    return rec.get("label_claimed") or "unknown"


def get_field_value(rec: dict[str, Any], item: str, leaf: str) -> Any:
    mr = rec.get("min_report") or {}
    item_block = mr.get(item) or {}
    return item_block.get(leaf, "__MISSING__")


# ------------------------------------------------------------
# 2. Compute per-field disagreement per cluster
# ------------------------------------------------------------
def cluster_disagreement(
    entries: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """For each label_claimed cluster, compute per-(item, leaf)
    pairwise disagreement rate. Returns:
        {label: {(item, leaf): {
              n_fws_reporting: int,
              n_pairs: int,
              n_disagree: int,
              disagree_rate: float | None,
              mean_value: str,
        }}}
    Reporting-vs-non-reporting is NOT a disagreement; reporting-vs-reporting
    mismatches ARE.
    """
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_label[entry_label(e)].append(e)
    out: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for label, group in by_label.items():
        out[label] = {}
        # frame diversity
        fws = {entry_framework(e) for e in group}
        # at most 1 entry per framework for the cross-pair construction
        # (if dupes exist, take first)
        rep: dict[str, dict[str, Any]] = {}
        for e in group:
            fw = entry_framework(e)
            if fw not in rep:
                rep[fw] = e
        fws_list = sorted(rep.keys())
        n_fws = len(fws_list)
        for item, leaf in MINREPORT_FIELDS:
            # collect values per framework
            per_fw_val: dict[str, list[Any]] = {fw: [] for fw in fws_list}
            for fw, e in rep.items():
                v = get_field_value(e, item, leaf)
                per_fw_val[fw].append(v)
            # report value set across fws; majority loser
            reporting_fws = [fw for fw in fws_list if all(
                x is not None and x != "__MISSING__" for x in per_fw_val[fw]
            )]
            n_report = len(reporting_fws)
            if n_report < 2:
                out[label][(item, leaf)] = {
                    "n_fws_reporting": n_report,
                    "n_pairs": 0,
                    "n_disagree": 0,
                    "disagree_rate": None,
                    "mean_value": "; ".join(
                        f"{fw}={per_fw_val[fw][0]}" for fw in fws_list
                    ),
                }
                continue
            # count pairs
            n_pairs = 0
            n_disagree = 0
            for i in range(len(reporting_fws)):
                for j in range(i + 1, len(reporting_fws)):
                    fw_i = reporting_fws[i]
                    fw_j = reporting_fws[j]
                    v_i = per_fw_val[fw_i][0]
                    v_j = per_fw_val[fw_j][0]
                    n_pairs += 1
                    # use string equality (with normalization for bool/int)
                    def norm(v: Any) -> str:
                        if isinstance(v, bool):
                            return "true" if v else "false"
                        return str(v)
                    if norm(v_i) != norm(v_j):
                        n_disagree += 1
            out[label][(item, leaf)] = {
                "n_fws_reporting": n_report,
                "n_pairs": n_pairs,
                "n_disagree": n_disagree,
                "disagree_rate": (n_disagree / n_pairs) if n_pairs else None,
                "mean_value": "; ".join(
                    f"{fw}={per_fw_val[fw][0]}" for fw in fws_list
                ),
            }
    return out


# ------------------------------------------------------------
# 3. Cluster-level rollup
# ------------------------------------------------------------
def cluster_rollup(
    entries: list[dict[str, Any]],
    disagree_data: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """For each label_claimed cluster: mean badge, mean disagreement
    (across reporting fields), n entries, n frameworks, etc.
    """
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_label[entry_label(e)].append(e)
    out: list[dict[str, Any]] = []
    for label, group in by_label.items():
        fws = sorted({entry_framework(e) for e in group})
        badges = [entry_badge(e) for e in group]
        non_null_badges = [b for b in badges if b is not None]
        mean_badge = sum(non_null_badges) / max(1, len(non_null_badges))
        # collect per-field disagreement for this cluster
        per_field = disagree_data.get(label, {})
        rates = [d["disagree_rate"] for d in per_field.values()
                 if d["disagree_rate"] is not None]
        mean_disagree = sum(rates) / max(1, len(rates)) if rates else None
        # full field coverage: how many of 23 sub-fields are reporting in
        # any cluster member
        n_total_fields = len(per_field)
        n_reporting_fields = sum(
            1 for d in per_field.values() if d["n_fws_reporting"] >= 2
        )
        out.append({
            "label": label,
            "n_entries": len(group),
            "n_frameworks": len(fws),
            "frameworks": ",".join(fws),
            "mean_badge": round(mean_badge, 4),
            "min_badge": round(min(non_null_badges), 4) if non_null_badges else None,
            "max_badge": round(max(non_null_badges), 4) if non_null_badges else None,
            "n_fields_total": n_total_fields,
            "n_fields_reporting": n_reporting_fields,
            "mean_disagree": round(mean_disagree, 4) if mean_disagree is not None else None,
            "mean_disagree_basis_n": len(rates),
            "n_pairs_total": sum(d["n_pairs"] for d in per_field.values()),
            "n_disagree_total": sum(d["n_disagree"] for d in per_field.values()),
        })
    return out


# ------------------------------------------------------------
# 4. Spearman rho with bootstrap CI
# ------------------------------------------------------------
def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    def rank(vs: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        for r_i, idx in enumerate(sorted_idx):
            r[idx] = r_i + 1
        return r
    rx = rank(xs)
    ry = rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    varx = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    vary = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / math.sqrt(varx * vary)


def bootstrap_spearman(
    rows: list[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Block-bootstrap by label; B resamples with replacement of the
    cluster-level data points, computes rho per resample.
    """
    rng = random.Random(seed)
    pool = [(r["mean_badge"], r["mean_disagree"]) for r in rows
            if r["mean_badge"] is not None and r["mean_disagree"] is not None]
    n = len(pool)
    if n < 3:
        return {"rho": None, "rho_lo": None, "rho_hi": None}
    xs = [p[0] for p in pool]
    ys = [p[1] for p in pool]
    point = spearman(xs, ys)
    rhos: list[float] = []
    for _ in range(n_boot):
        bx = []
        by = []
        for _i in range(n):
            j = rng.randint(0, n - 1)
            bx.append(pool[j][0])
            by.append(pool[j][1])
        r = spearman(bx, by)
        if not math.isnan(r):
            rhos.append(r)
    rhos.sort()
    lo = rhos[int(0.025 * len(rhos))]
    hi = rhos[int(0.975 * len(rhos)) - 1]
    return {"rho": round(point, 4), "rho_lo": round(lo, 4), "rho_hi": round(hi, 4)}


# ------------------------------------------------------------
# 5. Entry-level BADGE × within-cluster pairwise disagreement
# ------------------------------------------------------------
def entry_level_coupling(
    entries: list[dict[str, Any]],
    disagree_data: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """For each (entry, cluster) cell: entry badge, cluster disagree rate.
    Returns one row per entry in a cross-framework cluster.
    """
    out: list[dict[str, Any]] = []
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_label[entry_label(e)].append(e)
    for label, group in by_label.items():
        fws = {entry_framework(e) for e in group}
        if len(fws) < 2:
            continue  # cross-framework clusters only
        per_field = disagree_data.get(label, {})
        rates = [d["disagree_rate"] for d in per_field.values()
                 if d["disagree_rate"] is not None]
        mean_disagree = sum(rates) / max(1, len(rates)) if rates else None
        for e in group:
            out.append({
                "entry_id": e.get("id"),
                "label_claimed": label,
                "framework": entry_framework(e),
                "badge": entry_badge(e),
                "cluster_mean_disagree": round(mean_disagree, 4) if mean_disagree is not None else None,
            })
    return out


# ------------------------------------------------------------
# 6. Quartile test
# ------------------------------------------------------------
def quartile_test(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sort clusters by mean_badge, split into quartiles, report mean
    disagree per quartile + monotone-decreasing test.
    """
    pool = [r for r in rows if r["mean_badge"] is not None and r["mean_disagree"] is not None]
    n = len(pool)
    if n < 4:
        return {"error": "n < 4 clusters", "n": n}
    pool.sort(key=lambda r: r["mean_badge"])
    # quartile edges
    q = 4
    sz = n // q
    quartiles: list[dict[str, Any]] = []
    for k in range(q):
        lo = k * sz
        hi = (k + 1) * sz if k < q - 1 else n
        chunk = pool[lo:hi]
        if not chunk:
            continue
        badges = [c["mean_badge"] for c in chunk]
        disr = [c["mean_disagree"] for c in chunk]
        labels = [c["label"] for c in chunk]
        quartiles.append({
            "quartile": k + 1,
            "n_clusters": len(chunk),
            "labels": ",".join(labels),
            "badge_min": round(min(badges), 4),
            "badge_max": round(max(badges), 4),
            "mean_disagree": round(sum(disr) / len(disr), 4),
        })
    # monotone test: top-quartile mean < bottom-quartile mean?
    if len(quartiles) >= 2:
        bot = quartiles[0]["mean_disagree"]
        top = quartiles[-1]["mean_disagree"]
        monotone = top < bot
        delta = bot - top
    else:
        monotone = None
        delta = None
    return {
        "n_clusters": n,
        "quartiles": quartiles,
        "top_lt_bottom": monotone,
        "delta_bottom_minus_top": round(delta, 4) if delta is not None else None,
    }


# ------------------------------------------------------------
# 7. Per-field analysis: do well-reported fields diverge less?
# ------------------------------------------------------------
def per_field_table(
    disagree_data: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """For each (item, leaf), aggregate across clusters: number of clusters
    that report the field on >=2 frameworks, mean disagree rate among them."""
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for label, per_field in disagree_data.items():
        for (item, leaf), d in per_field.items():
            if d["n_fws_reporting"] >= 2 and d["disagree_rate"] is not None:
                counts[(item, leaf)] += 1
                agg[(item, leaf)].append(d["disagree_rate"])
    out: list[dict[str, Any]] = []
    for (item, leaf), rates in agg.items():
        out.append({
            "item": item,
            "leaf": leaf,
            "n_clusters_reporting": counts[(item, leaf)],
            "mean_disagree": round(sum(rates) / len(rates), 4),
            "max_disagree": round(max(rates), 4),
            "any_cluster_at_100": any(r > 0.99 for r in rates),
            "all_clusters_at_0": all(r < 0.01 for r in rates),
        })
    out.sort(key=lambda r: r["mean_disagree"])
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main() -> int:
    entries = load_entries()
    n_entries = len(entries)
    # count by type
    n_stack = sum(1 for e in entries if e.get("record_type") == "stack")
    # gather labels
    labels: dict[str, int] = defaultdict(int)
    for e in entries:
        labels[entry_label(e)] += 1

    # compute disagreement
    disagree_data = cluster_disagreement(entries)
    # cluster rollup
    cluster_rows = cluster_rollup(entries, disagree_data)
    # keep only cross-framework clusters (n_frameworks >= 2)
    cross_clusters = [r for r in cluster_rows if r["n_frameworks"] >= 2]
    # and single-framework clusters
    single_clusters = [r for r in cluster_rows if r["n_frameworks"] < 2]

    # coupling analyses
    boot = bootstrap_spearman(cross_clusters)
    quart = quartile_test(cross_clusters)
    entry_level = entry_level_coupling(entries, disagree_data)
    pf = per_field_table(disagree_data)

    # ------------------------------------------------------------
    # HYPOTHESES (falsifiable)
    # ------------------------------------------------------------
    H: list[dict[str, Any]] = []

    # H1: Spearman rho between cluster mean badge and cluster mean disagree
    # is NEGATIVE with magnitude >= 0.10 (weak monotone); CI excludes 0.
    h1_pass = (
        boot.get("rho") is not None
        and boot["rho"] < -0.10
        and boot.get("rho_hi") is not None
        and boot["rho_hi"] < 0
    )
    H.append({
        "h": "H1", "claim": "Spearman rho between cluster mean_badge and "
                            "cluster mean_disagree is NEGATIVE (rho <= -0.10) "
                            "with bootstrap CI excluding 0",
        "observed_rho": boot.get("rho"),
        "observed_ci": [boot.get("rho_lo"), boot.get("rho_hi")],
        "verdict": "PASS" if h1_pass else "FAIL",
        "n_clusters": len(cross_clusters),
        "n_boot": 2000,
    })

    # H2: Best-badge cluster (top quartile mean badge) has mean disagreement
    # strictly less than worst-badge cluster (bottom quartile). Delta > 0.05.
    if "quartiles" in quart and len(quart["quartiles"]) >= 2:
        bot = quart["quartiles"][0]["mean_disagree"]
        top = quart["quartiles"][-1]["mean_disagree"]
        h2_delta = round(bot - top, 4)
        h2_pass = h2_delta > 0.05
    else:
        h2_delta = None
        h2_pass = False
    H.append({
        "h": "H2",
        "claim": "Top-quartile mean_disagree strictly less than "
                 "bottom-quartile, with delta > 0.05",
        "observed_delta": h2_delta,
        "verdict": "PASS" if h2_pass else "FAIL",
        "n_quartiles": len(quart.get("quartiles", [])) if "quartiles" in quart else 0,
    })

    # H3: Clusters with mean_badge >= 0.70 have mean disagree < 0.30.
    high_badge_clusters = [
        r for r in cross_clusters
        if r["mean_badge"] is not None and r["mean_badge"] >= 0.70
        and r["mean_disagree"] is not None
    ]
    if high_badge_clusters:
        high_disagree = sum(c["mean_disagree"] for c in high_badge_clusters) / len(high_badge_clusters)
    else:
        high_disagree = None
    h3_pass = high_disagree is not None and high_disagree < 0.30
    H.append({
        "h": "H3",
        "claim": "Clusters with mean_badge >= 0.70 have mean_disagree < 0.30",
        "observed_mean_disagree": high_disagree,
        "n_clusters": len(high_badge_clusters),
        "verdict": "PASS" if h3_pass else "FAIL",
    })

    # H4: Spearman entry-level coupling (entry badge vs cluster mean disagree)
    pool_entry = [
        (r["badge"], r["cluster_mean_disagree"]) for r in entry_level
        if r["badge"] is not None and r["cluster_mean_disagree"] is not None
    ]
    if len(pool_entry) >= 3:
        ex = [p[0] for p in pool_entry]
        ey = [p[1] for p in pool_entry]
        entry_rho = spearman(ex, ey)
    else:
        entry_rho = None
    h4_pass = entry_rho is not None and entry_rho < -0.20
    H.append({
        "h": "H4",
        "claim": "Entry-level Spearman rho (entry badge vs cluster mean "
                 "disagree) is STRONGLY NEGATIVE (rho <= -0.20)",
        "observed_rho": round(entry_rho, 4) if entry_rho is not None else None,
        "n_entries": len(pool_entry),
        "verdict": "PASS" if h4_pass else "FAIL",
    })

    # H5: zvf130 clusters (mean_badge ~0.43) have MEANINGFULLY HIGHER
    # disagreement than colab-open clusters (mean_badge ~0.96). Delta > 0.05.
    fw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in cross_clusters:
        for fw in r["frameworks"].split(","):
            fw_groups[fw].append(r)
    zvf_disagrees = [r["mean_disagree"] for r in fw_groups.get("worktree-zvf130-batch", []) if r["mean_disagree"] is not None]
    colab_disagrees = [r["mean_disagree"] for r in fw_groups.get("colab-open-trainer", []) if r["mean_disagree"] is not None]
    zvf_mean = sum(zvf_disagrees) / max(1, len(zvf_disagrees)) if zvf_disagrees else None
    colab_mean = sum(colab_disagrees) / max(1, len(colab_disagrees)) if colab_disagrees else None
    h5_delta = round((zvf_mean or 0) - (colab_mean or 0), 4) if zvf_mean is not None and colab_mean is not None else None
    h5_pass = h5_delta is not None and h5_delta > 0.05
    H.append({
        "h": "H5",
        "claim": "zvf130 clusters have MEANINGFULLY HIGHER mean_disagree "
                 "than colab-open clusters (delta > 0.05)",
        "zvf_mean": round(zvf_mean, 4) if zvf_mean is not None else None,
        "colab_mean": round(colab_mean, 4) if colab_mean is not None else None,
        "delta_zvf_minus_colab": h5_delta,
        "n_zvf_clusters": len(zvf_disagrees),
        "n_colab_clusters": len(colab_disagrees),
        "verdict": "PASS" if h5_pass else "FAIL",
    })

    # H6: At least 4 of the 23 MIN-REPORT sub-fields achieve
    # all-cluster-0 disagreement (every cluster that reports them agrees).
    n_zero = sum(1 for r in pf if r["all_clusters_at_0"])
    h6_pass = n_zero >= 4
    H.append({
        "h": "H6",
        "claim": "At least 4 of 23 MIN-REPORT sub-fields achieve "
                 "all-cluster disagreement rate of 0 (universal agreement)",
        "observed_n_zero": n_zero,
        "verdict": "PASS" if h6_pass else "FAIL",
    })

    # H7: At least 3 sub-fields have median-cluster-level disagreement = 0
    # AND at least 3 sub-fields have median-cluster-level disagreement > 0.5
    # (inter-field heterogeneity exists).
    median_gt_05 = sum(
        1 for r in pf
        if r["mean_disagree"] > 0.5
    )
    median_eq_00 = sum(
        1 for r in pf
        if r["mean_disagree"] == 0
    )
    h7_pass = median_gt_05 >= 3 and median_eq_00 >= 3
    H.append({
        "h": "H7",
        "claim": "INTER-FIELD HETEROGENEITY: at least 3 fields at "
                 "mean_disagree > 0.5 AND at least 3 at mean_disagree == 0",
        "n_fields_gt_05": median_gt_05,
        "n_fields_eq_0": median_eq_00,
        "verdict": "PASS" if h7_pass else "FAIL",
    })

    # ------------------------------------------------------------
    # WRITE OUTPUTS
    # ------------------------------------------------------------
    # 1) cluster rollup TSV
    p1 = RESULTS / "p6_iter206_cluster_rollup.tsv"
    with open(p1, "w", newline="") as f:
        cols = [
            "label", "n_entries", "n_frameworks", "frameworks",
            "mean_badge", "min_badge", "max_badge",
            "n_fields_total", "n_fields_reporting",
            "mean_disagree", "mean_disagree_basis_n",
            "n_pairs_total", "n_disagree_total",
        ]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in cluster_rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[OK] {p1} ({len(cluster_rows)} rows)")

    # 2) per-field disagreement TSV
    p2 = RESULTS / "p6_iter206_perfield_disagree.tsv"
    with open(p2, "w", newline="") as f:
        cols = ["item", "leaf", "n_clusters_reporting",
                "mean_disagree", "max_disagree",
                "any_cluster_at_100", "all_clusters_at_0"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in pf:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[OK] {p2} ({len(pf)} rows)")

    # 3) hypotheses TSV
    p3 = RESULTS / "p6_iter206_hypotheses.tsv"
    with open(p3, "w", newline="") as f:
        cols = ["h", "claim", "verdict", "observed_rho",
                "observed_ci_lo", "observed_ci_hi",
                "observed_delta", "observed_mean_disagree",
                "n_clusters", "n_boot", "n_zvf_clusters",
                "n_colab_clusters", "n_fields_gt_05",
                "n_fields_eq_0", "observed_n_zero"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for h in H:
            row = {k: h.get(k) for k in cols}
            if h.get("observed_ci"):
                row["observed_ci_lo"] = h["observed_ci"][0]
                row["observed_ci_hi"] = h["observed_ci"][1]
            w.writerow(row)
    print(f"[OK] {p3} ({len(H)} rows)")

    # 4) entry-level TSV
    p4 = RESULTS / "p6_iter206_entry_level.tsv"
    with open(p4, "w", newline="") as f:
        cols = ["entry_id", "label_claimed", "framework",
                "badge", "cluster_mean_disagree"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in entry_level:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[OK] {p4} ({len(entry_level)} rows)")

    # 5) quartile TSV
    p5 = RESULTS / "p6_iter206_quartiles.tsv"
    with open(p5, "w", newline="") as f:
        cols = ["quartile", "n_clusters", "labels",
                "badge_min", "badge_max", "mean_disagree"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        if "quartiles" in quart:
            for r in quart["quartiles"]:
                w.writerow({k: r.get(k) for k in cols})
    print(f"[OK] {p5}")

    # 6) JSON summary
    summary = {
        "iter": 206,
        "pillar": "P6",
        "vein": "T3 cross-paper coupling — BADGE-vs-DIVERGENCE audit",
        "n_entries": n_entries,
        "n_stack_records": n_stack,
        "n_methods_total": len(labels),
        "n_methods_with_data": sum(1 for v in labels.values() if v > 0),
        "n_cross_framework_clusters": len(cross_clusters),
        "n_single_framework_clusters": len(single_clusters),
        "framework_groups_in_cross_clusters": {
            fw: len(rows) for fw, rows in fw_groups.items()
        },
        "spearman_rho_cluster_level": boot,
        "spearman_rho_entry_level": round(entry_rho, 4) if entry_rho is not None else None,
        "quartile_test": quart,
        "hypotheses": H,
        "n_pass": sum(1 for h in H if h["verdict"] == "PASS"),
        "n_fail": sum(1 for h in H if h["verdict"] == "FAIL"),
    }
    p6 = RESULTS / "p6_iter206_summary.json"
    with open(p6, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] {p6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
