#!/usr/bin/env python3
"""P5 MIN-REPORT item discriminative-power audit (iter 61, #72).

Three angles on the 7-item MIN-REPORT auditor (n=103 manifests):
  A) Multi-rater weight robustness — Dirichlet(α=2) perturbations,
     recompute badge and Spearman ρ with the canonical ranking.
  B.1) Per-item variance share on the full 103-row audit.
  B.2) Per-item × outcome Spearman ρ on the 98 mega cells (NaN if
        an item is constant on the corpus).
  C) Inter-item Pearson r on the 103-row audit.

Outputs:
  experiments/results/p5p8/p5_item_discriminative_power.{tsv,json}
  docs/p5p8_improvements/72_p5_item_discriminative_power.md  (doc-only)
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT_TSV = ROOT / "experiments" / "results" / "p5p8" / "minreport_audit.tsv"
CELLS_TSV = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"

# Canonical weight vector from scripts/p5p8/minreport_auditor.py
# (sums to 100).
CANONICAL_WEIGHTS = {
    1: 10.0,
    2: 10.0,
    3: 20.0,
    4: 20.0,
    5: 10.0,
    6: 10.0,
    7: 20.0,
}
ITEMS = list(range(1, 8))
OUTCOMES = ["mean_reward", "zvf", "pcd"]

# Per-item columns in minreport_audit.tsv (used by A/B/C)
ITEM_COL = {1: "item1_loss", 2: "item2_kl", 3: "item3_backend",
            4: "item4_zvf", 5: "item5_G", 6: "item6_heldout",
            7: "item7_decontam"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_audit_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            for k in ("item1_loss", "item2_kl", "item3_backend", "item4_zvf",
                      "item5_G", "item6_heldout", "item7_decontam", "badge"):
                row[k] = float(row[k])
            # The quick_20260704 corpus rows have empty G/temp/seed.
            row["G"] = int(row["G"]) if row.get("G", "").strip() else -1
            row["temperature"] = float(row["temperature"]) if row.get("temperature", "").strip() else -1.0
            row["seed"] = int(row["seed"]) if row.get("seed", "").strip() else -1
            rows.append(row)
    return rows


def load_cells_tsv(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            cid = row["cell_id"]
            for k in ("mean_reward", "zvf", "pcd", "mean_completion_len",
                      "std_completion_len"):
                row[k] = float(row[k])
            row["G"] = int(row["G"])
            row["temperature"] = float(row["temperature"])
            row["seed"] = int(row["seed"])
            row["n_groups"] = int(row["n_groups"])
            out[cid] = row
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Spearman rhovia rank correlation. Returns (rho, p_approx, n)."""
    n = len(xs)
    if n < 3:
        return (math.nan, math.nan, n)
    rx = rank(xs)
    ry = rank(ys)
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx2 = sum((a - mx) ** 2 for a in rx)
    dy2 = sum((b - my) ** 2 for b in ry)
    if dx2 == 0 or dy2 == 0:
        return (math.nan, math.nan, n)
    rho = num / math.sqrt(dx2 * dy2)
    # Approximate p-value via t distribution: n.b.  the MBH-M3 model is
    # good enough for ranking purposes; we only report CIs on it via
    # bootstrap below.
    if abs(rho) < 1.0 - 1e-12:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        p = 2 * (1 - normal_cdf(abs(t)))
    else:
        p = 0.0
    return (rho, p, n)


def rank(vs: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(vs), key=lambda p: (p[1], p[0]))
    ranks = [0.0] * len(vs)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg
        i = j + 1
    return ranks


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return math.nan
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx2 = sum((a - mx) ** 2 for a in xs)
    dy2 = sum((b - my) ** 2 for b in ys)
    if dx2 == 0 or dy2 == 0:
        return math.nan
    return num / math.sqrt(dx2 * dy2)


# ---------------------------------------------------------------------------
# Dirichlet (small n, 7 categories — use Gamma samples via stdlib gammavariate)
# ---------------------------------------------------------------------------

def dirichlet_sample(alpha: list[float], seed: int) -> list[float]:
    import random
    rng = random.Random(seed)
    g = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(g)
    return [x / s for x in g] if s > 0 else [1.0 / len(alpha) for _ in alpha]  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def per_item_score(row: dict, item: int) -> float:
    col = {1: "item1_loss", 2: "item2_kl", 3: "item3_backend",
           4: "item4_zvf", 5: "item5_G", 6: "item6_heldout",
           7: "item7_decontam"}[item]
    return float(row[col])


def recompute_badge(row: dict, weights: dict[int, float]) -> float:
    total = 0.0
    for it in ITEMS:
        col = ITEM_COL[it]
        total += float(row[col]) * (weights[it] / CANONICAL_WEIGHTS[it])
    return total


def main() -> int:
    audit = load_audit_tsv(AUDIT_TSV)
    cells = load_cells_tsv(CELLS_TSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Split into mega + quick subsets
    mega = [r for r in audit if r["corpus"] == "manifests"]
    quick = [r for r in audit if r["corpus"] != "manifests"]

    # For outcome-correlation: only mega cells have measured outcomes
    outcomes_per_cell: dict[str, dict[str, float]] = {}
    for r in mega:
        cid = r["cell_id"]
        if cid in cells:
            outcomes_per_cell[cid] = {
                "mean_reward": cells[cid]["mean_reward"],
                "zvf": cells[cid]["zvf"],
                "pcd": cells[cid]["pcd"],
            }

    # =====================================================================
    # (A) MULTI-RATER WEIGHT ROBUSTNESS — keep mega corpus only
    # =====================================================================
    n_boot_w = 500
    alpha = [2.0] * 7  # symmetric Dirichlet, "mild" perturbation
    canonical_badges = [r["badge"] for r in mega]
    per_item_deltas: dict[int, list[float]] = defaultdict(list)

    # Per-item weight-tilt diagnostic: how sensitive is the badge to
    # *swapping* this item's weight to a higher vs lower value?
    item_swing_acc = {it: 0.0 for it in ITEMS}
    item_swing_acc2 = {it: 0.0 for it in ITEMS}

    seed = 20260705
    rhos_perturb = []

    for b in range(n_boot_w):
        seed += 1
        w_raw = dirichlet_sample(alpha, seed)
        new_w = {it: 100 * w_raw[it - 1] for it in ITEMS}
        perturbed = [recompute_badge(r, new_w) for r in mega]
        rho_b, _, _ = spearman(canonical_badges, perturbed)
        if not math.isnan(rho_b):
            rhos_perturb.append(rho_b)
        for r in mega:
            for it in ITEMS:
                subscore = float(r[ITEM_COL[it]])
                delta = (new_w[it] - CANONICAL_WEIGHTS[it]) * subscore
                per_item_deltas[it].append(delta)
                item_swing_acc[it] += delta
                item_swing_acc2[it] += delta * delta

    # Aggregate (A): ranking robustness via percentile CI on rhos_perturb
    rhos_perturb_sorted = sorted(rhos_perturb)
    rho_mean = statistics.mean(rhos_perturb)
    rho_lo = rhos_perturb_sorted[int(0.025 * len(rhos_perturb_sorted))]
    rho_hi = rhos_perturb_sorted[int(0.975 * len(rhos_perturb_sorted))]

    a_rows = []
    a_rows.append({
        "metric": "A_ranking_rho_mean",
        "item": 0,
        "point": rho_mean,
        "ci_lo": rho_lo,
        "ci_hi": rho_hi,
        "n": len(rhos_perturb),
    })
    # Per-item weight-swing diagnostic: point estimate (mean over
    # perturbation-batches) and SD; no internal resampling needed
    # because the n_boot_w axis itself IS the bootstrap.
    for it in ITEMS:
        n = len(per_item_deltas[it])
        m = item_swing_acc[it] / n if n else math.nan
        v = max(0.0, item_swing_acc2[it] / n - m * m) if n else math.nan
        sd = math.sqrt(v)
        a_rows.append({
            "metric": "A_weight_swing_sd",
            "item": it,
            "point": m,
            "ci_lo": m - 1.96 * sd,
            "ci_hi": m + 1.96 * sd,
            "n": n,
        })

    # (B) PER-ITEM DISCRIMINATIVE POWER
    #   B.1: variance share on the full 103-row audit.
    #   B.2: bootstrap Spearman ρ per-item × outcome on the 98 mega cells
    #        (NaN where the item is constant on the mega corpus).
    n_boot_b = 1000
    b_rows = []

    # B.1 Variance-share diagnostic across full audit (n=103)
    item_var = {}
    item_n_unique = {}
    for it in ITEMS:
        col = ITEM_COL[it]
        vals = [float(r[col]) for r in audit]
        item_var[it] = statistics.pvariance(vals)
        item_n_unique[it] = len(set(vals))
    total_var = sum(item_var.values())
    for it in ITEMS:
        share = item_var[it] / total_var if total_var > 0 else 0.0
        b_rows.append({
            "metric": "B1_var_share_full103",
            "item": it,
            "point": share,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "n": len(audit),
        })
        b_rows.append({
            "metric": "B1_n_unique_values",
            "item": it,
            "point": float(item_n_unique[it]),
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "n": len(audit),
        })

    # B.2 Per-item × outcome Spearman ρ on the 98 mega cells.
    import random
    rng_b = random.Random(20260705)
    def bootstrap_rho(xs, ys, idx_seed):
        n = len(xs)
        r_pt, _, _ = spearman(xs, ys)
        rng = random.Random(idx_seed)
        rhos = sorted(spearman([xs[i] for i in idx := [rng.randrange(n) for _ in range(n)]],
                                [ys[i] for i in idx])[0]
                       for _ in range(n_boot_b))
        rhos = [r for r in rhos if not math.isnan(r)]
        if not rhos:
            return r_pt, math.nan, math.nan
        return r_pt, rhos[int(0.025*len(rhos))], rhos[int(0.975*len(rhos))]
    xs_per_item = {it: [float(r[ITEM_COL[it]]) for r in mega
                       if r["cell_id"] in outcomes_per_cell]
                   for it in ITEMS}
    ys_per_outcome = {o: [outcomes_per_cell[r["cell_id"]][o] for r in mega
                          if r["cell_id"] in outcomes_per_cell]
                      for o in OUTCOMES}
    for it in ITEMS:
        xs = xs_per_item[it]
        n = len(xs)
        if n < 5:
            continue
        if len(set(xs)) <= 1:
            for outcome in OUTCOMES:
                b_rows.append({"metric": f"B2_spearman_{outcome}_mega",
                               "item": it, "point": math.nan,
                               "ci_lo": math.nan, "ci_hi": math.nan, "n": n})
            continue
        for outcome in OUTCOMES:
            r_pt, lo, hi = bootstrap_rho(xs, ys_per_outcome[outcome],
                                          20260705 + it*31 + OUTCOMES.index(outcome))
            b_rows.append({"metric": f"B2_spearman_{outcome}_mega",
                           "item": it, "point": r_pt,
                           "ci_lo": lo, "ci_hi": hi, "n": n})

    # (C) INTER-ITEM REDUNDANCY — Pearson r between every pair of items
    # =====================================================================
    c_rows = []
    subscore_cols = {1: "item1_loss", 2: "item2_kl", 3: "item3_backend",
                     4: "item4_zvf", 5: "item5_G", 6: "item6_heldout",
                     7: "item7_decontam"}
    item_vectors = {it: [per_item_score(r, it) for r in audit]
                    for it in ITEMS}
    for i in ITEMS:
        for j in ITEMS:
            if j <= i:
                continue
            r = pearson(item_vectors[i], item_vectors[j])
            c_rows.append({
                "metric": "C_pearson_pair",
                "item": i * 10 + j,
                "point": r,
                "ci_lo": math.nan,
                "ci_hi": math.nan,
                "n": len(audit),
            })

    # Write tsv
    out_rows = a_rows + b_rows + c_rows
    cols = ["metric", "item", "point", "ci_lo", "ci_hi", "n"]
    out_tsv = OUT_DIR / "p5_item_discriminative_power.tsv"
    with out_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    # =====================================================================
    # Summary json
    # =====================================================================
    summary = {
        "n_manifests_total": len(audit),
        "n_mega": len(mega),
        "n_quick": len(quick),
        "n_outcome_mega_cells": len(outcomes_per_cell),
        "A_weight_robustness": {
            "n_perturbations": n_boot_w,
            "ranking_rho_mean": rho_mean,
            "ranking_rho_ci_lo": rho_lo,
            "ranking_rho_ci_hi": rho_hi,
            "interpretation": (
                "robust" if rho_mean > 0.95 else
                "moderate" if rho_mean > 0.85 else "fragile"
            ),
        },
        "B1_var_share_full103": {
            it: {
                "share": next(r for r in b_rows if r["metric"] == "B1_var_share_full103" and r["item"] == it)["point"],
                "n_unique": int(next(r for r in b_rows if r["metric"] == "B1_n_unique_values" and r["item"] == it)["point"]),
            }
            for it in ITEMS
        },
        "B2_outcome_correlation_mega": {
            it: {
                outcome: {
                    "rho": next((r for r in b_rows if r["metric"] == f"B2_spearman_{outcome}_mega" and r["item"] == it), {"point": math.nan})["point"],
                    "ci_lo": next((r for r in b_rows if r["metric"] == f"B2_spearman_{outcome}_mega" and r["item"] == it), {"ci_lo": math.nan})["ci_lo"],
                    "ci_hi": next((r for r in b_rows if r["metric"] == f"B2_spearman_{outcome}_mega" and r["item"] == it), {"ci_hi": math.nan})["ci_hi"],
                }
                for outcome in OUTCOMES
            }
            for it in ITEMS
        },
        "C_inter_item_pearson": [{"item_i": i, "item_j": j, "r": r["point"]}
                                for i in ITEMS for j in ITEMS if j > i
                                for r in c_rows if r["item"] == i*10+j],
    }
    out_json = OUT_DIR / "p5_item_discriminative_power_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"Iter 61 P5 #72 audit rows={len(audit)} (mega={len(mega)}, quick={len(quick)}) outcome_cells={len(outcomes_per_cell)}")
    print(f"(A) ranking Spearman rho B={n_boot_w} perturbations: {rho_mean:.4f} [{rho_lo:.4f}, {rho_hi:.4f}] -- {summary['A_weight_robustness']['interpretation']}")
    for it in ITEMS:
        sv = next(r for r in b_rows if r["metric"] == "B1_var_share_full103" and r["item"] == it)
        su = next(r for r in b_rows if r["metric"] == "B1_n_unique_values" and r["item"] == it)
        def rho(outcome):
            rs = [r for r in b_rows if r["metric"] == f"B2_spearman_{outcome}_mega" and r["item"] == it]
            if not rs or math.isnan(rs[0]['point']):
                return "n/a"
            return f"{rs[0]['point']:+.3f}[{rs[0]['ci_lo']:+.3f},{rs[0]['ci_hi']:+.3f}]"
        print(f"  item {it}: share={sv['point']:.4f} unique={int(su['point'])}  rho(mean_r/zvf/pcd)={rho('mean_reward')}/{rho('zvf')}/{rho('pcd')}")
    print(f"Wrote {out_tsv.relative_to(ROOT)} ({len(out_rows)} rows) and {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
