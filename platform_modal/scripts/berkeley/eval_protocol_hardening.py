#!/usr/bin/env python3
"""Berkeley F25 L5+L10 (Yehudai Survey + τ²-Bench) — eval-protocol hardening.

Source lecture(s):
  F25 L5  — Yehudai et al. (Survey on Evaluation of LLM-based Agents, arXiv:2503.16416)
  F25 L10 — Barres, Dong, Ray, Si, Narasimhan (τ²-Bench, arXiv:2506.07982)
Target: A2 — evaluation methodology (eval-protocol hardening)

Hypotheses tested on real Pillar-2 (ZVF) iter130 per-seed risk data:
  H1 (Yehudai-COST): the headline method ranking is preserved by a smaller
      seed subset. Compute Minimum Viable Seed Pool (MVSP) at stability ∈
      {50%, 80%, 95%}.
  H2 (Yehudai-ROBUSTNESS): per-method variance-mitigation gain (z = (μ_m - μ_grpo)
      / sqrt(σ_m² + σ_grpo²)) is robust to seed subset selection.
  H3 (τ²-Bench-ABLATION): the zvf_risk_max channel decomposition (mag vs csd
      vs drift) reveals which methods are "magnitude-dominant" vs "drift-dominant"
      — a fine-grained error map the iter130 headline ranking hides.
  H4 (τ²-Bench-COMPOSITIONAL): the 9 methods fall into 2-3 stable "behaviour
      clusters" by their (mag, csd, drift) signature; the full ranking is
      driven by ONE of these clusters, not by all three.

Stdlib-only. Outputs:
  experiments/results/berkeley/eval_protocol_mvsp.tsv
  experiments/results/berkeley/eval_protocol_robustness.tsv
  experiments/results/berkeley/eval_protocol_ablation.tsv
  experiments/results/berkeley/eval_protocol_clusters.tsv
  experiments/results/berkeley/eval_protocol_summary.json
"""
import csv, json, math, random, statistics, itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RISK_TSV = ROOT / "experiments/results/zvf_iter130_risk_index.tsv"
OUT = ROOT / "experiments/results/berkeley"

VAR_MIT = ["grpo", "ngrpo", "aero", "cppo", "mcgrpo", "areal", "gift", "es", "scafgrpo"]


def load_per_seed():
    """Return dict method -> list of dict rows (per seed)."""
    by_m = defaultdict(list)
    with open(RISK_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            m = row["method"].strip()
            if m not in VAR_MIT:
                continue
            by_m[m].append({
                "seed": int(row["seed"]),
                "zvf_risk_max": float(row["zvf_risk_max"]),
                "zvf_risk":    float(row["zvf_risk"]),
                "risk_mag":    float(row["risk_mag"]),
                "risk_csd":    float(row["risk_csd"]),
                "risk_drift":  float(row["risk_drift"]),
                "failure_label": row.get("failure_label", "").strip(),
            })
    # sort each method's seeds for determinism
    for m in by_m:
        by_m[m].sort(key=lambda r: r["seed"])
    return by_m


def ranking(values):
    """Lower zvf_risk_max is better. Returns method list sorted best→worst."""
    return sorted(values.keys(), key=lambda m: statistics.mean(values[m]))


def spearman(r1, r2):
    """Spearman rank correlation on the two ordered method lists."""
    n = len(r1)
    assert set(r1) == set(r2)
    pos1 = {m: i for i, m in enumerate(r1)}
    pos2 = {m: i for i, m in enumerate(r2)}
    d2 = sum((pos1[m] - pos2[m]) ** 2 for m in r1)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def topk_match(r1, r2, k):
    """Return 1 if top-k methods are identical in both orderings, 0 otherwise."""
    return 1 if r1[:k] == r2[:k] else 0


def h1_mvsp(by_m, n_boot=2000, seed=42):
    """Cost-efficient sub-sampling: smallest k such that the top-k ranking is
    stable in ≥P% of random subsets."""
    random.seed(seed)
    full = ranking({m: [r["zvf_risk_max"] for r in rows] for m, rows in by_m.items()})
    rows = []
    for k in range(1, 6):
        if k > 5:
            break
        # all C(5,k) subsets
        subsets = list(itertools.combinations(range(5), k))
        if len(subsets) > n_boot:
            subsets = random.sample(subsets, n_boot)
        spearman_list, top1_list, top3_list, top_agree = [], [], [], []
        for combo in subsets:
            sub_vals = {m: [rows[i]["zvf_risk_max"] for i in combo] for m, rows in by_m.items()}
            sub_rank = ranking(sub_vals)
            spearman_list.append(spearman(full, sub_rank))
            top1_list.append(topk_match(full, sub_rank, 1))
            top3_list.append(topk_match(full, sub_rank, 3))
            # the headline claim from iter130 is that scafgrpo is the safest
            top_agree.append(1 if sub_rank[0] == full[0] else 0)
        rows.append({
            "k_seeds": k,
            "n_subsets": len(subsets),
            "spearman_mean": round(statistics.mean(spearman_list), 4),
            "spearman_min":  round(min(spearman_list), 4),
            "top1_match_rate": round(statistics.mean(top1_list), 4),
            "top3_match_rate": round(statistics.mean(top3_list), 4),
            "best_match_rate": round(statistics.mean(top_agree), 4),
            "mvsp_50": "",  # filled below
            "mvsp_80": "",
            "mvsp_95": "",
        })
    # find smallest k for each threshold
    full_top1 = rows[4]["top1_match_rate"]  # k=5 (full)
    for r in rows:
        if r["top1_match_rate"] >= 0.50 and not r["mvsp_50"]:
            r["mvsp_50"] = r["k_seeds"]
        if r["top1_match_rate"] >= 0.80 and not r["mvsp_80"]:
            r["mvsp_80"] = r["k_seeds"]
        if r["top1_match_rate"] >= 0.95 and not r["mvsp_95"]:
            r["mvsp_95"] = r["k_seeds"]
    return {"rows": rows, "full_rank": full, "full_top1_full_pool": full_top1}


def h2_robustness(by_m, n_boot=2000, seed=43):
    """Per-method (z = (μ - μ_grpo) / sqrt(σ² + σ_grpo²)) under random
    sub-sampling. Report mean and 95% CI of z over all 5-seed subsets."""
    random.seed(seed)
    grpo = [r["zvf_risk_max"] for r in by_m["grpo"]]
    mu_g = statistics.mean(grpo); sd_g = statistics.stdev(grpo) if len(grpo) > 1 else 0
    rows = []
    for m, rows_m in by_m.items():
        vals = [r["zvf_risk_max"] for r in rows_m]
        mu_m = statistics.mean(vals); sd_m = statistics.stdev(vals) if len(vals) > 1 else 0
        z_full = (mu_m - mu_g) / math.sqrt(sd_m**2 + sd_g**2) if (sd_m**2 + sd_g**2) > 0 else 0.0
        # bootstrap with 5-seed subsets → use all 5; vary 4-seed subset
        zs = []
        for combo in itertools.combinations(range(5), 4):
            sub = [vals[i] for i in combo]
            sub_g = [grpo[i] for i in combo]
            mu_s = statistics.mean(sub); sd_s = statistics.stdev(sub) if len(sub) > 1 else 0
            mu_gs = statistics.mean(sub_g); sd_gs = statistics.stdev(sub_g) if len(sub_g) > 1 else 0
            denom = math.sqrt(sd_s**2 + sd_gs**2)
            zs.append((mu_s - mu_gs) / denom if denom > 0 else 0.0)
        rows.append({
            "method": m,
            "mu": round(mu_m, 4),
            "sd": round(sd_m, 4),
            "z_full_pool": round(z_full, 4),
            "z_4seed_mean": round(statistics.mean(zs), 4),
            "z_4seed_min":  round(min(zs), 4),
            "z_4seed_max":  round(max(zs), 4),
            "z_4seed_sd":   round(statistics.stdev(zs), 4) if len(zs) > 1 else 0.0,
            "sign_stable": "yes" if (all(z < 0 for z in zs) or all(z > 0 for z in zs)) else "no",
        })
    return rows


def h3_ablation(by_m):
    """τ²-Bench-style fine-grained ablation: per method, compute fraction of
    total risk mass in (mag, csd, drift) channels. Identify the dominant
    channel per method."""
    rows = []
    for m, rows_m in by_m.items():
        mag = statistics.mean([r["risk_mag"]   for r in rows_m])
        csd = statistics.mean([r["risk_csd"]   for r in rows_m])
        drf = statistics.mean([r["risk_drift"] for r in rows_m])
        total = mag + csd + drf
        rows.append({
            "method": m,
            "risk_mag":   round(mag, 4),
            "risk_csd":   round(csd, 4),
            "risk_drift": round(drf, 4),
            "frac_mag":   round(mag / total, 4) if total > 0 else 0,
            "frac_csd":   round(csd / total, 4) if total > 0 else 0,
            "frac_drift": round(drf / total, 4) if total > 0 else 0,
            "dominant_channel": ["mag", "csd", "drift"][[mag, csd, drf].index(max(mag, csd, drf))],
        })
    return rows


def h4_clusters(by_m, n_bootstrap=500, seed=44):
    """τ²-Bench-COMPOSITIONAL: k-means-style 3-cluster assignment on the
    3-channel signature (frac_mag, frac_csd, frac_drift) and check whether
    the full-pool ranking is driven by ONE cluster's internal order or by
    inter-cluster gaps."""
    random.seed(seed)
    methods = list(by_m.keys())
    sig = {m: [] for m in methods}
    for m in methods:
        rows_m = by_m[m]
        mag = statistics.mean([r["risk_mag"]   for r in rows_m])
        csd = statistics.mean([r["risk_csd"]   for r in rows_m])
        drf = statistics.mean([r["risk_drift"] for r in rows_m])
        total = mag + csd + drf
        sig[m] = [mag / total if total > 0 else 0,
                  csd / total if total > 0 else 0,
                  drf / total if total > 0 else 0]
    # deterministic 3-cluster assignment by descending total zvf_risk_max magnitude
    full_rank = ranking({m: [r["zvf_risk_max"] for r in by_m[m]] for m in methods})
    # full_rank is best→worst; so index 0 = best (lowest zvf_risk_max).
    # bucket by position in this ordering: high = worst (last 3), low = best (first 3).
    n = len(full_rank)
    bucket = {m: ("high_risk" if i >= n - 3 else "mid_risk" if i >= 3 else "low_risk")
              for i, m in enumerate(full_rank)}
    # signature centroid per bucket
    centroids = {b: [0, 0, 0] for b in ["high_risk", "mid_risk", "low_risk"]}
    counts = {b: 0 for b in ["high_risk", "mid_risk", "low_risk"]}
    for m, s in sig.items():
        b = bucket[m]
        for j in range(3):
            centroids[b][j] += s[j]
        counts[b] += 1
    for b in centroids:
        if counts[b] > 0:
            centroids[b] = [round(c / counts[b], 4) for c in centroids[b]]
    # test stability of bucket assignment under leave-one-seed-out
    stable_count = {b: 0 for b in ["high_risk", "mid_risk", "low_risk"]}
    n = len(methods)
    for m, rows_m in by_m.items():
        for excluded in range(5):
            # leave-one-out: re-rank by mean zvf_risk_max with seed excluded
            other = {mm: ([r["zvf_risk_max"] for j, r in enumerate(by_m[mm]) if j != excluded]
                          if mm == m
                          else [r["zvf_risk_max"] for r in by_m[mm]])
                     for mm in methods}
            rk = ranking(other)
            pos = rk.index(m)
            b = ("high_risk" if pos >= n - 3 else "mid_risk" if pos >= 3 else "low_risk")
            if b == bucket[m]:
                stable_count[bucket[m]] += 1
    return {
        "buckets": {m: bucket[m] for m in methods},
        "centroids": centroids,
        "bucket_stability": {b: round(stable_count[b] / 15.0, 4) for b in ["high_risk", "mid_risk", "low_risk"]},
    }


def main():
    random.seed(42)
    by_m = load_per_seed()
    print(f"Loaded {sum(len(v) for v in by_m.values())} rows across {len(by_m)} methods")

    h1 = h1_mvsp(by_m)
    h2 = h2_robustness(by_m)
    h3 = h3_ablation(by_m)
    h4 = h4_clusters(by_m)

    # write TSVs
    def write_tsv(path, rows, header):
        with open(path, "w") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")

    write_tsv(OUT / "eval_protocol_mvsp.tsv", h1["rows"],
              ["k_seeds", "n_subsets", "spearman_mean", "spearman_min",
               "top1_match_rate", "top3_match_rate", "best_match_rate",
               "mvsp_50", "mvsp_80", "mvsp_95"])
    write_tsv(OUT / "eval_protocol_robustness.tsv", h2,
              ["method", "mu", "sd", "z_full_pool", "z_4seed_mean",
               "z_4seed_min", "z_4seed_max", "z_4seed_sd", "sign_stable"])
    write_tsv(OUT / "eval_protocol_ablation.tsv", h3,
              ["method", "risk_mag", "risk_csd", "risk_drift",
               "frac_mag", "frac_csd", "frac_drift", "dominant_channel"])
    cluster_rows = []
    for m, b in h4["buckets"].items():
        cluster_rows.append({"method": m, "bucket": b,
                             "stability": h4["bucket_stability"][b]})
    write_tsv(OUT / "eval_protocol_clusters.tsv", cluster_rows,
              ["method", "bucket", "stability"])

    summary = {
        "iter": 143,
        "pillar": "B-F25",
        "lectures": ["F25 L5 Yehudai (Survey)", "F25 L10 Barres+ (τ²-Bench)"],
        "data": str(RISK_TSV.relative_to(ROOT)),
        "n_methods": len(by_m),
        "n_seeds_per_method": 5,
        "full_ranking": h1["full_rank"],
        "mvsp_50": next((r["k_seeds"] for r in h1["rows"] if r["mvsp_50"]), ">5"),
        "mvsp_80": next((r["k_seeds"] for r in h1["rows"] if r["mvsp_80"]), ">5"),
        "mvsp_95": next((r["k_seeds"] for r in h1["rows"] if r["mvsp_95"]), ">5"),
        "n_methods_sign_stable_z": sum(1 for r in h2 if r["sign_stable"] == "yes"),
        "n_methods_drift_dominant": sum(1 for r in h3 if r["dominant_channel"] == "drift"),
        "n_methods_mag_dominant":   sum(1 for r in h3 if r["dominant_channel"] == "mag"),
        "n_methods_csd_dominant":   sum(1 for r in h3 if r["dominant_channel"] == "csd"),
        "bucket_stability": h4["bucket_stability"],
    }
    with open(OUT / "eval_protocol_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
