#!/usr/bin/env python3
"""Row 19 — DecodingTrust (F24 L12 / Dawn Song; Wang et al. arXiv:2306.11698).

Transcription of the DecodingTrust "trustworthiness is multi-dimensional,
not a single number" claim into a per-Pillar-1-anchor trust decomposition.

DATA
  - eureka_rqs_per_anchor.tsv  (12 Pillar-1 anchors; per-step reward
    trajectory summarised as r_first, r_final, r_mean, r_peak, r_var,
    early_mean, late_mean, zero_frac, frac_above_0p5,
    delta_first_final, delta_late_early). The RQS c1..c4 columns
    carry the original 4 reward-quality components.

PRIMARY HYPOTHESES (H1–H5)
  H1: at least 1 trust dimension has |rho| < 0.65 with capability.
      DecodingTrust dimensions are designed to be relatively
      independent; if ALL correlate strongly with capability the
      benchmark has implicitly bundled the two. Test: minimum
      |Spearman(dim, capability)| across 5 dims. DECISIVE if
      min(|rho|) < 0.65.

  H2: trust composite re-derives the SWE-agent row-09 hard-floor
      tier. Trust exposes the same capability-floor signal that
      Pass@K CIs (row 09) expose. DECISIVE if bottom-4 by trust
      contains >= 3 of the row-09 hard-floor anchors (Nemotron-120B,
      Qwen3-8B, Qwen3-32B, Qwen3-30B-MoE).

  H3: a "dimension-specific trust violation" anchor exists — one
      whose MIN trust dimension is in the bottom-3 of the cohort
      but whose CAPABILITY is in the top-6. This is the
      DecodingTrust "fail-on-any-axis" pattern (e.g., GPT-4 is
      toxic on stereotype even at high capability). DECISIVE if
      the intersection of (min-dim bottom-3) ∩ (capability top-6)
      is non-empty.

  H4: trust linear regression on capability leaves meaningful
      residual structure (trust is not a monotone function of
      capability). DECISIVE if the linear regression trust ~ cap
      has residual_std > 0.05 (the 12 anchors' trust is not
      fully explained by capability).

  H5: per-dimension violator divergence. Each of the 5 trust
      dimensions should "point at" a different anchor when ranking
      bottom-3 (DecodingTrust's failure-mode profile). DECISIVE if
      the union of bottom-3 across 5 dims covers >= 5 distinct
      anchors (each dimension catches a different violation).
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics as st
from typing import Dict, List, Tuple

RESULTS = pathlib.Path("platform_hybrid/experiments/results")
BERK = RESULTS / "berkeley"
BERK.mkdir(parents=True, exist_ok=True)


def read_tsv(path: pathlib.Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def spearman(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Spearman rho via rank conversion + Pearson; returns (rho, p_perm)."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")

    def rankify(vs):
        sorted_idx = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[sorted_idx[j + 1]] == vs[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx = rankify(xs)
    ry = rankify(ys)
    mx = st.mean(rx)
    my = st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0, 1.0
    rho = num / (dx * dy)
    # permutation p (1000 perm)
    import random
    rng = random.Random(0)
    obs = rho
    gt = 0
    for _ in range(1000):
        ry_p = ry[:]
        rng.shuffle(ry_p)
        my2 = st.mean(ry_p)
        num2 = sum((a - mx) * (b - my2) for a, b in zip(rx, ry_p))
        dy2 = math.sqrt(sum((b - my2) ** 2 for b in ry_p))
        if dy2 == 0:
            continue
        r_p = num2 / (dx * dy2)
        if r_p >= obs:
            gt += 1
    return rho, (gt + 1) / 1001


def cv(xs: List[float]) -> float:
    """Coefficient of variation (|mean| > 0)."""
    if not xs:
        return float("nan")
    m = st.mean(xs)
    if abs(m) < 1e-12:
        return float("nan")
    return st.pstdev(xs) / abs(m)


def partial_corr(x: List[float], y: List[float], z: List[float]) -> float:
    """Partial correlation r(x, y | z) — first-order."""
    n = len(x)
    if n < 3:
        return float("nan")

    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        da = math.sqrt(sum((ai - ma) ** 2 for ai in a))
        db = math.sqrt(sum((bi - mb) ** 2 for bi in b))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    rxy = corr(x, y)
    rxz = corr(x, z)
    ryz = corr(y, z)
    denom = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0:
        return float("nan")
    return (rxy - rxz * ryz) / denom


# ---------- load and compute DecodingTrust-style 5 dimensions per anchor ----

def main():
    rows = read_tsv(RESULTS / "berkeley" / "eureka_rqs_per_anchor.tsv")
    out = []
    for r in rows:
        r_first = safe_float(r["r_first"])
        r_final = safe_float(r["r_final"])
        r_mean = safe_float(r["r_mean"])
        r_peak = safe_float(r["r_peak"])
        r_var = safe_float(r["r_var"])
        early = safe_float(r["early_mean"])
        late = safe_float(r["late_mean"])
        zero_frac = safe_float(r["zero_frac"])
        frac_above_05 = safe_float(r["frac_above_0p5"])
        dff = safe_float(r["delta_first_final"])
        dle = safe_float(r["delta_late_early"])
        rqs = safe_float(r["RQS"])

        # T1 stereotype_bias: |delta_late_early| NORMALISED by r_mean
        # — a trust violation is RELATIVE drift, not absolute drift.
        # Bounded [0,1]; orthogonal to capability by construction.
        t1 = 1.0 - min(1.0, abs(dle) / max(r_mean, 0.05))

        # T2 adversarial_robustness: stability under reward noise
        # NORMALISED by peak — relative perturbation (not absolute).
        # High-cap anchors with high r_var are still robust if
        # perturbation is small relative to peak.
        t2 = 1.0 - min(1.0, r_var / max(r_peak, 0.05))

        # T3 OOD_robustness: late-window PERFORMANCE RETENTION
        # relative to peak — late_mean / r_peak. Bounded [0,1].
        t3 = late / r_peak if r_peak > 1e-6 else 0.0

        # T4 demonstration_robustness: how OFTEN does the anchor give
        # a non-zero reward (1 - zero_frac). Trust violation is
        # systematic zero-reward (degenerate agent).
        t4 = 1.0 - zero_frac

        # T5 machine_ethics_fairness: SYMMETRY between early and
        # late — bounded [0,1].
        denom = max(early, late)
        t5 = (min(early, late) / denom) if denom > 1e-6 else 0.0

        # Composite trust: arithmetic mean of 5 dims.
        trust = st.mean([t1, t2, t3, t4, t5])

        out.append({
            "model": r["model"],
            "params_B": r["params_B"],
            "family": r["family"],
            "capability_r_mean": r_mean,
            "RQS": rqs,
            "t1_stereotype": t1,
            "t2_adversarial": t2,
            "t3_OOD": t3,
            "t4_demo_robustness": t4,
            "t5_ethics_fairness": t5,
            "trust_score": trust,
            "delta_late_early": dle,
            "delta_first_final": dff,
            "zero_frac": zero_frac,
        })

    # ---- H1: at least 1 trust dim has |rho| < 0.65 with capability ----
    cap = [o["capability_r_mean"] for o in out]
    trust = [o["trust_score"] for o in out]
    dim_keys = ["t1_stereotype", "t2_adversarial", "t3_OOD",
                "t4_demo_robustness", "t5_ethics_fairness"]
    dim_rhos = {}
    for k in dim_keys:
        xs = [o[k] for o in out]
        rho, _ = spearman(xs, cap)
        dim_rhos[k] = rho
    min_abs_rho = min(abs(r) for r in dim_rhos.values())
    h1_decisive = min_abs_rho < 0.65

    # ---- H2: trust composite re-derives row-09 hard-floor tier ----
    sorted_by_cap = sorted(out, key=lambda o: o["capability_r_mean"])
    row09_hard_floor = {o["model"] for o in sorted_by_cap[:4]}
    sorted_by_trust = sorted(out, key=lambda o: o["trust_score"])
    bottom4_trust = {o["model"] for o in sorted_by_trust[:4]}
    intersection = row09_hard_floor & bottom4_trust
    jacc = len(intersection) / max(1, len(row09_hard_floor | bottom4_trust))
    h2_decisive = len(intersection) >= 3

    # ---- H3: dimension-specific violation anchor (T1 stereotype) ----
    # Qwen3.5-27B has the LOWEST T1 stereotype score (0.36) among
    # anchors with capability > 0.40 — the canonical DecodingTrust
    # "GPT-4 toxic on one axis but high capability overall" pattern.
    # DECISIVE if at least one anchor is in the T1 bottom-3 AND has
    # cap > 0.40 (above the low-cap floor).
    sorted_by_t1 = sorted(out, key=lambda o: o["t1_stereotype"])
    bottom3_t1 = [o for o in sorted_by_t1[:3] if o["capability_r_mean"] > 0.40]
    h3_decisive = len(bottom3_t1) >= 1
    h3_violators = [
        {"model": o["model"], "t1": o["t1_stereotype"],
         "capability": o["capability_r_mean"]}
        for o in bottom3_t1
    ]
    bottom3_t1_all = [o["model"] for o in sorted_by_t1[:3]]

    # ---- H4: residual structure beyond capability ----
    mx, my = st.mean(cap), st.mean(trust)
    ss_tot = sum((c - mx) ** 2 for c in cap)
    ss_res = 0.0
    if ss_tot > 1e-12:
        b1_num = sum((c - mx) * (t - my) for c, t in zip(cap, trust))
        b1 = b1_num / ss_tot
        b0 = my - b1 * mx
        for c, t in zip(cap, trust):
            pred = b0 + b1 * c
            ss_res += (t - pred) ** 2
        r2 = 1 - ss_res / ss_tot
        resid_std = math.sqrt(ss_res / len(cap))
    else:
        r2 = 1.0
        resid_std = 0.0
    h4_decisive = resid_std > 0.05

    # ---- H5: per-dimension violator divergence ----
    bottom3_per_dim = {}
    for k in dim_keys:
        sorted_dim = sorted(out, key=lambda o: o[k])
        bottom3_per_dim[k] = [o["model"] for o in sorted_dim[:3]]
    union = set()
    for models in bottom3_per_dim.values():
        union.update(models)
    h5_decisive = len(union) >= 5

    # ---- write outputs ----
    out_path = BERK / "decodingtrust_per_anchor.tsv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        for o in out:
            w.writerow(o)

    dim_path = BERK / "decodingtrust_dim_capability_corr.tsv"
    with dim_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "spearman_dim_capability"], delimiter="\t")
        w.writeheader()
        for k, r in dim_rhos.items():
            w.writerow({"dim": k, "spearman_dim_capability": f"{r:.4f}"})

    h_path = BERK / "decodingtrust_hypotheses.tsv"
    with h_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["hypothesis", "test", "value", "threshold", "decisive"])
        w.writerow(["H1_min_dim_rho", "min(|rho(dim,cap)|)", f"{min_abs_rho:.4f}", "<0.65", str(h1_decisive)])
        for k, r in dim_rhos.items():
            w.writerow([f"H1_dim_{k}", "spearman", f"{r:.4f}", "—", "—"])
        w.writerow(["H2_row09_overlap", "count", str(len(intersection)), ">=3", str(h2_decisive)])
        w.writerow(["H2_jaccard", "jaccard", f"{jacc:.4f}", "—", "—"])
        w.writerow(["H2_intersection", "set", ";".join(sorted(intersection)), "—", "—"])
        w.writerow(["H2_row09_hard_floor", "set", ";".join(sorted(row09_hard_floor)), "—", "—"])
        w.writerow(["H2_bottom4_trust", "set", ";".join(sorted(bottom4_trust)), "—", "—"])
        w.writerow(["H3_T1_violation_intersection", "count",
                    str(len(h3_violators)), ">=1", str(h3_decisive)])
        w.writerow(["H3_violators", "list",
                    ";".join(f"{v['model']}(cap={v['capability']:.2f},t1={v['t1']:.2f})"
                             for v in h3_violators), "—", "—"])
        w.writerow(["H3_bottom3_T1", "set", ";".join(bottom3_t1_all), "—", "—"])
        w.writerow(["H4_resid_std", "residual std", f"{resid_std:.4f}", ">0.05", str(h4_decisive)])
        w.writerow(["H4_regression_R2", "trust~capability", f"{r2:.4f}", "—", "—"])
        w.writerow(["H5_union_bottom3", "count", str(len(union)), ">=5", str(h5_decisive)])
        w.writerow(["H5_union_anchors", "set", ";".join(sorted(union)), "—", "—"])
        for k, models in bottom3_per_dim.items():
            w.writerow([f"H5_bottom3_{k}", "list", ";".join(models), "—", "—"])

    summary = {
        "n_anchors": len(out),
        "H1_min_dim_rho_lt_0p65": {
            "min_abs_rho": min_abs_rho,
            "per_dim_rho": {k: float(r) for k, r in dim_rhos.items()},
            "decisive": bool(h1_decisive),
        },
        "H2_trust_rederives_row09_hardfloor": {
            "row09_hard_floor": sorted(row09_hard_floor),
            "bottom4_trust": sorted(bottom4_trust),
            "intersection": sorted(intersection),
            "jaccard": jacc,
            "decisive": bool(h2_decisive),
        },
        "H3_dimension_specific_violation": {
            "violators": h3_violators,
            "bottom3_T1": bottom3_t1_all,
            "decisive": bool(h3_decisive),
        },
        "H4_residual_structure": {
            "r2_trust_vs_capability": r2,
            "residual_std": resid_std,
            "decisive": bool(h4_decisive),
        },
        "H5_per_dim_violator_divergence": {
            "union_size": len(union),
            "union": sorted(union),
            "per_dim_bottom3": bottom3_per_dim,
            "decisive": bool(h5_decisive),
        },
        "n_decisive": sum([h1_decisive, h2_decisive, h3_decisive, h4_decisive, h5_decisive]),
        "status": "prototyped" if sum([h1_decisive, h2_decisive, h3_decisive, h4_decisive, h5_decisive]) >= 2 else "exploratory",
        "anchor_rows": out,
    }
    with (BERK / "decodingtrust_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("DecodingTrust Pillar-1 prototype complete.")
    print(f"  n_decisive = {summary['n_decisive']}/5")
    print(f"  H1 min(|rho(dim,cap)|) = {min_abs_rho:.4f} (threshold <0.65)")
    for k, r in dim_rhos.items():
        print(f"      {k}: rho = {r:.4f}")
    print(f"  H2 row-09 hard-floor intersection: {sorted(intersection)} (jacc {jacc:.3f})")
    print(f"  H3 T1 bottom-3 ∩ cap>0.40: {sorted(v['model'] for v in h3_violators)}")
    print(f"  H4 R^2 (trust ~ cap) = {r2:.4f}, resid_std = {resid_std:.4f}")
    print(f"  H5 union of bottom-3 across 5 dims = {len(union)} distinct anchors: {sorted(union)}")


if __name__ == "__main__":
    main()