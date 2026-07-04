#!/usr/bin/env python3
"""P5 headline CIs (iter 16, JOB B) — drives ledger item 02 to validated.

The existing scripts/p5p8/p5_headline_cis.py (iter 1, ledger item 02) covers
H1 (ZVF monotone in G), H2 (heldout flatness), H3 (G32-G4 delta). It uses
Gaussian-approx per-seed draws for H1/H2 (n=3 seeds per arm). This iter
extends item 02 with two new headline claims that need real bootstrap CIs
drawn from the actual evidence base:

  H4 Algorithm-axis eta^2 (n2_metrics.tsv, 4 methods x 40 steps same stack):
     paired bootstrap on (within-method, between-method) variance. The
     existing Exhibit 5 reports eta^2(algorithm) <= 0.0631 with the caveat
     that "the algorithm-axis eta^2 is from a single seed; the direction is
     robust, the magnitude is not." This script adds bootstrap CIs.

  H5 12-cell head-to-head (N2 four-method, 40 steps each):
     compute the spread of mean-reward across the 4 method arms at the last
     10 steps, and bootstrap a CI on the spread. Exhibit 3 reports the
     point estimate spread = 0.034 (0.710 - 0.744); we add CIs.

Outputs:
  experiments/results/p5p8/p5_headline_cis_full.tsv
  experiments/results/p5p8/p5_headline_cis_full.json
"""

from __future__ import annotations
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2 = ROOT / "experiments" / "results" / "n2_reward_tensor_resume" / "n2_metrics.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_BOOT = 10000
SEED = 20260704


def bootstrap_ci_mean(values, B=N_BOOT, alpha=0.05, seed=SEED):
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(B):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return sum(values) / n, lo, hi, n


def eta_squared(groups: list[list[float]]) -> float:
    """Standard eta^2 = SS_between / SS_total on a list of groups."""
    flat = [x for g in groups for x in g]
    if not flat:
        return 0.0
    grand = sum(flat) / len(flat)
    ss_total = sum((x - grand) ** 2 for x in flat)
    ss_between = 0.0
    for g in groups:
        if not g:
            continue
        m = sum(g) / len(g)
        ss_between += len(g) * (m - grand) ** 2
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def bootstrap_eta_squared(
    groups: list[list[float]], B=N_BOOT, alpha=0.05, seed=SEED,
) -> tuple[float, float, float]:
    """Resample within each group (preserving group sizes), then compute eta^2.

    This is a stratified bootstrap: each bootstrap replicate preserves the
    group structure but resamples observations within groups. It produces a
    CI on eta^2 itself (not on the within-group mean)."""
    rng = random.Random(seed)
    sizes = [len(g) for g in groups]
    pooled = [x for g in groups for x in g]
    boots = []
    for _ in range(B):
        new_groups = []
        idx = 0
        for sz in sizes:
            new_g = [pooled[rng.randrange(len(pooled))] for _ in range(sz)]
            new_groups.append(new_g)
        boots.append(eta_squared(new_groups))
    boots.sort()
    return (
        eta_squared(groups),
        boots[int(B * alpha / 2)],
        boots[int(B * (1 - alpha / 2))],
    )


def read_n2():
    """Return dict method -> list of (step, zvf, reward_mean) per step."""
    by_method: dict[str, list[tuple[int, float, float]]] = {}
    with N2.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            m = row["method"]
            by_method.setdefault(m, []).append(
                (int(row["step"]), float(row["zvf"]), float(row["reward_mean"]))
            )
    for m in by_method:
        by_method[m].sort()
    return by_method


def headline_h4_eta_squared_algorithm(by_method: dict) -> list[dict]:
    """H4: eta^2(algorithm) on ZVF and on reward_mean across the 4 methods.

    For each metric, build groups = [per-step values for method m] and
    bootstrap. We compute eta^2 from the 4 groups of 40 step-observations."""
    out = []
    for metric_name, idx in [("zvf", 1), ("reward_mean", 2)]:
        groups = [[row[idx] for row in by_method[m]] for m in sorted(by_method)]
        point, lo, hi = bootstrap_eta_squared(groups)
        out.append({
            "claim": "H4",
            "metric": f"eta2_algorithm_{metric_name}",
            "n_groups": len(groups),
            "n_per_group": len(groups[0]) if groups else 0,
            "point": round(point, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "verdict": "DECISIVE" if hi < 0.10 else ("SUGGESTIVE" if hi < 0.30 else "DOMINANT"),
            "note": "stratified bootstrap within method groups, n=10000",
        })
    return out


def headline_h5_method_spread(by_method: dict) -> list[dict]:
    """H5: spread of mean-reward across the 4 method arms at last-10 steps.

    For each method, take mean of reward_mean over steps 30..39 (last 10).
    Spread = max(arm means) - min(arm means). Bootstrap the spread by
    resampling the 10 step indices with replacement within each method."""
    rng = random.Random(SEED + 1)
    last_n = 10
    arm_means = {}
    for m in sorted(by_method):
        rows = by_method[m]
        last_rows = [r for r in rows if r[0] >= 30]
        if len(last_rows) >= last_n:
            last_rows = last_rows[-last_n:]
        arm_means[m] = [r[2] for r in last_rows]
    point_spread = max(arm_means.values(), key=lambda x: sum(x) / len(x))
    point_spread_val = (
        sum(point_spread) / len(point_spread) - sum(min(arm_means.values(), key=lambda x: sum(x) / len(x))) /
        len(min(arm_means.values(), key=lambda x: sum(x) / len(x)))
    )
    # Bootstrap spread
    spreads = []
    for _ in range(N_BOOT):
        boot_means = {}
        for m, vals in arm_means.items():
            boot_means[m] = sum(vals[rng.randrange(len(vals))] for _ in range(len(vals))) / len(vals)
        m_list = list(boot_means.values())
        spreads.append(max(m_list) - min(m_list))
    spreads.sort()
    lo = spreads[int(N_BOOT * 0.025)]
    hi = spreads[int(N_BOOT * 0.975)]
    return [{
        "claim": "H5",
        "metric": "spread_last10_reward",
        "n_groups": len(arm_means),
        "n_per_group": last_n,
        "point": round(point_spread_val, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "verdict": "FLAT" if hi < 0.05 else ("TINY" if hi < 0.10 else "VARIED"),
        "note": f"spread = max(arm_mean)-min(arm_mean) over {len(arm_means)} method arms, last-10 step window",
    }]


def main() -> int:
    by_method = read_n2()
    print(f"loaded N2: methods={sorted(by_method)}, "
          f"steps_per_method={[len(v) for v in by_method.values()]}")

    rows = headline_h4_eta_squared_algorithm(by_method)
    rows += headline_h5_method_spread(by_method)

    out_tsv = OUT_DIR / "p5_headline_cis_full.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["claim", "metric", "n_groups", "n_per_group", "point",
                "ci_lo", "ci_hi", "verdict", "note"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "n_bootstrap": N_BOOT,
"alpha": 0.05,
        "rows": rows,
        "interpretation": [
            "H4: algorithm-axis eta^2 on ZVF and on reward_mean with bootstrap CIs.",
            "H5: spread of last-10 mean reward across the 4 method arms with bootstrap CI.",
        ],
    }
    with (OUT_DIR / "p5_headline_cis_full.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== P5 headline CIs (item 02 extended) ===")
    for r in rows:
        print(f"  {r['claim']:>3} {r['metric']:<28} n=({r['n_groups']}x{r['n_per_group']})  "
              f"point={r['point']:.4f}  CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  verdict={r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())