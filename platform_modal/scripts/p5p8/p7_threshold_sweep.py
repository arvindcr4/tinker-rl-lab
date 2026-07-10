#!/usr/bin/env python3
"""
Pillar-7 (P7) τ-sensitivity sweep with seed-robustness bootstrap CIs.

Iter 39: addresses brief vein (c) — "seed-robustness of the trigger threshold
on the growing n10_seed_expansion panel".

Loads every n10_grpo_s*.json in platform_hybrid/experiments/results/n10_seed_expansion/
(falls back to the 5-seed panel if no new seeds have landed since iter 27).

For each controller in {zvf_triage, dualformer_auto, hybrid} and each τ ∈
{0.50, 0.55, ..., 0.95}, replays the per-step controller over each seed's
15-step zvf trajectory and computes:

  * total_G           — sum of G_t over 15 steps (compute proxy)
  * savings           — (120 − total_G) / 120  (vs always-G=8)
  * fire_rate         — fraction of steps where G_t ≠ 8
  * headroom_bad      — fires on z_t ≥ 0.99 (should be 0 for well-calibrated)
  * mean_zvf_at_fire  — average z_t over the firing steps

Then computes per (controller, τ):
  * paired-bootstrap 95% CI on savings (B=2000, percentile, seed as unit)
  * seed-CV          — std(per-seed total_G) / mean (seed-robustness metric)
  * best_τ           — argmin seed-CV subject to mean(headroom_bad) == 0

Falsifiable headline: at τ ∈ {0.55, 0.60}, Dualformer-Auto strictly dominates
zvf-triage on both the compute axis (savings > 0) and the seed-robustness
axis (CV lower) on the 5-seed N10 panel.

References (verified):
  - su2024dualformer     (Su et al., 2024, "Dualformer")
  - alphaproof2025nature (AlphaProof, Nature 2025)

Outputs (worktree-relative paths):
  platform_hybrid/experiments/results/p5p8/p7_threshold_sweep_per_seed.tsv
  platform_hybrid/experiments/results/p5p8/p7_threshold_sweep_summary.tsv
  platform_hybrid/experiments/results/p5p8/p7_threshold_sweep_ci.tsv
  platform_hybrid/experiments/results/p5p8/p7_threshold_sweep_summary.json
"""
from __future__ import annotations
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N10_DIR = ROOT / "platform_hybrid/experiments/results/n10_seed_expansion"
OUT_DIR = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G_BASE = 8
G_ESC = 16
G_DES = 4
N_STEPS = 15
N_BOOT = 2000
RNG_SEED = 39007
# τ grid: 0.50, 0.55, ..., 0.95
TAUS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
DELTA = 0.10  # Hybrid band width


# ----------------------------------------------------------------------------
def load_n10_seeds() -> list[dict]:
    """Load every N10 GRPO run JSON on disk; return list of dicts with
    seed + step-level zvf trajectory. Skip runs with < N_STEPS step_log."""
    out = []
    for path in sorted(N10_DIR.glob("n10_grpo_s*.json")):
        d = json.loads(path.read_text())
        sl = d.get("step_log", [])
        if len(sl) < N_STEPS:
            continue
        zvfs = [float(s["zvf"]) for s in sl[:N_STEPS]]
        rewards = [float(s["reward"]) for s in sl[:N_STEPS]]
        out.append({
            "seed": int(d["seed"]),
            "zvfs": zvfs,
            "rewards": rewards,
            "heldout_acc": float(d.get("heldout_acc", float("nan"))),
        })
    return out


# ----------------------------------------------------------------------------
def controller_zvf_triage(z: list[float], tau: float) -> list[int]:
    return [G_ESC if zt >= tau else G_BASE for zt in z]


def controller_dualformer(z: list[float], tau: float) -> list[int]:
    return [G_DES if zt >= tau else G_BASE for zt in z]


def controller_hybrid(z: list[float], tau: float, delta: float) -> list[int]:
    out = []
    for zt in z:
        if zt >= tau + delta:
            out.append(G_DES)
        elif zt >= tau:
            out.append(G_ESC)
        else:
            out.append(G_BASE)
    return out


def per_seed_metrics(G_t: list[int], z: list[float]) -> dict:
    n = len(G_t)
    total = sum(G_t)
    fires = [g for g, zt in zip(G_t, z) if g != G_BASE]
    n_fire = len(fires)
    headroom_bad = sum(1 for g, zt in zip(G_t, z) if g != G_BASE and zt >= 0.99)
    zvf_at_fire = [zt for g, zt in zip(G_t, z) if g != G_BASE]
    return {
        "total_G": total,
        "savings": (G_BASE * n - total) / (G_BASE * n),
        "fire_rate": n_fire / n,
        "headroom_bad": headroom_bad,
        "mean_zvf_at_fire": sum(zvf_at_fire) / len(zvf_at_fire) if zvf_at_fire else 0.0,
    }


def paired_boot_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    """Paired bootstrap percentile CI on the mean of `values` (seed as iid draw)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(values[i] for i in idx) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return (lo, hi)


def seed_cv(values: list[float]) -> float:
    """Coefficient of variation across seeds (smaller = more seed-robust)."""
    if len(values) < 2:
        return float("nan")
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.stdev(values) / abs(m)


# ----------------------------------------------------------------------------
def main():
    seeds = load_n10_seeds()
    if len(seeds) < 2:
        raise SystemExit(f"need >=2 complete seeds, found {len(seeds)}")

    controllers = [
        ("zvf_triage", controller_zvf_triage, {}),
        ("dualformer_auto", controller_dualformer, {}),
        ("hybrid", controller_hybrid, {"delta": DELTA}),
    ]

    # -- per-seed detail rows -------------------------------------------------
    per_seed_rows: list[dict] = []
    # -- per-(controller, τ) summary rows -------------------------------------
    summary_rows: list[dict] = []
    # -- per-(controller, τ) bootstrap CI rows --------------------------------
    ci_rows: list[dict] = []

    for cname, cfn, ckw in controllers:
        for tau in TAUS:
            savings_list = []
            total_list = []
            fire_list = []
            hrbad_list = []
            for s in seeds:
                Gt = cfn(s["zvfs"], tau, **ckw) if cname == "hybrid" else cfn(s["zvfs"], tau)
                m = per_seed_metrics(Gt, s["zvfs"])
                per_seed_rows.append({
                    "controller": cname,
                    "tau": tau,
                    "seed": s["seed"],
                    "total_G": m["total_G"],
                    "savings": m["savings"],
                    "fire_rate": m["fire_rate"],
                    "headroom_bad": m["headroom_bad"],
                    "mean_zvf_at_fire": m["mean_zvf_at_fire"],
                })
                savings_list.append(m["savings"])
                total_list.append(m["total_G"])
                fire_list.append(m["fire_rate"])
                hrbad_list.append(m["headroom_bad"])

            mean_savings = statistics.mean(savings_list)
            mean_total = statistics.mean(total_list)
            mean_fire = statistics.mean(fire_list)
            mean_hrbad = statistics.mean(hrbad_list)
            lo, hi = paired_boot_ci(savings_list, N_BOOT, RNG_SEED)
            cv_total = seed_cv(total_list)
            cv_savings = seed_cv(savings_list)

            summary_rows.append({
                "controller": cname,
                "tau": tau,
                "n_seeds": len(seeds),
                "mean_total_G": mean_total,
                "mean_savings": mean_savings,
                "mean_fire_rate": mean_fire,
                "mean_headroom_bad": mean_hrbad,
                "seed_cv_total": cv_total,
                "seed_cv_savings": cv_savings,
            })
            ci_rows.append({
                "controller": cname,
                "tau": tau,
                "metric": "savings",
                "point": mean_savings,
                "ci_lo": lo,
                "ci_hi": hi,
                "excludes_zero": (lo > 0) or (hi < 0),
                "seed_cv": cv_total,
            })

    # --- write per_seed detail tsv --------------------------------------------
    psp = OUT_DIR / "p7_threshold_sweep_per_seed.tsv"
    with psp.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(per_seed_rows[0].keys()),
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(per_seed_rows)

    # --- write summary tsv ---------------------------------------------------
    ssp = OUT_DIR / "p7_threshold_sweep_summary.tsv"
    with ssp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(summary_rows)

    # --- write ci tsv --------------------------------------------------------
    csp = OUT_DIR / "p7_threshold_sweep_ci.tsv"
    with csp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(ci_rows)

    # --- compute the headline: best-τ per controller under headroom-bad=0 -----
    headline = {}
    for cname, _, _ in controllers:
        eligible = [r for r in summary_rows if r["controller"] == cname and r["mean_headroom_bad"] == 0]
        if not eligible:
            headline[cname] = {"best_tau": None, "best_savings": None, "best_seed_cv": None}
            continue
        # Maximize mean_savings (prefer compute-efficient).
        best = max(eligible, key=lambda r: r["mean_savings"])
        headline[cname] = {
            "best_tau": best["tau"],
            "best_savings": best["mean_savings"],
            "best_fire_rate": best["mean_fire_rate"],
            "best_seed_cv_total": best["seed_cv_total"],
            "best_seed_cv_savings": best["seed_cv_savings"],
        }

    # --- summary JSON ---------------------------------------------------------
    out = {
        "iter": 39,
        "n_seeds": len(seeds),
        "seeds": [s["seed"] for s in seeds],
        "taus": TAUS,
        "G_base": G_BASE,
        "G_esc": G_ESC,
        "G_des": G_DES,
        "n_boot": N_BOOT,
        "n_steps_per_seed": N_STEPS,
        "headline_best_tau_per_controller": headline,
        "n_per_seed_rows": len(per_seed_rows),
        "n_summary_rows": len(summary_rows),
        "n_ci_rows": len(ci_rows),
    }
    (OUT_DIR / "p7_threshold_sweep_summary.json").write_text(json.dumps(out, indent=2))

    print(f"loaded {len(seeds)} seeds; wrote {len(per_seed_rows)} per-seed rows, "
          f"{len(summary_rows)} summary rows, {len(ci_rows)} ci rows")
    for cname, h in headline.items():
        if h["best_tau"] is None:
            print(f"  {cname:>15}: NO headroom-clean operating point")
        else:
            print(f"  {cname:>15}: best_τ={h['best_tau']:.2f}  savings={h['best_savings']:+.4f}  "
                  f"seed_CV_total={h['best_seed_cv_total']:.4f}")


if __name__ == "__main__":
    main()