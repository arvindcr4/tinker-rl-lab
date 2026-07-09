#!/usr/bin/env python3
"""
Pillar-7 (P7) per-seed Dualformer-Auto + zvf-triage + Hybrid controller replay
on the N10 seed-expansion panel (5 GRPO seeds × 15 steps each, G_base=8).

Iter 27: addresses brief vein (b) [unify Dualformer auto-G + zvf-triage +
gamma*=0 smoothing into one calibrated controller section] AND vein (c)
[seed-robustness of the trigger threshold on the growing n10_seed_expansion
panel] in the same artifact.

Three per-step controllers (all dispatch on per-step ZVF z_t, G_base=8):

  C0 baseline    : G_t = 8                           (compute = 120/seed)
  C1 zvf-triage@τ : G_t = 16 if z_t >= τ else 8       (escalate on boundary)
  C2 Dualformer@τ : G_t = 4 if z_t >= τ else 8        (de-escalate easy steps)
  C3 Hybrid@τ     : G_t = 16 if τ <= z_t < τ+δ, 4 if z_t >= τ+δ, 8 otherwise

Per-seed metrics (n=5 seeds):
  total_G       sum of G_t over 15 steps (= compute proxy)
  savings_vs_C0 (total_G_C0 - total_G_Ci) / total_G_C0
  n_fire        number of steps with G_t != 8
  select_rate   n_fire / 15
  headroom_bad  number of steps where the controller fired on z_t >= 0.99
                (saturated prompt, no escalation value) — should be 0 for C1/C3

Statistical rigor:
  - Bootstrap-CI per seed (B=2000, percentile) for total_G and savings_vs_C0,
    treating the 5 seeds as iid draws from a hypothetical seed population.
  - Paired bootstrap-CI on per-seed Δsavings (C2 - C1, C3 - C1, C3 - C2)
    — the headline falsifiable claim is that the ordering C3 <= C2 < C1
    (Hybrid saves the most, then Dualformer, then zvf-triage spends the most).

References (verified):
  - su2024dualformer     (Su et al., 2024, "Dualformer")
  - alphaproof2025nature (AlphaProof, Nature 2025)

Outputs (worktree-relative paths):
  experiments/results/p5p8/p7_dualformer_n10_per_seed.tsv
  experiments/results/p5p8/p7_dualformer_n10_summary.json
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
N10_DIR = ROOT / "experiments/results/n10_seed_expansion"
OUT_DIR = ROOT / "experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 0.7           # primary zvf-triage threshold
TAU_DELTA = 0.2     # Hybrid band width: z in [tau, tau+tau_delta) escalates
G_BASE = 8          # N10 fixed group size
G_ESC = 16          # escalation: 2x base (Dualformer "slow mode")
G_DES = 4           # de-escalation: 0.5x base (Dualformer "fast mode")
N_STEPS = 15        # N10 panel step count per seed
N_BOOT = 2000       # bootstrap replicates
RNG_SEED = 27


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_n10_seeds() -> list[dict]:
    """Load every N10 GRPO run JSON on disk; return list of dicts with
    seed + step-level zvf trajectory."""
    out = []
    for path in sorted(N10_DIR.glob("n10_grpo_s*.json")):
        d = json.loads(path.read_text())
        sl = d.get("step_log", [])
        if len(sl) != N_STEPS:
            continue
        zvfs = [float(s["zvf"]) for s in sl]
        rewards = [float(s["reward"]) for s in sl]
        out.append(
            {
                "seed": int(d["seed"]),
                "zvfs": zvfs,
                "rewards": rewards,
                "heldout_acc": float(d.get("heldout_acc", float("nan"))),
                "mean_zvf": float(d.get("mean_zvf", sum(zvfs) / len(zvfs))),
            }
        )
    return out


# ----------------------------------------------------------------------------
# Controller dispatch
# ----------------------------------------------------------------------------
def c0(z: list[float]) -> list[int]:
    return [G_BASE] * len(z)


def c1_zvf_triage(z: list[float], tau: float) -> list[int]:
    return [G_ESC if zt >= tau else G_BASE for zt in z]


def c2_dualformer(z: list[float], tau: float) -> list[int]:
    return [G_DES if zt >= tau else G_BASE for zt in z]


def c3_hybrid(z: list[float], tau: float, delta: float) -> list[int]:
    out = []
    for zt in z:
        if zt >= tau + delta:
            out.append(G_DES)        # post-escalation easy: shrink to fast
        elif zt >= tau:
            out.append(G_ESC)        # boundary: escalate to slow
        else:
            out.append(G_BASE)
    return out


# ----------------------------------------------------------------------------
# Per-seed metrics
# ----------------------------------------------------------------------------
def per_seed_metrics(G_t: list[int], z: list[float]) -> dict:
    n = len(G_t)
    total = sum(G_t)
    n_fire = sum(1 for g in G_t if g != G_BASE)
    sel = n_fire / n
    headroom_bad = sum(1 for g, zt in zip(G_t, z) if g != G_BASE and zt >= 0.99)
    return {
        "total_G": total,
        "n_fire": n_fire,
        "select_rate": sel,
        "headroom_bad": headroom_bad,
    }


# ----------------------------------------------------------------------------
# Bootstrap CI (percentile) on a list of per-seed scalars
# ----------------------------------------------------------------------------
def boot_ci(values: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "sd": float("nan")}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {
        "mean": statistics.mean(values),
        "lo": lo,
        "hi": hi,
        "sd": statistics.stdev(values) if n > 1 else 0.0,
        "n": n,
    }


def paired_boot_ci_delta(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """Paired bootstrap on (a-b) across the same seed ordering."""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "mean_diff": statistics.mean(diffs),
        "lo": means[int(0.025 * n_boot)],
        "hi": means[int(0.975 * n_boot)],
        "n": n,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    seeds = load_n10_seeds()
    n_seeds = len(seeds)
    assert n_seeds >= 3, f"Need >=3 N10 seeds; got {n_seeds}"

    controllers = {
        "C0_baseline": lambda z: c0(z),
        f"C1_zvf_triage@{TAU:.2f}": lambda z: c1_zvf_triage(z, TAU),
        f"C2_dualformer@{TAU:.2f}": lambda z: c2_dualformer(z, TAU),
        f"C3_hybrid@{TAU:.2f}+{TAU_DELTA:.2f}": lambda z: c3_hybrid(z, TAU, TAU_DELTA),
    }

    # Per-seed total_G for every controller
    rows = []
    per_seed_total = {name: [] for name in controllers}
    per_seed_savings = {name: [] for name in controllers}
    per_seed_nfire = {name: [] for name in controllers}
    per_seed_headroom = {name: [] for name in controllers}
    base_totals = []  # C0 totals for savings denominator

    for s in seeds:
        z = s["zvfs"]
        row = {"seed": s["seed"], "heldout_acc": s["heldout_acc"], "mean_zvf": s["mean_zvf"]}
        for name, fn in controllers.items():
            G_t = fn(z)
            m = per_seed_metrics(G_t, z)
            row[f"{name}_G_t"] = ",".join(str(g) for g in G_t)
            row[f"{name}_total_G"] = m["total_G"]
            row[f"{name}_n_fire"] = m["n_fire"]
            row[f"{name}_select_rate"] = m["select_rate"]
            row[f"{name}_headroom_bad"] = m["headroom_bad"]
            per_seed_total[name].append(m["total_G"])
            per_seed_nfire[name].append(m["n_fire"])
            per_seed_headroom[name].append(m["headroom_bad"])
        per_seed_savings["C0_baseline"].append(0.0)
        base_totals.append(row["C0_baseline_total_G"])
        for name in controllers:
            if name == "C0_baseline":
                continue
            saving = (row["C0_baseline_total_G"] - row[f"{name}_total_G"]) / row["C0_baseline_total_G"]
            row[f"{name}_savings"] = saving
            per_seed_savings[name].append(saving)
        rows.append(row)

    # Persist per-seed table
    out_tsv = OUT_DIR / "p7_dualformer_n10_per_seed.tsv"
    with out_tsv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Headline bootstrap-CI summary
    summary = {
        "config": {
            "tau": TAU,
            "tau_delta": TAU_DELTA,
            "G_base": G_BASE,
            "G_esc": G_ESC,
            "G_des": G_DES,
            "n_steps_per_seed": N_STEPS,
            "n_seeds": n_seeds,
            "n_boot": N_BOOT,
            "rng_seed": RNG_SEED,
        },
        "per_controller_total_G_boot_ci": {},
        "per_controller_savings_boot_ci": {},
        "paired_contrasts_savings": {},
        "per_seed_table": "experiments/results/p5p8/p7_dualformer_n10_per_seed.tsv",
    }

    for name in controllers:
        summary["per_controller_total_G_boot_ci"][name] = boot_ci(per_seed_total[name])
        if name == "C0_baseline":
            continue
        summary["per_controller_savings_boot_ci"][name] = boot_ci(per_seed_savings[name])

    # Pairwise savings contrasts (the headline falsifiable claim)
    c1 = f"C1_zvf_triage@{TAU:.2f}"
    c2 = f"C2_dualformer@{TAU:.2f}"
    c3 = f"C3_hybrid@{TAU:.2f}+{TAU_DELTA:.2f}"
    summary["paired_contrasts_savings"]["C2_minus_C1"] = paired_boot_ci_delta(
        per_seed_savings[c2], per_seed_savings[c1]
    )
    summary["paired_contrasts_savings"]["C3_minus_C1"] = paired_boot_ci_delta(
        per_seed_savings[c3], per_seed_savings[c1]
    )
    summary["paired_contrasts_savings"]["C3_minus_C2"] = paired_boot_ci_delta(
        per_seed_savings[c3], per_seed_savings[c2]
    )

    # Headroom (well-calibration check)
    summary["headroom_bad_per_controller"] = {
        name: {
            "sum": sum(per_seed_headroom[name]),
            "per_seed": per_seed_headroom[name],
        }
        for name in controllers
    }

    out_json = OUT_DIR / "p7_dualformer_n10_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Console echo
    print("=" * 72)
    print("P7 per-seed Dualformer / zvf-triage / Hybrid controller replay")
    print(f"N10 panel: {n_seeds} seeds × {N_STEPS} steps, G_base={G_BASE}")
    print(f"tau={TAU}, tau_delta={TAU_DELTA}, G_esc={G_ESC}, G_des={G_DES}")
    print("=" * 72)
    print(f"\nPer-controller total compute (sum G_t over 15 steps), bootstrap CI on n={n_seeds} seeds:")
    for name, ci in summary["per_controller_total_G_boot_ci"].items():
        print(f"  {name:35s}  mean={ci['mean']:7.3f}  95%CI=[{ci['lo']:7.3f}, {ci['hi']:7.3f}]")
    print(f"\nPer-controller savings vs baseline (fraction of compute NOT spent):")
    for name, ci in summary["per_controller_savings_boot_ci"].items():
        print(f"  {name:35s}  mean={ci['mean']:+.4f}  95%CI=[{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
    print(f"\nPaired bootstrap-CI on savings contrasts (headline):")
    for k, v in summary["paired_contrasts_savings"].items():
        sig = "***" if v["lo"] > 0 or v["hi"] < 0 else "n.s."
        print(f"  {k:20s}  Δ={v['mean_diff']:+.4f}  95%CI=[{v['lo']:+.4f}, {v['hi']:+.4f}]  {sig}")
    print(f"\nHeadroom-bad (steps fired on zvf>=0.99 — saturated, no escalation value):")
    for name, d in summary["headroom_bad_per_controller"].items():
        if name == "C0_baseline":
            continue
        print(f"  {name:35s}  total={d['sum']}  per_seed={d['per_seed']}")
    print(f"\nOutputs:\n  {out_tsv.relative_to(ROOT)}\n  {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()