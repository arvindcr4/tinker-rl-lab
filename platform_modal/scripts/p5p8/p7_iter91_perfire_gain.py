"""Iter 91 — P7 per-fire contrast gain (closed-form binomial Δ_ZVF per fired step).

Vein (fresh, not in 107 prior rows):
The P7 controller family (iter 67, 71, 75, 79, 83, 87, 88) reports fires, saves,
flips, hysteresis, and postpred restore_prob, but **never** reports the
closed-form expected ZVF-DROP per fired step — the metric that justifies the
controller's existence ("did the fire actually restore contrast?").

Iter 91 computes, on the real N2 four-method reward tensors, the per-(method,
step) closed-form Δ_ZVF(8→16) under the binomial model and ranks the
per-method Pareto frontier by **benefit per 1000 extra rollouts in
ZVF-drop units** — a falsifiable, per-fire metric that decouples the "fire
count" axis from the "fire value" axis.

The CORRECT metric is per-prompt binomial-predicted ZVF (NOT empirical
boundary fraction). For each prompt p with k_p successes in 8 rollouts:
  p̂_p = k_p / 8
  z_8(p̂) = p̂^8 + (1-p̂)^8    (binomial ZVF at G=8)
  z_16(p̂) = p̂^16 + (1-p̂)^16  (binomial ZVF at G=16)
  Δ_z(p̂) = z_8(p̂) - z_16(p̂)  (per-prompt benefit of escalating)
Per-step benefit: (1/16) Σ_p Δ_z(p̂_p).

Boundary prompts (k=0, 8): Δ_z = 0 (both z_8 and z_16 = 1)
Mixed prompts (k=1, 7): Δ_z ≈ 0.24 (z_8 = 0.34, z_16 = 0.10)
Mid prompts (k=2..6): Δ_z < 0.01

Outputs:
  platform_hybrid/experiments/results/p5p8/p7_iter91_perfire_gain_per_step.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter91_perfire_gain_per_method.tsv
  platform_hybrid/experiments/results/p5p8/p7_iter91_perfire_gain_summary.json
  platform_hybrid/experiments/results/p5p8/p7_iter91_perfire_gain_pareto.tsv
"""
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

WORK = Path("/home/claude/tinker-rl-lab-minimax")
N2_DIR = WORK / "platform_hybrid/experiments/results/n2_reward_tensor_resume"
OUT_DIR = WORK / "platform_hybrid/experiments/results/p5p8"

METHODS = ["grpo", "aero", "gift", "areal"]
G_BASE = 8
G_ESC = 16
N_STEPS = 40
N_PROMPTS = 16
N_BOOT = 4000
SEED = 20260705

TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ETA_MIN_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

# ---------- Closed-form helpers ----------

def zvf_binom(p_hat: float, G: int) -> float:
    """i.i.d. binomial predicted ZVF at group size G for an observed success rate p_hat."""
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    return p ** G + (1.0 - p) ** G


def restore_at_G(p_hat: float, G: int) -> float:
    return 1.0 - zvf_binom(p_hat, G)


# ---------- Load N2 tensors ----------

def load_n2():
    by_method = {}
    for m in METHODS:
        path = N2_DIR / f"{m}_s0_tensors.jsonl"
        rows = [json.loads(l) for l in open(path)]
        rows.sort(key=lambda r: r["step"])
        by_method[m] = rows
    return by_method


# ---------- Per-step metrics ----------

def compute_per_step_metrics(by_method):
    """For each (method, step) compute the per-prompt binomial ZVF drop from escalating."""
    per_step = []
    for m, rows in by_method.items():
        for r in rows:
            rewards = r["rewards"]
            ks = [int(round(sum(p))) for p in rewards]
            # Per-prompt binomial ZVF at G_BASE and G_ESC
            p_hats = [k / G_BASE for k in ks]
            z8_per_p = [zvf_binom(p, G_BASE) for p in p_hats]
            z16_per_p = [zvf_binom(p, G_ESC) for p in p_hats]
            mean_z8 = sum(z8_per_p) / len(z8_per_p)
            mean_z16 = sum(z16_per_p) / len(z16_per_p)
            zvf_drop = mean_z8 - mean_z16  # per-step mean ZVF reduction
            # Per-step restore_sum (iter 26 metric): sum_p (1 - z16_per_p)
            restore_sum = sum(1.0 - z for z in z16_per_p)
            # Per-step zvf_obs (empirical boundary fraction)
            zvf_obs = sum(1 for k in ks if k in (0, G_BASE)) / len(ks)
            per_step.append({
                "method": m,
                "step": r["step"],
                "n_prompts": len(ks),
                "n_degenerate_at_base": sum(1 for k in ks if k in (0, G_BASE)),
                "zvf_obs": zvf_obs,
                "mean_zvf_at_g8": mean_z8,
                "mean_zvf_at_g16": mean_z16,
                "zvf_drop": zvf_drop,
                "restore_sum": restore_sum,
                "n_k0": sum(1 for k in ks if k == 0),
                "n_k8": sum(1 for k in ks if k == 8),
            })
    return per_step


# ---------- Controller replay ----------

def replay_controller(per_step, controller: str, threshold, eta_min: float = 0.0):
    fired = []
    for s in per_step:
        if controller == "zvf_triage":
            fired_now = s["zvf_obs"] >= threshold
        elif controller == "zvf_then_drop":
            fired_now = (s["zvf_obs"] >= threshold) and (s["zvf_drop"] >= eta_min)
        elif controller == "drop_gated":
            fired_now = s["zvf_drop"] >= threshold
        else:
            raise ValueError(controller)
        if fired_now:
            fired.append(s)
    return fired


# ---------- Bootstrap CI ----------

def boot_mean_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return sum(values) / n, lo, hi


# ---------- Main ----------

def main():
    by_method = load_n2()
    per_step = compute_per_step_metrics(by_method)
    print(f"Loaded {len(per_step)} (method,step) rows")

    zvf_obs_means = {m: sum(s["zvf_obs"] for s in per_step if s["method"] == m) / N_STEPS
                     for m in METHODS}
    zvf_drop_means = {m: sum(s["zvf_drop"] for s in per_step if s["method"] == m) / N_STEPS
                      for m in METHODS}
    restore_means = {m: sum(s["restore_sum"] for s in per_step if s["method"] == m) / N_STEPS
                     for m in METHODS}
    mean_z8 = {m: sum(s["mean_zvf_at_g8"] for s in per_step if s["method"] == m) / N_STEPS
               for m in METHODS}
    print(f"Mean zvf_obs per method: {zvf_obs_means}")
    print(f"Mean z8 (binomial avg) per method: {mean_z8}")
    print(f"Mean zvf_drop per method: {zvf_drop_means}")
    print(f"Mean restore_sum per method: {restore_means}")

    # Write per-step TSV
    per_step_path = OUT_DIR / "p7_iter91_perfire_gain_per_step.tsv"
    with open(per_step_path, "w") as f:
        cols = ["method", "step", "n_prompts", "n_degenerate_at_base", "zvf_obs",
                "mean_zvf_at_g8", "mean_zvf_at_g16", "zvf_drop", "restore_sum",
                "n_k0", "n_k8"]
        f.write("\t".join(cols) + "\n")
        for s in per_step:
            f.write("\t".join(str(s[c]) for c in cols) + "\n")
    print(f"Wrote {per_step_path}")

    # Controller sweep — for each method, replay each (controller, τ, eta) combo
    rows = []
    for m in METHODS:
        mps = [s for s in per_step if s["method"] == m]
        # zvf_triage sweep over τ
        for tau in TAUS:
            fired = [s for s in mps if s["zvf_obs"] >= tau]
            n_fires = len(fired)
            if n_fires > 0:
                zvf_drops = [s["zvf_drop"] for s in fired]
                mean_drop, lo, hi = boot_mean_ci(zvf_drops)
                sum_drop = sum(zvf_drops)
                sum_restore = sum(s["restore_sum"] for s in fired)
            else:
                mean_drop = lo = hi = sum_drop = sum_restore = 0.0
            extra_rollouts = n_fires * (G_ESC - G_BASE) * N_PROMPTS
            benefit_per_1k = (sum_drop * N_PROMPTS) / extra_rollouts * 1000.0 if extra_rollouts > 0 else 0.0
            rows.append({
                "method": m,
                "controller": "zvf_triage",
                "threshold": f"{tau:.2f}",
                "eta_min": "0.0",
                "n_fires": n_fires,
                "extra_rollouts": extra_rollouts,
                "sum_zvf_drop": round(sum_drop, 4),
                "mean_zvf_drop_per_fire": round(mean_drop, 4),
                "boot_ci_lo": round(lo, 4),
                "boot_ci_hi": round(hi, 4),
                "sum_restore_prompts": round(sum_restore, 2),
                "zvf_drop_per_1k_rollouts": round(benefit_per_1k, 4),
            })
        # zvf_then_drop: requires BOTH step-ZVF >= tau AND per-step zvf_drop >= eta
        for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            for eta in ETA_MIN_GRID:
                fired = [s for s in mps if s["zvf_obs"] >= tau and s["zvf_drop"] >= eta]
                n_fires = len(fired)
                if n_fires > 0:
                    zvf_drops = [s["zvf_drop"] for s in fired]
                    mean_drop, lo, hi = boot_mean_ci(zvf_drops)
                    sum_drop = sum(zvf_drops)
                    sum_restore = sum(s["restore_sum"] for s in fired)
                else:
                    mean_drop = lo = hi = sum_drop = sum_restore = 0.0
                extra_rollouts = n_fires * (G_ESC - G_BASE) * N_PROMPTS
                benefit_per_1k = (sum_drop * N_PROMPTS) / extra_rollouts * 1000.0 if extra_rollouts > 0 else 0.0
                rows.append({
                    "method": m,
                    "controller": "zvf_then_drop",
                    "threshold": f"{tau:.2f}",
                    "eta_min": f"{eta:.3f}",
                    "n_fires": n_fires,
                    "extra_rollouts": extra_rollouts,
                    "sum_zvf_drop": round(sum_drop, 4),
                    "mean_zvf_drop_per_fire": round(mean_drop, 4),
                    "boot_ci_lo": round(lo, 4),
                    "boot_ci_hi": round(hi, 4),
                    "sum_restore_prompts": round(sum_restore, 2),
                    "zvf_drop_per_1k_rollouts": round(benefit_per_1k, 4),
                })
        # drop_gated: threshold is the eta value itself
        for eta in ETA_MIN_GRID:
            fired = [s for s in mps if s["zvf_drop"] >= eta]
            n_fires = len(fired)
            if n_fires > 0:
                zvf_drops = [s["zvf_drop"] for s in fired]
                mean_drop, lo, hi = boot_mean_ci(zvf_drops)
                sum_drop = sum(zvf_drops)
                sum_restore = sum(s["restore_sum"] for s in fired)
            else:
                mean_drop = lo = hi = sum_drop = sum_restore = 0.0
            extra_rollouts = n_fires * (G_ESC - G_BASE) * N_PROMPTS
            benefit_per_1k = (sum_drop * N_PROMPTS) / extra_rollouts * 1000.0 if extra_rollouts > 0 else 0.0
            rows.append({
                "method": m,
                "controller": "drop_gated",
                "threshold": "n/a",
                "eta_min": f"{eta:.3f}",
                "n_fires": n_fires,
                "extra_rollouts": extra_rollouts,
                "sum_zvf_drop": round(sum_drop, 4),
                "mean_zvf_drop_per_fire": round(mean_drop, 4),
                "boot_ci_lo": round(lo, 4),
                "boot_ci_hi": round(hi, 4),
                "sum_restore_prompts": round(sum_restore, 2),
                "zvf_drop_per_1k_rollouts": round(benefit_per_1k, 4),
            })

    per_method_path = OUT_DIR / "p7_iter91_perfire_gain_per_method.tsv"
    with open(per_method_path, "w") as f:
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"Wrote {per_method_path}")

    # Pareto summary: per (controller, threshold, eta) cross-method mean
    pareto_groups = defaultdict(list)
    pareto_meta = {}
    for r in rows:
        key = (r["controller"], r["threshold"], r["eta_min"])
        pareto_groups[key].append(r["zvf_drop_per_1k_rollouts"])
        pareto_meta[key] = {
            "n_fires_total": pareto_meta.get(key, {}).get("n_fires_total", 0) + r["n_fires"],
            "sum_zvf_drop_total": pareto_meta.get(key, {}).get("sum_zvf_drop_total", 0.0) + r["sum_zvf_drop"],
            "extra_rollouts_total": pareto_meta.get(key, {}).get("extra_rollouts_total", 0) + r["extra_rollouts"],
        }

    pareto_path = OUT_DIR / "p7_iter91_perfire_gain_pareto.tsv"
    items = []
    for (ctrl, thr, eta), values in pareto_groups.items():
        mean_v, lo, hi = boot_mean_ci(values, n_boot=2000, seed=SEED)
        meta = pareto_meta[(ctrl, thr, eta)]
        items.append((ctrl, thr, eta, meta["n_fires_total"],
                      meta["sum_zvf_drop_total"], meta["extra_rollouts_total"],
                      mean_v, lo, hi))
    items.sort(key=lambda x: -x[6])  # by mean benefit desc

    with open(pareto_path, "w") as f:
        f.write("controller\tthreshold\teta_min\tn_fires_total\tsum_zvf_drop_total\textra_rollouts_total\tzvf_drop_per_1k_mean\tci_lo\tci_hi\n")
        for ctrl, thr, eta, n, sd, er, m, lo, hi in items:
            f.write(f"{ctrl}\t{thr}\t{eta}\t{n}\t{sd:.4f}\t{er}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}\n")
    print(f"Wrote {pareto_path}")

    # JSON summary
    summary = {
        "n_total_observations": len(per_step),
        "n_methods": len(METHODS),
        "n_steps_per_method": N_STEPS,
        "n_prompts_per_step": N_PROMPTS,
        "zvf_obs_mean_per_method": zvf_obs_means,
        "zvf_drop_mean_per_method": zvf_drop_means,
        "restore_sum_mean_per_method": restore_means,
        "pareto_top10": [
            {"controller": ctrl, "threshold": thr, "eta_min": eta,
             "n_fires_total": n, "sum_zvf_drop_total": round(sd, 4),
             "extra_rollouts_total": er, "zvf_drop_per_1k_mean": round(m, 4),
             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
            for ctrl, thr, eta, n, sd, er, m, lo, hi in items[:10]
        ],
        "n_boot": N_BOOT,
        "seed": SEED,
    }
    summary_path = OUT_DIR / "p7_iter91_perfire_gain_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n=== TOP-10 PARETO (zvf_drop per 1000 extra rollouts, cross-method) ===")
    for ctrl, thr, eta, n, sd, er, m, lo, hi in items[:10]:
        print(f"  {ctrl:14s} thr={thr:5s} eta={eta:5s} fires={n:4d} "
              f"zvf_drop_sum={sd:7.4f} rolls={er:6d} benefit={m:7.4f} CI=[{lo:.4f},{hi:.4f}]")


if __name__ == "__main__":
    main()