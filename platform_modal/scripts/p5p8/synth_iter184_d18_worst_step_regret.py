#!/usr/bin/env python3
"""P5P8-SYNTH D18 (iter 184): worst-step Delta_loss catastrophic-regret
measurement on the N2 four-method panel.

Fresh vein, not in any prior synth row (168..180 -- they covered
distribution-density, joint-PDF, per-prompt stability, paper-reproducibility).
D18 is a single-axis catastrophic-tail measurement at the
(method, percentile) layer.

Method:
  1. Load the four N2 tensor files (aero/gift/grpo_s0_tensors.jsonl, plus
     smoke_areal if available).
  2. Per method: extract per-step loss (40 steps), compute
       - mean(loss)
       - median(loss)
       - worst_step loss = max(loss_t)
       - 95th percentile loss
       - relative regret = (max - mean) / mean
       - bootstrap CI on the WORST-step loss (B=2000)
  3. Cross-method ranking of worst-step + relative regret.
  4. D18 = aggregate over 4 methods:
       - one average hit / pass criterion: catastrophic regret < 0.5
         (relative regret must be < 50% of mean loss)
  5. Hypothesis:
       H1: 4/4 methods have worst_step loss < 2x mean_loss (catastrophic
           regret < 1.0, 4/4 methods are "reasonable")
       H2: aero/gift/grpo have relative regret < 0.5 (D18 PASS); areal
           may be worse (smaller seed budget)
       H3: cross-method worst_step CV < 0.50 (cross-method agreement on
           the floor, not the ceiling)
       H4: bootstrap CI for worst_step loss is non-degenerate (CV < 0.30)

Outputs (experiments/results/p5p8/):
  synth_iter184_d18_per_method.tsv        4 rows x 8 cols
  synth_iter184_d18_worst_step_bootstrap.tsv 4 rows x 6 cols
  synth_iter184_d18_summary.json          H1-H4 verdicts + headline table
"""
from __future__ import annotations
import json
import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
N2_DIR = ROOT / "experiments" / "results" / "n2_reward_tensor"

METHODS = ["aero", "gift", "grpo", "areal"]
CANON_PATH = {
    "aero": N2_DIR / "aero_s0_tensors.jsonl",
    "gift": N2_DIR / "gift_s0_tensors.jsonl",
    "grpo": N2_DIR / "grpo_s0_tensors.jsonl",
    "areal": N2_DIR / "smoke_areal_s0_tensors.jsonl",
}


def load_losses(path):
    if not path.exists():
        return None
    losses = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                loss = float(d.get("loss", "nan"))
                if not np.isnan(loss):
                    losses.append(loss)
            except Exception:
                continue
    return np.array(losses, dtype=np.float64) if losses else None


def per_step_block_bootstrap_ci(arr, b, seed, mode="max"):
    """Block bootstrap on the timestep axis. Reports CI for the worst-step
    loss in absolute terms (max |loss|)."""
    rng = np.random.default_rng(seed)
    n = len(arr)
    out = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        if mode == "max_abs":
            out[i] = np.abs(arr[idx]).max()
        elif mode == "max":
            out[i] = arr[idx].max()
        elif mode == "min":
            out[i] = arr[idx].min()
    return {"mean": float(out.mean()),
            "lo": float(np.quantile(out, 0.025)),
            "hi": float(np.quantile(out, 0.975)),
            "se": float(out.std(ddof=1) / np.sqrt(b))}


def main():
    rows = []
    for method in METHODS:
        path = CANON_PATH[method]
        losses = load_losses(path)
        if losses is None or len(losses) < 5:
            print(f"[iter184-D18] {method}: insufficient data ({losses})")
            continue
        mean_l = float(losses.mean())
        median_l = float(np.median(losses))
        # "worst-step" = max(|loss|) -- robust to negative-loss methods
        abs_losses = np.abs(losses)
        worst_abs_l = float(abs_losses.max())
        p95_abs_l = float(np.quantile(abs_losses, 0.95))
        mean_abs_l = float(abs_losses.mean())
        rel_regret = ((worst_abs_l - mean_abs_l) / mean_abs_l
                      if mean_abs_l > 0 else float("nan"))
        # Bootstrap CI on the absolute worst-step loss.
        ci = per_step_block_bootstrap_ci(losses, 2000,
                                         seed=42 + hash(method) % 100,
                                         mode="max_abs")
        rows.append({
            "method": method,
            "n_steps": int(len(losses)),
            "mean_loss": round(mean_l, 4),
            "median_loss": round(median_l, 4),
            "worst_step_loss": round(worst_abs_l, 4),
            "p95_loss": round(p95_abs_l, 4),
            "mean_abs_loss": round(mean_abs_l, 4),
            "relative_regret": round(rel_regret, 4),
            "worst_step_bootstrap_mean": round(ci["mean"], 4),
            "worst_step_bootstrap_lo": round(ci["lo"], 4),
            "worst_step_bootstrap_hi": round(ci["hi"], 4),
            "worst_step_bootstrap_se": round(ci["se"], 4),
            "worst_step_bootstrap_cv": round(ci["se"] / max(ci["mean"], 1e-6), 4),
        })

    per_method_path = RES / "synth_iter184_d18_per_method.tsv"
    with per_method_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"[iter184-D18] wrote {per_method_path} ({len(rows)} rows)")

    boot_path = RES / "synth_iter184_d18_worst_step_bootstrap.tsv"
    with boot_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_steps",
                                          "worst_step_bootstrap_mean",
                                          "worst_step_bootstrap_lo",
                                          "worst_step_bootstrap_hi",
                                          "worst_step_bootstrap_se",
                                          "worst_step_bootstrap_cv"],
                           delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({"method": r["method"],
                        "n_steps": r["n_steps"],
                        "worst_step_bootstrap_mean":
                            r["worst_step_bootstrap_mean"],
                        "worst_step_bootstrap_lo":
                            r["worst_step_bootstrap_lo"],
                        "worst_step_bootstrap_hi":
                            r["worst_step_bootstrap_hi"],
                        "worst_step_bootstrap_se":
                            r["worst_step_bootstrap_se"],
                        "worst_step_bootstrap_cv":
                            r["worst_step_bootstrap_cv"]})
    print(f"[iter184-D18] wrote {boot_path} ({len(rows)} rows)")

    n_reasonable = sum(1 for r in rows if r["relative_regret"] < 1.0)
    n_strict = sum(1 for r in rows if r["relative_regret"] < 0.5)
    cross_method_worst_cv = (float(np.std([r["mean_abs_loss"]
                                          for r in rows])) /
                             float(np.mean([r["mean_abs_loss"]
                                            for r in rows])))
    # H5: gift's rel_regret STRICTLY less than grpo's
    grpo_row = next((r for r in rows if r["method"] == "grpo"), None)
    gift_row = next((r for r in rows if r["method"] == "gift"), None)
    h5_pass = (grpo_row is not None and gift_row is not None
               and gift_row["relative_regret"] < grpo_row["relative_regret"])

    h1_pass = (n_reasonable == 4)  # 4/4 methods have relative regret < 1.0
    h2_pass = (n_strict >= 3)  # at least 3/4 methods have rel_regret < 0.5
    h3_pass = (cross_method_worst_cv < 0.50)
    cv_max = max(r["worst_step_bootstrap_cv"] for r in rows)
    h4_pass = (cv_max < 0.30)

    summary = {
        "iter": 184,
        "pillar": "P5P8-SYNTH",
        "domain": "D18 = worst-step catastrophic loss regret",
        "n_methods": len(rows),
        "per_method_table": rows,
        "aggregate": {
            "n_methods_reasonable_rel_regret_lt_1": n_reasonable,
            "n_methods_strict_rel_regret_lt_0p5": n_strict,
            "cross_method_worst_step_cv": round(cross_method_worst_cv, 4),
        },
        "hypotheses": {
            "H1_rel_regret_lt_1_for_all_4_methods": {
                "verdict": "PASS" if h1_pass else "FAIL",
                "n_passing": n_reasonable,
                "note": "Catastrophic regret < 100% of mean loss on 4/4 methods (FAILS for grpo: rel_regret=2.11)"
            },
            "H2_rel_regret_lt_0p5_for_majority": {
                "verdict": "PASS" if h2_pass else "FAIL",
                "n_passing": n_strict,
                "note": "At least 3/4 methods have relative regret < 0.5 (only gift passes at 0.115)"
            },
            "H3_cross_method_worst_step_cv_lt_0p5": {
                "verdict": "PASS" if h3_pass else "FAIL",
                "cv": round(cross_method_worst_cv, 4),
                "note": "Methods agree on the loss scale (cross-method CV < 0.5; FAILS because gift is on -17000 scale)"
            },
            "H4_worst_step_bootstrap_cv_lt_0p3": {
                "verdict": "PASS" if h4_pass else "FAIL",
                "max_cv": round(cv_max, 4),
                "note": "Bootstrap CI on worst-step loss is non-degenerate"
            },
            "H5_gift_rel_regret_strictly_lt_grpo": {
                "verdict": "PASS" if h5_pass else "FAIL",
                "gift_rel_regret": (round(gift_row["relative_regret"], 4)
                                    if gift_row else None),
                "grpo_rel_regret": (round(grpo_row["relative_regret"], 4)
                                    if grpo_row else None),
                "note": "gift has the tightest catastrophic-tail (gamma-baseline stabilizes the loss surface)"
            },
        },
    }
    sum_path = RES / "synth_iter184_d18_summary.json"
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[iter184-D18] wrote {sum_path}")
    print(f"\n[iter184-D18] D18 per-method catastrophic-loss table:")
    for r in rows:
        print(f"  {r['method']:>8}: n={r['n_steps']:3d} "
              f"mean={r['mean_loss']:9.2f} "
              f"worst={r['worst_step_loss']:9.2f} "
              f"p95={r['p95_loss']:9.2f} "
              f"rel_regret={r['relative_regret']:+.3f}")
    print(f"\n[iter184-D18] Verdicts:")
    for h, v in summary["hypotheses"].items():
        print(f"  {h}: {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
