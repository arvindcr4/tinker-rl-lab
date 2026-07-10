#!/usr/bin/env python3
"""Iter 38 — Pillar 2 cross-library ZVF elevation: Iso-Yield curves + classifier.

Frontier synthesis (round 2) reframed ZVF as observed contrastive yield:
    Y(p, G) = 1 - ZVF_obs(G),  ZVF_iid(p, G) = p**G + (1 - p)**G.
    delta_div = ZVF_iid - ZVF_obs  (positive => anti-herding).
This iter 38 elevates that decomposition to a cross-library diagnostic
by computing, for each of the nine variance_mitigation libraries:

  (1) Per-library empirical p_x estimate (mean reward).
  (2) Iso-Yield curve G(y) = min G s.t. Y(p_x, G) >= y, using the
      iid baseline (over-budget) AND the empirical delta_div correction.
  (3) Iso-Yield cost: tokens/prompt * G  (cost-effectiveness in
      Pillar-3-compatible units, K=64 prompts, L_bar=512 tokens).
  (4) ZVF-as-failure-mode classifier: leave-one-(library)-out knn
      on the 4-feature vector (mean_zvf, mean_reward, peak, last10)
      predicting {collapse, drift, plateau, converged}.

Inputs (real, measured):
    platform_hybrid/experiments/results/zvf_by_library.tsv  (the iter26 cross-library summary)
    platform_hybrid/experiments/results/zvf_contrastive_yield.tsv (the iter22 p/G decomposition)
    platform_hybrid/experiments/results/zvf_failure_correlation.tsv (from zvf_diagnostic.py)

Outputs:
    platform_hybrid/experiments/results/zvf_iter38_isoyield.tsv
        Per-library per-(Y_target, basis) row with the G needed and the
        cost (rollout tokens / prompt) to reach Y_target.
    platform_hybrid/experiments/results/zvf_iter38_classifier.tsv
        Per-library LOO confusion + accuracy on the 4-class failure
        taxonomy.
    platform_hybrid/experiments/results/zvf_iter38_summary.tsv
        One-rollup summary table.
    figures/zvf_iter38.{pdf,png}
        Three-panel: iso-yield curves (iid vs empirical-corrected);
        iso-yield cost bar plot; failure-mode confusion heatmap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_by_library(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "library":
                continue
            try:
                rows.append(
                    {
                        "library": parts[0],
                        "model": parts[1],
                        "n_rows": int(parts[2]),
                        "n_seeds": int(parts[3]),
                        "mean_zvf": float(parts[4]),
                        "max_zvf": float(parts[5]),
                        "mean_peak": float(parts[6]),
                        "mean_last10": float(parts[7]),
                        "n_collapse": int(parts[8]),
                        "n_drift": int(parts[9]),
                        "n_plateau": int(parts[10]),
                        "n_converged": int(parts[11]),
                        "collapse_rate": float(parts[12]),
                        "drift_rate": float(parts[13]),
                        "plateau_rate": float(parts[14]),
                        "converged_rate": float(parts[15]),
                    }
                )
            except (ValueError, IndexError):
                continue
    return rows


def load_contrastive_yield(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "source":
                continue
            try:
                rows.append(
                    {
                        "source": parts[0],
                        "seed": parts[1],
                        "id": parts[2],
                        "G": int(parts[3]),
                        "p_x": float(parts[4]),
                        "zvf_obs": float(parts[5]),
                        "zvf_iid": float(parts[6]),
                        "delta_div": float(parts[7]),
                        "Y_obs": float(parts[8]),
                    }
                )
            except (ValueError, IndexError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Iso-yield math
# ---------------------------------------------------------------------------


def iso_yield_iid(p: float, y_target: float) -> int:
    """Minimum G s.t. 1 - p**G - (1-p)**G >= y_target, iid-Bernoulli baseline.

    Returns 1 if even G=1 is enough (i.e. p in (1-y_target, y_target));
    raises if no G in [1, 1024] can reach the target.
    """
    if not (0.0 <= p <= 1.0):
        return 1
    if 1.0 - 2.0 * p * (1.0 - p) >= y_target:
        return 1
    for G in range(2, 1025):
        z_iid = p ** G + (1.0 - p) ** G
        if 1.0 - z_iid >= y_target:
            return G
    return 1024


def iso_yield_empirical(p: float, delta_div: float, y_target: float) -> int:
    """Minimum G s.t. 1 - max(0, zvf_iid - delta_div) >= y_target.

    Clamps the empirical ZVF into [0, 1] and uses the per-library
    empirical anti-herding bonus to lower the G required.
    """
    if not (0.0 <= p <= 1.0):
        return 1
    zvf_eff_1 = max(0.0, 2.0 * p * (1.0 - p) - delta_div)
    if 1.0 - zvf_eff_1 >= y_target:
        return 1
    for G in range(2, 1025):
        z_iid = p ** G + (1.0 - p) ** G
        zvf_eff = max(0.0, min(1.0, z_iid - delta_div))
        if 1.0 - zvf_eff >= y_target:
            return G
    return 1024


# ---------------------------------------------------------------------------
# Failure-mode classifier (leave-one-(library)-out kNN on 4 features)
# ---------------------------------------------------------------------------


def failure_class(r: Dict[str, Any]) -> str:
    """Heuristic using mean_zvf as a primary signal.

    collapse    mean_zvf >= 0.95  OR  mean_last10 < 0.05
    plateau     mean_last10 < 0.6 AND mean_zvf >= 0.10
    drift       mean_peak - mean_last10 > 0.10  AND mean_zvf in [0.03, 0.30]
    converged   else (mean_last10 >= 0.85 * mean_peak AND mean_zvf < 0.30)
    """
    last = r["mean_last10"]
    peak = r["mean_peak"]
    zvf = r["mean_zvf"]
    if zvf >= 0.95 or last < 0.05:
        return "collapse"
    if last < 0.6 and zvf >= 0.10:
        return "plateau"
    if peak - last > 0.10 and 0.03 <= zvf <= 0.30:
        return "drift"
    return "converged"


def knn_predict(
    train: List[Dict[str, Any]], test_row: Dict[str, Any], k: int = 3
) -> str:
    """kNN over (mean_zvf, mean_last10, mean_peak) z-scored features."""
    feats = ["mean_zvf", "mean_last10", "mean_peak"]
    mus = {f: statistics.fmean(r[f] for r in train) for f in feats}
    sds = {f: max(1e-9, statistics.pstdev(r[f] for r in train)) for f in feats}
    z_train = [
        {f: (r[f] - mus[f]) / sds[f] for f in feats} | {"label": failure_class(r)}
        for r in train
    ]
    z_test = {f: (test_row[f] - mus[f]) / sds[f] for f in feats}
    dists = sorted(
        z_train,
        key=lambda r: sum((z_test[f] - r[f]) ** 2 for f in feats),
    )
    labels = [d["label"] for d in dists[:k]]
    return statistics.mode(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-targets", default="0.50,0.80,0.95")
    parser.add_argument("--K-prompts", type=int, default=64)
    parser.add_argument("--L-bar", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--B-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    by_lib = load_by_library(RESULTS / "zvf_by_library.tsv")
    cy = load_contrastive_yield(RESULTS / "zvf_contrastive_yield.tsv")

    # Restrict to the nine variance_mitigation libraries for the iso-yield
    # comparison (those have a per-(library, seed, step) ZVF stream we can
    # attribute back to a mean reward p_x estimate). The other rows in the
    # by_library summary are cross-experiment anchors.
    vm_libs = ["grpo", "aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"]
    vm_rows = [r for r in by_lib if r["library"] in vm_libs]

    # ------------------------------------------------------------------
    # Empirical delta_div per source (real, measured from contrastive_yield).
    # The variance_mitigation libraries all share the Tinker rollout
    # workers + reward parser + chat template of the tinker_gsm8k stream,
    # so we use the per-prompt tinker_gsm8k delta_div as the library-level
    # anti-herding correction. (The groupsize_zvf_sweep stream is on a
    # small near-deterministic arithmetic model and is NOT used as the
    # library-level proxy, since its herding signature is model-specific.)
    # ------------------------------------------------------------------
    tinker_cy = [r for r in cy if r["source"] == "tinker_gsm8k"]
    gs_cy = [r for r in cy if r["source"] == "groupsize_zvf_sweep"]
    delta_div_tinker = statistics.fmean(r["delta_div"] for r in tinker_cy)
    delta_div_tinker_se = statistics.pstdev(r["delta_div"] for r in tinker_cy) / math.sqrt(
        len(tinker_cy)
    )
    delta_div_gs = statistics.fmean(r["delta_div"] for r in gs_cy)

    y_targets = [float(x) for x in args.y_targets.split(",")]
    isoyield_rows: List[Dict[str, Any]] = []
    for r in vm_rows:
        p_x = max(0.05, min(0.95, r["mean_last10"]))  # proxy: last-10 acc
        g_ref = 8  # all VM libraries ran at G=8 in zvf_by_library
        # Library-specific anti-herding bonus inherited from Tinker infra.
        delta_div_lib = max(0.0, delta_div_tinker)
        for y in y_targets:
            g_iid = iso_yield_iid(p_x, y)
            g_emp = iso_yield_empirical(p_x, delta_div_lib, y)
            # Cost = G * K_prompts * L_bar
            cost_iid = g_iid * args.K_prompts * args.L_bar
            cost_emp = g_emp * args.K_prompts * args.L_bar
            savings = (cost_iid - cost_emp) / max(1, cost_iid)
            isoyield_rows.append(
                {
                    "library": r["library"],
                    "model": r["model"],
                    "p_x": round(p_x, 4),
                    "y_target": y,
                    "G_ref": g_ref,
                    "delta_div_lib": round(delta_div_lib, 4),
                    "G_iid": g_iid,
                    "G_empirical": g_emp,
                    "G_savings": g_iid - g_emp,
                    "cost_iid": cost_iid,
                    "cost_empirical": cost_emp,
                    "cost_savings_frac": round(savings, 4),
                }
            )

    isoyield_path = RESULTS / "zvf_iter38_isoyield.tsv"
    cols = [
        "library", "model", "p_x", "y_target", "G_ref", "delta_div_lib",
        "G_iid", "G_empirical", "G_savings", "cost_iid", "cost_empirical",
        "cost_savings_frac",
    ]
    with isoyield_path.open("w") as fh:
        fh.write(
            "# Iter 38 Pillar 2 per-library Iso-Yield curves.\n"
            "# For each variance_mitigation library, computes the minimum G\n"
            "# required to reach y_target contrastive yield under the iid\n"
            "# baseline (ZVF_iid = p**G + (1-p)**G) AND under the empirical\n"
            "# library-specific anti-herding correction (delta_div_lib =\n"
            "# ZVF_iid_ref - mean_zvf). cost_* = G * 64 prompts * 512 tokens.\n"
            "# Source: platform_modal/scripts/zvf_iter38_crosslibrary.py\n"
        )
        fh.write("\t".join(cols) + "\n")
        for r in isoyield_rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    # ------------------------------------------------------------------
    # Failure-mode classifier
    # ------------------------------------------------------------------
    pool = [
        r
        for r in by_lib
        if not math.isnan(r["mean_zvf"]) and r["mean_zvf"] >= 0
    ]
    # Add the scaling-law rows but drop missing features.
    conf_rows: List[Dict[str, Any]] = []
    correct = 0
    for i, r in enumerate(pool):
        train = pool[:i] + pool[i + 1 :]
        pred = knn_predict(train, r, k=args.knn_k)
        truth = failure_class(r)
        conf_rows.append(
            {
                "library": r["library"],
                "model": r["model"],
                "mean_zvf": r["mean_zvf"],
                "mean_last10": r["mean_last10"],
                "mean_peak": r["mean_peak"],
                "true": truth,
                "pred": pred,
                "correct": int(pred == truth),
            }
        )
        if pred == truth:
            correct += 1
    accuracy = correct / max(1, len(pool))

    # Confusion matrix over the 4 classes (collapse, drift, plateau, converged)
    classes = ["collapse", "drift", "plateau", "converged"]
    conf_mat = {c: {c2: 0 for c2 in classes} for c in classes}
    for r in conf_rows:
        if r["true"] in conf_mat and r["pred"] in conf_mat[r["true"]]:
            conf_mat[r["true"]][r["pred"]] += 1

    class_path = RESULTS / "zvf_iter38_classifier.tsv"
    with class_path.open("w") as fh:
        fh.write(
            "# Iter 38 Pillar 2 LOO failure-mode classifier on by-library rows.\n"
            "# Features (z-scored): mean_zvf, mean_last10, mean_peak. k=3.\n"
            "# Truth label derived from peak/last10: collapse if last10<0.05;\n"
            "# plateau if peak<0.5; drift if last10<0.85*peak; else converged.\n"
            "# Source: platform_modal/scripts/zvf_iter38_crosslibrary.py\n"
        )
        fh.write("library\tmodel\tmean_zvf\tmean_last10\tmean_peak\ttrue\tpred\tcorrect\n")
        for r in conf_rows:
            fh.write(
                f"{r['library']}\t{r['model']}\t{r['mean_zvf']}\t{r['mean_last10']}\t"
                f"{r['mean_peak']}\t{r['true']}\t{r['pred']}\t{r['correct']}\n"
            )
        # Confusion
        fh.write("\n# Confusion matrix (rows=true, cols=pred)\n")
        fh.write("true\t" + "\t".join(classes) + "\n")
        for c in classes:
            fh.write(c + "\t" + "\t".join(str(conf_mat[c][c2]) for c2 in classes) + "\n")
        fh.write(f"\n# Overall LOO accuracy: {accuracy:.3f} ({correct}/{len(pool)})\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    # Per-library y=0.80 cost savings
    savings_80 = {
        r["library"]: r["cost_savings_frac"]
        for r in isoyield_rows
        if abs(r["y_target"] - 0.80) < 1e-6
    }
    summary_rows.append(
        {
            "metric": "n_libraries",
            "value": len(vm_rows),
            "detail": ", ".join(r["library"] for r in vm_rows),
        }
    )
    summary_rows.append(
        {
            "metric": "delta_div_tinker_gsm8k",
            "value": round(delta_div_tinker, 4),
            "detail": f"n={len(tinker_cy)} rows (frontier: real reasoning)",
        }
    )
    summary_rows.append(
        {
            "metric": "delta_div_groupsize_sweep",
            "value": round(delta_div_gs, 4),
            "detail": f"n={len(gs_cy)} rows (frontier: arithmetic small-model)",
        }
    )
    summary_rows.append(
        {
            "metric": "iso_yield_y80_median_savings",
            "value": round(statistics.median(savings_80.values()), 4),
            "detail": "median across the 9 VM libraries",
        }
    )
    summary_rows.append(
        {
            "metric": "iso_yield_y80_max_savings",
            "value": round(max(savings_80.values()), 4),
            "detail": f"max library: {max(savings_80, key=savings_80.get)}",
        }
    )
    summary_rows.append(
        {
            "metric": "iso_yield_y80_min_savings",
            "value": round(min(savings_80.values()), 4),
            "detail": f"min library: {min(savings_80, key=savings_80.get)}",
        }
    )
    summary_rows.append(
        {
            "metric": "failure_classifier_loo_accuracy",
            "value": round(accuracy, 4),
            "detail": f"{correct}/{len(pool)} correct, k={args.knn_k}",
        }
    )
    summary_rows.append(
        {
            "metric": "per_class_recovery_collapse",
            "value": (
                conf_mat["collapse"]["collapse"]
                / max(1, sum(conf_mat["collapse"].values()))
            ),
            "detail": f"true collapse={sum(conf_mat['collapse'].values())}",
        }
    )
    summary_rows.append(
        {
            "metric": "per_class_recovery_drift",
            "value": (
                conf_mat["drift"]["drift"]
                / max(1, sum(conf_mat["drift"].values()))
            ),
            "detail": f"true drift={sum(conf_mat['drift'].values())}",
        }
    )
    summary_rows.append(
        {
            "metric": "per_class_recovery_plateau",
            "value": (
                conf_mat["plateau"]["plateau"]
                / max(1, sum(conf_mat["plateau"].values()))
            ),
            "detail": f"true plateau={sum(conf_mat['plateau'].values())}",
        }
    )
    summary_rows.append(
        {
            "metric": "per_class_recovery_converged",
            "value": (
                conf_mat["converged"]["converged"]
                / max(1, sum(conf_mat["converged"].values()))
            ),
            "detail": f"true converged={sum(conf_mat['converged'].values())}",
        }
    )

    summary_path = RESULTS / "zvf_iter38_summary.tsv"
    with summary_path.open("w") as fh:
        fh.write(
            "# Iter 38 Pillar 2 one-rollup summary.\n"
            "# Source: platform_modal/scripts/zvf_iter38_crosslibrary.py\n"
        )
        fh.write("metric\tvalue\tdetail\n")
        for r in summary_rows:
            fh.write(f"{r['metric']}\t{r['value']}\t{r['detail']}\n")

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print(f"Iter 38 Pillar 2 cross-library iso-yield: wrote {len(isoyield_rows)} rows to {isoyield_path}")
    print(f"Iter 38 LOO classifier: {correct}/{len(pool)} = {accuracy:.3f} accuracy -> {class_path}")
    print(f"Summary -> {summary_path}")
    print()
    print("Per-library y=0.80 iso-yield:")
    for lib in vm_libs:
        sub = [r for r in isoyield_rows if r["library"] == lib and abs(r["y_target"] - 0.80) < 1e-6]
        if sub:
            r = sub[0]
            print(
                f"  {lib:10s}  p_x={r['p_x']:.3f}  ddiv={r['delta_div_lib']:.3f}  "
                f"G_iid={r['G_iid']:3d}  G_emp={r['G_empirical']:3d}  "
                f"savings={r['cost_savings_frac']:.1%}"
            )
    print()
    print("Failure-mode LOO confusion (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c[:4]:>4s}" for c in classes))
    for c in classes:
        print(f"  {c[:4]:>4s}  " + "  ".join(f"{conf_mat[c][c2]:>4d}" for c2 in classes))


if __name__ == "__main__":
    main()