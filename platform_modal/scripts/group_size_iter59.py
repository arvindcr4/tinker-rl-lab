#!/usr/bin/env python3
"""Iter 59 — Pillar 3: Equivalence-Region Estimation and Multiplicative Decomposition.

Three fresh, first-class diagnostics of the G=4 vs G=32 question:

  (A) EQUIVALENCE REGION in (T, R) plane.  Empirically, retention
      R(T) = acc(G=4) / acc(G=32) on the iso-token grid is non-monotone
      in T.  We define three equivalence regions:
        - PRAGMATIC EQUIVALENCE:  R >= 0.85 (Wu-2025 spirit)
        - OPERATIONAL EQUIVALENCE: R >= 0.75 (paper-grade)
        - HARD DIVERGENCE:       R <= 0.70 (paper's empirical crossovers)
      and report the budget at which each threshold is crossed.

  (B) MIN-TOKEN-TO-TARGET FRONTIER.  For each target accuracy
      a* in {0.55, 0.65, 0.75, 0.85}, find the G with the smallest T
      that achieves a >= a* (log-linear interpolation in T).  This is
      the practitioner-relevant inversion of the argmax-G question.

  (C) MULTIPLICATIVE DECOMPOSITION.  R(T) = (Y4/Y32) * (steps4/steps32)
      * (1 + noise(T)), where Y is contrastive yield and steps is the
      per-token optimizer step ratio.  For iso-token comparison,
      steps_4/steps_32 = G32/G4 = 8 always, so the decomposition is
      R(T) = 8 * (Y4/Y32) * (1 + noise(T)).
      Since Y(G) is approximately T-invariant (ZVF depends on policy
      not budget), noise(T) captures the *budget-specific* residual
      and is a direct probe of the iter-55 budget exponent gamma.

Inputs (read-only):
  experiments/results/group_size_token_normalized.tsv   (4 budgets x 5 G)
  experiments/results/groupsize_zvf_sweep.tsv          (4 G rows from n=3 seeds)

Outputs (TSVs):
  group_size_iter59_equivalence.tsv       region thresholds and budgets
  group_size_iter59_min_tokens.tsv        min-token-to-target frontier
  group_size_iter59_decomp.tsv            multiplicative decomposition
  group_size_iter59_summary.tsv           headline rollup
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
SEED = 20240702
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_token_norm():
    out = []
    with open(RES / "group_size_token_normalized.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "T": int(row["budget_tokens"]),
                    "G": int(row["G"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "ci_lo": float(row["heldout_acc_ci_low"]),
                    "ci_hi": float(row["heldout_acc_ci_high"]),
                    "gu": float(row["gu_estimate"]),
                }
            )
    return out


def load_zvf_sweep():
    out = []
    with open(RES / "groupsize_zvf_sweep.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            out.append(
                {
                    "G": int(row["G"]),
                    "n_seeds": int(row["n_seeds"]),
                    "acc": float(row["heldout_acc_mean"]),
                    "acc_se": float(row["heldout_acc_se"]),
                    "last10": float(row["last10_mean"]),
                    "mean_zvf": float(row["mean_zvf"]),
                    "zvf_th": float(row["zvf_theory_at_mean_p"]),
                    "mean_reward_train": float(row["mean_reward_train"]),
                }
            )
    return out


def write_tsv(path, header, rows):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in header) + "\n")


def acc_at(rows, T, G):
    """Linear interp log T -> acc at (T, G)."""
    cells = sorted([r for r in rows if r["G"] == G], key=lambda r: r["T"])
    if not cells:
        return float("nan")
    if T <= cells[0]["T"]:
        return cells[0]["acc"]
    if T >= cells[-1]["T"]:
        return cells[-1]["acc"]
    # log-log interp in T (acc is approximately log-linear in T at fixed G)
    logTs = [math.log(c["T"]) for c in cells]
    accs = [c["acc"] for c in cells]
    logT = math.log(T)
    # linear interp
    for i in range(len(cells) - 1):
        if logTs[i] <= logT <= logTs[i + 1]:
            w = (logT - logTs[i]) / (logTs[i + 1] - logTs[i])
            return accs[i] * (1 - w) + accs[i + 1] * w
    return float("nan")


def find_T_for_acc(rows, G, target_acc):
    """Find the smallest T at which acc(G, T) >= target_acc.

    Linear interpolation in log T.  Returns T (tokens) or inf if unreachable.
    """
    cells = sorted([r for r in rows if r["G"] == G], key=lambda r: r["T"])
    if not cells:
        return float("inf")
    # Already met at smallest budget?
    if cells[0]["acc"] >= target_acc:
        return cells[0]["T"]
    # Already exceeded at largest budget?
    if cells[-1]["acc"] < target_acc:
        return float("inf")
    for i in range(len(cells) - 1):
        a0, a1 = cells[i]["acc"], cells[i + 1]["acc"]
        if a0 < target_acc <= a1:
            logT0 = math.log(cells[i]["T"])
            logT1 = math.log(cells[i + 1]["T"])
            w = (target_acc - a0) / (a1 - a0)
            logT = logT0 * (1 - w) + logT1 * w
            return math.exp(logT)
    return float("inf")


# ---------------------------------------------------------------------------
# A. Equivalence region in (T, R) plane
# ---------------------------------------------------------------------------


def compute_retention_curve(rows):
    """Compute R(T) = acc(G=4)/acc(G=32) at every budget."""
    out = []
    budgets = sorted({r["T"] for r in rows})
    for T in budgets:
        a4 = next(r["acc"] for r in rows if r["T"] == T and r["G"] == 4)
        a32 = next(r["acc"] for r in rows if r["T"] == T and r["G"] == 32)
        R = a4 / a32 if a32 > 0 else float("nan")
        out.append({"T": T, "a4": a4, "a32": a32, "R": R})
    return out


def equivalence_regions(ret_curve):
    """Find the smallest T at which R crosses each threshold.

    R(T) is non-monotone; we report (i) the first budget where R drops
    below each threshold, (ii) the last budget where R is still above it,
    and (iii) the min/max T over the iso-token grid for context.
    """
    THRESHOLDS = {
        "PRAGMATIC_0.85": 0.85,
        "OPERATIONAL_0.75": 0.75,
        "HARD_DIVERGENCE_0.70": 0.70,
    }
    out = []
    sorted_rc = sorted(ret_curve, key=lambda r: r["T"])
    T_min = sorted_rc[0]["T"]
    T_max = sorted_rc[-1]["T"]
    for name, thr in THRESHOLDS.items():
        # Find smallest T at which R drops below threshold
        T_cross_down = None
        for r in sorted_rc:
            if r["R"] < thr:
                T_cross_down = r["T"]
                break
        # Find largest T at which R is still above threshold
        T_last_above = None
        for r in sorted_rc:
            if r["R"] >= thr:
                T_last_above = r["T"]
        # Equivalence region = [T_min, T_last_above] if any T has R >= thr
        in_region = "yes" if T_last_above is not None else "no"
        out.append(
            {
                "threshold_name": name,
                "threshold_R": thr,
                "T_min_grid_M": T_min // 1_000_000,
                "T_max_grid_M": T_max // 1_000_000,
                "T_first_below_threshold_M": (T_cross_down // 1_000_000) if T_cross_down is not None else "none",
                "T_last_above_threshold_M": (T_last_above // 1_000_000) if T_last_above is not None else "none",
                "equivalence_region_exists": in_region,
            }
        )
    # Append the actual retention curve for context
    for r in sorted_rc:
        out.append(
            {
                "threshold_name": f"R_observed_T{r['T']//1_000_000}M",
                "threshold_R": r["R"],
                "T_min_grid_M": r["T"] // 1_000_000,
                "T_max_grid_M": r["T"] // 1_000_000,
                "T_first_below_threshold_M": "",
                "T_last_above_threshold_M": "",
                "equivalence_region_exists": "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# B. Min-token-to-target frontier
# ---------------------------------------------------------------------------


def min_tokens_frontier(rows):
    """For each (target_acc, G), find the smallest T that achieves the target.

    Then, for each target, the G with the smallest T is the
    'token-efficient G at that target'.  This is the operational inversion
    of the iso-token argmax-G question.
    """
    targets = [0.55, 0.65, 0.75, 0.85]
    Gs = [4, 8, 16, 32, 64]
    out = []
    for a_star in targets:
        cell = []
        for G in Gs:
            T_star = find_T_for_acc(rows, G, a_star)
            cell.append((G, T_star))
        # Min T (skip inf)
        feasible = [(g, t) for g, t in cell if math.isfinite(t)]
        if feasible:
            argmin_G = min(feasible, key=lambda x: x[1])[0]
        else:
            argmin_G = "none_feasible"
        for G, T_star in cell:
            T_M = T_star / 1_000_000 if math.isfinite(T_star) else "inf"
            is_argmin = "yes" if G == argmin_G else "no"
            out.append(
                {
                    "target_acc": a_star,
                    "G": G,
                    "T_min_tokens_M": T_M,
                    "reachable": "yes" if math.isfinite(T_star) else "no",
                    "is_argmin_at_target": is_argmin,
                    "argmin_G_at_target": argmin_G,
                }
            )
    return out


# ---------------------------------------------------------------------------
# C. Multiplicative decomposition R = (Y4/Y32) * (steps4/steps32) * (1+noise)
# ---------------------------------------------------------------------------


def contrast_yields(zvf):
    """Return {G: Y_obs = 1 - zvf_obs} for G in {2,4,8,16} from the zvf sweep."""
    out = {}
    for r in zvf:
        out[r["G"]] = 1.0 - r["mean_zvf"]
    return out


def extend_to_G32_64(Y_map):
    """Extrapolate Y(G) for G not in zvf sweep via OLS of Y vs log G.

    The raw i.i.d. formula ZVF = p^G + (1-p)^G - delta_div underflows
    at G=32 with p=0.86 (i.i.d. term ~0.008, delta_div ~0.3, gives
    zvf_pred < 0, which clips to Y=1).  Instead, fit a linear model
    Y = a*log G + b on the zvf-sweep data {G=2,4,8,16} and extrapolate.
    Y(G) saturates so we also cap the extrapolation at 0.95 to keep the
    Y4/Y32 ratio finite.
    """
    Gs = np.array(sorted(Y_map.keys()), dtype=float)
    Ys = np.array([Y_map[int(g)] for g in Gs], dtype=float)
    logG = np.log(Gs)
    n = len(logG)
    sx, sy = logG.sum(), Ys.sum()
    sxx, sxy = (logG * logG).sum(), (logG * Ys).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx + 1e-12)
    intercept = (sy - slope * sx) / n
    out = dict(Y_map)
    for G in (32, 64):
        Y_extrap = intercept + slope * math.log(G)
        # Cap to keep Y4/Y32 finite and physically reasonable
        out[G] = min(0.95, max(0.0, Y_extrap))
    return out, slope, intercept


def multiplicative_decomposition(rows, Y_map, zvf_raw):
    """Decompose R(T) = (Y4/Y32) * (steps4/steps32) * (1+noise(T)).

    For iso-token comparison, steps4/steps32 = G32/G4 = 8 always
    (T / (G * L_per_prompt) is 8x larger for G=4 than G=32).  This
    means the EXPECTED retention under a pure-Y model is 8*(Y4/Y32).

    The factor 1 + noise(T) is then:
        noise(T) = R(T) / (8 * Y4/Y32) - 1
    and is a direct probe of the iter-55 budget exponent gamma.

    If Y4 < Y32 (which is the case empirically), then 8*(Y4/Y32) < 8
    but still > 1, so the "Y-ratio * steps-ratio" upper bound is a
    number between 1 and 8.  Empirical R(T) < 1 in all our data, so
    noise is large and negative — the structural model is an upper
    bound, not a predictor.
    """
    Y_full, slope, intercept = extend_to_G32_64(Y_map)
    Y4 = Y_full[4]
    Y32 = Y_full[32]
    steps_ratio = 32 / 4  # = 8 always on iso-token grid
    structural_R = steps_ratio * (Y4 / Y32)
    out = []
    for T in sorted({r["T"] for r in rows}):
        a4 = next(r["acc"] for r in rows if r["T"] == T and r["G"] == 4)
        a32 = next(r["acc"] for r in rows if r["T"] == T and r["G"] == 32)
        R_emp = a4 / a32 if a32 > 0 else float("nan")
        # Decomposition:
        # R_emp = structural_R * (1 + noise)
        # noise = R_emp / structural_R - 1
        noise = R_emp / structural_R - 1.0 if structural_R > 0 else float("nan")
        # Equivalent decomposition: log R = log(steps_ratio) + log(Y4/Y32) + log(1+noise)
        log_R = math.log(R_emp) if R_emp > 0 else float("nan")
        log_struct = math.log(structural_R) if structural_R > 0 else float("nan")
        log_noise = math.log(1.0 + noise) if (1.0 + noise) > 0 else float("nan")
        out.append(
            {
                "T_tokens": T,
                "G4_acc": a4,
                "G32_acc": a32,
                "empirical_R": R_emp,
                "structural_R_Y_times_steps": structural_R,
                "noise_residual": noise,
                "log_R_emp": log_R,
                "log_structural_R": log_struct,
                "log_noise": log_noise,
                "is_structural_overestimate": "yes" if R_emp < structural_R else "no",
            }
        )
    # Also report the static structural quantities
    out.append(
        {
            "T_tokens": "structural_constants",
            "G4_acc": "",
            "G32_acc": "",
            "empirical_R": "",
            "structural_R_Y_times_steps": structural_R,
            "noise_residual": f"Y4={Y4:.4f}",
            "log_R_emp": "",
            "log_structural_R": log_struct,
            "log_noise": f"Y32={Y32:.4f}",
            "is_structural_overestimate": "",
        }
    )
    out.append(
        {
            "T_tokens": "log_decomposition",
            "G4_acc": "",
            "G32_acc": "",
            "empirical_R": "",
            "structural_R_Y_times_steps": "",
            "noise_residual": "",
            "log_R_emp": "",
            "log_structural_R": f"log_steps={math.log(steps_ratio):.4f}",
            "log_noise": f"log_Y={math.log(Y4/Y32):.4f}",
            "is_structural_overestimate": "",
        }
    )
    return out


# ---------------------------------------------------------------------------
# D. Summary
# ---------------------------------------------------------------------------


def summarize(regions, frontier, decomp):
    out = []
    # Equivalence regions
    for r in regions:
        if r["threshold_name"].startswith("R_observed"):
            continue
        out.append(
            {
                "metric": f"equiv_{r['threshold_name']}_last_above_T_M",
                "value": r["T_last_above_threshold_M"],
            }
        )
        out.append(
            {
                "metric": f"equiv_{r['threshold_name']}_first_below_T_M",
                "value": r["T_first_below_threshold_M"],
            }
        )
    # Argmin-G at each target
    argmin_map = {}
    for r in frontier:
        if r["is_argmin_at_target"] == "yes":
            argmin_map[r["target_acc"]] = r["G"]
    for a_star, g in argmin_map.items():
        out.append({"metric": f"argmin_G_at_target_{a_star}", "value": str(g)})
    # Decomposition constants
    for r in decomp:
        if r["T_tokens"] == "structural_constants":
            out.append(
                {
                    "metric": "structural_R_Y_times_steps",
                    "value": f"{r['structural_R_Y_times_steps']:.4f}",
                }
            )
        if r["T_tokens"] == "log_decomposition":
            out.append(
                {"metric": "log_steps_ratio", "value": f"{r['log_structural_R']}"}
            )
    # Per-budget noise
    for r in decomp:
        if isinstance(r["T_tokens"], int):
            T_M = r["T_tokens"] // 1_000_000
            out.append(
                {
                    "metric": f"noise_residual_at_T{T_M}M",
                    "value": f"{r['noise_residual']:.4f}",
                }
            )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rows = load_token_norm()
    zvf = load_zvf_sweep()
    print(f"Loaded {len(rows)} iso-token cells, {len(zvf)} zvf-sweep rows.")

    # A. Equivalence regions
    ret_curve = compute_retention_curve(rows)
    regions = equivalence_regions(ret_curve)
    write_tsv(
        RES / "group_size_iter59_equivalence.tsv",
        [
            "threshold_name",
            "threshold_R",
            "T_min_grid_M",
            "T_max_grid_M",
            "T_first_below_threshold_M",
            "T_last_above_threshold_M",
            "equivalence_region_exists",
        ],
        regions,
    )
    print(f"Wrote group_size_iter59_equivalence.tsv ({len(regions)} rows)")

    # B. Min-token-to-target frontier
    frontier = min_tokens_frontier(rows)
    write_tsv(
        RES / "group_size_iter59_min_tokens.tsv",
        [
            "target_acc",
            "G",
            "T_min_tokens_M",
            "reachable",
            "is_argmin_at_target",
            "argmin_G_at_target",
        ],
        frontier,
    )
    print(f"Wrote group_size_iter59_min_tokens.tsv ({len(frontier)} rows)")

    # C. Multiplicative decomposition
    Y_map = contrast_yields(zvf)
    # Bind zvf_raw into a closure-equivalent
    global zvf_raw
    zvf_raw = zvf
    decomp = multiplicative_decomposition(rows, Y_map, zvf)
    write_tsv(
        RES / "group_size_iter59_decomp.tsv",
        [
            "T_tokens",
            "G4_acc",
            "G32_acc",
            "empirical_R",
            "structural_R_Y_times_steps",
            "noise_residual",
            "log_R_emp",
            "log_structural_R",
            "log_noise",
            "is_structural_overestimate",
        ],
        decomp,
    )
    print(f"Wrote group_size_iter59_decomp.tsv ({len(decomp)} rows)")

    # D. Summary
    summ = summarize(regions, frontier, decomp)
    write_tsv(RES / "group_size_iter59_summary.tsv", ["metric", "value"], summ)
    print(f"Wrote group_size_iter59_summary.tsv ({len(summ)} rows)")

    # Headline
    print("\n=== Iter 59 Headline ===")
    print("\nRetention curve:")
    for r in ret_curve:
        T_M = r["T"] // 1_000_000
        print(f"  T={T_M:2d}M  acc4={r['a4']:.3f}  acc32={r['a32']:.3f}  R={r['R']:.4f}")
    print("\nEquivalence regions:")
    for r in regions:
        if not r["threshold_name"].startswith("R_observed"):
            print(
                f"  {r['threshold_name']:25s}  R>={r['threshold_R']:.2f}: "
                f"last_above_T={r['T_last_above_threshold_M']}M  "
                f"first_below_T={r['T_first_below_threshold_M']}M  "
                f"exists={r['equivalence_region_exists']}"
            )
    print("\nMin-token-to-target frontier:")
    for r in frontier:
        if r["is_argmin_at_target"] == "yes":
            print(f"  target={r['target_acc']:.2f}  argmin_G={r['G']}  T={r['T_min_tokens_M']}M")
    print("\nMultiplicative decomposition:")
    for r in decomp:
        if isinstance(r["T_tokens"], int):
            T_M = r["T_tokens"] // 1_000_000
            print(
                f"  T={T_M:2d}M  R_emp={r['empirical_R']:.4f}  "
                f"R_struct(Y*steps)={r['structural_R_Y_times_steps']:.4f}  "
                f"noise={r['noise_residual']:+.4f}"
            )
        elif r["T_tokens"] == "structural_constants":
            print(
                f"  constants: Y4={r['noise_residual']}  Y32={r['log_noise']}  "
                f"R_struct={r['structural_R_Y_times_steps']:.4f}"
            )


if __name__ == "__main__":
    main()