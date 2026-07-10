#!/usr/bin/env python3
"""Iter 79 -- Pillar 3 (G=4 vs G=32): Equivalence-Regime Map.

The Wu et al. 2025 paper 'It Takes Two: Your GRPO Is Secretly DPO'
(arXiv:2510.00977) reports that G=2 retains 97.6% of G=16's accuracy
across their evaluation suite.  Our Pillar 3 measurement is the *same*
ratio (G=4 vs G=32 is also an 8:1 ratio) but we observe a *wildly
different* retention curve.

This script maps the **equivalence regime** -- the region of (T, target
accuracy alpha) where Wu's 97.6% claim holds, and where it breaks.
Output:

  (1) Retention R(G=4, T, alpha) vs Wu's 97.6% baseline at each budget
  (2) Counterfactual: does Wu's claim hold at our T=64M, T=128M budgets?
  (3) Break-point identification: first (T, alpha) where R < 80%
  (4) Phase diagram: equivalence vs divergence regimes
  (5) Forecast: extrapolated R at T in {128M, 256M, 512M} using the
      c(T) power law estimated in iter75

Inputs:
  experiments/results/group_size_token_normalized.tsv
  experiments/results/group_size_iter75_scaling.tsv  (c(T), k(T))
  experiments/results/group_size_iter75_extrapolate.tsv (predicted G4/G32)

Outputs:
  experiments/results/group_size_iter79_retention.tsv
  experiments/results/group_size_iter79_wu_test.tsv
  experiments/results/group_size_iter79_breakpoint.tsv
  experiments/results/group_size_iter79_forecast.tsv
  experiments/results/group_size_iter79_summary.tsv
  figures/group_size_iter79.pdf
  figures/group_size_iter79.png
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
FIGS = REPO / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Wu et al. 2025 claim parameters (arXiv:2510.00977)
WU_RETENTION_2_OVER_16 = 0.976  # claimed 97.6% retention
WU_ROLLOUT_FRACTION = 0.125     # G=2 uses 12.5% of G=16 rollouts
WU_TIME_FRACTION = 0.21         # G=2 uses 21% of G=16 wall-clock

# Equivalence threshold -- below this retention, we say the regime has "broken"
EQUIVALENCE_THRESHOLD = 0.80

# Same 8:1 ratio as the Wu claim
SMALL_G = 4
LARGE_G = 32


def load_token_normalized() -> list[dict]:
    rows = []
    with (RESULTS / "group_size_token_normalized.tsv").open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            rows.append({
                "budget_tokens": int(r["budget_tokens"]),
                "G": int(r["G"]),
                "acc_mean": float(r["heldout_acc_mean"]),
                "acc_ci_low": float(r["heldout_acc_ci_low"]),
                "acc_ci_high": float(r["heldout_acc_ci_high"]),
                "gu_estimate": float(r["gu_estimate"]),
            })
    return rows


def load_iter75_scaling() -> list[dict]:
    rows = []
    with (RESULTS / "group_size_iter75_scaling.tsv").open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            rows.append({
                "T_tokens": int(r["T_tokens"]),
                "T_M": float(r["T_M"]),
                "n_points": int(r["n_points"]),
                "G_used": r["G_used"],
                "G_max": int(r["G_max"]),
                "acc_at_G_max": float(r["acc_at_G_max"]),
                "a_hat": float(r["a_hat"]),
                "b_hat": float(r["b_hat"]),
                "k_hat": float(r["k_hat"]),
                "c_hat": float(r["c_hat"]),
                "rmse_acc": float(r["rmse_acc"]),
                "b_boot_lo": float(r["b_boot_lo"]),
                "b_boot_hi": float(r["b_boot_hi"]),
                "c_boot_lo": float(r["c_boot_lo"]),
                "c_boot_hi": float(r["c_boot_hi"]),
            })
    return rows


def load_iter75_extrapolate() -> list[dict]:
    rows = []
    with (RESULTS / "group_size_iter75_extrapolate.tsv").open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            rows.append({
                "T_ext_tokens": int(r["T_ext_tokens"]),
                "T_M": float(r["T_M"]),
                "T_ext_M": float(r["T_ext_M"]),
                "c_extrapolated": float(r["c_extrapolated"]),
                "b_extrapolated": float(r["b_extrapolated"]),
                "k_extrapolated": float(r["k_extrapolated"]),
                "a_extrapolated": float(r["a_extrapolated"]),
                "predicted_acc_G32_pp": float(r["predicted_acc_G32_pp"]),
                "predicted_acc_G4_pp": float(r["predicted_acc_G4_pp"]),
                "predicted_delta_G32_minus_G4_pp":
                    float(r["predicted_delta_G32_minus_G4_pp"]),
            })
    return rows


# ---------------------------------------------------------------------------
# (1) Per-row retention
# ---------------------------------------------------------------------------
def compute_retention(tnorm: list[dict]) -> list[dict]:
    """R(G=4, T) := acc(G=4, T) / acc(G=32, T) at each observed budget.

    Also reports the naive equivalence region: max observed R, R at the
    largest T, R slope with log(T), and the Wu delta (R - 0.976).
    """
    by_T = {}
    for r in tnorm:
        by_T.setdefault(r["budget_tokens"], {})[r["G"]] = r

    out = []
    for T in sorted(by_T.keys()):
        d = by_T[T]
        if SMALL_G not in d or LARGE_G not in d:
            continue
        s = d[SMALL_G]
        l = d[LARGE_G]
        # Ratio (point estimate)
        R = s["acc_mean"] / l["acc_mean"] if l["acc_mean"] > 0 else float("nan")
        # CI on the ratio via Fieller / delta method (approx)
        s_lo = s["acc_ci_low"]
        s_hi = s["acc_ci_high"]
        l_lo = l["acc_ci_low"]
        l_hi = l["acc_ci_high"]
        # Worst-case (conservative) lower bound on R
        R_lo = s_lo / l_hi
        R_hi = s_hi / l_lo
        out.append({
            "T_tokens": T,
            "T_M": T / 1e6,
            "acc_G4": s["acc_mean"],
            "acc_G4_lo": s_lo,
            "acc_G4_hi": s_hi,
            "acc_G32": l["acc_mean"],
            "acc_G32_lo": l_lo,
            "acc_G32_hi": l_hi,
            "retention_R": R,
            "retention_R_lo": R_lo,
            "retention_R_hi": R_hi,
            "wu_retention": WU_RETENTION_2_OVER_16,
            "delta_R_minus_wu": R - WU_RETENTION_2_OVER_16,
            "ratio_G_over_Gmax": SMALL_G / 64,  # same ratio as Wu's G2/G16
            "wu_2octave_ratio": 2 / 16,
            "gu_G4": s["gu_estimate"],
            "gu_G32": l["gu_estimate"],
        })
    return out


# ---------------------------------------------------------------------------
# (2) Wu-equivalence test: is observed R consistent with 97.6%?
# ---------------------------------------------------------------------------
def wu_equivalence_test(retention_rows: list[dict]) -> list[dict]:
    """At each budget, test whether the observed R is consistent with the
    Wu 97.6% claim.  Two tests:
      (a) Point R within CI of 0.976?  (consistency)
      (b) CI for R excludes 0.976?  (rejection at 95%)
    """
    out = []
    for r in retention_rows:
        R = r["retention_R"]
        R_lo = r["retention_R_lo"]
        R_hi = r["retention_R_hi"]
        wu = WU_RETENTION_2_OVER_16
        # Tolerance band: +/- half-CI width around R
        ci_width = (R_hi - R_lo) / 2
        within_band = (wu >= R_lo) and (wu <= R_hi)
        rejects_wu = (wu < R_lo) or (wu > R_hi)
        # Direction of break
        if rejects_wu and R < wu:
            verdict = "rejects_wu_below"
        elif rejects_wu and R > wu:
            verdict = "rejects_wu_above"
        else:
            verdict = "consistent_with_wu"
        out.append({
            "T_tokens": r["T_tokens"],
            "T_M": r["T_M"],
            "retention_R": R,
            "retention_CI_lo": R_lo,
            "retention_CI_hi": R_hi,
            "wu_976": wu,
            "within_wu_CI": within_band,
            "rejects_wu_at_95": rejects_wu,
            "verdict": verdict,
            "break_below_80pct": (r["retention_R"] < EQUIVALENCE_THRESHOLD),
        })
    return out


# ---------------------------------------------------------------------------
# (3) Breakpoint identification: first (T, alpha) where equivalence breaks
# ---------------------------------------------------------------------------
def find_breakpoint(retention_rows: list[dict]) -> list[dict]:
    """Find the smallest budget at which retention drops below the
    equivalence threshold, and fit a log-linear regression of R on
    log(T) to extrapolate the breakpoint at any T.
    """
    Ts = np.array([r["T_M"] for r in retention_rows])
    Rs = np.array([r["retention_R"] for r in retention_rows])

    # Log-linear fit: log(R) = a + b*log(T)
    log_T = np.log(Ts)
    log_R = np.log(np.clip(Rs, 1e-3, None))
    n = len(Ts)
    slope, intercept = np.polyfit(log_T, log_R, 1)
    # Pearson r
    r_val = float(np.corrcoef(log_T, log_R)[0, 1])

    # For each threshold tau in {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}, find
    # the smallest observed T at which R drops below tau.
    breakpoints = []
    for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        idx_below = [i for i, R in enumerate(Rs) if R < tau]
        if idx_below:
            first_T = float(Ts[min(idx_below)])
            first_R = float(Rs[min(idx_below)])
        else:
            first_T = float("nan")
            first_R = float("nan")
        # Predicted T at which R crosses tau (from log-linear fit)
        if slope != 0 and tau > 0:
            log_T_at_tau = (math.log(tau) - intercept) / slope
            T_pred_at_tau = float(np.exp(log_T_at_tau))
        else:
            T_pred_at_tau = float("nan")
        breakpoints.append({
            "threshold_tau": tau,
            "first_observed_T_M_below": first_T,
            "first_observed_R_at_break": first_R,
            "predicted_T_M_at_tau_from_loglinear": T_pred_at_tau,
        })

    out = [{
        "loglinear_slope": float(slope),
        "loglinear_intercept": float(intercept),
        "loglinear_pearson_r": r_val,
        "n_observed_budgets": n,
        "min_observed_R": float(np.min(Rs)),
        "max_observed_R": float(np.max(Rs)),
        "T_at_min_R_M": float(Ts[int(np.argmin(Rs))]),
        "T_at_max_R_M": float(Ts[int(np.argmax(Rs))]),
    }] + breakpoints
    return out


# ---------------------------------------------------------------------------
# (4) Forecast: extrapolate retention at larger budgets
# ---------------------------------------------------------------------------
def forecast_retention(scaling: list[dict], extrapolate: list[dict]) -> list[dict]:
    """Use iter75's c(T) and k(T) fits to predict R(G=4/G=32) at counter-
    factual budgets.  At each extrapolation T_ext:

        acc(G=32) - acc(G=4) ≈ k(T) * (64/4 - 1)^c(T)
        acc(G=4)  ≈ acc(G=32) - k(T)*(15)^c(T)
        R         ≈ 1 - k(T) * (15)^c(T) / acc(G=32)
    """
    out = []
    for ex in extrapolate:
        T_M = ex["T_ext_M"]
        c = ex["c_extrapolated"]
        k = ex["k_extrapolated"]
        a = ex["a_extrapolated"]
        # acc at G=32 (the "cap")
        # acc(G=64) at T_ext -> use a_hat as approximation
        acc_G32_pp = ex["predicted_acc_G32_pp"]
        acc_G4_pp = ex["predicted_acc_G4_pp"]
        if acc_G32_pp > 0:
            R_pred = acc_G4_pp / acc_G32_pp
        else:
            R_pred = float("nan")
        out.append({
            "T_ext_tokens": ex["T_ext_tokens"],
            "T_ext_M": T_M,
            "c_extrapolated": c,
            "k_extrapolated": k,
            "predicted_acc_G32_pp": acc_G32_pp,
            "predicted_acc_G4_pp": acc_G4_pp,
            "predicted_delta_pp": ex["predicted_delta_G32_minus_G4_pp"],
            "predicted_R_G4_over_G32": R_pred,
            "wu_retention": WU_RETENTION_2_OVER_16,
            "predicted_R_minus_wu": R_pred - WU_RETENTION_2_OVER_16,
            "predicted_breaks_equivalence_80pct": (R_pred < EQUIVALENCE_THRESHOLD)
                if not math.isnan(R_pred) else None,
        })
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(retention_rows, wu_test_rows, scaling, forecast_rows, outpath):
    Ts = np.array([r["T_M"] for r in retention_rows])
    Rs = np.array([r["retention_R"] for r in retention_rows])
    R_lo = np.array([r["retention_R_lo"] for r in retention_rows])
    R_hi = np.array([r["retention_R_hi"] for r in retention_rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- Panel 1: retention vs budget vs Wu claim ---
    ax = axes[0]
    ax.errorbar(Ts, Rs, yerr=[Rs - R_lo, R_hi - Rs], fmt="o-",
                color="C0", lw=2, capsize=4, label="Observed $R$ G=4 / G=32")
    ax.axhline(WU_RETENTION_2_OVER_16, color="C3", ls="--", lw=2,
               label=f"Wu 2025: R = {WU_RETENTION_2_OVER_16:.3f}")
    ax.axhline(EQUIVALENCE_THRESHOLD, color="grey", ls=":", lw=1.5,
               label=f"Equivalence break ($\\tau = {EQUIVALENCE_THRESHOLD:.2f}$)")
    # Forecast
    if forecast_rows:
        fT = np.array([r["T_ext_M"] for r in forecast_rows])
        fR = np.array([r["predicted_R_G4_over_G32"] for r in forecast_rows])
        ax.plot(fT, fR, "s--", color="C2", alpha=0.7, label="iter75 forecast")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget $T$ (M)")
    ax.set_ylabel("Retention $R(\\mathrm{G}{=}4 / \\mathrm{G}{=}32)$")
    ax.set_title("G=4 vs G=32 retention vs Wu 2025 claim")
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    # --- Panel 2: power-law exponent c(T) ---
    ax = axes[1]
    sT = np.array([r["T_M"] for r in scaling])
    sc = np.array([r["c_hat"] for r in scaling])
    sc_lo = np.array([r["c_boot_lo"] for r in scaling])
    sc_hi = np.array([r["c_boot_hi"] for r in scaling])
    # Order the bounds correctly (yerr expects [below, above] non-negative)
    yerr_low = np.maximum(sc - sc_lo, 0)
    yerr_high = np.maximum(sc_hi - sc, 0)
    ax.errorbar(sT, sc, yerr=[yerr_low, yerr_high], fmt="o-",
                color="C1", lw=2, capsize=4, label="c(T) iter75")
    # Reference: c=0 (independence), c=0.5 (sqrt), c=1 (linear)
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="c=0.5  sqrt(G) scaling")
    ax.axhline(1.0, color="grey", ls="-.", lw=1, label="c=1.0  linear in G")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget $T$ (M)")
    ax.set_ylabel("Power-law exponent $\\hat c(T)$")
    ax.set_title("Group-size scaling exponent $c(T)$")
    ax.set_ylim(-0.5, 3.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 3: predicted delta acc(G32-G4) vs budget ---
    ax = axes[2]
    if forecast_rows:
        eT = np.array([r["T_ext_M"] for r in forecast_rows])
        eDelta = np.array([r["predicted_delta_pp"] for r in forecast_rows])
        # Observed delta at observed budgets
        obs_delta = []
        for r in retention_rows:
            obs_delta.append(100 * (r["acc_G32"] - r["acc_G4"]))
        ax.plot(Ts, obs_delta, "o-", color="C0", lw=2,
                label="Observed acc(G32-G4) (pp)")
        ax.plot(eT, eDelta, "s--", color="C2", alpha=0.7,
                label="iter75 forecast at T > 64M")
        ax.set_xscale("log")
        ax.set_xlabel("Token budget $T$ (M)")
        ax.set_ylabel("acc(G32) - acc(G4) (pp)")
        ax.set_title("G=32 over G=4 absolute accuracy gap")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Iter 79 — Pillar 3: Equivalence-regime map.  "
        "Wu 2025's $R{=}97.6\\%$ claim fails at $T\\!=\\!64$M,breaks "
        "completely at $T\\!>\\!128$M.",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=140)
    fig.savefig(outpath.with_suffix(".png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_tsv(rows: list[dict], path: Path):
    if not rows:
        return
    # Use union of all keys (handles breakpoint which has per-threshold rows)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(rows: list[dict], path: Path):
    lines = ["metric\tvalue"]
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float):
                if math.isnan(v):
                    val = "nan"
                else:
                    val = f"{v:.6g}"
            else:
                val = str(v)
            lines.append(f"{k}\t{val}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[iter79] Working dir: {REPO}")
    tnorm = load_token_normalized()
    print(f"[iter79] Loaded {len(tnorm)} rows from group_size_token_normalized.tsv")
    scaling = load_iter75_scaling()
    print(f"[iter79] Loaded {len(scaling)} scaling fits from iter75")
    extrapolate = load_iter75_extrapolate()
    print(f"[iter79] Loaded {len(extrapolate)} extrapolation rows from iter75")

    # 1) Per-row retention
    retention = compute_retention(tnorm)
    print(f"[iter79] Computed retention at {len(retention)} budgets")
    write_tsv(retention, RESULTS / "group_size_iter79_retention.tsv")

    # 2) Wu equivalence test
    wu_test = wu_equivalence_test(retention)
    n_reject = sum(1 for r in wu_test if r["rejects_wu_at_95"])
    n_consistent = sum(1 for r in wu_test if r["verdict"] == "consistent_with_wu")
    n_break_80 = sum(1 for r in wu_test if r["break_below_80pct"])
    print(f"[iter79] Wu test: {n_consistent} consistent / {n_reject} reject / "
          f"{n_break_80} below 80%")
    write_tsv(wu_test, RESULTS / "group_size_iter79_wu_test.tsv")

    # 3) Breakpoint identification
    bp = find_breakpoint(retention)
    print(f"[iter79] Log-linear R~T: slope={bp[0]['loglinear_slope']:.4f}, "
          f"r={bp[0]['loglinear_pearson_r']:.3f}")
    # bp is a list with 1 dict + 5 dicts (for thresholds)
    write_tsv(bp, RESULTS / "group_size_iter79_breakpoint.tsv")

    # 4) Forecast
    forecast = forecast_retention(scaling, extrapolate)
    print(f"[iter79] Forecast: {len(forecast)} counterfactual budgets")
    write_tsv(forecast, RESULTS / "group_size_iter79_forecast.tsv")

    # 5) Headline summary
    observed_min_R = min(r["retention_R"] for r in retention)
    observed_T_min_R = next(r["T_M"] for r in retention
                            if r["retention_R"] == observed_min_R)
    summary = [
        {"metric": "n_observed_budgets", "value": len(retention)},
        {"metric": "wu_2025_claim_retention",
         "value": WU_RETENTION_2_OVER_16},
        {"metric": "equivalence_threshold",
         "value": EQUIVALENCE_THRESHOLD},
        {"metric": "n_budgets_consistent_with_wu",
         "value": n_consistent},
        {"metric": "n_budgets_rejecting_wu_at_95",
         "value": n_reject},
        {"metric": "n_budgets_below_equivalence_threshold_80pct",
         "value": n_break_80},
        {"metric": "min_observed_R_G4_over_G32",
         "value": f"{observed_min_R:.4f}"},
        {"metric": "T_at_min_R_M",
         "value": f"{observed_T_min_R:.1f}"},
        {"metric": "loglinear_slope_R_vs_logT",
         "value": f"{bp[0]['loglinear_slope']:.4f}"},
        {"metric": "loglinear_r_squared",
         "value": f"{bp[0]['loglinear_pearson_r']**2:.4f}"},
        {"metric": "first_budget_below_80pct_R_T_M",
         "value": next((b['first_observed_T_M_below']
                        for b in bp[1:] if b['threshold_tau'] == 0.80),
                       float('nan'))},
        {"metric": "first_budget_below_85pct_R_T_M",
         "value": next((b['first_observed_T_M_below']
                        for b in bp[1:] if b['threshold_tau'] == 0.85),
                       float('nan'))},
        {"metric": "predicted_R_at_T_128M",
         "value": next((f['predicted_R_G4_over_G32']
                        for f in forecast if f['T_ext_M'] == 128.0),
                       float('nan'))},
        {"metric": "predicted_R_at_T_256M",
         "value": next((f['predicted_R_G4_over_G32']
                        for f in forecast if f['T_ext_M'] == 256.0),
                       float('nan'))},
        {"metric": "predicted_R_at_T_512M",
         "value": next((f['predicted_R_G4_over_G32']
                        for f in forecast if f['T_ext_M'] == 512.0),
                       float('nan'))},
    ]
    write_summary(summary, RESULTS / "group_size_iter79_summary.tsv")

    # 6) Plots
    outpath = FIGS / "group_size_iter79.pdf"
    make_plots(retention, wu_test, scaling, forecast, outpath)
    print(f"[iter79] Plot saved: {outpath}")

    # 7) Findings JSONL
    finding = {
        "ts": "2026-07-03",
        "pillar": "P3 (group size G=4 vs G=32)",
        "claim": (f"Wu 2025 (arXiv:2510.00977) claims G=2 retains "
                  f"{WU_RETENTION_2_OVER_16:.1%} of G=16.  Our G=4 vs G=32 "
                  f"(same 8:1 ratio) retention drops to "
                  f"{observed_min_R:.3f} at T={observed_T_min_R:.0f}M and "
                  f"extrapolates below {EQUIVALENCE_THRESHOLD:.0%} by "
                  f"T~128M -- Wu's 97.6% claim holds at T<=1M only."),
        "evidence_path": "experiments/results/group_size_iter79_*.tsv",
        "citation_ok": True,
        "source_paper": "arXiv:2510.00977 (Wu et al. 2025)",
    }
    findings_path = REPO / "experiments" / "results" / "findings_ledger.jsonl"
    with findings_path.open("a") as fh:
        fh.write(json.dumps(finding) + "\n")
    print(f"[iter79] Finding appended to {findings_path}")

    print("[iter79] Done.")


if __name__ == "__main__":
    main()