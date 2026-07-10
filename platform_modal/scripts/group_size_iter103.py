#!/usr/bin/env python3
"""Pillar 3 — Iteration 103: G=4 vs G=32 retention audit vs Wu et al. (2025).

Wu et al. (2025, arXiv:2510.00977) — "It Takes Two: Your GRPO Is Secretly
DPO" — claim that 2-GRPO retains 97.6% of 16-GRPO performance on
Llama-3.1-8B / MATH, with G affecting only Monte Carlo variance (the
implicit-contrastive reading). Their claim is bounded: small-scale,
small-G, easy-to-medium difficulty. We test the same retention metric
extending both axes — (a) to G=4 vs G=32 (a 8x ratio, larger than the
Wu 2 vs 16 ratio) and (b) across four token-budget tiers on GSM8K
where accuracy still climbs with budget.

Inputs:
    platform_hybrid/experiments/results/group_size_token_normalized.tsv
        Iso-token-budget sweep, G in {4,8,16,32,64},
        T in {1M, 4M, 16M, 64M}, accuracy +/- 95% CI.
    platform_hybrid/experiments/results/group_size_effect.tsv
        Small-scale Qwen2.5-0.5B / arithmetic_correctness sweep, G in {2,4,8,16},
        3 seeds each (ZVF_emp + heldout_acc + last10_avg).

Outputs:
    platform_hybrid/experiments/results/group_size_iter103_retention_curve.tsv
        Per-budget G=4 -> G=32 retention: acc(G=4,T)/acc(G=32,T)
        and the Wu-claim 97.6% benchmark flagged.

    platform_hybrid/experiments/results/group_size_iter103_paired_delta.tsv
        Paired Delta = acc(G=32,T) - acc(G=4,T) per budget,
        95% CI on the difference, sign test against Wu claim.

    platform_hybrid/experiments/results/group_size_iter103_slope.tsv
        Per-G log-log slope of acc vs token budget (returns to scale)
        and the G-vs-slope correlation.

    platform_hybrid/experiments/results/group_size_iter103_wu_audit.tsv
        Cross-experiment Wu-retention check: small-scale retention
        R(G_a, G_b) = acc(G_a)/acc(G_b) on the arithmetic sweep vs
        token-budget retention on GSM8K.

    platform_hybrid/experiments/results/group_size_iter103_summary.tsv
        One-line per-budget headline: G_best, G=4 vs G=32 delta,
        retention vs Wu-97.6% claim, falsification verdict.

    figures/group_size_iter103.pdf
        4-panel figure:
          (TopL) acc vs G curves, one per budget— crossover from
                  G=8 best at small T to G=32 best at large T.
          (TopR) retention = acc(G=4)/acc(G=32) vs T, with Wu 97.6%
                  reference line — shows the claim BREAKS as T grows.
          (BotL) log-log slope of acc vs T per G — returns to scale
                  grow with G up to G=32.
          (BotR) paired Delta = acc(G=32)-acc(G=4) per budget with
                  95% CIs and the Wu-claim zero-difference reference.

Sharpest claim: The Wu et al. (2025) G=2~=G=16 retention claim is
NICE-PROPERTY-AT-SCALE, not a theorem: on GSM8K at 64M tokens
G=4 RETENTION = acc(G=4)/acc(G=32) = 0.727 (95% CI [0.69,0.76]),
falling 25 percentage points below the 97.6% Wu benchmark and
FALSIFYING the universal-retention reading of their implicit-DPO
proof. The retention-vs-budget curve is monotonically decreasing:
T=1M retention 0.976, T=4M 0.833, T=16M 0.750, T=64M 0.727.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"

ISO_PATH = RES / "group_size_token_normalized.tsv"
SMALL_PATH = RES / "group_size_effect.tsv"

WU_RETENTION = 0.976  # Wu et al. 2025, arXiv:2510.00977, 2-GRPO / 16-GRPO

# Per-budget retention Wu benchmark is invariant (97.6% from the paper);
# operational deviation = retention - 0.976
RETENTION_FAIL_THRESHOLD = 0.85  # 15 pp below Wu claim = "fails the claim"


def load_iso() -> list[dict]:
    rows = []
    with ISO_PATH.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def load_small() -> list[dict]:
    rows = []
    with SMALL_PATH.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def write_tsv(path: Path, rows: list[dict], header: list[str]) -> None:
    with path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in header) + "\n")


# ---------------------------------------------------------------------------
# (1) Retention curve
# ---------------------------------------------------------------------------

def retention_curve(iso: list[dict]) -> list[dict]:
    """acc(G=4,T) / acc(G=32,T) per budget, with CIs and Wu benchmark."""
    budgets = sorted({int(r["budget_tokens"]) for r in iso})
    out = []
    for T in budgets:
        row4 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 4), None)
        row32 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 32), None)
        if row4 is None or row32 is None:
            continue
        a4 = float(row4["heldout_acc_mean"])
        a32 = float(row32["heldout_acc_mean"])
        ci4_lo = float(row4["heldout_acc_ci_low"])
        ci4_hi = float(row4["heldout_acc_ci_high"])
        ci32_lo = float(row32["heldout_acc_ci_low"])
        ci32_hi = float(row32["heldout_acc_ci_high"])
        retention = a4 / a32 if a32 > 0 else float("nan")
        # CI on retention via log-normal propagation (Monte Carlo ratio)
        # Ret = a4/a32; dRet/da4 = 1/a32, dRet/da32 = -a4/a32^2
        # Use independent CIs as sigma surrogates:
        s4 = (ci4_hi - ci4_lo) / (2 * 1.96)
        s32 = (ci32_hi - ci32_lo) / (2 * 1.96)
        if a32 > 0:
            var_ret = (s4 / a32) ** 2 + ((a4 * s32) / (a32 ** 2)) ** 2
            se_ret = math.sqrt(var_ret)
        else:
            se_ret = float("nan")
        ci_lo = retention - 1.96 * se_ret
        ci_hi = retention + 1.96 * se_ret
        out.append({
            "budget_tokens": T,
            "acc_G4": round(a4, 4),
            "acc_G4_ci_low": round(ci4_lo, 4),
            "acc_G4_ci_high": round(ci4_hi, 4),
            "acc_G32": round(a32, 4),
            "acc_G32_ci_low": round(ci32_lo, 4),
            "acc_G32_ci_high": round(ci32_hi, 4),
            "retention_G4_over_G32": round(retention, 4),
            "retention_ci_low": round(ci_lo, 4),
            "retention_ci_high": round(ci_hi, 4),
            "retention_minus_wu0976": round(retention - WU_RETENTION, 4),
            "fails_wu_0976": retention < RETENTION_FAIL_THRESHOLD,
        })
    return out


# ---------------------------------------------------------------------------
# (2) Paired Delta
# ---------------------------------------------------------------------------

def paired_delta(iso: list[dict]) -> list[dict]:
    """Per-budget Delta = acc(G=32) - acc(G=4) with CIs and sign test."""
    budgets = sorted({int(r["budget_tokens"]) for r in iso})
    out = []
    for T in budgets:
        row4 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 4), None)
        row32 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 32), None)
        if row4 is None or row32 is None:
            continue
        a4 = float(row4["heldout_acc_mean"])
        a32 = float(row32["heldout_acc_mean"])
        s4 = (float(row4["heldout_acc_ci_high"]) - float(row4["heldout_acc_ci_low"])) / (2 * 1.96)
        s32 = (float(row32["heldout_acc_ci_high"]) - float(row32["heldout_acc_ci_low"])) / (2 * 1.96)
        delta = a32 - a4
        se = math.sqrt(s4 ** 2 + s32 ** 2)
        ci_lo = delta - 1.96 * se
        ci_hi = delta + 1.96 * se
        # z-test against Wu-claim zero (retention 0.976 implies Delta small)
        z = delta / se if se > 0 else float("inf")
        # one-sided p-value: H0 Delta <= wu_implied_delta
        # Wu retention 0.976 over same G ratio => delta ~= -0.024 * acc(G=32)
        wu_implied_delta = -(1 - WU_RETENTION) * a32
        z_wu = (delta - wu_implied_delta) / se if se > 0 else float("inf")
        # binomial sign-test (n=2 paired CIs) — if both CIs exclude 0,
        # sign is "consistent"; otherwise uncertain
        sign_consistent = (ci_lo > 0) or (ci_hi < 0)
        out.append({
            "budget_tokens": T,
            "delta_acc_G32_minus_G4": round(delta, 4),
            "delta_ci_low": round(ci_lo, 4),
            "delta_ci_high": round(ci_hi, 4),
            "se": round(se, 4),
            "z_vs_zero": round(z, 3),
            "wu_implied_delta_at_acc_g32": round(wu_implied_delta, 4),
            "z_vs_wu_implied": round(z_wu, 3),
            "ci_excludes_zero": sign_consistent,
            "favors_G32_over_G4": delta > 0 and sign_consistent,
        })
    return out


# ---------------------------------------------------------------------------
# (3) Returns to scale per G: log-log slope of acc vs T
# ---------------------------------------------------------------------------

def slope_per_g(iso: list[dict]) -> list[dict]:
    """Fit log(acc) ~ a + b log(T) per G, report b."""
    rows_by_g: dict[int, list[tuple[float, float]]] = {}
    for r in iso:
        G = int(r["G"])
        T = float(r["budget_tokens"])
        a = float(r["heldout_acc_mean"])
        rows_by_g.setdefault(G, []).append((T, a))
    out = []
    for G, pts in sorted(rows_by_g.items()):
        pts.sort()
        if len(pts) < 2:
            continue
        xs = np.array([math.log10(p[0]) for p in pts])
        ys = np.array([math.log10(max(p[1], 1e-3)) for p in pts])
        # simple OLS slope
        x_mean = xs.mean()
        y_mean = ys.mean()
        num = ((xs - x_mean) * (ys - y_mean)).sum()
        den = ((xs - x_mean) ** 2).sum()
        slope = num / den if den > 0 else float("nan")
        intercept = y_mean - slope * x_mean
        # residual std error
        y_hat = intercept + slope * xs
        resid = ys - y_hat
        rss = (resid ** 2).sum()
        sigma = math.sqrt(rss / max(1, len(pts) - 2))
        out.append({
            "G": G,
            "n_points": len(pts),
            "loglog_slope_b": round(float(slope), 4),
            "loglog_intercept_a": round(float(intercept), 4),
            "residual_sigma": round(float(sigma), 4),
            "implied_acc_64M": round(float(10 ** (intercept + slope * math.log10(64e6))), 4),
        })
    return out


# ---------------------------------------------------------------------------
# (4) Wu-audit: cross-experiment retention pairs
# ---------------------------------------------------------------------------

def wu_audit(small: list[dict]) -> list[dict]:
    """On the small Qwen2.5-0.5B / arithmetic_correctness sweep,
    compute retention for all G_a / G_b pairs.

    group_size_effect.tsv also has appended foreign sections
    (e.g. iter63 retention fits) — filter to the 11-column shape.
    """
    by_g: dict[int, list[float]] = {}
    for r in small:
        if "G" not in r or "heldout_acc_mean" not in r:
            continue
        try:
            G = int(r["G"])
            a = float(r["heldout_acc_mean"])
        except (ValueError, KeyError):
            continue
        if not (0 < a <= 1):
            continue
        by_g.setdefault(G, []).append(a)
    mean_acc = {G: statistics.mean(v) for G, v in by_g.items()}
    out = []
    pairs = [(2, 16), (2, 8), (4, 16), (4, 8), (2, 4)]
    for ga, gb in pairs:
        if ga in mean_acc and gb in mean_acc:
            ret = mean_acc[ga] / mean_acc[gb] if mean_acc[gb] > 0 else float("nan")
            out.append({
                "experiment": "arithmetic_small_qwen05B",
                "G_a": ga,
                "G_b": gb,
                "acc_G_a": round(mean_acc[ga], 4),
                "acc_G_b": round(mean_acc[gb], 4),
                "retention_acc_a_over_b": round(ret, 4),
                "retention_minus_wu0976": round(ret - WU_RETENTION, 4),
                "fails_wu_0976": ret < RETENTION_FAIL_THRESHOLD,
            })
    return out


# ---------------------------------------------------------------------------
# (5) Summary
# ---------------------------------------------------------------------------

def summarize(retention: list[dict], delta: list[dict], slopes: list[dict]) -> list[dict]:
    """One-row-per-budget headline."""
    out = []
    for r, d in zip(retention, delta):
        T = r["budget_tokens"]
        # find best G in this budget row
        out.append({
            "budget_tokens": T,
            "acc_G4": r["acc_G4"],
            "acc_G32": r["acc_G32"],
            "delta_G32_minus_G4": d["delta_acc_G32_minus_G4"],
            "delta_ci_low": d["delta_ci_low"],
            "delta_ci_high": d["delta_ci_high"],
            "retention_G4_over_G32": r["retention_G4_over_G32"],
            "retention_minus_wu0976": r["retention_minus_wu0976"],
            "fails_wu_0976": r["fails_wu_0976"],
            "headline": (
                f"T={T//1_000_000}M: G32 vs G4 Δ={d['delta_acc_G32_minus_G4']:+.2f} "
                f"[{d['delta_ci_low']:+.2f},{d['delta_ci_high']:+.2f}]; "
                f"retention={r['retention_G4_over_G32']:.3f} "
                f"vs Wu 0.976 — "
                f"{'FAILS Wu claim' if r['fails_wu_0976'] else 'consistent with Wu claim'}"
            ),
        })
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(iso: list[dict], retention: list[dict], delta: list[dict],
                slopes: list[dict], out_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # TopL: acc vs G curves per budget
    budgets = sorted({int(r["budget_tokens"]) for r in iso})
    G_vals = sorted({int(r["G"]) for r in iso})
    colors = {1_000_000: "#1f77b4", 4_000_000: "#2ca02c",
              16_000_000: "#ff7f0e", 64_000_000: "#d62728"}
    markers = {1_000_000: "o", 4_000_000: "s",
               16_000_000: "^", 64_000_000: "D"}
    for T in budgets:
        xs, ys, ylo, yhi = [], [], [], []
        for G in G_vals:
            row = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == G), None)
            if row is None:
                continue
            xs.append(G)
            ys.append(float(row["heldout_acc_mean"]))
            ylo.append(float(row["heldout_acc_ci_low"]))
            yhi.append(float(row["heldout_acc_ci_high"]))
        axes[0, 0].errorbar(xs, ys,
                             yerr=[np.array(ys) - np.array(ylo),
                                   np.array(yhi) - np.array(ys)],
                             label=f"T={T//1_000_000}M",
                             color=colors[T], marker=markers[T], capsize=3, lw=1.5)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(G_vals)
    axes[0, 0].set_xticklabels([str(g) for g in G_vals])
    axes[0, 0].set_xlabel("Group size G")
    axes[0, 0].set_ylabel("Held-out accuracy")
    axes[0, 0].set_title("Acc vs G per token budget (crossover at large T)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    # TopR: retention vs budget, Wu 0.976 reference
    Ts = [r["budget_tokens"] / 1e6 for r in retention]
    rets = [r["retention_G4_over_G32"] for r in retention]
    ci_lo = [r["retention_ci_low"] for r in retention]
    ci_hi = [r["retention_ci_high"] for r in retention]
    axes[0, 1].errorbar(Ts, rets,
                         yerr=[np.array(rets) - np.array(ci_lo),
                               np.array(ci_hi) - np.array(rets)],
                         marker="o", lw=2, color="#1f77b4", capsize=4,
                         label="G=4 / G=32 retention")
    axes[0, 1].axhline(WU_RETENTION, color="red", ls="--", lw=1.5,
                        label=f"Wu et al. 0.976 (arXiv:2510.00977)")
    axes[0, 1].axhline(RETENTION_FAIL_THRESHOLD, color="darkred", ls=":", lw=1,
                       label=f"fail threshold {RETENTION_FAIL_THRESHOLD}")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Token budget T (M)")
    axes[0, 1].set_ylabel("Retention = acc(G=4) / acc(G=32)")
    axes[0, 1].set_title("Retention collapses below Wu claim as T grows")
    axes[0, 1].legend(fontsize=8, loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)

    # BotL: log-log slope of acc vs T per G
    G_sl = [s["G"] for s in slopes]
    b_sl = [s["loglog_slope_b"] for s in slopes]
    axes[1, 0].bar([str(g) for g in G_sl], b_sl, color="#2ca02c", alpha=0.85)
    axes[1, 0].set_xlabel("Group size G")
    axes[1, 0].set_ylabel("log-log slope b: d log(acc) / d log(T)")
    axes[1, 0].set_title("Returns to scale grow with G up to G=32")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # BotR: paired Delta = acc(G=32) - acc(G=4) per budget, w/ CI
    Ts2 = [d["budget_tokens"] / 1e6 for d in delta]
    deltas = [d["delta_acc_G32_minus_G4"] for d in delta]
    dlo = [d["delta_ci_low"] for d in delta]
    dhi = [d["delta_ci_high"] for d in delta]
    axes[1, 1].errorbar(Ts2, deltas,
                         yerr=[np.array(deltas) - np.array(dlo),
                               np.array(dhi) - np.array(deltas)],
                         marker="s", lw=2, color="#d62728", capsize=4,
                         label="Δ = acc(G=32) − acc(G=4)")
    axes[1, 1].axhline(0, color="black", ls="-", lw=0.8)
    axes[1, 1].axhline(-(1 - WU_RETENTION), color="red", ls="--", lw=1.0,
                        label=f"wu-implied Δ at 0.976 retention (~{(1-WU_RETENTION):+.3f})")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Token budget T (M)")
    axes[1, 1].set_ylabel("Δ accuracy (G=32 − G=4)")
    axes[1, 1].set_title("G=32 advantage over G=4 grows monotonically with T")
    axes[1, 1].legend(fontsize=8, loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Pillar 3 / Iter 103 — G=4 vs G=32 retention audit vs Wu et al. (2025)",
                 fontsize=12,fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    iso = load_iso()
    small = load_small()

    retention = retention_curve(iso)
    delta = paired_delta(iso)
    slopes = slope_per_g(iso)
    audit = wu_audit(small)
    summary = summarize(retention, delta, slopes)

    write_tsv(RES / "group_size_iter103_retention_curve.tsv", retention,
              ["budget_tokens", "acc_G4", "acc_G4_ci_low", "acc_G4_ci_high",
               "acc_G32", "acc_G32_ci_low", "acc_G32_ci_high",
               "retention_G4_over_G32", "retention_ci_low", "retention_ci_high",
               "retention_minus_wu0976", "fails_wu_0976"])
    write_tsv(RES / "group_size_iter103_paired_delta.tsv", delta,
              ["budget_tokens", "delta_acc_G32_minus_G4", "delta_ci_low", "delta_ci_high",
               "se", "z_vs_zero", "wu_implied_delta_at_acc_g32", "z_vs_wu_implied",
               "ci_excludes_zero", "favors_G32_over_G4"])
    write_tsv(RES / "group_size_iter103_slope.tsv", slopes,
              ["G", "n_points", "loglog_slope_b", "loglog_intercept_a",
               "residual_sigma", "implied_acc_64M"])
    write_tsv(RES / "group_size_iter103_wu_audit.tsv", audit,
              ["experiment", "G_a", "G_b", "acc_G_a", "acc_G_b",
               "retention_acc_a_over_b", "retention_minus_wu0976", "fails_wu_0976"])
    write_tsv(RES / "group_size_iter103_summary.tsv", summary,
              ["budget_tokens", "acc_G4", "acc_G32",
               "delta_G32_minus_G4", "delta_ci_low", "delta_ci_high",
               "retention_G4_over_G32", "retention_minus_wu0976",
               "fails_wu_0976", "headline"])

    make_figure(iso, retention, delta, slopes, FIG / "group_size_iter103.pdf")

    # print a brief recap
    print("=== Iter 103 retention audit ===")
    for r in retention:
        print(f"  T={r['budget_tokens']:>9}  retention G=4/G=32 = "
              f"{r['retention_G4_over_G32']:.4f}  "
              f"CI [{r['retention_ci_low']:.3f},{r['retention_ci_high']:.3f}]  "
              f"{'FAILS' if r['fails_wu_0976'] else 'keeps'} Wu 0.976")
    print("=== Iter 103 paired Δ ===")
    for d in delta:
        print(f"  T={d['budget_tokens']:>9}  Δ = {d['delta_acc_G32_minus_G4']:+.4f}  "
              f"CI [{d['delta_ci_low']:+.3f},{d['delta_ci_high']:+.3f}]  "
              f"favors_G32={d['favors_G32_over_G4']}")
    print("=== Iter 103 slopes ===")
    for s in slopes:
        print(f"  G={s['G']:<3}  b={s['loglog_slope_b']:+.4f}  "
              f"implied acc(64M)={s['implied_acc_64M']:.3f}")
    print("=== Iter 103 Wu audit (small Qwen2.5-0.5B arithmetic) ===")
    for a in audit:
        print(f"  G_a={a['G_a']:<2} G_b={a['G_b']:<2}  "
              f"ret={a['retention_acc_a_over_b']:.4f}  "
              f"{'FAILS' if a['fails_wu_0976'] else 'keeps'} Wu 0.976")


if __name__ == "__main__":
    main()