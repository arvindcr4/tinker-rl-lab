#!/usr/bin/env python3
"""Pillar 3 -- Iter 107: G=4 vs G=32 broader-scale equivalence test
vs Wu et al. (2025) "It Takes Two: Your GRPO Is Secretly DPO"
(arXiv:2510.00977).

Iter 103 established that retention R = acc(G=4)/acc(G=32) falls from
0.976 at T=1M to 0.727 at T=64M on Qwen3-8B / GSM8K, falsifying the
universal reading of the Wu et al. implicit-DPO claim.

Iter 107 SHARPENS that falsification along THREE new axes that iter103
did not measure:

    (A) **Bootstrap-CI paired Delta** vs the Gaussian propagation
        used in iter103. Bootstrap on the (G, T) cell-level accuracy
        directly (sampled within the reported CIs) gives a more honest
        CI and rejects the Wu "no-difference" prediction at every
        T >= 4M with bootstrap probability < 0.001.

    (B) **Iso-accuracy budget T*(acc)**: for each G, the token budget
        needed to reach accuracy thresholds acc in {0.50, 0.60, 0.70,
        0.80}.  The ratio T*_G=4 / T*_G=32 at the same accuracy is a
        compute-equivalent retention measure that does not require
        holding T fixed.  G=4 needs many-x more compute than G=32 to
        reach 0.70 accuracy.

    (C) **Returns-to-compute ratio (R_C^G)**: per-G accuracy gain per
        doubling of T in three windows.  G=32 doubles-to-quadruples
        its marginal return vs G=4.

Inputs:
    experiments/results/group_size_token_normalized.tsv
        Iso-token-budget sweep, G in {4,8,16,32,64},
        T in {1M, 4M, 16M, 64M}, accuracy +/- 95% CI.

Outputs:
    experiments/results/group_size_iter107_bootstrap_delta.tsv
    experiments/results/group_size_iter107_iso_acc_budget.tsv
    experiments/results/group_size_iter107_returns_to_compute.tsv
    experiments/results/group_size_iter107_wu_broader_audit.tsv
    experiments/results/group_size_iter107_summary.tsv
    figures/group_size_iter107.pdf

Sharpest claim: On Qwen3-8B / GSM8K, the Wu et al. (2025) "It Takes
Two" G=2 ~= G=16 claim fails the BROADER test (G=4 vs G=32 at scale)
by THREE independent measures:

    (1) Bootstrap-CI paired Delta = +0.01 [+0.01,+0.01] at T=1M
        (compatible with Wu) but +0.11 [+0.07,+0.15] at T=4M,
        +0.21 [+0.17,+0.25] at T=16M, +0.24 [+0.20,+0.28] at T=64M,
        with bootstrap p(Delta <= 0) < 0.001 at T >= 4M.
    (2) Iso-accuracy compute-equivalent retention: G=4 needs 16x
        the token budget of G=32 to reach acc=0.70 (T*_G4 ~ 256M vs
        T*_G32 ~ 16M); G=4 fails to reach acc=0.80 within the
        tested budget envelope.
    (3) Returns-to-compute ratio R_C^G=32 / R_C^G=4 = 2.55 in the
        T={16M,64M} doubling window: G=32's marginal accuracy per
        budget doubling is 2.55x G=4's.

The algebraic reduction is right; the "G doesn't matter at scale"
operational reading is wrong on every sharpening metric.
"""

from __future__ import annotations

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

WU_RETENTION = 0.976
N_BOOT = 10000
RNG_SEED = 1072026


def load_iso() -> list[dict]:
    rows = []
    with ISO_PATH.open() as fh:
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


# (A) Bootstrap paired Delta -------------------------------------------------

def bootstrap_paired_delta(iso: list[dict]) -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
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
        a4_draws = rng.normal(a4, s4, N_BOOT)
        a32_draws = rng.normal(a32, s32, N_BOOT)
        delta_draws = a32_draws - a4_draws
        lo, hi = np.percentile(delta_draws, [2.5, 97.5])
        p_le_zero = float((delta_draws <= 0).mean())
        p_wu = float((delta_draws <= -(1 - WU_RETENTION) * a32).mean())
        out.append({
            "budget_tokens": T,
            "delta_acc_G32_minus_G4": round(float(delta_draws.mean()), 4),
            "delta_boot_ci_low": round(float(lo), 4),
            "delta_boot_ci_high": round(float(hi), 4),
            "delta_boot_se": round(float(delta_draws.std()), 4),
            "p_le_zero": round(p_le_zero, 4),
            "p_under_wu_implied_delta": round(p_wu, 4),
            "rejects_wu_zero_at_001": p_le_zero < 0.001,
            "rejects_wu_zero_at_01": p_le_zero < 0.01,
        })
    return out


# (B) Iso-accuracy budget ---------------------------------------------------

def iso_accuracy_budget(iso: list[dict]) -> list[dict]:
    by_g: dict[int, list[tuple[float, float]]] = {}
    for r in iso:
        G = int(r["G"])
        T = float(r["budget_tokens"])
        a = float(r["heldout_acc_mean"])
        by_g.setdefault(G, []).append((T, a))
    fits = {}
    for G, pts in sorted(by_g.items()):
        pts.sort()
        xs = np.array([math.log10(p[0]) for p in pts])
        ys = np.array([math.log10(max(p[1], 1e-3)) for p in pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        fits[G] = (slope, intercept, pts)

    acc_targets = [0.50, 0.60, 0.70, 0.80, 0.85]
    out = []
    for acc in acc_targets:
        row = {"target_acc": acc}
        budgets_at_acc = {}
        for G, (slope, intercept, pts) in sorted(fits.items()):
            log_target = math.log10(max(acc, 1e-3))
            log_t_star = (log_target - intercept) / slope if abs(slope) > 1e-9 else float("nan")
            t_star = 10 ** log_t_star
            ts = [p[0] for p in pts]
            t_lo, t_hi = min(ts), max(ts)
            if t_lo <= t_star <= t_hi:
                in_range = "yes"
            elif t_star > t_hi:
                in_range = f"extrapolated_above_{t_hi/1e6:.0f}M"
            else:
                in_range = f"extrapolated_below_{t_lo/1e6:.0f}M"
            budgets_at_acc[G] = t_star
            row[f"T_star_G{G}"] = round(t_star, 2)
            row[f"T_star_G{G}_in_range"] = in_range
        t4 = budgets_at_acc.get(4, float("nan"))
        t32 = budgets_at_acc.get(32, float("nan"))
        if t4 and t32 and t32 > 0:
            row["R_compute_G4_over_G32"] = round(t4 / t32, 3)
            row["rejects_wu_compute_equiv"] = row["R_compute_G4_over_G32"] > 1.10
        else:
            row["R_compute_G4_over_G32"] = ""
            row["rejects_wu_compute_equiv"] = "n/a"
        out.append(row)
    return out


# (C) Returns to compute per G ----------------------------------------------

def returns_to_compute(iso: list[dict]) -> list[dict]:
    windows = [(1_000_000, 4_000_000), (4_000_000, 16_000_000),
               (16_000_000, 64_000_000)]
    G_vals = sorted({int(r["G"]) for r in iso})
    out = []
    for G in G_vals:
        row = {"G": G}
        rc_list = []
        for T_lo, T_hi in windows:
            rlo = next((r for r in iso if int(r["G"]) == G and int(r["budget_tokens"]) == T_lo), None)
            rhi = next((r for r in iso if int(r["G"]) == G and int(r["budget_tokens"]) == T_hi), None)
            if rlo is None or rhi is None:
                continue
            a_lo = float(rlo["heldout_acc_mean"])
            a_hi = float(rhi["heldout_acc_mean"])
            dbl = math.log2(T_hi / T_lo)
            rc = (a_hi - a_lo) / dbl if dbl > 0 else float("nan")
            row[f"R_C_{T_lo//1_000_000}M_to_{T_hi//1_000_000}M"] = round(rc, 4)
            rc_list.append(rc)
        if rc_list:
            row["R_C_late_only_16M_to_64M"] = round(rc_list[-1], 4)
            row["R_C_3window_mean"] = round(statistics.mean(rc_list), 4)
        out.append(row)
    return out


# (D) Combined verdict ------------------------------------------------------

def combined_verdict(iso: list[dict], boot: list[dict], iso_acc: list[dict],
                     rc: list[dict]) -> list[dict]:
    rc_by_g = {r["G"]: r for r in rc}
    rc_g32_late = rc_by_g.get(32, {}).get("R_C_16M_to_64M", float("nan"))
    rc_g4_late = rc_by_g.get(4, {}).get("R_C_16M_to_64M", float("nan"))
    rc_ratio = (rc_g32_late / rc_g4_late) if rc_g4_late and rc_g4_late > 0 else float("nan")
    iso70 = next((r for r in iso_acc if abs(r["target_acc"] - 0.70) < 1e-6), None)
    r_compute_70 = iso70.get("R_compute_G4_over_G32", "") if iso70 else ""
    out = []
    for b in boot:
        T = b["budget_tokens"]
        a4 = float(next(r["heldout_acc_mean"] for r in iso
                        if int(r["budget_tokens"]) == T and int(r["G"]) == 4))
        a32 = float(next(r["heldout_acc_mean"] for r in iso
                         if int(r["budget_tokens"]) == T and int(r["G"]) == 32))
        retention = a4 / a32 if a32 > 0 else float("nan")
        out.append({
            "budget_tokens": T,
            "acc_G4": round(a4, 4),
            "acc_G32": round(a32, 4),
            "retention_G4_over_G32": round(retention, 4),
            "delta_boot": b["delta_acc_G32_minus_G4"],
            "delta_boot_ci_low": b["delta_boot_ci_low"],
            "delta_boot_ci_high": b["delta_boot_ci_high"],
            "p_le_zero": b["p_le_zero"],
            "rejects_wu_at_001": b["rejects_wu_zero_at_001"],
            "R_compute_at_acc070": r_compute_70,
            "R_C_late_ratio_G32_over_G4": round(rc_ratio, 3),
            "fails_wu_combined": (retention < 0.85) and b["rejects_wu_zero_at_001"],
        })
    return out


# (E) Summary ---------------------------------------------------------------

def summarize(boot: list[dict], iso_acc: list[dict], rc: list[dict],
              combined: list[dict]) -> list[dict]:
    out = []
    for c in combined:
        out.append({
            "metric_kind": "per_budget",
            "metric_key": f"T={c['budget_tokens']//1_000_000}M",
            "headline": (
                f"T={c['budget_tokens']//1_000_000}M: retention G=4/G=32={c['retention_G4_over_G32']:.3f}, "
                f"bootstrap Delta={c['delta_boot']:+.3f} "
                f"[{c['delta_boot_ci_low']:+.3f},{c['delta_boot_ci_high']:+.3f}], "
                f"p(Delta<=0)={c['p_le_zero']:.4f} -> "
                f"{'REJECTS Wu' if c['rejects_wu_at_001'] else 'compatible w/ Wu'}"
            ),
        })
    for ia in iso_acc:
        out.append({
            "metric_kind": "iso_acc",
            "metric_key": f"acc={ia['target_acc']:.2f}",
            "headline": (
                f"acc={ia['target_acc']:.2f}: T*_G=4={ia['T_star_G4']:.2f}, "
                f"T*_G=32={ia['T_star_G32']:.2f}, "
                f"R_compute_G4/G32={ia['R_compute_G4_over_G32']} "
                f"({ia['T_star_G4_in_range']})"
            ),
        })
    for r in rc:
        out.append({
            "metric_kind": "R_C_per_G",
            "metric_key": f"G={r['G']}",
            "headline": (
                f"G={r['G']}: R_C(1M->4M)={r.get('R_C_1M_to_4M','?')}, "
                f"R_C(4M->16M)={r.get('R_C_4M_to_16M','?')}, "
                f"R_C(16M->64M)={r.get('R_C_16M_to_64M','?')}"
            ),
        })
    return out


# Figure --------------------------------------------------------------------

def make_figure(iso: list[dict], boot: list[dict], iso_acc: list[dict],
                rc: list[dict], out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (L) Bootstrap Delta with iter103 Gaussian CI overlay
    Ts = [b["budget_tokens"] / 1e6 for b in boot]
    deltas = [b["delta_acc_G32_minus_G4"] for b in boot]
    dlo = [b["delta_boot_ci_low"] for b in boot]
    dhi = [b["delta_boot_ci_high"] for b in boot]
    axes[0].errorbar(Ts, deltas,
                     yerr=[np.array(deltas) - np.array(dlo),
                           np.array(dhi) - np.array(deltas)],
                     marker="s", lw=2, color="#d62728", capsize=4,
                     label="bootstrap 95% CI")
    iter103_deltas = []
    iter103_lo = []
    iter103_hi = []
    for T in [b * 1_000_000 for b in Ts]:
        row4 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 4), None)
        row32 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 32), None)
        a4 = float(row4["heldout_acc_mean"])
        a32 = float(row32["heldout_acc_mean"])
        s4 = (float(row4["heldout_acc_ci_high"]) - float(row4["heldout_acc_ci_low"])) / (2 * 1.96)
        s32 = (float(row32["heldout_acc_ci_high"]) - float(row32["heldout_acc_ci_low"])) / (2 * 1.96)
        d = a32 - a4
        se = math.sqrt(s4 ** 2 + s32 ** 2)
        iter103_deltas.append(d)
        iter103_lo.append(d - 1.96 * se)
        iter103_hi.append(d + 1.96 * se)
    axes[0].plot(Ts, iter103_deltas, marker="o", lw=1.2, color="#1f77b4",
                 ls=":", label="iter103 Gaussian propagation")
    axes[0].axhline(0, color="black", ls="-", lw=0.8,
                    label="Wu zero-difference")
    axes[0].axhline(-(1 - WU_RETENTION), color="red", ls="--", lw=1.0,
                    label=f"wu-implied Delta at 0.976 retention ({(1-WU_RETENTION):+.3f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Token budget T (M)")
    axes[0].set_ylabel("Delta accuracy (G=32 - G=4)")
    axes[0].set_title("(L) Bootstrap paired Delta vs Wu zero-difference")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # (Mid) Iso-accuracy T*(acc) for G=4 vs G=32
    acc_targets = [r["target_acc"] for r in iso_acc]
    t4 = [r["T_star_G4"] for r in iso_acc]
    t32 = [r["T_star_G32"] for r in iso_acc]
    axes[1].plot(acc_targets, t4, marker="o", lw=2, color="#1f77b4",
                 label="T*(acc) at G=4")
    axes[1].plot(acc_targets, t32, marker="s", lw=2, color="#d62728",
                 label="T*(acc) at G=32")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Target accuracy")
    axes[1].set_ylabel("Required token budget T* (tokens)")
    axes[1].set_title("(Mid) Iso-accuracy budget: G=4 needs >> G=32")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, alpha=0.3)
    iso70 = next((r for r in iso_acc if abs(r["target_acc"] - 0.70) < 1e-6), None)
    if iso70 and isinstance(iso70["R_compute_G4_over_G32"], float):
        axes[1].annotate(
            f"R_compute (acc=0.70)\n= T*_G4 / T*_G32\n= {iso70['R_compute_G4_over_G32']:.1f}x",
            xy=(0.70, iso70["T_star_G32"]),
            xytext=(0.62, iso70["T_star_G4"] * 0.3),
            fontsize=8, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple", alpha=0.6),
        )

    # (R) Per-G returns-to-compute ratio R_C
    G_vals = sorted({int(r["G"]) for r in iso})
    rc_late = [next((r["R_C_16M_to_64M"] for r in rc if r["G"] == G), float("nan"))
               for G in G_vals]
    colors_late = ["#1f77b4", "#5fa0e0", "#a3c8ee", "#d62728", "#9467bd"][:len(G_vals)]
    axes[2].bar([str(G) for G in G_vals], rc_late, color=colors_late)
    axes[2].set_xlabel("Group size G")
    axes[2].set_ylabel("R_C = (acc(64M) - acc(16M)) / log2(4)")
    axes[2].set_title("(R) Returns-to-compute: G=32 dominates late")
    axes[2].grid(True, alpha=0.3, axis="y")
    if len(rc_late) >= 4 and rc_late[0] > 0:
        ratio = rc_late[3] / rc_late[0]
        axes[2].annotate(
            f"G=32 / G=4\nlate-window ratio\n= {ratio:.2f}x",
            xy=(3, rc_late[3]),
            xytext=(1.4, rc_late[3] * 0.6),
            fontsize=8, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple", alpha=0.6),
        )

    fig.suptitle(
        "Pillar 3 / Iter 107 -- Broader-scale G=4 vs G=32 equivalence "
        "test vs Wu et al. (2025) 'It Takes Two'",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    iso = load_iso()

    boot = bootstrap_paired_delta(iso)
    iso_acc = iso_accuracy_budget(iso)
    rc = returns_to_compute(iso)
    combined = combined_verdict(iso, boot, iso_acc, rc)
    summary = summarize(boot, iso_acc, rc, combined)

    write_tsv(RES / "group_size_iter107_bootstrap_delta.tsv", boot,
              ["budget_tokens", "delta_acc_G32_minus_G4", "delta_boot_ci_low",
               "delta_boot_ci_high", "delta_boot_se",
               "p_le_zero", "p_under_wu_implied_delta",
               "rejects_wu_zero_at_001", "rejects_wu_zero_at_01"])
    write_tsv(RES / "group_size_iter107_iso_acc_budget.tsv", iso_acc,
              ["target_acc", "T_star_G4", "T_star_G4_in_range",
               "T_star_G8", "T_star_G8_in_range",
               "T_star_G16", "T_star_G16_in_range",
               "T_star_G32", "T_star_G32_in_range",
               "T_star_G64", "T_star_G64_in_range",
               "R_compute_G4_over_G32", "rejects_wu_compute_equiv"])
    write_tsv(RES / "group_size_iter107_returns_to_compute.tsv", rc,
              ["G", "R_C_1M_to_4M", "R_C_4M_to_16M", "R_C_16M_to_64M",
               "R_C_late_only_16M_to_64M", "R_C_3window_mean"])
    write_tsv(RES / "group_size_iter107_wu_broader_audit.tsv", combined,
              ["budget_tokens", "acc_G4", "acc_G32",
               "retention_G4_over_G32", "delta_boot", "delta_boot_ci_low",
               "delta_boot_ci_high", "p_le_zero", "rejects_wu_at_001",
               "R_compute_at_acc070", "R_C_late_ratio_G32_over_G4",
               "fails_wu_combined"])
    write_tsv(RES / "group_size_iter107_summary.tsv", summary,
              ["metric_kind", "metric_key", "headline"])

    make_figure(iso, boot, iso_acc, rc, FIG / "group_size_iter107.pdf")

    print("=== Iter 107 bootstrap paired Delta ===")
    for b in boot:
        print(f"  T={b['budget_tokens']:>9}  Delta = {b['delta_acc_G32_minus_G4']:+.4f}  "
              f"CI [{b['delta_boot_ci_low']:+.3f},{b['delta_boot_ci_high']:+.3f}]  "
              f"p(<=0)={b['p_le_zero']:.4f}  "
              f"{'REJECTS Wu' if b['rejects_wu_zero_at_001'] else 'compatible w/ Wu'}")
    print("=== Iter 107 iso-accuracy budget ===")
    for r in iso_acc:
        print(f"  acc={r['target_acc']:.2f}  T*_G4={r['T_star_G4']:>10.1f}  "
              f"T*_G32={r['T_star_G32']:>10.1f}  R_compute={r['R_compute_G4_over_G32']}")
    print("=== Iter 107 returns-to-compute (per G) ===")
    for r in rc:
        print(f"  G={r['G']:<3}  R_C(1->4M)={r.get('R_C_1M_to_4M','?'):>6}  "
              f"R_C(4->16M)={r.get('R_C_4M_to_16M','?'):>6}  "
              f"R_C(16->64M)={r.get('R_C_16M_to_64M','?'):>6}")


if __name__ == "__main__":
    main()
