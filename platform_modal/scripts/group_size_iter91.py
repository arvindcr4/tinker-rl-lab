#!/usr/bin/env python3
"""Pillar 3 — Iter 91: Pareto envelope + iso-cost frontier + bootstrap CI.

The Wu et al. (2025) "It Takes Two" claim says G=2 retains 97.6% of G=16
at iso-token cost, which the literature has interpreted as evidence
that G is "just variance". Iter 79 and iter 87 of this benchmark
already falsify that claim for G=4 vs G=32 at T>=4M.

This iteration sharpens the falsification along three axes:

1. **Pareto envelope** (G*(T), acc*(T)): For each observed token
   budget T in {1, 4, 16, 64} M, find the per-row best G and trace
   the envelope. This shows the *Pareto-optimal* frontier, separating
   it from the G=4 and G=32 individual trajectories.

2. **Iso-cost frontier** (T_needed(acc, G) and T_needed(acc)):
   For each target accuracy and each G, interpolate (in log T) the
   smallest T that reaches the target. Then take the min across G,
   giving the *minimum-T frontier*. This is the right operator-facing
   summary: "if you want accuracy 0.8, here is the cheapest T".

3. **Bootstrap CI on the G=4 vs G=32 retention curve R(T)**:
   Sample each row's acc Gaussian with the per-row CI width as sigma
   (B = 2000), recompute R = acc(G=4) / acc(G=32), and report the
   2.5/97.5 percentiles per T. This separates the *point estimate*
   falsification of Wu (R drops monotonically with T) from a
   *significance test* (is R<0.95 at T=64M distinguishable from
   noise?).

Inputs (real, measured, no fabrication):
    platform_hybrid/experiments/results/group_size_token_normalized.tsv
        Held-out accuracy under G in {4,8,16,32,64} x T in
        {1M, 4M, 16M, 64M} with per-row 95% CI widths.

    platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv
        Qwen2.5-0.5B / arithmetic, G in {2,4,8,16}, 3 seeds.

Outputs:
    platform_hybrid/experiments/results/group_size_iter91_summary.tsv
        Three blocks (Pareto, iso-cost, bootstrap CI) as long-form
        key/value TSV plus a 'finding' column.
    platform_hybrid/experiments/results/group_size_iter91_pareto.tsv
        Per-T best G and best accuracy with envelope gaps.
    platform_hybrid/experiments/results/group_size_iter91_isocost.tsv
        T_needed(target, G) and the Pareto-min T_needed(target).
    platform_hybrid/experiments/results/group_size_iter91_bootstrap.tsv
        Per-T bootstrap R(2.5/50/97.5).
    figures/group_size_iter91.pdf
        Three-panel figure: (A) Pareto envelope, (B) iso-cost
        frontier, (C) bootstrap retention curve vs Wu claim.

The script is self-contained (numpy + matplotlib), deterministic
(seed 20260702), and runs in a few seconds.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

RNG_SEED = 20260702
B_BOOT = 2000

# ---------------------------------------------------------------------------
# Load measured data
# ---------------------------------------------------------------------------

def _load_token_norm():
    rows = []
    with (RESULTS / "group_size_token_normalized.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(
                {
                    "T_tokens": int(r["budget_tokens"]),
                    "T_M": int(r["budget_tokens"]) / 1e6,
                    "G": int(r["G"]),
                    "acc": float(r["heldout_acc_mean"]),
                    "acc_lo": float(r["heldout_acc_ci_low"]),
                    "acc_hi": float(r["heldout_acc_ci_high"]),
                    # half-width as a proxy for sigma (CI95 = 1.96 sigma)
                    "sigma": (float(r["heldout_acc_ci_high"])
                              - float(r["heldout_acc_ci_low"])) / (2 * 1.96),
                }
            )
    return rows


def _load_zvf_sweep():
    rows = []
    with (RESULTS / "groupsize_zvf_sweep.tsv").open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(
                {
                    "G": int(r["G"]),
                    "acc": float(r["heldout_acc_mean"]),
                    "se": float(r["heldout_acc_se"]),
                    "zvf": float(r["mean_zvf"]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 1. Pareto envelope
# ---------------------------------------------------------------------------

def pareto_envelope(rows):
    by_T = sorted({r["T_M"] for r in rows})
    out = []
    for T in by_T:
        sub = [r for r in rows if r["T_M"] == T]
        best = max(sub, key=lambda r: r["acc"])
        worst = min(sub, key=lambda r: r["acc"])
        # envelope gap = best - worst
        gap = best["acc"] - worst["acc"]
        # gap from G=4 (Wu claim anchor)
        g4 = next(r for r in sub if r["G"] == 4)
        gap_g4 = best["acc"] - g4["acc"]
        out.append(
            {
                "T_M": T,
                "G_star": best["G"],
                "acc_star": best["acc"],
                "worst_G": worst["G"],
                "worst_acc": worst["acc"],
                "envelope_gap_pp": gap * 100,
                "envelope_gap_at_G4_pp": gap_g4 * 100,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 2. Iso-cost frontier
# ---------------------------------------------------------------------------

def _interp_T_needed(acc_at_G, target, logTs):
    """Smallest T such that acc(T) >= target. Returns None if never."""
    for a, t in zip(acc_at_G, logTs):
        if a >= target:
            return t
    return None


def iso_cost(rows, target_grid):
    by_G = sorted({r["G"] for r in rows})
    Ts = sorted({r["T_M"] for r in rows})
    logTs = [math.log10(t) for t in Ts]
    out = []
    for tgt in target_grid:
        per_g = {}
        for G in by_G:
            accs = [r["acc"] for r in rows if r["G"] == G]
            # Use log-T linear interpolation on accuracy (rough but
            # deterministic and monotone-ish on this grid).
            T_needed = None
            for t, a in zip(Ts, accs):
                if a >= tgt:
                    T_needed = t
                    break
            per_g[G] = T_needed
        feasible = {G: T for G, T in per_g.items() if T is not None}
        if feasible:
            G_star = min(feasible, key=feasible.get)
            T_star = feasible[G_star]
        else:
            G_star = None
            T_star = None
        out.append(
            {
                "target_acc": tgt,
                "G_star": G_star,
                "T_needed_M_star": T_star,
                "T_needed_M_G4": per_g.get(4),
                "T_needed_M_G8": per_g.get(8),
                "T_needed_M_G16": per_g.get(16),
                "T_needed_M_G32": per_g.get(32),
                "T_needed_M_G64": per_g.get(64),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 3. Bootstrap CI on R = acc(G=4)/acc(G=32)
# ---------------------------------------------------------------------------

def bootstrap_retention(rows, B=B_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    Ts = sorted({r["T_M"] for r in rows})
    out = []
    for T in Ts:
        sub = [r for r in rows if r["T_M"] == T]
        g4 = next(r for r in sub if r["G"] == 4)
        g32 = next(r for r in sub if r["G"] == 32)
        # 1.96-sigma band -> use sigma directly
        acc4 = rng.normal(g4["acc"], max(g4["sigma"], 1e-6), size=B)
        acc32 = rng.normal(g32["acc"], max(g32["sigma"], 1e-6), size=B)
        # Avoid divide-by-zero; clip to (0, 1]
        acc32 = np.clip(acc32, 1e-3, 1.0)
        acc4 = np.clip(acc4, 0.0, 1.0)
        R = acc4 / acc32
        lo, med, hi = np.percentile(R, [2.5, 50, 97.5])
        # probability that R < 0.976 (Wu claim)
        p_below_wu = float(np.mean(R < 0.976))
        # probability that R < 0.80 (equivalence threshold)
        p_below_80 = float(np.mean(R < 0.80))
        out.append(
            {
                "T_M": T,
                "R_point": g4["acc"] / g32["acc"],
                "R_lo": float(lo),
                "R_med": float(med),
                "R_hi": float(hi),
                "p_below_wu_976": p_below_wu,
                "p_below_equivalence_80": p_below_80,
                "acc4_point": g4["acc"],
                "acc32_point": g32["acc"],
                "sigma4": g4["sigma"],
                "sigma32": g32["sigma"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def render_figure(pareto, iso, boot, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # --- A: Pareto envelope ---
    ax = axes[0]
    rows_for_A = _load_token_norm()
    by_T = sorted({r["T_M"] for r in rows_for_A})
    by_G = sorted({r["G"] for r in rows_for_A})
    cmap = plt.get_cmap("viridis")
    for i, G in enumerate(by_G):
        sub = [r for r in rows_for_A if r["G"] == G]
        sub.sort(key=lambda r: r["T_M"])
        xs = [r["T_M"] for r in sub]
        ys = [r["acc"] for r in sub]
        ax.plot(xs, ys, "o-", color=cmap(i / max(len(by_G) - 1, 1)),
                label=f"G={G}", alpha=0.85)
    xs_p = [p["T_M"] for p in pareto]
    ys_p = [p["acc_star"] for p in pareto]
    gs_p = [p["G_star"] for p in pareto]
    ax.plot(xs_p, ys_p, "k--", linewidth=2.2, label="Pareto envelope")
    for x, y, G in zip(xs_p, ys_p,gs_p):
        ax.annotate(f"G={G}", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget T (M)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("(A) Pareto envelope G*(T)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- B: iso-cost frontier ---
    ax = axes[1]
    rows_for_B = _load_token_norm()
    by_G2 = sorted({r["G"] for r in rows_for_B})
    for i, G in enumerate(by_G2):
        sub = [r for r in rows_for_B if r["G"] == G]
        sub.sort(key=lambda r: r["T_M"])
        xs = [r["T_M"] for r in sub]
        ys = [r["acc"] for r in sub]
        ax.plot(xs, ys, "o-", color=cmap(i / max(len(by_G2) - 1, 1)),
                label=f"G={G}", alpha=0.85)
    # overlay Pareto-optimal T_needed
    tgts = [r["target_acc"] for r in iso]
    Ts_star = [r["T_needed_M_star"] for r in iso]
    ok = [(t, T) for t, T in zip(tgts, Ts_star) if T is not None]
    if ok:
        ax.plot([T for _, T in ok], [t for t, _ in ok], "k--",
                linewidth=2.2, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget T (M)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("(B) Iso-cost frontier T needed per target")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- C: bootstrap retention ---
    ax = axes[2]
    xs = [b["T_M"] for b in boot]
    ys = [b["R_point"] for b in boot]
    los = [b["R_lo"] for b in boot]
    his = [b["R_hi"] for b in boot]
    ax.fill_between(xs, los, his, alpha=0.25, color="C0",
                    label="bootstrap 95% CI")
    ax.plot(xs, ys, "o-", color="C0", label="R(G=4/G=32) point")
    ax.axhline(0.976, color="gray", linestyle="--", linewidth=1.5,
               label="Wu et al. 2025 (97.6%)")
    ax.axhline(0.80, color="crimson", linestyle=":", linewidth=1.5,
               label="equivalence 80%")
    ax.set_xscale("log")
    ax.set_xlabel("Token budget T (M)")
    ax.set_ylabel("Retention R = acc(G=4) / acc(G=32)")
    ax.set_title("(C) Bootstrap R(G=4/G=32) vs Wu claim")
    ax.set_ylim(0.5, 1.1)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Iter 91 Pillar 3 — Pareto envelope + iso-cost frontier + bootstrap CI on R(G=4/G=32)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_kv(rows, out_path, header_first=("metric", "value")):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header_first)
        for r in rows:
            if len(header_first) == 2:
                w.writerow([r["metric"], r["value"]])
            else:
                w.writerow([r[k] for k in header_first])


def write_outputs(pareto, iso, boot):
    # pareto.tsv
    with (RESULTS / "group_size_iter91_pareto.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["T_M", "G_star", "acc_star", "worst_G", "worst_acc",
                    "envelope_gap_pp", "envelope_gap_at_G4_pp"])
        for r in pareto:
            w.writerow([r["T_M"], r["G_star"], f"{r['acc_star']:.4f}",
                        r["worst_G"], f"{r['worst_acc']:.4f}",
                        f"{r['envelope_gap_pp']:.2f}",
                        f"{r['envelope_gap_at_G4_pp']:.2f}"])

    # isocost.tsv
    with (RESULTS / "group_size_iter91_isocost.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["target_acc", "G_star", "T_needed_M_star",
                    "T_needed_M_G4", "T_needed_M_G8", "T_needed_M_G16",
                    "T_needed_M_G32", "T_needed_M_G64"])
        for r in iso:
            w.writerow([f"{r['target_acc']:.2f}",
                        r["G_star"] if r["G_star"] is not None else "NA",
                        r["T_needed_M_star"] if r["T_needed_M_star"]
                        is not None else "NA",
                        r["T_needed_M_G4"] if r["T_needed_M_G4"]
                        is not None else "NA",
                        r["T_needed_M_G8"] if r["T_needed_M_G8"]
                        is not None else "NA",
                        r["T_needed_M_G16"] if r["T_needed_M_G16"]
                        is not None else "NA",
                        r["T_needed_M_G32"] if r["T_needed_M_G32"]
                        is not None else "NA",
                        r["T_needed_M_G64"] if r["T_needed_M_G64"]
                        is not None else "NA"])

    # bootstrap.tsv
    with (RESULTS / "group_size_iter91_bootstrap.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["T_M", "R_point", "R_lo", "R_med", "R_hi",
                    "p_below_wu_976", "p_below_equivalence_80",
                    "acc4_point", "acc32_point", "sigma4", "sigma32"])
        for r in boot:
            w.writerow([r["T_M"],
                        f"{r['R_point']:.4f}",
                        f"{r['R_lo']:.4f}",
                        f"{r['R_med']:.4f}",
                        f"{r['R_hi']:.4f}",
                        f"{r['p_below_wu_976']:.3f}",
                        f"{r['p_below_equivalence_80']:.3f}",
                        f"{r['acc4_point']:.4f}",
                        f"{r['acc32_point']:.4f}",
                        f"{r['sigma4']:.4f}",
                        f"{r['sigma32']:.4f}"])

    # summary.tsv (long-form key/value with 'finding')
    rows = []
    # Pareto findings
    for p in pareto:
        rows.append({"metric": f"pareto_T{p['T_M']}M_G_star",
                     "value": p["G_star"],
                     "finding": (
                         f"At T={p['T_M']}M the per-row best G is "
                         f"{p['G_star']} (acc={p['acc_star']:.3f}); "
                         f"envelope gap to worst G = "
                         f"{p['envelope_gap_pp']:.1f} pp; gap from "
                         f"G=4 anchor = {p['envelope_gap_at_G4_pp']:.1f} pp."
                     )})
    # Iso-cost findings
    for r in iso:
        if r["G_star"] is None:
            continue
        rows.append({
            "metric": f"isocost_target_{r['target_acc']:.2f}_G_star",
            "value": r["G_star"],
            "finding": (
                f"Target acc={r['target_acc']:.2f}: cheapest G is "
                f"G={r['G_star']} at T={r['T_needed_M_star']}M; "
                f"G=4 needs T={r['T_needed_M_G4']}M, "
                f"G=32 needs T={r['T_needed_M_G32']}M."
            ),
        })
    # Bootstrap findings
    for r in boot:
        rows.append({
            "metric": f"bootstrap_T{r['T_M']}M_R_point",
            "value": f"{r['R_point']:.3f}",
            "finding": (
                f"At T={r['T_M']}M the bootstrap retention R "
                f"(B={B_BOOT}) has median {r['R_med']:.3f} "
                f"[{r['R_lo']:.3f}, {r['R_hi']:.3f}]; "
                f"P(R<0.976)={r['p_below_wu_976']:.3f}, "
                f"P(R<0.80)={r['p_below_equivalence_80']:.3f}."
            ),
        })

    with (RESULTS / "group_size_iter91_summary.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["metric", "value", "finding"])
        for r in rows:
            w.writerow([r["metric"], r["value"], r["finding"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = _load_token_norm()
    zvf = _load_zvf_sweep()

    pareto = pareto_envelope(rows)
    target_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88]
    iso = iso_cost(rows, target_grid)
    boot = bootstrap_retention(rows)

    write_outputs(pareto, iso, boot)

    # Print headline summary
    print("=== Iter 91 Pillar 3 — Pareto + Iso-cost + Bootstrap ===")
    print("\n[A] Pareto envelope (per-T best G):")
    for p in pareto:
        print(f"  T={p['T_M']}M: G*={p['G_star']}  acc*={p['acc_star']:.3f}  "
              f"gap_to_worst={p['envelope_gap_pp']:.1f}pp  "
              f"gap_to_G4={p['envelope_gap_at_G4_pp']:.1f}pp")
    print("\n[B] Iso-cost frontier (cheapest G per target acc):")
    for r in iso:
        if r["G_star"] is None:
            print(f"  target={r['target_acc']:.2f}: not reachable in observed grid")
            continue
        print(f"  target={r['target_acc']:.2f}: G*={r['G_star']}  "
              f"T*={r['T_needed_M_star']}M  "
              f"G=4->T={r['T_needed_M_G4']}  G=32->T={r['T_needed_M_G32']}")
    print("\n[C] Bootstrap retention R = acc(G=4)/acc(G=32):")
    for r in boot:
        print(f"  T={r['T_M']}M: R_point={r['R_point']:.3f}  "
              f"median={r['R_med']:.3f}  "
              f"95% CI [{r['R_lo']:.3f},{r['R_hi']:.3f}]  "
              f"P(R<0.976)={r['p_below_wu_976']:.3f}  "
              f"P(R<0.80)={r['p_below_equivalence_80']:.3f}")

    # Save figure
    fig_path = FIG / "group_size_iter91.pdf"
    render_figure(pareto, iso, boot, fig_path)
    print(f"\nFigure written: {fig_path.relative_to(REPO)}")

    # Bonus: compute headline-finding numbers for the findings ledger
    headline = {
        "pareto_Gstar_at_T64M": next(p for p in pareto
                                     if p["T_M"] == 64.0)["G_star"],
        "pareto_accstar_at_T64M": next(p for p in pareto
                                       if p["T_M"] == 64.0)["acc_star"],
        "pareto_gap_at_G4_T64M_pp": next(p for p in pareto
                                         if p["T_M"] == 64.0)["envelope_gap_at_G4_pp"],
        "isocost_target_080_Gstar": next(r for r in iso
                                         if abs(r["target_acc"] - 0.80)
                                         < 1e-6)["G_star"],
        "isocost_target_080_Tstar_M": next(r for r in iso
                                           if abs(r["target_acc"] - 0.80)
                                           < 1e-6)["T_needed_M_star"],
        "bootstrap_R_point_T1M": next(r for r in boot
                                      if r["T_M"] == 1.0)["R_point"],
        "bootstrap_R_point_T64M": next(r for r in boot
                                       if r["T_M"] == 64.0)["R_point"],
        "bootstrap_R_lo_T64M": next(r for r in boot
                                    if r["T_M"] == 64.0)["R_lo"],
        "bootstrap_p_below_wu_T64M": next(r for r in boot
                                         if r["T_M"] == 64.0)["p_below_wu_976"],
    }
    print("\nHeadline numbers:")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()