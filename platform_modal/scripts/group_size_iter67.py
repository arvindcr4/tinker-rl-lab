#!/usr/bin/env python3
"""Iter 67 — Pillar 3 (G=4 vs G=32): Iso-Accuracy Frontier (IAF).

For each target accuracy alpha in [0.50, 0.85], find the minimum token
budget at which each G in {4,8,16,32,64} first reaches alpha.  The
envelope is the IAF; the wasted-token ratio W(alpha, G) compares
each G to the operationally-dominant G*(alpha).  Compares reachable
G-ratio to the arXiv:2510.00977 two-octave structural-equivalence bound.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "experiments" / "results"
FIG = REPO / "figures"


def read_tsv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def solve_min_T_for_acc(target_acc: float, log_t_by_G: dict[int, list[float]],
                        acc_by_G: dict[int, list[float]]) -> tuple[int | None, float | None]:
    """For each G, find the smallest log10(T) at which a linear segment
    in (log T, acc) space crosses target_acc.  Returns (best_G, best_log_T).
    """
    best_G: int | None = None
    best_log_T: float | None = None
    for G, log_t in log_t_by_G.items():
        accs = acc_by_G[G]
        log_T_at_cross: float | None = None
        for i in range(len(log_t) - 1):
            a0, a1 = accs[i], accs[i + 1]
            t0, t1 = log_t[i], log_t[i + 1]
            if (a0 - target_acc) * (a1 - target_acc) <= 0 and (a1 - a0) != 0:
                log_T_at_cross = t0 if a1 == a0 else t0 + ((target_acc - a0) / (a1 - a0)) * (t1 - t0)
                break
        if log_T_at_cross is None:
            continue
        if best_log_T is None or log_T_at_cross < best_log_T:
            best_log_T = log_T_at_cross
            best_G = G
    return best_G, best_log_T


def T_at_cross(alpha: float, log_t: list[float], accs: list[float]) -> float | None:
    """Smallest log10(T) at which a (log T, acc) linear segment crosses alpha."""
    for i in range(len(log_t) - 1):
        a0, a1 = accs[i], accs[i + 1]
        t0, t1 = log_t[i], log_t[i + 1]
        if (a0 - alpha) * (a1 - alpha) <= 0 and (a1 - a0) != 0:
            return t0 if a1 == a0 else t0 + ((alpha - a0) / (a1 - a0)) * (t1 - t0)
    return None


def main() -> None:
    rows = read_tsv(RES / "group_size_token_normalized.tsv")
    log_t_by_G: dict[int, list[float]] = {}
    acc_by_G: dict[int, list[float]] = {}
    ci_lo_by_G: dict[int, list[float]] = {}
    ci_hi_by_G: dict[int, list[float]] = {}
    for r in rows:
        G = int(r["G"])
        log_t_by_G.setdefault(G, []).append(math.log10(float(r["budget_tokens"])))
        acc_by_G.setdefault(G, []).append(float(r["heldout_acc_mean"]))
        ci_lo_by_G.setdefault(G, []).append(float(r["heldout_acc_ci_low"]))
        ci_hi_by_G.setdefault(G, []).append(float(r["heldout_acc_ci_high"]))

    targets = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    iaf_rows: list[dict] = []
    for alpha in targets:
        G_opt, log_T_opt = solve_min_T_for_acc(alpha, log_t_by_G, acc_by_G)
        iaf_rows.append({
            "target_acc": alpha,
            "operationally_optimal_G": G_opt if G_opt is not None else "NA",
            "iso_acc_token_budget_M": round(10 ** log_T_opt / 1e6, 4)
            if log_T_opt is not None else "NA",
            "iso_acc_log10_T": round(log_T_opt, 4)
            if log_T_opt is not None else "NA",
            "reached_by_observed": "yes" if G_opt is not None else "no",
        })

    ratio_rows: list[dict] = []
    for iaf in iaf_rows:
        alpha = iaf["target_acc"]
        if iaf["reached_by_observed"] == "no":
            continue
        G_opt = int(iaf["operationally_optimal_G"])
        T_opt_M = float(iaf["iso_acc_token_budget_M"])
        for G_other in sorted(log_t_by_G):
            log_T_at_cross = T_at_cross(alpha, log_t_by_G[G_other], acc_by_G[G_other])
            base = {
                "target_acc": alpha, "G_compare": G_other,
                "G_is_optimal": "yes" if G_other == G_opt else "no"}
            if log_T_at_cross is None:
                ratio_rows.append({**base, "T_needed_M": "NA",
                                   "waste_ratio_vs_optimal": "NA",
                                   "log10_waste_ratio": "NA"})
                continue
            T_needed_M = 10 ** log_T_at_cross / 1e6
            waste = T_needed_M / T_opt_M
            ratio_rows.append({**base,
                               "T_needed_M": round(T_needed_M, 4),
                               "waste_ratio_vs_optimal": round(waste, 4),
                               "log10_waste_ratio": round(math.log10(waste), 4)})

    se_rows: list[dict] = []
    for alpha in targets:
        reachable = sorted(G for G, accs in acc_by_G.items() if max(accs) >= alpha)
        if len(reachable) >= 2:
            G_ratio = reachable[-1] / reachable[0]
            row = {"target_acc": alpha, "reachable_Gs": ",".join(map(str, reachable)),
                   "G_min": reachable[0], "G_max": reachable[-1],
                   "G_ratio_max": G_ratio, "log2_G_ratio": round(math.log2(G_ratio), 4),
                   "literature_2octave_ratio": 8.0,
                   "exceeds_literature_ratio": "yes" if G_ratio > 8.0 else "no"}
        else:
            row = {"target_acc": alpha, "reachable_Gs": ",".join(map(str, reachable)),
                   "G_min": reachable[0] if reachable else "NA",
                   "G_max": reachable[0] if reachable else "NA",
                   "G_ratio_max": 1.0 if reachable else "NA",
                   "log2_G_ratio": 0.0 if reachable else "NA",
                   "literature_2octave_ratio": 8.0,
                   "exceeds_literature_ratio": "no"}
        se_rows.append(row)

    # Persist outputs
    def write_tsv(path: Path, dicts: list[dict]) -> None:
        if not dicts:
            return
        with path.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()), delimiter="\t")
            w.writeheader()
            for r in dicts:
                w.writerow(r)

    write_tsv(RES / "group_size_iter67_iso_acc_frontier.tsv", iaf_rows)
    write_tsv(RES / "group_size_iter67_iaf_ratios.tsv", ratio_rows)
    write_tsv(RES / "group_size_iter67_se_width.tsv", se_rows)

    # Headline summary
    headline: dict[str, object] = {
        "n_target_accuracies": len(targets),
        "n_Gs": len(log_t_by_G),
        "G_range": f"{min(log_t_by_G)}..{max(log_t_by_G)}",
        "operationally_optimal_G_at_each_acc": ",".join(
            str(r["operationally_optimal_G"]) for r in iaf_rows),
        "literature_2octave_threshold_ratio": 8.0,
    }
    g32_over_g4_ratios: dict[float, float] = {}
    g4_wastes: list[float] = []
    g32_wastes: list[float] = []
    for r in ratio_rows:
        if r["waste_ratio_vs_optimal"] == "NA":
            continue
        w = float(r["waste_ratio_vs_optimal"])
        if r["G_compare"] == 4:
            g4_wastes.append(w)
        elif r["G_compare"] == 32:
            g32_wastes.append(w)
    for r in ratio_rows:
        if r["G_compare"] != 4 or r["waste_ratio_vs_optimal"] == "NA":
            continue
        a = r["target_acc"]
        g32_row = next((rr for rr in ratio_rows
                        if rr["target_acc"] == a and rr["G_compare"] == 32
                        and rr["waste_ratio_vs_optimal"] != "NA"), None)
        if g32_row is None:
            continue
        g4_w = float(r["waste_ratio_vs_optimal"])
        g32_w = float(g32_row["waste_ratio_vs_optimal"])
        if g32_w > 0:
            g32_over_g4_ratios[float(a)] = g4_w / g32_w
    headline["max_waste_G4_vs_optimal"] = max(g4_wastes) if g4_wastes else "NA"
    headline["max_waste_G32_vs_optimal"] = max(g32_wastes) if g32_wastes else "NA"
    headline["max_G32_over_G4_efficiency_ratio"] = (
        max(g32_over_g4_ratios.values()) if g32_over_g4_ratios else "NA")
    headline["G32_over_G4_efficiency_ratio_by_acc"] = json.dumps(
        {f"{a:.2f}": round(v, 4) for a, v in g32_over_g4_ratios.items()})
    headline["max_observed_G_ratio_in_se_test"] = max(
        (float(r["G_ratio_max"]) for r in se_rows
         if isinstance(r["G_ratio_max"], (int, float))),
        default="NA")
    headline["all_se_ratios_exceed_2octave_lit_threshold"] = all(
        r["exceeds_literature_ratio"] == "yes" for r in se_rows
        if isinstance(r["G_ratio_max"], (int, float)))

    write_tsv(RES / "group_size_iter67_iaf_summary.tsv",
              [{"metric": k, "value": v} for k, v in headline.items()])

    meta = {
        "iteration": 67,
        "pillar": "P3-Group-Size",
        "inputs": [
            "platform_hybrid/experiments/results/group_size_token_normalized.tsv",
        ],
        "outputs": [
            "platform_hybrid/experiments/results/group_size_iter67_iso_acc_frontier.tsv",
            "platform_hybrid/experiments/results/group_size_iter67_iaf_ratios.tsv",
            "platform_hybrid/experiments/results/group_size_iter67_se_width.tsv",
            "platform_hybrid/experiments/results/group_size_iter67_iaf_summary.tsv",
            "figures/group_size_iter67_iaf.pdf",
            "figures/group_size_iter67_iaf.png",
        ],
        "method": "Iso-Accuracy Frontier (IAF): for each target accuracy α, the "
                  "minimum token budget at which each G ∈ {4, 8, 16, 32, 64} "
                  "first reaches α.  Envelope is the IAF; comparison yields "
                  "wasted-token ratios and a structural-equivalence half-width.",
        "comparison_to_arXiv_2510.00977":
            "ArXiv:2510.00977 reports G=2 ≈ G=16 (structural equivalence within "
            "a 2-octave G span).  We test whether this ratio (G_max/G_min = 8) "
            "still bounds the structurally-equivalent regime in our setup.",
        "headline_metrics": {k: v for k, v in headline.items()},
    }
    (RES / "group_size_iter67_iter_meta.json").write_text(json.dumps(meta, indent=2))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {4: "#1f77b4", 8: "#ff7f0e", 16: "#2ca02c", 32: "#d62728", 64: "#9467bd"}
    for G in sorted(log_t_by_G):
        ax[0].plot(log_t_by_G[G], acc_by_G[G], "o-",
                   color=colors.get(G, "gray"), label=f"G={G}", alpha=0.85)
        ax[0].fill_between(log_t_by_G[G], ci_lo_by_G[G], ci_hi_by_G[G],
                           color=colors.get(G, "gray"), alpha=0.08)
    for iaf in iaf_rows:
        if iaf["reached_by_observed"] == "yes":
            ax[0].axvline(float(iaf["iso_acc_log10_T"]),
                          linestyle=":", color="gray", alpha=0.5)
            ax[0].text(float(iaf["iso_acc_log10_T"]), float(iaf["target_acc"]) + 0.005,
                       f"α={iaf['target_acc']:.2f}\nG*={iaf['operationally_optimal_G']}",
                       fontsize=7, ha="center")
    ax[0].set_xlabel(r"$\log_{10}(\mathrm{token\ budget\ T})$")
    ax[0].set_ylabel("held-out accuracy")
    ax[0].set_title("Iso-Accuracy Frontier (IAF)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    alpha_axis = sorted({r["target_acc"] for r in ratio_rows})
    G_axis = sorted({int(r["G_compare"]) for r in ratio_rows})
    grid = np.full((len(G_axis), len(alpha_axis)), np.nan)
    for r in ratio_rows:
        if r["waste_ratio_vs_optimal"] == "NA":
            continue
        grid[G_axis.index(int(r["G_compare"])),
             alpha_axis.index(r["target_acc"])] = float(r["waste_ratio_vs_optimal"])
    vmax = max(2.0, np.nanmax(grid))
    im = ax[1].imshow(grid, aspect="auto", origin="lower",
                      cmap="Reds", vmin=1.0, vmax=vmax)
    ax[1].set_xticks(range(len(alpha_axis)))
    ax[1].set_xticklabels([f"{a:.2f}" for a in alpha_axis], rotation=45)
    ax[1].set_yticks(range(len(G_axis)))
    ax[1].set_yticklabels([f"G={g}" for g in G_axis])
    ax[1].set_xlabel(r"target accuracy $\alpha$")
    ax[1].set_ylabel("compare group size")
    ax[1].set_title("Wasted-token ratio W(α, G)")
    fig.colorbar(im, ax=ax[1], label="wasted-token ratio")
    fig.tight_layout()
    fig.savefig(FIG / "group_size_iter67_iaf.pdf")
    fig.savefig(FIG / "group_size_iter67_iaf.png", dpi=130)
    plt.close(fig)

    print("[iter67] headline summary:")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()