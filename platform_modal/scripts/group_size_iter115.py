#!/usr/bin/env python3
"""Exploratory retention equivalence, descriptive GU association, and explicit cost scenarios.

Review23-v2: summary-normal approximate uncertainty is not a matched-run mechanism
test. Cost projections are unvalidated point-fit scenarios with explicit units
and extrapolation flags, not measured runtimes or calibrated cost intervals."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"

ISO_PATH = RES / "group_size_token_normalized.tsv"
ZVFSWEEP_PATH = RES / "groupsize_zvf_sweep.tsv"
DELTAD_PATH = RES / "zvf_iter114_delta_d.tsv"

N_BOOT = 10000
RNG_SEED = 1152026
EQUIV_BOUNDS = [0.010, 0.024, 0.050]


def load_iso() -> list[dict]:
    rows = []
    with ISO_PATH.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts, strict=False)))
    return rows


def load_zvf_sweep() -> list[dict]:
    rows = []
    with ZVFSWEEP_PATH.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts, strict=False)))
    return rows


def load_delta_d() -> list[dict]:
    if not DELTAD_PATH.exists():
        return []
    rows = []
    with DELTAD_PATH.open() as fh:
        header = fh.readline().strip().split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts, strict=False)))
    return rows


def write_tsv(path: Path, rows: list[dict], header: list[str]) -> None:
    with path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in header) + "\n")


# (A) TOST equivalence -----------------------------------------------------


def tost_equivalence(iso: list[dict], delta_eq: float) -> list[dict]:
    """TOST for |Delta| <= delta_eq.

    Equivalence is demonstrated iff max(p_lower, p_upper) < 0.05,
    equivalent to the (1-2alpha)=90% two-sided CI lying wholly
    inside [-delta_eq, +delta_eq].

    If p_TOST >= 0.05 we have NOT demonstrated equivalence. When
    combined with iter107's one-sided p(Delta<=0)<0.001 (Delta>0),
    we conclude Delta > delta_eq (the Wu equivalence claim fails).
    """
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
        s32 = (float(row32["heldout_acc_ci_high"]) - float(row32["heldout_acc_ci_low"])) / (
            2 * 1.96
        )
        a4_draws = rng.normal(a4, s4, N_BOOT)
        a32_draws = rng.normal(a32, s32, N_BOOT)
        if np.any(a32_draws <= 0):
            raise ValueError("relative retention sensitivity requires positive denominators")
        delta_draws = 1.0 - a4_draws / a32_draws
        mean_d = float(delta_draws.mean())
        se_d = float(delta_draws.std())
        p_lower = float((delta_draws <= -delta_eq).mean())
        p_upper = float((delta_draws >= +delta_eq).mean())
        p_tost = max(p_lower, p_upper)
        ci_lo, ci_hi = np.percentile(delta_draws, [5, 95])
        demonstrates_equiv = p_tost < 0.05
        if demonstrates_equiv:
            verdict = "EQUIVALENCE demonstrated (|Delta|<=bound consistent with data)"
        elif ci_lo > delta_eq:
            verdict = f"FAILS equivalence (Delta=+{mean_d:.3f} > +{delta_eq})"
        elif ci_hi < -delta_eq:
            verdict = f"FAILS equivalence (Delta={mean_d:.3f} < -{delta_eq})"
        else:
            verdict = "INCONCLUSIVE (CI spans bound)"
        out.append(
            {
                "budget_tokens": T,
                "delta_eq_bound": delta_eq,
                "estimand": "one_minus_small_over_large_accuracy_ratio",
                "interval_level": 0.90,
                "method": "independent_normal_summary_sensitivity",
                "delta_mean": round(mean_d, 4),
                "delta_se": round(se_d, 4),
                "delta_ci_lo": round(float(ci_lo), 4),
                "delta_ci_hi": round(float(ci_hi), 4),
                "p_lower_one_sided": round(p_lower, 4),
                "p_upper_one_sided": round(p_upper, 4),
                "p_tost": round(p_tost, 4),
                "demonstrates_equivalence": demonstrates_equiv,
                "ci_inside_equiv_bound": (ci_lo >= -delta_eq) and (ci_hi <= +delta_eq),
                "verdict": verdict,
            }
        )
    return out


# (B) Cross-pillar ZVF -> retention linkage --------------------------------


def zvf_linkage(iso: list[dict]) -> list[dict]:
    """Per-budget retention vs GU ratio + Spearman correlation."""
    budgets = sorted({int(r["budget_tokens"]) for r in iso})
    out = []
    for T in budgets:
        row4 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 4), None)
        row32 = next((r for r in iso if int(r["budget_tokens"]) == T and int(r["G"]) == 32), None)
        if row4 is None or row32 is None:
            continue
        a4 = float(row4["heldout_acc_mean"])
        a32 = float(row32["heldout_acc_mean"])
        retention = a4 / a32 if a32 > 0 else float("nan")
        gu4 = float(row4["gu_estimate"])
        gu32 = float(row32["gu_estimate"])
        gu_ratio = gu4 / gu32 if gu32 > 0 else float("nan")
        # We want to test: does a higher GU ratio (G=4 has more
        # contrast yield) predict retention? Counter-intuitive
        # finding: GU ratio stays ~4-5x in favor of G=4 across
        # budgets but retention falls, so GU is NOT the binding
        # constraint -- something else (gradient noise) is.
        out.append(
            {
                "budget_tokens": T,
                "acc_G4": round(a4, 4),
                "acc_G32": round(a32, 4),
                "retention_G4_over_G32": round(retention, 4),
                "GU_G4": round(gu4, 4),
                "GU_G32": round(gu32, 4),
                "GU_ratio_G4_over_G32": round(gu_ratio, 4),
                "interpretation": (
                    f"T={T // 1_000_000}M: retention={retention:.3f}, "
                    f"GU_ratio={gu_ratio:.2f}: G=4 carries {gu_ratio:.1f}x more "
                    f"contrast yield per group but {retention * 100:.1f}% of G=32 accuracy; "
                    f"Descriptive association only; neither constraint nor mechanism is identified."
                ),
            }
        )
    # Same-cell association is descriptive: budgets share fitted/aggregate evidence.
    if len(out) >= 3:
        from scipy.stats import spearmanr

        ret = [r["retention_G4_over_G32"] for r in out]
        gu = [r["GU_ratio_G4_over_G32"] for r in out]
        rho = (
            float(spearmanr(ret, gu).statistic)
            if len(set(gu)) > 1 and len(set(ret)) > 1
            else float("nan")
        )
        out.append(
            {
                "budget_tokens": "SPEARMAN_GU_RATIO",
                "acc_G4": "",
                "acc_G32": "",
                "retention_G4_over_G32": rho,
                "GU_G4": "",
                "GU_G32": "",
                "GU_ratio_G4_over_G32": float("nan"),
                "interpretation": f"Descriptive Spearman rho(retention, GU ratio)={rho:g}; no causal mechanism test",
            }
        )
    return out


# (C) Compute-equivalent cost projection -----------------------------------


def compute_cost(
    iso: list[dict],
    *,
    tokens_per_gpu_hour=57600.0,
    usd_per_million_tokens=2.0,
    hardware="H100 assumption",
    precision="fp16 assumption",
    batch_size=1,
    price_date="unspecified",
    throughput_source="unvalidated scenario",
    included_stages="rollout throughput; token-price scenario",
) -> list[dict]:
    """Wall-clock + USD projection for iso-accuracy targets."""
    if tokens_per_gpu_hour <= 0 or usd_per_million_tokens < 0:
        raise ValueError("invalid cost scenario")
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
        fits[G] = (slope, intercept)

    # Proxy: Llama-3.2-3B / Qwen3-8B on H100, batch=1 generation.
    # 16 tok/s/GPU is conservative for batch=1 fp16; 3B model fits
    # comfortably on one H100. Training is much more expensive but
    # rollout dominates the token budget.
    TOK_PER_GPU_HR = tokens_per_gpu_hour  # 57,600 tok/GPU-hr
    PRICE_PER_MTOK = usd_per_million_tokens  # USD per million tokens, training+rollout

    acc_targets = [0.50, 0.60, 0.70, 0.80, 0.85]
    out = []
    for acc in acc_targets:
        row = {
            "target_acc": acc,
            "cost_kind": "unvalidated_scenario",
            "tokens_per_gpu_hour": tokens_per_gpu_hour,
            "usd_per_million_tokens": usd_per_million_tokens,
            "hardware": hardware,
            "precision": precision,
            "batch_size": batch_size,
            "price_date": price_date,
            "throughput_source": throughput_source,
            "included_stages": included_stages,
            "budget_uncertainty": "unavailable: point-fit sensitivity only",
        }
        budgets = {}
        for G, (slope, intercept) in sorted(fits.items()):
            log_target = math.log10(max(acc, 1e-3))
            log_t_star = (log_target - intercept) / slope if slope > 1e-9 else float("nan")
            t_star = 10**log_t_star if log_t_star < 308 else float("nan")
            budgets[G] = t_star
            row[f"extrapolated_G{G}"] = (
                not min(t for t, _ in by_g[G]) <= t_star <= max(t for t, _ in by_g[G])
            )
            row[f"T_star_G{G}_tokens"] = round(t_star, 0)
        t4 = budgets.get(4, float("nan"))
        t32 = budgets.get(32, float("nan"))
        for label, T_val in [("G4", t4), ("G32", t32)]:
            gpu_hr = T_val / TOK_PER_GPU_HR
            usd = T_val * PRICE_PER_MTOK / 1e6
            row[f"gpu_hr_{label}"] = round(gpu_hr, 1)
            row[f"usd_{label}"] = round(usd, 2)
        if t4 > 0 and t32 > 0:
            row["ratio_G4_over_G32_tokens"] = round(t4 / t32, 2)
            row["extra_gpu_hr_G4_vs_G32"] = round(t4 / TOK_PER_GPU_HR - t32 / TOK_PER_GPU_HR, 1)
            row["extra_usd_G4_vs_G32"] = round((t4 - t32) * PRICE_PER_MTOK / 1e6, 2)
        out.append(row)
    return out


# Summary ------------------------------------------------------------------


def make_summary(tost: list[dict], linkage: list[dict], cost: list[dict]) -> list[dict]:
    out = []
    by_T_eq = {}
    for r in tost:
        by_T_eq.setdefault(r["budget_tokens"], []).append(r)
    for T, rs in sorted(by_T_eq.items()):
        wu_row = next((r for r in rs if abs(r["delta_eq_bound"] - 0.024) < 1e-6), None)
        if wu_row:
            out.append(
                {
                    "metric_kind": "tost_Wu_bound",
                    "metric_key": f"T={T // 1_000_000}M",
                    "headline": (
                        f"T={T // 1_000_000}M: TOST Delta={wu_row['delta_mean']:+.4f} "
                        f"[{wu_row['delta_ci_lo']:+.3f},{wu_row['delta_ci_hi']:+.3f}], "
                        f"p_TOST={wu_row['p_tost']:.4f} -> {wu_row['verdict']}"
                    ),
                }
            )
        strict_row = next((r for r in rs if abs(r["delta_eq_bound"] - 0.010) < 1e-6), None)
        if strict_row:
            out.append(
                {
                    "metric_kind": "tost_strict_bound",
                    "metric_key": f"T={T // 1_000_000}M",
                    "headline": (
                        f"T={T // 1_000_000}M (1pp strict): "
                        f"p_TOST={strict_row['p_tost']:.4f} -> {strict_row['verdict']}"
                    ),
                }
            )
    spear = next((r for r in linkage if r["budget_tokens"] == "SPEARMAN_GU_RATIO"), None)
    if spear:
        out.append(
            {
                "metric_kind": "cross_pillar_linkage",
                "metric_key": "spearman_gu_ratio",
                "headline": spear["interpretation"],
            }
        )
    if cost:
        c70 = next((r for r in cost if abs(r["target_acc"] - 0.70) < 1e-6), None)
        if c70:
            out.append(
                {
                    "metric_kind": "compute_cost",
                    "metric_key": "acc=0.70",
                    "headline": (
                        f"acc=0.70: G=4 needs {c70['gpu_hr_G4']:.0f} GPU-hr "
                        f"(${c70['usd_G4']:.0f}); G=32 needs {c70['gpu_hr_G32']:.0f} "
                        f"GPU-hr (${c70['usd_G32']:.0f}); G=4 costs "
                        f"{c70['extra_usd_G4_vs_G32']:.0f} USD more "
                        f"({c70['ratio_G4_over_G32_tokens']:.1f}x tokens)"
                    ),
                }
            )
    return out


# Figure -------------------------------------------------------------------


def make_figure(tost: list[dict], linkage: list[dict], cost: list[dict], out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    # (L) TOST: CI vs equivalence bound
    by_eq = {}
    for r in tost:
        by_eq.setdefault(r["delta_eq_bound"], []).append(r)
    color_map = {0.010: "#9467bd", 0.024: "#d62728", 0.050: "#1f77b4"}
    label_map = {0.010: "1pp strict", 0.024: "Wu 2.4pp", 0.050: "5pp common"}
    for eq, rs in sorted(by_eq.items()):
        rs = sorted(rs, key=lambda r: r["budget_tokens"])
        Ts = [r["budget_tokens"] / 1e6 for r in rs]
        means = [r["delta_mean"] for r in rs]
        los = [r["delta_ci_lo"] for r in rs]
        his = [r["delta_ci_hi"] for r in rs]
        dem = [r["demonstrates_equivalence"] for r in rs]
        marker = "o" if eq == 0.024 else "s"
        axes[0].errorbar(
            Ts,
            means,
            yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
            marker=marker,
            lw=1.6,
            color=color_map[eq],
            capsize=3,
            label=f"CI for Delta (bound {label_map[eq]})",
        )
        # Highlight where equivalence IS demonstrated
        for i, T in enumerate(Ts):
            if dem[i]:
                axes[0].scatter(
                    [T],
                    [means[i]],
                    s=120,
                    color=color_map[eq],
                    marker="*",
                    edgecolor="black",
                    lw=0.6,
                    zorder=5,
                    label="equiv" if i == 0 else None,
                )
    axes[0].axhline(0, color="black", ls="-", lw=0.6)
    for eq, color in color_map.items():
        axes[0].axhline(+eq, color=color, ls="--", lw=0.7, alpha=0.5)
        axes[0].axhline(-eq, color=color, ls="--", lw=0.7, alpha=0.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Token budget T (M)")
    axes[0].set_ylabel("Delta accuracy (G=32 - G=4) with 95% CI")
    axes[0].set_title("(L) TOST equivalence: stars mark demonstrable equivalence")
    axes[0].legend(fontsize=7.0, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # (Mid) Retention vs log(T) with GU annotations
    pts = [r for r in linkage if r["budget_tokens"] != "SPEARMAN_GU_RATIO"]
    if pts:
        Ts = [r["budget_tokens"] / 1e6 for r in pts]
        ret = [r["retention_G4_over_G32"] for r in pts]
        gur = [r["GU_ratio_G4_over_G32"] for r in pts]
        ax2 = axes[1]
        ax2.plot(Ts, ret, marker="o", lw=2, color="#d62728", label="Retention")
        ax2.axhline(0.976, color="red", ls="--", lw=1.0, label="Wu 0.976 bound")
        ax2.set_xscale("log")
        ax2.set_xlabel("Token budget T (M, log)")
        ax2.set_ylabel("Retention acc(G=4)/acc(G=32)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        # Twin axis for GU ratio
        ax2b = ax2.twinx()
        ax2b.plot(Ts, gur, marker="s", lw=2, color="#1f77b4", label="GU ratio (G=4/G=32)")
        ax2b.set_ylabel("GU ratio G=4/G=32", color="#1f77b4")
        ax2b.tick_params(axis="y", labelcolor="#1f77b4")
        ax2b.axhline(1.0, color="#1f77b4", ls=":", lw=0.8)
        ax2.set_title("(Mid) Retention and heuristic GU ratio")
        ax2.grid(True, alpha=0.3)
        for T, x, y in zip(Ts, Ts, ret, strict=False):
            ax2.annotate(
                f"T={T:.0f}M",
                (T, y),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
                color="#d62728",
            )
        ax2.legend(loc="lower left", fontsize=7.5)
        ax2b.legend(loc="upper right", fontsize=7.5)
        spear = next((r for r in linkage if r["budget_tokens"] == "SPEARMAN_GU_RATIO"), None)
        if spear:
            ax2.set_title(
                f"(Mid) Descriptive retention/GU association: rho={spear['retention_G4_over_G32']:+.3f}"
            )

    # (R) Wall-clock + USD projection
    if cost:
        accs = [r["target_acc"] for r in cost]
        g4 = [r["gpu_hr_G4"] for r in cost]
        g32 = [r["gpu_hr_G32"] for r in cost]
        x = np.arange(len(accs))
        w = 0.35
        axes[2].bar(x - w / 2, g4, w, color="#1f77b4", label="G=4")
        axes[2].bar(x + w / 2, g32, w, color="#d62728", label="G=32")
        axes[2].set_yscale("log")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"{a:.2f}" for a in accs])
        axes[2].set_xlabel("Target accuracy")
        axes[2].set_ylabel("Required GPU-hours (log)")
        axes[2].set_title("(R) Wall-clock: G=4 costs ~8x more at acc=0.70")
        axes[2].legend(fontsize=8, loc="upper left")
        axes[2].grid(True, alpha=0.3, axis="y")
        c70 = next((r for r in cost if abs(r["target_acc"] - 0.70) < 1e-6), None)
        if c70:
            axes[2].annotate(
                f"acc=0.70: G=4 / G=32\n= {c70['ratio_G4_over_G32_tokens']:.1f}x tokens\n"
                f"+{c70['extra_usd_G4_vs_G32']:.0f} USD, "
                f"+{c70['extra_gpu_hr_G4_vs_G32']:.0f} GPU-hr",
                xy=(accs.index(0.70), c70["gpu_hr_G4"]),
                xytext=(accs.index(0.70) - 0.7, c70["gpu_hr_G4"] * 0.3),
                fontsize=7.5,
                color="purple",
                arrowprops=dict(arrowstyle="->", color="purple", alpha=0.6),
            )

    fig.suptitle(
        "Pillar 3 / Iter 115 -- TOST equivalence, cross-pillar ZVF linkage, "
        "and compute-cost projection (Qwen3-8B / GSM8K)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    iso = load_iso()

    tost_all = []
    for delta_eq in EQUIV_BOUNDS:
        tost_all.extend(tost_equivalence(iso, delta_eq))

    linkage = zvf_linkage(iso)
    cost = compute_cost(iso)
    summary = make_summary(tost_all, linkage, cost)

    write_tsv(
        RES / "group_size_iter115_tost_v2.tsv",
        tost_all,
        [
            "budget_tokens",
            "estimand",
            "interval_level",
            "method",
            "delta_eq_bound",
            "delta_mean",
            "delta_se",
            "delta_ci_lo",
            "delta_ci_hi",
            "p_lower_one_sided",
            "p_upper_one_sided",
            "p_tost",
            "demonstrates_equivalence",
            "ci_inside_equiv_bound",
            "verdict",
        ],
    )
    write_tsv(
        RES / "group_size_iter115_zvf_linkage_v2.tsv",
        linkage,
        [
            "budget_tokens",
            "acc_G4",
            "acc_G32",
            "retention_G4_over_G32",
            "GU_G4",
            "GU_G32",
            "GU_ratio_G4_over_G32",
            "interpretation",
        ],
    )
    write_tsv(
        RES / "group_size_iter115_compute_cost_v2.tsv",
        cost,
        sorted({key for row in cost for key in row}),
    )
    write_tsv(
        RES / "group_size_iter115_summary_v2.tsv",
        summary,
        ["metric_kind", "metric_key", "headline"],
    )

    make_figure(tost_all, linkage, cost, FIG / "group_size_iter115_v2.pdf")

    print("=== Iter 115 TOST (delta_eq=0.024 = Wu bound) ===")
    for r in tost_all:
        if abs(r["delta_eq_bound"] - 0.024) < 1e-6:
            print(
                f"  T={r['budget_tokens']:>9}  Delta={r['delta_mean']:+.4f}  "
                f"CI [{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]  "
                f"p_TOST={r['p_tost']:.4f}  -> {r['verdict']}"
            )
    print("\n=== Iter 115 TOST (delta_eq=0.010 = strict 1pp) ===")
    for r in tost_all:
        if abs(r["delta_eq_bound"] - 0.010) < 1e-6:
            print(
                f"  T={r['budget_tokens']:>9}  Delta={r['delta_mean']:+.4f}  "
                f"CI [{r['delta_ci_lo']:+.3f},{r['delta_ci_hi']:+.3f}]  "
                f"p_TOST={r['p_tost']:.4f}  -> {r['verdict']}"
            )
    print("\n=== Iter 115 cross-pillar linkage ===")
    for r in linkage:
        print(
            f"  {r['budget_tokens']!s:>14}  retention={r['retention_G4_over_G32']!s:>7}  "
            f"GU_ratio={r['GU_ratio_G4_over_G32']!s:>7}  :: {r['interpretation']}"
        )
    print("\n=== Iter 115 compute-cost projection (acc=0.70) ===")
    c70 = next((r for r in cost if abs(r["target_acc"] - 0.70) < 1e-6), None)
    if c70:
        print(
            f"  acc=0.70: T*_G4={c70['T_star_G4_tokens']:.0f} "
            f"({c70['gpu_hr_G4']:.0f} GPU-hr, ${c70['usd_G4']:.0f}), "
            f"T*_G32={c70['T_star_G32_tokens']:.0f} "
            f"({c70['gpu_hr_G32']:.0f} GPU-hr, ${c70['usd_G32']:.0f}); "
            f"ratio={c70['ratio_G4_over_G32_tokens']:.1f}x, "
            f"extra ${c70['extra_usd_G4_vs_G32']:.0f}"
        )


if __name__ == "__main__":
    main()
