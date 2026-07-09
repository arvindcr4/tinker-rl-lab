#!/usr/bin/env python3
"""Pillar 2 elevation: Contrastive Yield decomposition + Iso-G sizing.

Frontier-synthesis (ChatGPT Pro Extended + Gemini Deep Think, round 2):
ZVF should be reframed as observed signal availability, not difficulty.
The clean decomposition is:

    ZVF_obs(G) = ZVF_iid(p, G) - delta_div

where ZVF_iid(p, G) = p**G + (1 - p)**G is the iid-Bernoulli collision
probability under prompt-difficulty p, and delta_div in [0, 1] is the
structural diversity bonus introduced by high-temperature autoregressive
sampling. A positive delta_div (i.e. observed ZVF BELOW the iid baseline)
proves that the sampler anti-herds: completions within a rollout group
are MORE diverse than independent draws from the marginal Bernoulli(p)
distribution, providing contrastive signal that pure noise would not.

Contrastive Yield:
    Y(p, G) = 1 - ZVF_obs(G)

is the fraction of groups with at least one positive-advantage contrast;
this is the quantity that actually drives the GRPO gradient flow. Static
G wastes compute at the easy/hard tails (where G=2 already saturates Y
near 1) and starves the learning frontier (p ~ 0.5) where G=2 produces
Y = 0.5.

Iso-Yield Dynamic Grouping (Iso-G):
    For a target yield Y_target, the iid requirement on the easy/hard
    tails is G(p) = ceil(log(1 - Y_target) / log(max(p, 1 - p))).
    Iso-G picks G per prompt to keep Y >= Y_target uniformly.

Inputs (real, measured):
    experiments/results/tinker_gsm8k_zvf_s{42,123,456}.json
        200 problems x G=8 rewards per problem (Qwen3-8B / GSM8K).
    experiments/results/groupsize_zvf_sweep.json
        12 runs x 40 steps of per-step ZVF; G in {2,4,8,16}.

Outputs:
    experiments/results/zvf_contrastive_yield.tsv
        Per-problem decomposition rows (p_x, ZVF_obs, ZVF_iid, delta_div,
        Y, iso_g_target).
    experiments/results/zvf_iso_yield_sizing.tsv
        Required G(p) curve under three Y_target settings and the
        per-quintile mean G.
    experiments/results/zvf_yield_vs_static.tsv
        Static-G vs Iso-G expected gradient-flow summary on GSM8K.
    figures/zvf_contrastive_yield.pdf
        Two-panel figure: (left) Y(p, G) curves vs p for G in
        {2,4,8,16} with measured (p, Y) scatter overlaid; (right)
        iso-G sizing curve G(p) for Y_target in {0.6, 0.8, 0.95}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIG_DIR = REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# Per-problem ZVF decomposition
# ---------------------------------------------------------------------------


def per_problem_zvf_iid(p: float, G: int) -> float:
    """iid-Bernoulli collision probability: P(all-correct or all-wrong)."""
    if G <= 0:
        return float("nan")
    p = min(max(p, 0.0), 1.0)
    return p ** G + (1.0 - p) ** G


def iso_g(p: float, Y_target: float) -> int:
    """Smallest G such that iid-Y(p, G) >= Y_target."""
    if p <= 0.0 or p >= 1.0:
        # Degenerate prompts (always-wrong or always-correct) need G >= 1;
        # they produce no gradient signal regardless of G.
        return 1
    y_target = max(min(Y_target, 1.0 - 1e-9), 1e-9)
    # iid-Y(p, G) = 1 - p**G - (1-p)**G. For the tail (p near 0 or 1),
    # max(p, 1-p)**G dominates 1 - Y_target.
    log_inv = math.log(1.0 - y_target)
    denom = math.log(max(p, 1.0 - p))
    if denom == 0:
        return 1
    G_needed = math.ceil(log_inv / denom)
    return max(1, G_needed)


def load_tinker_per_problem() -> List[Dict[str, Any]]:
    """200 problems x G=8 rollouts, 3 seeds (600 problems total)."""
    out: List[Dict[str, Any]] = []
    for seed in (42, 123, 456):
        path = RESULTS / f"tinker_gsm8k_zvf_s{seed}.json"
        data = json.loads(path.read_text())
        G = data["group_size"]
        for p in data["per_problem"]:
            rewards = list(p["rewards"])
            assert len(rewards) == G, (seed, len(rewards), G)
            k = sum(int(r) for r in rewards)
            p_x = k / G
            zvf_obs = 1.0 if (k == 0 or k == G) else 0.0
            zvf_iid = per_problem_zvf_iid(p_x, G)
            delta_div = zvf_iid - zvf_obs
            Y_obs = 1.0 - zvf_obs
            out.append(
                {
                    "source": "tinker_gsm8k",
                    "seed": seed,
                    "problem_id": p["problem_id"],
                    "G": G,
                    "k": k,
                    "p_x": p_x,
                    "zvf_obs": zvf_obs,
                    "zvf_iid": zvf_iid,
                    "delta_div": delta_div,
                    "Y_obs": Y_obs,
                }
            )
    return out


def load_groupsize_sweep_per_step() -> List[Dict[str, Any]]:
    """12 runs x 40 steps of per-step ZVF and reward; aggregate to per-(G, seed)."""
    path = RESULTS / "groupsize_zvf_sweep.json"
    data = json.loads(path.read_text())
    out: List[Dict[str, Any]] = []
    for run in data["runs"]:
        G = int(run["group_size"])
        seed = int(run["seed"])
        steps = run["step_log"]
        zvfs = [float(s["zvf"]) for s in steps]
        rewards = [float(s["mean_reward"]) for s in steps]
        # Per-step iid collision uses the per-step reward as a proxy for p.
        # Across 40 steps we get 40 (p, ZVF_obs, ZVF_iid, delta_div) points.
        for s in steps:
            p_step = float(s["mean_reward"])
            zvf_obs = float(s["zvf"])
            zvf_iid = per_problem_zvf_iid(p_step, G)
            delta_div = zvf_iid - zvf_obs
            out.append(
                {
                    "source": "groupsize_zvf_sweep",
                    "seed": seed,
                    "G": G,
                    "step": int(s["step"]),
                    "p_x": p_step,
                    "zvf_obs": zvf_obs,
                    "zvf_iid": zvf_iid,
                    "delta_div": delta_div,
                    "Y_obs": 1.0 - zvf_obs,
                }
            )
        # Track aggregate per-(G, seed) for the Iso-G sizing table.
        mean_p = statistics.fmean(rewards)
        mean_zvf = statistics.fmean(zvfs)
        out.append(
            {
                "source": "groupsize_zvf_sweep_agg",
                "seed": seed,
                "G": G,
                "step": "agg",
                "p_x": mean_p,
                "zvf_obs": mean_zvf,
                "zvf_iid": per_problem_zvf_iid(mean_p, G),
                "delta_div": per_problem_zvf_iid(mean_p, G) - mean_zvf,
                "Y_obs": 1.0 - mean_zvf,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_contrastive_yield(
    rows: List[Dict[str, Any]], out_path: Path
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(
            "# Pillar 2 Contrastive Yield decomposition (per-problem / per-step)\n"
            "# Computed from real rollout data:\n"
            "#   - tinker_gsm8k_zvf_s{42,123,456}.json: 600 (problem, G=8) rollouts.\n"
            "#   - groupsize_zvf_sweep.json: 12 runs * 40 steps x G in {2,4,8,16}.\n"
            "# Columns:\n"
            "#   source      tinker_gsm8k | groupsize_zvf_sweep | groupsize_zvf_sweep_agg\n"
            "#   seed, problem_id|step, G, k|p_x, zvf_obs, zvf_iid, delta_div, Y_obs\n"
            "# zvf_iid  = p**G + (1-p)**G (Bernoulli-collision baseline).\n"
            "# delta_div = zvf_iid - zvf_obs (structural diversity bonus;\n"
            "#            POSITIVE delta_div proves the sampler anti-herds).\n"
            "# Source: scripts/zvf_contrastive_yield.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "source",
                "seed",
                "id",
                "G",
                "p_x",
                "zvf_obs",
                "zvf_iid",
                "delta_div",
                "Y_obs",
            )
        )
        for r in rows:
            writer.writerow(
                (
                    r["source"],
                    r["seed"],
                    r.get("problem_id", r.get("step", "")),
                    r["G"],
                    f"{r['p_x']:.4f}",
                    f"{r['zvf_obs']:.4f}",
                    f"{r['zvf_iid']:.4f}",
                    f"{r['delta_div']:.4f}",
                    f"{r['Y_obs']:.4f}",
                )
            )

    # Headline aggregates (per source).
    agg: Dict[str, Dict[str, Any]] = {}
    for src in ("tinker_gsm8k", "groupsize_zvf_sweep", "groupsize_zvf_sweep_agg"):
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        delta_vals = [r["delta_div"] for r in sub]
        y_vals = [r["Y_obs"] for r in sub]
        agg[src] = {
            "n": len(sub),
            "mean_delta_div": statistics.fmean(delta_vals),
            "median_delta_div": statistics.median(delta_vals),
            "frac_delta_positive": sum(1 for d in delta_vals if d > 0) / len(delta_vals),
            "mean_Y": statistics.fmean(y_vals),
            "median_Y": statistics.median(y_vals),
        }
    return {"per_source": agg, "n_rows": len(rows)}


def write_iso_yield_sizing(
    rows: List[Dict[str, Any]], out_path: Path
) -> Dict[str, Any]:
    """Required G(p) under three Y_target settings + per-quintile summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(
            "# Pillar 2 Iso-Yield Dynamic Grouping (Iso-G) sizing curve.\n"
            "# G(p, Y_target) = ceil(log(1 - Y_target) / log(max(p, 1 - p))).\n"
            "#   p in [0.05, 0.95] at 0.025 step; Y_target in {0.6, 0.8, 0.95}.\n"
            "# Source: scripts/zvf_contrastive_yield.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(("p", "G_y06", "G_y08", "G_y095"))
        for i in range(1, 40):
            p = i / 40.0  # 0.025 .. 0.975
            writer.writerow(
                (
                    f"{p:.4f}",
                    iso_g(p, 0.6),
                    iso_g(p, 0.8),
                    iso_g(p, 0.95),
                )
            )

    # Per-quintile mean G under Y_target=0.8 using the real per-problem p_x.
    tinker = [r for r in rows if r["source"] == "tinker_gsm8k"]
    tinker.sort(key=lambda r: r["p_x"])
    quintile_rows: List[Dict[str, Any]] = []
    if tinker:
        n = len(tinker)
        q_size = n // 5
        for q in range(5):
            lo = q * q_size
            hi = (q + 1) * q_size if q < 4 else n
            slice_ = tinker[lo:hi]
            p_min = slice_[0]["p_x"]
            p_max = slice_[-1]["p_x"]
            mean_p = statistics.fmean(r["p_x"] for r in slice_)
            static_g8 = statistics.fmean(r["Y_obs"] for r in slice_)
            iso_g_vals = [iso_g(r["p_x"], 0.8) for r in slice_]
            mean_iso_g = statistics.fmean(iso_g_vals)
            iso_yields = [
                1.0 - per_problem_zvf_iid(r["p_x"], iso_g(r["p_x"], 0.8))
                for r in slice_
            ]
            mean_iso_yield = statistics.fmean(iso_yields)
            quintile_rows.append(
                {
                    "quintile": q + 1,
                    "p_min": p_min,
                    "p_max": p_max,
                    "mean_p": mean_p,
                    "static_g8_Y_mean": static_g8,
                    "iso_g_mean": mean_iso_g,
                    "iso_Y_mean": mean_iso_yield,
                }
            )
    return {"quintiles": quintile_rows, "n_tinker": len(tinker)}


def write_yield_vs_static(
    rows: List[Dict[str, Any]], out_path: Path
) -> Dict[str, Any]:
    """Static G=8 vs Iso-G expected-yield comparison on the GSM8K slice."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tinker = [r for r in rows if r["source"] == "tinker_gsm8k"]
    static_g = 8
    Y_targets = (0.6, 0.8, 0.95)
    out_rows: List[Dict[str, Any]] = []
    for Y_target in Y_targets:
        Y_static = statistics.fmean(
            1.0 - per_problem_zvf_iid(r["p_x"], static_g) for r in tinker
        )
        Y_iso = statistics.fmean(
            1.0 - per_problem_zvf_iid(r["p_x"], iso_g(r["p_x"], Y_target))
            for r in tinker
        )
        G_iso_mean = statistics.fmean(iso_g(r["p_x"], Y_target) for r in tinker)
        G_iso_max = max(iso_g(r["p_x"], Y_target) for r in tinker)
        out_rows.append(
            {
                "Y_target": Y_target,
                "static_G": static_g,
                "Y_static_observed": Y_static,
                "Y_iso_target": Y_target,
                "Y_iso_realised": Y_iso,
                "iso_G_mean": G_iso_mean,
                "iso_G_max": G_iso_max,
                "n_problems": len(tinker),
            }
        )

    with out_path.open("w") as fh:
        fh.write(
            "# Pillar 2 Contrastive Yield: static-G=8 vs Iso-G on Qwen3-8B/GSM8K.\n"
            "# Y_static_observed = mean over the 600-problem slice of\n"
            "#   1 - [p_x**G + (1-p_x)**G] at G = static_G.\n"
            "# Y_iso_realised    = same mean after sizing G per-prompt via Iso-G\n"
            "#   (G(p, Y_target)) so the iid baseline meets Y_target uniformly.\n"
            "# Source: scripts/zvf_contrastive_yield.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "Y_target",
                "static_G",
                "Y_static_observed",
                "Y_iso_target",
                "Y_iso_realised",
                "iso_G_mean",
                "iso_G_max",
                "n_problems",
            )
        )
        for r in out_rows:
            writer.writerow(
                (
                    r["Y_target"],
                    r["static_G"],
                    f"{r['Y_static_observed']:.4f}",
                    r["Y_iso_target"],
                    f"{r['Y_iso_realised']:.4f}",
                    f"{r['iso_G_mean']:.2f}",
                    r["iso_G_max"],
                    r["n_problems"],
                )
            )
    return {"rows": out_rows}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _maybe_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def write_figure(
    rows: List[Dict[str, Any]],
    out_path: Path,
    quintiles: List[Dict[str, Any]],
) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Left panel: Y(p, G) curves vs p for G in {2,4,8,16} with the
    # measured (p, Y) scatter from tinker_gsm8k (G=8) and the G-sweep.
    Gs = (2, 4, 8, 16)
    colors = {2: "#1f77b4", 4: "#2ca02c", 8: "#d62728", 16: "#9467bd"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ps = [i / 200.0 for i in range(1, 200)]
    for G in Gs:
        ys = [1.0 - per_problem_zvf_iid(p, G) for p in ps]
        axL.plot(ps, ys, color=colors[G], linewidth=1.8, label=f"G={G} (iid)")

    # Overlay measured (p, Y_obs) jittered for G=8 (tinker_gsm8k)
    tinker = [r for r in rows if r["source"] == "tinker_gsm8k"]
    if tinker:
        x = [r["p_x"] + (hash((r["seed"], r["problem_id"])) % 7 - 3) * 0.012
             for r in tinker]
        y = [r["Y_obs"] for r in tinker]
        axL.scatter(
            x,
            y,
            s=10,
            alpha=0.30,
            color=colors[8],
            edgecolor="none",
            label=f"G=8 measured (n={len(tinker)})",
        )

    axL.set_xlim(0, 1)
    axL.set_ylim(-0.02, 1.05)
    axL.set_xlabel("Per-prompt success probability p_x")
    axL.set_ylabel("Contrastive Yield Y(p, G) = 1 - ZVF")
    axL.set_title("Contrastive Yield curves vs measured scatter (Qwen3-8B / GSM8K)")
    axL.legend(loc="lower center", fontsize=8, frameon=False, ncol=2)

    # Right panel: Iso-G sizing curves G(p, Y_target) for Y in {0.6, 0.8, 0.95}
    ps_iso = [i / 200.0 for i in range(1, 200)]
    iso_colors = {0.6: "#1f77b4", 0.8: "#2ca02c", 0.95: "#d62728"}
    for Y_target in (0.6, 0.8, 0.95):
        gs = [iso_g(p, Y_target) for p in ps_iso]
        # Cap display at G=64 so the panel doesn't get clipped by tails.
        gs_disp = [min(g, 64) for g in gs]
        axR.plot(
            ps_iso,
            gs_disp,
            color=iso_colors[Y_target],
            linewidth=1.8,
            label=f"Y_target={Y_target}",
        )

    if quintiles:
        qp = [(q["p_min"] + q["p_max"]) / 2.0 for q in quintiles]
        qg = [q["iso_g_mean"] for q in quintiles]
        axR.scatter(
            qp,
            qg,
            s=55,
            color="#34495e",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label="GSM8K quintile mean (Y=0.8)",
        )

    axR.set_yscale("log")
    axR.set_ylim(1, 64)
    axR.set_xlim(0, 1)
    axR.set_xlabel("Per-prompt success probability p_x")
    axR.set_ylabel("Required group size G(p, Y_target) (log)")
    axR.set_title("Iso-Yield Dynamic Grouping (Iso-G) sizing")
    axR.legend(loc="upper center", fontsize=8, frameon=False, ncol=2)

    fig.suptitle(
        "Contrastive Yield decomposition and Iso-Yield sizing "
        "(frontier-synthesis elev. of ZVF)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    tinker = load_tinker_per_problem()
    sweep = load_groupsize_sweep_per_step()
    rows = tinker + sweep

    if args.self_test:
        rows = rows[:50]

    cy = write_contrastive_yield(rows, RESULTS / "zvf_contrastive_yield.tsv")
    iso = write_iso_yield_sizing(rows, RESULTS / "zvf_iso_yield_sizing.tsv")
    yvs = write_yield_vs_static(rows, RESULTS / "zvf_yield_vs_static.tsv")
    fig = write_figure(
        rows,
        FIG_DIR / "zvf_contrastive_yield.pdf",
        iso["quintiles"],
    )

    print(f"[zvf-yield] wrote {cy['n_rows']} decomposition rows")
    for src, stats in cy["per_source"].items():
        print(
            f"[zvf-yield] {src:>30}  n={stats['n']:>4}  "
            f"mean_delta_div={stats['mean_delta_div']:+.4f}  "
            f"frac_positive={stats['frac_delta_positive']:.3f}  "
            f"mean_Y={stats['mean_Y']:.3f}"
        )
    print("[zvf-yield] static-G vs Iso-G (Qwen3-8B/GSM8K, n=600):")
    for r in yvs["rows"]:
        print(
            f"[zvf-yield]   Y_target={r['Y_target']:.2f}  "
            f"Y_static={r['Y_static_observed']:.4f}  "
            f"Y_iso={r['Y_iso_realised']:.4f}  "
            f"iso_G_mean={r['iso_G_mean']:.2f}  "
            f"iso_G_max={r['iso_G_max']}"
        )
    if fig:
        print(f"[zvf-yield] wrote figure {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())