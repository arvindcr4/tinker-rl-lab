#!/usr/bin/env python3
"""Iter 112 figure -- Severship-reward coupling test.

3-panel figure:
  (A) Scatter: per-(task, window, seed) (-Δ_bwd) on x-axis, Δ_R on y-axis;
      2 tasks side-by-side with per-window median markers.
  (B) Per-task pooled Spearman rho bar with bootstrap CI; significance
      region shaded at |rho| >= 0.5.
  (C) Per-window rank correlation across (seed, window) pairs; the 4
      windows per task are plotted side-by-side to show the temporal
      evolution of the severship-reward coupling.
  (D) Cumulative-window regression rho for k_end in {1,2,3,4};
      tests whether severship compounds into reward as windows accumulate.

Reads : platform_hybrid/experiments/results/length_bias_iter112_per_window.tsv
        platform_hybrid/experiments/results/length_bias_iter112_pooled.tsv
        platform_hybrid/experiments/results/length_bias_iter112_cumulative.tsv
        platform_hybrid/experiments/results/length_bias_iter112_rho_bootstrap.tsv
        platform_hybrid/experiments/results/length_bias_iter112_permutation_null.tsv
        platform_hybrid/experiments/results/length_bias_iter108_perrun_progress.tsv
Writes: figures/length_bias_iter112_sever_reward.pdf and .png;
        mirrored to paper/figures/.
"""
from __future__ import annotations
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIGS = ROOT / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

TASKS = ["arithmetic_easy", "gsm8k_cot"]
N_W = 4


def load_tsv(path: Path) -> list[dict]:
    with open(path) as fh:
        hdr = fh.readline().rstrip().split("\t")
        rows = [dict(zip(hdr, line.rstrip().split("\t"))) for line in fh]
    return rows


def to_f(rows, *keys):
    out = []
    for r in rows:
        rr = dict(r)
        for k in keys:
            if k in rr and rr[k] != "":
                try:
                    rr[k] = float(rr[k])
                except ValueError:
                    pass
        out.append(rr)
    return out


def build_long_table() -> list[dict]:
    """Rebuild the per-(task, window, seed) (severship, R_delta) from sources."""
    perrun = to_f(load_tsv(RES / "length_bias_iter108_perrun_progress.tsv"),
                  "phi_L", "phi_R", "bwd", "fwd", "bwd_signed", "fwd_signed")
    # iter108 perrun gives (task, algo, seed, window, bwd) per run
    # Need to pair Dr.GR - GR per seed per window
    by_key = {(r["task"], r["algo"], int(r["seed"]), int(r["window"])): float(r["bwd"])
              for r in perrun}

    # also need per-window ΔR from windowed step_log
    import json
    def load_sl(p, task_label):
        with open(p) as fh:
            d = json.load(fh)
        runs = []
        for r in d["runs"]:
            sl = r.get("step_log") or []
            if len(sl) < 5:
                continue
            R = np.array([float(s["mean_reward"]) for s in sl], dtype=np.float64)
            n = len(R)
            edges = [int(np.floor(n * w / N_W)) for w in range(N_W + 1)]
            for i in range(1, N_W + 1):
                edges[i] = max(edges[i], edges[i - 1] + 4)
                edges[i] = min(edges[i], n)
            R_w = [float(R[edges[w]:edges[w + 1]].mean()) for w in range(N_W)]
            runs.append({"task": task_label, "algo": r["algo"],
                         "seed": int(r["seed"]), "R_w": R_w})
        return runs

    runs = load_sl(RES.parent / "results" / "drgrpo_vs_grpo.json", "arithmetic_easy") \
         + load_sl(RES.parent / "results" / "drgrpo_gsm8k_cot_full.json", "gsm8k_cot")

    # pair by seed
    by_seed: dict = {}
    for r in runs:
        by_seed.setdefault((r["task"], r["algo"]), {})[r["seed"]] = r
    out = []
    for task in TASKS:
        sg = by_seed.get((task, "grpo"), {})
        sd = by_seed.get((task, "dr_grpo"), {})
        for seed in sorted(set(sg) & set(sd)):
            for w in range(N_W):
                bwd_g = by_key.get((task, "grpo", seed, w), 0.0)
                bwd_d = by_key.get((task, "dr_grpo", seed, w), 0.0)
                severship = -(bwd_d - bwd_g)  # larger = more severship
                dR = sd[seed]["R_w"][w] - sg[seed]["R_w"][w]
                out.append({"task": task, "window": w, "seed": seed,
                            "severship": severship, "delta_R": dR,
                            "bwd_g": bwd_g, "bwd_d": bwd_d})
    return out


def main() -> int:
    pooled = to_f(load_tsv(RES / "length_bias_iter112_pooled.tsv"),
                  "spearman_rho_sever_vs_reward", "spearman_p_param",
                  "p_perm_two_sided", "null_mean", "null_std",
                  "null_q025", "null_q500", "null_q975")
    perm = to_f(load_tsv(RES / "length_bias_iter112_permutation_null.tsv"),
                "obs_rho", "abs_obs", "p_perm",
                "null_mean", "null_std",
                "null_q025", "null_q500", "null_q975")
    perwin = to_f(load_tsv(RES / "length_bias_iter112_per_window.tsv"),
                  "mean_sever", "mean_dR", "rho_window", "spearman_p_param")
    cum = to_f(load_tsv(RES / "length_bias_iter112_cumulative.tsv"),
               "rho_cum_sever_vs_cum_R", "spearman_p_param",
               "mean_sever", "mean_cum_dR")
    boot = to_f(load_tsv(RES / "length_bias_iter112_rho_bootstrap.tsv"),
                "obs_rho", "ci_lo", "ci_hi")

    long_tab = build_long_table()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ---------------- (A) scatter ----------------
    ax = axes[0, 0]
    cmap = {"arithmetic_easy": "#117733", "gsm8k_cot": "#882211"}
    for ti, task in enumerate(TASKS):
        pts = [r for r in long_tab if r["task"] == task]
        for r in pts:
            x = r["severship"]; y = r["delta_R"]
            ax.scatter(x, y, color=cmap[task], alpha=0.65, s=42,
                       edgecolor="black", linewidth=0.4,
                       marker=("o" if int(r["window"]) % 2 == 0 else "s"))
        # median per window
        for w in range(N_W):
            wpts = [r for r in pts if int(r["window"]) == w]
            if wpts:
                xs = [p["severship"] for p in wpts]
                ys = [p["delta_R"] for p in wpts]
                ax.scatter(float(np.mean(xs)), float(np.mean(ys)),
                           color="white", edgecolor=cmap[task], s=120,
                           linewidth=2.0, zorder=5)
                ax.text(float(np.mean(xs)), float(np.mean(ys)), str(w),
                        ha="center", va="center", fontsize=8,
                        color=cmap[task], fontweight="bold")
    # annotate pooled Spearman per task
    ymax = ax.get_ylim()[1]
    for ti, task in enumerate(TASKS):
        r = next(p for p in pooled if p["task"] == task)
        ax.text(0.02, 0.96 - ti * 0.07,
                f"{task}: ρ={r['spearman_rho_sever_vs_reward']:+.2f} "
                f"(p={r['spearman_p_param']:.2f}, p_perm={r['p_perm_two_sided']:.2f})",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=cmap[task],
                          edgecolor="black", alpha=0.18))
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(r"$-\Delta_{bwd}\,|CCF(e_L,e_R;k<0)|$  (Dr.GRPO - GRPO, "
                  r"severship intensity)")
    ax.set_ylabel(r"$\Delta_R\,(\overline{R}_w)$  (Dr.GRPO - GRPO, mean window reward)")
    ax.set_title("(A) severship vs reward improvement per (window, seed)",
                 fontsize=10.5, loc="left")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="None", color=cmap[t],
                          markersize=8, label=t) for t in TASKS]
    handles += [plt.Line2D([0], [0], marker="o", linestyle="None",
                            color="grey", markersize=8, label="even window"),
                plt.Line2D([0], [0], marker="s", linestyle="None",
                            color="grey", markersize=8, label="odd window")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # ---------------- (B) per-task pooled Spearman with bootstrap CI ----------------
    ax = axes[0, 1]
    width = 0.35
    xs = np.arange(len(TASKS))
    obs = [next(p for p in pooled if p["task"] == t)["spearman_rho_sever_vs_reward"]
           for t in TASKS]
    ci_lo = [next(b for b in boot if b["task"] == t)["ci_lo"] for t in TASKS]
    ci_hi = [next(b for b in boot if b["task"] == t)["ci_hi"] for t in TASKS]
    p_perm = [next(p for p in pooled if p["task"] == t)["p_perm_two_sided"]
              for t in TASKS]
    ax.bar(xs, obs, width=width * 1.4, color=[cmap[t] for t in TASKS], alpha=0.7)
    ax.errorbar(xs, obs,
                 yerr=[[o - l for o, l in zip(obs, ci_lo)],
                       [h - o for o, h in zip(obs, ci_hi)]],
                 fmt="none", ecolor="black", elinewidth=1, capsize=4)
    for x, o, pp in zip(xs, obs, p_perm):
        ax.text(x, o + (0.04 if o >= 0 else -0.06),
                 f"ρ={o:+.2f}\np_perm={pp:.2f}",
                 ha="center", va=("bottom" if o >= 0 else "top"),
                 fontsize=9)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axhspan(0.5, 1.05, alpha=0.08, color="green",
                label="|ρ|≥0.5 strong")
    ax.axhspan(-1.05, -0.5, alpha=0.08, color="green")
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("_", r"\_") for t in TASKS])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r"Spearman($\rho$,  $severship$ vs $\Delta_R$)  pooled (window, seed)")
    ax.set_title("(B) pooled Spearman with bootstrap 95% CI",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # ---------------- (C) per-window Spearman ----------------
    ax = axes[1, 0]
    width = 0.18
    for ti, task in enumerate(TASKS):
        base = ti * (N_W + 1.2)
        for w in range(N_W):
            r = next((r for r in perwin
                       if r["task"] == task and int(r["window"]) == w),
                      None)
            if r is None:
                continue
            xs_w = base + (w - (N_W - 1) / 2.0) * width * 1.05
            ax.bar(xs_w, r["rho_window"], width=width * 0.95,
                    color=cmap[task], alpha=0.7)
            ax.text(xs_w, r["rho_window"] + (0.05 if r["rho_window"] >= 0 else -0.07),
                     f"{r['rho_window']:+.2f}",
                     ha="center", va=("bottom" if r["rho_window"] >= 0 else "top"),
                     fontsize=8)
        ax.text(base, 1.10, task.replace("_", r"\_"),
                 ha="center", va="bottom", fontsize=10,
                 transform=ax.get_xaxis_transform())
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(np.arange(N_W))
    ax.set_xticklabels([f"w{w}" for w in range(N_W)])
    ax.set_xlabel("training-progress window")
    ax.set_ylabel(r"per-window Spearman(severship, $\Delta_R$)")
    ax.set_title("(C) per-window Spearman  (Dr.GRPO - GRPO, n=3 or 5 seeds)",
                 fontsize=10.5, loc="left")
    ax.set_ylim(-1.15, 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[t], alpha=0.7)
                for t in TASKS]
    ax.legend(handles=handles, labels=[t for t in TASKS], frameon=False,
               fontsize=8, loc="upper right")

    # ---------------- (D) cumulative-window Spearman ----------------
    ax = axes[1, 1]
    for ti, task in enumerate(TASKS):
        rows = sorted([c for c in cum if c["task"] == task],
                       key=lambda r: int(r["k_end"]))
        if not rows:
            continue
        ks = [int(r["k_end"]) for r in rows]
        rhos = [r["rho_cum_sever_vs_cum_R"] for r in rows]
        ax.plot(ks, rhos, marker="o", color=cmap[task], label=task,
                 alpha=0.85, linewidth=1.6)
        for k, r in zip(ks, rhos):
            ax.text(k, r + (0.06 if r >= 0 else -0.10), f"{r:+.2f}",
                     ha="center", va=("bottom" if r >= 0 else "top"),
                     fontsize=9, color=cmap[task])
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(np.arange(1, N_W + 1))
    ax.set_xticklabels([f"k={k}" for k in range(1, N_W + 1)])
    ax.set_xlabel("cumulative-window cutoff  k_end")
    ax.set_ylabel(r"$\rho$(cumulative severship, cumulative $\Delta_R$)")
    ax.set_title("(D) cumulative-window regression  (does severship compound?)",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    out = FIGS / "length_bias_iter112_sever_reward.pdf"
    out_png = FIGS / "length_bias_iter112_sever_reward.png"
    fig.savefig(out)
    fig.savefig(out_png, dpi=130)
    print(f"[iter112 fig] wrote {out}")
    for ext in ("pdf", "png"):
        shutil.copy(str(out).replace(".pdf", f".{ext}"),
                    str(PAPER_FIGS / out.name).replace(".pdf", f".{ext}"))
    print(f"[iter112 fig] mirrored -> paper/figures/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
