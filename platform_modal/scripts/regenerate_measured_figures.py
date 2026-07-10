#!/usr/bin/env python3
"""Regenerate paper figures from MEASURED experiment artifacts only.

Sources (all measured this session):
  modal_results_all.json                         -> comparison_bars
  platform_hybrid/experiments/results/groupsize_zvf_sweep.json   -> group_size_ablation
  platform_hybrid/experiments/results/samestack_ppo_grpo.json    -> ppo_vs_grpo (same-stack)
  platform_hybrid/experiments/results/tinker_gsm8k_zvf_s*.json   -> zvf_correlation (frontier)
  platform_hybrid/experiments/results/groupsize_zvf_sweep.json   -> zvf_correlation (small-scale ZVF vs G)

Writes 300-dpi PNG+PDF into paper/figures/v2/, overwriting the stale versions.
Only figures with clean measured backing are regenerated; others are left as-is.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spilnes.top": False} if False else {"font.size": 11})
OK = "#0072B2"   # blue
OK2 = "#D55E00"  # vermillion
OK3 = "#009E73"  # green
GREY = "#999999"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def fig_comparison_bars():
    d = json.load(open(ROOT / "modal_results_all.json"))
    order = [("trl_grpo_math", "TRL\n(GRPO)"), ("sb3_ppo_math", "SB3\n(PPO)"),
             ("cleanrl_ppo_math", "CleanRL\n(PPO)"), ("tianshou_ppo_math", "Tianshou\n(PPO)")]
    labels, means, ses = [], [], []
    for key, lab in order:
        accs = [r["final_accuracy"] for r in d.get(key, []) if "final_accuracy" in r]
        if not accs:
            continue
        labels.append(lab)
        means.append(np.mean(accs))
        ses.append(np.std(accs) / np.sqrt(len(accs)))
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [OK] + [OK2] * (len(labels) - 1)
    bars = ax.bar(labels, means, yerr=ses, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Final accuracy (arithmetic)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-library comparison (5 seeds, measured)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "comparison_bars")


def fig_group_size():
    d = json.load(open(ROOT / "platform_hybrid/experiments/results/groupsize_zvf_sweep.json"))
    s = d["summary"]
    Gs = sorted(int(k) for k in s)
    acc = [s[str(g)]["heldout_acc_mean"] for g in Gs]
    accse = [s[str(g)]["heldout_acc_se"] for g in Gs]
    zvf = [s[str(g)]["mean_zvf"] for g in Gs]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(Gs))
    h_acc = ax.errorbar(x, acc, yerr=accse, marker="o", color=OK, lw=2, capsize=4)
    best = int(np.argmax(acc))
    ax.scatter([x[best]], [acc[best]], s=160, facecolors="none", edgecolors=OK2, lw=2, zorder=5)
    ax.annotate(f"peak G={Gs[best]}", (x[best], acc[best]), textcoords="offset points",
                xytext=(0, 12), ha="center", color=OK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G={g}" for g in Gs])
    ax.set_ylabel("Held-out accuracy", color=OK)
    ax.set_ylim(0.95, 1.0)
    ax2 = ax.twinx()
    (h_zvf,) = ax2.plot(x, zvf, marker="s", color=GREY, lw=2, ls="--")
    ax2.set_ylabel("Mean ZVF (fraction)", color=GREY)
    ax2.set_ylim(0, 1.0)
    ax.set_title("Group-size sweep (Qwen2.5-0.5B/arith., 3 seeds, measured)")
    ax.legend([h_acc, h_zvf], ["Held-out accuracy", "Mean ZVF"], loc="center right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "group_size_ablation")


def fig_ppo_vs_grpo():
    d = json.load(open(ROOT / "platform_hybrid/experiments/results/samestack_ppo_grpo.json"))
    runs = d["runs"]
    def traj(algo):
        rs = [r for r in runs if r["algo"] == algo]
        M = np.array([[s["mean_reward"] for s in r["step_log"]] for r in rs])
        return M.mean(0), M.std(0) / np.sqrt(len(rs))
    gm, gse = traj("grpo")
    pm, pse = traj("ppo")
    x = np.arange(len(gm))
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, se, c, lab in [(gm, gse, OK, "GRPO (group baseline)"), (pm, pse, OK2, "PPO (value baseline)")]:
        ax.plot(x, m, color=c, lw=2, label=lab)
        ax.fill_between(x, m - se, m + se, color=c, alpha=0.2)
    sm = d["summary"]
    pr = d.get("paired_grpo_vs_ppo", {})
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward (correctness)")
    ax.set_title("Same-stack PPO vs GRPO (Qwen2.5-0.5B, 5 seeds, measured)")
    txt = (f"held-out: GRPO {sm['grpo']['heldout_mean']:.3f} vs PPO {sm['ppo']['heldout_mean']:.3f}\n"
           f"paired diff {pr.get('mean_diff_grpo_minus_ppo', float('nan')):+.3f}, "
           f"p={pr.get('p_two_sided', float('nan')):.2f} (n.s.)")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "ppo_vs_grpo")


def fig_zvf_correlation():
    # frontier per-problem observed vs theoretical ZVF
    files = [f for f in sorted(glob.glob(str(ROOT / "platform_hybrid/experiments/results/tinker_gsm8k_zvf_s*.json")))
             if re.search(r"_s\d+\.json$", f)]
    G = 8
    obs, th = [], []
    for f in files:
        for pp in json.load(open(f))["per_problem"]:
            p = pp["mean_reward"]
            obs.append(pp["zvf"]); th.append(p ** G + (1 - p) ** G)
    obs = np.array(obs); th = np.array(th)
    r = np.corrcoef(obs, th)[0, 1] if len(obs) else float("nan")

    # small-scale mean ZVF vs G with theory at mean reward
    gd = json.load(open(ROOT / "platform_hybrid/experiments/results/groupsize_zvf_sweep.json"))["summary"]
    Gs = sorted(int(k) for k in gd)
    zvf_g = [gd[str(g)]["mean_zvf"] for g in Gs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    # jittered scatter (ZVF is 0/1 per problem) -> show as 2D density via mean theory bins
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(th, bins) - 1, 0, len(bins) - 2)
    bx = 0.5 * (bins[:-1] + bins[1:])
    by = [obs[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(bx))]
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="identity")
    ax.scatter(th, obs + np.random.default_rng(0).normal(0, 0.015, len(obs)), s=6, alpha=0.15, color=OK)
    ax.plot(bx, by, marker="o", color=OK2, lw=2, label="binned mean")
    ax.set_xlabel(r"theoretical $\mathrm{ZVF}=p^G+(1-p)^G$")
    ax.set_ylabel("observed ZVF (per problem)")
    ax.set_title(f"Frontier Qwen3-8B/GSM8K (n={len(obs)})\nobs-vs-theory r={r:.2f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Gs, zvf_g, marker="s", color=OK, lw=2, label="measured mean ZVF")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Gs); ax.set_xticklabels([str(g) for g in Gs])
    ax.set_xlabel("group size G")
    ax.set_ylabel("mean ZVF")
    ax.set_title("ZVF decreases with G\n(Qwen2.5-0.5B/arith., measured)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "zvf_correlation")


if __name__ == "__main__":
    print("Regenerating measured figures into", OUT)
    fig_comparison_bars()
    fig_group_size()
    fig_ppo_vs_grpo()
    fig_zvf_correlation()
    print("done.")
