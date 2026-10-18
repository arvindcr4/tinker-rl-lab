"""Variance-mitigation comparison on the real e1 confirmatory campaign (P2).

Input: platform_hybrid/experiments/results/e1_confirmatory_wandb_export.tsv
(W&B arvindcr4-pes-university/tinker-rl-lab, e1-confirmatory-*; Qwen3-8B, TRL
GRPOTrainer, num_generations=4, max_steps=30, lr=1e-6, beta=0, T=1.0; held-out
GSM8K n=500 per run). Every arm uses the same 8 seeds, so contrasts against
grpo are paired by seed.

Output: e1_variance_mitigation_summary.tsv with one row per arm: held-out mean,
sd, t-based 95% CI, paired difference vs grpo (paired t, df=7) and a
seed-level paired TOST at eps=0.02, plus mean ZVF, last-10 reward, rollouts
and generated tokens.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments/results"
SRC = RES / "e1_confirmatory_wandb_export.tsv"
OUT = RES / "e1_variance_mitigation_summary.tsv"
FIG = RES.parents[1] / "paper/figures/e1_variance_mitigation.png"
EPS = 0.02
ARMS = ("grpo", "aero", "dapo", "drgrpo", "gspo")


def load():
    rows = [
        r for r in csv.DictReader((l for l in open(SRC) if not l.startswith("#")), delimiter="\t")
    ]
    by = {a: {int(r["seed"]): r for r in rows if r["arm"] == a} for a in ARMS}
    seeds = sorted(by["grpo"])
    assert all(sorted(by[a]) == seeds for a in ARMS), "arms must share seeds"
    return by, seeds


def col(by, arm, seeds, key):
    return np.array([float(by[arm][s][key]) for s in seeds])


def main():
    by, seeds = load()
    n = len(seeds)
    tcrit = stats.t.ppf(0.975, n - 1)
    base = col(by, "grpo", seeds, "heldout_score")
    out = []
    for arm in ARMS:
        h = col(by, arm, seeds, "heldout_score")
        row = {
            "arm": arm,
            "n_seeds": n,
            "heldout_mean": h.mean(),
            "heldout_sd": h.std(ddof=1),
            "heldout_ci_low": h.mean() - tcrit * h.std(ddof=1) / np.sqrt(n),
            "heldout_ci_high": h.mean() + tcrit * h.std(ddof=1) / np.sqrt(n),
            "mean_zvf": col(by, arm, seeds, "mean_zvf").mean(),
            "last10_reward": col(by, arm, seeds, "last10_reward").mean(),
            "rollouts": col(by, arm, seeds, "rollouts").mean(),
            "num_tokens": col(by, arm, seeds, "num_tokens").mean(),
        }
        if arm != "grpo":
            d = h - base
            se = d.std(ddof=1) / np.sqrt(n)
            row.update(
                {
                    "diff_vs_grpo": d.mean(),
                    "diff_ci_low": d.mean() - tcrit * se,
                    "diff_ci_high": d.mean() + tcrit * se,
                    "paired_t_p": stats.ttest_rel(h, base).pvalue,
                    "tost_p_eps0.02": max(
                        stats.t.sf((d.mean() + EPS) / se, n - 1),
                        stats.t.cdf((d.mean() - EPS) / se, n - 1),
                    ),
                }
            )
        out.append(row)
    keys = list(out[1])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", restval="")
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {OUT.name}")
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for i, arm in enumerate(ARMS):
        h = col(by, arm, seeds, "heldout_score")
        ax_a.scatter(np.full(n, i) + np.linspace(-0.15, 0.15, n), h, s=14, alpha=0.6)
        r = out[i]
        ax_a.errorbar(
            i, r["heldout_mean"], yerr=r["heldout_mean"] - r["heldout_ci_low"], fmt="ko", capsize=4
        )
        z = col(by, arm, seeds, "mean_zvf")
        ax_b.scatter(np.full(n, i) + np.linspace(-0.15, 0.15, n), z, s=14, alpha=0.6)
        ax_b.plot([i - 0.25, i + 0.25], [z.mean()] * 2, "k-")
    for ax, lab, t in (
        (ax_a, "Held-out GSM8K accuracy (n=500)", "(A) Held-out accuracy, 8 seeds"),
        (ax_b, "Mean ZVF over training", "(B) Mean zero-variance fraction"),
    ):
        ax.set_xticks(range(len(ARMS)), ARMS)
        ax.set_ylabel(lab)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    print(f"wrote {FIG.name}")


if __name__ == "__main__":
    main()
