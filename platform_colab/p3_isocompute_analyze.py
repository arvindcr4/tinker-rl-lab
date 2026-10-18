"""Summarize the Colab iso-compute group-size sweep (p3_isocompute_groupsize.py).

Reads platform_hybrid/experiments/results/p3_isocompute/*.json and writes
p3_isocompute_summary.tsv plus a LaTeX table body on stdout.

The logged per-step "zvf_closed_form_hetero" plugs each prompt's pass rate,
estimated from that same group of G samples, into p^G + (1-p)^G; that is
>= the measured indicator by construction, so it is not used.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "platform_hybrid/experiments/results"
runs = [json.load(open(p)) for p in sorted((RES / "p3_isocompute").glob("*.json"))]


def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


rows = []
for w in (1, 0):
    for g in sorted({r["group_size"] for r in runs}):
        rs = [r for r in runs if r["group_size"] == g and int(r["whiten"]) == w]
        if not rs:
            continue
        post = np.array([r["heldout_post"] for r in rs])
        gain = np.array([r["heldout_post"] - r["heldout_pre"] for r in rs])
        zvf = np.array([r["mean_zvf"] for r in rs])
        homog = np.array(
            [
                np.mean(
                    [s["mean_reward"] ** g + (1 - s["mean_reward"]) ** g for s in r["step_log"]]
                )
                for r in rs
            ]
        )
        rows.append(
            {
                "whiten": w,
                "G": g,
                "n": len(rs),
                "prompts_per_step": rs[0]["prompts_per_step"],
                "heldout_post": post.mean(),
                "heldout_post_ci95": ci95(post),
                "heldout_gain": gain.mean(),
                "heldout_gain_ci95": ci95(gain),
                "last10": np.mean([r["last10_avg"] for r in rs]),
                "zvf": zvf.mean(),
                "zvf_homog_closed_form": homog.mean(),
                "seeds": ",".join(str(r["seed"]) for r in rs),
            }
        )

with open(RES / "p3_isocompute_summary.tsv", "w") as fh:
    fh.write("\t".join(rows[0]) + "\n")
    for r in rows:
        fh.write(
            "\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n"
        )

for r in rows:
    print(
        f"{'yes' if r['whiten'] else 'no'} & {r['G']} & {r['prompts_per_step']} & {r['n']} & "
        f"${r['heldout_post']:.3f} \\pm {r['heldout_post_ci95']:.3f}$ & ${r['heldout_gain']:+.3f}$ & "
        f"{r['last10']:.3f} & {r['zvf']:.3f} & {r['zvf_homog_closed_form']:.3f} \\\\"
    )

# Pairwise held-out contrasts among whitened arms (Welch; seeds are the unit).
w1 = {
    r["G"]: np.array(
        [x["heldout_post"] for x in runs if x["group_size"] == r["G"] and int(x["whiten"]) == 1]
    )
    for r in rows
    if r["whiten"] == 1
}
gs = sorted(w1)
best = max(gs, key=lambda g: w1[g].mean())
print(f"\nbest whitened G by mean held-out: {best}")
for g in gs:
    if g != best and w1[g].size > 1:
        t = stats.ttest_ind(w1[best], w1[g], equal_var=False)
        print(
            f"G={best} vs G={g}: diff={w1[best].mean() - w1[g].mean():+.3f} Welch p={t.pvalue:.3g}"
        )
# Whitening effect at each G (unpaired; seeds 42/123/456 shared).
for g in gs:
    a = np.array(
        [
            x["heldout_post"]
            for x in runs
            if x["group_size"] == g and int(x["whiten"]) == 1 and x["seed"] in (42, 123, 456)
        ]
    )
    b = np.array(
        [x["heldout_post"] for x in runs if x["group_size"] == g and int(x["whiten"]) == 0]
    )
    if a.size > 1 and b.size > 1:
        t = stats.ttest_rel(a, b) if a.size == b.size else stats.ttest_ind(a, b, equal_var=False)
        print(
            f"G={g} whiten-unwhiten (seeds 42/123/456): {a.mean() - b.mean():+.3f} p={t.pvalue:.3g}"
        )


# Figure: held-out accuracy and ZVF vs G at equal rollouts per step.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.0))
for w, color, label in (
    (1, "#1f77b4", "GRPO (whitened)"),
    (0, "#ff7f0e", r"unwhitened ($r-\bar r$)"),
):
    rs = [r for r in rows if r["whiten"] == w]
    gx = np.array([r["G"] for r in rs])
    off = 1.04 if w == 0 else 1.0
    for g in gx:
        pts = [x["heldout_post"] for x in runs if x["group_size"] == g and int(x["whiten"]) == w]
        ax_a.scatter([g * off] * len(pts), pts, color=color, alpha=0.3, s=14)
    ax_a.errorbar(
        gx * off,
        [r["heldout_post"] for r in rs],
        yerr=[r["heldout_post_ci95"] for r in rs],
        fmt="o-",
        color=color,
        capsize=4,
        lw=2,
        label=label,
    )
    ax_b.plot(
        gx * off, [r["zvf"] for r in rs], "o-", color=color, lw=2, label=f"measured ZVF, {label}"
    )
    if w == 1:
        ax_b.plot(
            gx,
            [r["zvf_homog_closed_form"] for r in rs],
            "k:",
            lw=1.2,
            label=r"closed form at batch-mean $p_t$",
        )
pre = np.mean([r["heldout_pre"] for r in runs])
ax_a.axhline(pre, color="gray", ls="--", lw=1, label=f"base model ({pre:.2f})")
for ax, title, ylabel in (
    (ax_a, "(A) Held-out accuracy after 60 steps", "Held-out accuracy (400 problems)"),
    (ax_b, "(B) Zero-variance fraction", "Mean ZVF over training"),
):
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="Group size G (128 rollouts per step for every G)", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(
    Path(__file__).resolve().parents[1]
    / "platform_hybrid/paper/figures/group_size_isocompute_mul2.png",
    dpi=200,
)
print("wrote group_size_isocompute_mul2.png")
