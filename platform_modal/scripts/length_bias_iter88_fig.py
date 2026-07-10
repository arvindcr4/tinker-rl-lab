#!/usr/bin/env python3
"""Iter 88 figure script: 4-panel (one per task x 2 metrics).

Panels:
  (a) GSM8K CoT:  beta_q vs q  with bootstrap CI by algo
  (b) arithmetic: beta_q vs q  with bootstrap CI by algo
  (c) GSM8K CoT:  per-run iqr_q vs partial_spear_rho scatter with
                  algo-averaged ellipses
  (d) arithmetic: per-run iqr_q vs partial_spear_rho scatter

Outputs: figures/length_bias_iter88.{pdf,png} and mirror to paper/figures/.
Stdlib + numpy + matplotlib.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/claude/tinker-rl-lab-minimax"
RES = os.path.join(W, "experiments", "results")
FIG = os.path.join(W, "figures")
PAPERFIG = os.path.join(W, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PAPERFIG, exist_ok=True)

QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

# ---------- load per-run ----------
def load_perrun():
    rows = []
    path = os.path.join(RES, "length_bias_iter88_perrun.tsv")
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            for k in ("n", "L_mean", "L_std", "R_mean", "R_std",
                      "partial_spear_rho",
                      "q10", "q25", "q50", "q75", "q90",
                      "iqr_q", "tail_ratio", "monotone_q"):
                row[k] = float(row[k])
            row["task"]  = row["task"]
            row["algo"]  = row["algo"]
            row["seed"]  = int(row["seed"])
            rows.append(row)
    return rows


def load_paired():
    rows = []
    path = os.path.join(RES, "length_bias_iter88_paired.tsv")
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            row["n_pairs"] = int(row["n_pairs"])
            for k in ("mean_diff", "ci_lo", "ci_hi", "p"):
                row[k] = float(row[k])
            row["task"] = row["task"]
            row["key"]  = row["key"]
            rows.append(row)
    return rows


def beta_by_task_algo(rows, task, algo):
    """Return (qs, means, cilo, cihi) for the beta_q profile."""
    sub = [r for r in rows if r["task"] == task and r["algo"] == algo]
    qs = QUANTILES.copy()
    arr = np.array([[r[f"q{int(q*100)}"] for q in QUANTILES] for r in sub])
    if arr.size == 0:
        return qs, np.full(5, np.nan), np.full(5, np.nan), np.full(5, np.nan)
    means = arr.mean(axis=0)
    # bootstrap CI on each quantile's mean
    rng = np.random.default_rng(88)
    n = arr.shape[0]
    B = 2000
    boots = np.empty((B, 5))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr[idx].mean(axis=0)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return qs, means, lo, hi


def annotate_pval(ax, x_qs, y_diffs, y_cilo, y_cihi, y_off=0.0,
                  color="#555"):
    """Place asterisks above each q where the per-task paired p < 0.05."""
    # This is computed from the paired.tsv per-key per-task
    pass


def main():
    rows = load_perrun()
    paired = load_paired()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---------------- panel (a): GSM8K CoT, beta_q vs q ----------------
    for algo, color, marker, label in (
        ("grpo",    "#3a78d4", "o",  "GRPO"),
        ("dr_grpo", "#d4574a", "s",  "Dr.GRPO"),
    ):
        qs, mu, lo, hi = beta_by_task_algo(rows, "gsm8k_cot", algo)
        ax_a.plot(qs, mu, color=color, marker=marker, lw=2.0, ms=6,
                  label=label)
        ax_a.fill_between(qs, lo, hi, color=color, alpha=0.15)
        # per-seed thin lines for context
        sub = [r for r in rows if r["task"] == "gsm8k_cot" and r["algo"] == algo]
        for r in sub:
            y = [r[f"q{int(q*100)}"] for q in QUANTILES]
            ax_a.plot(qs, y, color=color, alpha=0.18, lw=0.6)

    # annotate per-quantile p-values
    for key in ("q10", "q25", "q50", "q75", "q90"):
        for d in paired:
            if d["task"] == "gsm8k_cot" and d["key"] == key:
                q_label = int(key[1:]) / 100.0
                # position annotation at the algo-mean of one curve
                qs, mu_g, _, _ = beta_by_task_algo(rows, "gsm8k_cot", "grpo")
                i = list(QUANTILES).index(q_label)
                p = d["p"]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else
                                                ("*" if p < 0.05 else "ns"))
                y_pos = max(mu_g[i], 0.0) + 1e-4 + (0.0004 if i % 2 == 0 else 0.0008)
                ax_a.text(q_label, y_pos, sig, ha="center", va="bottom",
                          fontsize=8, color="#222")

    ax_a.axhline(0, color="#888", lw=0.6, ls="--")
    ax_a.set_xlabel("Quantile of per-step reward $R_t$")
    ax_a.set_ylabel(r"Quantile-regression slope $\beta_q$ (per-token)")
    ax_a.set_title("(a) GSM8K CoT — $\\beta_q$ vs $q$")
    ax_a.legend(loc="lower left", frameon=False, fontsize=9)
    ax_a.grid(alpha=0.25)

    # ---------------- panel (b): arithmetic, beta_q vs q ----------------
    for algo, color, marker, label in (
        ("grpo",    "#3a78d4", "o",  "GRPO"),
        ("dr_grpo", "#d4574a", "s",  "Dr.GRPO"),
    ):
        qs, mu, lo, hi = beta_by_task_algo(rows, "arithmetic_easy", algo)
        ax_b.plot(qs, mu, color=color, marker=marker, lw=2.0, ms=6,
                  label=label)
        ax_b.fill_between(qs, lo, hi, color=color, alpha=0.15)
        sub = [r for r in rows if r["task"] == "arithmetic_easy" and r["algo"] == algo]
        for r in sub:
            y = [r[f"q{int(q*100)}"] for q in QUANTILES]
            ax_b.plot(qs, y, color=color, alpha=0.10, lw=0.5)

    ax_b.axhline(0, color="#888", lw=0.6, ls="--")
    ax_b.set_xlabel("Quantile of per-step reward $R_t$")
    ax_b.set_ylabel(r"Quantile-regression slope $\beta_q$ (per-token)")
    ax_b.set_title("(b) Arithmetic — $\\beta_q$ vs $q$ (null control)")
    ax_b.legend(loc="lower left", frameon=False, fontsize=9)
    ax_b.grid(alpha=0.25)

    # ---------------- panel (c)/(d): iqr_q vs partial_spear_rho scatter ----------------
    def scatter_panel(ax, task, title):
        for algo, color, marker, label in (
            ("grpo",    "#3a78d4", "o",  "GRPO"),
            ("dr_grpo", "#d4574a", "s",  "Dr.GRPO"),
        ):
            sub = [r for r in rows if r["task"] == task and r["algo"] == algo]
            xs = np.array([r["iqr_q"] for r in sub])
            ys = np.array([r["partial_spear_rho"] for r in sub])
            ax.scatter(xs, ys, color=color, marker=marker, s=55, alpha=0.85,
                       label=label, edgecolor="white", linewidths=0.6)
            if len(xs) > 1:
                mx, my = xs.mean(), ys.mean()
                ax.scatter([mx], [my], color=color, marker="X", s=120,
                           edgecolor="white", linewidths=1.0, zorder=10)
        ax.axhline(0, color="#888", lw=0.6, ls="--")
        ax.axvline(0, color="#888", lw=0.6, ls="--")
        ax.set_xlabel(r"Heteroscedasticity proxy: $\mathrm{IQR}_Q = \beta_{0.75} - \beta_{0.25}$")
        ax.set_ylabel(r"Partial Spearman $\rho(L_t, R_t \mid t)$")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.grid(alpha=0.25)

    scatter_panel(ax_c, "gsm8k_cot",
                  "(c) GSM8K CoT — iqr vs partial-$\\rho$")
    scatter_panel(ax_d, "arithmetic_easy",
                  "(d) Arithmetic — iqr vs partial-$\\rho$ (null)")

    fig.suptitle(
        "Iter 88 — Pillar 4: Quantile-Regression Coupling Decomposition "
        "(Dr.GRPO strengthens L–R coupling at every central quantile on CoT)",
        fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_pdf = os.path.join(FIG, "length_bias_iter88.pdf")
    out_png = os.path.join(FIG, "length_bias_iter88.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")

    # mirror to paper/figures
    import shutil
    for src in (out_pdf, out_png):
        dst = os.path.join(PAPERFIG, os.path.basename(src))
        shutil.copy(src, dst)
        print(f"mirrored to {dst}")


if __name__ == "__main__":
    main()
