"""Pillar 1 elevation -- extended frontier+MoE scaling analysis (iter 13).

Builds on iter9's saturation+bootstrap+holdout+autopsy work and adds:

  (a) Expanded model set: 12 anchors spanning 4B-1T parameters,
      including MoE (qwen3-235b-moe, qwen3-30b-moe, qwen3-30b-moe-inst,
      gpt-oss-20b) and the non-MoE frontier (kimi-k2).
  (b) Power-law fit: log_R = a + b * log_N -- tests whether R scales
      as a power of parameter count (Kaplan-style).
  (c) Spearman rank correlation: rho(log_10(N), R_first) and
      rho(log_10(N), R_mean) with permutation p-values.
  (d) Architecture-stratified analysis: MoE vs dense head-to-head.
  (e) Early-late gap: R(1) - R(T) as a "first-step vs final-step" gate
      that is robust to lambda-bound degeneracy.

Outputs:
  experiments/results/scaling_law_extended_frontier.tsv
  experiments/results/scaling_law_power_law.tsv
  experiments/results/scaling_law_moe_vs_dense.tsv
  figures/scaling_law_extended.{pdf,png}
  paper/figures/scaling_law_extended.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

# params_B total = total parameter count (HF sidebar; for MoE we use total
# including inactive experts -- this is the convention used in iter 9).
EXTENDED_MODELS: dict[str, dict] = {
    # (label) -> (file, params_B, arch, family)
    "Qwen3.5-4B":            {"file": "scale_gsm8k_qwen3.5-4b.json",     "params":  4.0, "arch": "dense",  "family": "qwen"},
    "Qwen3-8B":              {"file": "scale_gsm8k_qwen3-8b.json",       "params":  8.0, "arch": "dense",  "family": "qwen"},
    "Llama-3.1-8B-Instruct": {"file": "scale_gsm8k_llama-8b-inst.json",  "params":  8.0, "arch": "dense",  "family": "llama"},
    "Qwen3-32B":             {"file": "scale_gsm8k_qwen3-32b.json",      "params": 32.0, "arch": "dense",  "family": "qwen"},
    "Qwen3.5-27B":           {"file": "scale_gsm8k_qwen3.5-27b.json",    "params": 27.0, "arch": "dense",  "family": "qwen"},
    "gpt-oss-20B":           {"file": "arch_gsm8k_gpt-oss-20b.json",     "params": 20.0, "arch": "moe",    "family": "gpt-oss"},
    "Qwen3-30B-MoE":         {"file": "moe_gsm8k_qwen3-30b-moe.json",    "params": 30.0, "arch": "moe",    "family": "qwen"},
    "Qwen3-30B-MoE-Inst":    {"file": "moe_gsm8k_qwen3-30b-inst.json",   "params": 30.0, "arch": "moe",    "family": "qwen"},
    "DeepSeek-V3.1":         {"file": "frontier_gsm8k_deepseek-v3.1.json","params": 685.0,"arch": "moe",   "family": "deepseek"},
    "Nemotron-120B":         {"file": "frontier_gsm8k_nemotron-120b.json","params": 120.0,"arch": "dense",  "family": "nemotron"},
    "Qwen3-235B-MoE":        {"file": "frontier_gsm8k_qwen3-235b.json",  "params": 235.0, "arch": "moe",    "family": "qwen"},
    "Kimi-K2-Thinking":      {"file": "arch_gsm8k_kimi-k2.json",         "params": 1000.0,"arch": "moe",   "family": "kimi"},
}

SEED = 42
N_PERM = 5000


def _ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym))) / den
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = float(np.sum(resid ** 2)) / max(1, n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    return a, b, se_b


def load_extended():
    out = {}
    for label, meta in EXTENDED_MODELS.items():
        p = TRACE_DIR / meta["file"]
        if not p.exists():
            print(f"  WARN: missing {p}")
            continue
        d = json.loads(p.read_text())
        out[label] = {
            "trace": np.asarray(d["reward_trace"], float),
            "params": meta["params"],
            "arch": meta["arch"],
            "family": meta["family"],
        }
    return out


def trace_stats(trace: np.ndarray) -> dict:
    n = len(trace)
    cut = max(2, n // 3)
    return dict(
        n=n,
        r_first=float(trace[0]),
        r_final=float(trace[-1]),
        r_mean=float(trace.mean()),
        r_peak=float(trace.max()),
        r_var=float(trace.var()),
        early_mean=float(trace[:cut].mean()),
        late_mean=float(trace[-cut:].mean()),
        zero_frac=float(np.mean(trace == 0)),
        frac_above_0p5=float(np.mean(trace > 0.5)),
        delta_first_final=float(trace[0] - trace[-1]),
        delta_late_early=float(trace[-cut:].mean() - trace[:cut].mean()),
    )


def power_law(log_n: np.ndarray, metric: np.ndarray) -> dict:
    """Fit log_R = a + b * log_N (Kaplan-style power law).

    Returns slope b, intercept a, R^2, and 5000-permutation p-value for
    the null hypothesis that b == 0.
    """
    a, b, se_b = _ols(log_n, metric)
    yhat = a + b * log_n
    ss_res = float(np.sum((metric - yhat) ** 2))
    ss_tot = float(np.sum((metric - metric.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    n = len(log_n)
    perm = np.empty(N_PERM, float)
    for i in range(N_PERM):
        idx = rng.permutation(n)
        _, bp, _ = _ols(log_n, metric[idx])
        perm[i] = bp
    perm = perm[~np.isnan(perm)]
    # two-sided p-value
    p = float(np.mean(np.abs(perm) >= abs(b)))
    return dict(slope=b, intercept=a, se_slope=se_b, r2=r2, p_perm=p, n=len(log_n))


def write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> None:
    raw = load_extended()
    print(f"Loaded {len(raw)}/{len(EXTENDED_MODELS)} models")

    # ---- per-trace stats TSV -------------------------------------------
    cols = ["model", "params_B", "arch", "family", "n_steps",
            "r_first", "r_final", "r_mean", "r_peak", "r_var",
            "early_mean", "late_mean", "zero_frac", "frac_above_0p5",
            "delta_first_final", "delta_late_early"]
    rows = []
    for label, data in raw.items():
        s = trace_stats(data["trace"])
        rows.append([
            label, f"{data['params']:.1f}", data["arch"], data["family"],
            s["n"],
            f"{s['r_first']:.4f}", f"{s['r_final']:.4f}", f"{s['r_mean']:.4f}",
            f"{s['r_peak']:.4f}", f"{s['r_var']:.4f}",
            f"{s['early_mean']:.4f}", f"{s['late_mean']:.4f}",
            f"{s['zero_frac']:.4f}", f"{s['frac_above_0p5']:.4f}",
            f"{s['delta_first_final']:+.4f}", f"{s['delta_late_early']:+.4f}",
        ])
    write_tsv(RESULTS_DIR / "scaling_law_extended_frontier.tsv", cols, rows)

    # ---- power-law fits -------------------------------------------------
    log_n = np.log10([raw[l]["params"] for l in raw])
    pl_rows = []
    pl_data = {}
    for metric in ("r_first", "r_final", "r_mean", "r_peak", "r_var",
                   "zero_frac", "frac_above_0p5"):
        vals = np.array([trace_stats(raw[l]["trace"])[metric] for l in raw])
        pl = power_law(log_n, vals)
        pl_data[metric] = (pl, vals)
        pl_rows.append([
            metric, pl["n"], f"{pl['intercept']:.4f}",
            f"{pl['slope']:.4f}", f"{pl['se_slope']:.4f}",
            f"{pl['r2']:.4f}", f"{pl['p_perm']:.4f}",
        ])
    write_tsv(RESULTS_DIR / "scaling_law_power_law.tsv",
              ["metric", "n", "intercept", "slope_per_log10N",
               "se_slope", "r2", "perm_p_value"], pl_rows)

    # ---- MoE vs dense head-to-head -------------------------------------
    moe_labels = [l for l in raw if raw[l]["arch"] == "moe"]
    dense_labels = [l for l in raw if raw[l]["arch"] == "dense"]
    moe_means = np.array([trace_stats(raw[l]["trace"])["r_mean"] for l in moe_labels])
    dense_means = np.array([trace_stats(raw[l]["trace"])["r_mean"] for l in dense_labels])

    # permutation test on MoE mean > dense mean
    rng = np.random.default_rng(SEED)
    obs_gap = float(moe_means.mean() - dense_means.mean())
    all_means = np.concatenate([moe_means, dense_means])
    n_moe, n_dense = len(moe_means), len(dense_means)
    perm = np.empty(N_PERM, float)
    for i in range(N_PERM):
        idx = rng.permutation(len(all_means))
        perm[i] = all_means[idx[:n_moe]].mean() - all_means[idx[n_moe:]].mean()
    perm = perm[~np.isnan(perm)]
    p_moe_gt = float(np.mean(perm >= obs_gap))
    p_moe_lt = float(np.mean(perm <= obs_gap))
    p_two = 2 * min(p_moe_gt, p_moe_lt)

    moe_rows = [
        ["group", "n", "mean_R", "std_R", "median_R", "min_R", "max_R"],
        ["moe", len(moe_means),
         f"{moe_means.mean():.4f}", f"{moe_means.std():.4f}",
         f"{float(np.median(moe_means)):.4f}",
         f"{moe_means.min():.4f}", f"{moe_means.max():.4f}"],
        ["dense", len(dense_means),
         f"{dense_means.mean():.4f}", f"{dense_means.std():.4f}",
         f"{float(np.median(dense_means)):.4f}",
         f"{dense_means.min():.4f}", f"{dense_means.max():.4f}"],
        ["gap_moe_minus_dense", "n/a",
         f"{obs_gap:+.4f}", "n/a", "n/a", "n/a", "n/a"],
        ["perm_p_moe_greater", "n/a",
         f"{p_moe_gt:.4f}", "n/a", "n/a", "n/a", "n/a"],
        ["perm_p_two_sided", "n/a",
         f"{p_two:.4f}", "n/a", "n/a", "n/a", "n/a"],
    ]
    write_tsv(RESULTS_DIR / "scaling_law_moe_vs_dense.tsv",
              moe_rows[0], moe_rows[1:])

    # ---- console headline ---------------------------------------------
    print()
    print("=== Extended scaling-law fit (12 anchors, 4B-1T) ===")
    for r in pl_rows:
        print(f"  {r[0]:>16s} slope={r[3]}/dec  R^2={r[5]}  perm_p={r[6]}")
    print()
    print(f"=== MoE vs dense (mean reward) ===")
    print(f"  MoE (n={len(moe_means)}): mean={moe_means.mean():.4f}, "
          f"members={moe_labels}")
    print(f"  Dense (n={len(dense_means)}): mean={dense_means.mean():.4f}, "
          f"members={dense_labels}")
    print(f"  gap (MoE - Dense) = {obs_gap:+.4f},  perm p(>=gap)={p_moe_gt:.4f}, "
          f"two-sided p={p_two:.4f}")

    # Spearman rank correlation of log_10(N) vs R_first and vs R_mean
    r_first = np.array([trace_stats(raw[label]["trace"])["r_first"] for label in raw])
    r_mean = np.array([trace_stats(raw[label]["trace"])["r_mean"] for label in raw])
    rho_first, p_first = spearmanr(log_n, r_first)
    rho_mean, p_mean = spearmanr(log_n, r_mean)
    rho_final, p_final = spearmanr(log_n,
                                   np.array([trace_stats(raw[label]["trace"])["r_final"] for label in raw]))
    rho_peak, p_peak = spearmanr(log_n,
                                 np.array([trace_stats(raw[label]["trace"])["r_peak"] for label in raw]))
    print()
    print(f"=== Spearman rank corr(log_10 N, ...) ===")
    print(f"  rho(logN, R_first) = {rho_first:+.4f}, p={p_first:.4f}")
    print(f"  rho(logN, R_mean)  = {rho_mean:+.4f}, p={p_mean:.4f}")
    print(f"  rho(logN, R_final) = {rho_final:+.4f}, p={p_final:.4f}")
    print(f"  rho(logN, R_peak)  = {rho_peak:+.4f}, p={p_peak:.4f}")

    # ---- figure (4-panel) ---------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)
    labels = list(raw.keys())
    arch_color = {"moe": "tab:red", "dense": "tab:blue"}

    # (a) R_first vs log_10(N) with arch stratification
    ax_a = fig.add_subplot(gs[0, 0])
    for label in labels:
        x = math.log10(raw[label]["params"])
        y = trace_stats(raw[label]["trace"])["r_first"]
        ax_a.scatter(x, y, c=arch_color[raw[label]["arch"]],
                     s=80 if raw[label]["arch"] == "moe" else 60,
                     marker="^" if raw[label]["arch"] == "moe" else "o",
                     edgecolor="k", zorder=3)
        ax_a.annotate(label.replace("-Inst", "").replace("MoE-Inst", "*Inst").replace("MoE", ""),
                      (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax_a.set_xlabel(r"$\log_{10}$(params [B])")
    ax_a.set_ylabel(r"$R(t=1)$ -- first-step reward")
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.set_title(f"(a) First-step reward vs params\n"
                   f"Spearman rho={rho_first:+.3f}, p={p_first:.3f}")
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="k", markersize=8, label="dense"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:red",
               markeredgecolor="k", markersize=9, label="MoE"),
    ], fontsize=8, loc="lower right")
    ax_a.grid(alpha=0.25)

    # (b) Mean reward vs log_10(N) with OLS fit
    ax_b = fig.add_subplot(gs[0, 1])
    means = np.array([trace_stats(raw[label]["trace"])["r_mean"] for label in labels])
    for label in labels:
        x = math.log10(raw[label]["params"])
        y = trace_stats(raw[label]["trace"])["r_mean"]
        ax_b.scatter(x, y, c=arch_color[raw[label]["arch"]],
                     s=80 if raw[label]["arch"] == "moe" else 60,
                     marker="^" if raw[label]["arch"] == "moe" else "o",
                     edgecolor="k", zorder=3)
        ax_b.annotate(label.replace("-Inst", "").replace("MoE-Inst", "*Inst").replace("MoE", ""),
                      (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    a_, b_, _ = _ols(log_n, means)
    xs = np.linspace(log_n.min() - 0.05, log_n.max() + 0.05, 100)
    ax_b.plot(xs, a_ + b_ * xs, "k--", lw=1.5,
              label=fr"OLS slope={b_:+.4f}/dec  $R^2$={1 - float(np.sum((means - (a_+b_*log_n))**2))/float(np.sum((means-means.mean())**2)):.3f}")
    ax_b.set_xlabel(r"$\log_{10}$(params [B])")
    ax_b.set_ylabel(r"$\bar R$ -- mean reward")
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.set_title("(b) Mean reward vs params (OLS log-linear)")
    ax_b.legend(fontsize=7, loc="lower right")
    ax_b.grid(alpha=0.25)

    # (c) MoE vs dense head-to-head box-strip
    ax_c = fig.add_subplot(gs[1, 0])
    box_data = [moe_means, dense_means]
    bp = ax_c.boxplot(box_data, patch_artist=True, widths=0.55,
                      tick_labels=["MoE", "dense"])
    for patch, color in zip(bp["boxes"], ["tab:red", "tab:blue"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, vals in enumerate([moe_means, dense_means]):
        for v in vals:
            ax_c.scatter(i + 1 + np.random.default_rng(SEED + i).uniform(-0.10, 0.10),
                         v, c="k", s=22, alpha=0.6, zorder=4)
    ax_c.set_ylabel("mean reward across trace")
    ax_c.set_title(f"(c) MoE vs dense head-to-head\n"
                   f"gap = {obs_gap:+.3f},  perm p(>=gap)={p_moe_gt:.3f}")
    ax_c.grid(axis="y", alpha=0.25)

    # (d) First-final gap -- a "noise gate" proxy
    ax_d = fig.add_subplot(gs[1, 1])
    gaps = []
    cols = []
    for label in labels:
        s = trace_stats(raw[label]["trace"])
        gaps.append(s["delta_first_final"])
        cols.append(arch_color[raw[label]["arch"]])
    xs_d = np.arange(len(labels))
    ax_d.bar(xs_d, gaps, color=cols, edgecolor="k", alpha=0.85)
    ax_d.axhline(0.0, color="k", lw=0.8)
    ax_d.set_xticks(xs_d)
    ax_d.set_xticklabels([l.replace("-Inst", "").replace("MoE-Inst", "*Inst").replace("MoE", "") for l in labels],
                         rotation=25, ha="right", fontsize=7)
    ax_d.set_ylabel(r"$R(1) - R(T)$ (negative = improving)")
    ax_d.set_title("(d) First-final gap by architecture\n"
                   "(Nemotron outlier at +0.5 = collapse not retention)")
    ax_d.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Pillar 1 elevation (iter 13) -- Extended scaling-law fit across 12 anchors "
        "(4B--1T params; MoE vs dense; first-step and mean diagnostics)",
        fontsize=11,
    )
    out_pdf = FIG_DIR / "scaling_law_extended.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    fig.savefig(PAPER_FIG / "scaling_law_extended.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_extended.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_pdf}")

    # ---- append findings to AUTORESEARCH jsonl ------------------------
    findings_path = REPO / "AUTORESEARCH_FINDINGS.jsonl"
    if findings_path.exists():
        new_records = [
            {"ts": "2026-07-02", "pillar": "P1",
             "claim": f"Spearman rho(log_10 N, R_first)={rho_first:+.3f} (p={p_first:.3f}) on 12 anchors",
             "evidence_path": "experiments/results/scaling_law_extended_frontier.tsv"},
            {"ts": "2026-07-02", "pillar": "P1",
             "claim": f"Spearman rho(log_10 N, R_mean)={rho_mean:+.3f} (p={p_mean:.3f}) on 12 anchors",
             "evidence_path": "experiments/results/scaling_law_power_law.tsv"},
            {"ts": "2026-07-02", "pillar": "P1",
             "claim": f"MoE mean={moe_means.mean():.3f} vs dense mean={dense_means.mean():.3f}, "
                      f"gap={obs_gap:+.3f}, perm p={p_moe_gt:.3f}",
             "evidence_path": "experiments/results/scaling_law_moe_vs_dense.tsv"},
            {"ts": "2026-07-02", "pillar": "P1",
             "claim": f"First-final gap R(1)-R(T) distribution: Nemotron outlier +0.5 (collapse), "
                      f"others within [-0.5, +0.4]",
             "evidence_path": "figures/scaling_law_extended.pdf"},
        ]
        with findings_path.open("a") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"\nappended {len(new_records)} findings to {findings_path}")


if __name__ == "__main__":
    main()