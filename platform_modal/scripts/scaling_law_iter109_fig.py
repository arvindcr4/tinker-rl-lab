"""scaling_law_iter109_fig.py -- Generate the iter109 figure with 4 panels:
(a) traces + 3-param fits, (b) lambda-vs-N log-log with perm-test annotation,
(c) time-to-saturation bars, (d) Nemotron collapse zoom.

Reads from experiments/results/scaling_law_iter109_{saturation,lambdaN,nemotron,
permtest,family,stability}.tsv + the canonical reward_trace JSONs.

Outputs:
  figures/scaling_law_iter109.{pdf,png}
  paper/figures/scaling_law_iter109.{pdf,png}
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "experiments" / "tinker-runs" / "results"
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
PAPER_FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)
PAPER_FIG.mkdir(exist_ok=True)

# Same MODELS dict as iter109.
MODELS: dict[str, tuple[float, str, str]] = {
    "Qwen3.5-4B":            (4.0,    "scale_gsm8k_qwen3.5-4b.json",     "dense"),
    "Qwen3-8B":              (8.0,    "scale_gsm8k_qwen3-8b.json",       "dense"),
    "Llama-3.1-8B-Instruct": (8.0,    "scale_gsm8k_llama-8b-inst.json",  "dense"),
    "Qwen3-32B":             (32.0,   "scale_gsm8k_qwen3-32b.json",      "dense"),
    "Qwen3.5-27B":           (27.0,   "scale_gsm8k_qwen3.5-27b.json",    "dense"),
    "gpt-oss-20B":           (20.0,   "arch_gsm8k_gpt-oss-20b.json",     "moe"),
    "Qwen3-30B-MoE":         (30.0,   "moe_gsm8k_qwen3-30b-moe.json",    "moe"),
    "Qwen3-30B-MoE-Inst":    (30.0,   "moe_gsm8k_qwen3-30b-inst.json",   "moe"),
    "DeepSeek-V3.1":         (685.0,  "frontier_gsm8k_deepseek-v3.1.json","moe"),
    "Nemotron-120B":         (120.0,  "frontier_gsm8k_nemotron-120b.json","dense"),
    "Qwen3-235B-MoE":        (235.0,  "frontier_gsm8k_qwen3-235b.json",  "moe"),
    "Kimi-K2-Thinking":      (1000.0, "arch_gsm8k_kimi-k2.json",         "moe"),
}


def f_sat_3p(t, r0, rinf, lam):
    return r0 + (rinf - r0) * (1.0 - np.exp(-lam * t))


def load_traces() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, (_, fname, _) in MODELS.items():
        d = json.loads((TR / fname).read_text())
        rt = np.asarray(d["reward_trace"], float)
        t = np.arange(1, len(rt) + 1, dtype=float)
        out[name] = (t, rt)
    return out


def main() -> None:
    raw = load_traces()

    # Read fitted values from iter109 saturation TSV.
    sat_rows = {}
    with (RES / "scaling_law_iter109_saturation.tsv").open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sat_rows[parts[idx["model"]]] = {
                "r0": float(parts[idx["r0"]]),
                "rinf": float(parts[idx["R_inf"]]),
                "lam": float(parts[idx["lambda"]]),
                "lam_lo": float(parts[idx["lambda_lo"]]),
                "lam_hi": float(parts[idx["lambda_hi"]]),
                "rmse": float(parts[idx["rmse"]]),
                "family": parts[idx["family"]],
            }
    # Lambda-vs-N regression rows
    with (RES / "scaling_law_iter109_lambdaN.tsv").open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_l = {h: i for i, h in enumerate(header)}
        lamN = [line.rstrip("\n").split("\t") for line in f]
    a_all = float(lamN[0][idx_l["intercept"]]); b_all = float(lamN[0][idx_l["slope_per_log10N"]])
    a_f = float(lamN[1][idx_l["intercept"]]); b_f = float(lamN[1][idx_l["slope_per_log10N"]])
    boot_lo = float(lamN[1][idx_l["boot_slope_lo"]])
    boot_hi = float(lamN[1][idx_l["boot_slope_hi"]])

    # Permutation test results
    perm_meta = json.loads((RES / "scaling_law_iter109b_meta.json").read_text())
    p_lam = perm_meta["E_permutation_test"]["lambda_vs_N"]["p_two_sided"]
    z_lam = perm_meta["E_permutation_test"]["lambda_vs_N"]["z_score"]
    p_ri = perm_meta["E_permutation_test"]["R_inf_vs_N"]["p_two_sided"]

    # Family rows
    with (RES / "scaling_law_iter109b_family.tsv").open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_f = {h: i for i, h in enumerate(header)}
        family_rows = [line.rstrip("\n").split("\t") for line in f]
    fam_dense = next((r for r in family_rows if r[idx_f["family"]] == "dense"), None)
    fam_moe = next((r for r in family_rows if r[idx_f["family"]] == "moe"), None)

    # Nemotron row
    with (RES / "scaling_law_iter109_nemotron.tsv").open() as f:
        header = f.readline().rstrip("\n").split("\t")
        idx_n = {h: i for i, h in enumerate(header)}
        nem = f.readline().rstrip("\n").split("\t")
    peak_v = float(nem[idx_n["peak_R"]])
    late_mean = float(nem[idx_n["late_mean"]])
    nem_lam = float(nem[idx_n["lambda_3p"]])
    nem_lam_lo = float(nem[idx_n["lambda_3p_lo"]])
    nem_lam_hi = float(nem[idx_n["lambda_3p_hi"]])

    # ---- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)

    # (a) traces with fitted 3-param curves + 80% asymptote lines
    ax_a = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("viridis")
    names = list(MODELS.keys())
    for i, name in enumerate(names):
        t, y = raw[name]
        s = sat_rows[name]
        col = cmap(i / max(1, len(names) - 1))
        ax_a.plot(t, y, "o", color=col, markersize=4, alpha=0.85,
                  label=f"{name} ({MODELS[name][0]:.0f}B)")
        t_dense = np.linspace(t.min(), max(t.max() * 1.5, 10.0), 100)
        yhat = np.clip(f_sat_3p(t_dense, s["r0"], s["rinf"], s["lam"]), -0.1, 1.5)
        ax_a.plot(t_dense, yhat, "-", color=col, lw=1.0, alpha=0.55)
    ax_a.set_xlabel("training step"); ax_a.set_ylabel("reward")
    ax_a.set_ylim(-0.05, 1.20)
    ax_a.set_title("(a) 12-anchor traces + 3-param saturation fits")
    ax_a.grid(alpha=0.25); ax_a.legend(fontsize=6, loc="lower right", ncol=2)

    # (b) lambda-vs-N (log-log) with perm-test annotation
    ax_b = fig.add_subplot(gs[0, 1])
    fam_color = {"dense": "tab:blue", "moe": "tab:red"}
    for name in names:
        s = sat_rows[name]
        if s["lam"] > 0 and not math.isnan(s["lam"]):
            params_b = MODELS[name][0]
            col = fam_color[s["family"]]
            ax_b.scatter(math.log10(params_b), math.log10(s["lam"]),
                         color=col, edgecolor="k", s=70, zorder=3, alpha=0.85)
            if s["lam_lo"] > 0 and s["lam_hi"] > 0:
                ax_b.plot([math.log10(params_b), math.log10(params_b)],
                          [math.log10(s["lam_lo"]), math.log10(s["lam_hi"])],
                          color=col, lw=1.0, alpha=0.6)
    xs = np.linspace(-0.1, 3.2, 50)
    ax_b.plot(xs, a_all + b_all * xs, "k--", lw=1.2,
              label=f"all: slope={b_all:+.2f}/dec")
    ax_b.plot(xs, a_f + b_f * xs, "k:", lw=1.2,
              label=f"filtered: slope={b_f:+.2f}/dec")
    ax_b.fill_between(xs,
                      a_f + boot_lo * xs,
                      a_f + boot_hi * xs,
                      color="grey", alpha=0.18,
                      label=f"boot 95% CI [{boot_lo:+.2f},{boot_hi:+.2f}]")
    ax_b.axhline(np.median([math.log10(sat_rows[n]["lam"]) for n in names
                            if sat_rows[n]["lam"] > 0]),
                 color="grey", lw=0.7, alpha=0.5,
label="H0: slope=0 (median)")
    ax_b.set_xlabel(r"$\log_{10}$(params [B])")
    ax_b.set_ylabel(r"$\log_{10}(\lambda_{3p})$")
    permtxt = (f"permutation test: p_two={p_lam:.3f}\n"
               f"reject H0 at 0.05: {p_lam<0.05}\n"
               f"R_inf perm p_two={p_ri:.3f}")
    ax_b.text(0.02, 0.02, permtxt, transform=ax_b.transAxes, fontsize=8,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
              va="bottom", ha="left")
    ax_b.set_title(f"(b) lambda-vs-N falsification (z={z_lam:+.2f})")
    ax_b.grid(alpha=0.25)
    from matplotlib.lines import Line2D
    ax_b.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="k", markersize=9, label="dense"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red",
               markeredgecolor="k", markersize=9, label="moe"),
        Line2D([0], [0], color="k", ls="--", label=f"all: {b_all:+.2f}/dec"),
        Line2D([0], [0], color="k", ls=":", label=f"filt: {b_f:+.2f}/dec"),
    ], fontsize=7, loc="upper right")
    if fam_dense is not None:
        d_slope = fam_dense[idx_f["slope_per_log10N"]]
        d_p = fam_dense[idx_f["perm_p_two_sided"]]
        m_slope = fam_moe[idx_f["slope_per_log10N"]] if fam_moe else "nan"
        m_p = fam_moe[idx_f["perm_p_two_sided"]] if fam_moe else "nan"
        ax_b.text(0.98, 0.55,
                  f"family-stratified:\n"
                  f"  dense: slope={d_slope}, p={d_p}\n"
                  f"  moe:   slope={m_slope}, p={m_p}",
                  transform=ax_b.transAxes, fontsize=7, ha="right",
                  bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                            ec="grey", alpha=0.85))

    # (c) time-to-saturation bars
    ax_c = fig.add_subplot(gs[1, 0])
    fracs = [0.5, 0.7, 0.8, 0.9]
    width = 0.20
    xpos = np.arange(len(names))
    for j, f in enumerate(fracs):
        vals = []
        for name in names:
            t, y = raw[name]
            peak = float(np.max(y))
            idx_first = np.where(y >= f * peak)[0]
            v = float(t[idx_first[0]]) if len(idx_first) else 0.0
            vals.append(v)
        ax_c.bar(xpos + (j - 1.5) * width, vals, width=width,
                 label=f"t_{int(f*100)}%", alpha=0.85, edgecolor="k")
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels([n.replace("-Inst", "") for n in names],
                         rotation=25, ha="right", fontsize=8)
    ax_c.set_ylabel("training step")
    ax_c.set_title("(c) Time-to-saturation onset (t_50/70/80/90 of peak)")
    ax_c.grid(axis="y", alpha=0.25); ax_c.legend(fontsize=7)

    # (d) Nemotron collapse zoom
    ax_d = fig.add_subplot(gs[1, 1])
    nt, ny = raw["Nemotron-120B"]
    s = sat_rows["Nemotron-120B"]
    ax_d.bar(nt, ny, color="tab:red", alpha=0.55, edgecolor="k",
             label=f"trace (peak={peak_v:.2f})")
    t_dense = np.linspace(1, len(nt), 100)
    yhat_n = f_sat_3p(t_dense, s["r0"], s["rinf"], s["lam"])
    ax_d.plot(t_dense, yhat_n, "k-", lw=1.4,
              label=fr"3-param fit $\lambda$={nem_lam:.3f} [{nem_lam_lo:.3f},{nem_lam_hi:.3f}]")
    ax_d.axhline(0.8 * s["rinf"], ls=":", color="tab:purple", lw=1.0,
                 label=fr"$0.8 R_{{\infty}}={0.8*s['rinf']:.2f}$")
    ax_d.axhline(late_mean, ls="--", color="tab:blue", lw=1.0,
                 label=f"late mean={late_mean:.2f}")
    pi = int(np.argmax(ny))
    ax_d.annotate(f"peak {peak_v:.2f} @ step {pi+1}",
                  xy=(pi + 1, peak_v), xytext=(pi + 1.5, peak_v + 0.06),
                  arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=8)
    ax_d.set_xlabel("training step"); ax_d.set_ylabel("reward")
    ax_d.set_ylim(0, 1.05)
    ax_d.set_title("(d) Nemotron-120B: peak 0.875 not retained (recovery=4.20x)")
    ax_d.legend(fontsize=7); ax_d.grid(alpha=0.25)

    fig.suptitle(
        "Pillar 1 iter109 -- 3-param saturation R(t)=r0+(R_inf-r0)*(1-exp(-lambda t)): "
        "lambda-vs-N scaling FAILS (perm p_two=0.44, n=12)",
        fontsize=11,
    )
    for fpx in (FIG / "scaling_law_iter109.pdf",
                FIG / "scaling_law_iter109.png",
                PAPER_FIG / "scaling_law_iter109.pdf",
                PAPER_FIG / "scaling_law_iter109.png"):
        fig.savefig(fpx, dpi=150 if fpx.suffix == ".png" else None)
    plt.close(fig)
    print("wrote figures/scaling_law_iter109.{pdf,png} and paper/figures/...")


if __name__ == "__main__":
    main()