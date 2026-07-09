#!/usr/bin/env python3
"""
zvf_iter94.py — Pillar 2 (ZVF): Cross-Library Operational Diagnostic Dashboard.

Four fresh analyses, all on real iter-90 / iter-82 / dynamics data:

  1. SHUFFLE-NULL ANTI-HERDING BONUS
     For each (prompt-batch mean reward p, group size G) in the 12 per-step
     groupsize_zvf_sweep runs, compute the i.i.d. baseline ZVF_iid = p**G + (1-p)**G.
     Compare to empirical mean ZVF. The gap delta_div = ZVF_empirical - ZVF_iid
     is the anti-herding bonus (frontier synthesis: ~+0.13..+0.23).

  2. CROSS-LIBRARY ZVF STABILITY SCORE
     For each of the 9 variance-mitigation algorithms (grpo, aero, cppo, ngrpo,
     mcgrpo, gift, areal, scafgrpo, es) combine iter90 recovery_rate,
     iter86 alarm_rate (=1 - n_never_alarmed/trajectory_count), and
     iter90 post_episode delta into a single [0, 100] stability score.

  3. RECOVERY-FORECAST CALIBRATION
     Using the 12 per-step groupsize_zvf_sweep traces, fit the empirical
     distribution of (mean_zvf, std_zvf, lag1) per run and compute the
     predictive separation between healthy runs (last10_acc > 0.90) and
     at-risk runs (last10_acc < 0.90).

  4. ZVF vs AERO HEADLINE
     AERO is the only non-GRPO-derivative reference at iter90's group.
     Compute the (recovery_rate_AERO - recovery_rate_method) gap and the
     decision-table winners per metric.

Outputs:
    experiments/results/zvf_iter94_shuffle_null.tsv
    experiments/results/zvf_iter94_stability.tsv
    experiments/results/zvf_iter94_calibration.tsv
    experiments/results/zvf_summary.tsv            <-- headline
    figures/zvf_vs_failure.{pdf,png}
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load_groupsize_perstep():
    """Return list of runs with their per-step time series."""
    with (RES / "groupsize_zvf_sweep.json").open() as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        zvf = np.array([s["zvf"] for s in sl], dtype=float)
        rew = np.array([s["mean_reward"] for s in sl], dtype=float)
        out.append(dict(
            method=r["model"],
            group_size=r["group_size"],
            seed=r["seed"],
            n_steps=r["n_steps"],
            n_prompts=r["n_prompts"],
            heldout_acc=r["heldout_acc"],
            last10=r["last10_avg"],
            mean_zvf=float(np.mean(zvf)),
            std_zvf=float(np.std(zvf)),
            lag1=float(np.corrcoef(zvf[:-1], zvf[1:])[0, 1]) if len(zvf) > 2 else float("nan"),
            mean_reward=float(np.mean(rew)),
            zvf=zvf,
            rew=rew,
        ))
    return out


def load_iter90_recovery():
    """Return per-method recovery stats table."""
    rows = []
    with (RES / "zvf_iter90_recovery.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def load_iter90_post_episode():
    rows = []
    with (RES / "zvf_iter90_post_episode.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def load_iter90_episodes():
    rows = []
    with (RES / "zvf_iter90_episodes.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def load_dynamics_summary():
    rows = []
    with (RES / "zvf_dynamics_summary.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


# ---------- 1. SHUFFLE-NULL ANTI-HERDING BONUS ----------
def shuffle_null_perstep(perstep):
    """For each (G, run) compute ZVF_iid(p, G) and gap to empirical mean ZVF."""
    out = []
    for r in perstep:
        G = r["group_size"]
        p = r["mean_reward"]
        # i.i.d. baseline: P(all-correct) + P(all-wrong)
        zvf_iid = p ** G + (1 - p) ** G
        # empirical mean ZVF
        zvf_emp = r["mean_zvf"]
        delta_div = zvf_emp - zvf_iid
        out.append(dict(
            model=r["method"],
            G=G,
            seed=r["seed"],
            mean_p=round(p, 4),
            zvf_iid=round(zvf_iid, 4),
            zvf_emp=round(zvf_emp, 4),
            delta_div=round(delta_div, 4),
            abs_delta_div=round(abs(delta_div), 4),
            last10_acc=round(r["last10"], 4),
        ))
    return out


# ---------- 2. CROSS-LIBRARY STABILITY SCORE ----------
def stability_scores(recovery, post, eps):
    """Composite stability score [0,100] per method.

    Inputs: iter90 per-method recovery_rate and iter86 alarm_rate proxy.
    Higher score = more stable (lower alarm + higher recovery + smaller
    post-recovery accuracy gap).
    """
    alarm_rate = {r["method"]: 1 - int(r["n_never_alarmed_traces"]) / max(int(r["trajectory_count"]), 1)
                  for r in recovery}
    recov = {r["method"]: float(r["recovery_rate"]) for r in recovery}

    # post-recovery delta per method: avg(ha_after_recovery) - avg(ha_after_sustained)
    by_method_delta = {}
    for r in post:
        m = r["method"]
        cat = r["cat"]
        # iter90 columns: method, cat (recovered/sustained), n_obs, mean_ha, median_ha
        try:
            ha_str = r["mean_ha"]
            ha = float(ha_str) if ha_str not in ("", "NA", "nan") else float("nan")
        except (KeyError, ValueError):
            ha = float("nan")
        # group "recovered" -> "recovery" label for parity with internal vocab
        norm_cat = "recovery" if cat == "recovered" else "sustained"
        by_method_delta.setdefault(m, {"recovery": [], "sustained": []})[norm_cat].append(ha)
    delta_h = {}
    for m, v in by_method_delta.items():
        if v["recovery"] and v["sustained"]:
            rec_vals = [x for x in v["recovery"] if not math.isnan(x)]
            sust_vals = [x for x in v["sustained"] if not math.isnan(x)]
            if rec_vals and sust_vals:
                delta_h[m] = float(np.mean(rec_vals)) - float(np.mean(sust_vals))
            else:
                delta_h[m] = 0.0
        else:
            delta_h[m] = 0.0

    out = []
    for m in sorted(alarm_rate):
        ar = alarm_rate[m]
        rr = recov.get(m, 0.0)
        dh = delta_h.get(m, 0.0)
        # Stability = (1 - alarm_rate) * 60 + recovery_rate * 30 + max(0, 1 - |delta_h|) * 10
        score = (1 - ar) * 60 + rr * 30 + max(0.0, 1 - abs(dh)) * 10
        out.append(dict(
            method=m,
            alarm_rate_proxy=round(ar, 4),
            recovery_rate_K5=round(rr, 4),
            post_recovery_delta=round(dh, 4),
            stability_score=round(score, 2),
        ))
    return out


# ---------- 3. RECOVERY-FORECAST CALIBRATION ----------
def recovery_calibration(perstep):
    """Split runs into healthy / at-risk by last10 threshold; compute
    the per-step ZVF distributional separation (mean, std, lag1, max)."""
    rows = []
    healthy_stats, atrisk_stats = [], []
    for r in perstep:
        z = r["zvf"]
        rec = dict(
            method=r["method"],
            G=r["group_size"],
            seed=r["seed"],
            last10=r["last10"],
            mean_zvf=round(r["mean_zvf"], 4),
            std_zvf=round(r["std_zvf"], 4),
            lag1=round(r["lag1"], 4),
            max_zvf=round(float(np.max(z)), 4),
            min_zvf=round(float(np.min(z)), 4),
            auc_above_07=round(float(np.mean(np.maximum(z - 0.7, 0))), 4),
            auc_above_09=round(float(np.mean(np.maximum(z - 0.9, 0))), 4),
        )
        rows.append(rec)
        if r["last10"] >= 0.90:
            healthy_stats.append([r["mean_zvf"], r["std_zvf"], r["lag1"],
                                  float(np.max(z)), float(np.min(z))])
        else:
            atrisk_stats.append([r["mean_zvf"], r["std_zvf"], r["lag1"],
                                 float(np.max(z)), float(np.min(z))])
    summary = {
        "n_healthy": len(healthy_stats),
        "n_atrisk": len(atrisk_stats),
        "healthy_mean": np.nanmean(healthy_stats, axis=0).tolist() if healthy_stats else [],
        "atrisk_mean": np.nanmean(atrisk_stats, axis=0).tolist() if atrisk_stats else [],
    }
    return rows, summary


# ---------- 4. ZVF vs AERO HEADLINE ----------
def zvf_vs_aero(stability):
    a = next((s for s in stability if s["method"] == "aero"), None)
    out = []
    for s in stability:
        if a is None:
            gap = 0.0
        else:
            gap = s["stability_score"] - a["stability_score"]
        out.append(dict(
            method=s["method"],
            stability_score=s["stability_score"],
            aero_gap=round(gap, 2),
            beats_aero=gap > 0,
        ))
    return out


# ---------- WRITERS ----------
def write_tsv(path, rows, header_comment=None):
    with path.open("w") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
        if not rows:
            f.write("(empty)\n")
            return
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def write_summary_tsv(path, stability, zvfaero, calibration_summary, deltas, recovery, post):
    """Compose zvf_summary.tsv — the headline dashboard."""
    by_m = {r["method"]: r for r in recovery}
    post_by_m = {}
    for r in post:
        m = r["method"]
        cat_norm = "recovery" if r["cat"] == "recovered" else "sustained"
        try:
            ha_str = r["mean_ha"]
            ha = float(ha_str) if ha_str not in ("", "NA", "nan") else float("nan")
        except (KeyError, ValueError):
            ha = float("nan")
        post_by_m.setdefault(m, {"recovery": [], "sustained": []})[cat_norm].append(ha)

    cols = [
        "method", "alarm_rate", "recovery_rate_K5", "post_recovery_ha",
        "post_sustained_ha", "delta_h_recovery_minus_sustained",
        "stability_score", "aero_gap", "beats_aero",
        "delta_div_at_mean_p_obs_G4_med", "n_episodes", "n_sustained",
        "n_recovered", "iter90_recovery_rate",
    ]
    rows = []
    for s in stability:
        m = s["method"]
        pr = post_by_m.get(m, {"recovery": [], "sustained": []})
        pra_vals = [v for v in pr["recovery"] if not math.isnan(v)]
        psa_vals = [v for v in pr["sustained"] if not math.isnan(v)]
        pra = float(np.mean(pra_vals)) if pra_vals else float("nan")
        psa = float(np.mean(psa_vals)) if psa_vals else float("nan")
        d_h = pra - psa if (not math.isnan(pra) and not math.isnan(psa)) else float("nan")
        d4 = next((d["delta_div"] for d in deltas if d["G"] == 4 and d["model"] == "Qwen/Qwen2.5-0.5B"), float("nan"))
        # median across G=2/4/8/16 across the 3 seeds:
        d4_med = float(np.median([d["delta_div"] for d in deltas if d["G"] == 4])) if any(d["G"] == 4 for d in deltas) else float("nan")
        rec = by_m.get(m, {})
        rows.append({
            "method": m,
            "alarm_rate": s["alarm_rate_proxy"],
            "recovery_rate_K5": s["recovery_rate_K5"],
            "post_recovery_ha": round(pra, 4),
            "post_sustained_ha": round(psa, 4),
            "delta_h_recovery_minus_sustained": round(d_h, 4),
            "stability_score": s["stability_score"],
            "aero_gap": s["stability_score"] - next(x["stability_score"] for x in stability if x["method"] == "aero"),
            "beats_aero": s["stability_score"] > next(x["stability_score"] for x in stability if x["method"] == "aero"),
            "delta_div_at_mean_p_obs_G4_med": round(d4_med, 4),
            "n_episodes": rec.get("n_episodes", "0"),
            "n_sustained": rec.get("n_sustained", "0"),
            "n_recovered": rec.get("n_recovered", "0"),
            "iter90_recovery_rate": rec.get("recovery_rate", "0.0"),
        })

    with path.open("w") as f:
        f.write("# zvf_summary.tsv — Pillar 2 headline ZVF dashboard\n")
        f.write("# Aggregates iter90 recovery + iter90 post-episode + iter82 alarm rates\n")
        f.write("# + iter94 shuffle-null delta_div at G=4 + iter94 stability score.\n")
        f.write("# stability_score = (1 - alarm_rate)*60 + recovery_rate*30 + max(0,1-|delta_h|)*10\n")
        f.write("# delta_div = empirical ZVF - i.i.d. baseline ZVF at mean prompt accuracy p\n")
        f.write("# aero_gap > 0: this method's stability_score exceeds AERO\n")
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def make_figure(stability, deltas, calibration_summary, recovery, zvfaero, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # Panel (a) — Per-method stability score (sorted descending) with AERO highlighted
    ax = axes[0, 0]
    methods = [s["method"] for s in stability]
    scores = [s["stability_score"] for s in stability]
    order = np.argsort(scores)[::-1]
    colors = ["#d62728" if m == "aero" else "#1f77b4" for m in methods]
    ax.bar([methods[i] for i in order], [scores[i] for i in order], color=[colors[i] for i in order])
    ax.axhline(scores[methods.index("aero")] if "aero" in methods else 0,
               ls="--", c="#d62728", lw=0.9, alpha=0.6, label="AERO score")
    ax.set_ylabel("ZVF stability score [0, 100]")
    ax.set_title("(a) Cross-library ZVF stability score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower right")

    # Panel (b) — Shuffle-null anti-herding bonus delta_div per G
    ax = axes[0, 1]
    Gs = sorted({d["G"] for d in deltas})
    bp = ax.boxplot(
        [[d["delta_div"] for d in deltas if d["G"] == g] for g in Gs],
        tick_labels=[str(g) for g in Gs],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ca02c")
        patch.set_alpha(0.6)
    ax.axhline(0, ls=":", c="k", lw=0.7)
    ax.set_xlabel("Group size G")
    ax.set_ylabel("delta_div = empirical ZVF − i.i.d. ZVF")
    ax.set_title("(b) Anti-herding bonus per G")

    # Panel (c) — Recovery rate vs post-episode delta_h scatter
    ax = axes[1, 0]
    xs, ys = [], []
    for s in stability:
        if not np.isnan(s["post_recovery_delta"]):
            xs.append(s["recovery_rate_K5"])
            ys.append(s["post_recovery_delta"])
    methods_c = [s["method"] for s in stability if not np.isnan(s["post_recovery_delta"])]
    ax.scatter(xs, ys, s=70, c=["#d62728" if m == "aero" else "#1f77b4" for m in methods_c])
    for i, m in enumerate(methods_c):
        ax.annotate(m, (xs[i], ys[i]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, ls=":", c="k", lw=0.7)
    ax.set_xlabel("Recovery rate at K=5 (iter90)")
    ax.set_ylabel("Δ(ha_recovery − ha_sustained) [iter90]")
    ax.set_title("(c) Recovery vs post-episode accuracy")

    # Panel (d) — Recovery lead bar: recov rate − baseline (0.5 random)
    ax = axes[1, 1]
    methods = [s["method"] for s in stability]
    xs = np.arange(len(methods))
    ys = [s["recovery_rate_K5"] - 0.5 for s in stability]
    colors = ["#d62728" if m == "aero" else "#1f77b4" for m in methods]
    ax.bar(xs, ys, color=colors)
    ax.axhline(0, ls=":", c="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=30)
    ax.set_ylabel("Recovery rate − 0.5 (random)")
    ax.set_title("(d) Above-chance recovery")

    fig.suptitle("Iter 94 — Pillar 2 ZVF cross-library diagnostic dashboard", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    perstep = load_groupsize_perstep()
    recovery = load_iter90_recovery()
    post = load_iter90_post_episode()
    eps = load_iter90_episodes()
    dyn_summary = load_dynamics_summary()

    # 1. Shuffle-null
    deltas = shuffle_null_perstep(perstep)
    write_tsv(
        RES / "zvf_iter94_shuffle_null.tsv",
        deltas,
        header_comment=(
            "zvf_iter94_shuffle_null.tsv — anti-herding bonus calibration. "
            "Per (model, G, seed) of groupsize_zvf_sweep; "
            "zvf_iid = p^G + (1-p)^G, delta_div = empirical ZVF − zvf_iid. "
            "delta_div > 0 means anti-herding (frontier synthesis)."
        ),
    )

    # 2. Stability score
    stability = stability_scores(recovery, post, eps)
    write_tsv(
        RES / "zvf_iter94_stability.tsv",
        stability,
        header_comment=(
            "zvf_iter94_stability.tsv — cross-library ZVF stability score. "
            "alarm_rate_proxy = 1 - n_never_alarmed/n_traces (iter90). "
            "recovery_rate = recovered/total_episodes at K=5. "
            "post_recovery_delta = mean(post-recovery ha) - mean(post-sustained ha). "
            "stability_score = (1-alarm_rate)*60 + recovery_rate*30 + max(0, 1-|delta_h|)*10."
        ),
    )

    # 3. Calibration
    cal_rows, cal_summary = recovery_calibration(perstep)
    write_tsv(
        RES / "zvf_iter94_calibration.tsv",
        cal_rows,
        header_comment=(
            "zvf_iter94_calibration.tsv — per-run ZVF shape stats for "
            "12 groupsize_zvf_sweep runs. Use last10_acc>=0.90 to split healthy/at-risk."
        ),
    )

    # 4. ZVF vs AERO
    zvfaero = zvf_vs_aero(stability)
    write_tsv(
        RES / "zvf_iter94_vs_aero.tsv",
        zvfaero,
        header_comment=(
            "zvf_iter94_vs_aero.tsv — signed gap to AERO on iter94 stability_score."
        ),
    )

    # Headline dashboard
    write_summary_tsv(RES / "zvf_summary.tsv", stability, zvfaero,
                      cal_summary, deltas, recovery, post)

    # Figure
    out_pdf = FIG / "zvf_vs_failure.pdf"
    out_png = FIG / "zvf_vs_failure.png"
    make_figure(stability, deltas, cal_summary, recovery, zvfaero, out_pdf, out_png)

    # Console echo
    print("[iter94] zvf_summary.tsv written")
    print("[iter94] zvf_iter94_shuffle_null.tsv rows:", len(deltas))
    print("[iter94] zvf_iter94_stability.tsv rows:", len(stability))
    print("[iter94] zvf_iter94_calibration.tsv rows:", len(cal_rows))
    print("[iter94] zvf_iter94_vs_aero.tsv rows:", len(zvfaero))
    print("[iter94] figure:", out_pdf, "+", out_png)
    # Highlight: median delta_div by G
    for g in sorted({d["G"] for d in deltas}):
        vals = [d["delta_div"] for d in deltas if d["G"] == g]
        print(f"[iter94]   median delta_div(G={g}) = {np.median(vals):+.4f}, "
              f"min={min(vals):+.4f}, max={max(vals):+.4f}")
    # Stability winner
    winner = max(stability, key=lambda x: x["stability_score"])
    print(f"[iter94] STABILITY WINNER: {winner['method']} "
          f"(score={winner['stability_score']:.1f}, "
          f"recovery={winner['recovery_rate_K5']:.3f}, "
          f"alarm={winner['alarm_rate_proxy']:.3f})")


if __name__ == "__main__":
    main()
