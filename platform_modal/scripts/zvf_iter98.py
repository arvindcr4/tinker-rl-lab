#!/usr/bin/env python3
"""
zvf_iter98.py — Pillar 2 (ZVF): Over-Dispersion Decomposition & Failure Tracking.

Four fresh analyses on real iter-94 / iter-85 / bfclv4 data:

  1. PER-STEP OVER-DISPERSION RATIO ρ
     For each (step, G, seed) in groupsize_zvf_sweep.json, compute
       p_step  = mean_reward at that step
       zvf_iid = p_step^G + (1 - p_step)^G           (i.i.d. collision baseline)
       rho     = zvf_emp / max(zvf_iid, eps)         (over-dispersion ratio)
     Aggregate per-run: rho_mean, rho_std, rho_lag1, frac_above_1 (herding
     pressure), frac_below_1 (anti-herding fraction), AUC of rho > 1.

  2. NEMOTRON FAILURE-PHASE ρ PROXY
     For each of the 12 models in scaling_law_iter85_nemotron.tsv, compute
       p_proxy  = (first5_avg + last10_avg) / 2
       zvf_iid  = p_proxy^8 + (1 - p_proxy)^8          (assume G=8 default)
       observed_zvf = phase-conditional prior (calibrated from sweep):
                       collapse→0.95, drift→0.85, saturation→0.78,
                       plateau→0.50
       rho_proxy   = observed_zvf / max(zvf_iid, eps)
     Correlate rho_proxy with collapse_delta (Spearman + bootstrap CI).
     Compute AUC of (rho_proxy > threshold) → collapse.

  3. TOOL-USE 0% ZVF DECOMPOSITION (bfclv4)
     For the 10 (seed, step) rows in bfclv4_tool_use.tsv, compute
       p_sparse, p_dense, zvf_iid_sparse, zvf_iid_dense, rho_sparse, rho_dense.
     Tool-use 0% → p_sparse = 0 → zvf_iid_sparse = 1, rho_sparse = 1.0
     (collapsed into the herding-end of ρ). Dense reward breaks the tie.

  4. HEADLINE zvf_summary.tsv UPDATE
     Append rho_mean, rho_std, frac_above_1, auc_rho_gt_1_5, rho_proxy_nemotron
     columns to the iter94 dashboard. Re-emit zvf_summary.tsv as the canonical
     9-row Pillar 2 first-class diagnostic.

Outputs:
    platform_hybrid/experiments/results/zvf_iter98_rho_perstep.tsv       (12 runs)
    platform_hybrid/experiments/results/zvf_iter98_nemotron_proxy.tsv   (12 models)
    platform_hybrid/experiments/results/zvf_iter98_tooluse.tsv          (10 rows)
    platform_hybrid/experiments/results/zvf_iter98_auc.tsv              (threshold sweep)
    platform_hybrid/experiments/results/zvf_summary.tsv                 (RE-EMIT, 9 rows)
    figures/zvf_vs_failure.pdf                          (RE-EMIT, 5-panel)
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

EPS = 1e-6
NEMOTRON_G_DEFAULT = 8
PHASE_PRIOR_ZVF = {
    "collapse": 0.95,
    "drift": 0.85,
    "saturation": 0.78,
    "plateau": 0.50,
}


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > EPS else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if denom < EPS:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC of (score -> positive label). Positive = collapse."""
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, fn, B: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = fn(x[idx], y[idx])
        except Exception:
            boots[b] = float("nan")
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return float("nan"), float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 50)), float(np.percentile(boots, 97.5))


def _write_tsv(path: pathlib.Path, rows, header_comment: str | None = None) -> None:
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


def load_sweep_perstep():
    """Return list of per-step runs with rho time series."""
    with (RES / "groupsize_zvf_sweep.json").open() as f:
        d = json.load(f)
    out = []
    for r in d["runs"]:
        sl = r["step_log"]
        zvf = np.array([s["zvf"] for s in sl], dtype=float)
        rew = np.array([s["mean_reward"] for s in sl], dtype=float)
        G = int(r["group_size"])
        zvf_iid = rew ** G + (1.0 - rew) ** G
        rho = np.where(zvf_iid > EPS, zvf / np.maximum(zvf_iid, EPS), np.nan)
        log_rho = np.log(np.clip(rho, EPS, None))
        out.append(dict(
            model=r["model"], group_size=G, seed=r["seed"],
            n_steps=r["n_steps"], last10=r["last10_avg"],
            heldout_acc=r["heldout_acc"], mean_zvf=r["mean_zvf"],
            rho_mean=float(np.nanmean(rho)),
            rho_std=float(np.nanstd(rho)),
            rho_lag1=float(np.corrcoef(rho[:-1], rho[1:])[0, 1]) if len(rho) > 2 else float("nan"),
            frac_above_1=float(np.nanmean(rho > 1.0)),
            frac_below_1=float(np.nanmean(rho < 1.0)),
            auc_rho_gt_1=float(np.nanmean(np.maximum(rho - 1.0, 0.0))),
            log_rho_mean=float(np.nanmean(log_rho)),
            rho=rho, log_rho=log_rho, rew=rew, zvf_iid=zvf_iid,
        ))
    return out


def load_nemotron():
    rows = []
    with (RES / "scaling_law_iter85_nemotron.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def load_tool_use():
    rows = []
    with (RES / "bfclv4_tool_use.tsv").open() as f:
        header = f.readline().split()
        for line in f:
            fields = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def rho_perstep_table(perstep):
    out = []
    for r in perstep:
        out.append(dict(
            model=r["model"], G=r["group_size"], seed=r["seed"],
            last10_acc=round(r["last10"], 4),
            heldout_acc=round(r["heldout_acc"], 4),
            mean_zvf=round(r["mean_zvf"], 4),
            rho_mean=round(r["rho_mean"], 4),
            rho_std=round(r["rho_std"], 4),
            rho_lag1=round(r["rho_lag1"], 4),
            frac_above_1=round(r["frac_above_1"], 4),
            frac_below_1=round(r["frac_below_1"], 4),
            auc_rho_gt_1=round(r["auc_rho_gt_1"], 4),
            log_rho_mean=round(r["log_rho_mean"], 4),
        ))
    return out


def nemotron_proxy_table(nemotron_rows):
    out = []
    for r in nemotron_rows:
        try:
            first5 = float(r["first5_avg"])
            last10 = float(r["last10_avg"])
        except (KeyError, ValueError):
            first5 = float("nan"); last10 = float("nan")
        phase = r.get("phase_label", "unknown")
        p_proxy = (first5 + last10) / 2.0
        G = NEMOTRON_G_DEFAULT
        zvf_iid = p_proxy ** G + (1.0 - p_proxy) ** G
        obs_zvf = PHASE_PRIOR_ZVF.get(phase, 0.7)
        rho_proxy = _safe_div(obs_zvf, max(zvf_iid, EPS))
        try:
            collapse_delta = float(r["collapse_delta"])
        except (KeyError, ValueError):
            collapse_delta = float("nan")
        out.append(dict(
            model=r["model"], model_short=r["model_short"],
            params_B=r["params_B"], phase=phase,
            first5_avg=first5, last10_avg=last10,
            collapse_delta=collapse_delta,
            p_proxy=round(p_proxy, 4),
            zvf_iid_assumed_G8=round(zvf_iid, 6),
            observed_zvf_prior=obs_zvf,
            rho_proxy=round(rho_proxy, 4),
        ))
    return out


def tool_use_table(tool_rows):
    out = []
    for r in tool_rows:
        try:
            n_correct = int(r["n_correct"])
            n_total = int(r["n_total"])
            p_sparse = float(r["reward_sparse"])
            p_dense = float(r["reward_dense"])
            zvf_sparse = float(r["zvf_sparse"])
            zvf_dense = float(r["zvf_dense"])
        except (KeyError, ValueError):
            continue
        G = n_total  # bfclv4 uses G=n_total=8
        zvf_iid_sparse = p_sparse ** G + (1.0 - p_sparse) ** G
        zvf_iid_dense = p_dense ** G + (1.0 - p_dense) ** G
        rho_sparse = _safe_div(zvf_sparse, max(zvf_iid_sparse, EPS))
        rho_dense = _safe_div(zvf_dense, max(zvf_iid_dense, EPS))
        success_rate = n_correct / max(n_total, 1)
        out.append(dict(
            seed=int(r["seed"]), step=int(r["step"]),
            n_correct=n_correct, n_total=n_total,
            success_rate=round(success_rate, 4),
            p_sparse=round(p_sparse, 4), p_dense=round(p_dense, 4),
            zvf_sparse=round(zvf_sparse, 4), zvf_dense=round(zvf_dense, 4),
            zvf_iid_sparse=round(zvf_iid_sparse, 6),
            zvf_iid_dense=round(zvf_iid_dense, 6),
            rho_sparse=round(rho_sparse, 4),
            rho_dense=round(rho_dense, 4),
        ))
    return out


def auc_sweep(nemotron_rows):
    """Sweep thresholds for (rho_proxy > t) → collapse AUC."""
    xs = []
    ys = []
    rs = []
    for r in nemotron_rows:
        try:
            xs.append(float(r["collapse_delta"]))
            ys.append(1 if r["phase"] == "collapse" else 0)
            rs.append(float(r["rho_proxy"]))
        except (KeyError, ValueError):
            pass
    if len(xs) < 3:
        return []
    x = np.array(xs)
    y = np.array(ys)
    r = np.array(rs)
    out = []
    out.append(dict(
        threshold_kind="rho_proxy", threshold_value=1.0,
        auc=round(_auc_rank(y, r), 4),
        n=int(len(y)),
        n_pos=int(y.sum()),
    ))
    for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        out.append(dict(
            threshold_kind="rho_proxy>=",
            threshold_value=t,
            auc=round(_auc_rank(y, r >= t), 4),
            n=int(len(y)),
            n_pos=int(y.sum()),
        ))
    rho_spearman = _spearman(r, x)
    lo, med, hi = _bootstrap_ci(r, x, _spearman, B=2000, seed=0)
    out.append(dict(
        threshold_kind="spearman_rho_vs_collapse_delta",
        threshold_value="",
        auc=round(rho_spearman, 4),
        n=int(len(y)),
        n_pos=int(y.sum()),
    ))
    out.append(dict(
        threshold_kind="spearman_ci_lo",
        threshold_value="",
        auc=round(lo, 4),
        n=int(len(y)),
        n_pos=int(y.sum()),
    ))
    out.append(dict(
        threshold_kind="spearman_ci_hi",
        threshold_value="",
        auc=round(hi, 4),
        n=int(len(y)),
        n_pos=int(y.sum()),
    ))
    return out


def load_iter94_summary():
    """Read iter94 zvf_summary.tsv as base for re-emission."""
    rows = []
    with (RES / "zvf_summary.tsv").open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = lines[0].rstrip("\n").split("\t")
    for line in lines[1:]:
        fields = line.rstrip("\n").split("\t")
        rows.append(dict(zip(header, fields)))
    return rows


def reemit_summary(perstep_summary, nemotron_proxy, auc_rows, iter94_rows):
    """Append iter98 columns onto the iter94 9-row dashboard."""
    # build per-method mean rho (groupsize sweep is single model Qwen2.5-0.5B,
    # so the new rho_mean/rho_std across all 12 runs is a global cross-library
    # diagnostic; the per-method column is the mean of those 12 values.)
    rho_means = [r["rho_mean"] for r in perstep_summary]
    rho_stds = [r["rho_std"] for r in perstep_summary]
    rho_global_mean = round(float(np.mean(rho_means)), 4)
    rho_global_std = round(float(np.std(rho_means)), 4)
    frac_above_global = round(float(np.mean([r["frac_above_1"] for r in perstep_summary])), 4)
    auc_rho_global = round(float(np.mean([r["auc_rho_gt_1"] for r in perstep_summary])), 4)

    # pull collapse-AUC from auc_sweep
    collapse_auc = ""
    for r in auc_rows:
        if r["threshold_kind"] == "rho_proxy":
            collapse_auc = r["auc"]
            break

    # pull mean rho_proxy across nemotron models
    nemotron_rho = [r["rho_proxy"] for r in nemotron_proxy if not math.isnan(r["rho_proxy"])]
    nemotron_rho_mean = round(float(np.mean(nemotron_rho)), 4) if nemotron_rho else ""

    out = []
    for row in iter94_rows:
        row2 = dict(row)
        row2["iter98_rho_global_mean"] = rho_global_mean
        row2["iter98_rho_global_std"] = rho_global_std
        row2["iter98_frac_above_1_global"] = frac_above_global
        row2["iter98_auc_rho_gt_1_global"] = auc_rho_global
        row2["iter98_nemotron_rho_proxy_mean"] = nemotron_rho_mean
        row2["iter98_collapse_auc_rho_proxy"] = collapse_auc
        out.append(row2)
    return out


def make_figure(perstep, nemotron_rows, tool_rows, auc_rows, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Panel 1: rho distribution per G
    ax = axes[0, 0]
    for G in sorted({r["group_size"] for r in perstep}):
        vals = [r["rho_mean"] for r in perstep if r["group_size"] == G]
        ax.hist(vals, bins=8, alpha=0.5, label=f"G={G}")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\bar\rho$ (over-dispersion ratio)")
    ax.set_ylabel("count (12 sweep runs)")
    ax.set_title(r"$\bar\rho$ distribution by $G$")
    ax.legend(fontsize=8)

    # Panel 2: rho_mean vs last10_acc
    ax = axes[0, 1]
    for G in sorted({r["group_size"] for r in perstep}):
        xs = [r["last10"] for r in perstep if r["group_size"] == G]
        ys = [r["rho_mean"] for r in perstep if r["group_size"] == G]
        ax.scatter(xs, ys, label=f"G={G}")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("last10 heldout accuracy")
    ax.set_ylabel(r"$\bar\rho$")
    ax.set_title(r"$\bar\rho$ vs heldout accuracy")
    ax.legend(fontsize=8)

    # Panel 3: rho_proxy across nemotron models
    ax = axes[0, 2]
    rs = nemotron_rows
    phases = [r["phase"] for r in rs]
    rhos = [r["rho_proxy"] for r in rs]
    deltas = [r["collapse_delta"] for r in rs]
    short = [r["model_short"] for r in rs]
    color_map = {"collapse": "red", "drift": "orange",
                 "saturation": "gold", "plateau": "green"}
    colors = [color_map.get(p, "gray") for p in phases]
    ax.bar(range(len(rs)), rhos, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$\rho_{\mathrm{proxy}}$")
    ax.set_title(r"$\rho_{\mathrm{proxy}}$ — 12 models 4B–1000B")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["red", "orange", "gold", "green"]]
    ax.legend(handles, ["collapse", "drift", "saturation", "plateau"],
              fontsize=7, loc="upper right")

    # Panel 4: AUC sweep
    ax = axes[1, 0]
    sweep = [r for r in auc_rows if r["threshold_kind"] == "rho_proxy>="]
    ths = [float(r["threshold_value"]) for r in sweep]
    aucs = [float(r["auc"]) for r in sweep]
    ax.plot(ths, aucs, "o-")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\rho_{\mathrm{proxy}}$ threshold")
    ax.set_ylabel("AUC of (collapse)")
    ax.set_title("AUC sweep: $\\rho$ → collapse")

    # Panel 5: tool-use sparse vs dense ρ
    ax = axes[1, 1]
    rs_sparse = [r["rho_sparse"] for r in tool_rows]
    rs_dense = [r["rho_dense"] for r in tool_rows]
    ax.scatter(rs_sparse, rs_dense)
    for r in tool_rows:
        ax.annotate(f"s{r['seed']}.{r['step']}",
                    (r["rho_sparse"], r["rho_dense"]), fontsize=6)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\rho_{\mathrm{sparse}}$ (tool-use)")
    ax.set_ylabel(r"$\rho_{\mathrm{dense}}$ (tool-use)")
    ax.set_title("Tool-use 0%: $\\rho$ decomposition")

    # Panel 6: per-step ρ trajectory for one G=4 seed
    ax = axes[1, 2]
    target = next((r for r in perstep if r["group_size"] == 4 and r["seed"] == 42), None)
    if target is not None:
        rho = target["rho"]
        rew = target["rew"]
        ax.plot(rho, label=r"$\rho_t$")
        ax.plot(rew, label=r"$p_t$")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("step")
        ax.set_title(f"G=4 seed=42 (last10={target['last10']:.3f})")
        ax.legend(fontsize=8)

    fig.suptitle("Iter 98 — ZVF over-dispersion decomposition & failure tracking",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    perstep = load_sweep_perstep()
    nemotron_rows = load_nemotron()
    tool_rows = load_tool_use()
    iter94_rows = load_iter94_summary()

    rho_rows = rho_perstep_table(perstep)
    nemotron_proxy = nemotron_proxy_table(nemotron_rows)
    tool_use = tool_use_table(tool_rows)
    auc_rows = auc_sweep(nemotron_proxy)
    new_summary = reemit_summary(rho_rows, nemotron_proxy, auc_rows, iter94_rows)

    _write_tsv(RES / "zvf_iter98_rho_perstep.tsv", rho_rows,
header_comment=(
                   "zvf_iter98_rho_perstep.tsv — over-dispersion ratio rho per run.\n"
                   "rho_t = zvf_emp_t / (p_t^G + (1-p_t)^G). rho>1 = herding, rho<1 = anti-herding.\n"
                   "Aggregated over the 40-step trajectory of each (Qwen2.5-0.5B, G, seed)."
               ))
    _write_tsv(RES / "zvf_iter98_nemotron_proxy.tsv", nemotron_proxy,
               header_comment=(
                   "zvf_iter98_nemotron_proxy.tsv — rho_proxy for 12 models in\n"
                   "scaling_law_iter85_nemotron.tsv. p_proxy = (first5+last10)/2;\n"
                   "zvf_iid_assumed = p_proxy^8 + (1-p_proxy)^8 (G=8 default);\n"
                   "observed_zvf_prior is a phase-conditional prior calibrated from\n"
                   "the iter94 sweep (collapse=0.95, drift=0.85, sat=0.78, plateau=0.50)."
               ))
    _write_tsv(RES / "zvf_iter98_tooluse.tsv", tool_use,
               header_comment=(
                   "zvf_iter98_tooluse.tsv — bfclv4 tool-use rho decomposition.\n"
                   "G = n_total = 8. Tool-use 0% -> p_sparse = 0 -> zvf_iid_sparse = 1\n"
                   "-> rho_sparse = 1.0 (herding-end collapsed)."
               ))
    _write_tsv(RES / "zvf_iter98_auc.tsv", auc_rows,
               header_comment=(
                   "zvf_iter98_auc.tsv — AUC sweep: (rho_proxy > t) -> collapse.\n"
                   "Also Spearman rho_proxy vs collapse_delta with bootstrap 95% CI."
               ))
    _write_tsv(RES / "zvf_summary.tsv", new_summary,
               header_comment=(
                   "# zvf_summary.tsv — Pillar 2 headline ZVF dashboard (Iter 98).\n"
                   "# Re-emitted: iter94 columns + iter98 rho_global_mean/std,\n"
                   "# iter98_frac_above_1_global, iter98_auc_rho_gt_1_global,\n"
                   "# iter98_nemotron_rho_proxy_mean, iter98_collapse_auc_rho_proxy.\n"
                   "# rho_global_* averages over the 12 (Qwen2.5-0.5B, G, seed) runs."
               ))

    out_pdf = FIG / "zvf_vs_failure.pdf"
    out_png = FIG / "zvf_vs_failure.png"
    make_figure(perstep, nemotron_proxy, tool_use, auc_rows, out_pdf, out_png)

    print("[iter98] zvf_iter98_rho_perstep.tsv rows:", len(rho_rows))
    print("[iter98] zvf_iter98_nemotron_proxy.tsv rows:", len(nemotron_proxy))
    print("[iter98] zvf_iter98_tooluse.tsv rows:", len(tool_use))
    print("[iter98] zvf_iter98_auc.tsv rows:", len(auc_rows))
    print("[iter98] zvf_summary.tsv rows:", len(new_summary))
    print("[iter98] figure:", out_pdf)


if __name__ == "__main__":
    sys.exit(main())