"""Pillar 1 iter37b -- Dynamic-only subset model selection.

Iter37's initial model-selection battery showed that linear wins on 9/12
anchors. Two of the 12 are completely flat (Qwen3-30B-MoE-Inst, Qwen3-235B-MoE;
r_var=0) and cannot, by construction, discriminate any non-linear candidate;
the remaining 7/9 "linear wins" are short-trace artefacts (n_steps<=30, where
2-parameter nonlinear fits lack curvature signal).

This follow-up filter to dynamic traces (r_var > 0.005) and reruns the same
model-selection battery. The expectation is that on the dynamic subset:

 (1) the literature's exponential saturation no longer trivially dominates;
 (2) at least one of MM / Hill provides comparable or better AIC;
 (3) the bootstrap win rate for the saturation form is well below 1.

We then test the central claim "the saturation form is identifiable on
dynamic GRPO traces" with an explicit multi-model permutation test on a
class of synthetic breakpoints.
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

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "experiments" / "results" / "scaling_law_extended_frontier.tsv"
ROOTCAUSE = REPO / "experiments" / "results" / "scaling_law_nemotron_rootcause.tsv"
FITS = REPO / "experiments" / "results" / "scaling_law_iter37_fits.tsv"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260702)
B_BOOT = 500


def model_saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def model_michaelis_menten(t, r_max, t_half):
    return r_max * t / (t_half + t)


def model_hill(t, r_max, k):
    return r_max * (t * t) / (k * k + t * t)


def model_power(t, r_max, alpha):
    return r_max * (1.0 - np.power(1.0 + t, -alpha))


def model_linear(t, a, b):
    return a + b * t


CANDIDATES = [
    {
        "name": "A_saturation_exp",
        "fn": model_saturation,
        "p0": [0.8, 0.3],
        "bounds": ([0.0, 1e-3], [2.0, 5.0]),
        "n_params": 2,
    },
    {
        "name": "B_michaelis_menten",
        "fn": model_michaelis_menten,
        "p0": [0.9, 5.0],
        "bounds": ([0.0, 1e-3], [2.0, 1e4]),
        "n_params": 2,
    },
    {
        "name": "C_hill_n2",
        "fn": model_hill,
        "p0": [0.9, 5.0],
        "bounds": ([0.0, 1e-3], [2.0, 1e4]),
        "n_params": 2,
    },
    {
        "name": "D_power_law",
        "fn": model_power,
        "p0": [0.9, 0.4],
        "bounds": ([0.0, 1e-3], [2.0, 5.0]),
        "n_params": 2,
    },
    {
        "name": "E_linear",
        "fn": model_linear,
        "p0": [0.3, 0.01],
        "bounds": ([-1.0, -1.0], [2.0, 2.0]),
        "n_params": 2,
    },
]


def aic_bic(n, k, ss_res):
    if ss_res <= 0 or not math.isfinite(ss_res):
        return float("inf"), float("inf")
    log_lik = -0.5 * n * (1.0 + math.log(2.0 * math.pi * ss_res / n))
    aic = -2.0 * log_lik + 2 * k
    bic = -2.0 * log_lik + k * math.log(n)
    return float(aic), float(bic)


def fit_candidate(t, y, cand):
    try:
        popt, _ = curve_fit(
            cand["fn"], t, y, p0=cand["p0"], bounds=cand["bounds"], maxfev=8000,
        )
    except Exception:
        return None, float("inf"), float("inf"), float("inf"), -math.inf, True
    y_hat = cand["fn"](t, *popt)
    resid = y - y_hat
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    aic, bic = aic_bic(len(y), cand["n_params"], ss_res)
    return popt, ss_res, aic, bic, r2, False


def synth_trace(r, rc_lookup):
    n = int(r["n_steps"])
    peak = int(float(rc_lookup[r["model"]]["peak_step"])) if r["model"] in rc_lookup else (
        1 if abs(r["r_peak"] - r["r_first"]) < 0.05 else max(1, n // 2)
    )
    if peak < 1:
        peak = 1
    if peak > n - 1:
        peak = n - 1
    peak_val = r["r_peak"]
    early, late, mean, zf = r["early_mean"], r["late_mean"], r["r_mean"], r["zero_frac"]
    t = np.arange(1, n + 1, dtype=float)
    out = np.linspace(early, late, n)
    out[peak - 1] = max(out[peak - 1], peak_val)
    if peak - 2 >= 0:
        out[peak - 2] = max(out[peak - 2], 0.5 * (out[peak - 1] + out[peak]))
    if peak < n:
        out[peak] = max(out[peak], 0.5 * (out[peak - 1] + out[peak + 1]))
    n_zero = int(round(zf * n))
    if n_zero > 0 and r["model"] == "Nemotron-120B":
        out[:n_zero] = 0.0
        if n - 1 > peak:
            out[(n_zero + peak) // 2] = 0.0
    cur = float(np.mean(out))
    if cur > 1e-9:
        out = out * (mean / cur)
    out = np.clip(out, 0.0, 1.0)
    return t, out, peak


def main() -> None:
    with open(DATA) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    with open(ROOTCAUSE) as f:
        rc_lookup = {r["model"]: r for r in csv.DictReader(f, delimiter="\t")}

    # ---- dynamic-only filter (r_var > 0.005) ----
    var_threshold = 0.005
    dynamic_rows = [r for r in rows if r["r_var"] > var_threshold]
    flat_rows = [r for r in rows if r["r_var"] <= var_threshold]
    print(f"Total anchors: {len(rows)}, dynamic (r_var>{var_threshold}): "
          f"{len(dynamic_rows)}, flat: {len(flat_rows)}")

    dynamic_rows.sort(key=lambda r: -r["params_B"])

    # ---- per-anchor × per-candidate fits on dynamic subset ----
    fits_rows = []
    aic_rows = []
    for r in dynamic_rows:
        t, y, _ = synth_trace(r, rc_lookup)
        per_model = {"model": r["model"], "params_B": r["params_B"],
                     "arch": r["arch"], "n_steps": int(r["n_steps"]),
                     "r_var": r["r_var"]}
        per_cand = {}
        for cand in CANDIDATES:
            popt, ss_res, aic, bic, r2, hit = fit_candidate(t, y, cand)
            per_cand[cand["name"]] = {
                "params": popt, "ss_res": ss_res, "aic": aic, "bic": bic,
                "r2": r2, "hit": hit,
            }
            params = (
                {f"param_{k_}": float(p) for k_, p in zip(["p1", "p2"], popt)}
                if popt is not None
                else {"param_p1": float("nan"), "param_p2": float("nan")}
            )
            row = dict(per_model)
            row["model_name"] = cand["name"]
            row.update(params)
            row["ss_res"] = round(ss_res, 6)
            row["r2"] = round(r2, 4) if math.isfinite(r2) else float("nan")
            row["aic"] = round(aic, 4) if math.isfinite(aic) else float("inf")
            row["bic"] = round(bic, 4) if math.isfinite(bic) else float("inf")
            row["hit_bound"] = hit
            fits_rows.append(row)

        aics = np.array([per_cand[c["name"]]["aic"] for c in CANDIDATES], dtype=float)
        finite = np.isfinite(aics)
        if not finite.any():
            best = "NONE"
            second = "NONE"
            delta_best_second = float("nan")
            delta_best_worst = float("nan")
        else:
            sorted_idx = np.argsort(np.where(finite, aics, np.inf))
            best = CANDIDATES[sorted_idx[0]]["name"]
            second = CANDIDATES[sorted_idx[1]]["name"]
            delta_best_second = float(aics[sorted_idx[1]] - aics[sorted_idx[0]])
            delta_best_worst = float(aics[sorted_idx[-1]] - aics[sorted_idx[0]])
        # Akaike weights within all 5 (treat infinite as -large penalty below)
        delta = aics - np.where(finite, aics, np.inf).min()
        # mask out infeasible ones
        delta_clip = np.where(np.isfinite(delta), delta, np.inf)
        # softmax-style:
        exp_neg_half = np.exp(-0.5 * np.clip(delta_clip, 0, 700))
        # infeasible get weight 0
        exp_neg_half[~finite] = 0.0
        w = exp_neg_half / max(exp_neg_half.sum(), 1e-12)
        w_sat = float(w[CANDIDATES.index(
            next(c for c in CANDIDATES if c["name"] == "A_saturation_exp"))])
        aic_rows.append({
            "model": per_model["model"],
            "params_B": per_model["params_B"],
            "arch": per_model["arch"],
            "r_var": round(r["r_var"], 6),
            "best_aic_model": best,
            "second_aic_model": second,
            "delta_aic_best_second": round(delta_best_second, 4),
            "delta_aic_best_worst": round(delta_best_worst, 4),
            "w_saturation": round(w_sat, 4),
            "w_michaelis_menten": round(float(w[1]), 4),
            "w_hill_n2": round(float(w[2]), 4),
            "w_power_law": round(float(w[3]), 4),
            "w_linear": round(float(w[4]), 4),
        })

    # ---- (2) bootstrap AIC selection rate on dynamic subset ----
    # We draw a random sample of anchors (with replacement) and refit; the
    # within-bootstrap "best AIC winner" is recorded.
    boot_rows = []
    n_dyn = len(dynamic_rows)
    rc_models = list(rc_lookup.keys())
    seeds_per_anchor = []
    n_obs_per_anchor = []
    for r in dynamic_rows:
        t, y, _ = synth_trace(r, rc_lookup)
        seeds_per_anchor.append((t, y, r))
        n_obs_per_anchor.append(len(y))
    for _ in range(B_BOOT):
        # resample anchors (with replacement, n=n_dyn)
        idx = RNG.integers(0, n_dyn, size=n_dyn)
        # concatenate traces (preserves model identity)
        t_concat = np.concatenate([seeds_per_anchor[i][0] for i in idx])
        # remap t to be globally increasing by anchor block
        # easier: refit on each resampled anchor separately, take modal winner
        winner_idx = []
        for i in idx:
            t_i, y_i, r_i = seeds_per_anchor[i]
            y_noisy = np.clip(
                y_i + RNG.normal(0, 0.10, size=len(y_i)), 0, 1
            )
            aics_i = []
            for cand in CANDIDATES:
                _, _, aic_i, _, _, _ = fit_candidate(t_i, y_noisy, cand)
                aics_i.append(aic_i)
            aics_i = np.array(aics_i, dtype=float)
            if not np.any(np.isfinite(aics_i)):
                continue
            winner_idx.append(int(np.argmin(np.where(
                np.isfinite(aics_i), aics_i, np.inf
            ))))
        if not winner_idx:
            continue
        winner_names = [CANDIDATES[w]["name"] for w in winner_idx]
        win_share = {c["name"]: sum(1 for w in winner_names if w == c["name"]) / len(winner_names) for c in CANDIDATES}
        boot_rows.append({
            "n_anchors_resampled": n_dyn,
            "boot_n_resampled_traces": len(winner_names),
            **win_share,
        })
    # aggregate bootstrap win shares across B_BOOT replicates
    if boot_rows:
        boot_summary = {
            c["name"]: round(float(np.mean([b[c["name"]] for b in boot_rows])), 4)
            for c in CANDIDATES
        }
        sat_wins_share = sum(1 for b in boot_rows
                             if b["A_saturation_exp"] > max(
                                 v for k, v in b.items() if k.startswith("B_") or
                                 k.startswith("C_") or k.startswith("D_") or
                                 k.startswith("E_")
                             )) / len(boot_rows)
    else:
        boot_summary = {c["name"]: 0.0 for c in CANDIDATES}
        sat_wins_share = 0.0

    # ---- (3) Central permutation test ----
    # H0: AICs of {A_saturation_exp, B_michaelis_menten, C_hill_n2, D_power_law}
    # are equally informative on dynamic traces (i.e. w_sat ~ 0.25). We compute
    # w_sat under the observed ordering of deltas, then shuffle the deltas
    # across models and recompute. The observed w_sat vs the null distribution
    # gives a clean p-value for "saturation is uniquely informative".
    w_sat_obs = float(np.mean([r["w_saturation"] for r in aic_rows]))
    B_PERM = 2000
    w_sat_perm = []
    w_arr = np.array([[r["w_saturation"], r["w_michaelis_menten"],
                        r["w_hill_n2"], r["w_power_law"], r["w_linear"]] for r in aic_rows])
    for _ in range(B_PERM):
        # permute columns to break the "this column = A_saturation" identity
        perm = RNG.permutation(5)
        permuted = w_arr[:, perm]
        # pick column that lands in position 0 (saturation slot)
        w_sat_perm.append(float(np.mean(permuted[:, 0])))
    w_sat_perm = np.array(w_sat_perm)
    p_sat_uniquely_better = float(np.mean(w_sat_perm >= w_sat_obs))

    # ---- summary ----
    summary = {
        "n_dynamic": len(dynamic_rows),
        "var_threshold": var_threshold,
        "n_candidates": len(CANDIDATES),
        "n_anchors_where_sat_wins_aic": sum(1 for r in aic_rows
                                             if r["best_aic_model"] == "A_saturation_exp"),
        "n_anchors_where_mm_wins_aic": sum(1 for r in aic_rows
                                            if r["best_aic_model"] == "B_michaelis_menten"),
        "n_anchors_where_hill_wins_aic": sum(1 for r in aic_rows
                                              if r["best_aic_model"] == "C_hill_n2"),
        "n_anchors_where_power_wins_aic": sum(1 for r in aic_rows
                                               if r["best_aic_model"] == "D_power_law"),
        "n_anchors_where_linear_wins_aic": sum(1 for r in aic_rows
                                                if r["best_aic_model"] == "E_linear"),
        "mean_w_saturation_dynamic": round(float(np.mean([r["w_saturation"] for r in aic_rows])), 4),
        "mean_w_michaelis_menten_dynamic": round(float(np.mean([r["w_michaelis_menten"] for r in aic_rows])), 4),
        "mean_w_hill_dynamic": round(float(np.mean([r["w_hill_n2"] for r in aic_rows])), 4),
        "mean_w_power_dynamic": round(float(np.mean([r["w_power_law"] for r in aic_rows])), 4),
        "mean_w_linear_dynamic": round(float(np.mean([r["w_linear"] for r in aic_rows])), 4),
        "median_delta_aic_best_second": round(float(np.median([r["delta_aic_best_second"] for r in aic_rows])), 4),
        "bootstrap_win_share_saturation": boot_summary["A_saturation_exp"],
        "bootstrap_win_share_mm": boot_summary["B_michaelis_menten"],
        "bootstrap_win_share_hill": boot_summary["C_hill_n2"],
        "bootstrap_win_share_power": boot_summary["D_power_law"],
        "bootstrap_win_share_linear": boot_summary["E_linear"],
        "bootstrap_share_sat_wins_each_replicate": round(sat_wins_share, 4),
        "permutation_p_saturation_uniquely_better": p_sat_uniquely_better,
        "n_perm_replicates": B_PERM,
    }

    # ---- write outputs ----
    out_files = {
        "scaling_law_iter37b_fits.tsv": fits_rows,
        "scaling_law_iter37b_aic.tsv": aic_rows,
    }
    for fname, drows in out_files.items():
        path = RESULTS / fname
        with open(path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(drows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(drows)
        print(f"wrote {path}  ({len(drows)} rows)")
    with open(RESULTS / "scaling_law_iter37b_summary.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        for k, v in summary.items():
            w.writerow([k, v])
    print(f"wrote {RESULTS / 'scaling_law_iter37b_summary.tsv'}")

    # ---- figure: Akaike-weight heat-map + bootstrap histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    w_mat = np.array([
        [r["w_saturation"], r["w_michaelis_menten"], r["w_hill_n2"],
         r["w_power_law"], r["w_linear"]] for r in aic_rows
    ])
    im = ax.imshow(w_mat.T, cmap="viridis", aspect="auto", vmin=0, vmax=0.6)
    ax.set_yticks(range(5))
    ax.set_yticklabels([c["name"].split("_", 1)[1] for c in CANDIDATES], fontsize=9)
    ax.set_xticks(range(len(aic_rows)))
    short = [m.replace("-Instruct", "-Inst") for m in [r["model"] for r in aic_rows]]
    ax.set_xticklabels([f"{s}\n{r['params_B']:.0f}B" for s, r in zip(short, aic_rows)],
                       rotation=0, fontsize=8)
    ax.set_title("Iter37b -- Akaike weights across the dynamic frontier set")
    fig.colorbar(im, ax=ax, label="weight", shrink=0.85)

    ax2 = axes[1]
    win_arr = np.array([[b[c["name"]] for b in boot_rows] for c in CANDIDATES])
    short_names = [c["name"].split("_", 1)[1] for c in CANDIDATES]
    bp = ax2.boxplot(win_arr.T,
                     tick_labels=short_names,
                     patch_artist=True,
                     boxprops=dict(facecolor="#ccebc5", lw=0.5),
                     medianprops=dict(color="#2b8cbe", lw=1.2),
                     flierprops=dict(marker="o", ms=3, mfc="grey"),
                     whiskerprops=dict(lw=0.7),
                     capprops=dict(lw=0.7),
                     widths=0.55)
    ax2.set_xticklabels(short_names, rotation=20, ha="right", fontsize=8)
    ax2.axhline(0.25, ls="--", color="red", lw=0.8, label="uniform=1/4")
    ax2.set_ylabel("AIC winner share per bootstrap replicate")
    ax2.set_title(f"Iter37b -- Bootstrap AIC winner share (B={B_BOOT})")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"scaling_law_iter37b.{ext}", bbox_inches="tight")
        fig.savefig(PAPER_FIG / f"scaling_law_iter37b.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/scaling_law_iter37b.{{pdf,png}}")

    # ---- console ----
    print("\n=== Iter 37b summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== Per-anchor AIC winner (dynamic only) ===")
    for r in aic_rows:
        print(f"  {r['model']:30s}  best={r['best_aic_model']:25s} "
              f"second={r['second_aic_model']:25s} "
              f"delta={r['delta_aic_best_second']:.2f}")


if __name__ == "__main__":
    main()
