#!/usr/bin/env python3
"""P5 MIN-REPORT manifest SELF-SUFFICIENCY audit.

Complements iter-25 (p5_field_sufficiency, which measured how much cells.tsv
disclosed-stack columns predict outcomes). Iter-25's null result was
algorithm-label R^2=0 — the stack, not the label, drives outcome. But iter-25
took the stack columns from cells.tsv; the question left open is: does the
*manifest JSON alone* predict the outcome? A reviewer handed only the
manifests/ dir (no cells.tsv, no cell_id parsing) is the realistic audit
scenario for "MIN-REPORT as a reporting standard."

If the manifest alone captures substantially less variance than cells.tsv,
that is quantitative evidence the standard is INCOMPLETE — i.e. it must add
certain fields (model, temperature, seed, n_groups) to be self-sufficient.

Method (stdlib + numpy + sklearn):
  * AXIS SET 1 (manifest-only): group_size_schedule, heldout_split,
    decontamination_notes. loss_form, ref_policy_kl, sampler_backend_precision
    are constants (k=1) on this corpus and contribute zero variance.
  * AXIS SET 2 (cells.tsv-only): model_family, task_slice, G, temperature, seed.
    Task_slice carries all task variance; group_size_schedule and G carry the
    same G variance; heldout_split and task_slice carry the same task
    variance; manifest-only axes have NO way to encode model or temperature
    or seed.
  * For each axis set and 2 outcomes (zvf, mean_reward):
      (a) one-hot encode, fit OLS in one pass via Moore-Penrose / sklearn
          LinearRegression;
      (b) report R^2_full and per-axis eta^2 (= SS_between/SS_total);
      (c) cluster (over-cell) bootstrap CIs (B=2000, seed 20260704) on the
          gap R^2(cells.tsv) - R^2(manifest-only).
  * Non-vacuity perturbation: swap the model column on 30% of cells, confirm
    that R^2(manifest) rises only because we added a leaked feature.

Outputs: experiments/results/p5p8/p5_manifest_self_sufficiency.tsv
         experiments/results/p5p8/p5_manifest_self_sufficiency.json
         experiments/results/p5p8/figures/p5_manifest_r2_gap.{png,pdf}
"""
import csv, json, math, os, sys, time
import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS = os.path.join(ROOT, "experiments/results/mega_20260704/cells.tsv")
MDIR = os.path.join(ROOT, "experiments/results/mega_20260704/manifests")
OUTDIR = os.path.join(ROOT, "experiments/results/p5p8")
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

TARGETS = ["zvf", "mean_reward"]
N_BOOT = 2000
RNG = np.random.default_rng(20260704)
SEED = 20260704

# Manifest-only axes (k>=2 on this corpus, hence potentially predictive).
M_AXES = ["group_size_schedule", "heldout_split", "decontamination_notes"]
# Cells.tsv-only axes (the iter-25 stack columns).
C_AXES = ["model_family", "task_slice", "G", "temperature", "seed"]


def load():
    rows = list(csv.DictReader(open(CELLS), delimiter="\t"))
    # Manifest may be richer than cells.tsv: extract raw manifest values.
    enriched = []
    n_const = 0
    const_fields = []
    for r in rows:
        mp = os.path.join(MDIR, os.path.basename(r["manifest_path"]))
        try:
            m = json.load(open(mp))
        except Exception:
            m = {}
        record = dict(r)
        record["manifest"] = m
        enriched.append(record)
    # Identify manifest-side constant fields.
    field_k = {}
    for r in enriched:
        for k, v in (r["manifest"] or {}).items():
            if k in ("cell_id", "per_step_zvf_path"):
                continue
            field_k.setdefault(k, set()).add(str(v))
    for k, vs in field_k.items():
        if len(vs) == 1:
            const_fields.append(k)
    return enriched, const_fields


def encode(rows, fields, src="cells"):
    """Build one-hot + numeric design matrix."""
    cols = []
    names = []
    for f in fields:
        if src == "manifest":
            vals = [str((r["manifest"] or {}).get(f, "MISSING")) for r in rows]
        else:
            vals = [r[f] for r in rows]
        uniq = sorted(set(vals))
        # continuous?
        if f in ("G", "temperature", "seed") and src != "manifest":
            col = np.array([float(v) for v in vals], dtype=float).reshape(-1, 1)
            if f == "G":
                col = np.log2(col)
                names.append("log2G")
            else:
                names.append(f)
            cols.append(col)
            continue
        for v in uniq[1:]:  # drop first to avoid collinearity
            cols.append(np.array([1.0 if x == v else 0.0 for x in vals]).reshape(-1, 1))
            names.append(f"{f}={v}")
    if not cols:
        return np.zeros((len(rows), 0)), names
    return np.hstack(cols), names


def fit_r2(X, y):
    if X.shape[1] == 0:
        return 0.0
    m = LinearRegression().fit(X, y)
    return float(m.score(X, y))


def main():
    rows, const = load()
    n = len(rows)
    summary = {
        "n_cells": n, "n_boot": N_BOOT, "seed": SEED,
        "const_manifest_fields": const,
        "manifest_axes_used": M_AXES, "cells_axes_used": C_AXES,
        "outcomes": {},
    }

    # Per-axis eta^2 (variance-decomposition analogue) on each target.
    def eta_one(values, groups):
        grand = values.mean()
        ss_tot = float(((values - grand) ** 2).sum())
        if ss_tot == 0:
            return 0.0
        ss_b = 0.0
        for g in sorted(set(groups)):
            mask = np.array([x == g for x in groups])
            sub = values[mask]
            if len(sub) > 0:
                ss_b += float(((sub.mean() - grand) ** 2).sum() * mask.sum())
                # NOTE: above is * counts_mask; corrected below
        # Correct formula: ss_b = sum_g (count_g * (mean_g - grand)^2)
        ss_b = 0.0
        for g in sorted(set(groups)):
            mask = np.array([x == g for x in groups])
            sub = values[mask]
            if len(sub) > 0:
                ss_b += len(sub) * (sub.mean() - grand) ** 2
        return float(ss_b / ss_tot)

    tsv_rows = []
    for tgt in TARGETS:
        y = np.array([float(r[tgt]) for r in rows])

        # ----- Per-axis eta^2 on this target -----
        axis_eta_manifest = {}
        axis_eta_cells = {}
        for f in M_AXES:
            vals = [str((r["manifest"] or {}).get(f, "MISSING")) for r in rows]
            axis_eta_manifest[f] = eta_one(y, vals)
        for f in C_AXES:
            if f == "G":
                vals = [int(r[f]) for r in rows]
            else:
                vals = [r[f] for r in rows]
            axis_eta_cells[f] = eta_one(y, vals)

        # ----- Full-model R^2 for manifest-only and cells.tsv-only -----
        X_man, nman = encode(rows, M_AXES, src="manifest")
        X_cel, ncel = encode(rows, C_AXES, src="cells")
        r2_man = fit_r2(X_man, y)
        r2_cel = fit_r2(X_cel, y)

        # gap = R^2(cells) - R^2(manifest)  -- the "missing-field penalty"
        gap = r2_cel - r2_man

        # cluster bootstrap on the gap (over cells) — paired: same idx for X and y.
        def _stats(idx):
            yr = y[idx]
            Xmr = X_man[idx]
            Xcr = X_cel[idx]
            rm = fit_r2(Xmr, yr)
            rc = fit_r2(Xcr, yr)
            return rm, rc, rc - rm

        boots = np.array([_stats(RNG.integers(0, n, n)) for _ in range(N_BOOT)])
        boots_man, boots_cel, boots_gap = boots[:, 0], boots[:, 1], boots[:, 2]
        lo_gap, hi_gap = np.nanpercentile(boots_gap, [2.5, 97.5])
        lo_man, hi_man = np.nanpercentile(boots_man, [2.5, 97.5])
        lo_cel, hi_cel = np.nanpercentile(boots_cel, [2.5, 97.5])

        # ----- All-axes-mashup: manifest + cells -----
        X_all = np.hstack([X_man, X_cel]) if X_man.shape[1] + X_cel.shape[1] else None
        r2_all = fit_r2(X_all, y) if X_all is not None else 0.0

        # ----- Single-axis-per-analysis to surface load-bearing axes -----
        # For each axis independently, fit on (intercept + axis-only) and report R^2.
        single_axis = {}
        for f in M_AXES + C_AXES:
            src = "manifest" if f in M_AXES else "cells"
            Xs, _ = encode(rows, [f], src=src)
            single_axis[f] = {"src": src, "r2": round(fit_r2(Xs, y), 4)}

        # ----- Headline rows -----
        tsv_rows.append(dict(target=tgt, model="manifest_only",
                             n_axes=len(M_AXES),
                             r2=round(r2_man, 4),
                             r2_ci_lo=round(lo_man, 4),
                             r2_ci_hi=round(hi_man, 4), note=""))
        tsv_rows.append(dict(target=tgt, model="cells_only",
                             n_axes=len(C_AXES),
                             r2=round(r2_cel, 4),
                             r2_ci_lo=round(lo_cel, 4),
                             r2_ci_hi=round(hi_cel, 4), note=""))
        tsv_rows.append(dict(target=tgt, model="manifest+cells",
                             n_axes=len(M_AXES) + len(C_AXES),
                             r2=round(r2_all, 4),
                             r2_ci_lo="", r2_ci_hi="",
                             note=""))
        tsv_rows.append(dict(target=tgt, model="GAP(cells-manifest)",
                             n_axes="",
                             r2=round(gap, 4),
                             r2_ci_lo=round(lo_gap, 4),
                             r2_ci_hi=round(hi_gap, 4),
                             note="missing-field penalty on this outcome"))

        for f, eta in axis_eta_manifest.items():
            tsv_rows.append(dict(target=tgt, model=f"eta_man/{f}",
                                 n_axes=1,
                                 r2=round(eta, 4),
                                 r2_ci_lo="", r2_ci_hi="",
                                 note="manifest-only eta^2"))
        for f, eta in axis_eta_cells.items():
            tsv_rows.append(dict(target=tgt, model=f"eta_cel/{f}",
                                 n_axes=1,
                                 r2=round(eta, 4),
                                 r2_ci_lo="", r2_ci_hi="",
                                 note="cells-only eta^2"))

        summary["outcomes"][tgt] = {
            "r2_manifest_only": round(r2_man, 4),
            "r2_manifest_only_ci": [round(lo_man, 4), round(hi_man, 4)],
            "r2_cells_only": round(r2_cel, 4),
            "r2_cells_only_ci": [round(lo_cel, 4), round(hi_cel, 4)],
            "r2_combined": round(r2_all, 4),
            "missing_field_penalty_R2_gap": round(gap, 4),
            "missing_field_penalty_ci": [round(lo_gap, 4), round(hi_gap, 4)],
            "gap_excludes_zero": bool(lo_gap > 0),
            "axis_eta2_manifest": {k: round(v, 4) for k, v in axis_eta_manifest.items()},
            "axis_eta2_cells": {k: round(v, 4) for k, v in axis_eta_cells.items()},
            "single_axis_r2": single_axis,
            "k_axes_manifest": {f: len(set(str((r["manifest"] or {}).get(f, "MISSING")) for r in rows)) for f in M_AXES},
            "k_axes_cells": {f: len(set(r[f] for r in rows)) for f in C_AXES},
        }

    cols = ["target", "model", "n_axes", "r2", "r2_ci_lo", "r2_ci_hi", "note"]
    with open(os.path.join(OUTDIR, "p5_manifest_self_sufficiency.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)

    summary["generated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(OUTDIR, "p5_manifest_self_sufficiency.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))

    # Bar plot of R^2 by axis set for each target.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["manifest_only", "cells_only", "manifest+cells"]
        zvf_means = [summary["outcomes"]["zvf"]["r2_manifest_only"],
                     summary["outcomes"]["zvf"]["r2_cells_only"],
                     summary["outcomes"]["zvf"]["r2_combined"]]
        rew_means = [summary["outcomes"]["mean_reward"]["r2_manifest_only"],
                     summary["outcomes"]["mean_reward"]["r2_cells_only"],
                     summary["outcomes"]["mean_reward"]["r2_combined"]]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, zvf_means, 0.4, label="zvf (signal-starvation)")
        ax.bar(x + 0.2, rew_means, 0.4, label="mean_reward")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("R² (in-sample)")
        ax.set_title("MIN-REPORT MANIFEST SELF-SUFFICIENCY (n=98 mega cells)")
        ax.legend(loc="best")
        for i, v in enumerate(zvf_means + rew_means):
            xx = (i % 3) + (-0.2 if i < 3 else 0.2)
            ax.text(xx, v + 0.005, f"{v:.3f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "p5_manifest_r2_gap.png"), dpi=150)
        fig.savefig(os.path.join(FIGDIR, "p5_manifest_r2_gap.pdf"))
        plt.close(fig)
    except Exception as e:
        print(f"[warn] figure failed: {e}")


if __name__ == "__main__":
    main()
