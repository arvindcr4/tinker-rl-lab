#!/usr/bin/env python3
"""P5 MIN-REPORT stratified coverage audit (ledger item 28, iter 21).

For every (MIN-REPORT item x stack-axis) cell on the 98 mega_20260704
manifests, computes eta^2 of per-cell item_score + concrete% + max-min
contrast, plus reward-quartile and ZVF-quartile strata. Identifies whether
the manifest emitter is stack-invariant.

Reuses iter-18's auditor output: experiments/results/p5p8/minreport_audit.tsv.

Headline falsifiable claim: items 1-6 are perfectly stack-invariant
(every cell in every stratum gets the same per-item score -> eta^2 is
NaN by zero-within-variance). Item 7 is the only item with non-zero
stack-axis eta^2 (task_slice = 1.000, contrast 3.33pp; reward/ZVF
quartile correlations are confound artifacts because reward and ZVF
are themselves task-stratified). The "add the missing fields" work-list
is therefore corpus-wide, not stack-conditional.

Output:
  experiments/results/p5p8/minreport_stratified.tsv
  experiments/results/p5p8/minreport_stratified_summary.json
  experiments/results/p5p8/figures/minreport_stratified_heatmap.{png,pdf}
  experiments/results/p5p8/figures/minreport_stratified_contrast.{png,pdf}
"""
from __future__ import annotations
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P5P8 = ROOT / "experiments" / "results" / "p5p8"
MEGA = ROOT / "experiments" / "results" / "mega_20260704"
CELLS = MEGA / "cells.tsv"
MANIFESTS = MEGA / "manifests"
AUDIT_TSV = P5P8 / "minreport_audit.tsv"
OUT_TSV = P5P8 / "minreport_stratified.tsv"
OUT_JSON = P5P8 / "minreport_stratified_summary.json"
FIG = P5P8 / "figures"

ITEM_NAMES = {1: "loss_form", 2: "ref_policy_kl", 3: "sampler_backend",
              4: "per_step_zvf", 5: "group_size_schedule", 6: "heldout_split",
              7: "decontam_parser"}
ITEM_COLS = {1: "item1_loss", 2: "item2_kl", 3: "item3_backend", 4: "item4_zvf",
             5: "item5_G", 6: "item6_heldout", 7: "item7_decontam"}
STACK_AXES = ["model_family", "task_slice", "G", "temperature", "seed"]
ALL_AXES = STACK_AXES + ["reward_quartile", "zvf_quartile"]
ITEM_KEYS = {1: "loss_form", 2: "ref_policy_kl", 3: "sampler_backend_precision",
             4: "per_step_zvf_path", 5: "group_size_schedule", 6: "heldout_split",
             7: "decontamination_notes"}
VALIDATORS = {1: [r"^(grpo|gspo|dapo|drgrpo|dpo|sequence|ppo|sft|n/a-sampling)$"],
              2: [r"^(kl-[a-z]+(\d+(\.\d+)?)?|kl-est-[a-z]+|no-kl|n/a(?:-[a-z]+)?)$"],
              3: [r"^(tinker-closed|vllm|sglang|hf|trtllm|openai|anthropic)[-@a-zA-Z0-9._/]*$"],
              4: [r"/.*\.json$"],
              5: [r"^(fixed-G=\d+|adaptive[-+a-zA-Z0-9=<>]*|escalating|decaying)$"],
              6: [r"^[a-z0-9_]+$"], 7: [r".*"]}


def eta2(vals, groups):
    """eta^2 = SS_between/SS_total. NaN if SS_total == 0."""
    pairs = [(v, g) for v, g in zip(vals, groups)
             if v is not None and g is not None
             and not (isinstance(v, float) and math.isnan(v))]
    if len(pairs) < 2: return float("nan")
    flat = [v for v, _ in pairs]
    g_mean = sum(flat) / len(flat)
    ss_t = sum((v - g_mean) ** 2 for v in flat)
    by_g = defaultdict(list)
    for v, g in pairs: by_g[g].append(v)
    ss_a = sum(len(vs) * (sum(vs)/len(vs) - g_mean) ** 2 for vs in by_g.values() if vs)
    if ss_t <= 1e-12: return float("nan")
    return ss_a / ss_t


def load_cells():
    out = {}
    with CELLS.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[r["cell_id"]] = {
                "model_family": r["model_family"], "task_slice": r["task_slice"],
                "G": int(r["G"]), "temperature": float(r["temperature"]),
                "seed": int(r["seed"]),
                "mean_reward": float(r["mean_reward"]), "zvf": float(r["zvf"]),
            }
    return out


def load_manifests(cells):
    out = {}
    for cid, meta in cells.items():
        for p in [(MEGA / "manifests" / f"{cid}.json")]:
            if p.is_file():
                try:
                    with p.open() as f: m = json.load(f)
                    m["_meta"] = meta
                    out[cid] = m
                except Exception: pass
                break
    return out


def coverage_status(m, it):
    v = m.get(ITEM_KEYS[it])
    if v is None or (isinstance(v, str) and v.strip() == ""): return "absent"
    s = str(v)
    if s.lower().startswith("n/a"): return "n/a"
    return "concrete" if any(re.match(p, s, re.IGNORECASE) for p in VALIDATORS[it]) else "invalid"


def axis_value(meta, axis, r_q, z_q):
    if axis == "model_family": return meta["model_family"]
    if axis == "task_slice":   return meta["task_slice"]
    if axis == "G":            return meta["G"]
    if axis == "temperature":  return meta["temperature"]
    if axis == "seed":         return meta["seed"]
    if axis == "reward_quartile":
        v = meta["mean_reward"]
        return "Q1_lo" if v <= r_q[0] else "Q2_midlo" if v <= r_q[1] else "Q3_midhi" if v <= r_q[2] else "Q4_hi"
    if axis == "zvf_quartile":
        v = meta["zvf"]
        return "Q1_lo" if v <= z_q[0] else "Q2_midlo" if v <= z_q[1] else "Q3_midhi" if v <= z_q[2] else "Q4_hi"
    return None


def qcut(sv, q):
    return sv[max(0, min(len(sv)-1, int(len(sv) * q)))]


def main():
    cells = load_cells()
    manifests = load_manifests(cells)
    if not manifests:
        print("no manifests loaded", file=sys.stderr); return 1
    n = len(manifests)
    print(f"loaded {n} mega manifests")

    auditor = {}
    with AUDIT_TSV.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["cell_id"] in manifests: auditor[r["cell_id"]] = r

    all_rewards = sorted(m["_meta"]["mean_reward"] for m in manifests.values())
    all_zvfs = sorted(m["_meta"]["zvf"] for m in manifests.values())
    r_q = [qcut(all_rewards, 0.25), qcut(all_rewards, 0.5), qcut(all_rewards, 0.75)]
    z_q = [qcut(all_zvfs, 0.25), qcut(all_zvfs, 0.5), qcut(all_zvfs, 0.75)]

    # bucket[(it, axis)][ax_v] = (list of item_scores, list of concrete 0/1)
    bucket = defaultdict(lambda: defaultdict(lambda: [[], []]))
    per_item_overall = {it: defaultdict(int) for it in range(1, 8)}
    for cid, m in manifests.items():
        meta = m["_meta"]; aud = auditor.get(cid)
        if aud is None: continue
        for it in range(1, 8):
            try: score = float(aud[ITEM_COLS[it]])
            except (KeyError, ValueError): continue
            per_item_overall[it][coverage_status(m, it)] += 1
            is_concrete = 1 if coverage_status(m, it) == "concrete" else 0
            for axis in ALL_AXES:
                ax = axis_value(meta, axis, r_q, z_q)
                if ax is None: continue
                bucket[(it, axis)][ax][0].append(score)
                bucket[(it, axis)][ax][1].append(is_concrete)

    # per (item, axis) -> {ax_v -> {n, mean, std, concrete_pct, eta2_score}}
    coverage, eta2_score, contrast_score = {}, {}, {}
    axis_values_seen = {ax: set() for ax in ALL_AXES}
    for it in range(1, 8):
        for axis in ALL_AXES:
            tbl = {}
            for ax_v, (scores, concretes) in bucket[(it, axis)].items():
                axis_values_seen[axis].add(ax_v)
                m_score = sum(scores) / len(scores) if scores else float("nan")
                std = (sum((s - m_score)**2 for s in scores) / max(1, len(scores)-1)) ** 0.5 if scores else 0
                tbl[str(ax_v)] = {
                    "n": len(scores),
                    "mean_item_score": round(m_score, 3),
                    "std_item_score": round(std, 3),
                    "concrete_pct": round(100.0 * sum(concretes) / max(1, len(concretes)), 2),
                }
            coverage[(it, axis)] = tbl
            all_s, all_g = [], []
            for ax_v, (scores, _) in bucket[(it, axis)].items():
                all_s.extend(scores); all_g.extend([ax_v]*len(scores))
            eta2_score[(it, axis)] = eta2(all_s, all_g)
            means = [t["mean_item_score"] for t in tbl.values()]
            contrast_score[(it, axis)] = round(max(means) - min(means), 3) if means else float("nan")

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item_no", "item_name", "axis", "axis_value", "n",
                    "mean_item_score", "std_item_score", "concrete_pct",
                    "eta2_axis_item_score", "contrast_score"])
        for it in range(1, 8):
            for axis in ALL_AXES:
                e2s = eta2_score[(it, axis)]
                for ax_v in sorted(axis_values_seen[axis],
                                    key=lambda x: (0, float(x)) if str(x).replace('.', '').replace('-', '').isdigit() else (1, str(x))):
                    tab = coverage[(it, axis)].get(str(ax_v))
                    if tab is None: continue
                    w.writerow([it, ITEM_NAMES[it], axis, ax_v, tab["n"],
                                tab["mean_item_score"], tab["std_item_score"],
                                tab["concrete_pct"],
                                f"{e2s:.4f}" if not math.isnan(e2s) else "nan",
                                f"{contrast_score[(it, axis)]:.3f}" if not math.isnan(contrast_score[(it, axis)]) else "nan"])
    print(f"wrote {OUT_TSV}")

    # per-item best axis contrast
    per_item_best = {}
    for it in range(1, 8):
        best_ax, best_c = None, -1.0
        for axis in ALL_AXES:
            c = contrast_score[(it, axis)]
            if math.isnan(c): continue
            if c > best_c: best_c, best_ax = c, axis
        per_item_best[it] = {"best_axis": best_ax,
                              "best_contrast_pp": round(best_c, 3),
                              "axis_values": coverage[(it, best_ax)] if best_ax else {}}

    # stack-invariance test
    flat_stack = [(it, ax, eta2_score[(it, ax)]) for it in range(1, 8)
                  for ax in STACK_AXES if not math.isnan(eta2_score[(it, ax)])]
    n_lt_005 = sum(1 for _, _, e in flat_stack if e < 0.05)
    n_lt_010 = sum(1 for _, _, e in flat_stack if e < 0.10)

    summary = {
        "n_manifests": n,
        "reward_quartile_cutoffs": r_q,
        "zvf_quartile_cutoffs": z_q,
        "per_item_overall_counts": {str(k): dict(v) for k, v in per_item_overall.items()},
        "stack_invariance_test": {
            "n_pairs": len(flat_stack),
            "n_eta2_lt_005": n_lt_005, "n_eta2_lt_010": n_lt_010,
            "interpretation": ("If most (item, stack-axis) eta^2 of per-cell "
                               "item_score are < 0.05, item coverage is "
                               "STACK-INVARIANT — the manifest emitter treats "
                               "every stack uniformly. (NaN entries on items "
                               "1-6 are NOT missing — they are zero-within-"
                               "variance, the strongest possible invariance.)"),
        },
        "eta2_per_item_per_axis": {
            f"item{it}/{ax}": (round(e, 4) if not math.isnan(e) else None)
            for it in range(1, 8) for ax in ALL_AXES
            for e in [eta2_score[(it, ax)]]
        },
        "contrast_per_item_per_axis": {
            f"item{it}/{ax}": (round(contrast_score[(it, axis)], 3)
                                if not math.isnan(contrast_score[(it, axis)]) else None)
            for it in range(1, 8) for ax in ALL_AXES
        },
        "per_item_best_axis": per_item_best,
        "per_item_coverage_per_axis_sample": {
            f"item{it}/{ax}": coverage[(it, ax)]
            for it in (1, 2, 4, 7) for ax in STACK_AXES
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_JSON}")

    # console summary
    print("\n=== eta^2 of per-cell item_score across stack axes ===")
    print(f"{'item':>20s}  " + "  ".join(f"{ax:>15s}" for ax in STACK_AXES))
    for it in range(1, 8):
        row = f"{ITEM_NAMES[it]:>20s}  "
        for ax in STACK_AXES:
            v = eta2_score[(it, ax)]
            row += f"  {v:>15.4f}" if not math.isnan(v) else f"  {'nan':>15s}"
        print(row)
    print(f"\n  n(eta^2<0.05) = {n_lt_005}/{len(flat_stack)}  (stack-invariance test)")
    for it in range(1, 8):
        c = per_item_best[it]
        print(f"  item{it} ({ITEM_NAMES[it]:>20s}): best axis={c['best_axis']}, contrast={c['best_contrast_pp']}pp")

    # ---- heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return 0
    items = list(range(1, 8))
    M_e = np.array([[0 if math.isnan(eta2_score[(it, ax)]) else eta2_score[(it, ax)]
                      for ax in ALL_AXES] for it in items])
    M_c = np.array([[0 if math.isnan(contrast_score[(it, ax)]) else contrast_score[(it, ax)]
                      for ax in ALL_AXES] for it in items])
    FIG.mkdir(parents=True, exist_ok=True)
    for M, fname, title, cmap, vmax in [
        (M_e, "minreport_stratified_heatmap", "eta^2 of per-cell item_score", "YlOrRd", max(0.05, M_e.max())),
        (M_c, "minreport_stratified_contrast", "max-min contrast (score units)", "YlGnBu", max(0.5, M_c.max())),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(ALL_AXES)))
        ax.set_xticklabels(ALL_AXES, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"item{it} {ITEM_NAMES[it]}" for it in items], fontsize=8)
        ax.set_title(f"{title} (n={n} mega manifests)")
        for i in range(len(items)):
            for j in range(len(ALL_AXES)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if M[i,j] < M.max()*0.6 else "white")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIG / f"{fname}.png", dpi=150)
        fig.savefig(FIG / f"{fname}.pdf")
        plt.close(fig)
    print(f"figure: {FIG}/minreport_stratified_heatmap.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())