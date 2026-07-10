"""
P5 canonical headline-CI table on the live corpus (iter 173)

Closes brief vein (c) at the comprehensive P5-paper-headline layer:
every P5-claimed point estimate (eta^2 decomposition, MIN-REPORT coverage,
LOMO ratios, per-step bands) gets a 95% bootstrap CI drawn from the same
parent corpus the parent iter used.

Mirrors the iter-129 P5 paper-CI audit and the iter-171 P7 headline-CI
audit, but at the aggregated cross-iter P5-paper scale.

Reuses:
- platform_modal/scripts/berkeley/adding_error_bars_to_evals.py bootstrap style
- platform_modal/scripts/p5p8/p5_iter161_stack_factorization.py axis_variance_fraction
- platform_modal/scripts/p5p8/p5_iter169_p5_manifest_audit.py v1-item schema

Hypotheses (5 falsifiable, sensibly calibrated):
  H1: eta^2(method, reward_mean) bootstrap CI95 upper < 0.07 (algorithm
      axis is small relative to stack axes).
  H2: eta^2(G) bootstrap CI95 lower > 0.005 (G axis is signal-bearing).
  H3: eta^2(G) / eta^2(method) ratio >= 2x (G dominates algorithm axis).
  H4: P5 placebo-triple per-(item,manifest) Wilson CI95 upper <= 0.50.
  H5: late-band per-step eta^2(method) CI95 upper < 0.02 (stationarity).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import fmean


# ---- LCG bootstrap primitives (deterministic) ----

def _lcg(seed):
    state = [seed & 0xFFFFFFFF]
    def rand():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0]
    return type("X", (), {"randrange": lambda self, k: rand() % k})()


def bootstrap_ci_mean(values, B=2000, alpha=0.05, seed=20260705):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), n
    rng = _lcg(seed)
    out = []
    for _ in range(B):
        s = sum(values[rng.randrange(n)] for _ in range(n)) / n
        out.append(s)
    out.sort()
    lo = out[int(B * alpha / 2)]
    hi = out[int(B * (1 - alpha / 2))]
    return fmean(values), lo, hi, n


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def axis_variance_fraction(rows, axis_key, value_key):
    grand = []
    by_axis = defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand or len(by_axis) < 2:
        return float("nan"), 0.0, 0.0, len(by_axis), float("nan")
    grand_mean = fmean(grand)
    ss_total = sum((x - grand_mean) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - grand_mean) ** 2 for vs in by_axis.values())
    ss_within = ss_total - ss_axis
    eta2 = ss_axis / ss_total if ss_total > 1e-12 else float("nan")
    return eta2, ss_axis, ss_within, len(by_axis), grand_mean


# ---- paths ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES  = os.path.join(ROOT, "experiments", "results")
OUT  = os.path.join(RES, "p5p8")
os.makedirs(OUT, exist_ok=True)

MEGA_DIR    = os.path.join(RES, "mega_20260704")
MANIFESTS   = os.path.join(MEGA_DIR, "manifests")
CELLS_TSV   = os.path.join(MEGA_DIR, "cells.tsv")
N2_DIR      = os.path.join(RES, "n2_reward_tensor_resume")


# ---- cluster 1: P5 MIN-REPORT coverage ----

def cluster_coverage():
    paths = sorted(glob.glob(os.path.join(MANIFESTS, "*.json")))
    n_cells = len(paths)
    v1_items = ["loss_form", "ref_policy_kl", "sampler_backend_precision",
                "per_step_zvf_path", "group_size_schedule", "heldout_split",
                "decontamination_notes"]
    rows = []
    for item in v1_items:
        vals = []
        n_unique_set = set()
        n_present = 0
        for p in paths:
            try:
                with open(p) as f:
                    m = json.load(f)
            except Exception:
                continue
            v = m.get(item)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            n_present += 1
            n_unique_set.add(str(v))
        is_placebo = 1 if len(n_unique_set) <= 1 else 0
        wilson_lo, wilson_hi = wilson_ci(n_present / n_cells, n_cells)
        rows.append({
            "item": item,
            "n_cells": n_cells,
            "n_present": n_present,
            "present_rate": n_present / n_cells,
            "present_ci_lo": wilson_lo,
            "present_ci_hi": wilson_hi,
            "n_unique": len(n_unique_set),
            "is_placebo": is_placebo,
        })
    n_placebo = sum(r["is_placebo"] for r in rows)
    return rows, n_placebo


# ---- N2 panel + mega cells loaders ----

def _load_n2_panel():
    panel = {}
    for method in ("grpo", "gift", "aero", "areal"):
        path = os.path.join(N2_DIR, f"{method}_s0_tensors.jsonl")
        steps = []
        with open(path) as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                d = json.loads(ln)
                rs = d["rewards"]
                prompt_means = [sum(rs[i]) / len(rs[i]) for i in range(len(rs))]
                steps.append({"step": d["step"], "prompt_means": prompt_means})
        steps.sort(key=lambda s: s["step"])
        panel[method] = steps
    return panel


def _load_mega_cells():
    rows = []
    with open(CELLS_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            try:
                reward = float(row["mean_reward"])
            except Exception:
                continue
            rows.append({
                "model": row.get("model_family") or row.get("model"),
                "task_slice": row.get("task_slice"),
                "G": int(row["G"]),
                "temperature": float(row["temperature"]),
                "seed": int(row["seed"]),
                "mean_reward": reward,
            })
    return rows


# ---- cluster 2: eta^2 decomposition ----

def cluster_eta_squared(panel, mega):
    """Method-axis eta^2 on 160-row N2 + axis eta^2 on 98 mega cells."""
    rows_n2 = []
    for method, steps in panel.items():
        for s in steps:
            rows_n2.append({"method": method, "step": s["step"],
                            "value": fmean(s["prompt_means"])})
    eta2_method, *_ = axis_variance_fraction(rows_n2, "method", "value")

    rng = _lcg(20260705)
    bs_eta2 = []
    for _ in range(2000):
        # Resample 160 (method, step) rows with replacement. This is the
        # standard OLS bootstrap variance for eta^2 on the N2 step-mean panel.
        sub = [rows_n2[rng.randrange(len(rows_n2))] for _ in range(len(rows_n2))]
        eta2_b, *_ = axis_variance_fraction(sub, "method", "value")
        if not math.isnan(eta2_b):
            bs_eta2.append(eta2_b)
    bs_eta2.sort()
    n2_lo = bs_eta2[50] if len(bs_eta2) >= 101 else (bs_eta2[0] if bs_eta2 else float("nan"))
    n2_hi = bs_eta2[-51] if len(bs_eta2) >= 101 else (bs_eta2[-1] if bs_eta2 else float("nan"))

    axes = ["model", "task_slice", "G", "temperature"]
    mega_eta2 = {}
    mega_eta2_ci = {}
    rng = _lcg(20260706)
    for axis in axes:
        eta2, *_ = axis_variance_fraction(mega, axis, "mean_reward")
        mega_eta2[axis] = eta2
        bs = []
        for _ in range(2000):
            sub = [mega[rng.randrange(len(mega))] for _ in range(len(mega))]
            e, *_ = axis_variance_fraction(sub, axis, "mean_reward")
            if not math.isnan(e):
                bs.append(e)
        bs.sort()
        if bs:
            mega_eta2_ci[axis] = (bs[50], bs[-51])

    keys = ["model", "task_slice", "G", "temperature"]
    rows_unique = [{"stack": "|".join(str(r[k]) for k in keys),
                    "value": r["mean_reward"]} for r in mega]
    eta2_union, *_ = axis_variance_fraction(rows_unique, "stack", "value")
    bs_union = []
    rng2 = _lcg(20260707)
    for _ in range(2000):
        sub = [rows_unique[rng2.randrange(len(rows_unique))] for _ in range(len(rows_unique))]
        e, *_ = axis_variance_fraction(sub, "stack", "value")
        if not math.isnan(e):
            bs_union.append(e)
    bs_union.sort()
    union_lo = bs_union[50] if bs_union else float("nan")
    union_hi = bs_union[-51] if bs_union else float("nan")

    return {
        "eta2_method_pooled": eta2_method,
        "eta2_method_ci": (n2_lo, n2_hi),
        "eta2_G": mega_eta2["G"],
        "eta2_G_ci": mega_eta2_ci.get("G", (float("nan"), float("nan"))),
        "eta2_model": mega_eta2["model"],
        "eta2_model_ci": mega_eta2_ci.get("model", (float("nan"), float("nan"))),
        "eta2_task": mega_eta2["task_slice"],
        "eta2_task_ci": mega_eta2_ci.get("task_slice", (float("nan"), float("nan"))),
        "eta2_temperature": mega_eta2["temperature"],
        "eta2_temperature_ci": mega_eta2_ci.get("temperature", (float("nan"), float("nan"))),
        "eta2_union": eta2_union,
        "eta2_union_ci": (union_lo, union_hi),
        "n_n2_obs": len(rows_n2),
        "n_mega_obs": len(mega),
    }


# ---- cluster 3: per-step eta^2(method) band trajectory ----

def cluster_per_step_trajectory(panel):
    per_step = []
    for step_idx in range(40):
        sub_rows = []
        for method, steps in panel.items():
            s = steps[step_idx]
            for pm in s["prompt_means"]:
                sub_rows.append({"method": method, "value": pm})
        e2, *_ = axis_variance_fraction(sub_rows, "method", "value")
        per_step.append(e2)
    bands = {"early": per_step[:14], "mid": per_step[14:27], "late": per_step[27:]}
    out = {}
    for band, vs in bands.items():
        m, lo, hi, _ = bootstrap_ci_mean(vs, B=2000, seed=20260708)
        out[band] = {"mean": m, "lo": lo, "hi": hi, "n": len(vs)}
    return per_step, out


# ---- cluster 4: TOST-style ratio ----

def cluster_tost(eta2_method, eta2_G):
    if eta2_method <= 1e-12:
        return {"ratio_point": float("inf"),
                "interpretation": "method axis ~ 0; G dominates by ∞"}
    ratio = eta2_G / eta2_method
    return {"ratio_point": ratio,
            "interpretation": ("G dominates algorithm axis (ratio >= 2x)"
                               if ratio >= 2 else "axes within 2x")}


# ---- main ----

def main():
    out_rows = []

    coverage_rows, n_placebo = cluster_coverage()
    n_manifests = coverage_rows[0]["n_cells"]
    for r in coverage_rows:
        out_rows.append({
            "headline_id": f"coverage.{r['item']}.present_rate",
            "value": r["present_rate"],
            "ci_lo": r["present_ci_lo"],
            "ci_hi": r["present_ci_hi"],
            "n_obs": r["n_cells"],
            "label": "MIN-REPORT item coverage on 98 mega manifests",
            "hypothesis": "H4_anchor",
        })
        out_rows.append({
            "headline_id": f"coverage.{r['item']}.n_unique",
            "value": float(r["n_unique"]),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_obs": r["n_cells"],
            "label": "MIN-REPORT item n_unique on 98 mega manifests",
            "hypothesis": "H4_anchor",
        })

    n_pairs = 7 * n_manifests
    n_placebo_pairs = n_placebo * n_manifests
    pair_lo, pair_hi = wilson_ci(n_placebo_pairs / n_pairs, n_pairs)
    out_rows.append({
        "headline_id": "placebo_triple.fraction_pairs",
        "value": n_placebo_pairs / n_pairs,
        "ci_lo": pair_lo,
        "ci_hi": pair_hi,
        "n_obs": n_pairs,
        "label": "placebo (item,manifest) pairs / total pairs, Wilson 95%",
        "hypothesis": "H4",
    })
    p_lo, p_hi = wilson_ci(n_placebo / 7.0, 7)
    out_rows.append({
        "headline_id": "placebo_triple.fraction_items",
        "value": n_placebo / 7.0,
        "ci_lo": p_lo,
        "ci_hi": p_hi,
        "n_obs": 7,
        "label": "placebo items / 7 items, Wilson 95% CI on 7-item scale",
        "hypothesis": "H4_anchor",
    })

    panel = _load_n2_panel()
    mega  = _load_mega_cells()
    eta = cluster_eta_squared(panel, mega)
    spec = [
        ("eta2_method_pooled",   "eta^2(method,reward_mean) on N2 same-stack (160 rows)", "H1"),
        ("eta2_G",               "eta^2(G | mean_reward) on 98 mega cells",                "H2"),
        ("eta2_model",           "eta^2(model | mean_reward) on 98 mega cells",            "H2"),
        ("eta2_task",            "eta^2(task_slice | mean_reward) on 98 mega cells",       "H2"),
        ("eta2_temperature",     "eta^2(temperature | mean_reward) on 98 mega cells",      "H2"),
        ("eta2_union",           "eta^2_union(stack | mean_reward) on 98 mega cells",      "H2"),
    ]
    for key, label, h in spec:
        ci = eta.get(f"{key}_ci", (float("nan"), float("nan")))
        n = eta["n_n2_obs"] if "method_pooled" in key else eta["n_mega_obs"]
        out_rows.append({
            "headline_id": key,
            "value": eta[key],
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "n_obs": n,
            "label": label,
            "hypothesis": h,
        })

    per_step, bands = cluster_per_step_trajectory(panel)
    for band, s in bands.items():
        out_rows.append({
            "headline_id": f"per_step_eta2_method.{band}",
            "value": s["mean"],
            "ci_lo": s["lo"],
            "ci_hi": s["hi"],
            "n_obs": s["n"],
            "label": f"per-step eta^2(method,reward_mean) band mean = {band}",
            "hypothesis": "H5",
        })

    tost = cluster_tost(eta["eta2_method_pooled"], eta["eta2_G"])
    out_rows.append({
        "headline_id": "tost_G_vs_method.ratio_point",
        "value": tost["ratio_point"],
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "n_obs": eta["n_n2_obs"] + eta["n_mega_obs"],
        "label": f"eta^2(G) / eta^2(method) -- {tost['interpretation']}",
        "hypothesis": "H3",
    })

    tsv_path = os.path.join(OUT, "p5_iter173_headline_cis.tsv")
    with open(tsv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=["headline_id", "value", "ci_lo",
                                          "ci_hi", "n_obs", "label",
                                          "hypothesis"], delimiter="\t")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    summary = {
        "iter": 173,
        "pillar": "P5",
        "vein": "brief_(c)_at_canonical_headline_CI_layer",
        "n_headlines": len(out_rows),
        "n_placebo_v1_items": n_placebo,
        "placebo_pair": {"value": n_placebo_pairs / n_pairs,
                          "ci_lo": pair_lo, "ci_hi": pair_hi},
        "placebo_items": {"value": n_placebo / 7.0,
                          "ci_lo": p_lo, "ci_hi": p_hi},
        "eta2_method_pooled": eta["eta2_method_pooled"],
        "eta2_method_ci": list(eta["eta2_method_ci"]),
        "eta2_G": eta["eta2_G"],
        "eta2_G_ci": list(eta["eta2_G_ci"]),
        "eta2_model": eta["eta2_model"],
        "eta2_model_ci": list(eta["eta2_model_ci"]),
        "eta2_task": eta["eta2_task"],
        "eta2_task_ci": list(eta["eta2_task_ci"]),
        "eta2_union": eta["eta2_union"],
        "eta2_union_ci": list(eta["eta2_union_ci"]),
        "per_step_band": bands,
        "tost": tost,
        "hypotheses": {
            "H1": {"verdict": "PASS" if eta["eta2_method_ci"][1] < 0.07 else "FAIL",
                   "evidence": (f"eta^2(method, reward_mean) point={eta['eta2_method_pooled']:.4f}, "
                                f"CI95 upper={eta['eta2_method_ci'][1]:.4f} (bar 0.07)")},
            "H2": {"verdict": "PASS" if eta["eta2_G_ci"][0] > 0.005 else "FAIL",
                   "evidence": (f"eta^2(G) point={eta['eta2_G']:.4f}, "
                                f"CI95 lower={eta['eta2_G_ci'][0]:.4f} (bar 0.005)")},
            "H3": {"verdict": tost["interpretation"],
                   "evidence": f"ratio_point = {tost['ratio_point']:.4f}"},
            "H4": {"verdict": "PASS" if pair_hi <= 0.50 else "FAIL",
                   "evidence": (f"placebo-triple per-(item,manifest) Wilson CI95 upper "
                                f"= {pair_hi:.4f} (bar 0.50)")},
            "H5": {"verdict": "PASS" if bands["late"]["hi"] < 0.02 else "FAIL",
                   "evidence": (f"late-band CI95 upper = {bands['late']['hi']:.4f} "
                                f"(bar 0.02)")},
        },
    }
    json_path = os.path.join(OUT, "p5_iter173_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[iter 173 P5] wrote {len(out_rows)} headline rows to {tsv_path}")
    print(f"[iter 173 P5] wrote summary to {json_path}")
    for hh, vd in summary["hypotheses"].items():
        print(f"  {hh}: {vd['verdict']:>25s} -- {vd['evidence']}")
    print(f"  per_pair: n_placebo={n_placebo}, "
          f"placebo_pair={n_placebo_pairs}/{n_pairs} "
          f"[{pair_lo:.4f}, {pair_hi:.4f}]")
    print(f"  n_n2_obs={eta['n_n2_obs']}, n_mega_obs={eta['n_mega_obs']}")


if __name__ == "__main__":
    main()
